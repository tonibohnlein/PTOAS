// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Construct a private retained graph, then certify all omitted hazard envelopes.
#include "OrderCoveredPeriodic.h"
#include "../Shared/RecognitionInternal.h"
#include "../Shared/CountedLoop.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
bool appendMaterializedCells(RotatingPrimitives& result, ArrayRef<std::size_t> effectIds,
                             const SyncInput& input, scf::ForOp loop)
{
    // Shared partition IDs are disjoint across address spaces and bases.
    // These are the supplied modeled ranges, including existing buffer bounds.
    const auto& storage = input.accesses();
    const auto protection = structuredProtection(storage);
    DenseMap<const CompoundInstanceElement*, uint32_t> positions;
    for (auto [position, phase] : llvm::enumerate(result.phases)) { positions[phase] = position; }
    DenseMap<std::pair<uint32_t, uint32_t>, std::size_t> fragments;
    for (auto id : effectIds) {
        const auto& effect = storage.effects()[id];
        auto position = positions.find(effect.phase);
        if (position == positions.end()) { return false; }
        for (auto cell : effect.cells) {
            if (cell >= storage.cells().size() || cell > UINT32_MAX || result.fragments.size() >= UINT32_MAX) {
                return false;
            }
            const auto& physical = storage.cells()[cell];
            if (physical.base || physical.space != effect.memory->scope || physical.begin >= physical.end) {
                return false;
            }
            const auto atom = static_cast<uint32_t>(cell);
            const auto [found, inserted] = fragments.try_emplace({position->second, atom}, result.fragments.size());
            if (inserted) { result.fragments.push_back({position->second, 0, atom, 1, 0, 0, false, false, 0}); }
            auto& fragment = result.fragments[found->second];
            fragment.read |= effect.mode == SyncAccessMode::Read;
            fragment.write |= effect.mode == SyncAccessMode::Write;
            // Protected accumulator RMW must be one writer fragment before the
            // extractor validates it; read-only cells retain ordinary hazards.
            if (fragment.write && physical.space == AddressSpace::ACC) {
                fragment.protectionGroup = protection.inLoop(effect.phase, loop);
            }
        }
    }
    return true;
}
bool coverEffects(OrderCoveredPeriodicResult& result, std::string& error)
{
    const auto& effects = result.input->accesses();
    DenseMap<const CompoundInstanceElement*, uint32_t> positions;
    for (auto [position, phase] : llvm::enumerate(result.retained.phases)) { positions[phase] = position; }
    const std::set<std::size_t> omitted(result.omittedEffects.begin(), result.omittedEffects.end());
    auto check = [&](std::size_t a, std::size_t b) {
        if (!effects.mayConflict(a, b)) { return true; }
        const auto left = positions.find(effects.effects()[a].phase);
        const auto right = positions.find(effects.effects()[b].phase);
        if (left == positions.end() || right == positions.end()) {
            error = "order-covered effect has a potential hazard outside its original owner"; return false;
        }
        const uint64_t forward = left->second < right->second ? 0 : 1;
        const auto threshold = result.retained.periodic.completionThreshold(
            left->second, {right->second, PeriodicEventKind::Start});
        if (threshold.error != PeriodicQueryError::None || !threshold.displacement ||
            *threshold.displacement > forward) {
            error = "retained periodic order does not cover every omitted forward hazard"; return false;
        }
        result.coveredPairs.push_back({a, b, left->second, right->second, forward, *threshold.displacement});
        return true;
    };
    // Enumerate each ordered pair touching an omitted effect once, in O(m*n).
    for (std::size_t a = 0; a < effects.effects().size(); ++a) {
        if (omitted.count(a)) {
            for (std::size_t b = 0; b < effects.effects().size(); ++b) {
                if (!check(a, b)) { return false; }
            }
        } else {
            for (auto b : result.omittedEffects) { if (!check(a, b)) { return false; } }
        }
    }
    return true;
}
} // namespace
bool OrderCoveredPeriodicResult::applies(
    const SyncInput* expectedInput, const PhaseIndex* expectedIndex, scf::ForOp loop) const
{
    auto function = loop ? dyn_cast<func::FuncOp>(loop->getParentOp()) : func::FuncOp{};
    return input && index && input.get() == expectedInput && index.get() == expectedIndex &&
        function && function.getBody().hasOneBlock() &&
        isa<func::ReturnOp>(function.getBody().front().getTerminator()) &&
        retained.loop == loop && retained.error.empty() &&
        retained.periodic.error.empty();
}
std::shared_ptr<const OrderCoveredPeriodicResult> analyzeOrderCoveredPeriodic(
    scf::ForOp loop, std::size_t originalRegion, std::shared_ptr<const SyncInput> input,
    std::shared_ptr<const PhaseIndex> index, uint64_t& reductions, std::string& error)
{
    error.clear();
    if (!input || !index || !CountedLoop::get(loop)) {
        error = "order coverage requires a representable original counted domain"; return {};
    }
    auto hypothesis = detail::recognizeOrderCoveredInvariant(loop, *input, *index, error);
    if (!hypothesis) { return {}; }
    auto primitives = collectRotatingPrimitives(loop, *index, *input, hypothesis->retained);
    if (!primitives.error.empty()) { error = primitives.error; return {}; }
    if (!appendMaterializedCells(primitives, hypothesis->materializedEffects, *input, loop)) {
        error = "materialized local cell identities are unavailable"; return {};
    }
    auto result = std::make_shared<OrderCoveredPeriodicResult>();
    result->input = std::move(input); result->index = std::move(index); result->originalRegion = originalRegion;
    result->omittedEffects = std::move(hypothesis->omittedEffects);
    auto& retained = result->retained;
    retained.loop = loop; retained.phases = std::move(primitives.phases);
    retained.fragments = std::move(primitives.fragments);
    retained.extraction = extractRotatingGenerators(primitives.payloads, retained.fragments, ptoStorageProtection());
    if (!retained.extraction.error.empty()) { error = retained.extraction.error; return {}; }
    for (const auto& edge : primitives.prerequisites) {
        retained.extraction.generators.push_back(edge);
        retained.extraction.refreshBound = std::max(retained.extraction.refreshBound, edge.displacement);
    }
    if (reductions != UINT64_MAX) { ++reductions; }
    retained.periodic = analyzePeriodicDemands(primitives.payloads, retained.extraction.generators,
                                              primitives.nativePrerequisites);
    if (!retained.periodic.error.empty()) { error = retained.periodic.error; return {}; }
    if (!coverEffects(*result, error)) { return {}; }
    return result;
}
} // namespace mlir::pto::frontiersynch
