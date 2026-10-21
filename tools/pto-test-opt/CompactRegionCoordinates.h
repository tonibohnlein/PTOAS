// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Test-only translation between supplied original visits and retained event frames.
#ifndef PTO_TEST_COMPACTREGIONCOORDINATES_H
#define PTO_TEST_COMPACTREGIONCOORDINATES_H
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceAnalysisInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RepeatedRegionInternal.h"
#include "mlir/IR/Matchers.h"
namespace compact_contract {
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
using Coordinates = llvm::DenseMap<Operation*, uint64_t>;
using Key = std::pair<uint64_t, std::vector<uint64_t>>;
inline std::optional<std::pair<uint64_t, uint64_t>> domain(scf::ForOp loop)
{
    APInt lower, step;
    if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
        !matchPattern(loop.getStep(), m_ConstantInt(&step)) || lower.isNegative() || step.isNegative() ||
        step.isZero() || lower.getActiveBits() > 64 || step.getActiveBits() > 64) {
        return {};
    }
    return std::make_pair(lower.getZExtValue(), step.getZExtValue());
}
inline std::optional<uint64_t> number(const fs::RegionalAnalysis& region, fs::RegionExpressions::Id value)
{
    auto& e = *region.expressions;
    if (auto literal = e.constantValue(value)) { return literal; }
    if (region.anchors.empty() || !region.anchors.front().phase) { return {}; }
    auto function = region.anchors.front().phase->elementOp->getParentOfType<func::FuncOp>();
    auto supplied = function->getAttrOfType<ArrayAttr>("test.contract_constants");
    if (!supplied) { return {}; }
    // Values are supplied by the independent finite interpreter, keyed by the
    // parsed operation's source line. Only values identical across all executed
    // visits are supplied; a changing predicate is never replaced by its last
    // observed value. This evaluates tests, not production availability.
    SmallVector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> bindings;
    function.walk([&](Operation* operation) {
        if (operation->getName().getDialectNamespace() != "arith" || operation->getNumResults() != 1) { return; }
        auto location = dyn_cast<FileLineColLoc>(operation->getLoc());
        auto result = operation->getResult(0);
        if (!location || (!isa<IntegerType>(result.getType()) && !result.getType().isIndex())) { return; }
        for (auto attribute : supplied) {
            auto row = dyn_cast<DenseI64ArrayAttr>(attribute);
            if (!row || row.size() != 2 || row[0] != location.getLine()) { continue; }
            auto literal = result.getType().isInteger(1) ? e.boolean(row[1] != 0) :
                e.constant(static_cast<uint64_t>(row[1]));
            bindings.emplace_back(e.input(result), literal);
        }
    });
    fs::RegionExpressions::Substitution substitution(bindings);
    return e.constantValue(e.substitute(value, substitution));
}
inline bool decode(const fs::RegionalAnalysis& region, fs::RegionalEvent event, Coordinates& values)
{
    if (event.type >= region.anchors.size()) {
        return false;
    }
    if (auto state = region.retainedRepetition.lock()) {
        if (event.visits.empty() || !state->phaseCount) {
            return false;
        }
        auto n = number(region, event.visits.front());
        auto bounds = domain(state->loop);
        if (!n || !bounds) {
            return false;
        }
        if (!state->typePhases.empty() && event.type >= state->typePhases.size()) {
            return false;
        }
        auto phase = state->typePhases.empty() ? 0 : state->typePhases[event.type];
        if (*n > (UINT64_MAX - phase) / state->phaseCount) {
            return false;
        }
        const auto ordinal = *n * state->phaseCount + phase;
        if (ordinal > (UINT64_MAX - bounds->first) / bounds->second) {
            return false;
        }
        values[state->loop] = bounds->first + bounds->second * ordinal;
        event.visits.erase(event.visits.begin());
        return decode(state->body, std::move(event), values);
    }
    if (auto state = region.retainedSequence.lock()) {
        for (const auto& child : state->children) {
            if (event.type < child.regional.anchors.size()) {
                return decode(child.regional, std::move(event), values);
            }
            event.type -= child.regional.anchors.size();
        }
        return false;
    }
    // Generic arithmetic exports retain all enclosing ordinal frames. The
    // innermost ordinal alone cannot reconstruct a nested original occurrence.
    if (!region.outerLoops.empty()) {
        if (event.type >= region.outerLoops.size() ||
            event.visits.size() != region.outerLoops[event.type].size()) { return false; }
        for (std::size_t i = 0; i < event.visits.size(); ++i) {
            const auto loop = region.outerLoops[event.type][i];
            const auto bounds = domain(loop);
            const auto ordinal = number(region, event.visits[i]);
            // Nonunit quotient frames need their provider's phase decoder.
            if (!region.outerDivisors.empty() &&
                (event.type >= region.outerDivisors.size() ||
                 i >= region.outerDivisors[event.type].size() || region.outerDivisors[event.type][i] != 1)) {
                return false;
            }
            if (!bounds || !ordinal || *ordinal > (UINT64_MAX - bounds->first) / bounds->second) {
                return false;
            }
            values[loop] = bounds->first + bounds->second * *ordinal;
        }
    }
    const auto& anchor = region.anchors[event.type];
    for (const auto& point : anchor.coordinates) {
        if (point.induction < 0) {
            return false;
        }
        values.try_emplace(point.loop, static_cast<uint64_t>(point.induction));
    }
    if (event.type >= region.occurrenceLoops.size()) {
        return false;
    }
    if (auto loop = region.occurrenceLoops[event.type]) {
        const auto bounds = domain(loop);
        const auto ordinal = number(region, event.ordinal);
        if (!bounds || !ordinal || *ordinal > (UINT64_MAX - bounds->first) / bounds->second) {
            return false;
        }
        values[loop] = bounds->first + bounds->second * *ordinal;
    }
    return true;
}
inline SmallVector<scf::ForOp> enclosing(Operation* operation)
{
    SmallVector<scf::ForOp> loops;
    for (auto* parent = operation->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            loops.push_back(loop);
        }
    }
    std::reverse(loops.begin(), loops.end());
    return loops;
}
inline std::optional<Key> key(
    const fs::RegionalAnalysis& region, fs::RegionalEvent event, ArrayRef<const pto::CompoundInstanceElement*> phases)
{
    if (event.type >= region.anchors.size()) {
        return {};
    }
    const auto* phase = region.anchors[event.type].phase;
    auto found = llvm::find(phases, phase);
    if (found == phases.end()) {
        return {};
    }
    Coordinates values;
    if (!decode(region, event, values)) {
        return {};
    }
    Key result{static_cast<uint64_t>(found - phases.begin()), {}};
    for (auto loop : enclosing(phase->elementOp)) {
        auto coordinate = values.find(loop);
        if (coordinate == values.end()) {
            return {};
        }
        result.second.push_back(coordinate->second);
    }
    return result;
}
inline std::optional<fs::RegionalEvent> candidate(
    const fs::RegionalAnalysis& region, uint32_t type, const Coordinates& values)
{
    if (!region.expressions || type >= region.anchors.size() || type >= region.occurrenceLoops.size() ||
        (!region.outerLoops.empty() && type >= region.outerLoops.size()) ||
        (!region.outerDivisors.empty() && (region.outerLoops.empty() || type >= region.outerDivisors.size() ||
                                           region.outerDivisors[type].size() != region.outerLoops[type].size()))) {
        return {};
    }
    auto& e = *region.expressions;
    fs::RegionalEvent event{type, e.constant(0), fs::PeriodicEventKind::Start};
    auto ordinal = [&](scf::ForOp loop) -> std::optional<uint64_t> {
        auto bounds = domain(loop);
        auto position = values.find(loop);
        if (!bounds || position == values.end() || position->second < bounds->first ||
            (position->second - bounds->first) % bounds->second) {
            return {};
        }
        return (position->second - bounds->first) / bounds->second;
    };
    if (!region.outerLoops.empty()) {
        for (std::size_t frame = 0; frame < region.outerLoops[type].size(); ++frame) {
            auto value = ordinal(region.outerLoops[type][frame]);
            const auto divisor = region.outerDivisors.empty() ? 1 : region.outerDivisors[type][frame];
            if (!value || !divisor) {
                return {};
            }
            event.visits.push_back(e.constant(*value / divisor));
        }
    }
    if (region.occurrenceLoops[type]) {
        auto value = ordinal(region.occurrenceLoops[type]);
        if (!value) {
            return {};
        }
        event.ordinal = e.constant(*value);
    }
    return event;
}
} // namespace compact_contract
#endif
