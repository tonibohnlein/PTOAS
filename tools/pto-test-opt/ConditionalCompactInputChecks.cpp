// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite alternative-word coverage oracle; production never enumerates words.
#include "PTO/Transforms/FrontierSynch/ConditionalCompactInput.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Expr = fs::RegionExpressions::Id;
using Key = std::tuple<uint32_t, uint32_t, fs::StorageHazard>;
bool coverage(const fs::ConditionalCompactInput& built, const pto::SyncInput& input, uint64_t& checked)
{
    const auto& slots = built.body->slots;
    const auto& model = input.accesses();
    const auto& upper = built.mathematical.upper;
    for (uint32_t trips = 0; trips <= 3; ++trips) {
        uint64_t choices = 1;
        for (uint32_t i = 0; i < trips; ++i) {
            for (const auto& slot : slots) {
                // Fixture size bound only. Every independent alternative word
                // is tested, a superset of the correlated original branch paths.
                if (slot.alternatives.empty() || choices > 65536 / slot.alternatives.size()) { return false; }
                choices *= slot.alternatives.size();
            }
        }
        for (uint64_t encoded = 0; encoded < choices; ++encoded) {
            auto remaining = encoded;
            std::vector<const pto::CompoundInstanceElement*> word;
            for (uint32_t i = 0; i < trips; ++i) {
                for (const auto& slot : slots) {
                    word.push_back(slot.alternatives[remaining % slot.alternatives.size()].phase);
                    remaining /= slot.alternatives.size();
                }
            }
            for (uint32_t a = 0; a < word.size(); ++a) {
                for (uint32_t b = a + 1; b < word.size(); ++b) {
                    if (fs::ptoStorageProtection().protectsScalar(static_cast<uint32_t>(word[a]->kPipeValue),
                                                                  static_cast<uint32_t>(word[b]->kPipeValue))) {
                        continue;
                    }
                    bool conflict = false;
                    for (auto x : model.effectsFor(word[a])) {
                        for (auto y : model.effectsFor(word[b])) {
                            conflict |= (model.effects()[x].mode == pto::SyncAccessMode::Write ||
                                         model.effects()[y].mode == pto::SyncAccessMode::Write) &&
                                        model.mayOverlap(x, y);
                        }
                    }
                    if (!conflict) { continue; }
                    const auto m = static_cast<uint32_t>(slots.size());
                    auto ordered = upper.eventPrecedes({a % m, fs::PeriodicEventKind::Completion}, a / m,
                        {b % m, fs::PeriodicEventKind::Start}, b / m, word.size());
                    if (ordered.error != fs::PeriodicQueryError::None || !ordered.value) { return false; }
                }
            }
            ++checked;
        }
    }
    return true;
}
bool bindings(scf::ForOp loop, const pto::SyncInput& input, const fs::PhaseIndex& index,
              const fs::ConditionalCompactInput& base)
{
    std::map<std::size_t, uint32_t> accessIds;
    for (uint32_t id = 0; id < base.storage.accessEffects.size(); ++id) {
        for (auto effect : base.storage.accessEffects[id]) { accessIds[effect] = id; }
    }
    std::map<Key, fs::OriginDistanceInterval> expected;
    fs::CompactWriterReaderBindings supplied;
    supplied.accessBounds = [&](std::size_t a, std::size_t b, fs::StorageHazard hazard)
        -> std::optional<fs::OriginDistanceInterval> {
        auto& bound = expected[{accessIds.at(a), accessIds.at(b), hazard}];
        const fs::OriginDistanceInterval next = a % 2 ? fs::OriginDistanceInterval{true, 4, 7} :
                                                       fs::OriginDistanceInterval{true, 0, std::nullopt};
        if (!bound.reachable) { bound = next; }
        else {
            bound.minimum = std::min(bound.minimum, next.minimum);
            bound.maximum = bound.maximum && next.maximum ?
                std::optional<uint64_t>(std::max(*bound.maximum, *next.maximum)) : std::nullopt;
        }
        return a % 2 ? std::optional<fs::OriginDistanceInterval>(next) : std::nullopt;
    };
    auto restricted = fs::buildConditionalCompactInput(loop, input, index, supplied);
    if (!restricted.error.empty() || restricted.storage.queries.size() != expected.size()) { return false; }
    for (const auto& query : restricted.storage.queries) {
        auto found = expected.find({query.sourceAccess, query.targetAccess, query.hazard});
        if (found == expected.end() || query.bounds.minimum != found->second.minimum ||
            query.bounds.maximum != found->second.maximum) { return false; }
    }
    return true;
}
bool exports(scf::ForOp loop, const pto::SyncInput& input, const fs::PhaseIndex& index,
             const fs::ConditionalCompactInput& built)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    const auto count = arena->input(loop.getUpperBound());
    const auto trips = arena->select(arena->slt(arena->constant(0), count), count, arena->constant(0));
    std::string error;
    auto domain = fs::captureBalancedCompactFixedBodyContext(loop, input, index, arena, trips, error);
    if (!domain || !error.empty()) { return false; }
    auto bounds = fs::buildCompactOrderBounds(domain, built.mathematical, {}, 3);
    if (!bounds.exportError.empty() || !bounds.certificate.error.empty() || !bounds.bounds.upper ||
        !bounds.bounds.lower) { return false; }
    const auto& region = bounds.bounds.upper->regional();
    for (uint32_t type = 0; type < built.body->slots.size(); ++type) {
        if (built.body->slots[type].alternatives.size() > 1 && region.anchors[type].phase) { return false; }
        for (uint64_t ordinal = 0; ordinal <= 3; ++ordinal) {
            const fs::RegionalEvent event{type, arena->constant(ordinal), fs::PeriodicEventKind::Start};
            auto present = fs::regionalPresence(region, event);
            auto reflexive = fs::regionalReachability(region, event, event);
            if (!present || !reflexive) { return false; }
            const std::pair<Expr, Expr> binding{count, arena->constant(3)};
            fs::RegionExpressions::Substitution substitution(binding);
            if (arena->constantValue(arena->substitute(*present, substitution)) != uint64_t(ordinal < 3) ||
                arena->constantValue(arena->substitute(*reflexive, substitution)) != uint64_t(ordinal < 3)) {
                return false;
            }
        }
    }
    return true;
}
bool check(func::FuncOp function, const pto::SyncInput& input, uint64_t& checked)
{
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
    scf::ForOp loop;
    bool multiple = false;
    function.walk([&](scf::ForOp candidate) {
        if (candidate->getParentOfType<scf::ForOp>()) { return; }
        multiple |= bool(loop); loop = candidate;
    });
    if (!loop || multiple) { return false; }
    auto built = fs::buildConditionalCompactInput(loop, input, index);
    if (function->hasAttr("test.balanced_reject")) { return !built.error.empty(); }
    if (!built.error.empty()) { llvm::errs() << built.error << "\n"; return false; }
    if (built.storage.payloads.size() != built.body->slots.size() ||
        built.storage.phaseAlternatives.size() != built.body->slots.size()) { return false; }
    std::set<std::size_t> original, actual;
    bool singleton = true, rmw = false;
    for (const auto& slot : built.body->slots) {
        singleton &= slot.alternatives.size() == 1;
        for (const auto& alternative : slot.alternatives) {
            for (auto effect : input.accesses().effectsFor(alternative.phase)) {
                const auto& record = input.accesses().effects()[effect];
                if (!(record.rangesMaterialized && record.ranges.empty())) { original.insert(effect); }
            }
        }
    }
    for (const auto& group : built.storage.accessEffects) {
        for (auto effect : group) { if (!actual.insert(effect).second) { return false; } }
    }
    for (const auto& access : built.storage.accesses) {
        if (access.fullOverwrite) { return false; }
        rmw |= access.read && access.write;
    }
    if (actual != original || (singleton ? built.storage.phases.size() != built.body->slots.size() :
                                            !built.storage.phases.empty())) { return false; }
    if (function->hasAttr("test.compact_rmw") && !rmw) { return false; }
    auto sizeIs = [&](StringRef attribute, std::size_t size) {
        auto expected = function->getAttrOfType<IntegerAttr>(attribute);
        return !expected || (expected.getInt() >= 0 && uint64_t(expected.getInt()) == size);
    };
    if (!sizeIs("test.compact_classes", built.storage.classEffects.size()) ||
        !sizeIs("test.compact_native", built.storage.native.size()) ||
        !sizeIs("test.compact_additional", built.storage.additional.size())) { return false; }
    if (singleton) {
        auto fixed = fs::buildCompactWriterReaderInput(loop, input, index);
        if (!fixed.error.empty() || fixed.payloads.size() != built.storage.payloads.size() ||
            fixed.queries.size() != built.storage.queries.size() ||
            fixed.native.size() != built.storage.native.size()) {
            return false;
        }
    }
    return coverage(built, input, checked) && bindings(loop, input, index, built) && exports(loop, input, index, built);
}
} // namespace
int runConditionalCompactInputChecks(func::FuncOp function, const pto::SyncInput& input)
{
    uint64_t checked = 0;
    if (!check(function, input, checked)) {
        llvm::errs() << "conditional compact input failed: " << function.getSymName() << "\n";
        return 1;
    }
    llvm::outs() << "conditional compact input passed: " << function.getSymName() << " words=" << checked << "\n";
    return 0;
}
