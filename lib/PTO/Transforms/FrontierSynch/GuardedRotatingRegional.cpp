// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite physical-cell boundary selectors for an immutable guarded periodic
// interior. Slot-table expansion is explicit; runtime trips/offsets are circuits.
#include "PTO/Transforms/FrontierSynch/GuardedRotatingRegional.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingInsertion.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/MapVector.h"
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
std::optional<int64_t> constant(Value value)
{
    APInt integer;
    if (!matchPattern(value, m_ConstantInt(&integer)) || !integer.isSignedIntN(64)) { return std::nullopt; }
    return integer.getSExtValue();
}
struct Visit {
    SyncStorageCell cell;
    uint32_t type = 0;
    Expr first = 0, last = 0, read = 0, write = 0;
};
// A numerical skeleton retains the existing cyclic allocation proof after
// interval restriction. Map its canonical record IDs back to the guarded
// producer's recipes; do not infer a budget from a finite sampled execution.
std::shared_ptr<RegionalAllocationSummary> sliceAllocation(const GuardedRotatingAnalysis& analysis,
    RegionExpressions::Id begin, RegionExpressions::Id end)
{
    auto& dag = *analysis.expressions;
    std::vector<PeriodicPayload> payloads;
    std::vector<PeriodicRecord> records, native;
    for (const auto& payload : analysis.payloads) {
        if (dag.constantValue(payload.presence) != 1) { return {}; }
        payloads.push_back({payload.pipe});
    }
    for (const auto& edge : analysis.periodic.nativePrerequisites) {
        auto enabled = dag.constantValue(edge.active), distance = dag.constantValue(edge.displacement);
        if (!enabled || !distance) { return {}; }
        if (*enabled) { native.push_back({edge.source, edge.target, *distance}); }
    }
    std::map<std::tuple<uint32_t, uint32_t, uint64_t>, uint32_t> original;
    for (uint32_t i = 0; i < analysis.generators.size(); ++i) {
        const auto& edge = analysis.generators[i];
        auto retained = dag.constantValue(analysis.periodic.retained[i]);
        if (!retained) { return {}; }
        if (!*retained) { continue; }
        auto distance = dag.constantValue(edge.displacement);
        if (!distance) { return {}; }
        records.push_back({edge.source, edge.target, *distance});
        if (!original.emplace(std::make_tuple(edge.source, edge.target, *distance), i).second) { return {}; }
    }
    auto numerical = analyzePeriodicDemands(payloads, records, native);
    auto allocation = buildPeriodicAllocation(numerical);
    if (!numerical.error.empty() || !allocation.error.empty()) { return {}; }
    auto out = std::make_shared<RegionalAllocationSummary>();
    for (const auto& direction : allocation.directions) {
        if (!direction.uniformBudget || !*direction.uniformBudget) { return {}; }
        RegionalAllocationGroup group{direction.sourcePipe, direction.targetPipe, *direction.uniformBudget, {}};
        for (auto [phase, handoff] : llvm::enumerate(direction.handoffs)) {
            auto source = RegionalEvent{handoff.source, begin, PeriodicEventKind::Start};
            auto target = RegionalEvent{handoff.target, dag.sub(end, dag.constant(1)), PeriodicEventKind::Completion};
            auto active = dag.land(dag.lt(begin, end),
                dag.lt(dag.constant(handoff.displacement), dag.sub(end, begin)));
            auto record = original.find(std::make_tuple(handoff.source, handoff.target, handoff.displacement));
            if (record == original.end()) { return {}; }
            group.members.push_back({record->second, direction.handoffs.size(), phase, source, target, active});
        }
        out->groups.push_back(std::move(group));
    }
    return out;
}
class Exporter {
public:
    Exporter(func::FuncOp function, const SyncInput& input, std::shared_ptr<GuardedRotatingAnalysis> analysis,
             std::string& error, std::optional<PeriodicSlice> slice)
        : function(function), input(input), analysis(std::move(analysis)),
          dag(*this->analysis->expressions), error(error), slice(slice) {}
    RegionalAnalysis result;
    bool run()
    {
        if (!domain() || !visits()) { return false; }
        storage();
        native();
        queries();
        result.cost.children = 1;
        result.cost.cells = result.storageBoundary.size();
        result.cost.physicalFragments = patterns.size();
        result.cost.expressionNodes = dag.size();
        result.expressions = analysis->expressions;
        result.firstOrdinal = begin;
        result.gmAliasPolicy = input.memory().gmPolicy();
        result.accessModel = &input.accesses();
        for (uint32_t type = 0; type < analysis->phases.size(); ++type) {
            auto present = dag.land(dag.lt(begin, trips), analysis->payloads[type].presence);
            RegionalSelector first{{type, begin, PeriodicEventKind::Start}, present};
            RegionalSelector last{{type, dag.sub(trips, c(1)), PeriodicEventKind::Start}, present};
            for (auto effect : input.accesses().effectsFor(analysis->phases[type])) {
                if (llvm::is_contained(analysis->dischargedEffects, effect)) {
                    result.deferredAccessBoundary.push_back({effect, first, last, false});
                    continue;
                }
                result.accessBoundary.push_back({effect, first, last,
                    !input.accesses().effects()[effect].regions.empty()});
            }
        }
        result.capabilities = {true, true, true, true, true};
        if (!dag.constructionError().empty()) { return fail(dag.constructionError()); }
        return true;
    }
private:
    func::FuncOp function;
    const SyncInput& input;
    std::shared_ptr<GuardedRotatingAnalysis> analysis;
    RegionExpressions& dag;
    std::string& error;
    Expr trips = RegionExpressions::invalid, begin = RegionExpressions::invalid;
    std::optional<PeriodicSlice> slice;
    std::vector<Visit> patterns;
    Expr c(uint64_t value) { return dag.constant(value); }
    bool fail(const std::string& message) { error = message; return false; }
    bool domain()
    {
        auto loop = analysis->loop;
        auto lower = constant(loop.getLowerBound()), step = constant(loop.getStep());
        if (!lower || *lower < 0 || !step || *step <= 0 || loop.getNumResults() ||
            loop->getParentOfType<func::FuncOp>() != function) {
            return fail("guarded regional export requires one original counted loop with constant nonnegative start");
        }
        const auto upper = dag.input(loop.getUpperBound());
        const auto span = dag.select(dag.slt(c(*lower), upper), dag.sub(upper, c(*lower)), c(0));
        trips = dag.add(dag.div(span, c(*step)), dag.select(dag.eq(dag.rem(span, c(*step)), c(0)), c(0), c(1)));
        begin = c(0);
        if (slice) {
            if (slice->begin >= dag.size() || slice->end >= dag.size() ||
                dag.isBoolean(slice->begin) || dag.isBoolean(slice->end)) {
                return fail("periodic slice requires integer endpoints in its expression arena");
            }
            begin = slice->begin;
            trips = slice->end;
        }
        for (const auto* phase : analysis->phases) {
            if (!phase || !phase->elementOp || !loop->isProperAncestor(phase->elementOp)) {
                return fail("guarded regional anchor is outside its original loop");
            }
            auto* operation = phase->elementOp;
            result.anchors.push_back({phase, {}, {operation->getBlock(), operation},
                                      {operation->getBlock(), operation->getNextNode()}});
            result.occurrenceLoops.push_back(loop);
        }
        return true;
    }
    Expr multiplyModulo(Expr value, uint64_t factor, uint64_t modulus)
    {
        auto output = c(0);
        factor %= modulus;
        while (factor) {
            if (factor & 1) { output = dag.rem(dag.add(output, value), c(modulus)); }
            factor >>= 1;
            if (factor) { value = dag.rem(dag.add(value, value), c(modulus)); }
        }
        return output;
    }
    bool visits()
    {
        uint64_t slotVisits = 0;
        for (const auto& fragment : analysis->fragments) {
            if (fragment.slots > maxRegionalSlotVisits - slotVisits) {
                return fail("regional physical slot expansion exceeds the supported visit limit");
            }
            slotVisits += fragment.slots;
        }
        for (const auto& fragment : analysis->fragments) {
            if (fragment.effect >= input.accesses().effects().size() || fragment.payload >= analysis->payloads.size() ||
                !fragment.slots || fragment.slots > INT64_MAX || !fragment.divisor || !fragment.refresh ||
                fragment.divisor > fragment.slots || fragment.refresh != fragment.slots / fragment.divisor ||
                fragment.slots % fragment.divisor || fragment.slotAtom.first >= fragment.slotAtom.second ||
                fragment.offset >= dag.size() || dag.isBoolean(fragment.offset) ||
                !dag.isBoolean(fragment.read) || !dag.isBoolean(fragment.write)) {
                return fail("guarded regional fragment has an invalid exact slot description");
            }
            const auto& effect = input.accesses().effects()[fragment.effect];
            if (!effect.memory || effect.phase != analysis->phases[fragment.payload]) {
                return fail("guarded regional fragment does not belong to its original effect");
            }
            auto slots = mlir::pto::detail::physicalSlotRanges(input, *effect.memory);
            if (fragment.firstPhysicalSlot) {
                const auto first = *fragment.firstPhysicalSlot;
                if (!fragment.physicalSlotStride || first.end < first.begin ||
                    fragment.slots - 1 > (UINT64_MAX - first.end) / fragment.physicalSlotStride) {
                    return fail("normalized physical slot family overflows its address space");
                }
                slots.clear();
                for (uint64_t slot = 0; slot < fragment.slots; ++slot) {
                    const auto offset = slot * fragment.physicalSlotStride;
                    slots.push_back({first.space, first.begin + offset, first.end + offset, first.base});
                }
            }
            if (slots.size() != fragment.slots ||
                (effect.selection && effect.selection->addresses.size() != fragment.slots)) {
                return fail("guarded regional export requires a finite encoded physical slot table");
            }
            for (std::size_t slot = 0; slot < slots.size(); ++slot) {
                const auto& physical = slots[slot];
                if (physical.begin > physical.end || fragment.slotAtom.second > physical.end - physical.begin ||
                    (effect.selection && effect.selection->addresses[slot] != physical.begin)) {
                    return fail("guarded regional slot atom lies outside its physical slot");
                }
                ++result.cost.rotatingResidues;
                // stride*j+offset == slot (mod S). The gcd test selects the
                // orbit; its modular inverse gives the first j in [0,refresh).
                // Both residue addends are <S<=INT64_MAX, so addition fits u64.
                auto difference = dag.rem(dag.sub(dag.add(c(slot), c(fragment.slots)), fragment.offset),
                                          c(fragment.slots));
                auto compatible = dag.eq(dag.rem(difference, c(fragment.divisor)), c(0));
                auto first = multiplyModulo(dag.div(difference, c(fragment.divisor)),
                                            fragment.inverseStride, fragment.refresh);
                const auto residue = dag.rem(begin, c(fragment.refresh));
                first = dag.add(begin, dag.rem(dag.sub(dag.add(first, c(fragment.refresh)), residue),
                                               c(fragment.refresh)));
                auto exists = dag.land(compatible, dag.lt(first, trips));
                const auto final = dag.sub(trips, c(1));
                auto last = dag.sub(final, dag.rem(dag.sub(final, first), c(fragment.refresh)));
                // Coordinates are total modular expressions. Presence guards
                // their use; replacing absent coordinates by zero would hide
                // equality with native boundary coordinates and create ports.
                patterns.push_back({{physical.space, physical.begin + fragment.slotAtom.first,
                                     physical.begin + fragment.slotAtom.second, physical.base},
                    fragment.payload, first, last, dag.land(exists, fragment.read), dag.land(exists, fragment.write)});
            }
        }
        return true;
    }
    Expr before(const RegionalSelector& a, const RegionalSelector& b)
    {
        ++result.cost.selectorComparisons;
        return dag.lor(dag.lt(a.event.ordinal, b.event.ordinal),
            dag.land(dag.eq(a.event.ordinal, b.event.ordinal), dag.boolean(a.event.type < b.event.type)));
    }
    std::vector<RegionalSelector> extrema(const std::vector<RegionalSelector>& candidates, bool first)
    {
        std::vector<RegionalSelector> output;
        for (auto selected : candidates) {
            for (const auto& other : candidates) {
                const auto earlier = first ? before(other, selected) : before(selected, other);
                selected.present = dag.land(selected.present, dag.lnot(dag.land(other.present, earlier)));
            }
            if (dag.constantValue(selected.present) != 0) { output.push_back(selected); }
        }
        return output;
    }
    void summarize(SyncStorageCell cell)
    {
        std::vector<RegionalSelector> firstWrites, lastWrites, firstReads, lastReads;
        for (const auto& pattern : patterns) {
            if (!sameStorageDomain(pattern.cell, cell) || pattern.cell.begin > cell.begin ||
                pattern.cell.end < cell.end) {
                continue;
            }
            auto selected = [&](Expr ordinal, Expr present) {
                return RegionalSelector{{pattern.type, ordinal, PeriodicEventKind::Start}, present};
            };
            firstWrites.push_back(selected(pattern.first, pattern.write));
            lastWrites.push_back(selected(pattern.last, pattern.write));
            firstReads.push_back(selected(pattern.first, pattern.read));
            lastReads.push_back(selected(pattern.last, pattern.read));
        }
        if (firstWrites.empty() && firstReads.empty()) { return; }
        RegionalStorageBoundary boundary;
        boundary.cell = cell;
        boundary.firstWriters = extrema(firstWrites, true);
        boundary.lastWriters = extrema(lastWrites, false);
        auto readers = [&](std::vector<RegionalSelector> reads, const std::vector<RegionalSelector>& writers,
                           bool first, std::map<uint32_t, std::vector<RegionalSelector>>& destination) {
            std::map<uint32_t, std::vector<RegionalSelector>> byPipe;
            for (auto read : reads) {
                for (const auto& writer : writers) {
                    const auto ordered = first ? before(read, writer) : before(writer, read);
                    read.present = dag.land(read.present, dag.lnot(dag.land(writer.present, dag.lnot(ordered))));
                }
                byPipe[analysis->payloads[read.event.type].pipe].push_back(read);
            }
            for (const auto& [pipe, candidates] : byPipe) { destination[pipe] = extrema(candidates, first); }
        };
        // A read at a writer occurrence is excluded by the strict comparison;
        // the previous-writer obligation supplies its read half (RMW semantics).
        readers(firstReads, firstWrites, true, boundary.firstReaders);
        readers(lastReads, lastWrites, false, boundary.lastReaders);
        result.storageBoundary.push_back(std::move(boundary));
    }
    void storage()
    {
        std::map<AddressSpace, llvm::MapVector<Value, std::set<uint64_t>>> endpoints;
        for (const auto& pattern : patterns) {
            endpoints[pattern.cell.space][pattern.cell.base].insert(pattern.cell.begin);
            endpoints[pattern.cell.space][pattern.cell.base].insert(pattern.cell.end);
        }
        for (const auto& [space, bases] : endpoints) {
            for (const auto& [base, points] : bases) {
                for (auto position = points.begin(); position != points.end() && std::next(position) != points.end();
                     ++position) {
                    summarize({space, *position, *std::next(position), base});
                }
            }
        }
    }
    void native()
    {
        std::map<uint32_t, std::vector<RegionalSelector>> firsts, lasts;
        const auto nonempty = dag.lt(begin, trips);
        const auto last = dag.sub(trips, c(1));
        for (uint32_t type = 0; type < analysis->payloads.size(); ++type) {
            const auto& payload = analysis->payloads[type];
            auto present = dag.land(nonempty, payload.presence);
            firsts[payload.pipe].push_back({{type, begin, PeriodicEventKind::Start}, present});
            lasts[payload.pipe].push_back({{type, last, PeriodicEventKind::Start}, present});
        }
        for (const auto& [pipe, selectors] : firsts) { result.firstPayloads[pipe] = extrema(selectors, true); }
        for (const auto& [pipe, selectors] : lasts) { result.lastPayloads[pipe] = extrema(selectors, false); }
    }
    void queries()
    {
        auto owned = analysis;
        const auto count = trips, firstOrdinal = begin;
        result.presence = [owned, count, firstOrdinal](RegionalEvent event) -> std::optional<Expr> {
            auto& expressions = *owned->expressions;
            if (event.type >= owned->payloads.size() || event.ordinal >= expressions.size() ||
                expressions.isBoolean(event.ordinal) || (event.kind != PeriodicEventKind::Start &&
                event.kind != PeriodicEventKind::Completion)) { return std::nullopt; }
            return expressions.land(owned->payloads[event.type].presence,
                expressions.land(expressions.le(firstOrdinal, event.ordinal), expressions.lt(event.ordinal, count)));
        };
        auto presence = result.presence;
        result.reachability = [owned, presence](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
            auto first = presence(a), second = presence(b);
            if (!first || !second) { return std::nullopt; }
            auto threshold = owned->periodic.eventThreshold({a.type, a.kind}, {b.type, b.kind});
            if (!threshold) { return std::nullopt; }
            auto& expressions = *owned->expressions;
            return expressions.land(expressions.land(*first, *second),
                expressions.land(threshold->reachable, expressions.land(expressions.le(a.ordinal, b.ordinal),
                    expressions.le(threshold->distance, expressions.sub(b.ordinal, a.ordinal)))));
        };
        const auto parent = function;
        const auto allocation = sliceAllocation(*owned, begin, trips);
        RegionalDemandFilter domainFilter;
        if (slice) {
            domainFilter = [presence, owned](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
                auto first = presence(a), second = presence(b);
                if (!first || !second) { return std::nullopt; }
                return owned->expressions->land(*first, *second);
            };
        }
        result.prepare = [owned, parent, domainFilter, allocation]()
            -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
            std::string error;
            auto prepared = prepareGuardedRotatingEndpoints(parent, *owned, error, domainFilter);
            if (succeeded(prepared)) {
                (*prepared)->completeInvocation = false;
                (*prepared)->regionalAllocation = allocation;
            }
            return prepared;
        };
        result.prepareFiltered = [owned, parent, domainFilter](const RegionalDemandFilter& filter)
            -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
            std::string error;
            RegionalDemandFilter combined = [owned, domainFilter, filter](RegionalEvent a, RegionalEvent b)
                -> std::optional<Expr> {
                auto first = domainFilter ? domainFilter(a, b) :
                    std::optional<Expr>(owned->expressions->boolean(true));
                auto second = filter ? filter(a, b) : std::optional<Expr>(owned->expressions->boolean(true));
                if (!first || !second) { return std::nullopt; }
                return owned->expressions->land(*first, *second);
            };
            auto prepared = prepareGuardedRotatingEndpoints(parent, *owned, error, combined);
            if (succeeded(prepared)) { (*prepared)->completeInvocation = false; }
            return prepared;
        };
    }
};
}
FailureOr<RegionalAnalysis> guardedRotatingRegionalResult(func::FuncOp function, const SyncInput& input,
    const GuardedRotatingAnalysis& analysis, std::string& error, std::optional<PeriodicSlice> slice)
{
    if (!function || !analysis.error.empty() || !analysis.loop || !analysis.expressions ||
        !analysis.periodic.error.empty() || analysis.phases.size() != analysis.payloads.size() ||
        analysis.phases.size() > UINT32_MAX || analysis.periodic.expressions != analysis.expressions) {
        error = "guarded rotating analysis lacks an exact regional contract";
        return failure();
    }
    Exporter exporter(function, input, std::make_shared<GuardedRotatingAnalysis>(analysis), error, slice);
    if (!exporter.run()) { return failure(); }
    return std::move(exporter.result);
}
} // namespace mlir::pto::frontiersynch
