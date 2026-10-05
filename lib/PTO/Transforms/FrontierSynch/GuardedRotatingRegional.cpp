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
class Exporter {
public:
    Exporter(func::FuncOp function, const SyncInput& input, std::shared_ptr<GuardedRotatingAnalysis> analysis,
             std::string& error)
        : function(function), input(input), analysis(std::move(analysis)),
          dag(*this->analysis->expressions), error(error) {}
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
        result.gmAliasPolicy = input.memory().gmPolicy();
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
    Expr trips = RegionExpressions::invalid;
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
            const auto slots = mlir::pto::detail::physicalSlotRanges(input, *effect.memory);
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
                auto exists = dag.land(compatible, dag.lt(first, trips));
                const auto final = dag.sub(trips, c(1));
                auto last = dag.sub(final, dag.rem(dag.sub(final, first), c(fragment.refresh)));
                first = dag.select(exists, first, c(0));
                last = dag.select(exists, last, c(0));
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
        const auto nonempty = dag.lt(c(0), trips);
        const auto last = dag.select(nonempty, dag.sub(trips, c(1)), c(0));
        for (uint32_t type = 0; type < analysis->payloads.size(); ++type) {
            const auto& payload = analysis->payloads[type];
            auto present = dag.land(nonempty, payload.presence);
            firsts[payload.pipe].push_back({{type, c(0), PeriodicEventKind::Start}, present});
            lasts[payload.pipe].push_back({{type, last, PeriodicEventKind::Start}, present});
        }
        for (const auto& [pipe, selectors] : firsts) { result.firstPayloads[pipe] = extrema(selectors, true); }
        for (const auto& [pipe, selectors] : lasts) { result.lastPayloads[pipe] = extrema(selectors, false); }
    }
    void queries()
    {
        auto owned = analysis;
        const auto count = trips;
        result.presence = [owned, count](RegionalEvent event) -> std::optional<Expr> {
            auto& expressions = *owned->expressions;
            if (event.type >= owned->payloads.size() || event.ordinal >= expressions.size() ||
                expressions.isBoolean(event.ordinal) || (event.kind != PeriodicEventKind::Start &&
                event.kind != PeriodicEventKind::Completion)) { return std::nullopt; }
            return expressions.land(owned->payloads[event.type].presence, expressions.lt(event.ordinal, count));
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
        result.prepare = [owned, parent]() -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
            std::string error;
            auto prepared = prepareGuardedRotatingEndpoints(parent, *owned, error);
            if (succeeded(prepared)) { (*prepared)->completeInvocation = false; }
            return prepared;
        };
        result.prepareFiltered = [owned, parent](const RegionalDemandFilter& filter)
            -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
            std::string error;
            auto prepared = prepareGuardedRotatingEndpoints(parent, *owned, error, filter);
            if (succeeded(prepared)) { (*prepared)->completeInvocation = false; }
            return prepared;
        };
    }
};
}
FailureOr<RegionalAnalysis> guardedRotatingRegionalResult(func::FuncOp function, const SyncInput& input,
    const GuardedRotatingAnalysis& analysis, std::string& error)
{
    if (!function || !analysis.error.empty() || !analysis.loop || !analysis.expressions ||
        !analysis.periodic.error.empty() || analysis.phases.size() != analysis.payloads.size() ||
        analysis.phases.size() > UINT32_MAX || analysis.periodic.expressions != analysis.expressions) {
        error = "guarded rotating analysis lacks an exact regional contract";
        return failure();
    }
    Exporter exporter(function, input, std::make_shared<GuardedRotatingAnalysis>(analysis), error);
    if (!exporter.run()) { return failure(); }
    return std::move(exporter.result);
}
} // namespace mlir::pto::frontiersynch
