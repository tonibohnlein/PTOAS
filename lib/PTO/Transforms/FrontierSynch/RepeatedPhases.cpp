// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RepeatedPhases.h"
#include "RepeatedRegionInternal.h"
#include "RepeatedReadOnlyStorage.h"
#include "CountedLoop.h"
#include "SequenceAnalysisInternal.h"
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Selectors = std::vector<RegionalSelector>;
Id any(RegionExpressions& e, const Selectors& values)
{
    auto result = e.boolean(false);
    for (const auto& value : values) { result = e.lor(result, value.present); }
    return result;
}
void restrictSelectors(RegionExpressions& e, Selectors& values, Id guard)
{
    for (auto& value : values) { value.present = e.land(value.present, guard); }
}
void appendFirst(RegionExpressions& e, Selectors& first, Selectors next, Id guard)
{
    restrictSelectors(e, next, e.land(guard, e.lnot(any(e, first))));
    first.insert(first.end(), next.begin(), next.end());
}
void appendLast(RegionExpressions& e, Selectors& last, const Selectors& next, Id guard)
{
    restrictSelectors(e, last, e.land(guard, e.lnot(any(e, next))));
    last.insert(last.end(), next.begin(), next.end());
}
void mergeCell(RegionExpressions& e, RegionalStorageBoundary& left, const RegionalStorageBoundary& right)
{
    auto noLeftWriter = e.lnot(any(e, left.firstWriters));
    auto noRightWriter = e.lnot(any(e, right.firstWriters));
    for (const auto& [pipe, values] : right.firstReaders) {
        appendFirst(e, left.firstReaders[pipe], values, noLeftWriter);
    }
    for (auto& [pipe, values] : left.lastReaders) { restrictSelectors(e, values, noRightWriter); }
    for (const auto& [pipe, values] : right.lastReaders) {
        appendLast(e, left.lastReaders[pipe], values, e.boolean(true));
    }
    appendFirst(e, left.firstWriters, right.firstWriters, e.boolean(true));
    appendLast(e, left.lastWriters, right.lastWriters, e.boolean(true));
}
void liftSelectors(RegionExpressions& e, Selectors& values, uint32_t offset, Id period, Id guard)
{
    for (auto& value : values) {
        value.event.type += offset;
        value.event.visits.insert(value.event.visits.begin(), period);
        value.present = e.land(value.present, guard);
    }
}
// On a certified nonempty singleton interval, every exported occurrence has
// the same absolute outer visit: begin == end - 1. Only selector coordinates
// are canonicalized; original interval guards and repeated query state remain.
bool normalizeSingletonSelectors(RegionalAnalysis& out, Id begin, Id end, uint64_t phases)
{
    auto& e = *out.expressions;
    auto nonempty = e.lt(begin, end);
    // end - 1 is used only under nonempty, which implies end > 0. Its modular
    // value on the empty interval is harmless because every selector is absent.
    auto visit = e.constantValue(begin) == 0 ? e.constant(0) : e.sub(end, e.constant(1));
    auto period = e.div(visit, e.constant(phases));
    auto normalize = [&](RegionalSelector& value) {
        if (value.event.visits.empty()) { return false; }
        value.event.visits.front() = period;
        value.present = e.select(nonempty, value.present, e.boolean(false));
        return true;
    };
    auto list = [&](Selectors& values) { return llvm::all_of(values, normalize); };
    for (auto& cell : out.storageBoundary) {
        if (!list(cell.firstWriters) || !list(cell.lastWriters)) { return false; }
        for (auto* readers : {&cell.firstReaders, &cell.lastReaders}) {
            for (auto& [pipe, values] : *readers) { if (!list(values)) { return false; } }
        }
    }
    for (auto* side : {&out.firstPayloads, &out.lastPayloads, &out.firstSitePayloads}) {
        for (auto& [key, values] : *side) { if (!list(values)) { return false; } }
    }
    for (auto* accesses : {&out.accessBoundary, &out.deferredAccessBoundary}) {
        for (auto& access : *accesses) {
            if (!normalize(access.first) || !normalize(access.last)) { return false; }
        }
    }
    return true;
}
RegionalStorageBoundary liftCell(RegionExpressions& e, RegionalStorageBoundary cell,
    uint32_t offset, Id period, Id guard)
{
    liftSelectors(e, cell.firstWriters, offset, period, guard);
    liftSelectors(e, cell.lastWriters, offset, period, guard);
    for (auto& [pipe, values] : cell.firstReaders) { liftSelectors(e, values, offset, period, guard); }
    for (auto& [pipe, values] : cell.lastReaders) { liftSelectors(e, values, offset, period, guard); }
    return cell;
}
void appendPartial(RegionalAnalysis& out, const std::vector<RegionalAnalysis>& phases,
    const std::vector<uint32_t>& starts, Id periods, Id remainder)
{
    auto& e = *out.expressions;
    for (std::size_t phase = 0; phase < phases.size(); ++phase) {
        auto enabled = e.lt(e.constant(phase), remainder);
        for (auto& destination : out.storageBoundary) {
            for (const auto& source : phases[phase].storageBoundary) {
                if (!sameStorageDomain(destination.cell, source.cell) ||
                    source.cell.begin > destination.cell.begin || source.cell.end < destination.cell.end) {
                        continue;
                    }
                mergeCell(e, destination, liftCell(e, source, starts[phase], periods, enabled));
            }
        }
        for (const auto& [pipe, source] : phases[phase].firstPayloads) {
            auto values = source;
            liftSelectors(e, values, starts[phase], periods, enabled);
            appendFirst(e, out.firstPayloads[pipe], std::move(values), e.boolean(true));
        }
        for (const auto& [pipe, source] : phases[phase].lastPayloads) {
            auto values = source;
            liftSelectors(e, values, starts[phase], periods, enabled);
            appendLast(e, out.lastPayloads[pipe], values, e.boolean(true));
        }
    }
}
// Each invariant phase executes at most once in these boundary windows. The
// first q visits contain the first writer/prefix readers; the last q visits
// contain the last writer/suffix readers, including phases absent for a byte.
// Folding whole per-byte summaries is essential: masking an unclipped first
// writer would not reveal the next writer after an initial partial period.
RegionalStorageSelectors projectStorage(RegionExpressions& e,
    const std::vector<RegionalStorageSelectors>& phases, const std::vector<uint32_t>& starts,
    Id begin, Id end)
{
    const auto q = e.constant(phases.size());
    const auto firstPeriod = e.div(begin, q), firstPhase = e.rem(begin, q);
    const auto lastPeriod = e.div(end, q), lastPhase = e.rem(end, q);
    RegionalStorageBoundary first, last;
    for (unsigned pass = 0; pass < 2; ++pass) {
        auto initial = e.add(firstPeriod, e.constant(pass));
        auto final = pass ? lastPeriod : e.sub(lastPeriod, e.constant(1));
        for (std::size_t phase = 0; phase < phases.size(); ++phase) {
            const auto p = e.constant(phase);
            auto belowEnd = [&](Id period) {
                return e.lor(e.lt(period, lastPeriod), e.land(e.eq(period, lastPeriod), e.lt(p, lastPhase)));
            };
            auto aboveBegin = [&](Id period) {
                return e.lor(e.lt(firstPeriod, period), e.land(e.eq(period, firstPeriod), e.le(firstPhase, p)));
            };
            auto firstGuard = e.land(pass ? e.lt(p, firstPhase) : e.le(firstPhase, p), belowEnd(initial));
            auto lastGuard = e.land(pass ? e.lt(p, lastPhase) : e.le(lastPhase, p),
                e.land(belowEnd(final), aboveBegin(final)));
            const auto& source = phases[phase];
            RegionalStorageBoundary cell{{}, source.firstWriters, source.lastWriters,
                                         source.firstReaders, source.lastReaders};
            mergeCell(e, first, liftCell(e, cell, starts[phase], initial, firstGuard));
            mergeCell(e, last, liftCell(e, std::move(cell), starts[phase], final, lastGuard));
        }
    }
    return {std::move(first.firstWriters), std::move(last.lastWriters),
            std::move(first.firstReaders), std::move(last.lastReaders)};
}
std::optional<RegionalStorageSelectors> phaseStorage(const RegionalAnalysis& phase, RegionalByteAddress address)
{
    if (phase.storageSelectors) { return phase.storageSelectors(address); }
    // A finite-only phase can answer the same byte query through its exact
    // partition. Residual effect relationships are not an absence certificate.
    for (const auto* accesses : {&phase.accessBoundary, &phase.deferredAccessBoundary}) {
        if (llvm::any_of(*accesses, [](const auto& access) { return !access.representedByCells; })) {
            return std::nullopt;
        }
    }
    auto& e = *phase.expressions;
    RegionalStorageSelectors result;
    for (const auto& cell : phase.storageBoundary) {
        if (cell.cell.space != address.space) { continue; }
        if (cell.cell.base != address.base) {
            SmallVector<SyncStorageCell> domains{cell.cell, {address.space, 0, 1, address.base}};
            if (!storageBasesAreComparable(domains, phase.gmAliasPolicy)) { return std::nullopt; }
            continue;
        }
        auto active = e.land(e.le(e.constant(cell.cell.begin), address.offset),
                             e.lt(address.offset, e.constant(cell.cell.end)));
        auto append = [&](Selectors& target, Selectors source) {
            restrictSelectors(e, source, active);
            target.insert(target.end(), source.begin(), source.end());
        };
        append(result.firstWriters, cell.firstWriters); append(result.lastWriters, cell.lastWriters);
        for (const auto& [pipe, values] : cell.firstReaders) { append(result.firstReaders[pipe], values); }
        for (const auto& [pipe, values] : cell.lastReaders) { append(result.lastReaders[pipe], values); }
    }
    return result;
}
} // namespace
RepeatedRegionAnalysis repeatPhasedRegions(func::FuncOp function, scf::ForOp loop,
    std::vector<RegionalAnalysis> phases, Id trips, ArrayRef<scf::ForOp> enclosing, Id begin,
    std::optional<uint64_t> maximumLength)
{
    RepeatedRegionAnalysis failure;
    if (!function || !loop || loop->getParentOfType<func::FuncOp>() != function || phases.empty() ||
        phases.size() > maxRegionalSlotVisits || !phases.front().expressions) {
        failure.error = "repeated region requires a valid loop and supported explicit phase count"; return failure;
    }
    auto arena = phases.front().expressions;
    auto& e = *arena;
    if (trips >= e.size() || e.isBoolean(trips) || !e.constructionError().empty()) {
        failure.error = "repeated phases require a valid integer trip expression"; return failure;
    }
    const auto domain = CountedLoop::get(loop);
    if (!domain) {
        failure.error = "phase endpoint binding requires a represented 64-bit counted domain"; return failure;
    }
    if (begin == RegionExpressions::invalid) { begin = e.constant(0); }
    if (begin >= e.size() || e.isBoolean(begin)) {
        failure.error = "phase interval requires an integer beginning"; return failure;
    }
    const auto q = phases.size();
    auto ordinal = domain->ordinal(e);
    auto period = e.div(ordinal, e.constant(q)), phaseValue = e.rem(ordinal, e.constant(q));
    auto periods = e.div(trips, e.constant(q)), remainder = e.rem(trips, e.constant(q));
    auto firstPeriod = e.div(begin, e.constant(q)), firstPhase = e.rem(begin, e.constant(q));
    auto atOrAbove = [&](Id visit, Id phase) {
        return e.lor(e.lt(firstPeriod, visit), e.land(e.eq(visit, firstPeriod), e.le(firstPhase, phase)));
    };
    std::vector<uint32_t> starts, typePhases;
    for (std::size_t phase = 0; phase < q; ++phase) {
        if (phases[phase].expressions != arena || phases[phase].anchors.size() > UINT32_MAX - typePhases.size()) {
            failure.error = "phase views require a common arena and bounded type identities"; return failure;
        }
        starts.push_back(static_cast<uint32_t>(typePhases.size()));
        typePhases.insert(typePhases.end(), phases[phase].anchors.size(), static_cast<uint32_t>(phase));
        phases[phase].endpointSiteGuard = e.eq(phaseValue, e.constant(phase));
        phases[phase].endpointInvocationGuard = e.land(atOrAbove(period, e.constant(phase)),
            e.lor(e.lt(period, periods), e.land(e.eq(period, periods), e.lt(e.constant(phase), remainder))));
    }
    bool endpoints = llvm::all_of(phases, [](const RegionalAnalysis& phase) {
        return phase.capabilities.endpointRecipes && (phase.prepare || phase.prepareWithVisits);
    });
    const bool symbolic = llvm::any_of(phases, [](const auto& phase) {
        return phase.storageSelectors || !phase.symbolicStorageEffects.empty();
    });
    std::shared_ptr<const std::vector<RegionalAnalysis>> originalPhases;
    if (symbolic) {
        originalPhases = std::make_shared<const std::vector<RegionalAnalysis>>(phases);
        const SyncStorageEffects* model = nullptr;
        std::vector<std::size_t> periodEffects;
        for (const auto& phase : phases) {
            if (phase.accessModel) {
                if (model && (model != phase.accessModel || phase.gmAliasPolicy != phases.front().gmAliasPolicy)) {
                    failure.error = "symbolic phases require one shared effect and alias context"; return failure;
                }
                model = phase.accessModel;
            }
            for (const auto* accesses : {&phase.accessBoundary, &phase.deferredAccessBoundary}) {
                for (const auto& access : *accesses) {
                    if (!phase.accessModel || access.effect >= phase.accessModel->effects().size() ||
                        !validRegionalEvent(phase, access.first.event) ||
                        !validRegionalEvent(phase, access.last.event) || access.first.present >= e.size() ||
                        access.last.present >= e.size() || !e.isBoolean(access.first.present) ||
                        !e.isBoolean(access.last.present)) {
                        failure.error = "symbolic phase has invalid shared effect extrema"; return failure;
                    }
                    // Only consumed sibling extrema discharge collective
                    // coverage. A deferred-only ordinary phase does not enter
                    // the crossing graph. Its missing bridge must remain an
                    // unavailable interface, even if its effect ID is known.
                    if (accesses == &phase.accessBoundary && !llvm::is_contained(periodEffects, access.effect)) {
                        periodEffects.push_back(access.effect);
                    }
                }
            }
        }
        // Keep qualification separate from graph construction. Finite atoms
        // produced here must survive both period composition and clipping.
        auto executed = e.select(e.lt(begin, trips), trips, e.constant(0));
        for (auto& phase : phases) {
            if (!phase.storageSelectors && phase.symbolicStorageEffects.empty()) { continue; }
            failure.error = prepareRepeatedSymbolicStorage(phase, loop, executed, periodEffects);
            if (!failure.error.empty()) { return failure; }
        }
    }
    SmallVector<scf::ForOp> protectionContext(enclosing.begin(), enclosing.end());
    protectionContext.push_back(loop);
    auto composed = composeRegionalSequenceWithin(function, arena, phases, false, false, protectionContext);
    if (!composed.error.empty()) { failure.error = composed.error; return failure; }
    composed.state->completeInvocation = false;
    composed.state->requiredOuterLoops.assign(enclosing.begin(), enclosing.end());
    composed.state->requiredOuterLoops.push_back(loop);
    auto body = sequenceRegionalResult(composed);
    if (!endpoints) {
        body.prepare = {}; body.prepareWithVisits = {}; body.prepareFiltered = {};
        body.capabilities.endpointRecipes = false;
    }
    auto count = e.add(periods, e.select(e.lt(e.constant(0), remainder), e.constant(1), e.constant(0)));
    RepeatedRegionAnalysis result;
    if (maximumLength && *maximumLength <= 1) {
        // Every present endpoint has the same absolute outer visit, hence the
        // same period. Preserve its coordinates but build no wrap graph: the
        // interval masks below exclude all other periods and phases.
        auto state = std::make_shared<RepeatedRegionState>();
        state->function = function; state->loop = loop; state->body = body; state->trips = count;
        state->infinity = 1; state->portEnable = e.boolean(true);
        result = exportRepeatedRegion(std::move(state));
    } else { result = repeatInvariantRegion(function, loop, body, count); }
    if (!result.error.empty()) { return result; }
    result.state->phaseCount = q;
    result.state->originalTrips = trips;
    result.state->originalBegin = begin;
    result.state->typePhases = typePhases;
    auto full = body;
    liftRepeatedSelectors(full, periods);
    auto& out = result.regional;
    out.firstSitePayloads.clear();
    for (std::size_t phase = 0; phase < q; ++phase) {
        auto n = e.add(firstPeriod, e.select(e.lt(e.constant(phase), firstPhase), e.constant(1), e.constant(0)));
        auto exists = e.lor(e.lt(n, periods), e.land(e.eq(n, periods), e.lt(e.constant(phase), remainder)));
        for (const auto& [type, first] : phases[phase].firstSitePayloads) {
            auto values = first;
            liftSelectors(e, values, starts[phase], n, exists);
            out.firstSitePayloads[type + starts[phase]] = std::move(values);
        }
    }
    for (auto& divisors : out.outerDivisors) { divisors.front() = q; }
    auto oldSite = out.endpointEventGuard;
    out.endpointEventGuard = [oldSite, arena, phaseValue, typePhases](RegionalEvent event) -> std::optional<Id> {
        if (event.type >= typePhases.size()) { return std::nullopt; }
        auto other = oldSite ? oldSite(event) : std::optional<Id>(arena->boolean(true));
        if (!other) { return std::nullopt; }
        return arena->land(*other, arena->eq(phaseValue, arena->constant(typePhases[event.type])));
    };
    out.storageBoundary = std::move(full.storageBoundary);
    out.firstPayloads = std::move(full.firstPayloads);
    out.lastPayloads = std::move(full.lastPayloads);
    appendPartial(out, phases, starts, periods, remainder);
    out.numerical.reset();
    out.arithmeticRelations.reset();
    out.symbolicStorage.reset();
    auto originalPresence = out.presence;
    auto originalQuery = out.reachability;
    auto actual = [arena, periods, remainder, firstPeriod, firstPhase, typePhases](RegionalEvent event)
        -> std::optional<Id> {
        if (event.type >= typePhases.size() || event.visits.empty()) { return std::nullopt; }
        auto& e = *arena;
        auto n = event.visits.front();
        auto phase = e.constant(typePhases[event.type]);
        auto above = e.lor(e.lt(firstPeriod, n), e.land(e.eq(n, firstPeriod), e.le(firstPhase, phase)));
        return e.land(above, e.lor(e.lt(n, periods), e.land(e.eq(n, periods), e.lt(phase, remainder))));
    };
    // The first occurrence of every phase is in the initial partial period
    // or the next period. Fold these two finite prefixes in reference order;
    // no number of runtime periods is unfolded.
    std::vector<RegionalStorageBoundary> initial;
    for (const auto& cell : out.storageBoundary) { initial.push_back({cell.cell, {}, {}, {}, {}}); }
    out.firstPayloads.clear();
    for (unsigned pass = 0; pass < 2; ++pass) {
        auto n = e.add(firstPeriod, e.constant(pass));
        for (std::size_t phase = 0; phase < q; ++phase) {
            auto phaseSelected = pass ? e.lt(e.constant(phase), firstPhase) : e.le(firstPhase, e.constant(phase));
            auto exists = e.land(phaseSelected,
                e.lor(e.lt(n, periods), e.land(e.eq(n, periods), e.lt(e.constant(phase), remainder))));
            for (auto& destination : initial) {
                for (const auto& source : phases[phase].storageBoundary) {
                    if (!sameStorageDomain(destination.cell, source.cell) ||
                        source.cell.begin > destination.cell.begin || source.cell.end < destination.cell.end) {
                        continue;
                    }
                    mergeCell(e, destination, liftCell(e, source, starts[phase], n, exists));
                }
            }
            for (const auto& [pipe, source] : phases[phase].firstPayloads) {
                auto values = source;
                liftSelectors(e, values, starts[phase], n, exists);
                appendFirst(e, out.firstPayloads[pipe], std::move(values), e.boolean(true));
            }
        }
    }
    auto maskActual = [&](Selectors& values) {
        for (auto& selector : values) {
            auto exists = actual(selector.event);
            selector.present = e.select(*exists, selector.present, e.boolean(false));
        }
    };
    for (std::size_t cell = 0; cell < out.storageBoundary.size(); ++cell) {
        auto& destination = out.storageBoundary[cell];
        destination.firstWriters = std::move(initial[cell].firstWriters);
        destination.firstReaders = std::move(initial[cell].firstReaders);
        maskActual(destination.lastWriters);
        for (auto& [pipe, readers] : destination.lastReaders) { maskActual(readers); }
    }
    for (auto& [pipe, values] : out.lastPayloads) { maskActual(values); }
    out.presence = [originalPresence, actual, arena](RegionalEvent event) -> std::optional<Id> {
        auto p = originalPresence(event), a = actual(event);
        if (!p || !a) { return std::nullopt; }
        return arena->land(*p, *a);
    };
    out.reachability = [originalQuery, actual, arena](RegionalEvent a, RegionalEvent b) -> std::optional<Id> {
        auto pa = actual(a), pb = actual(b);
        if (!pa || !pb) { return std::nullopt; }
        if (arena->constantValue(*pa) == 0 || arena->constantValue(*pb) == 0) {
            return arena->boolean(false);
        }
        auto value = originalQuery(a, b);
        if (!value) { return std::nullopt; }
        return arena->land(*value, arena->land(*pa, *pb));
    };
    if (originalPhases) {
        out.accessBoundary.clear(); out.deferredAccessBoundary.clear();
        for (std::size_t phase = 0; phase < q; ++phase) {
            const auto& source = (*originalPhases)[phase];
            for (auto effect : source.symbolicStorageEffects) {
                if (!llvm::is_contained(out.symbolicStorageEffects, effect)) {
                    out.symbolicStorageEffects.push_back(effect);
                }
            }
            auto append = [&](auto& target, const auto& accesses) {
                for (auto access : accesses) {
                    for (auto* selector : {&access.first, &access.last}) {
                        selector->event.type += starts[phase];
                        selector->event.visits.insert(selector->event.visits.begin(), e.constant(0));
                    }
                    target.push_back(std::move(access));
                }
            };
            append(out.accessBoundary, source.accessBoundary);
            append(out.deferredAccessBoundary, source.deferredAccessBoundary);
        }
        out.storageSelectors = [originalPhases, arena, starts, begin, trips](RegionalByteAddress address)
            -> std::optional<RegionalStorageSelectors> {
            if (address.offset >= arena->size() || arena->isBoolean(address.offset)) { return std::nullopt; }
            if (arena->constantValue(arena->lt(begin, trips)) == 0) { return RegionalStorageSelectors{}; }
            std::vector<RegionalStorageSelectors> selected;
            for (const auto& phase : *originalPhases) {
                auto selectors = phaseStorage(phase, address);
                if (!selectors) { return std::nullopt; }
                selected.push_back(std::move(*selectors));
            }
            return projectStorage(*arena, selected, starts, begin, trips);
        };
    }
    for (auto* accesses : {&out.accessBoundary, &out.deferredAccessBoundary}) {
        for (auto& access : *accesses) {
            const auto phase = typePhases[access.last.event.type];
            auto first = e.add(firstPeriod,
                e.select(e.lt(e.constant(phase), firstPhase), e.constant(1), e.constant(0)));
            auto exists = e.lor(e.lt(first, periods), e.land(e.eq(first, periods), e.lt(e.constant(phase), remainder)));
            access.first.present = e.land(access.first.present, exists);
            access.last.present = e.land(access.last.present, exists);
            access.first.event.visits.front() = first;
            access.last.event.visits.front() = e.select(e.lt(e.constant(phase), remainder), periods,
                e.sub(periods, e.constant(1)));
        }
    }
    if (maximumLength && *maximumLength <= 1 && !normalizeSingletonSelectors(out, begin, trips, q)) {
        failure.error = "singleton selector lacks its absolute outer coordinate";
        return failure;
    }
    out.cost.phaseDescriptions += q;
    out.cost.expressionNodes = e.size();
    return result;
}
} // namespace mlir::pto::frontiersynch
