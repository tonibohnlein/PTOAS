// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RotatingRegion.h"
#include "PTO/Transforms/FrontierSynch/BoundingRepetition.h"
#include "RepeatedRegionInternal.h"
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/DifferenceBoundRelations.h"
#include <numeric>
#include <set>
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
Id multiply(RegionExpressions& e, Id value, uint64_t coefficient)
{
    auto result = e.constant(0);
    while (coefficient) {
        if (coefficient & 1) { result = e.add(result, value); }
        coefficient >>= 1;
        if (coefficient) { value = e.add(value, value); }
    }
    return result;
}
Id multiplyModulo(RegionExpressions& e, Id value, uint64_t coefficient, uint64_t modulus)
{
    // Every operand is reduced below modulus <= INT64_MAX, so each sum fits
    // uint64 even when the unreduced product would need 126 bits.
    auto divisor = e.constant(modulus);
    value = e.rem(value, divisor);
    auto result = e.constant(0);
    while (coefficient) {
        if (coefficient & 1) { result = e.rem(e.add(result, value), divisor); }
        coefficient >>= 1;
        if (coefficient) { value = e.rem(e.add(value, value), divisor); }
    }
    return result;
}
uint64_t inverse(uint64_t value, uint64_t modulus)
{
    BoundInteger a(value), b(modulus), x(1), y(0);
    while (b != 0) {
        auto quotient = a / b;
        auto next = a - quotient * b; a = b; b = std::move(next);
        next = x - quotient * y; x = y; y = std::move(next);
    }
    return static_cast<uint64_t>(static_cast<int64_t>(mod(x, BoundInteger(modulus))));
}
struct FamilySelectors {
    RotatingRegionFamily family;
    uint64_t divisor = 1, distance = 1, inverseStride = 0;
    std::vector<RegionalStorageBoundary> cells;
};
struct Storage {
    std::shared_ptr<RegionExpressions> arena;
    Id trips;
    GMAliasPolicy gmAliasPolicy = GMAliasPolicy::MayAlias;
    std::vector<FamilySelectors> families;
    std::shared_ptr<RepeatedStorage> residual;
    std::function<std::optional<Id>(RegionalEvent, RegionalEvent)> before;
    bool normalizeReaders(std::vector<RegionalSelector>& selected, bool first) const
    {
        auto& e = *arena;
        std::vector<RegionalSelector> unique;
        for (const auto& candidate : selected) {
            auto same = llvm::find_if(unique, [&](const auto& other) {
                return candidate.event.type == other.event.type && candidate.event.kind == other.event.kind &&
                    candidate.event.ordinal == other.event.ordinal && candidate.event.visits == other.event.visits;
            });
            if (same == unique.end()) { unique.push_back(candidate); }
            else { same->present = e.lor(same->present, candidate.present); }
        }
        selected = std::move(unique);
        const auto candidates = selected;
        for (auto& candidate : selected) {
            for (const auto& other : candidates) {
                auto earlier = before(first ? other.event : candidate.event, first ? candidate.event : other.event);
                if (!earlier) { return false; }
                candidate.present = e.land(candidate.present, e.lnot(e.land(other.present, *earlier)));
            }
        }
        return true;
    }
    std::optional<RegionalStorageSelectors> query(RegionalByteAddress address) const
    {
        auto& e = *arena;
        if (address.offset >= e.size()) { return std::nullopt; }
        RegionalStorageSelectors result;
        for (const auto& item : families) {
            const auto& f = item.family;
            if (f.firstBank.space != address.space) { continue; }
            if (f.firstBank.base != address.base) {
                SmallVector<SyncStorageCell> domains{f.firstBank, {address.space, 0, 1, address.base}};
                if (!storageBasesAreComparable(domains, gmAliasPolicy)) { return std::nullopt; }
                continue;
            }
            auto relative = e.sub(address.offset, e.constant(f.firstBank.begin));
            auto bank = e.div(relative, e.constant(f.bankStride));
            auto local = e.rem(relative, e.constant(f.bankStride));
            auto delta = e.rem(e.add(bank, e.constant(f.banks - f.offset)), e.constant(f.banks));
            auto first = multiplyModulo(e, e.div(delta, e.constant(item.divisor)),
                                        item.inverseStride, item.distance);
            auto membership = e.land(e.le(e.constant(f.firstBank.begin), address.offset),
                e.land(e.lt(bank, e.constant(f.banks)), e.eq(e.rem(delta, e.constant(item.divisor)), e.constant(0))));
            membership = e.land(membership, e.lt(first, trips));
            auto last = e.add(first, multiply(e, e.div(e.sub(e.sub(trips, e.constant(1)), first),
                                                      e.constant(item.distance)), item.distance));
            const auto canonical = f.firstBank.begin + f.offset * f.bankStride;
            for (const auto& cell : item.cells) {
                auto present = e.land(membership, e.land(e.le(e.constant(cell.cell.begin - canonical), local),
                                                        e.lt(local, e.constant(cell.cell.end - canonical))));
                auto append = [&](auto& destination, const auto& source, bool final) {
                    for (auto selector : source) {
                        selector.present = e.land(present, selector.present);
                        selector.event.visits.insert(selector.event.visits.begin(), final ? last : first);
                        destination.push_back(std::move(selector));
                    }
                };
                append(result.firstWriters, cell.firstWriters, false);
                append(result.lastWriters, cell.lastWriters, true);
                for (const auto& [pipe, values] : cell.firstReaders) {
                    append(result.firstReaders[pipe], values, false);
                }
                for (const auto& [pipe, values] : cell.lastReaders) { append(result.lastReaders[pipe], values, true); }
            }
        }
        if (residual) {
            auto remaining = residual->projectedSelectors(address);
            if (!remaining) { return std::nullopt; }
            auto append = [](auto& target, const auto& source) {
                target.insert(target.end(), source.begin(), source.end());
            };
            append(result.firstWriters, remaining->firstWriters);
            append(result.lastWriters, remaining->lastWriters);
            for (const auto& [pipe, values] : remaining->firstReaders) { append(result.firstReaders[pipe], values); }
            for (const auto& [pipe, values] : remaining->lastReaders) { append(result.lastReaders[pipe], values); }
            // Only read-only families may overlap the checked reservations.
            for (auto& [pipe, values] : result.firstReaders) {
                if (!normalizeReaders(values, true)) { return std::nullopt; }
            }
            for (auto& [pipe, values] : result.lastReaders) {
                if (!normalizeReaders(values, false)) { return std::nullopt; }
            }
        }
        return result;
    }
};
Id any(RegionExpressions& e, ArrayRef<RegionalSelector> values)
{
    auto result = e.boolean(false);
    for (const auto& value : values) { result = e.lor(result, value.present); }
    return result;
}
bool refreshCell(RepeatedRegionState& state, const RegionalStorageBoundary& cell)
{
    auto& e = state.e();
    auto written = e.land(any(e, cell.firstWriters), any(e, cell.lastWriters));
    auto accessed = e.lor(any(e, cell.firstWriters), any(e, cell.lastWriters));
    for (const auto* side : {&cell.firstReaders, &cell.lastReaders}) {
        for (const auto& [pipe, values] : *side) { accessed = e.lor(accessed, any(e, values)); }
    }
    const bool readOnly = cell.firstWriters.empty() && cell.lastWriters.empty();
    if (!readOnly && !e.implies(accessed, written) && e.constantUnder(accessed, written) != 1) {
        state.error = "rotating child does not refresh every accessed local cell in each visit"; return false;
    }
    return true;
}
bool familyBridges(RepeatedRegionState& state, const FamilySelectors& family)
{
    SequenceAnalysisState pair(state.function, state.body.expressions);
    for (auto* parent = state.loop->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto enclosing = dyn_cast<scf::ForOp>(parent)) { pair.requiredOuterLoops.push_back(enclosing); }
    }
    for (unsigned i = 0; i < 2; ++i) {
        Child child;
        child.regional = state.body;
        child.regional.storageBoundary = family.cells;
        child.regional.accessBoundary.clear(); child.regional.deferredAccessBoundary.clear();
        child.regional.storageSelectors = {}; child.regional.symbolicStorageEffects.clear();
        child.regional.symbolicStorage.reset();
        child.regional.arithmeticRelations.reset(); child.regional.relations.reset();
        child.anchors = state.body.anchors;
        pair.children.push_back(std::move(child));
    }
    if (!pair.importSummaries(false)) { state.error = pair.error; return false; }
    pair.bridges();
    if (!pair.error.empty()) { state.error = pair.error; return false; }
    for (const auto& edge : pair.crossings) {
        const auto& source = pair.ports[edge.source];
        const auto& target = pair.ports[edge.target];
        if (source.child != 0 || target.child != 1) {
            state.error = "rotating bridge does not connect consecutive uses of one bank"; return false;
        }
        state.crossings.push_back({source.event(PeriodicEventKind::Completion), target.event(),
                                   edge.guard, false, family.distance});
    }
    return true;
}
} // namespace
RepeatedRegionAnalysis repeatRotatingRegion(func::FuncOp function, scf::ForOp loop,
    RotatingRegionInput input, Id trips)
{
    RepeatedRegionAnalysis failed;
    auto fail = [&](StringRef message) { failed.error = message.str(); return failed; };
    auto& body = input.child;
    if (!function || !loop || loop->getParentOfType<func::FuncOp>() != function ||
        !body.expressions || !body.accessModel || trips >= body.expressions->size() ||
        body.expressions->isBoolean(trips) ||
        !body.capabilities.completeStorageModel || !body.capabilities.exactQueries ||
        !body.capabilities.exactSelectors) {
        return fail("rotating child requires complete finite local selectors and exact child queries");
    }
    // Deferred globally read-only accesses retain exact endpoint identities.
    // Their physical maps must still pass the same residual selector proof.
    for (const auto& access : body.deferredAccessBoundary) {
        if (!llvm::is_contained(input.residualEffects, access.effect)) {
            return fail("rotating deferred access lacks a residual storage classification");
        }
        body.accessBoundary.push_back(access);
    }
    body.deferredAccessBoundary.clear();
    auto state = std::make_shared<RepeatedRegionState>();
    state->function = function; state->loop = loop; state->body = std::move(body); state->trips = trips;
    auto storage = std::make_shared<Storage>(); storage->arena = state->body.expressions; storage->trips = trips;
    storage->gmAliasPolicy = state->body.gmAliasPolicy;
    auto& e = state->e();
    SmallVector<SyncStorageCell> domains;
    std::set<std::size_t> effects;
    std::set<std::size_t> residualEffects(input.residualEffects.begin(), input.residualEffects.end());
    for (auto& family : input.families) {
        if (family.firstBank.begin >= family.firstBank.end) {
            return fail("rotating family has an empty or reversed bank interval");
        }
        const auto width = family.firstBank.end - family.firstBank.begin;
        if (!family.banks || family.banks > INT64_MAX || family.firstBank.begin > INT64_MAX ||
            !family.bankStride || family.bankStride < width ||
            family.firstBank.begin >= family.firstBank.end || family.stride >= family.banks ||
            family.offset >= family.banks || width > uint64_t(INT64_MAX) - family.firstBank.begin ||
            family.banks - 1 > (uint64_t(INT64_MAX) - family.firstBank.begin - width) / family.bankStride) {
            return fail("rotating family has invalid numerical bank geometry");
        }
        SyncStorageCell span{family.firstBank.space, family.firstBank.begin,
            family.firstBank.begin + (family.banks-1)*family.bankStride + width, family.firstBank.base};
        for (std::size_t previous = 0; previous < domains.size(); ++previous) {
            const auto& old = domains[previous];
            if (!sameStorageDomain(old, span) || span.end <= old.begin || old.end <= span.begin) { continue; }
            const auto& other = storage->families[previous].family;
            // Bank starts lie on affine lattices. Holes in their enclosing
            // intervals remain holes: overlap requires a multiple of gcd(S,T)
            // in this exact byte-difference interval. A nonempty intersection
            // is an unavailable proof, not a claim that physical bytes overlap.
            const auto divisor = std::gcd(family.bankStride, other.bankStride);
            const BoundInteger low = BoundInteger(family.firstBank.begin) - BoundInteger(other.firstBank.end) + 1;
            const BoundInteger high = BoundInteger(family.firstBank.end) - BoundInteger(other.firstBank.begin) - 1;
            const BoundInteger first = low + mod(-low, BoundInteger(divisor));
            if (first <= high) { return fail("rotating physical bank families lack a disjointness proof"); }
        }
        domains.push_back(span);
        for (auto id : family.effects) {
            if (id >= state->body.accessModel->effects().size() || residualEffects.count(id) ||
                !effects.insert(id).second) {
                return fail("rotating family effect identity is invalid or classified twice");
            }
        }
        FamilySelectors selectors;
        selectors.divisor = std::gcd(family.stride, family.banks);
        selectors.distance = family.banks / selectors.divisor;
        selectors.inverseStride = inverse(family.stride / selectors.divisor, selectors.distance);
        selectors.family = std::move(family);
        storage->families.push_back(std::move(selectors));
    }
    if (!storageBasesAreComparable(domains, state->body.gmAliasPolicy)) {
        return fail("rotating family physical relationships lack a common partition");
    }
    for (const auto& access : state->body.accessBoundary) {
        if ((!access.representedByCells || !effects.count(access.effect)) && !residualEffects.count(access.effect)) {
            return fail("rotating child has an unclassified access effect");
        }
    }
    for (const auto& anchor : state->body.anchors) {
        if (!anchor.phase) { return fail("rotating child has an invalid original phase"); }
        for (auto id : state->body.accessModel->effectsFor(anchor.phase)) {
            if ((!effects.count(id) && !residualEffects.count(id)) ||
                llvm::none_of(state->body.accessBoundary, [id](const auto& access) { return access.effect == id; })) {
                return fail("rotating child has an unexported original effect");
            }
        }
    }
    for (auto id : state->body.symbolicStorageEffects) {
        if (!residualEffects.count(id)) {
            return fail("rotating child needs finite local selectors for each bank effect");
        }
    }
    for (const auto& cell : state->body.storageBoundary) {
        bool matched = false;
        for (auto& item : storage->families) {
            const auto& f = item.family;
            const auto begin = f.firstBank.begin + f.offset*f.bankStride;
            if (!sameStorageDomain(cell.cell, f.firstBank) || cell.cell.begin < begin ||
                cell.cell.end > begin + f.firstBank.end-f.firstBank.begin) { continue; }
            if (!refreshCell(*state, cell)) { return fail(state->error); }
            item.cells.push_back(cell); matched = true; break;
        }
        if (!matched && residualEffects.empty()) {
            return fail("rotating child cell is outside the selected canonical banks");
        }
    }
    if (!residualEffects.empty()) {
        auto projected = state->body;
        projected.storageBoundary.clear();
        projected.symbolicStorageEffects.assign(residualEffects.begin(), residualEffects.end());
        std::vector<RepeatedPersistentStorage> persistent;
        for (std::size_t i = 0; i < storage->families.size(); ++i) {
            const auto& family = storage->families[i];
            projected.storageBoundary.insert(projected.storageBoundary.end(), family.cells.begin(), family.cells.end());
            const bool readOnly = llvm::all_of(family.family.effects, [&](auto id) {
                return state->body.accessModel->effects()[id].mode == SyncAccessMode::Read;
            });
            persistent.push_back({domains[i], family.family.effects, readOnly});
        }
        for (auto& access : projected.accessBoundary) {
            if (residualEffects.count(access.effect)) { access.representedByCells = false; }
        }
        auto proof = recognizeRepeatedStorageWithPersistent(projected, loop, trips, persistent);
        if (!proof.storage) { return fail("rotating residual ownership: " + proof.error); }
        for (auto id : residualEffects) {
            if (!proof.storage->contains(id)) {
                return fail("rotating residual effect lacks an ownership/read-only proof");
            }
        }
        storage->residual = std::move(proof.storage);
    }
    for (const auto& family : storage->families) {
        if (!familyBridges(*state, family)) { return fail(state->error); }
    }
    state->crossings.insert(state->crossings.end(), input.prerequisites.begin(), input.prerequisites.end());
    if (!state->buildNativeBoundary()) { return fail(state->error); }
    state->queryCrossings = state->crossings;
    using EventKey = std::tuple<uint32_t, Id, PeriodicEventKind, std::vector<Id>>;
    std::map<EventKey, RegionalEvent> uniquePorts;
    for (const auto& edge : state->crossings) {
        for (const auto& event : {edge.source, edge.target}) {
            uniquePorts.emplace(EventKey{event.type, event.ordinal, event.kind, event.visits}, event);
        }
    }
    std::vector<RegionalEvent> ports;
    for (const auto& entry : uniquePorts) { ports.push_back(entry.second); }
    auto weighted = buildBoundingRepeatedQuery(state->body, std::move(ports), state->crossings, state->error);
    if (!weighted) { return fail(state->error); }
    auto covers = weighted->covers();
    if (!covers) { return fail("rotating weighted crossing reduction unavailable"); }
    state->crossings = std::move(*covers);
    state->weightedAcross = [weighted](const RegionalEvent& a, const RegionalEvent& b, Id gap) {
        return weighted->across(a, b, gap);
    };
    auto result = exportRepeatedRegion(state);
    if (!result.error.empty()) { return result; }
    auto& out = result.regional;
    out.storageBoundary.clear(); out.arithmeticRelations.reset(); out.relations.reset(); out.symbolicStorage.reset();
    storage->before = out.referenceBefore;
    out.storageSelectors = [storage](RegionalByteAddress address) { return storage->query(address); };
    effects.insert(residualEffects.begin(), residualEffects.end());
    out.symbolicStorageEffects.assign(effects.begin(), effects.end());
    auto certificate = std::make_shared<RegionalSymbolicStorageCertificate>();
    certificate->expressions = out.expressions; certificate->accessModel = out.accessModel;
    certificate->gmAliasPolicy = out.gmAliasPolicy;
    for (const auto& item : storage->families) {
        RegionalStorageFamily family;
        family.space = item.family.firstBank.space; family.base = item.family.firstBank.base;
        family.effects = item.family.effects;
        family.kind = llvm::all_of(family.effects, [&](auto id) {
            return out.accessModel->effects()[id].mode == SyncAccessMode::Read;
        }) ? RegionalStorageFamilyKind::SharedReadOnly : RegionalStorageFamilyKind::General;
        // Membership is obtained from actual selectors, never the reservation.
        auto one = std::make_shared<Storage>();
        one->arena = storage->arena; one->trips = trips; one->gmAliasPolicy = storage->gmAliasPolicy;
        one->families.push_back(item);
        family.membership = [one](RegionalByteAddress address) -> std::optional<Id> {
            auto selected = one->query(address);
            if (!selected) { return std::nullopt; }
            auto& expressions = *one->arena;
            auto present = any(expressions, selected->firstWriters);
            for (const auto& [pipe, readers] : selected->firstReaders) {
                present = expressions.lor(present, any(expressions, readers));
            }
            return present;
        };
        certificate->families.push_back(std::move(family));
    }
    if (storage->residual) {
        auto residual = storage->residual->certificate();
        certificate->families.insert(certificate->families.end(), residual->families.begin(), residual->families.end());
    }
    out.symbolicStorage = std::move(certificate);
    for (auto& access : out.accessBoundary) { access.representedByCells = false; }
    out.capabilities.completeStorageModel = true; out.capabilities.exactSelectors = true;
    out.cost.ports += uniquePorts.size();
    out.cost.selectorComparisons += weighted->cost().bodyQueries;
    out.cost.numericalIndexOperations += weighted->cost().relaxations;
    out.cost.implicationChecks += weighted->cost().deletionTests;
    out.cost.crossingCandidates += state->queryCrossings.size();
    out.cost.expressionNodes = e.size();
    return result;
}
} // namespace mlir::pto::frontiersynch
