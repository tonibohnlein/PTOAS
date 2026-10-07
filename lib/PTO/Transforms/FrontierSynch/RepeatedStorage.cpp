// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owner inversion and per-byte selectors. Reservation membership is distinct
// from actual access membership: holes retain empty selectors.
#include "RepeatedStorageInternal.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using State = RepeatedStorage::State;
bool validAddress(const State& state, RegionalByteAddress address)
{
    return address.offset < state.expressions().size() && !state.expressions().isBoolean(address.offset);
}
bool distinctDomain(const State& state, AddressSpace space, Value base, RegionalByteAddress address)
{
    if (space != address.space) { return true; }
    if (base == address.base) { return false; }
    SmallVector<SyncStorageCell> domains{{space, 0, 1, base}, {address.space, 0, 1, address.base}};
    return storageBasesAreComparable(domains, state.body.gmAliasPolicy);
}
std::optional<RepeatedStorageOwner> lookupOwner(const State& state, std::size_t id, RegionalByteAddress address)
{
    if (id >= state.families.size() || !validAddress(state, address)) { return std::nullopt; }
    const auto& family = state.families[id];
    auto& e = state.expressions();
    const auto zero = e.constant(0);
    if (distinctDomain(state, family.spec.space, family.spec.origin.base, address)) {
        return RepeatedStorageOwner{e.boolean(false), zero, zero};
    }
    if (family.spec.space != address.space || family.spec.origin.base != address.base) { return std::nullopt; }
    auto present = e.land(e.le(family.origin, address.offset), e.lt(zero, state.trips));
    const auto delta = e.sub(address.offset, family.origin);
    if (family.spec.kind == RepeatedStorageKind::SharedReadOnly) {
        return RepeatedStorageOwner{e.land(present, e.lt(delta, e.constant(family.extent))), zero, delta};
    }
    if (family.extent > family.spec.stride) {
        // A finite union can be disjoint across visits even when its hull is
        // wider than the translation. Invert each interval; the construction
        // proved that overlapping candidates always identify the same visit.
        auto active = e.boolean(false), visit = zero, local = zero;
        const auto stride = e.constant(family.spec.stride);
        const bool single = e.constantValue(state.trips) == uint64_t(1);
        for (const auto& piece : family.pieces) {
            const auto start = e.constant(piece.begin), width = e.constant(piece.end - piece.begin);
            const auto offset = e.sub(delta, start);
            const auto candidate = single ? zero : e.div(offset, stride);
            const auto remainder = single ? offset : e.rem(offset, stride);
            auto matches = e.land(present, e.land(e.le(start, delta),
                e.land(e.lt(candidate, state.trips), e.lt(remainder, width))));
            visit = e.select(matches, candidate, visit);
            local = e.select(matches, e.add(start, remainder), local);
            active = e.lor(active, matches);
        }
        return RepeatedStorageOwner{active, visit, local};
    }
    const auto stride = e.constant(family.spec.stride);
    const auto visit = e.div(delta, stride);
    present = e.land(present, e.lt(visit, state.trips));
    return RepeatedStorageOwner{present, visit, e.rem(delta, stride)};
}
void append(std::vector<RegionalSelector>& out, ArrayRef<RegionalSelector> input,
            Id guard, Id visit, RegionExpressions& e)
{
    for (auto selected : input) {
        selected.present = e.land(guard, selected.present);
        if (e.constantValue(selected.present) == uint64_t(0)) { continue; }
        selected.event.visits.insert(selected.event.visits.begin(), visit);
        out.push_back(std::move(selected));
    }
}
void appendPiece(RegionalStorageSelectors& out, const State::Piece& piece,
                 RepeatedStorageOwner owner, const State::Family& family, const State& state)
{
    auto& e = state.expressions();
    const auto guard = e.land(owner.present, e.land(e.le(e.constant(piece.begin), owner.localByte),
                                                  e.lt(owner.localByte, e.constant(piece.end))));
    if (e.constantValue(guard) == uint64_t(0)) { return; }
    const auto finalVisit = family.spec.kind == RepeatedStorageKind::VisitOwned ? owner.visit :
        e.sub(state.trips, e.constant(1));
    if (piece.mode == SyncAccessMode::Write) {
        append(out.firstWriters, {piece.access.first}, guard, owner.visit, e);
        append(out.lastWriters, {piece.access.last}, guard, finalVisit, e);
    } else {
        append(out.firstReaders[piece.pipe], {piece.access.first}, guard, owner.visit, e);
        append(out.lastReaders[piece.pipe], {piece.access.last}, guard, finalVisit, e);
    }
}
std::optional<Id> before(const State& state, RegionalEvent a, RegionalEvent b)
{
    auto& e = state.expressions();
    const auto i = a.visits.front(), j = b.visits.front();
    a.visits.erase(a.visits.begin());
    b.visits.erase(b.visits.begin());
    auto inner = regionalReferenceBefore(state.body, a, b);
    if (!inner) { return std::nullopt; }
    return e.lor(e.lt(i, j), e.land(e.eq(i, j), *inner));
}
bool extremal(const State& state, std::vector<RegionalSelector>& selected, bool first)
{
    auto& e = state.expressions();
    std::vector<RegionalSelector> unique;
    for (const auto& candidate : selected) {
        auto same = llvm::find_if(unique, [&](const auto& other) {
            return candidate.event.type == other.event.type && candidate.event.ordinal == other.event.ordinal &&
                candidate.event.kind == other.event.kind && candidate.event.visits == other.event.visits;
        });
        if (same == unique.end()) { unique.push_back(candidate); }
        else { same->present = e.lor(same->present, candidate.present); }
    }
    selected = std::move(unique);
    const auto candidates = selected;
    for (auto& candidate : selected) {
        for (const auto& other : candidates) {
            auto a = first ? other.event : candidate.event;
            auto b = first ? candidate.event : other.event;
            auto earlier = before(state, a, b);
            if (!earlier) { return false; }
            candidate.present = e.land(candidate.present, e.lnot(e.land(other.present, *earlier)));
        }
    }
    return true;
}
bool boundaryReaders(const State& state, std::vector<RegionalSelector>& readers,
                     ArrayRef<RegionalSelector> writers, bool first)
{
    auto& e = state.expressions();
    for (auto& reader : readers) {
        for (const auto& writer : writers) {
            auto outside = first ? before(state, reader.event, writer.event) :
                before(state, writer.event, reader.event);
            if (!outside) { return false; }
            reader.present = e.land(reader.present, e.lor(e.lnot(writer.present), *outside));
        }
    }
    return extremal(state, readers, first);
}
bool finish(const State& state, RegionalStorageSelectors& out)
{
    if (!extremal(state, out.firstWriters, true) || !extremal(state, out.lastWriters, false)) { return false; }
    for (auto& [pipe, values] : out.firstReaders) {
        if (!boundaryReaders(state, values, out.firstWriters, true)) { return false; }
    }
    for (auto& [pipe, values] : out.lastReaders) {
        if (!boundaryReaders(state, values, out.lastWriters, false)) { return false; }
    }
    return true;
}
} // namespace
std::optional<RepeatedStorageOwner> RepeatedStorage::owner(std::size_t family, RegionalByteAddress address) const
{
    return lookupOwner(*state, family, address);
}
std::optional<RegionalStorageSelectors> RepeatedStorage::selectors(RegionalByteAddress address) const
{
    if (!validAddress(*state, address)) { return std::nullopt; }
    RegionalStorageSelectors result;
    auto& e = state->expressions();
    for (const auto& access : state->body.accessBoundary) {
        if (contains(access.effect) || access.representedByCells) { continue; }
        const auto& effect = state->body.accessModel->effects()[access.effect];
        if (effect.memory->scope != address.space) { continue; }
        if (effect.regions.empty() || llvm::any_of(effect.regions, [&](const auto& region) {
                return !distinctDomain(*state, effect.memory->scope, region.base, address);
            })) { return std::nullopt; }
    }
    for (std::size_t id = 0; id < state->families.size(); ++id) {
        auto owned = lookupOwner(*state, id, address);
        if (!owned) { return std::nullopt; }
        if (e.constantValue(owned->present) == uint64_t(0)) { continue; }
        for (const auto& piece : state->families[id].pieces) {
            appendPiece(result, piece, *owned, state->families[id], *state);
        }
    }
    for (const auto& cell : state->body.storageBoundary) {
        if (distinctDomain(*state, cell.cell.space, cell.cell.base, address)) { continue; }
        if (cell.cell.space != address.space || cell.cell.base != address.base) { return std::nullopt; }
        const auto guard = e.land(e.lt(e.constant(0), state->trips),
            e.land(e.le(e.constant(cell.cell.begin), address.offset), e.lt(address.offset, e.constant(cell.cell.end))));
        if (e.constantValue(guard) == uint64_t(0)) { continue; }
        const auto zero = e.constant(0), last = e.sub(state->trips, e.constant(1));
        append(result.firstWriters, cell.firstWriters, guard, zero, e);
        append(result.lastWriters, cell.lastWriters, guard, last, e);
        for (const auto& [pipe, values] : cell.firstReaders) {
            append(result.firstReaders[pipe], values, guard, zero, e);
        }
        for (const auto& [pipe, values] : cell.lastReaders) {
            append(result.lastReaders[pipe], values, guard, last, e);
        }
    }
    return finish(*state, result) ? std::optional<RegionalStorageSelectors>(std::move(result)) : std::nullopt;
}
bool RepeatedStorage::contains(std::size_t effect) const { return llvm::is_contained(state->effectIds, effect); }
ArrayRef<std::size_t> RepeatedStorage::effects() const { return state->effectIds; }
const RegionalAnalysis& RepeatedStorage::body() const { return state->body; }
scf::ForOp RepeatedStorage::loop() const { return state->loop; }
RegionExpressions::Id RepeatedStorage::trips() const { return state->trips; }
} // namespace mlir::pto::frontiersynch
