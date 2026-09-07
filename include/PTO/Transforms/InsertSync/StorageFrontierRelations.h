// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERRELATIONS_H
#define PTO_TRANSFORMS_INSERTSYNC_STORAGEFRONTIERRELATIONS_H

#include "PTO/Transforms/InsertSync/StorageFrontierDomain.h"
#include <map>
#include <numeric>
#include <set>

namespace mlir::pto::insert_sync_frontier {

// Relations refer to dynamic occurrences. They do not identify all executions
// of a static operation with one generation, and do not invent a flattened IV.
struct IterationDelta {
    unsigned loop = kInvalid;
    int64_t targetMinusSource = 0;
};
struct OccurrenceRelation {
    enum class Kind { SameInstance, KnownDistance, OrderedUnknown };
    Kind kind = Kind::OrderedUnknown;
    std::vector<IterationDelta> deltas;
    // SameInstance is meaningful only in this explicitly established domain.
    std::vector<unsigned> commonLoops;
    std::optional<int64_t> distance(unsigned loop) const
    {
        for (
            const auto& d : deltas) {
            if (
                d.loop == loop) {
                return d.targetMinusSource;
            }
        }
        if (kind == Kind::SameInstance &&
            std::find(commonLoops.begin(), commonLoops.end(), loop) != commonLoops.end()) {
            return 0;
        }
        return std::nullopt;
    }
};

// 'scope' is the dynamic SSA definition lifetime, NOT the lexical location of
// its use. Invariant conditions can remain correlated across an inner loop;
// a loop-local definition must not be correlated with its next occurrence.
struct GuardFact {
    unsigned expression = kInvalid;
    unsigned scope = kInvalid; // function/invariant when invalid
    int64_t value = 0;
    bool equal = true;
};
inline bool incompatibleGuards(
    const std::vector<GuardFact>& a, const std::vector<GuardFact>& b, const OccurrenceRelation& relation)
{
    for (
        const auto& x : a) {
        for (
            const auto& y : b) {
            if (
                x.expression == kInvalid || x.expression != y.expression || x.scope != y.scope) {
                continue;
            }
            if (
                x.scope != kInvalid && relation.distance(x.scope) != std::optional<int64_t>(0)) {
                continue;
            }
            if (
                x.value == y.value && x.equal != y.equal) {
                return true;
            }
            if (
                x.equal && y.equal && x.value != y.value) {
                return true;
            }
        }
    }
    return false;
}

struct IndexTerm {
    unsigned loop = kInvalid;
    int64_t coefficient = 0;
};
struct AffineIndex {
    int64_t constant = 0;
    std::vector<IndexTerm> terms;
    // Set only by a range/no-wrap proof or by a guard retained in the final plan.
    bool arithmeticQualified = false;
};
inline bool checkedSigned(int64_t a, int64_t b, bool multiply, int64_t& out)
{
    // Both products and sums fit a signed 128-bit intermediate for i64 operands.
    const __int128 v = multiply ? __int128(a) * b : __int128(a) + b;
    if (
        v < std::numeric_limits<int64_t>::min() || v > std::numeric_limits<int64_t>::max()) {
        return false;
    }
    out = static_cast<int64_t>(v);
    return true;
}
inline std::optional<AffineIndex> normalizeIndex(AffineIndex a)
{
    if (
        !a.arithmeticQualified || a.terms.size() > 32) {
        return std::nullopt;
    }
    std::map<unsigned, int64_t> terms;
    for (
        const auto& term : a.terms) {
        if (
            term.loop == kInvalid || !checkedSigned(terms[term.loop], term.coefficient, false, terms[term.loop])) {
            return std::nullopt;
        }
    }
    a.terms.clear();
    for (
        auto [loop, coefficient] : terms) {
        if (
            coefficient) {
            a.terms.push_back({loop, coefficient});
        }
    }
    return a;
}
inline std::optional<int64_t> constantDifference(
    const AffineIndex& source, const AffineIndex& target, const OccurrenceRelation& relation)
{
    auto a = normalizeIndex(source), b = normalizeIndex(target);
    if (
        !a || !b || a->terms.size() != b->terms.size()) {
        return std::nullopt;
    }
    __int128 difference = __int128(b->constant) - a->constant;
    for (
        unsigned i = 0; i < a->terms.size(); ++i) {
        if (
            a->terms[i].loop != b->terms[i].loop || a->terms[i].coefficient != b->terms[i].coefficient) {
            return std::nullopt;
        }
        auto d = relation.distance(a->terms[i].loop);
        if (
            !d) {
            return std::nullopt;
        }
        difference += __int128(a->terms[i].coefficient) * *d;
        if (
            difference < std::numeric_limits<int64_t>::min() || difference > std::numeric_limits<int64_t>::max()) {
            return std::nullopt;
        }
    }
    if (
        difference < std::numeric_limits<int64_t>::min() || difference > std::numeric_limits<int64_t>::max()) {
        return std::nullopt;
    }
    return static_cast<int64_t>(difference);
}

struct SlotMap {
    AffineIndex selector; // actual selector normalized to logical loop coordinates
    unsigned modulus = 0;
    std::vector<uint64_t> addresses; // arbitrary physical order, not necessarily contiguous
    uint64_t extent = 0;
};
struct AccessSlice {
    unsigned owner = 0, space = 0, root = kInvalid;
    bool global = false;
    bool read = false, write = false;
    bool known = false;
    AffineIndex byteStart;
    uint64_t extent = 0;
    std::optional<SlotMap> slots;
    std::vector<GuardFact> guard;
};
struct AccessRelationResult {
    enum class Kind { Disjoint, Conflict, Unknown, LimitExceeded };
    enum class Reason {
        NoWrite,
        OwnerOrSpace,
        ExclusiveGuard,
        CallerContract,
        ByteIntervals,
        PhysicalSlots,
        UnqualifiedArithmetic,
        UnknownRelation
    };
    Kind kind = Kind::Unknown;
    Reason reason = Reason::UnknownRelation;
    unsigned cases = 0;
};
inline bool disjointBytes(uint64_t a, uint64_t an, uint64_t b, uint64_t bn)
{
    if (
        !an || !bn || a > std::numeric_limits<uint64_t>::max() - an || b > std::numeric_limits<uint64_t>::max() - bn) {
        return false;
    }
    return a + an <= b || b + bn <= a;
}
inline unsigned residue(__int128 value, unsigned modulus)
{
    value %= modulus;
    if (
        value < 0) {
        value += modulus;
    }
    return static_cast<unsigned>(value);
}

// Exhaust a PROVED FINITE residue domain (not a trip-count horizon). Different
// physical mappings are compared by bytes even when their slot numbers differ.
inline AccessRelationResult compareSlotMaps(
    const SlotMap& a, const SlotMap& b, const OccurrenceRelation& relation, Budget& budget)
{
    using R = AccessRelationResult;
    auto x = normalizeIndex(a.selector), y = normalizeIndex(b.selector);
    if (
        !x || !y || !a.modulus || !b.modulus || a.modulus > 16 || b.modulus > 16 || a.addresses.size() != a.modulus ||
        b.addresses.size() != b.modulus || !a.extent || !b.extent) {
        return {R::Kind::Unknown, R::Reason::UnqualifiedArithmetic};
    }
    std::set<unsigned> loops;
    for (
        auto t : x->terms) {
        loops.insert(t.loop);
    }
    for (
        auto t : y->terms) {
        loops.insert(t.loop);
    }
    if (
        loops.size() > 4) {
        return {R::Kind::Unknown, R::Reason::UnknownRelation};
    }
    const unsigned period = std::lcm(a.modulus, b.modulus);
    std::vector<unsigned> ids(loops.begin(), loops.end());
    uint64_t cases = 1;
    for (
        unsigned loop : ids) {
        if (
            !relation.distance(loop)) {
            return {R::Kind::Unknown, R::Reason::UnknownRelation};
        }
        if (
            cases > 65536 / period) {
            return {R::Kind::LimitExceeded, R::Reason::PhysicalSlots};
        }
        cases *= period;
    }
    if (
        !budget.spend(cases)) {
        return {R::Kind::LimitExceeded, R::Reason::PhysicalSlots};
    }
    bool overlap = false, separate = false;
    for (
        uint64_t n = 0; n < cases; ++n) {
        uint64_t digits = n;
        std::map<unsigned, unsigned> values;
        for (
            unsigned loop : ids) {
            values[loop] = digits % period;
            digits /= period;
        }
        __int128 u = x->constant, v = y->constant;
        for (
            auto t : x->terms) {
            u += __int128(residue(t.coefficient, a.modulus)) * values[t.loop];
        }
        for (
            auto t : y->terms) {
            const unsigned shifted = residue(__int128(values[t.loop]) + *relation.distance(t.loop), b.modulus);
            v += __int128(residue(t.coefficient, b.modulus)) * shifted;
        }
        const bool disjoint =
            disjointBytes(a.addresses[residue(u, a.modulus)], a.extent, b.addresses[residue(v, b.modulus)], b.extent);
        separate |= disjoint;
        overlap |= !disjoint;
    }
    return {
        overlap ? (separate ? R::Kind::Unknown : R::Kind::Conflict) : R::Kind::Disjoint, R::Reason::PhysicalSlots,
        static_cast<unsigned>(cases)};
}

inline AccessRelationResult compareAccesses(
    const AccessSlice& a, const AccessSlice& b, const OccurrenceRelation& relation, bool callerProvesRootsDisjoint,
    Budget& budget)
{
    using R = AccessRelationResult;
    if (
        !a.write && !b.write) {
        return {R::Kind::Disjoint, R::Reason::NoWrite};
    }
    if (
        a.space != b.space || (!a.global && !b.global && a.owner != b.owner)) {
        return {R::Kind::Disjoint, R::Reason::OwnerOrSpace};
    }
    if (
        incompatibleGuards(a.guard, b.guard, relation)) {
        return {R::Kind::Disjoint, R::Reason::ExclusiveGuard};
    }
    if (
        a.global && b.global && a.root != b.root && callerProvesRootsDisjoint) {
        return {R::Kind::Disjoint, R::Reason::CallerContract};
    }
    if (
        !a.known || !b.known || a.global != b.global || !a.extent || !b.extent) {
        return {};
    }
    if (
        a.slots && b.slots && !a.global) {
        return compareSlotMaps(*a.slots, *b.slots, relation, budget);
    }
    if (
        a.slots || b.slots || (a.global && (a.root == kInvalid || a.root != b.root))) {
        return {};
    }
    auto difference = constantDifference(a.byteStart, b.byteStart, relation);
    if (
        !difference) {
        return {R::Kind::Unknown, R::Reason::UnknownRelation};
    }
    const bool disjoint =
        __int128(*difference) >= __int128(a.extent) || __int128(*difference) + __int128(b.extent) <= 0;
    return {disjoint ? R::Kind::Disjoint : R::Kind::Conflict, R::Reason::ByteIntervals};
}

// Source/target and storage witness stay immutable when synchronization moves.
struct RequirementWitness {
    Requirement obligation;
    unsigned sourceAccess = kInvalid, targetAccess = kInvalid;
    OccurrenceRelation relation;
    std::vector<GuardFact> guard;
    enum class Precision { Exact, Conservative };
    Precision precision = Precision::Conservative;
};
} // namespace mlir::pto::insert_sync_frontier
#endif
