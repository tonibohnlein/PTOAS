// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_PHYSICALPERMUTATION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_PHYSICALPERMUTATION_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::pto::frontiersynch::periodic_uses {

// All banks in one record are in the SAME qualified physical memory domain.
// The adapter establishes that domain and the original scalar transition. This
// core checks an EXPLICIT finite population; it never expands a modulus.
struct Bank {
    uint64_t begin = 0, bytes = 0;
    bool operator==(const Bank& other) const { return begin == other.begin && bytes == other.bytes; }
};
struct Permutation {
    bool exact = false;
    std::vector<Bank> banks;
    std::vector<std::size_t> next, previous, cycleLength, cycleId;
    std::string reason;
};
inline Permutation certify(std::vector<Bank> banks, std::vector<std::size_t> next)
{
    Permutation result;
    result.banks = std::move(banks);
    result.next = std::move(next);
    const auto n = result.banks.size();
    auto reject = [&](const char* reason) {
        result.reason = reason;
        return result;
    };
    if (!n || result.next.size() != n) {
        return reject("no explicit physical permutation");
    }
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    for (const auto& bank : result.banks) {
        if (!bank.bytes || bank.bytes > std::numeric_limits<uint64_t>::max() - bank.begin) {
            return reject("empty or overflowing physical bank");
        }
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return result.banks[a].begin < result.banks[b].begin;
    });
    for (std::size_t i = 1; i < n; ++i) {
        const auto& a = result.banks[order[i - 1]];
        const auto& b = result.banks[order[i]];
        if (b.begin - a.begin < a.bytes) {
            return reject("noninjective or overlapping physical bank selection");
        }
    }
    result.previous.assign(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto successor = result.next[i];
        if (successor >= n || result.previous[successor] != n) {
            return reject("physical transition is not a permutation");
        }
        result.previous[successor] = i;
    }
    result.cycleLength.assign(n, 0);
    result.cycleId.assign(n, n);
    // Each edge is followed once, independent of any invocation's trip count.
    for (std::size_t first = 0; first < n; ++first) {
        if (result.cycleLength[first]) {
            continue;
        }
        std::vector<std::size_t> cycle;
        auto at = first;
        do {
            cycle.push_back(at);
            at = result.next[at];
        } while (at != first);
        for (auto bank : cycle) {
            result.cycleLength[bank] = cycle.size();
            result.cycleId[bank] = first;
        }
    }
    result.exact = true;
    return result;
}

// A domain is a predicate over ORIGINAL body-visit ordinals. It is not a new
// runtime visit counter. Native clients retain the original IV/LB/UB/step and
// qualify a width-correct recipe for these tests at each endpoint separately.
struct Domain {
    enum class Boundary { Any, HasPrevious, Initial, HasNext, Final } boundary = Boundary::Any;
    uint64_t period = 0, phase = 0, distance = 0;
    bool contains(uint64_t ordinal, uint64_t tripCount) const
    {
        if (!period || phase >= period || ordinal >= tripCount || ordinal % period != phase) {
            return false;
        }
        switch (boundary) {
            case Boundary::Any:
                return true;
            case Boundary::HasPrevious:
                return ordinal >= distance;
            case Boundary::Initial:
                return ordinal < distance;
            case Boundary::HasNext:
                return distance < tripCount - ordinal;
            case Boundary::Final:
                return distance >= tripCount - ordinal;
        }
        return false;
    }
};
struct Link {
    std::size_t bank = 0;
    uint64_t period = 0, sourcePhase = 0, targetPhase = 0, distance = 0;
    Domain predecessor() const { return {Domain::Boundary::HasPrevious, period, targetPhase, distance}; }
    Domain initial() const { return {Domain::Boundary::Initial, period, targetPhase, distance}; }
    Domain successor() const { return {Domain::Boundary::HasNext, period, sourcePhase, distance}; }
    Domain final() const { return {Domain::Boundary::Final, period, sourcePhase, distance}; }
    std::optional<uint64_t> previous(uint64_t target, uint64_t trips) const
    {
        return predecessor().contains(target, trips) ? std::optional<uint64_t>{target - distance} : std::nullopt;
    }
    std::optional<uint64_t> next(uint64_t source, uint64_t trips) const
    {
        // The domain proves the addition is representable, even near UINT64_MAX.
        return successor().contains(source, trips) ? std::optional<uint64_t>{source + distance} : std::nullopt;
    }
};

// One mandatory role per ORIGINAL owner visit, in reference order. A role may
// have both effects. Independent reader roles are not contracted into one read.
// Its selection is an explicitly qualified cycle, aligned at owner ordinal 0.
struct Role {
    std::size_t operation = 0;
    bool read = false, write = false;
    std::vector<std::size_t> selection;
};
struct Correspondence {
    bool exact = false;
    std::vector<Link> links;
    std::string reason;
};
inline Correspondence relate(
    const Permutation& permutation, const std::vector<Role>& roles,
    std::size_t source, std::size_t target, bool sourceWrites, bool targetWrites,
    const std::vector<bool>& selectedBanks = {})
{
    Correspondence result;
    auto reject = [&](const char* reason) {
        result.links.clear();
        result.reason = reason;
        return result;
    };
    if (!permutation.exact || permutation.next.size() != permutation.banks.size() ||
        permutation.cycleLength.size() != permutation.banks.size() || source >= roles.size() || target >= roles.size() ||
        (!sourceWrites && !targetWrites)) {
        return reject("invalid periodic conflict roles");
    }
    if (!selectedBanks.empty() && selectedBanks.size() != permutation.banks.size()) {
        return reject("invalid physical-cell projection");
    }
    const auto& from = roles[source];
    const auto& to = roles[target];
    if (!(sourceWrites ? from.write : from.read) || !(targetWrites ? to.write : to.read)) {
        return reject("periodic endpoint has no requested access role");
    }
    const auto n = from.selection.size();
    if (!n || n > permutation.banks.size()) {
        return reject("missing explicit role cycle");
    }
    // An inverse phase map for each RELATED role, not a product of independent
    // selector states. A partial bank population or noninjective selection is
    // not inferred to be the same family from equal periods alone.
    std::vector<std::vector<std::size_t>> phase(roles.size(), std::vector<std::size_t>(permutation.banks.size(), n));
    for (std::size_t r = 0; r < roles.size(); ++r) {
        if (roles[r].selection.size() != n) {
            return reject("interfering selector has a different physical-use cycle");
        }
        for (std::size_t i = 0; i < n; ++i) {
            const auto bank = roles[r].selection[i];
            if (bank >= permutation.banks.size() || phase[r][bank] != n) {
                return reject("role selection is not injective on physical banks");
            }
            phase[r][bank] = i;
        }
        for (auto bank : from.selection) {
            if (bank >= permutation.banks.size() || phase[r][bank] == n) {
                return reject("interfering selector uses a different physical bank population");
            }
        }
    }
    // The source role traverses one of the supplied physical permutation's
    // cycles. Other roles can have a different starting phase or spelling.
    for (std::size_t i = 0; i < n; ++i) {
        const auto bank = from.selection[i];
        if (permutation.cycleLength[bank] != n ||
            permutation.next[bank] != from.selection[(i + 1) % n]) {
            return reject("role sequence does not follow its physical permutation");
        }
    }
    auto precedingDistance = [&](std::size_t r, std::size_t bank, std::size_t targetPhase) -> uint64_t {
        const auto sourcePhase = phase[r][bank];
        // Avoid targetPhase + n and its host-size overflow.
        auto distance = targetPhase >= sourcePhase ? targetPhase - sourcePhase : n - (sourcePhase - targetPhase);
        if (!distance && r >= target) {
            distance = n; // positive-length return, never a same-occurrence self edge
        }
        return distance;
    };
    for (std::size_t targetPhase = 0; targetPhase < n; ++targetPhase) {
        const auto bank = to.selection[targetPhase];
        if (!selectedBanks.empty() && !selectedBanks[bank]) {
            continue;
        }
        const auto distance = precedingDistance(source, bank, targetPhase);
        // Reads between a producer and a later reader do NOT sever it. A writer
        // inside this same-bank interval changes the applicable origin/episode;
        // without a separately qualified relation that case stays unresolved.
        for (std::size_t r = 0; r < roles.size(); ++r) {
            if (!roles[r].write) {
                continue;
            }
            const auto writerDistance = precedingDistance(r, bank, targetPhase);
            if (writerDistance < distance || (writerDistance == distance && r > source)) {
                return reject("another physical write intervenes between periodic roles");
            }
        }
        result.links.push_back({bank, n, phase[source][bank], targetPhase, distance});
    }
    if (result.links.empty()) {
        return reject("no selected physical bank");
    }
    std::sort(result.links.begin(), result.links.end(), [](const Link& a, const Link& b) { return a.bank < b.bank; });
    result.exact = true;
    return result;
}

} // namespace mlir::pto::frontiersynch::periodic_uses
#endif
