// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
// Independent finite execution checks of the PRODUCTION D2 core. These checks
// stay active in optimized/NDEBUG builds and are not a native-import test.
#include "PTO/Transforms/FrontierSynch/PhysicalPermutation.h"
#include <cstdlib>
#include <iostream>

namespace p = mlir::pto::frontiersynch::periodic_uses;
static uint64_t assertions = 0, traces = 0, permutations = 0;
static void check(bool ok, int line = __builtin_LINE())
{
    ++assertions;
    if (!ok) {
        std::cerr << "D2 core assertion failed at " << line << '\n';
        std::exit(1);
    }
}
static p::Permutation ring(std::size_t n)
{
    std::vector<p::Bank> banks;
    std::vector<std::size_t> next;
    for (std::size_t i = 0; i < n; ++i) {
        banks.push_back({1024 + 64 * i, 32});
        next.push_back((i + 1) % n);
    }
    return p::certify(banks, next);
}
static std::vector<std::size_t> rotated(std::size_t n, std::size_t shift)
{
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back((i + shift) % n);
    }
    return out;
}
struct Event { uint64_t ordinal; std::size_t role, bank; };
static void execute(const p::Permutation& permutation, const std::vector<p::Role>& roles,
                    std::size_t source, std::size_t target, bool sw, bool tw, uint64_t trips)
{
    ++traces;
    const auto result = p::relate(permutation, roles, source, target, sw, tw);
    check(result.exact);
    std::vector<Event> events;
    for (uint64_t i = 0; i < trips; ++i) {
        for (std::size_t r = 0; r < roles.size(); ++r) {
            events.push_back({i, r, roles[r].selection[i % roles[r].selection.size()]});
        }
    }
    for (std::size_t at = 0; at < events.size(); ++at) {
        const auto& e = events[at];
        const auto found = std::find_if(result.links.begin(), result.links.end(), [&](const p::Link& link) {
            return link.bank == e.bank;
        });
        check(found != result.links.end());
        const auto& link = *found;
        if (e.role == target) {
            std::optional<uint64_t> previous;
            for (auto i = at; i > 0; --i) {
                const auto& candidate = events[i - 1];
                if (candidate.role == source && candidate.bank == e.bank) {
                    previous = candidate.ordinal;
                    break;
                }
                // Every accepted link has no interior write on the same bank.
                check(candidate.bank != e.bank || !roles[candidate.role].write);
            }
            check(link.previous(e.ordinal, trips) == previous);
            check(link.predecessor().contains(e.ordinal, trips) == bool(previous));
            check(link.initial().contains(e.ordinal, trips) == !previous);
            for (const auto& other : result.links) {
                if (other.bank != e.bank) {
                    check(!other.predecessor().contains(e.ordinal, trips));
                    check(!other.initial().contains(e.ordinal, trips));
                }
            }
        }
        if (e.role == source) {
            std::optional<uint64_t> next;
            for (auto i = at + 1; i < events.size(); ++i) {
                const auto& candidate = events[i];
                if (candidate.role == target && candidate.bank == e.bank) {
                    next = candidate.ordinal;
                    break;
                }
            }
            check(link.next(e.ordinal, trips) == next);
            check(link.successor().contains(e.ordinal, trips) == bool(next));
            check(link.final().contains(e.ordinal, trips) == !next);
            if (next) {
                check(link.previous(*next, trips) == std::optional<uint64_t>(e.ordinal));
            }
        }
    }
    for (const auto& link : result.links) {
        check(!link.predecessor().contains(trips, trips));
        check(!link.successor().contains(trips, trips));
        check(!link.initial().contains(trips, trips));
        check(!link.final().contains(trips, trips));
    }
}
int main()
{
    // All explicit permutations up to seven banks, including several cycles
    // of DIFFERENT lengths in one permutation. Oracle: walk from each bank.
    for (std::size_t n = 1; n <= 7; ++n) {
        auto permutation = ring(n);
        std::vector<std::size_t> next(n);
        std::iota(next.begin(), next.end(), 0);
        do {
            ++permutations;
            const auto proof = p::certify(permutation.banks, next);
            check(proof.exact);
            for (std::size_t bank = 0; bank < n; ++bank) {
                std::size_t length = 1, at = next[bank];
                while (at != bank) {
                    ++length;
                    at = next[at];
                }
                check(length == proof.cycleLength[bank]);
                check(proof.previous[next[bank]] == bank);
                check(proof.next[proof.previous[bank]] == bank);
            }
        } while (std::next_permutation(next.begin(), next.end()));
    }
    for (std::size_t n = 1; n <= 5; ++n) {
        const auto permutation = ring(n);
        for (std::size_t shift = 0; shift < n; ++shift) {
            // One writer and two INDEPENDENT reader sites. The second reader
            // does not terminate the first reader's link or duplicate its ID.
            const std::vector<p::Role> roles{
                {41, false, true, rotated(n, 0)},
                {7, true, false, rotated(n, shift)},
                {99, true, false, rotated(n, shift)}};
            for (uint64_t trips = 0; trips <= 4 * n + 2; ++trips) {
                execute(permutation, roles, 0, 1, true, false, trips);
                execute(permutation, roles, 0, 2, true, false, trips);
                // Use the reader's permutation as the source's physical orbit.
                execute(permutation, roles, 1, 0, false, true, trips);
                execute(permutation, roles, 2, 0, false, true, trips);
                execute(permutation, roles, 0, 0, true, true, trips);
            }
        }
    }
    // Non-rotation physical traversal: still exact physical equality, not a
    // guessed scalar residue. Each role uses its own explicit sequence.
    {
        const auto permutation = ring(3);
        const std::vector<p::Role> roles{{0, false, true, {0, 1, 2}}, {1, true, false, {2, 1, 0}}};
        for (uint64_t trips = 0; trips < 15; ++trips) {
            execute(permutation, roles, 0, 1, true, false, trips);
        }
    }
    check(!p::certify({}, {}).exact);
    check(!p::certify({{0, 64}, {0, 64}}, {1, 0}).exact);
    check(!p::certify({{0, 64}, {32, 64}}, {1, 0}).exact);
    check(!p::certify({{0, 64}, {64, 64}}, {1, 1}).exact);
    check(!p::certify({{0, 0}}, {0}).exact);
    check(!p::certify({{UINT64_MAX - 10, 11}}, {0}).exact);
    check(!p::certify({{0, 8}}, {1}).exact);
    // A,A,B,B is scalar-periodic, but is NOT a physical permutation.
    check(!p::certify({{0, 32}, {0, 32}, {64, 32}, {64, 32}}, {1, 2, 3, 0}).exact);
    {
        const auto permutation = ring(2);
        std::vector<p::Role> roles{{0, false, true, {0, 1}}, {1, false, true, {0, 1}},
                                   {2, true, false, {0, 1}}};
        check(!p::relate(permutation, roles, 0, 2, true, false).exact); // intervening write
        roles[1].write = false;
        roles[1].read = true;
        check(p::relate(permutation, roles, 0, 2, true, false).exact); // read is harmless
        roles[1].selection = {0, 0};
        check(!p::relate(permutation, roles, 0, 2, true, false).exact);
        roles[1].selection = {0};
        check(!p::relate(permutation, roles, 0, 2, true, false).exact); // fixed interfering bank
        check(!p::relate(permutation, roles, 90, 2, true, false).exact);
    }
    // Cell-specific qualification must not be defeated by an intervening write
    // on an excluded bank, nor claim that this exclusion covers the whole family.
    {
        const auto permutation = ring(3);
        const std::vector<p::Role> roles{{0, false, true, {0, 1, 2}},
                                         {1, false, true, {0, 2, 1}},
                                         {2, true, false, {0, 1, 2}}};
        check(!p::relate(permutation, roles, 0, 2, true, false).exact);
        const auto selected = p::relate(permutation, roles, 0, 2, true, false, {false, true, true});
        check(selected.exact && selected.links.size() == 2);
        check(selected.links[0].bank == 1 && selected.links[1].bank == 2);
        check(!p::relate(permutation, roles, 0, 2, true, false, {true}).exact);
        check(!p::relate(permutation, roles, 0, 2, true, false, {false, false, false}).exact);
    }
    // Independent families retain their OWN populations (2+3, not lcm/product).
    std::size_t records = 0;
    for (std::size_t n : {2, 3}) {
        const auto permutation = ring(n);
        const std::vector<p::Role> roles{{0, false, true, rotated(n, 0)}, {1, true, false, rotated(n, 0)}};
        const auto result = p::relate(permutation, roles, 1, 0, false, true);
        check(result.exact);
        records += result.links.size();
    }
    check(records == 5);
    // The proof has no 256-state trial budget when the bank population is
    // explicitly supplied. This is not enumeration of a huge encoded modulus.
    {
        const auto permutation = ring(1025);
        const std::vector<p::Role> roles{{0, false, true, rotated(1025, 0)},
                                         {1, true, false, rotated(1025, 0)}};
        const auto result = p::relate(permutation, roles, 1, 0, false, true);
        check(result.exact && result.links.size() == 1025);
        check(result.links.front().previous(1025, 1026) == std::optional<uint64_t>(0));
    }
    const p::Link edge{0, 1, 0, 0, 3};
    check(edge.next(UINT64_MAX - 4, UINT64_MAX) == std::optional<uint64_t>(UINT64_MAX - 1));
    check(!edge.next(UINT64_MAX - 3, UINT64_MAX));
    check(edge.final().contains(UINT64_MAX - 1, UINT64_MAX));
    check(!edge.previous(2, UINT64_MAX));
    check(edge.initial().contains(2, UINT64_MAX));
    check(!edge.initial().contains(0, 0));
    std::cout << "PASS D2 production core: " << permutations << " explicit permutations, "
              << traces << " finite traces, " << assertions << " assertions\n";
}
