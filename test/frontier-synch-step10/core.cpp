// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "PTO/Transforms/FrontierSynch/ExactFrontierCore.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
using namespace mlir::pto::frontiersynch;
namespace core = mlir::pto::frontiersynch::exact_frontier;
namespace arithmetic = mlir::pto::frontiersynch::value_arithmetic;

// A small independently interpreted expression arena. The tested D3 equations,
// slice index and interval arithmetic are production code, not copies here.
// This does not emulate MLIR dominance, effects, or original-value qualification.
struct Arena {
    enum Kind { False, True, Atom, And, Or, Not };
    struct Predicate { Kind kind; std::size_t a = 0, b = 0; };
    struct Frontier { enum Kind { Empty, Leaf, Union, Guard } kind; std::size_t a = 0, b = 0; };
    std::vector<Predicate> predicates{{False}, {True}};
    std::vector<Frontier> frontiers{{Frontier::Empty}};
    std::size_t p(Kind kind, std::size_t a, std::size_t b = 0)
    {
        predicates.push_back({kind, a, b});
        return predicates.size() - 1;
    }
    std::size_t f(Frontier::Kind kind, std::size_t a, std::size_t b = 0)
    {
        frontiers.push_back({kind, a, b});
        return frontiers.size() - 1;
    }
    core::Algebra algebra()
    {
        return {
            [this](auto a, auto b) { return !a || !b ? 0 : a == 1 ? b : b == 1 ? a : p(And, a, b); },
            [this](auto a, auto b) { return a == 1 || b == 1 ? 1 : !a ? b : !b ? a : p(Or, a, b); },
            [this](auto a, auto b) { return !a ? b : !b ? a : f(Frontier::Union, a, b); },
            [this](auto a, auto b) { return !a || !b ? 0 : b == 1 ? a : f(Frontier::Guard, a, b); },
            [this](auto a) { return a < 2 ? 1 - a : p(Not, a); }};
    }
    core::Summary leaf(std::size_t operation)
    {
        auto id = f(Frontier::Leaf, operation);
        return {true, 1, id, id, {}};
    }
    std::vector<bool> valuation(const std::vector<bool>& atoms) const
    {
        std::vector<bool> out;
        for (const auto& v : predicates) {
            switch (v.kind) {
            case False: out.push_back(false); break;
            case True: out.push_back(true); break;
            case Atom: out.push_back(atoms.at(v.a)); break;
            case And: out.push_back(out.at(v.a) && out.at(v.b)); break;
            case Or: out.push_back(out.at(v.a) || out.at(v.b)); break;
            case Not: out.push_back(!out.at(v.a)); break;
            }
        }
        return out;
    }
    std::vector<std::size_t> accesses(std::size_t root, const std::vector<bool>& values) const
    {
        std::vector<std::size_t> out, pending{root};
        while (!pending.empty()) {
            auto id = pending.back(); pending.pop_back();
            const auto& v = frontiers.at(id);
            if (v.kind == Frontier::Leaf) {
                out.push_back(v.a);
            } else if (v.kind == Frontier::Union) {
                pending.push_back(v.b); pending.push_back(v.a);
            } else if (v.kind == Frontier::Guard && values.at(v.b)) {
                pending.push_back(v.a);
            }
        }
        return out;
    }
};
static std::size_t checkedExecutions = 0, checkedDomains = 0;
void check(Arena& arena, const core::Summary& summary, const std::vector<bool>& atoms,
           const std::vector<std::size_t>& trace)
{
    assert(summary.complete);
    const auto values = arena.valuation(atoms);
    const auto first = arena.accesses(summary.first, values), last = arena.accesses(summary.last, values);
    assert(values.at(summary.nonempty) == !trace.empty());
    assert(first.size() == (trace.empty() ? 0 : 1));
    assert(last.size() == (trace.empty() ? 0 : 1));
    if (!trace.empty()) {
        assert(first.front() == trace.front() && last.front() == trace.back());
    }
    ++checkedExecutions;
}
void optionalReaders()
{
    for (unsigned engine = 0; engine < 3; ++engine) {
        Arena arena;
        auto a = arena.algebra();
        std::vector<core::Summary> parts;
        for (unsigned i = 0; i < 8; ++i) {
            auto guard = arena.p(Arena::Atom, i);
            parts.push_back(core::choice(a, guard, i % 3 == engine ? arena.leaf(i) : core::Summary{}, {}));
        }
        const auto all = core::sequence(a, parts);
        // An unrelated suffix cannot become this selector's last access.
        const auto withSuffix = core::sequence(a, {all, core::Summary{}});
        for (unsigned mask = 0; mask < 256; ++mask) {
            std::vector<bool> atoms(8);
            std::vector<std::size_t> concrete;
            for (unsigned i = 0; i < 8; ++i) {
                atoms[i] = mask & (1u << i);
                if (atoms[i] && i % 3 == engine) {
                    concrete.push_back(i);
                }
            }
            check(arena, all, atoms, concrete);
            check(arena, withSuffix, atoms, concrete);
        }
    }
    Arena shared;
    auto a = shared.algebra();
    auto g = shared.p(Arena::Atom, 0);
    const auto first = core::choice(a, g, shared.leaf(1), {});
    const auto second = core::choice(a, g, shared.leaf(2), {});
    const auto repeatedGuard = core::sequence(a, {first, second});
    check(shared, repeatedGuard, {false}, {});
    check(shared, repeatedGuard, {true}, {1, 2});
    const auto alternative = core::choice(a, g, shared.leaf(3), shared.leaf(4));
    check(shared, alternative, {true}, {3});
    check(shared, alternative, {false}, {4});
    const auto unknown = core::unknown("unresolved match");
    assert(!core::sequence(a, {first, unknown, second}).complete);
    assert(!core::choice(a, g, first, unknown).complete);
    assert(core::choice(a, 0, unknown, second).complete);
    assert(core::choice(a, 1, first, unknown).complete);
}
void repeat()
{
    Arena arena;
    auto a = arena.algebra();
    const auto g = arena.p(Arena::Atom, 0), visits = arena.p(Arena::Atom, 1);
    const auto firstVisit = arena.p(Arena::Atom, 2), lastVisit = arena.p(Arena::Atom, 3);
    const auto body = core::choice(a, g, arena.leaf(7), {});
    const auto repeated = core::counted(a, body, visits, firstVisit, lastVisit, true);
    for (unsigned n = 0; n < 9; ++n) {
        for (bool participates : {false, true}) {
            std::vector<std::pair<std::size_t, unsigned>> first, last;
            for (unsigned i = 0; i < n; ++i) {
                const auto values = arena.valuation({participates, n > 0, i == 0, i + 1 == n});
                for (auto op : arena.accesses(repeated.first, values)) first.emplace_back(op, i);
                for (auto op : arena.accesses(repeated.last, values)) last.emplace_back(op, i);
            }
            const bool nonempty = participates && n > 0;
            assert(first.size() == unsigned(nonempty) && last.size() == unsigned(nonempty));
            if (nonempty) {
                assert(first.front().second == 0 && last.front().second == n - 1);
            }
            assert(arena.valuation({participates, n > 0, false, false})[repeated.nonempty] == nonempty);
            ++checkedExecutions;
        }
    }
    assert(!core::counted(a, body, visits, firstVisit, lastVisit, false).complete);
    const auto zero = core::counted(a, core::unknown("unsupported unexecuted body"), 0, firstVisit, lastVisit, false);
    assert(zero.complete && zero.nonempty == 0);
    // The first frontier is independent of the later child's optional guard;
    // the last frontier is not. Actual SSA availability is covered by native.cpp.
    Arena late;
    auto b = late.algebra();
    auto future = late.p(Arena::Atom, 0);
    const auto sequence = core::sequence(b, {late.leaf(10), core::choice(b, future, late.leaf(11), {})});
    check(late, sequence, {false}, {10});
    check(late, sequence, {true}, {10, 11});
}
void intervals()
{
    for (int n = 0; n < 11; ++n) {
        for (int l = -4; l < 15; ++l) {
            for (int u = -4; u < 15; ++u) {
                for (int l2 = -2; l2 < 3; ++l2) {
                    const arithmetic::Integer type{8, false};
                    const std::vector<uint64_t> values{0, uint64_t(n), uint64_t(l) & 255,
                                                       uint64_t(u) & 255, uint64_t(l2) & 255, 7};
                    const core::IntervalRecipe recipe{type, {{0, false}, {2, false}, {4, false}},
                                                           {{1, false}, {3, false}, {5, false}}};
                    unsigned lookups = 0;
                    const auto domain = core::intersection(recipe, [&](auto id) -> std::optional<uint64_t> {
                        ++lookups; return values.at(id);
                    });
                    std::vector<int> concrete;
                    for (int j = 0; j < n; ++j) {
                        if (l <= j && l2 <= j && j < u && j < 7) concrete.push_back(j);
                    }
                    assert(domain.valid && domain.nonempty == !concrete.empty() && lookups == 6);
                    auto first = core::coordinate(type, domain, false), last = core::coordinate(type, domain, true);
                    assert(bool(first) == !concrete.empty() && bool(last) == !concrete.empty());
                    if (first) {
                        assert(*first == unsigned(concrete.front()) && *last == unsigned(concrete.back()));
                    }
                    ++checkedDomains;
                }
            }
        }
    }
    for (bool isUnsigned : {false, true}) {
        for (unsigned width = 2; width <= 8; ++width) {
            const arithmetic::Integer type{width, isUnsigned};
            auto mathematical = [&](uint64_t bits) -> int64_t {
                return !isUnsigned && (bits & type.sign()) ? int64_t(bits) - (int64_t(1) << width) : int64_t(bits);
            };
            for (uint64_t lo = 0; lo <= type.mask(); ++lo) {
                for (uint64_t hi = 0; hi <= type.mask(); ++hi) {
                    const core::IntervalRecipe recipe{type, {{0, false}}, {{1, false}}};
                    const auto domain = core::intersection(recipe, [&](auto id) -> std::optional<uint64_t> {
                        return id == 0 ? lo : hi;
                    });
                    assert(domain.valid && domain.nonempty == (mathematical(lo) < mathematical(hi)));
                    const auto last = core::coordinate(type, domain, true);
                    assert(bool(last) == domain.nonempty);
                    if (last) assert(mathematical(*last) == mathematical(hi) - 1);
                    ++checkedDomains;
                }
            }
        }
    }
    for (bool isUnsigned : {false, true}) {
        const arithmetic::Integer type{64, isUnsigned};
        const core::IntervalRecipe recipe{type, {{0, false}}, {{1, false}}};
        const auto empty = core::intersection(recipe, [&](auto) -> std::optional<uint64_t> { return type.minimum(); });
        assert(empty.valid && !empty.nonempty && !core::coordinate(type, empty, true));
        unsigned work = 0;
        const auto huge = core::intersection(recipe, [&](auto id) -> std::optional<uint64_t> {
            ++work; return id == 0 ? type.maximum() - 2 : type.maximum();
        });
        assert(work == 2 && core::coordinate(type, huge, true) == type.maximum() - 1);
        auto badSuccessor = recipe;
        badSuccessor.upper[0].successor = true;
        assert(!core::intersection(badSuccessor, [&](auto) -> std::optional<uint64_t> {
            return type.maximum();
        }).valid);
        auto singleton = recipe;
        singleton.upper = {{0, true}};
        const auto one = core::intersection(singleton, [&](auto) -> std::optional<uint64_t> { return 5; });
        assert(one.valid && one.nonempty && core::coordinate(type, one, true) == 5);
    }
    core::IntervalRecipe invalid;
    assert(!core::intersection(invalid, [](auto) -> std::optional<uint64_t> { return 0; }).valid);
    const core::IntervalRecipe missing{{8, true}, {{0, false}}, {{1, false}}};
    assert(!core::intersection(missing, [](auto) -> std::optional<uint64_t> { return {}; }).valid);
    assert(!core::intersection(missing, [](auto) -> std::optional<uint64_t> { return 256; }).valid);
}
void comparisons()
{
    using C = core::Comparison;
    auto eval = [](C c, int a, int b) {
        switch (c) {
        case C::Equal: return a == b; case C::NotEqual: return a != b;
        case C::Less: return a < b; case C::LessEqual: return a <= b;
        case C::Greater: return a > b; case C::GreaterEqual: return a >= b;
        }
        throw std::logic_error("invalid comparison");
    };
    for (auto p : {C::Equal, C::NotEqual, C::Less, C::LessEqual, C::Greater, C::GreaterEqual})
        for (bool swap : {false, true}) for (bool positive : {false, true})
            for (int a = -3; a <= 3; ++a) for (int b = -3; b <= 3; ++b) {
                const auto expected = eval(p, swap ? b : a, swap ? a : b) == positive;
                assert(eval(core::normalizeComparison(p, swap, positive), a, b) == expected);
            }
}
Region leaf(std::size_t op) { Region r; r.kind = Region::Operation; r.operation = op; return r; }
Region sequence(std::vector<Region> parts) { Region r; r.children = std::move(parts); return r; }
void slices()
{
    Region loop; loop.kind = Region::For; loop.originalOwner = 10; loop.children = {sequence({leaf(2)})};
    Region choice; choice.kind = Region::Choice; choice.originalOwner = 11;
    choice.children = {sequence({leaf(3)}), sequence({leaf(4)})};
    auto root = sequence({leaf(0), sequence({sequence({leaf(1)}), loop, choice}), leaf(5)});
    core::SliceIndex index(root);
    assert(index.contextCount() == 4);
    OriginalIntervalRequest query;
    query.start = {0, OriginalCut::After}; query.stop = {5, OriginalCut::Before};
    auto s = index.slice(query);
    assert(s.complete && s.end - s.begin == 3);
    assert((*s.parts)[s.begin]->operation == 1);
    assert((*s.parts)[s.begin + 1]->kind == Region::For);
    assert((*s.parts)[s.begin + 2]->kind == Region::Choice);
    for (auto side : {OriginalCut::Before, OriginalCut::After}) {
        query.stop.side = side;
        for (bool include : {false, true}) {
            query.includeStoppingAccess = include;
            s = index.slice(query);
            assert(s.complete && s.end - s.begin == unsigned(3 + include));
        }
    }
    query.start = OriginalCut::childBoundary(10, 0, OriginalCut::Before);
    query.stop = OriginalCut::childBoundary(10, 0, OriginalCut::After);
    query.includeStoppingAccess = false;
    assert(!index.slice(query).complete);
    query.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::FirstReach;
    s = index.slice(query);
    assert(s.complete && s.end - s.begin == 1 && (*s.parts)[s.begin]->operation == 2);
    query.stop = OriginalCut::scope(10, OriginalCut::After);
    assert(!index.slice(query).complete); // an unfinished child is not widened
    query.start = OriginalCut::scope(10, OriginalCut::Before);
    s = index.slice(query);
    assert(s.complete && s.end - s.begin == 1 && (*s.parts)[s.begin]->kind == Region::For);
    query.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::AfterBackedge;
    assert(!index.slice(query).complete);
    query.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::FirstReach;
    query.start = {3, OriginalCut::Before}; query.stop = {4, OriginalCut::After};
    assert(!index.slice(query).complete); // incompatible arms cannot be joined
    query.start = {1, OriginalCut::After}; query.stop = query.start;
    for (bool include : {false, true}) {
        query.includeStoppingAccess = include;
        s = index.slice(query);
        assert(s.complete && s.begin == s.end);
    }
    auto named = sequence({leaf(6), leaf(7)});
    named.originalOwner = 12;
    auto namedRoot = sequence({leaf(0), named, leaf(8)});
    core::SliceIndex namedIndex(namedRoot);
    query = {};
    query.start = OriginalCut::scope(12, OriginalCut::Before);
    query.stop = OriginalCut::scope(12, OriginalCut::After);
    s = namedIndex.slice(query);
    assert(s.complete && s.end - s.begin == 1 && (*s.parts)[s.begin]->originalOwner == 12);
    query.start = {6, OriginalCut::Before};
    query.stop = {7, OriginalCut::Before};
    s = namedIndex.slice(query);
    assert(s.complete && s.end - s.begin == 1 && (*s.parts)[s.begin]->operation == 6);
    query.start = OriginalCut::scope(12, OriginalCut::Before);
    s = namedIndex.slice(query);
    assert(s.complete && s.end - s.begin == 1);
    // Semantic query keys distinguish all interval/occurrence/selector fields.
    query.version = OriginalProgramVersion::fresh();
    OriginalInterval a{query, NoControlId}, b = a;
    std::map<OriginalInterval, unsigned> cache;
    auto add = [&](auto mutate) { b = a; mutate(b); cache.emplace(b, 1); };
    cache.emplace(a, 1);
    add([](auto& v) { v.query.version = OriginalProgramVersion::fresh(); });
    add([](auto& v) { v.query.continuationOwner = 10; });
    add([](auto& v) { v.query.selector.engine = 2; });
    add([](auto& v) { v.query.selector.physicalRelation = 3; });
    add([](auto& v) { v.query.occurrence.incomingInterface = 5; });
    add([](auto& v) { v.query.includeStoppingAccess = !v.query.includeStoppingAccess; });
    assert(cache.size() == 7);
}
void sharing()
{
    Arena arena;
    auto a = arena.algebra();
    std::vector<core::Summary> parts;
    const unsigned count = 10000;
    for (unsigned i = 0; i < count; ++i) {
        parts.push_back(core::choice(a, arena.p(Arena::Atom, i), arena.leaf(i), {}));
    }
    const auto result = core::sequence(a, parts);
    assert(result.complete && arena.predicates.size() < 9 * count && arena.frontiers.size() < 9 * count);
    std::cout << "shared stress: " << count << " optional readers; " << arena.predicates.size()
              << " predicate nodes, " << arena.frontiers.size() << " frontier nodes; no valuations enumerated\n";
}
int main()
{
    optionalReaders(); repeat(); intervals(); comparisons(); slices(); sharing();
    std::cout << "PASS: " << checkedExecutions << " finite frontier executions, " << checkedDomains
              << " interval domains; comparison normalization, empty/overflow cases and structural cuts\n";
}
