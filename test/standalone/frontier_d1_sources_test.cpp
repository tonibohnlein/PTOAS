// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/FixedVisitSources.h"
#include <cstdlib>
#include <iostream>
#include <set>

using namespace mlir::pto::frontiersynch;
using Node = FactoredUseNode;
using Tree = FactoredUseRegion;
using Set = std::set<std::size_t>;
static std::size_t checks = 0, executions = 0;
static void check(bool pass, const char* why)
{
    ++checks;
    if (!pass) {
        std::cerr << "D1 failure: " << why << '\n';
        std::exit(EXIT_FAILURE);
    }
}
static bool evaluate(const FactoredUseArena& arena, std::size_t root, const std::map<std::uintptr_t, bool>& values)
{
    // Independent lazy interpreter: only the selected original branch is read.
    while (root > 1) {
        const auto& n = arena[root];
        if (n.kind == Node::Kind::Test) {
            return values.at(n.guard.value);
        }
        check(n.kind == Node::Kind::Choose, "non-condition in endpoint predicate");
        root = evaluate(arena, n.condition, values) ? n.left : n.right;
    }
    return root == 1;
}
struct Fixture {
    FactoredUseProjection projection;
    FactoredUseInterface boundary;
    Fixture()
    {
        projection.frame.cell = 0;
        projection.frame.snapshot = OriginalProgramVersion::fresh();
        boundary.arena = std::make_shared<FactoredUseArena>(projection.frame);
        boundary.incoming = {
            boundary.arena->incoming(NoControlId, Node::Boundary::Entry, Node::Role::Writer),
            boundary.arena->incoming(NoControlId, Node::Boundary::Entry, Node::Role::Reader)};
    }
    Tree access(bool read, bool write, bool full = true)
    {
        Tree t;
        t.kind = Tree::Kind::Access;
        t.access = projection.accesses.size();
        auto a = std::make_shared<FactoredUseAccess>();
        a->operation = t.access;
        a->read = read;
        a->write = write;
        a->definiteWrite = write && full;
        if (read) {
            a->readIncidences = {10 + t.access};
        }
        if (write) {
            a->writeIncidences = {20 + t.access};
        }
        projection.accesses.push_back(a);
        return t;
    }
    Tree choice(std::uintptr_t guard, Tree yes, Tree no = {})
    {
        Tree t;
        t.kind = Tree::Kind::Choice;
        t.condition = boundary.arena->test({guard, NoControlId});
        t.children = {std::move(yes), std::move(no)};
        return t;
    }
    FactoredUseResult build(Tree body)
    {
        projection.body = std::move(body);
        return FactoredUseBuilder(projection, boundary).take();
    }
};
static Tree sequence(std::initializer_list<Tree> children)
{
    Tree t;
    t.children = children;
    return t;
}
static Set selected(const FixedVisitSources& sources, const std::map<std::uintptr_t, bool>& values)
{
    Set result;
    for (const auto& source : sources.alternatives) {
        if (evaluate(*sources.predicates, source.condition, values)) {
            check(result.insert(source.operation).second, "duplicate semantic source alternative");
        }
    }
    return result;
}
static void trace(const Fixture& fixture, const Tree& t, const std::map<std::uintptr_t, bool>& values,
                  std::vector<std::size_t>& out)
{
    if (t.kind == Tree::Kind::Access) {
        out.push_back(t.access);
    } else if (t.kind == Tree::Kind::Choice) {
        trace(fixture, t.children[evaluate(*fixture.boundary.arena, t.condition, values) ? 0 : 1], values, out);
    } else {
        for (const auto& child : t.children) {
            trace(fixture, child, values, out);
        }
    }
}
static void compare(Fixture& f, const FactoredUseResult& uses, unsigned guards)
{
    // Only SMALL fixtures enumerate executions. The implementation never does.
    std::map<std::pair<std::size_t, Node::Hazard>, FixedVisitSources> queries;
    for (const auto& site : uses.sites) {
        for (auto hazard : {Node::Hazard::RAW, Node::Hazard::WAR, Node::Hazard::WAW}) {
            if (hazard == Node::Hazard::RAW ? site.access->read : site.access->write) {
                queries.emplace(std::make_pair(site.access->operation, hazard),
                                queryFixedVisitSources(uses, site.access->operation, hazard));
            }
        }
    }
    for (unsigned bits = 0; bits < (1u << guards); ++bits) {
        ++executions;
        std::map<std::uintptr_t, bool> values;
        for (unsigned i = 0; i < guards; ++i) {
            values[i + 1] = (bits >> i) & 1;
        }
        std::vector<std::size_t> path;
        trace(f, f.projection.body, values, path);
        Set writers{NoFactoredId}, readers{NoFactoredId};
        for (auto op : path) {
            const auto& effect = *f.projection.accesses[op];
            for (auto hazard : {Node::Hazard::RAW, Node::Hazard::WAR, Node::Hazard::WAW}) {
                if (!(hazard == Node::Hazard::RAW ? effect.read : effect.write)) {
                    continue;
                }
                const auto& result = queries.at({op, hazard});
                check(result.complete, "D1 failed on admitted fixed-use expression");
                const auto expected = hazard == Node::Hazard::WAR ? readers : writers;
                check(selected(result, values) == expected, "D1 disagrees with concrete original provenance scan");
                for (const auto& source : result.alternatives) {
                    if (evaluate(*result.predicates, source.condition, values)) {
                        check(evaluate(*result.predicates, source.targetCondition, values), "target loses applicable source");
                        if (!source.incoming) {
                            check(evaluate(*result.predicates, source.sourceCondition, values), "source loses applicable target");
                            check(bool(source.witness), "lost translated-effect witness");
                        }
                    }
                }
            }
            // Independent concrete OLD-state scanner, including RMW-before-update.
            if (effect.write && effect.definiteWrite) {
                writers = {op};
                readers.clear();
            } else {
                if (effect.write) {
                    writers.insert(op);
                }
                if (effect.read) {
                    readers.insert(op);
                }
            }
        }
        // Check BOTH endpoint execution domains, including paths where a source
        // executes but its optional target does not. Implication at executed
        // consumers alone would miss a spurious publication on an empty path.
        const Set executed(path.begin(), path.end());
        for (const auto& [key, query] : queries) {
            check(evaluate(*query.predicates, query.applicability, values) == bool(executed.count(key.first)),
                  "target participation differs from original control");
            for (const auto& source : query.alternatives) {
                const auto applicable = evaluate(*query.predicates, source.condition, values);
                if (source.incoming || executed.count(source.operation)) {
                    check(evaluate(*query.predicates, source.sourceCondition, values) == applicable,
                          "source endpoint executes without its corresponding target");
                }
                if (executed.count(key.first)) {
                    check(evaluate(*query.predicates, source.targetCondition, values) == applicable,
                          "target endpoint executes without its corresponding source");
                }
                check(!applicable || source.incoming || executed.count(source.operation),
                      "applicable origin is in an incompatible original arm");
            }
        }
    }
}
static void regressions()
{
    {
        Fixture f;
        auto w = f.access(false, true), a = f.access(true, false), b = f.access(true, false);
        auto overwrite = f.access(false, true);
        auto uses = f.build(sequence({w, a, b, overwrite}));
        compare(f, uses, 0);
        auto read = queryFixedVisitSources(uses, b.access, Node::Hazard::RAW);
        check(read.exclusiveWriters && read.alternatives.front().operation == w.access, "intervening read severs producer");
        auto release = queryFixedVisitSources(uses, overwrite.access, Node::Hazard::WAR);
        check(release.independentReaders && release.alternatives.size() == 2, "independent readers became alternatives");
    }
    {
        Fixture f;
        auto w0 = f.access(false, true), a = f.access(true, false), w1 = f.access(false, true);
        auto b = f.access(true, false), w2 = f.access(false, true);
        auto uses = f.build(sequence({w0, a, f.choice(1, w1), b, w2}));
        compare(f, uses, 1);
        const auto count = uses.nodes().size();
        auto q = queryFixedVisitSources(uses, b.access, Node::Hazard::RAW);
        check(q.exclusiveWriters && q.alternatives.size() == 2, "conditional replacement not a D1 alternative");
        check(q.alternatives[1].sourceCondition == 1, "source lexical guard not discharged at its own endpoint");
        check(q.alternatives[1].targetCondition != 1, "source qualification leaked into target condition");
        check(uses.nodes().size() == count, "query mutates original provenance");
    }
    {
        Fixture f;
        auto w = f.access(false, true), a = f.access(true, false), b = f.access(true, false);
        auto uses = f.build(sequence({f.choice(1, w, a), b}));
        compare(f, uses, 1);
        auto q = queryFixedVisitSources(uses, a.access, Node::Hazard::RAW);
        check(q.alternatives.size() == 1 && q.alternatives.front().incoming, "incompatible producer joined to opposite arm");
        q = queryFixedVisitSources(uses, b.access, Node::Hazard::RAW);
        check(q.alternatives.size() == 2 && q.alternatives.back().incoming, "no-producer branch was discarded");
    }
    {
        Fixture f;
        auto w = f.access(false, true), r = f.access(true, false), partial = f.access(true, true, false);
        auto later = f.access(true, false), end = f.access(false, true);
        auto uses = f.build(sequence({w, r, f.choice(1, partial), later, end}));
        compare(f, uses, 1);
        auto q = queryFixedVisitSources(uses, later.access, Node::Hazard::RAW);
        check(!q.exclusiveWriters && q.alternatives.size() == 2, "partial overwrite became an exclusive full origin");
    }
    {
        Fixture f;
        auto w = f.access(false, true), x = f.access(false, true), y = f.access(true, true), r = f.access(true, false);
        auto uses = f.build(sequence({w, f.choice(1, sequence({f.choice(2, x), f.choice(1, y)})), r}));
        compare(f, uses, 2);
    }
    {
        Fixture f;
        auto w = f.access(false, true), r = f.access(true, false);
        auto uses = f.build(sequence({f.choice(1, w), f.choice(1, {}, r)}));
        compare(f, uses, 1);
    }
    {
        Fixture f;
        auto w = f.access(false, true), r = f.access(true, false);
        auto uses = f.build(sequence({w, f.choice(1, r)}));
        compare(f, uses, 1);
    }
    // Systematic branch replacement, multiple readers, and RMW/weak-write mixtures.
    for (unsigned modes = 0; modes < 81; ++modes) {
        Fixture f;
        Tree body;
        body.children.push_back(f.access(false, true));
        auto code = modes;
        for (unsigned i = 0; i < 4; ++i) {
            const auto mode = code % 3;
            code /= 3;
            auto effect = mode == 0 ? f.access(true, false) : f.access(mode == 2, true, mode == 1);
            body.children.push_back(f.choice(i + 1, effect));
            body.children.push_back(f.access(true, false));
        }
        body.children.push_back(f.access(false, true));
        auto uses = f.build(body);
        compare(f, uses, 4);
    }
    {
        Fixture f;
        Tree body;
        body.children.push_back(f.access(false, true));
        constexpr unsigned count = 256;
        for (unsigned i = 0; i < count; ++i) {
            body.children.push_back(f.choice(i + 1, f.access(true, false)));
        }
        auto overwrite = f.access(false, true);
        body.children.push_back(overwrite);
        auto uses = f.build(body);
        const auto originalNodes = uses.nodes().size();
        auto q = queryFixedVisitSources(uses, overwrite.access, Node::Hazard::WAR);
        check(q.complete && q.alternatives.size() == count, "optional-reader obligations lost");
        check(q.predicates->nodes().size() <= 12 * count + 10, "D1 enumerated optional reader combinations");
        check(uses.nodes().size() == originalNodes, "D1 changed formation population");
    }
}
int main()
{
    regressions();
    std::cout << "D1: " << checks << " checks; " << executions << " concrete executions passed\n";
    return EXIT_SUCCESS;
}
