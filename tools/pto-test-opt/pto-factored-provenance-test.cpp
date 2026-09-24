// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.

// This target intentionally needs only the standard library. It executes the
// same transfer used by the native adapter, not a reimplementation or MLIR stub.
#include "PTO/Transforms/FrontierSynch/FactoredUse.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <cstdlib>

namespace fs = mlir::pto::frontiersynch;
using Node = fs::FactoredUseNode;
using Region = fs::FactoredUseRegion;
using Set = std::set<std::size_t>;
using Demand = std::tuple<std::size_t, std::size_t, Node::Hazard>;
using Demands = std::set<Demand>;
using Input = std::tuple<std::size_t, Node::Boundary, Node::Role>;
static std::size_t executions = 0, materializedOrigins = 0, materializedDemands = 0;

[[noreturn]] static void fail(const std::string& message)
{
    std::cerr << "factored provenance check failed: " << message << '\n';
    std::exit(1);
}
static void require(bool condition, const std::string& message)
{
    if (!condition) {
        fail(message);
    }
}

// Test-only interpretation of a fixed original execution. In particular, it
// never asks for a test in an unselected child. Unknown repeat interfaces are
// not silently interpreted as empty or as one iteration.
struct Evaluation {
    const fs::FactoredUseArena& arena;
    std::map<fs::FactoredGuardIdentity, bool> guards;
    std::map<Input, Set> inputs;
    std::map<std::size_t, bool> conditionCache;
    std::map<std::size_t, Set> originCache;
    std::map<std::size_t, Demands> demandCache;
    bool condition(std::size_t id)
    {
        const auto prior = conditionCache.find(id);
        if (prior != conditionCache.end()) {
            return prior->second;
        }
        const auto& n = arena[id];
        bool value;
        switch (n.kind) {
            case Node::Kind::Empty:
                value = false;
                break;
            case Node::Kind::True:
                value = true;
                break;
            case Node::Kind::Test:
                value = guards.at(n.guard);
                break;
            case Node::Kind::Choose:
                value = condition(condition(n.condition) ? n.left : n.right);
                break;
            default:
                fail("non-exact condition queried");
        }
        conditionCache.emplace(id, value);
        return value;
    }
    Set origins(std::size_t id)
    {
        const auto prior = originCache.find(id);
        if (prior != originCache.end()) {
            return prior->second;
        }
        const auto& n = arena[id];
        Set value;
        switch (n.kind) {
            case Node::Kind::Empty:
                break;
            case Node::Kind::Incoming:
                value = inputs.at({n.owner, n.boundary, n.role});
                break;
            case Node::Kind::Access:
                value.insert(n.operation);
                break;
            case Node::Kind::Both: {
                value = origins(n.left);
                const auto right = origins(n.right);
                value.insert(right.begin(), right.end());
                break;
            }
            case Node::Kind::Choose:
                value = origins(condition(n.condition) ? n.left : n.right);
                break;
            default:
                fail("non-exact origin queried");
        }
        materializedOrigins += value.size();
        originCache.emplace(id, value);
        return value;
    }
    Demands demands(std::size_t id)
    {
        const auto prior = demandCache.find(id);
        if (prior != demandCache.end()) {
            return prior->second;
        }
        const auto& n = arena[id];
        Demands value;
        switch (n.kind) {
            case Node::Kind::Empty:
                break;
            case Node::Kind::Demand:
                for (auto source : origins(n.left)) {
                    value.emplace(source, n.operation, n.hazard);
                }
                break;
            case Node::Kind::Both: {
                value = demands(n.left);
                const auto right = demands(n.right);
                value.insert(right.begin(), right.end());
                break;
            }
            case Node::Kind::Choose:
                value = demands(condition(n.condition) ? n.left : n.right);
                break;
            default:
                fail("unresolved demand queried as a concrete execution");
        }
        materializedDemands += value.size();
        demandCache.emplace(id, value);
        return value;
    }
};

struct Fixture {
    fs::FactoredUseProjection projection;
    fs::FactoredUseInterface boundary;
    Fixture(fs::FactoredUseFrame frame = {})
    {
        frame.cell = 0;
        projection.frame = frame;
        boundary.arena = std::make_shared<fs::FactoredUseArena>(frame);
    }
    Region op(bool read, bool write, bool full = true)
    {
        auto access = std::make_shared<fs::FactoredUseAccess>();
        access->operation = projection.accesses.size();
        access->read = read;
        access->write = write;
        access->definiteWrite = write && full;
        if (read) {
            access->readIncidences = {2, 5};
        }
        if (write) {
            access->writeIncidences = {3, 7};
        }
        Region out;
        out.kind = Region::Kind::Access;
        out.access = projection.accesses.size();
        projection.accesses.push_back(std::move(access));
        return out;
    }
    Region choice(std::size_t value, Region yes, Region no = {}, std::size_t scope = 0)
    {
        Region out;
        out.kind = Region::Kind::Choice;
        out.owner = 100 + value;
        out.condition = boundary.arena->test({value, scope});
        out.children = {std::move(yes), std::move(no)};
        return out;
    }
    void add(Region region) { projection.body.children.push_back(std::move(region)); }
    fs::FactoredUseResult build() const { return fs::FactoredUseBuilder(projection, boundary).take(); }
    void withIncoming()
    {
        boundary.incoming = {
            boundary.arena->incoming(70, Node::Boundary::Entry, Node::Role::Writer),
            boundary.arena->incoming(70, Node::Boundary::Entry, Node::Role::Reader)};
        boundary.following = {
            boundary.arena->incoming(71, Node::Boundary::Exit, Node::Role::Writer),
            boundary.arena->incoming(71, Node::Boundary::Exit, Node::Role::Reader)};
    }
};

// The independent oracle flattens only the selected *original* syntax, then
// scans concrete occurrence IDs. It reads no generated node or cached verdict.
static void trace(const Region& region, Evaluation& valuation, std::vector<std::size_t>& out)
{
    switch (region.kind) {
        case Region::Kind::Access:
            out.push_back(region.access);
            break;
        case Region::Kind::Choice:
            trace(region.children.at(valuation.condition(region.condition) ? 0 : 1), valuation, out);
            break;
        case Region::Kind::Sequence:
            for (const auto& child : region.children) {
                trace(child, valuation, out);
            }
            break;
        default:
            fail("non-fixed-use fixture passed to concrete scan");
    }
}
struct Concrete {
    Set writers, readers;
    Demands obligations;
    std::map<std::size_t, std::pair<Set, Set>> before, after;
};
static Concrete scan(const Fixture& f, const std::vector<std::size_t>& path, Set writers, Set readers, bool backward)
{
    Concrete out;
    out.writers = std::move(writers);
    out.readers = std::move(readers);
    auto visit = [&](std::size_t slot) {
        const auto& a = *f.projection.accesses.at(slot);
        const auto op = a.operation;
        out.before[op] = {out.writers, out.readers};
        if (!backward) {
            for (auto earlier : out.writers) {
                if (a.read) {
                    out.obligations.emplace(earlier, op, Node::Hazard::RAW);
                }
                if (a.write) {
                    out.obligations.emplace(earlier, op, Node::Hazard::WAW);
                }
            }
            if (a.write) {
                for (auto reader : out.readers) {
                    out.obligations.emplace(reader, op, Node::Hazard::WAR);
                }
            }
        }
        // Concrete fixed-cell last-writer scan. The weak extension below is tested
        // as a conservative may scan, not claimed exact for a particular subrange.
        if (a.write && a.definiteWrite) {
            out.writers.clear();
            out.readers.clear();
        }
        if (a.write) {
            out.writers.insert(op);
        }
        if (a.read && (backward || !a.write || !a.definiteWrite)) {
            out.readers.insert(op);
        }
        out.after[op] = {out.writers, out.readers};
    };
    if (backward) {
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            visit(*it);
        }
    } else {
        for (auto slot : path) {
            visit(slot);
        }
    }
    return out;
}
static std::set<std::pair<std::size_t, std::size_t>> closure(std::set<std::pair<std::size_t, std::size_t>> edges)
{
    bool changed;
    do {
        changed = false;
        const auto old = edges;
        for (auto a : old) {
            for (auto b : old) {
                if (a.second == b.first) {
                    changed |= edges.emplace(a.first, b.second).second;
                }
            }
        }
    } while (changed);
    return edges;
}
static void compare(const Fixture& f, std::map<fs::FactoredGuardIdentity, bool> choices, bool incoming = false)
{
    const auto formed = f.build();
    require(formed.complete, formed.reason);
    Evaluation values{*formed.arena, std::move(choices), {}, {}, {}, {}};
    Set iw, ir, nw, nr;
    if (incoming) {
        iw = {9000};
        ir = {9001, 9002};
        nw = {9010};
        nr = {9011, 9012};
        values.inputs = {
            {{70, Node::Boundary::Entry, Node::Role::Writer}, iw},
            {{70, Node::Boundary::Entry, Node::Role::Reader}, ir},
            {{71, Node::Boundary::Exit, Node::Role::Writer}, nw},
            {{71, Node::Boundary::Exit, Node::Role::Reader}, nr}};
    }
    std::vector<std::size_t> path;
    trace(f.projection.body, values, path);
    const auto forward = scan(f, path, iw, ir, false);
    const auto backward = scan(f, path, nw, nr, true);
    const auto actual = values.demands(formed.demands);
    require(actual == forward.obligations, "guarded requirements differ from independent scan");
    require(values.origins(formed.finalWriters) == forward.writers, "outgoing writers differ");
    require(values.origins(formed.finalReaders) == forward.readers, "outgoing readers differ");
    require(values.origins(formed.entryNextWriters) == backward.writers, "backward entry writers differ");
    require(values.origins(formed.entryNextReaders) == backward.readers, "backward entry readers differ");
    for (const auto& site : formed.sites) {
        const auto op = site.access->operation;
        const bool active = forward.before.count(op) != 0;
        require(values.condition(site.applicability) == active, "consumer applicability differs");
        Demands atSite;
        for (auto hazard : {Node::Hazard::RAW, Node::Hazard::WAR, Node::Hazard::WAW}) {
            const auto view = formed.requirement(op, hazard);
            if (!view.valid) {
                continue;
            }
            require(!view.hasUnresolved(), "fixed-use request unexpectedly needs a summary");
            require(
                view.targetEffects() ==
                    (hazard == Node::Hazard::RAW ? site.access->readIncidences : site.access->writeIncidences),
                "lost target-effect witnesses");
            const auto local = values.demands(view.demands);
            atSite.insert(local.begin(), local.end());
        }
        Demands expected;
        for (auto demand : actual) {
            if (std::get<1>(demand) == op) {
                expected.insert(demand);
            }
        }
        require(atSite == expected, "public demand view is not guarded by consumer applicability");
        if (!active) {
            continue;
        }
        require(values.origins(site.priorWriters) == forward.before.at(op).first, "incoming writers differ");
        require(values.origins(site.priorReaders) == forward.before.at(op).second, "incoming readers differ");
        require(values.origins(site.nextWriters) == backward.before.at(op).first, "next writers differ");
        require(values.origins(site.nextReaders) == backward.before.at(op).second, "next readers differ");
    }
    if (!incoming && formed.work.weakWrites == 0) {
        std::set<std::pair<std::size_t, std::size_t>> sparse, all;
        for (const auto& edge : actual) {
            sparse.emplace(std::get<0>(edge), std::get<1>(edge));
        }
        for (std::size_t i = 0; i < path.size(); ++i) {
            for (std::size_t j = i + 1; j < path.size(); ++j) {
                const auto &a = *f.projection.accesses[path[i]], &b = *f.projection.accesses[path[j]];
                if ((a.write && (b.read || b.write)) || (a.read && b.write)) {
                    all.emplace(a.operation, b.operation);
                }
            }
        }
        require(closure(sparse) == closure(all), "sparse generator lost an all-pairs concrete conflict");
    }
    ++executions;
}

static void cases()
{
    // Conditional replacement; the SAME value guards multiple distinct sites.
    for (bool weak : {false, true}) {
        Fixture f;
        f.withIncoming();
        f.add(f.op(false, true));                                 // W0
        f.add(f.op(true, false));                                 // A
        f.add(f.choice(1, f.op(true, true, !weak)));              // conditional RMW W1
        f.add(f.choice(1, f.op(true, false), f.op(true, false))); // same g, two consumers
        f.add(f.choice(2, f.op(true, false)));                    // independent optional reader
        f.add(f.op(false, true));
        for (bool g : {false, true}) {
            for (bool h : {false, true}) {
                compare(f, {{{1, 0}, g}, {{2, 0}, h}}, true);
            }
        }
        const auto& nodes = f.boundary.arena->nodes();
        require(
            std::count_if(nodes.begin(), nodes.end(), [](const Node& n) { return n.kind == Node::Kind::Test; }) == 2,
            "same original guard did not share its test reference");
    }
    // No local producer, explicit incoming writer AND multiple incoming readers.
    Fixture open;
    open.withIncoming();
    open.add(open.choice(3, open.op(false, true)));
    open.add(open.op(true, false));
    open.add(open.op(true, true));
    for (bool g : {false, true}) {
        compare(open, {{{3, 0}, g}}, true);
    }

    // Different occurrences of a defining value must not become one guard.
    Fixture scopes;
    scopes.add(scopes.choice(9, scopes.op(false, true), {}, 1));
    scopes.add(scopes.choice(9, scopes.op(true, false), {}, 2));
    for (bool a : {false, true}) {
        for (bool b : {false, true}) {
            compare(scopes, {{{9, 1}, a}, {{9, 2}, b}});
        }
    }

    // h need not have an original value on the g=false path. Do not speculate it.
    Fixture nested;
    nested.add(nested.op(false, true));
    nested.add(nested.choice(1, nested.choice(2, nested.op(true, false))));
    nested.add(nested.op(false, true));
    compare(nested, {{{1, 0}, false}});
    compare(nested, {{{1, 0}, true}, {{2, 0}, false}});
    compare(nested, {{{1, 0}, true}, {{2, 0}, true}});

    // Explicitly conservative partial/RMW updates keep the older histories.
    Fixture partial;
    partial.add(partial.op(false, true));
    partial.add(partial.op(true, false));
    partial.add(partial.op(true, true, false));
    partial.add(partial.op(true, false));
    partial.add(partial.op(false, true));
    compare(partial, {});
    auto p = partial.build();
    Evaluation ev{*p.arena, {}, {}, {}, {}, {}};
    require(ev.origins(p.site(3)->priorWriters) == Set({0, 2}), "partial write killed old contents");
    require(ev.origins(p.site(4)->priorReaders) == Set({1, 2, 3}), "partial RMW lost old readers");
    require(ev.demands(p.demands).count({1, 2, Node::Hazard::WAR}) == 1, "RMW queried after update");
    require(ev.origins(p.site(1)->nextReaders).count(2) == 1, "backward RMW read role missing");

    // Later overwrites must not mutate the old demand or witness node.
    const auto view = p.requirement(2, Node::Hazard::WAR);
    require(view.valid && view.targetEffects() == std::vector<std::size_t>({3, 7}), "RMW witness view missing");
    require(ev.demands(view.demands) == Demands({{1, 2, Node::Hazard::WAR}}), "old demand lost after full write");
}

static void composition()
{
    Fixture first;
    first.add(first.op(false, true));
    first.add(first.op(true, false));
    auto prefix = first.build();
    const auto oldSize = prefix.arena->nodes().size();
    const auto oldWriter = prefix.finalWriters;
    auto oldWitness = prefix.arena->nodes()[oldWriter].access;
    Fixture second;
    second.boundary.arena = prefix.arena;
    second.boundary.incoming = {prefix.finalWriters, prefix.finalReaders};
    auto r = second.op(true, false);
    auto w = second.op(false, true);
    // Keep original operation identities distinct across the two projected pieces.
    auto ra = std::make_shared<fs::FactoredUseAccess>(*second.projection.accesses[0]);
    ra->operation = 2;
    auto wa = std::make_shared<fs::FactoredUseAccess>(*second.projection.accesses[1]);
    wa->operation = 3;
    second.projection.accesses = {ra, wa};
    second.add(r);
    second.add(w);
    auto suffix = second.build();
    require(suffix.complete && suffix.arena == prefix.arena, "composition copied or changed the arena");
    require(suffix.site(2)->priorWriters == oldWriter, "incoming writer root was not referenced");
    require(suffix.work.inputNodes == oldSize, "incoming DAG not separately charged");
    first.boundary.following = {suffix.entryNextWriters, suffix.entryNextReaders};
    auto prefixWithContinuation = first.build();
    Evaluation ev{*suffix.arena, {}, {}, {}, {}, {}};
    require(
        ev.origins(prefixWithContinuation.site(1)->nextWriters) == Set({3}), "following interface lost next writer");
    require(
        ev.origins(prefixWithContinuation.site(1)->nextReaders) == Set({2}), "following interface lost next reader");
    prefix = {};
    require(
        suffix.arena->nodes()[oldWriter].access == oldWitness && oldWitness->writeIncidences.size() == 2,
        "composed source lost translated-effect witnesses");

    // Identical numeric roots in another occurrence frame are not interchangeable.
    auto incompatible = second.projection;
    incompatible.frame.kind = fs::FactoredUseFrame::Kind::ForBody;
    incompatible.frame.owner = 17;
    auto bad = fs::FactoredUseBuilder(incompatible, second.boundary).take();
    require(!bad.complete && !bad.reason.empty(), "cross-frame input accepted without transport");
    incompatible = second.projection;
    incompatible.frame.cell = 1;
    bad = fs::FactoredUseBuilder(incompatible, second.boundary).take();
    require(!bad.complete, "cross-cell input accepted");
    auto wrongSort = second.boundary;
    wrongSort.incoming.writers = suffix.demands;
    bad = fs::FactoredUseBuilder(second.projection, wrongSort).take();
    require(!bad.complete, "demand root used as an incoming writer expression");
}

static void repeats()
{
    Fixture f;
    f.add(f.op(false, true));
    Region repeat;
    repeat.kind = Region::Kind::OpaqueRepeat;
    repeat.owner = 20;
    f.add(repeat); // all-access exclusion: other cells only
    f.add(f.op(true, false));
    auto result = f.build();
    require(result.complete && result.repeatedInterfaces.empty(), "cell-disjoint loop rejected entire projection");
    f.projection.body.children[1].mayRead = true;
    result = f.build();
    require(
        !result.complete && result.repeatedInterfaces.size() == 2, "repeat premise not preserved in both directions");
    require(
        !result.requirement(1, Node::Hazard::RAW).hasUnresolved(), "read-only loop destroyed unchanged writer facts");
    f.add(f.op(false, true));
    f.add(f.op(true, false));
    result = f.build();
    require(result.requirement(2, Node::Hazard::WAR).hasUnresolved(), "unresolved loop readers disappeared");
    require(
        !result.requirement(3, Node::Hazard::RAW).hasUnresolved(),
        "full local overwrite failed to restore provenance precision");
    require(result.arena->nodes()[result.demands].containsUnresolved, "later full write erased loop demands");

    fs::FactoredUseFrame frame;
    frame.kind = fs::FactoredUseFrame::Kind::ForBody;
    frame.owner = 20;
    Fixture body(frame);
    body.withIncoming();
    body.add(body.op(true, false));
    body.add(body.op(false, true));
    body.add(body.op(true, false));
    compare(body, {}, true);
    auto visit = body.build();
    require(
        visit.arena->nodes()[visit.site(0)->priorWriters].containsIncoming,
        "a local body visit silently started with fresh history");
    require(
        !visit.arena->nodes()[visit.site(2)->priorWriters].containsIncoming,
        "body's proved full write failed to replace the incoming writer");
}

static void partialBytes()
{
    Fixture f;
    f.add(f.op(false, true));
    f.add(f.op(true, false));
    f.add(f.op(false, true, false));
    f.add(f.op(true, false));
    f.add(f.op(false, true));
    const auto result = f.build();
    Evaluation evaluated{*result.arena, {}, {}, {}, {}, {}};
    const auto possibleWriters = evaluated.origins(result.site(3)->priorWriters);
    const auto possibleReaders = evaluated.origins(result.site(4)->priorReaders);
    std::set<std::pair<std::size_t, std::size_t>> generated;
    for (auto [source, target, hazard] : evaluated.demands(result.demands)) {
        generated.emplace(source, target);
    }
    generated = closure(generated);
    // A coarse may-write covers zero, one, or both represented bytes. Compare
    // against a byte-level scanner, not against the production weak transfer.
    for (unsigned writeMask = 0; writeMask != 4; ++writeMask) {
        std::array<unsigned, 5> reads{{0, 3, 0, 3, 0}};
        std::array<unsigned, 5> writes{{3, 0, writeMask, 0, 3}};
        std::array<std::size_t, 2> last{{fs::NoFactoredId, fs::NoFactoredId}};
        std::array<Set, 2> pending;
        for (std::size_t op = 0; op < 5; ++op) {
            if (op == 3) {
                for (auto origin : last) {
                    require(possibleWriters.count(origin), "partial write lost a concrete byte origin");
                }
            }
            if (op == 4) {
                for (const auto& set : pending) {
                    for (auto reader : set) {
                        require(possibleReaders.count(reader), "partial write lost a concrete outstanding byte reader");
                    }
                }
            }
            for (std::size_t earlier = 0; earlier < op; ++earlier) {
                if ((writes[earlier] & (reads[op] | writes[op])) || (reads[earlier] & writes[op])) {
                    require(generated.count({earlier, op}), "coarse generator missed an exact byte conflict");
                }
            }
            for (unsigned byte = 0; byte != 2; ++byte) {
                if (writes[op] & (1U << byte)) {
                    last[byte] = op;
                    pending[byte].clear();
                } else if (reads[op] & (1U << byte)) {
                    pending[byte].insert(op);
                }
            }
        }
        ++executions;
    }
}

static void malformed()
{
    Fixture missing;
    missing.op(true, false); // Metadata without a syntax occurrence is not complete.
    require(!missing.build().complete, "unrepresented access accepted");
    Fixture duplicate;
    auto operation = duplicate.op(true, false);
    duplicate.add(operation);
    duplicate.add(operation);
    require(!duplicate.build().complete, "duplicate occurrence accepted as fixed use");
    Fixture malformedChoice;
    auto choice = malformedChoice.choice(1, malformedChoice.op(false, true));
    choice.children.pop_back();
    malformedChoice.add(choice);
    require(!malformedChoice.build().complete, "unary choice executed as a sequence");
    Fixture missingGuard;
    choice = missingGuard.choice(1, missingGuard.op(false, true));
    choice.condition = fs::NoFactoredId;
    missingGuard.add(choice);
    missingGuard.add(missingGuard.op(true, false));
    const auto unknown = missingGuard.build();
    require(
        !unknown.complete && unknown.requirement(1, Node::Hazard::RAW).hasUnresolved(),
        "unknown guard was silently made false");
    fs::FactoredUseInterface noArena;
    const auto invalid = fs::FactoredUseBuilder({}, noArena).take();
    require(
        !invalid.complete && invalid.nodes().empty() && !invalid.requirement(0, Node::Hazard::RAW).valid,
        "invalid public result is unsafe to inspect");
}

static void randomCases()
{
    std::mt19937 rng(0x440043);
    for (unsigned trial = 0; trial < 40; ++trial) {
        Fixture f;
        for (unsigned i = 0; i < 9; ++i) {
            const auto effect = 1 + rng() % 3;
            auto op = f.op(effect & 1, effect & 2, trial % 4 != 0 || i % 3 != 0);
            if (rng() % 3 != 0) {
                op = f.choice(1 + rng() % 3, std::move(op));
            }
            if (rng() % 5 == 0) {
                op = f.choice(1 + rng() % 3, std::move(op));
            }
            f.add(std::move(op));
        }
        for (unsigned bits = 0; bits < 8; ++bits) {
            compare(f, {{{1, 0}, bool(bits & 1)}, {{2, 0}, bool(bits & 2)}, {{3, 0}, bool(bits & 4)}});
        }
    }
}

static std::size_t syntaxSize(const Region& r)
{
    std::size_t count = 1;
    for (const auto& c : r.children) {
        count += syntaxSize(c);
    }
    return count;
}
static void formation()
{
    for (std::size_t m : {1, 8, 64, 512, 4096}) {
        Fixture f;
        f.add(f.op(false, true));
        for (std::size_t i = 0; i < m; ++i) {
            f.add(f.choice(i + 1, f.op(true, false)));
        }
        f.add(f.op(false, true));
        const auto beforeOrigins = materializedOrigins, beforeDemands = materializedDemands;
        const auto result = f.build();
        const auto q = syntaxSize(f.projection.body);
        require(result.complete && result.sites.size() == m + 2, "optional-reader formation failed");
        require(result.work.forwardSteps == q && result.work.backwardSteps == q, "revisited a fixed-use syntax node");
        require(
            result.work.addedNodes <= 32 * q && result.work.constructorCalls <= 64 * q, "nonlinear construction work");
        require(
            beforeOrigins == materializedOrigins && beforeDemands == materializedDemands,
            "formation enumerated output");
        std::cout << "readers=" << m << " syntax=" << q << " input-nodes=" << result.work.inputNodes
                  << " added-nodes=" << result.work.addedNodes << " constructor-calls=" << result.work.constructorCalls
                  << '\n';
    }
}

int main()
{
    cases();
    composition();
    repeats();
    partialBytes();
    malformed();
    randomCases();
    formation();
    std::cout << "factored provenance checks passed; concrete-executions=" << executions
              << " materialized-origin-members=" << materializedOrigins
              << " materialized-demand-members=" << materializedDemands << '\n';
    return 0;
}
