// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
// Standalone semantic checks of the production D4 cores. No assertions vanish
// under NDEBUG. Finite traces are test oracles, never the production algorithm.
#include "PTO/Transforms/FrontierSynch/D4Composition.h"
#include "PTO/Transforms/FrontierSynch/D4Relations.h"
#include <iostream>
#include <cstdlib>
#include <tuple>
#include <unordered_map>

namespace fs = mlir::pto::frontiersynch;
using N = fs::FactoredUseNode;
using R = fs::FactoredUseRegion;
using Origins = std::set<std::pair<bool, std::size_t>>; // incoming handle / original operation
using Demand = std::tuple<N::Hazard, bool, std::size_t, std::size_t>;
using Demands = std::set<Demand>;
static std::size_t checks = 0, executions = 0;
[[noreturn]] static void fail(const char* text)
{
    std::cerr << "D4 failure after " << checks << " checks: " << text << '\n';
    std::exit(1);
}
static void require(bool value, const char* text)
{
    ++checks;
    if (!value) {
        fail(text);
    }
}
struct Eval {
    const fs::FactoredUseArena& arena;
    std::map<fs::FactoredGuardIdentity, bool> guards;
    bool condition(std::size_t root) const
    {
        const auto& node = arena[root];
        switch (node.kind) {
            case N::Kind::Empty: return false;
            case N::Kind::True: return true;
            case N::Kind::Test: return guards.at(node.guard);
            case N::Kind::Choose: return condition(condition(node.condition) ? node.left : node.right);
            default: fail("unqualified condition in exact oracle");
        }
    }
    Origins origins(std::size_t root) const
    {
        const auto& node = arena[root];
        switch (node.kind) {
            case N::Kind::Empty: return {};
            case N::Kind::Incoming: return {{true, root}};
            case N::Kind::Access: return {{false, node.operation}};
            case N::Kind::Choose: return origins(condition(node.condition) ? node.left : node.right);
            case N::Kind::Both: {
                auto values = origins(node.left);
                const auto right = origins(node.right);
                values.insert(right.begin(), right.end());
                return values;
            }
            default: fail("unqualified origin in exact oracle");
        }
    }
    Demands demands(std::size_t root) const
    {
        const auto& node = arena[root];
        switch (node.kind) {
            case N::Kind::Empty: return {};
            case N::Kind::Choose: return demands(condition(node.condition) ? node.left : node.right);
            case N::Kind::Both: {
                auto values = demands(node.left);
                const auto right = demands(node.right);
                values.insert(right.begin(), right.end());
                return values;
            }
            case N::Kind::Demand: {
                Demands values;
                for (auto [incoming, source] : origins(node.left)) {
                    values.insert({node.hazard, incoming, source, node.operation});
                }
                return values;
            }
            default: fail("unqualified demand in exact oracle");
        }
    }
};
struct Fixture {
    fs::FactoredUseProjection projection;
    fs::FactoredUseInterface boundary;
    std::vector<fs::FactoredGuardIdentity> guards;
    Fixture()
    {
        projection.frame.original = 123;
        projection.frame.owner = 17;
        projection.frame.cell = 3;
        projection.frame.snapshot = fs::OriginalProgramVersion::fresh();
        boundary.arena = std::make_shared<fs::FactoredUseArena>(projection.frame);
        auto& arena = *boundary.arena;
        boundary.incoming = {arena.incoming(17, N::Boundary::Entry, N::Role::Writer),
                             arena.incoming(17, N::Boundary::Entry, N::Role::Reader)};
        boundary.following = {arena.incoming(17, N::Boundary::Exit, N::Role::Writer),
                              arena.incoming(17, N::Boundary::Exit, N::Role::Reader)};
    }
    R access(bool read, bool write, bool full = false)
    {
        auto value = std::make_shared<fs::FactoredUseAccess>();
        value->operation = projection.accesses.size();
        value->read = read;
        value->write = write;
        value->definiteWrite = full;
        if (read) { value->readIncidences = {2}; }
        if (write) { value->writeIncidences = {5}; }
        R out;
        out.kind = R::Kind::Access;
        out.access = value->operation;
        projection.accesses.push_back(value);
        return out;
    }
    std::size_t guard()
    {
        guards.push_back({100 + guards.size(), 17});
        return boundary.arena->test(guards.back());
    }
    R choice(std::size_t condition, R yes, R no = {})
    {
        R out;
        out.kind = R::Kind::Choice;
        out.condition = condition;
        out.children = {std::move(yes), std::move(no)};
        return out;
    }
};
static R seq(std::initializer_list<R> children)
{
    R out;
    out.children.assign(children.begin(), children.end());
    return out;
}
static void trace(const R& region, const Eval& eval, std::vector<std::size_t>& out)
{
    if (region.kind == R::Kind::Access) {
        out.push_back(region.access);
    } else if (region.kind == R::Kind::Choice) {
        trace(region.children[eval.condition(region.condition) ? 0 : 1], eval, out);
    } else if (region.kind == R::Kind::Sequence) {
        for (const auto& child : region.children) { trace(child, eval, out); }
    } else {
        fail("test oracle does not turn a repeat into one visit");
    }
}
// Independent concrete set scan (does not call the production transfer or DAG
// substitution). Check all old-state hazards before any provenance update.
static void checkTrace(const Fixture& fixture, const fs::FactoredUseResult& actual, const Eval& eval)
{
    std::vector<std::size_t> visits;
    trace(fixture.projection.body, eval, visits);
    Origins writers = eval.origins(fixture.boundary.incoming.writers);
    Origins readers = eval.origins(fixture.boundary.incoming.readers);
    Demands expected;
    std::set<std::size_t> visited;
    for (auto index : visits) {
        const auto& access = *fixture.projection.accesses[index];
        visited.insert(access.operation);
        const auto* site = actual.site(access.operation);
        require(site && site->access == fixture.projection.accesses[index], "lost translated-effect witness");
        require(eval.condition(site->applicability), "wrong child applicability");
        require(eval.origins(site->priorWriters) == writers, "incorrect incoming writer substitution");
        require(eval.origins(site->priorReaders) == readers, "incorrect incoming reader substitution");
        auto add = [&](const Origins& sources, N::Hazard hazard) {
            Demands here;
            for (auto [incoming, source] : sources) {
                here.insert({hazard, incoming, source, access.operation});
            }
            require(eval.demands(site->demands[static_cast<std::size_t>(hazard)]) == here,
                    "child lost an old-state demand");
            expected.insert(here.begin(), here.end());
        };
        if (access.read) { add(writers, N::Hazard::RAW); }
        if (access.write) {
            add(writers, N::Hazard::WAW);
            add(readers, N::Hazard::WAR);
        }
        if (access.write && access.definiteWrite) {
            writers = {{false, access.operation}};
            readers.clear();
        } else {
            if (access.write) { writers.insert({false, access.operation}); }
            if (access.read) { readers.insert({false, access.operation}); }
        }
    }
    require(eval.demands(actual.demands) == expected, "composed requirements differ from concrete scan");
    require(eval.origins(actual.finalWriters) == writers && eval.origins(actual.finalReaders) == readers,
            "incorrect composed outgoing history");
    for (const auto& site : actual.sites) {
        if (!visited.count(site.access->operation)) {
            require(!eval.condition(site.applicability), "bypassed child became applicable");
            for (auto root : site.demands) { require(eval.demands(root).empty(), "bypass generated a demand"); }
        }
    }
    writers = eval.origins(fixture.boundary.following.writers);
    readers = eval.origins(fixture.boundary.following.readers);
    for (auto it = visits.rbegin(); it != visits.rend(); ++it) {
        const auto& access = *fixture.projection.accesses[*it];
        const auto& site = *actual.site(access.operation);
        require(eval.origins(site.nextWriters) == writers && eval.origins(site.nextReaders) == readers,
                "incorrect enclosing continuation substitution");
        if (access.write && access.definiteWrite) {
            writers = {{false, access.operation}};
            readers = access.read ? Origins{{false, access.operation}} : Origins{};
        } else {
            if (access.write) { writers.insert({false, access.operation}); }
            if (access.read) { readers.insert({false, access.operation}); }
        }
    }
    require(eval.origins(actual.entryNextWriters) == writers && eval.origins(actual.entryNextReaders) == readers,
            "incorrect backward composition");
    ++executions;
}
static void checkFixture(Fixture& fixture)
{
    auto flat = fs::FactoredUseBuilder(fixture.projection, fixture.boundary).take();
    auto composed = fs::buildD4Composition(fixture.projection, fixture.boundary);
    require(flat.complete && composed.uses.complete, "supported D4 fixture failed formation");
    require(composed.uses.frame == fixture.projection.frame, "child replaced physical owner");
    require(composed.work.applications == composed.work.summaries, "summary application accounting");
    for (std::size_t bits = 0; bits < (std::size_t(1) << fixture.guards.size()); ++bits) {
        Eval eval{*fixture.boundary.arena, {}};
        for (std::size_t g = 0; g < fixture.guards.size(); ++g) {
            eval.guards[fixture.guards[g]] = (bits >> g) & 1;
        }
        checkTrace(fixture, composed.uses, eval);
        checkTrace(fixture, flat, eval);
        require(eval.demands(flat.demands) == eval.demands(composed.uses.demands), "flat/wrapped demand mismatch");
    }
}
static void provenance()
{
    // A split inside one unfinished use and a complete-use boundary. The empty
    // scalar child models a local selector reset; it cannot reset physical history.
    Fixture f;
    const auto w0 = f.access(false, true, true);
    const auto r0 = f.access(true, false);
    const auto reset = f.access(false, false);
    const auto r1 = f.access(true, false);
    const auto w1 = f.access(false, true, true);
    const auto r2 = f.access(true, false);
    f.projection.body = seq({seq({w0, seq({r0})}), seq({reset, r1}), seq({w1, r2})});
    checkFixture(f);
    const auto g = f.guard(), h = f.guard();
    // Reuse the SAME original guard on both sites; independent h stays factored.
    f.projection.body = seq({w0, f.choice(g, r0), reset, f.choice(g, w1), f.choice(h, r1), r2});
    checkFixture(f);
    for (bool full : {false, true}) {
        Fixture mixed;
        const auto guard = mixed.guard();
        const auto w = mixed.access(false, true, true);
        const auto r = mixed.access(true, false);
        const auto rmw = mixed.access(true, true, full);
        const auto final = mixed.access(false, true, true);
        mixed.projection.body = seq({seq({w, r}), mixed.choice(guard, seq({rmw})), final});
        checkFixture(mixed);
    }
    // Exhaust every small fixed-mode sequence, with nonfresh input AND following
    // histories. Child boundaries must not erase incoming WAR or backward RMW.
    for (unsigned code = 0; code < 256; ++code) {
        Fixture small;
        unsigned modes = code;
        for (unsigned i = 0; i < 4; ++i, modes >>= 2) {
            const auto mode = modes & 3;
            small.projection.body.children.push_back(seq({small.access(mode & 1, mode & 2, mode == 2)}));
        }
        checkFixture(small);
    }
}
static void invalidAndOpaque()
{
    Fixture f;
    f.projection.body = seq({f.access(false, true, true), f.access(true, false)});
    auto summary = std::make_shared<fs::D4ChildSummary>(f.projection, f.boundary.arena);
    const auto before = f.boundary.incoming;
    auto first = fs::composeD4Children({summary}, f.boundary);
    require(first.uses.complete, "single summary failed");
    auto fresh = f.boundary;
    fresh.incoming = {};
    auto second = fs::composeD4Children({summary}, fresh);
    require(second.uses.complete && second.uses.site(0)->priorReaders == 0, "explicit input not applied");
    require(first.uses.site(0)->priorReaders == before.readers, "new application mutated old roots");
    auto duplicate = fs::composeD4Children({summary, summary}, f.boundary);
    require(!duplicate.uses.complete && duplicate.uses.finalReaders == before.readers,
            "two dynamic visits accidentally share static origin identity");
    auto otherFrame = f.projection.frame;
    otherFrame.owner = 88;
    fs::FactoredUseInterface other;
    other.arena = std::make_shared<fs::FactoredUseArena>(otherFrame);
    require(!fs::composeD4Children({summary}, other).uses.complete, "cross-owner roots accepted without transport");
    otherFrame = f.projection.frame;
    otherFrame.snapshot = fs::OriginalProgramVersion::fresh();
    other.arena = std::make_shared<fs::FactoredUseArena>(otherFrame);
    require(!fs::composeD4Children({summary}, other).uses.complete, "stale original snapshot accepted");
    auto bad = f.boundary;
    bad.incoming.writers = 1; // true is not an origin set
    require(!fs::composeD4Children({summary}, bad).uses.complete, "invalid history sort accepted");
    R repeat;
    repeat.kind = R::Kind::OpaqueRepeat;
    repeat.owner = 99;
    f.projection.body = seq({f.projection.body.children[0], seq({repeat}), f.projection.body.children[1]});
    auto empty = fs::buildD4Composition(f.projection, f.boundary);
    require(empty.uses.complete, "All-excluded repeat should preserve physical histories");
    f.projection.body.children[1].children[0].mayWrite = true;
    auto unknown = fs::buildD4Composition(f.projection, f.boundary);
    require(!unknown.uses.complete && unknown.uses.repeatedInterfaces.size() == 2,
            "unknown repeated use was made into a single/fresh visit");
    require(unknown.uses.site(1) && unknown.uses.nodes()[unknown.uses.site(1)->priorWriters].containsUnresolved,
            "unknown child did not block exact provenance");
    require(unknown.uses.nodes()[unknown.uses.demands].containsIncoming &&
                unknown.uses.nodes()[unknown.uses.demands].containsUnresolved,
            "unsupported child lost incoming demands");
}
static fs::OriginalInterval interval(Fixture& f)
{
    fs::OriginalInterval out;
    out.owner = f.projection.frame.owner;
    out.query.version = f.projection.frame.snapshot;
    out.query.selector.cell = f.projection.frame.cell;
    out.query.start = fs::OriginalCut::childBoundary(out.owner, 0, fs::OriginalCut::Before);
    out.query.stop = fs::OriginalCut::childBoundary(out.owner, 0, fs::OriginalCut::After);
    out.query.continuationOwner = out.owner;
    out.query.occurrence.incomingInterface = 74;
    out.query.occurrence.qualification = 31;
    return out;
}
static fs::D4UseRole role(const fs::OriginalInterval& context, std::size_t site, std::size_t coordinate)
{
    return {site, coordinate, context.owner, context.query.selector.cell, {site, fs::OriginalCut::After}};
}
static fs::D4ChildUse child(
    const fs::OriginalInterval& context, std::size_t id, fs::D4UseRole first, fs::D4UseRole last,
    std::size_t condition = 1)
{
    fs::D4ChildUse out;
    out.kind = fs::D4ChildUse::Kind::CompleteUses;
    out.rule = fs::D4ChildUse::Rule::D2;
    out.interpretation = context;
    out.child = id;
    out.first = first;
    out.last = last;
    out.nonempty = condition;
    out.occurrenceQualified = true; // Supplied qualified child maps in these rule tests.
    auto existing = std::make_shared<fs::D4ExistingCorrespondence>();
    existing->rule = out.rule;
    existing->interpretation = context;
    existing->firstUse = first;
    existing->lastUse = last;
    existing->nonempty = condition;
    existing->source = first.site;
    existing->target = last.site;
    existing->selectorOwner = id;
    existing->previousDistance = existing->nextDistance = 2;
    existing->qualified = true;
    out.correspondence = std::move(existing);
    return out;
}
static const fs::D4UseExpression& selected(
    const fs::D4RelationComposition& result, std::size_t root, const Eval& eval)
{
    for (;;) {
        const auto& node = result.expressions.at(root);
        if (node.kind != fs::D4UseExpression::Kind::Choose) {
            return node;
        }
        root = eval.condition(node.condition) ? node.yes : node.no;
    }
}
static void relationComposition()
{
    Fixture f;
    const auto context = interval(f);
    const auto g = f.guard(), h = f.guard();
    auto a = child(context, 21, role(context, 0, 100), role(context, 1, 102), g);
    auto b = child(context, 22, role(context, 2, 200), role(context, 3, 202), h);
    auto c = child(context, 23, role(context, 4, 300), role(context, 5, 302), g);
    const auto result = fs::composeD4Relations(context, f.boundary.arena, {a, b, c});
    require(result.complete, "qualified D2 child composition rejected");
    require(result.children.front().correspondence == a.correspondence &&
                result.children.front().correspondence->previousDistance == 2 &&
                result.children.front().correspondence->initialDomain == fs::NoFactoredId,
            "D4 replaced a child relation or fabricated its unresolved entry domain");
    for (bool gv : {false, true}) {
        for (bool hv : {false, true}) {
            Eval eval{*f.boundary.arena, {{f.guards[0], gv}, {f.guards[1], hv}}};
            std::vector<fs::D4ChildUse> active;
            for (const auto& use : {a, b, c}) {
                if (eval.condition(use.nonempty)) { active.push_back(use); }
            }
            for (const auto& link : result.links) {
                if (!eval.condition(link.applicability)) { continue; }
                const auto at = std::find_if(active.begin(), active.end(), [&](const auto& use) {
                    return use.child == link.child;
                });
                require(at != active.end(), "link applies on wrong branch");
                const auto& target = selected(result, link.otherUse, eval);
                if ((!link.backward && at == active.begin()) || (link.backward && at + 1 == active.end())) {
                    require(target.kind == fs::D4UseExpression::Kind::Incoming,
                            "child reset fabricated a previous/next local use");
                } else {
                    const auto& expected = link.backward ? (at + 1)->first : (at - 1)->last;
                    require(target.kind == fs::D4UseExpression::Kind::Use && target.use == expected,
                            "physical predecessor/successor changed across child boundary");
                }
            }
        }
    }
    auto sameVisit = a;
    auto sameVisitRelation = std::make_shared<fs::D4ExistingCorrespondence>(*a.correspondence);
    sameVisitRelation->previousDistance = sameVisitRelation->nextDistance = 0;
    sameVisit.correspondence = sameVisitRelation;
    const auto zero = fs::composeD4Relations(context, f.boundary.arena, {sameVisit, b});
    require(zero.complete && zero.children.front().correspondence->previousDistance == 0,
            "qualified D2 same-visit relation was rejected or changed");
    sameVisitRelation->target = sameVisitRelation->source;
    require(!fs::composeD4Relations(context, f.boundary.arena, {sameVisit, b}).complete,
            "D2 same-occurrence self edge was accepted");
    // D1 profiles use the SAME composition, without manufacturing a period.
    for (auto* profile : {&a, &b, &c}) {
        profile->rule = fs::D4ChildUse::Rule::D1;
        auto existing = std::make_shared<fs::D4ExistingCorrespondence>(*profile->correspondence);
        existing->rule = profile->rule;
        existing->previousDistance.reset();
        existing->nextDistance.reset();
        profile->correspondence = std::move(existing);
    }
    require(fs::composeD4Relations(context, f.boundary.arena, {a, b, c}).complete, "D1 child transport failed");
    fs::D4ChildUse empty;
    empty.interpretation = context;
    empty.child = 88;
    empty.noUseProved = true;
    require(fs::composeD4Relations(context, f.boundary.arena, {a, empty, b}).complete,
            "empty structural child changed physical use sequence");
    empty.noUseProved = false;
    auto unknown = fs::composeD4Relations(context, f.boundary.arena, {a, empty, b});
    require(!unknown.complete && !unknown.unresolved.empty(), "unproved exclusion accepted");
    const auto link = std::find_if(unknown.links.begin(), unknown.links.end(), [](const auto& value) {
        return !value.backward && value.child == 22;
    });
    require(link != unknown.links.end() && unknown.expressions[link->otherUse].kind == fs::D4UseExpression::Kind::Unknown,
            "unknown child silently bypassed its physical obligations");
    const auto retained = b.correspondence;
    b.correspondence.reset();
    require(!fs::composeD4Relations(context, f.boundary.arena, {a, b}).complete,
            "D4 created a child correspondence from its boundary profile");
    b.correspondence = retained;
    auto mismatched = b;
    ++mismatched.first.coordinate;
    require(!fs::composeD4Relations(context, f.boundary.arena, {a, mismatched}).complete,
            "D4 attached a qualified relation to another dynamic boundary");
    mismatched = b;
    mismatched.nonempty = 1;
    require(!fs::composeD4Relations(context, f.boundary.arena, {a, mismatched}).complete,
            "D4 changed participation without requalifying its relation");
    b.occurrenceQualified = false;
    b.missingPremise = "enclosing first/last bank-use correspondence";
    unknown = fs::composeD4Relations(context, f.boundary.arena, {a, b});
    require(!unknown.complete && unknown.unresolved.front().second == b.missingPremise,
            "missing child relation not reported");
    b.occurrenceQualified = true;
    b.interpretation.query.includeStoppingAccess = true;
    require(!fs::composeD4Relations(context, f.boundary.arena, {a, b}).complete,
            "different stopping convention reused a D4 result");
    b.interpretation = context;
    b.interpretation.query.occurrence.incomingInterface = 999;
    require(!fs::composeD4Relations(context, f.boundary.arena, {a, b}).complete,
            "different incoming occurrence interpretation accepted");
    b.interpretation = context;
    b.first.owner = 22; // a restarted local child cannot become physical owner
    require(!fs::composeD4Relations(context, f.boundary.arena, {a, b}).complete, "local reset changed physical owner");
    // This test supplies the qualified maps, it does NOT claim to derive them
    // from general affine/carried selectors (draft I.1 explicitly leaves that open).
    for (bool resetSelector : {false, true}) {
        for (unsigned bank = 0; bank < 2; ++bank) {
            std::vector<std::pair<unsigned, unsigned>> visits;
            std::vector<fs::D4ChildUse> profiles;
            for (unsigned outer = 0; outer < 4; ++outer) {
                std::vector<unsigned> local;
                for (unsigned j = 0; j < 3; ++j) {
                    const auto selectedBank = (resetSelector ? j : 3 * outer + j) % 2;
                    if (selectedBank == bank) { local.push_back(j); visits.push_back({outer, j}); }
                }
                profiles.push_back(child(context, outer,
                    role(context, 90, 3 * outer + local.front()), role(context, 90, 3 * outer + local.back())));
            }
            auto composed = fs::composeD4Relations(context, f.boundary.arena, profiles);
            require(composed.complete, "qualified nested-selector profiles failed");
            Eval eval{*f.boundary.arena, {}};
            for (const auto& edge : composed.links) {
                if (edge.backward || edge.child == 0) { continue; }
                const auto current = std::find_if(visits.begin(), visits.end(), [&](auto visit) {
                    return 3 * visit.first + visit.second == edge.endpoint.coordinate;
                });
                require(current != visits.begin() && current != visits.end(), "bad finite oracle fixture");
                const auto& predecessor = selected(composed, edge.otherUse, eval);
                const auto previous = *std::prev(current);
                require(predecessor.kind == fs::D4UseExpression::Kind::Use &&
                            predecessor.use.coordinate == 3 * previous.first + previous.second,
                        "selector restart reset the physical predecessor");
            }
        }
    }
}
static void unfinishedUse()
{
    Fixture f;
    const auto context = interval(f);
    const auto g = f.guard(), h = f.guard();
    const auto writer = role(context, 0, 123);
    auto a = child(context, 30, role(context, 1, 123), role(context, 2, 123), g);
    auto b = child(context, 31, role(context, 3, 123), role(context, 4, 123), h);
    a.kind = b.kind = fs::D4ChildUse::Kind::ReadFragment;
    a.enclosingUse = b.enclosingUse = writer;
    auto retainEnclosing = [](fs::D4ChildUse& profile, const fs::D4UseRole& use) {
        auto relation = std::make_shared<fs::D4ExistingCorrespondence>(*profile.correspondence);
        relation->enclosingUse = use;
        profile.correspondence = std::move(relation);
    };
    retainEnclosing(a, writer);
    retainEnclosing(b, writer);
    auto readers = fs::composeD4ReadFragments(context, f.boundary.arena, writer, {a, b});
    require(readers.complete && readers.enclosingUse == writer && readers.boundaries.links.empty(),
            "read fragment incorrectly advanced complete-use predecessor");
    for (bool gv : {false, true}) {
        for (bool hv : {false, true}) {
            Eval eval{*f.boundary.arena, {{f.guards[0], gv}, {f.guards[1], hv}}};
            require(eval.condition(readers.nonempty) == (gv || hv), "wrong D3 nonempty composition");
            const auto& first = selected(readers.boundaries, readers.first, eval);
            const auto& last = selected(readers.boundaries, readers.last, eval);
            if (!gv && !hv) {
                require(first.kind == fs::D4UseExpression::Kind::NoUse && last.kind == first.kind,
                        "empty read path fabricated a frontier");
            } else {
                require(first.use == (gv ? a.first : b.first) && last.use == (hv ? b.last : a.last),
                        "D4 did not compose D3 first/last equations");
            }
        }
    }
    require(readers.boundaries.children.front().kind == fs::D4ChildUse::Kind::ReadFragment,
            "composed child lost its unfinished-use interpretation");
    require(!fs::composeD4Relations(context, f.boundary.arena, {a, b}).complete,
            "unfinished read fragment admitted as a complete use");
    const fs::D4UseRole incoming{fs::NoControlId, context.query.occurrence.incomingInterface,
        context.owner, context.query.selector.cell, fs::OriginalCut::scope(context.owner, fs::OriginalCut::Before)};
    auto importedA = a, importedB = b;
    importedA.enclosingUse = importedB.enclosingUse = incoming;
    retainEnclosing(importedA, incoming);
    retainEnclosing(importedB, incoming);
    require(fs::composeD4ReadFragments(context, f.boundary.arena, incoming, {importedA, importedB}).complete,
            "D4 read composition lost its explicit incoming use");
    b.enclosingUse = role(context, 9, 124); // A real reload, not a lexical boundary.
    require(!fs::composeD4ReadFragments(context, f.boundary.arena, writer, {a, b}).complete,
            "intervening writer allowed stale same-episode transport");
}
static void growth()
{
    for (std::size_t count : {64, 256, 1024, 4096}) {
        Fixture f;
        f.projection.body.children.push_back(f.access(false, true, true));
        for (std::size_t i = 0; i < count; ++i) {
            f.projection.body.children.push_back(f.choice(f.guard(), f.access(true, false)));
        }
        f.projection.body.children.push_back(f.access(false, true, true));
        const auto out = fs::buildD4Composition(f.projection, f.boundary);
        require(out.uses.complete && out.uses.sites.size() == count + 2, "large shared formation failed");
        require(out.work.substitutedNodes <= 40 * (count + 2), "D4 application expanded unrelated guard histories");
        require(out.uses.work.addedNodes <= 70 * (count + 2), "D4 node formation is not within restricted bound");
        std::cout << "readers=" << count << " summaries=" << out.work.summaries
                  << " nodes=" << out.uses.work.addedNodes << " substituted=" << out.work.substitutedNodes << '\n';
    }
}
// Canonicalization is shared by all native ProgramAnalysis consumers, not a
// private reader-query rewrite. Test payload order and real control boundaries.
static void originalWrappers()
{
    using Region = fs::Region;
    auto leaf = [](std::size_t op) {
        Region result;
        result.kind = Region::Operation;
        result.operation = op;
        return result;
    };
    auto wrap = [](Region child) {
        Region result;
        result.children.push_back(std::move(child));
        return result;
    };
    Region sequence;
    sequence.children = {wrap(wrap(leaf(0))), wrap(leaf(1))};
    Region choice;
    choice.kind = Region::Choice;
    choice.originalOwner = 30;
    choice.children = {wrap(wrap(leaf(2))), wrap(wrap(leaf(3)))};
    Region counted;
    counted.kind = Region::For;
    counted.originalOwner = 40;
    counted.zeroTripPossible = true;
    counted.qualifiedCounted = true;
    counted.children = {wrap(wrap(leaf(4)))};
    Region loop;
    loop.kind = Region::While;
    loop.originalOwner = 50;
    loop.children = {wrap(wrap(leaf(5))), wrap(wrap(leaf(6)))};
    Region named = wrap(wrap(leaf(7)));
    named.originalOwner = 60;
    Region root;
    root.children = {std::move(sequence), std::move(choice), std::move(counted), std::move(loop), std::move(named)};
    fs::normalizeOriginalSequences(root);
    require(root.children.size() == 6, "anonymous wrappers changed semantic sequence length");
    require(root.children[0].operation == 0 && root.children[1].operation == 1, "wrapper changed payload order");
    require(root.children[2].kind == Region::Choice && root.children[2].children.size() == 2 &&
                root.children[2].originalOwner == 30, "choice lost its arm/owner interpretation");
    require(root.children[3].kind == Region::For && root.children[3].children.size() == 1 &&
                root.children[3].zeroTripPossible && root.children[3].qualifiedCounted,
            "counted loop lost its body, bypass or qualification");
    require(root.children[4].kind == Region::While && root.children[4].children.size() == 2,
            "while-before/after regions were identified");
    require(root.children[5].kind == Region::Sequence && root.children[5].originalOwner == 60 &&
                root.children[5].children.size() == 1, "named enclosing scope was flattened");
    for (std::size_t i = 2; i <= 4; ++i) {
        for (const auto& child : root.children[i].children) {
            require(child.kind == Region::Sequence && child.children.size() == 1 &&
                        child.children.front().kind == Region::Operation,
                    "real structured child boundary was removed or nested wrapper remained");
        }
    }
    fs::normalizeOriginalSequences(root);
    require(root.children.size() == 6, "sequence canonicalization was not idempotent");
    Region deep = leaf(8);
    for (std::size_t i = 0; i < 4096; ++i) { deep = wrap(std::move(deep)); }
    fs::normalizeOriginalSequences(deep);
    require(deep.children.size() == 1 && deep.children[0].kind == Region::Operation && deep.children[0].operation == 8,
            "deep transparent wrapper lost a physical use");
}
int main()
{
    originalWrappers();
    provenance();
    invalidAndOpaque();
    relationComposition();
    unfinishedUse();
    growth();
    std::cout << "D4 checks=" << checks << " concrete-scans=" << executions << " PASS\n";
    return 0;
}
