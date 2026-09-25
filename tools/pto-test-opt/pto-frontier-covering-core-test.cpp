// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "DirectionalCovering.h"
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

namespace fs = mlir::pto::frontiersynch;
namespace detail = fs::detail;
using fs::OriginalCut;
using Status = fs::CoveringBoundary::Status;
using Obstruction = fs::CoveringBoundary::Obstruction;
using Match = detail::CoveringMatch;
static std::size_t checks = 0, executions = 0;
static void check(bool condition, const std::string& what)
{
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        std::exit(1);
    }
}
static fs::Region op(std::size_t id)
{
    fs::Region r;
    r.kind = fs::Region::Operation;
    r.operation = id;
    return r;
}
static fs::Region seq(std::initializer_list<fs::Region> children)
{
    fs::Region r;
    r.children = children;
    return r;
}
static fs::Region choice(std::size_t owner, fs::Region yes, fs::Region no = {})
{
    auto r = seq({std::move(yes), std::move(no)});
    r.kind = fs::Region::Choice;
    r.originalOwner = owner;
    return r;
}
static fs::Region repeat(std::size_t owner, fs::Region body, bool finite = true, bool zero = true)
{
    auto r = seq({std::move(body)});
    r.kind = fs::Region::For;
    r.originalOwner = owner;
    r.qualifiedCounted = finite;
    r.zeroTripPossible = zero;
    return r;
}
struct Effect {
    unsigned engine = 1;
    std::size_t cell = 0;
    bool read = true, write = false;
    Match certainty = Match::Must;
    std::size_t relation = 0;
};
static bool relevant(const Effect& e, const fs::OriginalAccessSelector& s)
{
    return (!s.engine || e.engine == *s.engine) && e.cell == s.cell &&
           ((s.read && e.read) || (s.write && e.write)) &&
           (s.physicalRelation == fs::NoControlId || s.physicalRelation == e.relation);
}
struct Fixture {
    fs::Region body;
    std::vector<Effect> effects;
    fs::OriginalProgramVersion version = fs::OriginalProgramVersion::fresh();
    detail::ControlGraph graph;
    detail::DirectionalCovering service;
    Fixture(fs::Region body, std::vector<Effect> effects, std::set<OriginalCut> illegal = {})
        : body(std::move(body)), effects(std::move(effects)),
          graph(detail::buildControlGraph(this->body, this->effects.size(),
                                         [illegal](const OriginalCut& c) { return !illegal.count(c); })),
          service(this->body, graph, version,
                  [this](std::size_t id, const fs::OriginalAccessSelector& s) {
                      return relevant(this->effects.at(id), s) ? this->effects.at(id).certainty : Match::NoHit;
                  },
                  [](const fs::Region& r) { return r.kind == fs::Region::For && r.qualifiedCounted; })
    {
        graph.version = version;
        check(graph.valid, "valid fixture control: " + graph.reason);
    }
    fs::OriginalIntervalRequest whole() const
    {
        fs::OriginalIntervalRequest q;
        q.selector = {};
        q.selector.read = true;
        q.selector.write = false;
        q.selector.engine = 1;
        q.start = OriginalCut::scope(fs::NoControlId, OriginalCut::Before);
        q.stop = OriginalCut::scope(fs::NoControlId, OriginalCut::After);
        q.occurrence.stopVisit = fs::OriginalOccurrenceContext::StopVisit::FirstReach;
        return q;
    }
    fs::OriginalInterval prepare(fs::OriginalIntervalRequest q) const
    {
        const auto p = detail::prepareOriginalInterval(graph, version, std::move(q));
        check(p.valid, "prepare test interval: " + p.reason);
        return p.interval;
    }
    fs::CoveringBoundary query(fs::OriginalIntervalRequest q, bool source = true) const
    {
        return service.query(prepare(std::move(q)), source ? fs::CoveringDirection::Source : fs::CoveringDirection::Target);
    }
};
static void expectCut(const fs::CoveringBoundary& answer, OriginalCut expected)
{
    check(answer.status == Status::Covering, "positive covering status: " + answer.reason);
    check(answer.cut && *answer.cut == expected, "prescribed directional cut");
    check(answer.referenceCoverage && answer.guardIsTrue && answer.exactlyOneVisitPerInterval,
          "reference coverage and unconditional per-interval multiplicity");
    check(answer.requiresWholePrefixCheck, "no whole-prefix ordering or completion claim");
}
static void examples()
{
    Effect unrelated;
    unrelated.cell = 1;
    // A; compute(g); if(g) B; U. The service does not even request a last-read
    // guard, so a value computed between A and B cannot disable this answer.
    Fixture f(seq({op(0), choice(10, seq({op(1)})), op(2)}), {{}, {}, unrelated});
    auto source = f.query(f.whole());
    expectCut(source, OriginalCut::scope(10, OriginalCut::After));
    check(source.enclosedWork.items.size() == 2 && source.trimmedWork.items.size() == 1 &&
          source.trimmedWork.items[0].operation == 2, "source trims U, keeps entire conditional");
    check(!source.mayExecuteWithoutAccess, "mandatory A excludes empty access paths");
    expectCut(f.query(f.whole(), false), {0, OriginalCut::Before});
    // U; if(g) A; B is the dual, not a source/target pair menu.
    Fixture dual(seq({op(0), choice(11, seq({op(1)})), op(2)}), {unrelated, {}, {}});
    auto target = dual.query(dual.whole(), false);
    expectCut(target, OriginalCut::scope(11, OriginalCut::Before));
    check(target.trimmedWork.items.size() == 1 && target.trimmedWork.items[0].operation == 0,
          "target trims unrelated prefix");
    expectCut(dual.query(dual.whole()), {2, OriginalCut::After});
    // No synthetic return on a no-reader path: the extra endpoint execution is
    // explicitly visible in both independent answers.
    Fixture optional(seq({choice(12, seq({op(0), op(1)}))}), {{}, unrelated});
    source = optional.query(optional.whole());
    target = optional.query(optional.whole(), false);
    expectCut(source, OriginalCut::scope(12, OriginalCut::After));
    expectCut(target, OriginalCut::scope(12, OriginalCut::Before));
    check(source.mayExecuteWithoutAccess && target.mayExecuteWithoutAccess, "optional empty paths retained");
    check(source.enclosedWork.items[0].kind == fs::Region::Choice,
          "unrelated work INSIDE conditional remains visible rather than trimmed");
    auto empty = optional.whole();
    empty.selector.cell = 9;
    for (bool s : {false, true}) {
        const auto a = optional.query(empty, s);
        check(a.status == Status::NoHit && !a.cut && !a.referenceCoverage, "all-no-hit has no endpoint");
    }
    Effect unknown;
    unknown.certainty = Match::Unknown;
    Fixture uncertain(seq({op(0), op(1), op(2)}), {{}, unknown, unrelated});
    source = uncertain.query(uncertain.whole());
    expectCut(source, {1, OriginalCut::After});
    check(source.conservativeEffects && source.mayAccesses == std::vector<std::size_t>({0, 1}),
          "unknown effect blocks suffix exclusion");
    Fixture uncertainOnly(seq({op(0)}), {unknown});
    check(uncertainOnly.query(uncertainOnly.whole()).mayExecuteWithoutAccess,
          "uncertain physical incidence does not prove participation");
    auto symbolic = uncertain.whole();
    symbolic.selector.qualification = 71;
    symbolic.selector.predicateDependencies = {19};
    source = uncertain.query(symbolic);
    check(source.conservativeSelection && source.mayExecuteWithoutAccess,
          "opaque narrower selector retains possible no-family paths");
    // Unavailable internal phase source cut cannot cause widening. Target
    // before that phase remains independently usable.
    Fixture illegal(seq({op(0), op(1)}), {{}, unrelated}, {{0, OriginalCut::After}});
    source = illegal.query(illegal.whole());
    check(source.status == Status::Unknown && source.obstruction == Obstruction::CutUnavailable && !source.cut,
          "unavailable internal phase cut rejected without another boundary search");
    expectCut(illegal.query(illegal.whole(), false), {0, OriginalCut::Before});
    Fixture wrapped(seq({seq({seq({op(0)}), choice(10, seq({seq({op(1)})}))}), seq({op(2)})}),
                    {{}, {}, unrelated});
    expectCut(wrapped.query(wrapped.whole()), OriginalCut::scope(10, OriginalCut::After));
    expectCut(wrapped.query(wrapped.whole(), false), {0, OriginalCut::Before});
    check(wrapped.query(wrapped.whole()).enclosedWork.items.size() == 2, "transparent wrappers do not add scope");
    auto partial = wrapped.whole();
    partial.start = {0, OriginalCut::After};
    partial.stop = {2, OriginalCut::Before};
    source = wrapped.query(partial);
    expectCut(source, OriginalCut::scope(10, OriginalCut::After));
    check(source.mayExecuteWithoutAccess, "cut-delimited subinterval does not borrow A's participation");
    // Different reader engine is a different query, never a merged maximum.
    Effect otherEngine;
    otherEngine.engine = 2;
    Fixture engines(seq({op(0), op(1)}), {{}, otherEngine});
    expectCut(engines.query(engines.whole()), {0, OriginalCut::After});
    auto engine2 = engines.whole();
    engine2.selector.engine = 2;
    expectCut(engines.query(engine2), {1, OriginalCut::After});
    engine2.selector.engine.reset();
    check(engines.query(engine2).obstruction == Obstruction::EngineRequired, "one engine required for positive result");
}
static void loopsAndIntervals()
{
    Effect unrelated;
    unrelated.cell = 1;
    // Even varying participation can use a finite loop exit; no D3 first/last
    // participating-coordinate premise is imposed on the covering rule.
    Fixture f(seq({op(0), repeat(20, seq({choice(21, seq({op(1)})), op(2)})), op(3)}),
              {unrelated, {}, unrelated, unrelated});
    auto q = f.whole();
    auto a = f.query(q);
    expectCut(a, OriginalCut::scope(20, OriginalCut::After));
    expectCut(f.query(q, false), OriginalCut::scope(20, OriginalCut::Before));
    check(a.mayExecuteWithoutAccess && !a.mayRepeatInInvocation && a.finiteLoopOwners == std::vector<std::size_t>{20},
          "external finite-loop boundary executes once even on zero trips");
    check(a.enclosedWork.items.back().kind == fs::Region::For, "whole loop work retained");
    q.start = OriginalCut::childBoundary(20, 0, OriginalCut::Before);
    q.stop = OriginalCut::childBoundary(20, 0, OriginalCut::After);
    a = f.query(q);
    expectCut(a, OriginalCut::scope(21, OriginalCut::After));
    check(a.mayRepeatInInvocation && a.visitEntry == q.start && a.visitExit == q.stop,
          "body boundary is once per qualified body interval, not once per invocation");
    q.occurrence.stopVisit = fs::OriginalOccurrenceContext::StopVisit::Unqualified;
    check(f.query(q).obstruction == Obstruction::OccurrenceUnqualified, "unqualified repeated cut cannot certify multiplicity");
    q.occurrence.stopVisit = fs::OriginalOccurrenceContext::StopVisit::AfterBackedge;
    q.occurrence.backedgeOwner = 20;
    check(f.query(q).status == Status::Unknown, "no guessed across-backedge covering correspondence");
    Fixture unqualified(seq({repeat(30, seq({op(0)}), false)}), {{}});
    a = unqualified.query(unqualified.whole());
    check(a.status == Status::Unknown && a.obstruction == Obstruction::RepetitionUnqualified && !a.cut &&
          a.mayAccesses == std::vector<std::size_t>{0} && a.loopOwners == std::vector<std::size_t>{30},
          "unqualified repeat retains may facts and failed premise");
    auto nohit = unqualified.whole();
    nohit.selector.cell = 2;
    check(unqualified.query(nohit).status == Status::NoHit, "all-no-hit needs no invented loop-exit endpoint");
    fs::Region whileLoop;
    whileLoop.kind = fs::Region::While;
    whileLoop.originalOwner = 31;
    whileLoop.children = {seq({op(0)}), seq({op(1)})};
    Fixture w(seq({whileLoop}), {{}, {}});
    check(w.query(w.whole()).obstruction == Obstruction::RepetitionUnqualified, "unqualified while boundary withheld");
    auto beforeVisit = w.whole();
    beforeVisit.start = OriginalCut::childBoundary(31, 0, OriginalCut::Before);
    beforeVisit.stop = OriginalCut::childBoundary(31, 0, OriginalCut::After);
    auto beforeBoundary = w.query(beforeVisit);
    expectCut(beforeBoundary, {0, OriginalCut::After});
    check(beforeBoundary.mayRepeatInInvocation,
          "one while-before visit is not a finite-exit or whole-loop multiplicity proof");
    // No-hit repetition is not a finiteness proof for an otherwise positive interval.
    Fixture mixed(seq({op(0), repeat(32, seq({op(1)}), false)}), {{}, unrelated});
    check(mixed.query(mixed.whole()).obstruction == Obstruction::RepetitionUnqualified,
          "unqualified no-hit suffix cannot qualify the finite interval");
    // Sibling arm cuts are different visits. An arbitrary reachability walk is
    // not the supplied structural SESE contract; neither is a child exit equal
    // to the enclosing next overwrite.
    Fixture branches(seq({choice(40, seq({op(0)}), seq({op(1)})), op(2)}), {{}, {}, {}});
    q = branches.whole();
    q.start = {0, OriginalCut::Before};
    q.stop = {2, OriginalCut::Before};
    check(branches.query(q).obstruction == Obstruction::NotSingleEntrySingleExit,
          "branch-local entry crossing a join needs a supplied occurrence/SESE relation");
    q = branches.whole();
    q.start = OriginalCut::scope(40, OriginalCut::After);
    q.stop = {2, OriginalCut::After};
    q.includeStoppingAccess = true;
    expectCut(branches.query(q), {2, OriginalCut::After});
    // Explicit stop-access convention is independent of the Before/After cut.
    Fixture single(seq({op(0), op(1), op(2)}), {{}, {}, {}});
    for (auto side : {OriginalCut::Before, OriginalCut::After}) {
        for (bool include : {false, true}) {
            q = single.whole();
            q.start = {1, OriginalCut::Before};
            q.stop = {1, side};
            q.includeStoppingAccess = include;
            for (bool source : {false, true}) {
                a = single.query(q, source);
                if (include) {
                    expectCut(a, {1, source ? OriginalCut::After : OriginalCut::Before});
                } else {
                    check(a.status == Status::NoHit && !a.cut, "stopping access excluded for either cut side");
                }
            }
        }
    }
    q = single.whole();
    q.start = {0, OriginalCut::Before};
    q.stop = {1, OriginalCut::After};
    a = single.query(q);
    expectCut(a, {0, OriginalCut::After});
    check(a.trimmedWork.items.size() == 1 && a.trimmedWork.items[0].operation == 1,
          "excluded stopping access still visible as enclosed original work before After stop");
    q.start = {1, OriginalCut::After};
    check(single.query(q).status == Status::NoHit, "same After gap is empty, not a backedge");
    q.stop = {1, OriginalCut::Before};
    q.includeStoppingAccess = true;
    check(single.query(q).status == Status::Unknown,
          "stop inclusion cannot change an earlier stop into a same-visit endpoint");
    q.start = {1, OriginalCut::Before};
    q.stop = {0, OriginalCut::After};
    q.includeStoppingAccess = false;
    check(single.query(q).status == Status::Unknown,
          "distinct cuts at one sequence gap retain their original order");

}
static void cacheAndFormation()
{
    Fixture f(seq({op(0), choice(10, seq({op(1)}))}), {{}, {}});
    auto q = f.whole();
    const auto interval = f.prepare(q);
    const auto first = f.service.query(interval, fs::CoveringDirection::Source);
    auto stats = f.service.stats();
    f.service.query(interval, fs::CoveringDirection::Source);
    check(f.service.stats().cacheHits == stats.cacheHits + 1, "same full key hits cache");
    check(f.service.stats().summaryEvaluations == stats.summaryEvaluations, "cached answer does not reform summaries");
    check(f.service.stats().materializedReferences > stats.materializedReferences, "copied output charged separately");
    stats = f.service.stats();
    f.service.query(interval, fs::CoveringDirection::Target);
    check(f.service.stats().cacheHits == stats.cacheHits, "direction is part of the answer key");
    auto different = q;
    different.occurrence.incomingInterface = 70;
    auto answer = f.query(different);
    check(answer.cases.incoming && answer.interval != interval, "incoming identity retained without fresh-state claim");
    const std::vector<fs::OriginalIntervalRequest> keys = [&] {
        std::vector<fs::OriginalIntervalRequest> out;
        auto add = [&](auto change) { auto x = q; change(x); out.push_back(std::move(x)); };
        add([](auto& x) { x.selector.engine = 2; });
        add([](auto& x) { x.selector.cell = 1; });
        add([](auto& x) { x.selector.write = true; });
        add([](auto& x) { x.selector.physicalRelation = 3; });
        add([](auto& x) { x.selector.qualification = 5; });
        add([](auto& x) { x.selector.predicateDependencies = {9}; });
        add([](auto& x) { x.occurrence.qualification = 7; });
        add([](auto& x) { x.occurrence.source = 0; });
        add([](auto& x) { x.occurrence.target = 1; });
        add([](auto& x) { x.continuationOwner = fs::NoControlId; });
        return out;
    }();
    for (const auto& key : keys) {
        stats = f.service.stats();
        f.query(key);
        check(f.service.stats().cacheHits == stats.cacheHits, "distinct semantic fields cannot reuse answer");
    }
    auto stale = interval;
    stale.query.version.revision++;
    check(f.service.query(stale, fs::CoveringDirection::Source).obstruction == Obstruction::InvalidInterval,
          "stale version cannot retrieve positive cache entry");
    stale = interval;
    stale.query.version = fs::OriginalProgramVersion::fresh();
    check(f.service.query(stale, fs::CoveringDirection::Source).status == Status::Unknown, "foreign snapshot rejected");
    stale = interval;
    stale.owner = 10;
    check(f.service.query(stale, fs::CoveringDirection::Source).status == Status::Unknown, "altered derived owner rejected");
    for (std::size_t m : {64u, 1024u, 4096u}) {
        fs::Region body;
        std::vector<Effect> effects(m);
        for (std::size_t i = 0; i < m; ++i) {
            body.children.push_back(choice(100 + i, seq({op(i)})));
        }
        Fixture stress(std::move(body), std::move(effects));
        auto s = stress.query(stress.whole());
        auto t = stress.query(stress.whole(), false);
        expectCut(s, OriginalCut::scope(100 + m - 1, OriginalCut::After));
        expectCut(t, OriginalCut::scope(100, OriginalCut::Before));
        check(stress.service.stats().summaryEvaluations == 2 * m,
              "one summary per relevant node/selector, no reader-set valuation product");
        check(stress.service.stats().effectQueries <= 5 * m && stress.service.stats().intervalNodes <= 6 * m,
              "formation and query inspections linear on independent-reader stress case");
        check(s.mayAccesses.size() == m && s.enclosedWork.items.size() == m,
              "charged static references retain every optional reader");
        std::cout << "formation m=" << m << " summaries=" << stress.service.stats().summaryEvaluations
                  << " effect-queries=" << stress.service.stats().effectQueries
                  << " output-refs=" << stress.service.stats().materializedReferences << '\n';
    }
}

// Independent reference oracle: execute the original syntax, preserving cut
// VISITS and payload occurrences. It shares neither the production lane index,
// region summaries, control-graph walk nor trimming procedure. Enumeration here
// is confined to small tests, not used in the production query.
struct Event {
    bool payload = false;
    OriginalCut cut;
    std::size_t operation = fs::NoControlId;
};
struct Execution {
    std::map<std::size_t, bool> choices;
    std::map<std::size_t, unsigned> trips;
    std::map<std::size_t, std::vector<bool>> choicesByVisit;
    bool uncertainPresent = true, qualifiedFamilyPresent = true;
};
static void execute(const fs::Region& r, const Execution& input, std::vector<Event>& trace,
                    std::map<std::size_t, std::size_t>& visits)
{
    auto cut = [&](OriginalCut c) { trace.push_back({false, c, fs::NoControlId}); };
    if (r.kind == fs::Region::Operation) {
        cut({r.operation, OriginalCut::Before});
        trace.push_back({true, {}, r.operation});
        cut({r.operation, OriginalCut::After});
        return;
    }
    if (r.kind == fs::Region::Sequence) {
        if (r.originalOwner != fs::NoControlId) {
            cut(OriginalCut::scope(r.originalOwner, OriginalCut::Before));
        }
        for (const auto& child : r.children) {
            execute(child, input, trace, visits);
        }
        if (r.originalOwner != fs::NoControlId) {
            cut(OriginalCut::scope(r.originalOwner, OriginalCut::After));
        }
        return;
    }
    cut(OriginalCut::scope(r.originalOwner, OriginalCut::Before));
    auto child = [&](std::size_t index) {
        cut(OriginalCut::childBoundary(r.originalOwner, index, OriginalCut::Before));
        execute(r.children[index], input, trace, visits);
        cut(OriginalCut::childBoundary(r.originalOwner, index, OriginalCut::After));
    };
    if (r.kind == fs::Region::Choice) {
        const auto varying = input.choicesByVisit.find(r.originalOwner);
        const auto visit = visits[r.originalOwner]++;
        const bool yes = varying == input.choicesByVisit.end() ? input.choices.at(r.originalOwner) :
                                                               varying->second.at(visit);
        child(yes ? 0 : 1);
    } else if (r.kind == fs::Region::For) {
        for (unsigned i = 0; i < input.trips.at(r.originalOwner); ++i) {
            child(0);
        }
    } else {
        for (unsigned i = 0; i <= input.trips.at(r.originalOwner); ++i) {
            child(0);
            if (i != input.trips.at(r.originalOwner)) {
                child(1);
            }
        }
    }
    cut(OriginalCut::scope(r.originalOwner, OriginalCut::After));
}
static void verify(const Fixture& f, const fs::OriginalIntervalRequest& query,
                   const fs::CoveringBoundary& answer, const Execution& input)
{
    ++executions;
    if (answer.status == Status::Unknown) {
        check(!answer.cut && !answer.referenceCoverage, "unknown answer exposes no certified endpoint");
        return;
    }
    std::vector<Event> trace{{false, OriginalCut::scope(fs::NoControlId, OriginalCut::Before), fs::NoControlId}};
    std::map<std::size_t, std::size_t> visits;
    execute(f.body, input, trace, visits);
    trace.push_back({false, OriginalCut::scope(fs::NoControlId, OriginalCut::After), fs::NoControlId});
    for (std::size_t start = 0; start < trace.size(); ++start) {
        if (trace[start].payload || trace[start].cut != query.start) {
            continue;
        }
        auto stop = start;
        while (stop < trace.size() && (trace[stop].payload || trace[stop].cut != query.stop)) {
            ++stop;
        }
        check(stop < trace.size(), "finite same-lane execution reaches specified stopping cut");
        auto end = stop;
        if (query.stop.kind == OriginalCut::Kind::Payload && query.includeStoppingAccess &&
            query.stop.side == OriginalCut::Before) {
            while (end < trace.size() && (trace[end].payload ||
                   trace[end].cut != OriginalCut{query.stop.operation, OriginalCut::After})) {
                ++end;
            }
            check(end < trace.size(), "included stopping access has its own after-cut visit");
        }
        std::vector<std::size_t> matches, boundaries;
        for (auto i = start; i <= end; ++i) {
            if (!trace[i].payload) {
                if (answer.cut && trace[i].cut == *answer.cut) {
                    boundaries.push_back(i);
                }
                continue;
            }
            const auto id = trace[i].operation;
            const auto& e = f.effects[id];
            if (query.stop.kind == OriginalCut::Kind::Payload && id == query.stop.operation &&
                !query.includeStoppingAccess) {
                continue;
            }
            const bool symbolic = query.selector.qualification != fs::NoControlId ||
                                  !query.selector.predicateDependencies.empty() ||
                                  query.occurrence.qualification != fs::NoControlId;
            if (relevant(e, query.selector) && e.certainty != Match::NoHit &&
                (e.certainty == Match::Must || input.uncertainPresent) &&
                (!symbolic || input.qualifiedFamilyPresent)) {
                matches.push_back(i);
            }
        }
        if (answer.status == Status::NoHit) {
            check(matches.empty() && !answer.cut, "NoHit agrees with concrete execution");
            continue;
        }
        check(boundaries.size() == 1, "one actual boundary VISIT per represented interval visit");
        check(!matches.empty() || answer.mayExecuteWithoutAccess, "extra no-access execution is visible");
        for (auto access : matches) {
            check(answer.direction == fs::CoveringDirection::Source ? access < boundaries[0] : boundaries[0] < access,
                  "every concrete required access is covered in the correct reference direction");
            check(std::find(answer.mayAccesses.begin(), answer.mayAccesses.end(), trace[access].operation) !=
                      answer.mayAccesses.end(), "all concrete matching incidences retained");
        }
    }
}
static void finiteReferenceChecks()
{
    {
        Fixture varying(seq({repeat(300, seq({choice(301, seq({op(0)}))}))}), {{}});
        const auto q = varying.whole();
        for (bool source : {false, true}) {
            const auto answer = varying.query(q, source);
            for (unsigned trips = 0; trips <= 5; ++trips) {
                for (unsigned mask = 0; mask < (1u << trips); ++mask) {
                    Execution execution;
                    execution.trips[300] = trips;
                    auto& guards = execution.choicesByVisit[301];
                    for (unsigned visit = 0; visit < trips; ++visit) {
                        guards.push_back((mask >> visit) & 1);
                    }
                    verify(varying, q, answer, execution);
                }
            }
        }
    }

    // A varying reader can appear in an inner conditional; a qualified loop
    // covers all its concrete trips without a last-participating-iteration test.
    for (unsigned layout = 0; layout < 16; ++layout) {
        auto body = seq({op(0), choice(100, seq({op(1)}), seq({op(2)})),
                         repeat(101, seq({choice(102, seq({op(3)})), op(4)})), op(5)});
        if (layout & 1) {
            body = seq({seq({body})});
        }
        std::vector<Effect> effects(6);
        for (std::size_t i = 0; i < effects.size(); ++i) {
            effects[i].cell = ((layout >> (i % 4)) & 1) ? 0 : 1;
            effects[i].engine = i == 2 ? 2 : 1;
            effects[i].write = i == 4;
            effects[i].read = i != 4;
        }
        effects[3].certainty = Match::Unknown;
        Fixture f(std::move(body), std::move(effects));
        std::vector<fs::OriginalIntervalRequest> queries{f.whole()};
        auto q = f.whole();
        q.start = {0, OriginalCut::After};
        q.stop = {5, OriginalCut::Before};
        queries.push_back(q);
        q = f.whole();
        q.start = OriginalCut::scope(100, OriginalCut::Before);
        q.stop = OriginalCut::scope(101, OriginalCut::After);
        queries.push_back(q);
        q = f.whole();
        q.start = OriginalCut::childBoundary(101, 0, OriginalCut::Before);
        q.stop = OriginalCut::childBoundary(101, 0, OriginalCut::After);
        queries.push_back(q);
        for (auto side : {OriginalCut::Before, OriginalCut::After}) {
            for (bool include : {false, true}) {
                q = f.whole();
                q.stop = {5, side};
                q.includeStoppingAccess = include;
                queries.push_back(q);
            }
        }
        q = f.whole();
        q.selector.qualification = 88;
        queries.push_back(q);
        for (const auto& base : queries) {
            for (unsigned engine : {1u, 2u}) {
                for (unsigned mode : {0u, 1u, 2u}) {
                    q = base;
                    q.selector.engine = engine;
                    q.selector.read = mode != 1;
                    q.selector.write = mode != 0;
                    for (bool source : {false, true}) {
                        const auto answer = f.query(q, source);
                        for (unsigned valuation = 0; valuation < 16; ++valuation) {
                            for (unsigned trips = 0; trips <= 3; ++trips) {
                                Execution input;
                                input.choices = {{100, bool(valuation & 1)}, {102, bool(valuation & 2)}};
                                input.trips = {{101, trips}};
                                input.uncertainPresent = bool(valuation & 4);
                                input.qualifiedFamilyPresent = bool(valuation & 8);
                                verify(f, q, answer, input);
                            }
                        }
                    }
                }
            }
        }
    }
}
int main()
{
    examples();
    loopsAndIntervals();
    cacheAndFormation();
    finiteReferenceChecks();
    std::cout << "directional covering core: " << checks << " checks; " << executions
              << " independent finite-execution comparisons; passed\n";
}
