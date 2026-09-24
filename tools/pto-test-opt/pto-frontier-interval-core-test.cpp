// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
// Tests the actual production records, cut graph and interval traversal without
// MLIR stubs. Native importer/cut resolution is exercised by the separate test.
#include "OriginalIntervals.h"
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>

namespace fs = mlir::pto::frontiersynch;
namespace detail = fs::detail;
using fs::OriginalCut;
using fs::OriginalIntervalRequest;
using fs::OriginalOccurrenceContext;
using fs::Region;
using StopVisit = OriginalOccurrenceContext::StopVisit;
static std::size_t checks = 0, oracleQueries = 0;
static void check(bool condition, const std::string& message)
{
    ++checks;
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
static Region op(std::size_t id)
{
    Region r;
    r.kind = Region::Operation;
    r.operation = id;
    return r;
}
static Region seq(std::vector<Region> children)
{
    Region r;
    r.children = std::move(children);
    return r;
}
static Region control(Region::Kind kind, std::size_t owner, std::vector<Region> children)
{
    auto r = seq(std::move(children));
    r.kind = kind;
    r.originalOwner = owner;
    return r;
}
static detail::ControlGraph graph(const Region& r, std::size_t count, const std::set<OriginalCut>& unavailable = {})
{
    return detail::buildControlGraph(r, count, [&](const OriginalCut& cut) { return !unavailable.count(cut); });
}
static fs::OriginalInterval prepare(
    const detail::ControlGraph& g, const fs::OriginalProgramVersion& version, OriginalIntervalRequest q)
{
    const auto result = detail::prepareOriginalInterval(g, version, std::move(q));
    check(result.valid, "interval preparation: " + result.reason);
    return result.interval;
}
static OriginalIntervalRequest query(OriginalCut start, OriginalCut stop, StopVisit visit = StopVisit::FirstReach)
{
    OriginalIntervalRequest q;
    q.start = start;
    q.stop = stop;
    q.occurrence.stopVisit = visit;
    return q;
}
static detail::OriginalIntervalWalk walk(const detail::ControlGraph& g, const fs::OriginalInterval& q)
{
    return detail::walkOriginalInterval(g, q, [](std::size_t) { return true; });
}

// Independent finite execution interpreter. It uses the Region grammar, not
// the graph or the query implementation; dynamic cut and backedge visits are
// explicit tokens. The graph's conservative counted-loop language admits zero
// trips, even for a qualified/nonempty loop, so the reference does too.
struct Token {
    enum Kind { Cut, Payload, Backedge } kind = Cut;
    OriginalCut cut;
    std::size_t operation = fs::NoControlId;
};
using Trace = std::vector<Token>;
using Traces = std::vector<Trace>;
static void append(Trace& to, const Trace& from) { to.insert(to.end(), from.begin(), from.end()); }
static Traces product(const Traces& left, const Traces& right)
{
    Traces out;
    for (const auto& a : left) {
        for (const auto& b : right) {
            auto joined = a;
            append(joined, b);
            out.push_back(std::move(joined));
        }
    }
    return out;
}
static Traces surround(Traces traces, OriginalCut before, OriginalCut after)
{
    for (auto& trace : traces) {
        trace.insert(trace.begin(), {Token::Cut, before, fs::NoControlId});
        trace.push_back({Token::Cut, after, fs::NoControlId});
    }
    return traces;
}
static Traces execute(const Region& r, unsigned maxIterations)
{
    Traces traces;
    if (r.kind == Region::Operation) {
        return {
            {{Token::Cut, {r.operation, OriginalCut::Before}, fs::NoControlId},
             {Token::Payload, {}, r.operation},
             {Token::Cut, {r.operation, OriginalCut::After}, fs::NoControlId}}};
    }
    auto child = [&](std::size_t i) {
        return surround(
            execute(r.children[i], maxIterations), OriginalCut::childBoundary(r.originalOwner, i, OriginalCut::Before),
            OriginalCut::childBoundary(r.originalOwner, i, OriginalCut::After));
    };
    if (r.kind == Region::Sequence) {
        traces = {{}};
        for (const auto& part : r.children) {
            traces = product(traces, execute(part, maxIterations));
        }
    } else if (r.kind == Region::Choice) {
        traces = child(0);
        const auto no = child(1);
        traces.insert(traces.end(), no.begin(), no.end());
    } else if (r.kind == Region::For) {
        const auto body = child(0);
        Traces repeated{{}};
        traces = repeated;
        for (unsigned i = 0; i < maxIterations; ++i) {
            repeated = product(repeated, body);
            for (auto& trace : repeated) {
                trace.push_back({Token::Backedge, {}, r.originalOwner});
            }
            traces.insert(traces.end(), repeated.begin(), repeated.end());
        }
    } else if (r.kind == Region::While) {
        const auto before = child(0), after = child(1);
        Traces prefix{{}};
        for (unsigned i = 0; i <= maxIterations; ++i) {
            const auto exitPaths = product(prefix, before);
            traces.insert(traces.end(), exitPaths.begin(), exitPaths.end());
            prefix = product(exitPaths, after);
            for (auto& trace : prefix) {
                trace.push_back({Token::Backedge, {}, r.originalOwner});
            }
        }
    }
    if (r.originalOwner != fs::NoControlId) {
        traces = surround(
            std::move(traces), OriginalCut::scope(r.originalOwner, OriginalCut::Before),
            OriginalCut::scope(r.originalOwner, OriginalCut::After));
    }
    return traces;
}
static std::set<std::size_t> referenceInterval(
    const Traces& traces, const fs::OriginalInterval& interval,
    detail::IntervalSelection selection = detail::IntervalSelection::All)
{
    const auto& q = interval.query;
    const auto ownerExit = OriginalCut::scope(interval.owner, OriginalCut::After);
    std::set<std::size_t> all;
    for (const auto& trace : traces) {
        for (std::size_t start = 0; start < trace.size(); ++start) {
            if (trace[start].kind != Token::Cut || trace[start].cut != q.start) {
                continue;
            }
            bool armed = q.occurrence.stopVisit != StopVisit::AfterBackedge;
            std::vector<std::size_t> hits;
            for (std::size_t i = start; i < trace.size(); ++i) {
                const auto& token = trace[i];
                if (token.kind == Token::Backedge && token.operation == q.occurrence.backedgeOwner) {
                    armed = true;
                }
                if (token.kind == Token::Payload) {
                    hits.push_back(token.operation);
                }
                if (token.kind == Token::Cut && armed && token.cut == q.stop) {
                    if (q.stop.kind == OriginalCut::Kind::Payload) {
                        if (q.stop.side == OriginalCut::Before && q.includeStoppingAccess) {
                            hits.push_back(q.stop.operation);
                        } else if (
                            q.stop.side == OriginalCut::After && !q.includeStoppingAccess && i != start &&
                            !hits.empty() && hits.back() == q.stop.operation) {
                            hits.pop_back();
                        }
                    }
                    break;
                }
                if (token.kind == Token::Cut && token.cut == ownerExit) {
                    break;
                }
            }
            if (!hits.empty() && selection == detail::IntervalSelection::First) {
                all.insert(hits.front());
            } else if (!hits.empty() && selection == detail::IntervalSelection::Last) {
                all.insert(hits.back());
            } else {
                all.insert(hits.begin(), hits.end());
            }
        }
    }
    return all;
}
static void compareOracle(const Region& body, std::size_t count, unsigned trips)
{
    const auto g = graph(body, count);
    check(g.valid, "valid original graph");
    const auto version = fs::OriginalProgramVersion::fresh();
    auto traces = surround(
        execute(body, trips), OriginalCut::scope(fs::NoControlId, OriginalCut::Before),
        OriginalCut::scope(fs::NoControlId, OriginalCut::After));
    // All cut pairs, all stopping-access conventions, and every supported
    // stop-visit interpretation. Each starts from all dynamic visits of its cut.
    for (const auto& [start, startNode] : g.cuts) {
        (void)startNode;
        for (const auto& [stop, stopNode] : g.cuts) {
            (void)stopNode;
            for (bool include : {false, true}) {
                if (include && stop.kind != OriginalCut::Kind::Payload) {
                    continue;
                }
                std::vector<std::size_t> loops{fs::NoControlId};
                for (const auto& [owner, entry] : g.loopEntries) {
                    (void)entry;
                    loops.push_back(owner);
                }
                for (auto loop : loops) {
                    auto q =
                        query(start, stop, loop == fs::NoControlId ? StopVisit::FirstReach : StopVisit::AfterBackedge);
                    q.includeStoppingAccess = include;
                    q.continuationOwner = fs::NoControlId;
                    q.occurrence.backedgeOwner = loop;
                    const auto interval = prepare(g, version, q);
                    for (const auto selection :
                         {detail::IntervalSelection::All, detail::IntervalSelection::First,
                          detail::IntervalSelection::Last}) {
                        const auto answer =
                            detail::walkOriginalInterval(g, interval, [](std::size_t) { return true; }, selection);
                        check(answer.complete, "complete represented interval walk");
                        const auto expected = referenceInterval(traces, interval, selection);
                        const std::set<std::size_t> actual(answer.operations.begin(), answer.operations.end());
                        if (actual != expected) {
                            std::cerr << "oracle mismatch start=" << int(start.kind) << ':' << start.operation << ':'
                                      << int(start.side) << ':' << start.owner << ':' << start.child
                                      << " stop=" << int(stop.kind) << ':' << stop.operation << ':' << int(stop.side)
                                      << ':' << stop.owner << ':' << stop.child << " include=" << include
                                      << " loop=" << loop << " selection=" << int(selection) << '\n';
                            std::cerr << "actual:";
                            for (auto op : actual) {
                                std::cerr << ' ' << op;
                            }
                            std::cerr << " expected:";
                            for (auto op : expected) {
                                std::cerr << ' ' << op;
                            }
                            std::cerr << '\n';
                        }
                        check(actual == expected, "interval agrees with independent finite reference executions");
                        ++oracleQueries;
                    }
                }
            }
        }
    }
}

static void recordsAndOwnership()
{
    const auto nested =
        seq({control(Region::For, 10, {seq({op(0), control(Region::For, 20, {seq({op(1), op(2)})}), op(3)})}), op(4)});
    const auto g = graph(nested, 5);
    const auto v = fs::OriginalProgramVersion::fresh();
    auto q = query({1, OriginalCut::After}, {2, OriginalCut::Before});
    q.occurrence.source = 1;
    q.occurrence.target = 2;
    const auto local = prepare(g, v, q);
    check(local.owner == 20, "least owner includes related uses");
    q.continuationOwner = 10;
    const auto enclosing = prepare(g, v, q);
    check(enclosing.owner == 10, "declared continuation lifts owner beyond lexical child");
    q.continuationOwner = fs::NoControlId;
    const auto function = prepare(g, v, q);
    check(function.owner == fs::NoControlId, "function horizon is explicit, not unspecified");
    check(local != enclosing && enclosing != function, "horizons have distinct identity");
    q.continuationOwner.reset();
    q.stop = OriginalCut::scope(20, OriginalCut::After);
    const auto exit = prepare(g, v, q);
    check(exit.owner == 20, "child exit remains a child interval");
    q.stop = {3, OriginalCut::Before};
    q.occurrence.target = 3;
    const auto overwrite = prepare(g, v, q);
    check(overwrite.owner == 10, "outside overwrite keeps enclosing physical owner");
    check(exit != overwrite, "child exit and enclosing overwrite are different stops");
    auto input = query(OriginalCut::scope(fs::NoControlId, OriginalCut::Before), {0, OriginalCut::Before});
    input.occurrence.incomingInterface = 7;
    const auto incoming = walk(g, prepare(g, v, input));
    check(incoming.cases.incoming && incoming.cases.childEntry, "incoming and child entry survive traversal");
    const auto loopEntry = g.cuts.at(OriginalCut::scope(10, OriginalCut::Before));
    const auto bodyEntry = g.cuts.at(OriginalCut::childBoundary(10, 0, OriginalCut::Before));
    check(
        loopEntry != bodyEntry && !g.repeated[loopEntry] && g.repeated[bodyEntry],
        "external entry is not a repeatedly visited body cut");
}
static void cacheKeys()
{
    const auto g = graph(seq({op(0), op(1)}), 2);
    auto base =
        prepare(g, fs::OriginalProgramVersion::fresh(), query({0, OriginalCut::Before}, {1, OriginalCut::After}));
    std::set<fs::OriginalInterval> keys{base};
    auto add = [&](auto change) {
        auto q = base;
        change(q);
        check(keys.insert(q).second, "complete key distinguishes changed semantic field");
    };
    add([](auto& q) { q.owner = 12; });
    add([](auto& q) { q.query.version = fs::OriginalProgramVersion::fresh(); });
    add([](auto& q) { ++q.query.version.revision; });
    add([](auto& q) { q.query.selector.cell = 4; });
    add([](auto& q) { q.query.selector.read = false; });
    add([](auto& q) { q.query.selector.write = false; });
    add([](auto& q) { q.query.selector.engine = 2; });
    add([](auto& q) { q.query.selector.physicalRelation = 3; });
    add([](auto& q) { q.query.selector.qualification = 4; });
    add([](auto& q) { q.query.selector.predicateDependencies = {5}; });
    add([](auto& q) { q.query.occurrence.source = 0; });
    add([](auto& q) { q.query.occurrence.target = 1; });
    add([](auto& q) { q.query.occurrence.stopVisit = StopVisit::Unqualified; });
    add([](auto& q) { q.query.occurrence.backedgeOwner = 7; });
    add([](auto& q) { q.query.occurrence.incomingInterface = 8; });
    add([](auto& q) { q.query.occurrence.qualification = 9; });
    add([](auto& q) { q.query.start.side = OriginalCut::After; });
    add([](auto& q) { q.query.stop.side = OriginalCut::Before; });
    add([](auto& q) { q.query.includeStoppingAccess = true; });
    add([](auto& q) { q.query.continuationOwner = fs::NoControlId; });
    check(!keys.insert(base).second, "unchanged full key shares an answer");
    check(
        !detail::prepareOriginalInterval(g, fs::OriginalProgramVersion::fresh(), base.query).valid,
        "foreign snapshot rejected, not rebound");
    auto versioned = g;
    versioned.version = base.query.version;
    auto foreign = base;
    foreign.query.version = fs::OriginalProgramVersion::fresh();
    check(!walk(versioned, foreign).complete, "graph cannot accept another original snapshot directly");
    auto forged = base;
    forged.owner = 99;
    check(!walk(g, forged).complete, "caller cannot forge a different owner");
}
static void boundaries()
{
    const auto body = seq({op(0), control(Region::Choice, 10, {seq({op(1)}), seq({})}), op(2)});
    const auto g = graph(body, 3);
    const auto v = fs::OriginalProgramVersion::fresh();
    auto q = query({0, OriginalCut::After}, OriginalCut::scope(10, OriginalCut::After));
    auto result = walk(g, prepare(g, v, q));
    check(result.operations == std::vector<std::size_t>{1}, "if join excludes unrelated suffix");
    check(result.cases.reachedStop && result.cases.childEntry, "join keeps both child cases");
    q = query({0, OriginalCut::After}, {1, OriginalCut::Before});
    result = walk(g, prepare(g, v, q));
    check(
        result.operations == std::vector<std::size_t>{2} && result.cases.reachedOwnerExit,
        "branch bypass remains in the designated owner continuation");
    const auto flat = graph(seq({op(0), op(1)}), 2);
    for (auto side : {OriginalCut::Before, OriginalCut::After}) {
        for (bool include : {false, true}) {
            auto one = query({0, OriginalCut::After}, {1, side});
            one.includeStoppingAccess = include;
            check(
                walk(flat, prepare(flat, v, one)).operations ==
                    (include ? std::vector<std::size_t>{1} : std::vector<std::size_t>{}),
                "stop inclusion independent of stopping side");
        }
    }
    auto same = query({0, OriginalCut::After}, {0, OriginalCut::After});
    same.includeStoppingAccess = true;
    check(walk(flat, prepare(flat, v, same)).operations.empty(), "start-after never imports preceding access");
    const auto phaseGraph = graph(seq({op(0), op(1)}), 2, {{0, OriginalCut::After}, {1, OriginalCut::Before}});
    check(
        !detail::prepareOriginalInterval(phaseGraph, v, query({0, OriginalCut::After}, {1, OriginalCut::After})).valid,
        "unavailable internal source cut rejected without widening");
    check(
        !detail::prepareOriginalInterval(phaseGraph, v, query({0, OriginalCut::Before}, {1, OriginalCut::Before}))
             .valid,
        "unavailable internal target cut rejected without widening");
    check(
        prepare(phaseGraph, v, query({0, OriginalCut::Before}, {1, OriginalCut::After})).owner == fs::NoControlId,
        "multi-phase outer cuts remain legal");
    auto wrapped =
        seq({seq({op(0), seq({control(Region::Choice, 10, {seq({seq({op(1)})}), seq({seq({})})})})}), seq({op(2)})});
    const auto wrappers = graph(wrapped, 3);
    q = query({0, OriginalCut::After}, OriginalCut::scope(10, OriginalCut::After));
    const auto a = prepare(g, v, q), b = prepare(wrappers, v, q);
    check(
        a == b && walk(g, a).operations == walk(wrappers, b).operations,
        "transparent sequences preserve semantic cuts and owner");
}
static void recurrenceAndWork()
{
    const auto body = seq({control(Region::For, 10, {seq({op(0), op(1)})}), op(2)});
    const auto g = graph(body, 3);
    const auto v = fs::OriginalProgramVersion::fresh();
    auto q = query({0, OriginalCut::Before}, {0, OriginalCut::Before});
    const auto first = prepare(g, v, q);
    check(walk(g, first).operations.empty(), "same visit is empty");
    q.occurrence.stopVisit = StopVisit::AfterBackedge;
    q.occurrence.backedgeOwner = 10;
    const auto next = prepare(g, v, q);
    const auto answer = walk(g, next);
    check(
        first != next && answer.operations == std::vector<std::size_t>({0, 1}),
        "same static endpoints with backedge interpretation have different answers");
    check(answer.cases.backedge && answer.cases.bypass, "backedge and possible exit preserved");
    q.occurrence.stopVisit = StopVisit::Unqualified;
    const auto unknown = walk(g, prepare(g, v, q));
    check(
        unknown.operations.empty() && unknown.unresolvedOccurrence,
        "unqualified repeated empty interval is not an exclusion");
    q.occurrence.stopVisit = StopVisit::AfterBackedge;
    q.occurrence.backedgeOwner = 999;
    check(!detail::prepareOriginalInterval(g, v, q).valid, "missing named backedge is explicit");
    const auto whileBody = seq({control(Region::While, 20, {seq({op(0)}), seq({op(1)})}), op(2)});
    const auto w = graph(whileBody, 3);
    const auto before = OriginalCut::childBoundary(20, 0, OriginalCut::Before);
    const auto exit = OriginalCut::scope(20, OriginalCut::After);
    const auto finalVisit = walk(w, prepare(w, v, query(before, exit)));
    check(finalVisit.operations == std::vector<std::size_t>({0, 1}), "while continuation retains final before visit");
    const auto wtraces = execute(whileBody, 0);
    check(
        referenceInterval(
            surround(
                wtraces, OriginalCut::scope(fs::NoControlId, OriginalCut::Before),
                OriginalCut::scope(fs::NoControlId, OriginalCut::After)),
            prepare(w, v, query(before, exit))) == std::set<std::size_t>{0},
        "zero-body while executes before, not after");
    constexpr std::size_t n = 512;
    Region many;
    for (std::size_t i = 0; i < n; ++i) {
        many.children.push_back(control(Region::Choice, n + i, {seq({op(i)}), seq({})}));
    }
    const auto shared = graph(many, n);
    const auto full = prepare(
        shared, v,
        query(
            OriginalCut::scope(fs::NoControlId, OriginalCut::Before),
            OriginalCut::scope(fs::NoControlId, OriginalCut::After)));
    const auto all = walk(shared, full);
    check(
        shared.sites.size() <= 16 * n + 8 && all.visitedSites <= shared.sites.size() && all.operations.size() == n,
        "independent choices retain linear graph and traversal work without valuations");
}
int main()
{
    recordsAndOwnership();
    cacheKeys();
    boundaries();
    recurrenceAndWork();
    compareOracle(seq({op(0), control(Region::Choice, 10, {seq({op(1)}), seq({})}), op(2)}), 3, 0);
    compareOracle(seq({op(0), control(Region::For, 10, {seq({op(1), op(2)})}), op(3)}), 4, 3);
    compareOracle(seq({control(Region::While, 10, {seq({op(0)}), seq({op(1)})}), op(2)}), 3, 3);
    compareOracle(
        seq(
            {control(Region::For, 10, {seq({op(0), control(Region::Choice, 20, {seq({op(1)}), seq({op(2)})})})}),
             op(3)}),
        4, 3);
    compareOracle(
        seq({control(Region::For, 10, {seq({op(0), control(Region::For, 20, {seq({op(1)})}), op(2)})}), op(3)}), 4, 3);
    std::cout << "original interval core: " << checks << " checks; " << oracleQueries
              << " exhaustive cut/occurrence queries matched independent finite executions\n";
    return 0;
}
