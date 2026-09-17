// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "../../lib/PTO/Transforms/OAHS/Control.h"
#include <iomanip>
#include <numeric>
using namespace selected_test;
namespace {
void bits(const o::FrontierBits& values)
{
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << (i ? "," : "") << values[i];
    }
    std::cout << ']';
}
void facts(const o::FrontierState& state)
{
    if (!state.reachable()) {
        std::cout << "null";
        return;
    }
    const auto& f = *state.facts();
    std::cout << "{\"reach\":[";
    for (std::size_t i = 0; i < f.reach.size(); ++i) {
        if (i) { std::cout << ','; }
        bits(f.reach[i]);
    }
    std::cout << "],\"history\":[";
    for (std::size_t i = 0; i < f.history.size(); ++i) {
        if (i) { std::cout << ','; }
        if (const auto* reached = f.history.find(i)) { bits(*reached); } else { std::cout << "null"; }
    }
    std::cout << "],\"events\":[";
    for (std::size_t i = 0; i < f.events.size(); ++i) {
        if (i) { std::cout << ','; }
        std::cout << "{\"occupancy\":" << unsigned(f.events[i].occupancy) << ",\"publishers\":[";
        for (std::size_t j = 0; j < f.events[i].publishers.size(); ++j) {
            const auto& b = f.events[i].publishers[j];
            std::cout << (j ? "," : "") << '[' << b.cut << ',' << b.command << ']';
        }
        std::cout << "]}";
    }
    std::cout << "]}";
}
void region(const o::Region& r)
{
    std::cout << "{\"kind\":" << unsigned(r.kind) << ",\"operation\":" << r.operation << ",\"children\":[";
    for (std::size_t i = 0; i < r.children.size(); ++i) {
        if (i) { std::cout << ','; }
        region(r.children[i]);
    }
    std::cout << "]}";
}
void emit(const std::string& name, const o::Program& p, bool expectedExact = false)
{
    const auto result = o::constructSelectedPlan(p);
    const auto graph = o::detail::buildControlGraph(p);
    o::CausalFrontier f(p);
    std::cout << "{\"name\":" << std::quoted(name) << ",\"success\":" << result.success
              << ",\"failure\":" << unsigned(result.failure) << ",\"reason\":" << std::quoted(result.reason)
              << ",\"expected_exact\":" << expectedExact << ",\"cells\":" << p.cells.size() << ",\"keys\":[";
    for (std::size_t i = 0; i < f.keys().size(); ++i) {
        const auto& key = f.keys()[i];
        std::cout << (i ? "," : "") << '[' << unsigned(key.source) << ',' << unsigned(key.observer) << ',' << key.key
              << ']';
    }
    std::cout << "],\"operations\":[";
    for (std::size_t i = 0; i < p.operations.size(); ++i) {
        const auto& op = p.operations[i];
        std::cout << (i ? "," : "") << "{\"pipe\":" << unsigned(op.pipe) << ",\"effects\":[";
        for (std::size_t j = 0; j < op.accesses.size(); ++j) {
            const auto& a = op.accesses[j];
            std::cout << (j ? "," : "") << '[' << a.cell << ',' << a.read << ',' << a.write << ']';
        }
        std::cout << "]}";
    }
    std::cout << "],\"body\":";
    region(p.body);
    std::cout << ",\"observed\":" << bool(p.observed) << ",\"entry\":" << graph.entry << ",\"exit\":" << graph.exit
              << ",\"sites\":[";
    for (std::size_t i = 0; i < graph.sites.size(); ++i) {
        std::cout << (i ? "," : "") << '[';
        if (graph.operations[i] == o::NoAnalysisId) { std::cout << "null"; } else { std::cout << graph.operations[i]; }
        std::cout << ",[";
        for (std::size_t j = 0; j < graph.sites[i].successors.size(); ++j) {
            std::cout << (j ? "," : "") << graph.sites[i].successors[j];
        }
        std::cout << "]]";
    }
    std::cout << "],\"commands\":[";
    for (std::size_t i = 0; i < result.commands.size(); ++i) {
        std::cout << (i ? "," : "") << '[';
        for (std::size_t j = 0; j < result.commands[i].size(); ++j) {
            const auto& c = result.commands[i][j];
            std::cout << (j ? "," : "") << '[' << unsigned(c.kind) << ',' << unsigned(c.source) << ','
                      << unsigned(c.observer) << ',' << c.key << ']';
        }
        std::cout << ']';
    }
    std::cout << "],\"cuts\":[";
    for (std::size_t i = 0; i < result.certificate.cuts.size(); ++i) {
        std::cout << (i ? "," : "") << '[';
        facts(result.certificate.cuts[i].incoming);
        std::cout << ',';
        facts(result.certificate.cuts[i].beforeIssue);
        std::cout << ',';
        facts(result.certificate.cuts[i].outgoing);
        std::cout << ']';
    }
    std::cout << "],\"work\":{\"visits\":" << result.work.frontierVisits << ",\"updates\":"
              << result.work.selectedUpdates
              << ",\"replay\":" << result.work.replaySiteEvaluations << ",\"forward\":"
              << result.work.forwardSiteEvaluations
              << ",\"microseconds\":" << result.work.elapsedMicroseconds << ",\"channels\":" << result.channels.size()
              << ",\"acknowledgments\":" << result.work.acknowledgments
              << ",\"preparation_microseconds\":" << result.work.preparationMicroseconds
              << ",\"validation_evaluations\":" << result.work.invariantSiteEvaluations
              << ",\"endpoints\":" << result.ledger.size() << ",\"sources\":" << result.sources.size()
              << ",\"key_queries\":" << result.work.keyQueries << "}}\n";
}
o::Program small(unsigned cells, unsigned keys = 2)
{
    auto p = base(cells, keys);
    for (unsigned a = 3; a < o::PipeCount; ++a) {
        p.target.supported[a] = false;
    }
    return p;
}
} // namespace
int main()
{
    const auto P = o::Pipe::S, Q = o::Pipe::V, R = o::Pipe::M;
    auto p = small(2);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                    op(Q, {{0, true, false}}), op(P, {{0, false, true, true}})};
    emit("T1-joint-return", p, true);
    p.operations[3] = op(Q, {{1, true, false}});
    emit("T2-early-prefixes", p, true);
    for (auto& row : p.target.keys) {
        for (auto& pool : row) {
            if (pool.size() > 1) { pool.resize(1); }
        }
    }
    emit("T6-consumption-acknowledgment", p, true);
    p = small(3);
    p.operations = {op(P, {{0, true, false}, {1, false, true, true}}),
                    op(Q, {{1, true, false}, {2, false, true, true}}), op(R, {{2, true, false}, {0, false, true, true}})};
    emit("movement-cross-domain", p, true);
    std::mt19937 rng(20260916);
    for (unsigned trial = 0; trial < 100; ++trial) {
        p = small(3);
        for (unsigned i = 0; i < 6; ++i) {
            const auto mode = 1 + rng() % 3;
            p.operations.push_back(op(o::Pipe(rng() % 3),
                {{unsigned(rng() % 3), bool(mode & 1), bool(mode & 2), bool(mode & 2)}}));
        }
        if (trial % 4 == 1) {
            p.body = {o::Region::For, {seq({leaf(0), leaf(1), leaf(2), leaf(3), leaf(4), leaf(5)})}, 0, true};
        } else if (trial % 4 == 2) {
            p.body = seq({{o::Region::Choice, {seq({leaf(0), leaf(1)}), seq({leaf(2), leaf(3)})}}, leaf(4), leaf(5)});
        } else if (trial % 4 == 3) {
            p.body = {o::Region::For, {seq({leaf(0), {o::Region::For, {seq({leaf(1), leaf(2), leaf(3)})}, 0, true},
                    leaf(4), leaf(5)})}, 0, true};
        }
        emit("generated-" + std::to_string(trial), p);
    }
    for (unsigned slots = 1; slots <= 4; ++slots) {
        p = small(slots, slots);
        p.target.supported[2] = false;
        p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}})};
        std::vector<unsigned> cells(slots);
        std::iota(cells.begin(), cells.end(), 0);
        auto input = o::makePeriodicLoop(p, slots, {{0, 0, cells, 1, 0}, {1, 0, cells, 1, 0}});
        require(input.success, input.reason);
        emit("cyclic-" + std::to_string(slots), input.program);
    }
    // Open first/steady/final roles, with surrounding accesses on a third engine.
    // The reference sees the entire original graph and all payload effects.
    p = small(1, 3);
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}}),
                    op(Q, {{0, true, false}})};
    auto open = o::makePeriodicLoop(p, 1, {});
    require(open.success, open.reason);
    auto& q = *open.program.observed;
    const auto preOp = open.program.operations.size();
    open.program.operations.push_back(op(R, {{0, false, true}}));
    const auto postOp = open.program.operations.size();
    open.program.operations.push_back(op(R, {{0, false, true}}));
    auto node = [&](std::size_t operation) {
        auto site = q.sites.size(), observation = q.observations.size();
        q.observations.push_back({1000 + site, {}, true});
        q.sites.push_back({operation, observation, {}, {}, 0});
        return site;
    };
    auto entry = q.entry, exit = q.exit;
    q.entry = node(preOp);
    auto post = node(postOp);
    q.exit = node(o::NoControlId);
    q.sites[q.entry].successors = {entry};
    q.sites[exit].successors = {post};
    q.sites[post].successors = {q.exit};
    emit("F8-open-multiple-readers", open.program);
    // An enclosing zero-or-more loop re-enters the qualified initializer. Its
    // final role return must prove rearming, including across an empty visit.
    q.sites[post].successors = {q.entry, q.exit};
    q.sites[post].backedgeOwners = {q.entry, o::NoControlId};
    emit("F8-reentered-open-region", open.program);

}
