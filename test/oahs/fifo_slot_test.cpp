// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
#include <functional>
using namespace selected_test;
namespace {
o::Program fixture()
{
    auto p = base(3);
    p.operations = {
        op(o::Pipe::MTE2, {{0, true, false, false}, {1, false, true, false}}),
        op(o::Pipe::V, {{1, true, false, false}, {2, false, true, false}}),
        op(o::Pipe::MTE3, {{2, true, false, false}, {0, false, true, false}})};
    for (unsigned i = 0; i < p.operations.size(); ++i)
        p.operations[i].original = i;
    o::ObservedControl q;
    q.qualification = "original guarded alternating FIFO";
    q.scopes = {{0, o::NoControlId, o::NoControlId}};
    q.entry = 0;
    q.exit = 9;
    q.sites.resize(10);
    const std::vector<std::vector<std::size_t>> edges = {{1}, {2, 9}, {3, 1}, {4}, {5}, {6}, {7}, {8}, {1}, {}};
    for (unsigned i = 0; i < 10; ++i) {
        q.sites[i].successors = edges[i];
        q.sites[i].observation = i;
        q.observations.push_back({i, {}, true});
    }
    q.sites[3].operation = 0;
    q.sites[5].operation = 1;
    q.sites[7].operation = 2;
    q.sites[8].backedgeOwners = {0};
    q.sites[2].backedgeOwners = {o::NoControlId, 0};
    p.observed = std::move(q);
    p.invocation.retirement = o::Program::InvocationContract::DrainAllAtReturn;
    p.invocation.boundary = "FIFO test invocation";
    p.target.barrierAll = true;
    return p;
}
void staticCursorSlots()
{
    auto p = base(2);
    for (unsigned i = 0; i < 8; ++i) {
        p.operations.push_back(
            op(i < 4 ? o::Pipe::FIX : o::Pipe::MTE2, {{0, i >= 4, i < 4, false}, {1, true, false, false}}));
        p.operations.back().original = i;
    }
    o::ObservedControl q;
    q.qualification = "independent producer/consumer cursors across three phases";
    q.scopes = {{0, o::NoControlId, o::NoControlId}, {1, 0, o::NoControlId}};
    q.entry = 0;
    q.exit = 11;
    q.sites.resize(12);
    const std::vector<std::vector<std::size_t>> edges = {{1, 11}, {2}, {3}, {4, 8}, {5}, {6},
                                                         {7},     {3}, {9}, {10},   {0}, {}};
    for (unsigned i = 0; i < q.sites.size(); ++i) {
        q.sites[i].successors = edges[i];
        q.sites[i].observation = i;
        q.observations.push_back({i, {}, true});
    }
    const unsigned sites[] = {1, 2, 4, 5, 6, 7, 8, 9};
    for (unsigned i = 0; i < 8; ++i)
        q.sites[sites[i]].operation = i;
    q.sites[7].backedgeOwners = {1};
    q.sites[10].backedgeOwners = {0};
    p.observed = std::move(q);
    o::AlternatingSlotRegion region{0, 2, 128, {4, 5, 6, 7}, {0, 1, 2, 3}};
    auto r = o::refineStaticSlots(p, region);
    require(r.success, "static cursor refinement: " + r.reason);
    require(
        r.program.observed->sites.size() == p.observed->sites.size() &&
            r.program.operations.size() == p.operations.size() && !r.program.alternatingSlots,
        "static slots must not expand control or install an alternating protocol");
    for (unsigned i = 0; i < 8; ++i) {
        const auto& a = r.program.operations[i].accesses;
        require(
            a[0].cell == 2 + i % 2 && !a[0].definiteWrite && a[1].cell == 1,
            "independent send/pop cursors and unrelated effects");
    }
    // Independently count participating operations, across empty, short,
    // repeated and varying-length entries. No reset at a child boundary.
    for (unsigned first = 0; first < 4; ++first)
        for (unsigned second = 0; second < 4; ++second) {
            std::vector<unsigned> visits;
            for (auto length : {first, second}) {
                visits.insert(visits.end(), {0, 1});
                for (unsigned j = 0; j < length; ++j)
                    visits.insert(visits.end(), {2, 3, 4, 5});
                visits.insert(visits.end(), {6, 7});
            }
            unsigned reads = 0, writes = 0;
            for (auto op : visits) {
                auto slot = (op < 4 ? writes++ : reads++) % 2;
                require(r.program.operations[op].accesses[0].cell == 2 + slot, "phase-boundary cursor mismatch");
            }
        }
    auto skipped = p;
    skipped.observed->sites[3].successors.push_back(5);
    require(!o::refineStaticSlots(skipped, region).success, "skipped send must not retain static slot identity");
    auto extra = region;
    extra.reads.pop_back();
    require(!o::refineStaticSlots(p, extra).success, "unrepresented FIFO user must decline");
    extra = region;
    extra.writes.push_back(4);
    require(!o::refineStaticSlots(p, extra).success, "op cannot advance both cursor roles");
    extra = region;
    extra.slotBytes = std::numeric_limits<uint64_t>::max();
    require(!o::refineStaticSlots(p, extra).success, "slot offset overflow must decline");
    require(!o::refineStaticSlots(r.program, region).success, "already split root must not be refined twice");
    require(
        !o::refineStaticSlots(fixture(), {0, 2, 128, {0}, {2}}).success,
        "one send/pop per repeated body requires a changing slot observation");
}
void checkTraces(const o::Program& p, const o::Commands& commands)
{
    const auto& q = *p.observed;
    std::vector<std::size_t> path;
    unsigned traces = 0;
    std::function<void(std::size_t, unsigned, unsigned, unsigned)> visit;
    visit = [&](std::size_t site, unsigned steps, unsigned reads, unsigned writes) {
        if (steps > 4)
            return;
        auto anchor = q.observations[q.sites[site].observation].anchor;
        auto phase = q.sites[site].operation;
        if (phase != o::NoControlId) {
            const auto& op = p.operations[phase];
            if (op.original == 0 || op.original == 2) {
                unsigned index = op.original == 0 ? reads++ : writes++;
                unsigned cell = p.alternatingSlots->cells[index % 2];
                require(
                    std::any_of(op.accesses.begin(), op.accesses.end(), [&](auto a) { return a.cell == cell; }),
                    "slot identity must follow participation, not loop visits");
            }
        }
        path.push_back(site);
        if (site == q.exit) {
            auto concrete = p;
            concrete.observed.reset();
            concrete.alternatingSlots.reset();
            concrete.operations.clear();
            o::Commands words(1);
            std::vector<unsigned> order;
            for (auto cut : path) {
                words.back().insert(words.back().end(), commands[cut].begin(), commands[cut].end());
                auto op = q.sites[cut].operation;
                if (op != o::NoControlId) {
                    order.push_back(concrete.operations.size());
                    concrete.operations.push_back(p.operations[op]);
                    words.emplace_back();
                }
            }
            require(bool(oahs_oracle::graph(concrete, words, order)), "FIFO finite memory/event oracle");
            ++traces;
        } else
            for (auto next : q.sites[site].successors)
                visit(next, steps + (anchor == 1), reads, writes);
        path.pop_back();
    };
    visit(q.entry, 0, 0, 0);
    require(traces > 10, "FIFO skipped/repeated episode coverage");
}
} // namespace
int main()
{
    staticCursorSlots();
    auto p = fixture();
    o::AlternatingSlotRegion region{0, 2, 128, {0}, {2}};
    auto refined = o::refineAlternatingSlots(p, region);
    require(refined.success, "qualified two-slot FIFO");
    const auto& r = refined.program;
    require(r.observed->sites.size() <= 2 * p.observed->sites.size(), "FIFO site bound");
    require(
        r.observed->observations.size() == p.observed->observations.size(), "hidden phase must not add runtime guards");
    auto plan = o::constructSelectedPlan(r);
    require(plan.success, "FIFO selected construction: " + plan.reason + " at " + std::to_string(plan.cut));
    require(plan.channels.size() == 2, "two logical directions, not per-cell channels");
    require(o::checkCausalFrontier(r, plan.commands).accepted, "FIFO cold validation");
    for (auto read : r.alternatingSlots->reads)
        for (auto c : plan.commands[read])
            require(
                !(c.kind == o::Command::Acquire && c.source == o::Pipe::MTE3 && c.observer == o::Pipe::MTE2),
                "output return gates receive");
    checkTraces(r, plan.commands);
    // Isolate the FIFO proof: the richer fixture can carry the same credit
    // through another necessary local-storage receipt, which is legitimate.
    auto bare = p;
    bare.operations[0].accesses = {{0, true, false, false}};
    bare.operations[1].accesses.clear();
    bare.operations[2].accesses = {{0, false, true, false}};
    auto bareSlots = o::refineAlternatingSlots(bare, region);
    auto barePlan = o::constructSelectedPlan(bareSlots.program);
    require(barePlan.success, "isolated FIFO construction");
    auto broken = barePlan.commands;
    for (auto& word : broken)
        word.erase(
            std::remove_if(
                word.begin(), word.end(),
                [](auto c) {
                    return c.source == o::Pipe::MTE3 && c.observer == o::Pipe::MTE2 &&
                           (c.kind == o::Command::Publish || c.kind == o::Command::Acquire);
                }),
            word.end());
    require(
        !o::checkCausalFrontier(bareSlots.program, broken).accepted,
        "older same-slot writer needs the actual preceding return");
    region.slots = 1;
    require(!o::refineAlternatingSlots(p, region).success, "one-slot delayed-return negative");
    region.slots = 2;
    auto mismatch = p;
    mismatch.observed->sites[6].successors = {7, 8};
    require(!o::refineAlternatingSlots(mismatch, region).success, "optional send is not a whole skipped episode");
    auto extra = p;
    extra.operations[1].accesses.push_back({0, true, false, false});
    require(!o::refineAlternatingSlots(extra, region).success, "unrepresented root user");
    require(!o::refineAlternatingSlots(r, region).success, "no hidden phase products");
    auto malformed = r;
    malformed.alternatingSlots->cells.clear();
    require(!o::constructSelectedPlan(malformed).success, "malformed slot metadata must decline");
    std::cout << "FIFO slot tests passed\n";
}
