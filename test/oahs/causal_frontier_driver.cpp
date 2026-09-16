// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Test-only numeric interchange for the actual production frontier service.
#include "PTO/Transforms/OAHS/CausalFrontier.h"
#include <iomanip>
#include <iostream>
namespace o = mlir::pto::oahs;
namespace {
bool region(o::Region& r)
{
    unsigned kind = 0, zero = 0;
    std::size_t children = 0;
    if (!(std::cin >> kind >> r.operation >> zero >> children) || kind > o::Region::Operation || zero > 1)
        return false;
    r.kind = o::Region::Kind(kind);
    r.zeroTripPossible = zero;
    r.children.resize(children);
    for (auto& child : r.children)
        if (!region(child))
            return false;
    return true;
}
bool command(o::Command& c)
{
    unsigned kind = 0, a = 0, b = 0;
    if (!(std::cin >> kind >> a >> b >> c.key) || kind > o::Command::BarrierAll || a >= o::PipeCount ||
        b >= o::PipeCount)
        return false;
    c.kind = o::Command::Kind(kind);
    c.source = o::Pipe(a);
    c.observer = o::Pipe(b);
    return true;
}
void bitset(const o::FrontierBits& b)
{
    std::cout << '[';
    for (std::size_t i = 0; i < b.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << b[i];
    }
    std::cout << ']';
}
void facts(const o::FrontierState& s)
{
    if (!s.reachable()) {
        std::cout << "null";
        return;
    }
    const auto& f = *s.facts();
    std::cout << "{\"reach\":[";
    for (std::size_t i = 0; i < f.reach.size(); ++i) {
        if (i)
            std::cout << ',';
        bitset(f.reach[i]);
    }
    std::cout << "],\"history\":[";
    for (std::size_t i = 0; i < f.history.size(); ++i) {
        if (i)
            std::cout << ',';
        if (f.history[i])
            bitset(*f.history[i]);
        else
            std::cout << "null";
    }
    std::cout << "],\"events\":[";
    for (std::size_t i = 0; i < f.events.size(); ++i) {
        if (i)
            std::cout << ',';
        std::cout << "{\"occupancy\":" << unsigned(f.events[i].occupancy) << ",\"publishers\":[";
        for (std::size_t j = 0; j < f.events[i].publishers.size(); ++j) {
            if (j)
                std::cout << ',';
            const auto& b = f.events[i].publishers[j];
            std::cout << '[' << b.cut << ',' << b.command << ']';
        }
        std::cout << "]}";
    }
    std::cout << "]}";
}
void residuals(const std::vector<o::FrontierRequirement>& requirements)
{
    std::cout << '[';
    for (std::size_t i = 0; i < requirements.size(); ++i) {
        if (i)
            std::cout << ',';
        const auto& r = requirements[i];
        std::cout << '[' << r.cell << ',' << unsigned(r.source) << ',' << r.sourceWrite << ',' << r.consumer << ','
                  << r.consumerRead << ',' << r.consumerWrite << ']';
    }
    std::cout << ']';
}
} // namespace
int main()
{
    unsigned version = 0, mode = 0;
    std::size_t operations = 0, cells = 0, keys = 0;
    if (!(std::cin >> version >> mode >> operations >> cells >> keys) || version != 1 || mode > 1)
        return 2;
    o::Program p;
    p.target.contract = "test bridge ordinary issue-only core";
    p.target.supported.fill(true);
    p.target.barriers.fill(true);
    p.cells.resize(cells);
    p.operations.resize(operations);
    for (std::size_t i = 0; i < keys; ++i) {
        unsigned a, b, k;
        if (!(std::cin >> a >> b >> k) || a >= o::PipeCount || b >= o::PipeCount || a == b)
            return 2;
        p.target.keys[a][b].push_back(k);
    }
    for (auto& op : p.operations) {
        unsigned pipe;
        std::size_t effects;
        if (!(std::cin >> pipe >> effects) || pipe >= o::PipeCount)
            return 2;
        op.pipe = o::Pipe(pipe);
        op.complete = true;
        for (std::size_t i = 0; i < effects; ++i) {
            unsigned cell, r, w;
            if (!(std::cin >> cell >> r >> w) || r > 1 || w > 1)
                return 2;
            op.accesses.push_back({cell, bool(r), bool(w)});
        }
    }
    if (!region(p.body))
        return 2;
    if (mode == 0) {
        o::Commands commands(operations + 1);
        for (auto& word : commands) {
            std::size_t n;
            if (!(std::cin >> n))
                return 2;
            word.resize(n);
            for (auto& c : word)
                if (!command(c))
                    return 2;
        }
        auto result = o::checkCausalFrontier(p, commands);
        std::cout << "{\"complete\":" << result.complete << ",\"accepted\":" << result.accepted
                  << ",\"failure\":" << unsigned(result.failure) << ",\"reason\":" << std::quoted(result.reason)
                  << ",\"cut\":" << result.cut << ",\"evaluations\":" << result.siteEvaluations << ",\"residuals\":";
        residuals(result.residuals);
        std::cout << ",\"cuts\":[";
        for (std::size_t i = 0; i < result.cuts.size(); ++i) {
            if (i)
                std::cout << ',';
            std::cout << '[';
            facts(result.cuts[i].incoming);
            std::cout << ',';
            facts(result.cuts[i].beforeIssue);
            std::cout << ',';
            facts(result.cuts[i].outgoing);
            std::cout << ']';
        }
        std::cout << "]}\n";
    } else {
        o::CausalFrontier frontier(p);
        if (!frontier.complete()) {
            std::cerr << frontier.reason() << '\n';
            return 3;
        }
        std::vector<o::FrontierState> states{frontier.initial()};
        std::size_t count;
        if (!(std::cin >> count))
            return 2;
        std::cout << '[';
        facts(states[0]);
        for (std::size_t i = 0; i < count; ++i) {
            unsigned kind;
            std::size_t input;
            if (!(std::cin >> kind >> input) || input >= states.size())
                return 2;
            o::FrontierStep step;
            if (kind == 0) {
                std::size_t op;
                if (!(std::cin >> op))
                    return 2;
                step = frontier.issue(states[input], op);
            } else if (kind == 1) {
                o::Command c;
                o::FrontierBinding binding;
                if (!command(c) || !(std::cin >> binding.cut >> binding.command))
                    return 2;
                step = frontier.command(states[input], c, binding);
            } else if (kind == 2) {
                std::size_t other;
                if (!(std::cin >> other) || other >= states.size())
                    return 2;
                step = frontier.join(states[input], states[other]);
            } else if (kind == 3)
                step = frontier.exit(states[input]);
            else
                return 2;
            states.push_back(step.state);
            std::cout << ",{\"applied\":" << step.applied << ",\"failure\":" << unsigned(step.failure)
                      << ",\"residuals\":";
            residuals(step.residuals);
            std::cout << ",\"state\":";
            facts(step.state);
            std::cout << '}';
        }
        std::cout << "]\n";
    }
    std::string trailing;
    return (std::cin >> trailing) ? 2 : 0;
}
