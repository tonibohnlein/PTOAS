// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include <numeric>
using namespace selected_test;
namespace {
void checkSlots(unsigned slots)
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V;
    auto body = base(slots, slots + 1);
    for (unsigned a = 0; a < o::PipeCount; ++a) {
        body.target.supported[a] = a == unsigned(P) || a == unsigned(Q);
        for (unsigned b = 0; b < o::PipeCount; ++b) {
            if (!((a == unsigned(P) && b == unsigned(Q)) || (a == unsigned(Q) && b == unsigned(P)))) {
                body.target.keys[a][b].clear();
            }
        }
    }
    body.reservations = {{P, Q, 0, false}, {Q, P, 0, false}};
    body.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}})};
    std::vector<unsigned> cells(slots);
    std::iota(cells.begin(), cells.end(), 0);
    auto input = o::makePeriodicLoop(body, slots, {{0, 0, cells, 1, 0}, {1, 0, cells, 1, 0}});
    require(input.success, "periodic input: " + input.reason);
    auto result = accepted(input.program);
    require(result.channels.size() == 2 * slots, "cyclic qualifier must derive both roles per slot");
    require(result.work.acknowledgments == 0 && count(result, o::Command::Barrier) == 0 &&
            count(result, o::Command::BarrierAll) == 0, "qualified slot cycle must not select extra barriers/helpers");
    for (const auto& channel : result.channels) {
        require(channel.key == channel.cell + 1, "stable slots reserve lowest unreserved directional keys");
    }
    unsigned deleted = 0;
    for (const auto& endpoint : result.ledger) {
        auto words = result.commands;
        const auto& original = result.commands[endpoint.cut];
        auto found = std::find_if(original.begin(), original.end(), [&](const auto& c) {
            return c.kind == endpoint.command.kind && c.source == endpoint.command.source &&
                   c.observer == endpoint.command.observer && c.key == endpoint.command.key;
        });
        require(found != original.end(), "ledger endpoint absent from emitted word");
        const auto position = std::size_t(found - original.begin());
        const auto observation = input.program.observed->sites[endpoint.cut].observation;
        for (std::size_t site = 0; site < words.size(); ++site) {
            if (input.program.observed->sites[site].observation == observation) {
                words[site].erase(words[site].begin() + position);
            }
        }
        require(!o::checkCausalFrontier(input.program, words).accepted, "deleted cyclic endpoint was accepted");
        ++deleted;
    }
    // One original observation cannot receive different words in two contexts.
    auto words = result.commands;
    bool mutated = false;
    for (std::size_t site = 0; site < words.size(); ++site) {
        if (!words[site].empty() && o::canonicalCommandCut(input.program, site) != site) {
            words[site].clear();
            mutated = true;
            break;
        }
    }
    if (mutated) {
        require(!o::checkCausalFrontier(input.program, words).accepted, "nonuniform observation accepted");
    }
    auto unavailable = input.program;
    for (auto& observation : unavailable.observed->observations) {
        observation.available = false;
    }
    require(!o::constructSelectedPlan(unavailable).success, "unavailable original guards accepted");
    auto scarce = input.program;
    scarce.target.keys[unsigned(P)][unsigned(Q)].pop_back();
    auto refusal = o::constructSelectedPlan(scarce);
    require(!refusal.success && refusal.failure == o::SelectedFailure::EventResource,
            "insufficient qualified role keys must be a resource refusal, not fallback");
    std::cout << "slots=" << slots << " channels=" << result.channels.size()
              << " mutated=" << deleted << " fixedpoint=" << result.work.invariantSiteEvaluations << '\n';
}
} // namespace
int main()
{
    for (unsigned slots = 1; slots <= 4; ++slots) {
        checkSlots(slots);
    }
}
