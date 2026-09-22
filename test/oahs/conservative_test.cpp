// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "GraphOracle.h"
using namespace selected_test;
namespace {
const auto A = o::Pipe::MTE2, B = o::Pipe::MTE1, C = o::Pipe::M;
o::SelectedPlan checked(const o::Program& program)
{
    auto plan = o::constructConservativePlan(program);
    require(plan.success && plan.conservative, "conservative refusal: " + plan.reason);
    require(o::checkCausalFrontier(program, plan.commands).accepted, "conservative cold check");
    return plan;
}
void repeatedWalk()
{
    auto p = base(1, 1);
    p.operations = {op(A, {{0, false, true}}), op(C, {{0, true, false}})};
    p.body = {o::Region::For, {seq({leaf(0), leaf(1)})}, 0, true};
    // The closed walk must traverse B despite B having no payload. Repeated
    // visits use one key per direction and retain actual consumption history.
    for (auto& row : p.target.keys) {
        for (auto& keys : row) {
            keys.clear();
        }
    }
    p.target.keys[unsigned(A)][unsigned(B)] = {0};
    p.target.keys[unsigned(B)][unsigned(C)] = {0};
    p.target.keys[unsigned(C)][unsigned(A)] = {0};
    auto plan = checked(p);
    for (unsigned count : {0u, 1u, 2u, 5u}) {
        std::vector<unsigned> visits;
        for (unsigned i = 0; i < count; ++i) {
            visits.insert(visits.end(), {0, 1});
        }
        require(bool(oahs_oracle::graph(p, plan.commands, visits, {})), "closed tour oracle/rearming");
    }
    auto broken = plan.commands;
    for (auto& word : broken) {
        word.erase(std::remove_if(word.begin(), word.end(), [](const auto& command) {
            return command.source == C && command.observer == A;
        }), word.end());
    }
    require(!o::checkCausalFrontier(p, broken).accepted, "missing tour return was accepted");
    p.reservations.push_back({B, C, 0});
    const auto refused = o::constructConservativePlan(p);
    require(!refused.success && refused.commands.empty(), "reserved direction granted cross-component credit");
}
void singletonAndDisconnected()
{
    auto p = base(2, 0);
    p.operations = {op(A, {{0, false, true}}), op(A, {{0, true, false}}),
                    op(C, {{1, false, true}}), op(C, {{1, true, false}})};
    auto plan = checked(p);
    require(count(plan, o::Command::Publish) == 0 && count(plan, o::Command::Barrier) != 0,
            "disconnected independent pipes did not use local completion");
    p.operations[3].accesses = {{0, true, false}};
    require(!o::constructConservativePlan(p).success, "disconnected cross-pipe hazard disappeared");
    p.operations.resize(2);
    p.target.barriers[unsigned(A)] = false;
    require(!o::constructConservativePlan(p).success, "missing local completion primitive accepted");
    p.target.keys[unsigned(A)][unsigned(B)] = {0};
    p.target.keys[unsigned(B)][unsigned(A)] = {0};
    require(count(checked(p), o::Command::Publish) != 0, "unused intermediary did not supply return-cycle completion");
}
void sharedWords()
{
    auto p = base(1, 1);
    p.operations = {op(A, {{0, false, true}}), op(C, {{0, true, false}})};
    o::ObservedControl q;
    q.qualification = "test-only original shared-word graph";
    q.entry = 1;
    q.exit = 4;
    q.sites.resize(5);
    q.observations = {{100, {}, true}, {101, {}, true}, {102, {}, true}};
    q.sites[0] = {1, 0, {4}, {}}; // unreachable representative of the read word
    q.sites[1] = {0, 1, {2, 3}, {}};
    q.sites[2] = {1, 0, {4}, {}};
    q.sites[3] = {1, 0, {4}, {}};
    q.sites[4].observation = 2;
    p.observed = q;
    const auto plan = checked(p);
    for (auto copy : {2u, 3u}) {
        require(plan.commands[0].size() == plan.commands[copy].size() && !plan.commands[copy].empty(),
                "shared receipt packet disappeared with unreachable representative");
    }
    auto broken = plan.commands;
    broken[2].clear();
    require(!o::checkCausalFrontier(p, broken).accepted, "different shared words accepted");
}
void originalChoices()
{
    auto p = base(1, 1);
    p.operations = {op(A, {{0, false, true}}), op(B, {{0, false, true}}), op(C, {{0, true, false}})};
    const o::Region choice{o::Region::Choice, {leaf(0), leaf(1)}};
    p.body = {o::Region::For, {seq({choice, leaf(2)})}, 0, true};
    const auto plan = checked(p);
    for (const auto& visits : std::vector<std::vector<unsigned>>{{}, {0, 2}, {1, 2}, {0, 2, 1, 2}, {1, 2, 0, 2}}) {
        require(bool(oahs_oracle::graph(p, plan.commands, visits, {})), "branch/loop conservative oracle");
    }
    auto observed = o::addStructuredBoundaryCuts(p);
    require(observed.success, "original control observation failed");
    checked(observed.program);
}
void contractBoundaries()
{
    auto p = base(1, 1);
    p.operations = {op(o::Pipe::S, {{0, false, true}}), op(A, {{0, true, false}})};
    p.target.synchronous[unsigned(o::Pipe::S)] = true;
    checked(p); // scalar participates through genuinely available event edges
    for (unsigned pipe = 0; pipe < o::PipeCount; ++pipe) {
        p.target.keys[unsigned(o::Pipe::S)][pipe].clear();
        p.target.keys[pipe][unsigned(o::Pipe::S)].clear();
    }
    require(!o::constructConservativePlan(p).success, "synchronous scalar supplied global completion");
    p.operations = {op(A, {{0, false, true}})};
    p.target.barrierAll = true;
    p.invocation.retirement = o::Program::InvocationContract::DrainAllAtReturn;
    auto plan = checked(p);
    require(plan.commands.back().back().kind == o::Command::BarrierAll, "missing terminal retirement");
    plan.commands.back().clear();
    require(!o::checkCausalFrontier(p, plan.commands).accepted, "pre-payload packet counted as final retirement");
    p.cells[0].exclusive = true;
    require(!o::constructConservativePlan(p).success, "unsupported exclusive contract silently weakened");
    p.cells[0].exclusive = false;
    p.operations[0].resources.push_back({"opaque", true, false});
    require(!o::constructConservativePlan(p).success, "typed effects silently discarded");
    p.operations[0].resources.clear();
    p.invocation.externalProgress = true;
    require(!o::constructConservativePlan(p).success, "local checking claimed external progress");
    p.invocation.externalProgress = false;
    o::SelectedOptions options;
    options.conservativeOnly = true;
    o::Commands fixed(2);
    fixed[0].push_back({o::Command::Barrier, A});
    const auto refused = o::constructSelectedPlan(p, fixed, options);
    require(!refused.success && refused.commands.empty(), "fixed word silently discarded");
}
} // namespace
int main()
{
    checked(base(0, 0));
    repeatedWalk();
    singletonAndDisconnected();
    sharedWords();
    originalChoices();
    contractBoundaries();
}
