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
const auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE3;
namespace {
void straightLine()
{
    auto p = base(2);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                    op(Q, {{0, true, false}}), op(P, {{0, false, true, true}})};
    auto result = accepted(p);
    require(count(result, o::Command::Publish) == 2, "T1 requires readiness and release only");
    require(count(result, o::Command::Barrier) == 0, "reader return should cover old writer WAW");
    require(result.commands[1].size() == 1 && result.commands[1][0].kind == o::Command::Publish,
            "first publication must precede unrelated write");
    p.operations.push_back(op(P, {{1, true, false}}));
    result = accepted(p);
    require(count(result, o::Command::Barrier) == 1, "reader return cannot cover a later unrelated write");
    p = base(2);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                    op(Q, {{0, true, false}}), op(Q, {{1, true, false}})};
    result = accepted(p);
    require(count(result, o::Command::Publish) == 2 && result.work.acknowledgments == 0,
            "independent first consumers retain separate early prefixes");
    p.operations.resize(3);
    p.operations[2] = op(Q, {{0, true, false}, {1, true, false}});
    result = accepted(p);
    require(count(result, o::Command::Publish) == 1, "common operands should share one source prefix");
}
void sharedCredit()
{
    auto p = base(3);
    p.operations = {op(P, {{0, true, false}, {1, false, true, true}}),
                    op(Q, {{1, true, false}, {2, false, true, true}}),
                    op(R, {{2, true, false}, {0, false, true, true}})};
    auto result = accepted(p);
    require(count(result, o::Command::Publish) == 2, "local chain must share GM source-read completion");
    p.operations.push_back(op(P, {{0, true, false}, {1, false, true, true}}));
    result = accepted(p);
    require(count(result, o::Command::Publish) == 3, "new load requires actual store return");
    p = base(1);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                    op(R, {{0, true, false}}), op(P, {{0, false, true, true}})};
    result = accepted(p);
    require(count(result, o::Command::Publish) == 4, "two independent readers require both releases");
    require(count(result, o::Command::Barrier) == 0, "external releases cover old writer");
    p.operations[1] = op(Q, {{0, true, true, true}});
    result = accepted(p);
    require(result.success, "RMW must query before replacing both histories");
}
void scarcity()
{
    auto p = base(2, 1);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                    op(Q, {{0, true, false}}), op(Q, {{1, true, false}})};
    auto result = accepted(p);
    require(result.ledger.size() == 6 && result.work.acknowledgments == 1, "one-key T6 should add one acknowledgment");
    require(result.commands[2].size() >= 2 && result.commands[2][0].kind == o::Command::Acquire &&
            result.commands[2][1].kind == o::Command::Publish && result.commands[2][1].source == Q,
            "acknowledgment belongs immediately after old wait, before its read");
    for (const auto& endpoint : result.ledger) {
        if (endpoint.purpose == o::EndpointPurpose::ConsumptionAcknowledgment) {
            require(endpoint.acknowledges != o::NoAnalysisId, "helper must identify its consumption obligation");
        }
    }
    // Remove every reverse/relay possibility. Do not silently coalesce consumers.
    for (auto& row : p.target.keys) {
        for (auto& pool : row) {
            pool.clear();
        }
    }
    p.target.keys[unsigned(P)][unsigned(Q)] = {0};
    result = o::constructSelectedPlan(p);
    require(!result.success && result.commands.empty() && result.failure == o::SelectedFailure::EventResource,
            "one forward key without return must refuse this policy");
    p = base(1, 2);
    p.reservations.push_back({P, Q, 0, false});
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}})};
    result = accepted(p);
    require(result.commands[1][0].key == 1, "reservation must be excluded at source allocation");
    for (auto& row : p.target.keys) {
        for (auto& pool : row) {
            pool.clear();
        }
    }
    p.reservations.clear();
    p.target.keys[unsigned(P)][unsigned(R)] = {0};
    p.target.keys[unsigned(R)][unsigned(Q)] = {0};
    result = accepted(p);
    require(count(result, o::Command::Publish) == 2, "canonical route should use actual relay endpoints");

    // v0.22 T6b: stable F7 repair selection must repair key 0 after its
    // preceding acquisition. Repairing key 1 would add completion of read x
    // before issue of read z, despite the two reads being independent.
    p = base(3, 2);
    p.operations = {op(P, {{0, false, true, true}}), op(P, {{1, false, true, true}}),
                    op(P, {{2, false, true, true}}), op(Q, {{0, true, false}}),
                    op(Q, {{1, true, false}}), op(Q, {{2, true, false}})};
    result = accepted(p);
    require(result.work.acknowledgments == 1 && result.ledger.size() == 8 &&
            count(result, o::Command::Barrier) == 0,
            "T6b requires the exact eight-endpoint, fence-free realization");
    const auto repaired = std::find_if(result.decisions.begin(), result.decisions.end(), [](const auto& decision) {
        return decision.repairedAcquisition != o::NoAnalysisId;
    });
    require(repaired != result.decisions.end() && repaired->repairedForwardKey == 0 &&
            repaired->repairReverseKey == 0 && repaired->repairInputVersion < repaired->repairOutputVersion,
            "T6b must record the complete stable repair certificate for key 0");
    require(repaired->endpoints.size() == 4,
            "T6b repair decision must name helper and forward endpoints together");
    require(repaired->repairedAcquisition < result.ledger.size() &&
            result.ledger[repaired->repairedAcquisition].cut == 3,
            "T6b acknowledgment must target the first acquisition");
    require(bool(oahs_oracle::graph(p, result.commands, {0, 1, 2, 3, 4, 5}, {{3, 5}})),
            "T6b stable repair added read-x completion before read-z issue");
}
void structured()
{
    auto p = base(1);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                    op(P, {{0, false, true, true}})};
    p.body = seq({leaf(0), {o::Region::Choice, {leaf(1), seq({})}}, leaf(2)});
    auto result = accepted(p);
    require(result.work.commonCutTransfers != 0, "branch participation uses qualified local endpoints");
    p.body = seq({leaf(0), {o::Region::For, {leaf(1)}, 0, true}, leaf(2)});
    result = accepted(p);
    require(result.success, "read-only loop retains incoming writer and enclosing overwrite");
    p.body = {o::Region::For, {seq({leaf(0), leaf(1), leaf(2)})}, 0, true};
    result = accepted(p);
    require(result.certificate.siteEvaluations > p.operations.size(), "loop accepted through a fixed point");
    p.body = {o::Region::While, {seq({leaf(0), leaf(1)}), leaf(2)}};
    result = accepted(p);
    require(result.success, "while-before is mandatory and participates in recurrence");
    p.body = {o::Region::For, {seq({leaf(0), {o::Region::For, {leaf(1)}, 0, true}, leaf(2)})}, 0, true};
    result = accepted(p);
    require(result.success, "nested loop keeps surrounding reuse");
}
void qualification()
{
    auto p = base(1);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}})};
    p.target.barrierAll = true;
    p.invocation.retirement = o::Program::InvocationContract::DrainAllAtReturn;
    auto result = accepted(p);
    require(result.commands.back().size() == 1 && result.commands.back()[0].kind == o::Command::BarrierAll,
            "terminal retirement must be emitted under the unchanged contract");
    auto changed = result.commands;
    changed.back().clear();
    require(!o::checkCausalFrontier(p, changed).accepted, "missing retirement must refuse");
    changed = result.commands;
    changed[1].clear();
    require(!o::checkCausalFrontier(p, changed).accepted, "terminal retirement cannot repair a missing handoff");
    p.operations = {op(o::Pipe::S, {{0, false, true, true}}), op(o::Pipe::S, {{0, true, false}})};
    p.target.synchronous[unsigned(o::Pipe::S)] = true;
    p.target.barriers[unsigned(o::Pipe::S)] = false;
    result = accepted(p);
    require(count(result, o::Command::Barrier) == 0, "synchronous lane needs no invented scalar fence");
    p.operations[0].visibility.push_back({0, true, false});
    result = o::constructSelectedPlan(p);
    require(!result.success && result.commands.empty(), "unsupported visibility cannot be erased");
}
} // namespace
int main()
{
    straightLine();
    sharedCredit();
    scarcity();
    structured();
    qualification();
    std::cout << "selected-plan named policy tests passed\n";
}
