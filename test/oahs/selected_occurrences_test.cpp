// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "../../lib/PTO/Transforms/OAHS/SelectedInternal.h"
using namespace selected_test;

namespace {
o::Program sharedWords()
{
    auto p = base(1);
    p.operations = {op(o::Pipe::MTE2, {{0, false, true}}), op(o::Pipe::V, {{0, true, false}})};
    o::ObservedControl graph;
    graph.qualification = "equivalent child paths with shared original command words";
    graph.scopes = {{0, o::NoControlId, o::NoControlId}};
    graph.entry = 0;
    graph.exit = 7;
    graph.sites.resize(8);
    const std::vector<std::vector<std::size_t>> edges{{1, 4}, {2}, {3}, {7}, {5}, {6}, {7}, {}};
    for (std::size_t at = 0; at < graph.sites.size(); ++at) {
        graph.sites[at].successors = edges[at];
        graph.sites[at].observation = at;
        graph.observations.push_back({at, {}, true});
    }
    for (std::size_t at = 4; at < 7; ++at) {
        graph.sites[at].observation = at - 3;
    }
    graph.sites[1].operation = graph.sites[4].operation = 0;
    graph.sites[3].operation = graph.sites[6].operation = 1;
    p.observed = std::move(graph);
    return p;
}

void correspondence()
{
    auto p = sharedWords();
    o::selected::Control control(p);
    require(control.complete, control.reason);
    const auto& relation = control.correspondence(1, 3);
    require(relation.proved() && relation.pairs == std::vector<std::pair<o::Cut, o::Cut>>{{1, 3}, {4, 6}},
            "shared command words lost a participating child occurrence");
    const auto work = control.occurrenceAnalysisSites;
    require(&relation == &control.correspondence(4, 6) && work == control.occurrenceAnalysisSites,
            "equivalent word query recomputed immutable correspondence");
    const auto& limited = control.correspondence(1, 3, 1);
    require(limited.outcome == o::selected::ProofOutcome::Unknown && limited.pairs.empty(),
            "budget exhaustion granted partial occurrence credit or disproved feasibility");
    require(control.correspondence(99, 3).outcome == o::selected::ProofOutcome::Unknown,
            "invalid boundary was treated as a proof");
    p.observed->sites[0].successors = {1, 5};
    o::selected::Control bypass(p);
    require(bypass.correspondence(1, 3).outcome == o::selected::ProofOutcome::Disproved,
            "acquisition bypass without a publication was accepted");
    p = sharedWords();
    p.observed->sites[1].successors = {4};
    o::selected::Control repeated(p);
    require(repeated.correspondence(1, 3).outcome == o::selected::ProofOutcome::Disproved,
            "second publication before consumption was accepted");
    p = sharedWords();
    p.observed->sites[2].successors = {7};
    o::selected::Control missing(p);
    require(missing.correspondence(1, 3).outcome == o::selected::ProofOutcome::Disproved,
            "unconsumed publication at region continuation was accepted");
}

void physicalDeadlines()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE1;
    auto p = base(2);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                    op(R, {{0, true, false}}), op(P, {{0, false, true, true}}),
                    op(Q, {{0, true, false}}), op(P, {{1, false, true, true}})};
    o::selected::Control control(p);
    o::StorageFrontierAnalysis storage(p);
    o::selected::RequirementFrontiers facts(p, control, storage);
    require(facts.complete(), facts.reason());
    const auto& producer = facts.use(0, 0);
    require(producer.roles == 2 && producer.release == 1 && producer.origin.site == 0,
            "physical use lost its original release boundary");
    require(facts.use(1, 0).returns == std::vector<o::Cut>{3} &&
                facts.use(2, 0).returns == std::vector<o::Cut>{3},
            "distinct readers lost the required storage-return deadline");
    const auto demands = facts.demandsAt(1, {{0, P, true, 1, true, false}});
    require(demands.size() == 1 && demands[0].release == 1 && demands[0].deadline == 1 &&
                demands[0].returnDeadlines == std::vector<o::Cut>{3},
            "readiness and possible return were conflated or lost");
    const auto& later = facts.at(4);
    require(later.size() == 1 && later[0].relationship.source.site == 3,
            "reload reused the old physical generation's readiness provenance");
    require(facts.use(5, 0).roles == 0, "unrelated storage became part of the lifetime");
    const auto plan = accepted(p);
    require(!plan.decisions.empty() && !plan.decisions.front().lifecycles.empty(),
            "ordinary binding discarded shared physical deadline facts");
}
} // namespace

namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static void everyInterveningOccurrence()
    {
        auto p = sharedWords();
        Constructor constructor(p);
        Id key = NoAnalysisId;
        for (Id i = 0; i < constructor.frontier.keys().size(); ++i) {
            const auto& identity = constructor.frontier.keys()[i];
            if (identity.source == Pipe::MTE2 && identity.observer == Pipe::V) {
                key = i;
                break;
            }
        }
        require(key != NoAnalysisId, "fixture has no eligible key");
        require(constructor.clearInterval(key, 4, 6), "paired empty intervals should be clear");
        constructor.ledger.append(2, {Command::Publish, Pipe::MTE2, Pipe::V, 0}, EndpointPurpose::Fixed);
        require(!constructor.clearInterval(key, 4, 6),
                "binding ignored the noncanonical occurrence of an intervening key use");
    }
};
}
int main()
{
    correspondence();
    physicalDeadlines();
    o::selected::ReplayTestAccess::everyInterveningOccurrence();
    std::cout << "shared occurrence and physical-deadline queries passed\n";
}
