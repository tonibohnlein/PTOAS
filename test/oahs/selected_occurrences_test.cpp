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

void alternativeEndpoints()
{
    auto p = sharedWords();
    // Distinct source/receipt words on two exclusive paths describe the same
    // logical obligation without requiring one canonical representative.
    for (o::Cut at = 4; at < 7; ++at) {
        p.observed->sites[at].observation = at;
    }
    o::selected::Control control(p);
    const auto& relation = control.correspondence(std::vector<o::Cut>{1, 4}, std::vector<o::Cut>{3, 6});
    require(relation.proved() && relation.pairs == std::vector<std::pair<o::Cut, o::Cut>>{{1, 3}, {4, 6}},
            "alternative endpoint sets lost their original correspondence");
    const auto work = control.occurrenceAnalysisSites;
    require(&relation == &control.correspondence(std::vector<o::Cut>{4, 1, 4}, std::vector<o::Cut>{6, 3}) &&
                control.occurrenceAnalysisSites == work,
            "endpoint enumeration order changed the immutable relation");
    require(!control.correspondence(std::vector<o::Cut>{1}, std::vector<o::Cut>{3, 6}).proved(),
            "one alternative publication supplied credit on the other path");
    require(!control.correspondence(std::vector<o::Cut>{1, 4}, std::vector<o::Cut>{3}).proved(),
            "unconsumed alternative publication was accepted");
    require(control.correspondence(std::vector<o::Cut>{}, std::vector<o::Cut>{3}).outcome ==
                o::selected::ProofOutcome::Unknown,
            "empty endpoint set was accepted");
    const auto& limited = control.correspondence(std::vector<o::Cut>{1, 4}, std::vector<o::Cut>{3, 6}, 1);
    require(limited.outcome == o::selected::ProofOutcome::Unknown && limited.pairs.empty(),
            "set matching granted partial correspondence after budget exhaustion");
}

// Independent empty/full execution oracle. Unlike the retired placement API,
// a same-word publication precedes its acquisition, and every terminal is
// checked. Matching is not a proof that an invocation terminates.
bool concreteBalance(const std::vector<std::vector<o::Cut>>& graph,
                     const std::vector<o::Cut>& publications, o::Cut acquisition)
{
    std::vector<bool> publishes(graph.size());
    for (auto cut : publications) {
        publishes[cut] = true;
    }
    std::vector<std::array<bool, 2>> seen(graph.size());
    std::vector<std::pair<o::Cut, bool>> todo{{0, false}};
    bool received = false;
    while (!todo.empty()) {
        auto [site, full] = todo.back();
        todo.pop_back();
        if (seen[site][full]) {
            continue;
        }
        seen[site][full] = true;
        if (publishes[site]) {
            if (full) {
                return false;
            }
            full = true;
        }
        if (site == acquisition) {
            if (!full) {
                return false;
            }
            full = false;
            received = true;
        }
        if (graph[site].empty() && full) {
            return false;
        }
        for (auto next : graph[site]) {
            todo.emplace_back(next, full);
        }
    }
    return received;
}

void exhaustiveMatching()
{
    std::size_t cases = 0;
    for (unsigned n = 1; n <= 4; ++n) {
        for (uint64_t mask = 0; mask < (uint64_t(1) << ((n - 1) * n)); ++mask) {
            std::vector<std::vector<o::Cut>> graph(n);
            for (unsigned a = 0; a + 1 < n; ++a) {
                for (unsigned b = 0; b < n; ++b) {
                    if (mask & (uint64_t(1) << (a * n + b))) {
                        graph[a].push_back(b);
                    }
                }
            }
            // Isolate the matching query on a bounded graph. This deliberately
            // does not claim admission by the structured-control importer.
            o::selected::Control control(base(1));
            control.graph.entry = 0;
            control.graph.exit = n - 1;
            control.graph.sites.resize(n);
            control.canonicalCut.resize(n);
            for (unsigned at = 0; at < n; ++at) {
                control.graph.sites[at].successors = graph[at];
                control.canonicalCut[at] = at;
            }
            for (unsigned bits = 1; bits < (1u << n); ++bits) {
                std::vector<o::Cut> publications;
                for (unsigned at = 0; at < n; ++at) {
                    if (bits & (1u << at)) {
                        publications.push_back(at);
                    }
                }
                for (unsigned receipt = 0; receipt < n; ++receipt) {
                    const auto& relation = control.correspondence(publications, std::vector<o::Cut>{receipt});
                    require(relation.proved() == concreteBalance(graph, publications, receipt),
                            "shared occurrence matching differs from concrete token execution");
                    ++cases;
                }
            }
        }
    }
    std::cout << "independent matching cases=" << cases << '\n';
}

void joinedSources()
{
    constexpr o::Cut alternatives = 32, suffix = 40, receipt = alternatives + suffix + 1;
    o::selected::Control control(base(1));
    control.graph.entry = 0;
    control.graph.exit = receipt + 1;
    control.graph.sites.resize(receipt + 2);
    control.canonicalCut.resize(receipt + 2);
    std::vector<o::Cut> sources;
    for (o::Cut at = 0; at <= receipt + 1; ++at) {
        control.canonicalCut[at] = at;
        control.graph.sites[at].successors.clear();
        if (at > 0 && at <= alternatives) {
            sources.push_back(at);
            control.graph.sites[0].successors.push_back(at);
            control.graph.sites[at].successors = {alternatives + 1};
        } else if (at > alternatives && at <= receipt) {
            control.graph.sites[at].successors = {at + 1};
        }
    }
    const auto& matched = control.correspondence(sources, std::vector<o::Cut>{receipt});
    require(matched.proved() && matched.pairs.size() == alternatives,
            "shared suffix lost alternative source identities");
    require(matched.siteEvaluations == 2 + alternatives * (suffix + 2),
            "shared suffix occurrence work was not charged by source identity");
    const auto& limited = control.correspondence(sources, std::vector<o::Cut>{receipt}, 64);
    require(limited.outcome == o::selected::ProofOutcome::Unknown && limited.pairs.empty(),
            "shared suffix work limit changed matching into partial credit");
}

void publicationBoundaries()
{
    auto p = sharedWords();
    o::selected::Control control(p);
    const auto& boundary = control.publicationAfter(1);
    require(boundary.proved() && boundary.word == 2,
            "equivalent payload copies lost the first shared release boundary");
    const auto work = control.boundaryAnalysisSites;
    require(&boundary == &control.publicationAfter(4) && work == control.boundaryAnalysisSites,
            "shared payload copies recomputed the release boundary");
    o::StorageFrontierAnalysis storage(p);
    o::selected::RequirementFrontiers facts(p, control, storage);
    require(facts.recurringRelease(1, 0) == 2,
            "publication release still requires a single-owner occurrence grammar");
    require(!control.publicationAfter(99).proved(), "invalid source gained a boundary");
    p.observed->sites[0].successors.push_back(5);
    o::selected::Control bypass(p);
    require(!bypass.publicationAfter(1).proved(),
            "release boundary can execute without its payload occurrence");
    p = sharedWords();
    p.observed->sites[5].observation = 5;
    o::selected::Control split(p);
    require(!split.publicationAfter(1).proved(),
            "distinct release words were silently merged");
    p = sharedWords();
    p.observed->sites[2].observation = p.observed->sites[1].observation;
    p.observed->sites[5].observation = p.observed->sites[4].observation;
    o::selected::Control sameWord(p);
    require(!sameWord.publicationAfter(1).proved(),
            "before-payload word became its own after-payload release");
    p = sharedWords();
    // The first path has an unobservable intervening payload. It cannot be
    // crossed merely because the second path still has its early boundary.
    p.observed->sites[2].observation = o::NoAnalysisId;
    p.observed->sites[2].operation = 1;
    o::selected::Control crossed(p);
    require(!crossed.publicationAfter(1).proved(), "release crossed an intervening payload");
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
    alternativeEndpoints();
    exhaustiveMatching();
    joinedSources();
    publicationBoundaries();
    physicalDeadlines();
    o::selected::ReplayTestAccess::everyInterveningOccurrence();
    std::cout << "shared occurrence and physical-deadline queries passed\n";
}
