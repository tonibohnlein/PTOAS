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

void composedPhysicalUses()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V;
    auto p = base(2);
    p.operations = {op(P, {{0, false, true, true}}), op(Q, {{0, true, false}}),
                    op(Q, {{0, true, false}}), op(P, {{0, false, true, true}}),
                    op(Q, {{0, true, false}}), op(P, {{1, false, true, true}})};
    p.body = seq({leaf(0), {o::Region::For, {leaf(1)}, 0, true},
                  {o::Region::For, {leaf(2)}, 0, true}, leaf(3), leaf(4), leaf(5)});
    o::selected::Control control(p);
    o::StorageFrontierAnalysis storage(p);
    require(control.complete && storage.complete(), "invalid physical-use fixture");
    const auto reader = storage.sitesForOperation(1).front();
    const auto nextReader = storage.sitesForOperation(2).front();
    auto contains = [](const o::PhysicalUseFrontier& frontier, o::Cut site) {
        return std::any_of(frontier.accesses.begin(), frontier.accesses.end(),
                           [&](const auto& origin) { return origin.site == site; });
    };
    const auto& next = storage.nearestUses(control.graph.sites[reader].successors, 0);
    require(next.complete && contains(next, reader) && contains(next, nextReader),
            "physical uses lost repeated or following child participation");
    const auto work = storage.stats().nearestUseEvaluations;
    require(&next == &storage.nearestUses(control.graph.sites[reader].successors, 0) &&
                storage.stats().nearestUseEvaluations == work,
            "immutable physical-use query repeated its graph traversal");
    const auto reload = storage.sitesForOperation(3).front();
    const auto& afterChild = storage.nearestUses(control.graph.sites[nextReader].successors, 0);
    require(contains(afterChild, reload), "next overwrite disappeared behind the lexical child boundary");
    const auto& afterReload = storage.nearestUses(control.graph.sites[reload].successors, 0);
    require(contains(afterReload, storage.sitesForOperation(4).front()) && !contains(afterReload, reader),
            "reload inherited the earlier generation's reader occurrence");
    const auto outside = storage.sitesForOperation(4).front();
    const auto& tail = storage.nearestUses(control.graph.sites[outside].successors, 0);
    require(tail.complete && tail.accesses.empty() && tail.boundaries == std::vector<o::Cut>{control.graph.exit},
            "unrelated storage obscured the open final-use obligation");
    const auto& stopped = storage.nearestUses(std::vector<o::Cut>{nextReader}, 0,
                                            std::vector<o::Cut>{nextReader});
    require(stopped.accesses.empty() && stopped.boundaries == std::vector<o::Cut>{nextReader},
            "open boundary was implicitly consumed as a storage access");
    require(!storage.nearestUses(std::vector<o::Cut>{control.graph.sites.size()}, 0).complete,
            "invalid use query returned a complete empty interface");
    const auto& previous = storage.nearestUses(control.predecessors[outside], 0, {}, true);
    require(previous.complete && contains(previous, reload), "backward use query crossed a reload");
    for (bool readModifyWrite : {false, true}) {
        auto partial = p;
        partial.operations[3].accesses = {{0, readModifyWrite, true, false}};
        o::StorageFrontierAnalysis uncertain(partial);
        const auto& front = uncertain.nearestUses(control.graph.sites[nextReader].successors, 0);
        require(front.complete && contains(front, reload) && !contains(front, outside),
                "may-write or RMW was skipped as if it established no physical obligation");
        const auto oldWriters = uncertain.previousWriters(outside, 0);
        require(std::any_of(oldWriters.begin(), oldWriters.end(), [](const auto& origin) {
                    return origin.operation == 0;
                }), "nearest-use query converted a partial write into a full-generation kill");
    }

}

void finiteModeTransitions()
{
    auto p = base(1);
    p.operations = {op(o::Pipe::MTE2, {{0, false, true}}), op(o::Pipe::V, {{0, true, false}})};
    o::ObservedControl graph;
    graph.qualification = "finite child modes inside an original outer recurrence";
    graph.entry = 0;
    graph.exit = 9;
    graph.scopes = {{o::AnalysisContext::Function, o::NoControlId, o::NoControlId},
                    {o::AnalysisContext::ForBody, 0, 1}, {o::AnalysisContext::ForBody, 1, 3}};
    const std::vector<std::vector<o::Cut>> edges{{1}, {2}, {3, 9}, {4}, {5}, {6}, {7}, {8}, {10}, {}, {2}};
    graph.sites.resize(edges.size());
    for (o::Cut at = 0; at < edges.size(); ++at) {
        auto& site = graph.sites[at];
        site.successors = edges[at];
        site.backedgeOwners.assign(edges[at].size(), o::NoControlId);
        site.observation = at;
        site.context = at >= 3 && at <= 8 ? 2 : (at == 2 || at == 10 ? 1 : 0);
        graph.observations.push_back({at, {}, true});
    }
    graph.sites[5].backedgeOwners[0] = graph.sites[7].backedgeOwners[0] = 3;
    graph.sites[10].backedgeOwners[0] = 1;
    graph.sites[5].operation = 0;
    graph.sites[7].operation = 1;
    p.observed = std::move(graph);
    o::selected::Control control(p);
    require(control.complete, control.reason);
    require(control.unsummarizedBackedges == 0 && control.finiteOccurrenceTransitions == 2,
            "finite occurrence transitions became natural loops or forced whole-graph replay");
    require(control.constructionEdges[5] == std::vector<o::Cut>{6} &&
                control.constructionEdges[7] == std::vector<o::Cut>{8},
            "finite occurrence transition lost its actual predecessor");
    require(control.headerAccesses[6].empty() && control.headerAccesses[8].empty(),
            "finite transition manufactured a previous-iteration history");
    o::selected::Constructor constructor(p);
    auto plan = constructor.run({}, false);
    require(plan.success, plan.reason);
    require(plan.channels.empty() && plan.work.contextualReplays == 0,
            "finite occurrence transitions depended on recurring admission or full replay");
    require(o::checkCausalFrontier(p, plan.commands).accepted,
            "finite-mode construction lost an original physical obligation");
}

void composedEndpointDemands()
{
    const auto P = o::Pipe::MTE2, Q = o::Pipe::V;
    auto p = base(2);
    p.operations = {op(P, {{0, false, true}, {1, true, false}}),
                    op(Q, {{0, true, false}, {1, false, true}}),
                    op(P, {{0, false, true}, {1, true, false}})};
    p.body = seq({leaf(0), {o::Region::For, {leaf(1)}}, leaf(2)});
    auto imported = o::addStructuredBoundaryCuts(p);
    require(imported.success, imported.reason);
    o::selected::Control control(imported.program);
    o::StorageFrontierAnalysis storage(imported.program);
    o::selected::RequirementFrontiers facts(imported.program, control, storage);
    o::Cut owner = o::NoAnalysisId;
    for (const auto& scope : control.graph.contexts) {
        if (scope.kind == o::AnalysisContext::ForBody) {
            owner = scope.ownerSite;
        }
    }
    require(owner != o::NoAnalysisId, "endpoint fixture lost its original reader owner");
    const auto& demands = facts.endpoints(owner);
    for (auto role : {o::selected::EndpointRequirement::FirstConsumer,
                      o::selected::EndpointRequirement::FirstWrite,
                      o::selected::EndpointRequirement::FinalReader}) {
        require(std::any_of(demands.begin(), demands.end(), [&](const auto& demand) {
                    return demand.role == role && demand.owner == owner && demand.access.operation == 1;
                }), "one child role silently discarded another role at the same original access");
    }
    require(facts.needsOccurrenceSeparation(owner), "mixed first/continuing roles lost refinement demand");
    require(facts.endpoints(o::NoAnalysisId).empty(), "unknown owner manufactured endpoint requests");
    auto rmw = base(1);
    rmw.operations = {op(P, {{0, true, true}}), op(Q, {{0, true, false}})};
    rmw.body = seq({leaf(0), {o::Region::For, {leaf(1)}}});
    auto rmwImported = o::addStructuredBoundaryCuts(rmw);
    require(rmwImported.success, rmwImported.reason);
    o::selected::Control rmwControl(rmwImported.program);
    o::StorageFrontierAnalysis rmwStorage(rmwImported.program);
    o::selected::RequirementFrontiers rmwFacts(rmwImported.program, rmwControl, rmwStorage);
    require(rmwFacts.needsOccurrenceSeparation(owner),
            "outside RMW suppressed a reader child's first/continuing demand");
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}})};
    p.operations.push_back(p.operations[0]);
    p.operations.push_back(p.operations[1]);
    p.body = seq({leaf(0), leaf(1), {o::Region::For, {seq({leaf(2), leaf(3)})}}});
    imported = o::addStructuredBoundaryCuts(p);
    require(imported.success, imported.reason);
    o::selected::Control uniformControl(imported.program);
    o::StorageFrontierAnalysis uniformStorage(imported.program);
    o::selected::RequirementFrontiers uniform(imported.program, uniformControl, uniformStorage);
    for (const auto& scope : uniformControl.graph.contexts) {
        if (scope.kind == o::AnalysisContext::ForBody) {
            require(!uniform.endpoints(scope.ownerSite).empty(), "uniform fixture has no endpoint demands");
            require(!uniform.needsOccurrenceSeparation(scope.ownerSite),
                    "uniform roles requested unnecessary control copies");
        }
    }
}

void mixedReaderParticipation()
{
    for (bool before : {false, true}) {
        auto p = base(1);
        p.operations = {op(o::Pipe::MTE2, {{0, false, true}}),
                        op(o::Pipe::V, {{0, true, false}}),
                        op(o::Pipe::V, {{0, true, false}}),
                        op(o::Pipe::MTE2, {{0, false, true}})};
        const o::Region choice{o::Region::Choice, {leaf(2), seq({})}};
        p.body = before ? seq({leaf(0), choice, leaf(1), leaf(3)})
                        : seq({leaf(0), leaf(1), choice, leaf(3)});
        o::selected::Control control(p);
        o::StorageFrontierAnalysis storage(p);
        o::selected::RequirementFrontiers facts(p, control, storage);
        const auto sites = storage.sitesForOperation(1);
        require(sites.size() == 1, "mixed participation fixture needs one queried reader");
        require(facts.readerParticipation(sites.front(), 0).outcome == o::selected::ProofOutcome::Unknown,
                before ? "mixed first/continuing participation was proved"
                       : "mixed final/continuing participation was proved");
    }
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
    const auto& first = facts.readerParticipation(1, 0);
    const auto& last = facts.readerParticipation(2, 0);
    const auto& reloaded = facts.readerParticipation(4, 0);
    require(first.proved() && first.first && !first.final && last.proved() && !last.first && last.final,
            "different reader owners did not retain one write episode");
    require(reloaded.proved() && reloaded.first && reloaded.final &&
                reloaded.preceding.size() == 1 && reloaded.preceding.front().site == 3,
            "reload did not start a separate reader episode");
    const auto queryWork = storage.stats().nearestUseEvaluations;
    require(&facts.readerParticipation(1, 0) == &first && storage.stats().nearestUseEvaluations == queryWork,
            "immutable reader participation was recomputed");
    auto uncertain = p;
    uncertain.operations[3].accesses = {{0, true, true, false}};
    o::selected::Control uncertainControl(uncertain);
    o::StorageFrontierAnalysis uncertainStorage(uncertain);
    o::selected::RequirementFrontiers uncertainFacts(uncertain, uncertainControl, uncertainStorage);
    require(!uncertainFacts.readerParticipation(4, 0).proved(),
            "RMW was treated as a fresh reader episode");
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
    static void omittedSupport()
    {
        const auto P = Pipe::MTE2, Q = Pipe::V, R = Pipe::MTE1;
        auto p = base(2, 4);
        p.operations = {op(P, {{1, false, true}}), op(P, {{0, false, true}}),
                        op(P, {{1, false, true}}), op(Q, {{0, true, false}})};
        Constructor constructor(p);
        std::string reason;
        require(constructor.ledger.initialize({}, reason), reason);
        RecurringRequirement direct;
        direct.source = P;
        direct.observer = Q;
        direct.publications = {2};
        direct.acquisitions = {3};
        direct.supportSeeds = {1};
        auto first = direct;
        first.observer = R;
        first.acquisitions = {2};
        first.supportSeeds.clear();
        first.qualifiedCycle = true;
        auto second = first;
        second.source = R;
        second.observer = Q;
        second.acquisitions = {3};
        require(!constructor.recurring({direct, first, second}) &&
                    constructor.result.reason.find("producer repair") != std::string::npos,
                "omitting a private channel dropped the proposal's residual-support obligation");
        require(constructor.result.work.redundantRecurringChannels == 1 && constructor.ledger.records().empty(),
                "support witness did not omit its private channel atomically");
    }
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
    composedPhysicalUses();
    finiteModeTransitions();
    composedEndpointDemands();
    mixedReaderParticipation();
    physicalDeadlines();
    o::selected::ReplayTestAccess::everyInterveningOccurrence();
    o::selected::ReplayTestAccess::omittedSupport();
    std::cout << "shared occurrence and physical-deadline queries passed\n";
}
