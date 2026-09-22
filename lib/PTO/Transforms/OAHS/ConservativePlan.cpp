// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Conservative construction uses actual target events, not synthetic ALL credit.
#include "Control.h"
#include "PTO/Transforms/OAHS/SelectedPlan.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <set>

namespace mlir::pto::oahs {
namespace {
using Id = std::size_t;
struct Routes {
    std::array<std::array<std::vector<Id>, PipeCount>, PipeCount> paths;
    explicit Routes(const std::vector<EventIdentity>& keys)
    {
        for (unsigned source = 0; source < PipeCount; ++source) {
            std::array<bool, PipeCount> seen{};
            std::vector<unsigned> todo{source};
            seen[source] = true;
            for (Id i = 0; i < todo.size(); ++i) {
                const auto from = todo[i];
                for (Id key = 0; key < keys.size(); ++key) {
                    const auto next = unsigned(keys[key].observer);
                    if (unsigned(keys[key].source) != from || seen[next]) {
                        continue;
                    }
                    seen[next] = true;
                    paths[source][next] = paths[source][from];
                    paths[source][next].push_back(key);
                    todo.push_back(next);
                }
            }
        }
    }
    bool connected(unsigned a, unsigned b) const
    {
        return a == b || (!paths[a][b].empty() && !paths[b][a].empty());
    }
};

void appendTour(std::vector<Command>& packet, const std::vector<Id>& tour,
                const std::vector<EventIdentity>& keys)
{
    for (auto key : tour) {
        const auto& event = keys[key];
        packet.push_back({Command::Publish, event.source, event.observer, event.key});
        packet.push_back({Command::Acquire, event.source, event.observer, event.key});
    }
}

bool serializeComponent(const Program& program, const std::vector<unsigned>& members,
                        const Routes& routes, const std::vector<EventIdentity>& keys,
                        std::vector<Command>& packet)
{
    auto visits = members;
    if (visits.size() == 1) {
        const auto pipe = members.front();
        if (program.target.synchronous[pipe]) {
            return true; // completion remains local; no cross-pipeline credit
        }
        if (program.target.barriers[pipe]) {
            packet.push_back({Command::Barrier, Pipe(pipe), Pipe(pipe), 0});
            return true;
        }
        for (unsigned other = 0; other < PipeCount; ++other) {
            if (other != pipe && routes.connected(pipe, other)) {
                visits.push_back(other);
                break;
            }
        }
        if (visits.size() == 1) {
            return false;
        }
    }
    std::vector<Id> tour;
    auto current = visits.front();
    for (Id i = 1; i <= visits.size(); ++i) {
        const auto next = visits[i % visits.size()];
        const auto& leg = routes.paths[current][next];
        tour.insert(tour.end(), leg.begin(), leg.end());
        current = next;
    }
    // The first closed tour gathers every participating prefix at its root.
    // The second distributes that prefix to all members. Each WAIT propagates
    // its consumption through the following walk before that key is reused.
    appendTour(packet, tour, keys);
    appendTour(packet, tour, keys);
    return true;
}

bool serializationPacket(const Program& program, const detail::ControlGraph& graph,
                         const std::vector<bool>& reachable, const CausalFrontier& frontier,
                         std::vector<Command>& packet)
{
    std::set<unsigned> used;
    for (Cut cut = 0; cut < graph.sites.size(); ++cut) {
        const auto op = graph.operations[cut];
        if (reachable[cut] && op != NoAnalysisId) {
            used.insert(unsigned(program.operations[op].pipe));
        }
    }
    const Routes routes(frontier.keys());
    while (!used.empty()) {
        const auto root = *used.begin();
        std::vector<unsigned> members;
        for (auto at = used.begin(); at != used.end();) {
            if (routes.connected(root, *at)) {
                members.push_back(*at);
                at = used.erase(at);
            } else {
                ++at;
            }
        }
        if (!serializeComponent(program, members, routes, frontier.keys(), packet)) {
            return false;
        }
    }
    return true;
}

bool materialize(const Program& program, const detail::ControlGraph& graph,
                 const std::vector<bool>& reachable, const std::vector<Command>& packet,
                 Commands& commands)
{
    std::set<Cut> words;
    for (Cut cut = 0; cut < graph.sites.size(); ++cut) {
        if (!reachable[cut] || graph.operations[cut] == NoAnalysisId) {
            continue;
        }
        if (!legalCommandCut(program, cut)) {
            return false;
        }
        words.insert(canonicalCommandCut(program, cut));
    }
    commands.resize(commandCutCount(program));
    for (Cut cut = 0; cut < commands.size(); ++cut) {
        if (words.count(canonicalCommandCut(program, cut))) {
            commands[cut] = packet;
        }
    }
    if (program.invocation.retirement == Program::InvocationContract::DrainAllAtReturn) {
        commands[invocationExitCut(program)].push_back({Command::BarrierAll});
    }
    return true;
}
} // namespace

SelectedPlan constructConservativePlan(const Program& program)
{
    SelectedPlan result;
    result.conservative = true;
    const auto start = std::chrono::steady_clock::now();
    if (program.invocation.authoredSynchronization || program.invocation.externalProgress) {
        result.failure = SelectedFailure::UnsupportedContract;
        result.reason = program.invocation.authoredSynchronization ?
            "conservative construction requires authored protocol import" :
            "conservative serialization requires an external-protocol progress certificate";
        return result;
    }
    if (!program.invocation.localProgressGap.empty()) {
        result.failure = SelectedFailure::UnsupportedContract;
        result.reason = "conservative local progress unqualified: " + program.invocation.localProgressGap;
        return result;
    }
    const CausalFrontier frontier(program);
    if (!frontier.complete()) {
        result.failure = SelectedFailure::UnsupportedContract;
        result.reason = frontier.reason();
        return result;
    }
    const auto graph = detail::buildControlGraph(program);
    const auto reachable = detail::reachableSites(graph);
    std::vector<Command> packet;
    if (!serializationPacket(program, graph, reachable, frontier, packet)) {
        result.failure = SelectedFailure::UnsupportedContract;
        result.reason = "conservative construction requires a supported local completion primitive";
        return result;
    }
    Commands commands;
    if (!materialize(program, graph, reachable, packet, commands)) {
        result.failure = SelectedFailure::UnqualifiedControl;
        result.reason = "conservative construction requires a legal original payload gap";
        return result;
    }
    result.work.constructedSites = graph.sites.size();
    result.work.cells = program.cells.size();
    result.work.eligibleKeys = frontier.keys().size();
    result.certificate = checkCausalFrontier(program, commands);
    result.work.finalCertificateSiteEvaluations = result.certificate.siteEvaluations;
    result.work.elapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();
    if (!result.certificate.accepted) {
        result.failure = SelectedFailure::FinalValidation;
        result.reason = "conservative protocol: " + result.certificate.reason;
        result.cut = result.certificate.cut;
        result.certificate = {};
        return result;
    }
    for (Cut cut = 0; cut < commands.size(); ++cut) {
        if (canonicalCommandCut(program, cut) != cut) {
            continue;
        }
        result.work.commandWords += !commands[cut].empty();
        for (const auto& command : commands[cut]) {
            result.ledger.push_back({result.ledger.size(), cut, command, EndpointPurpose::Completion});
        }
    }
    result.commands = std::move(commands);
    result.success = true;
    return result;
}
} // namespace mlir::pto::oahs
