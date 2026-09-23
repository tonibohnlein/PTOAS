// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SelectedTestSupport.h"
#include "../../lib/PTO/Transforms/OAHS/SelectedInternal.h"
#include <functional>
using namespace selected_test;
namespace mlir::pto::oahs::selected {
struct ReplayTestAccess {
    static void ordinaryFenceProof(unsigned mutation)
    {
        const auto P = Pipe::MTE2;
        auto p = base(2, 1);
        p.operations = {op(P, {{1, false, true}}), op(P, {{0, false, true}}),
                        op(P, {{1, false, true}})};
        if (mutation == 5) { p.target.barriers[unsigned(P)] = false; }
        if (mutation == 6) { p.operations[1] = op(Pipe::V, {{0, false, true}}); }
        if (mutation == 3) { p.operations.push_back(op(P, {{1, false, true}})); }
        if (mutation == 7) { p.operations.push_back(op(Pipe::V, {{0, false, true}})); }
        if (mutation == 3 || mutation == 7) {
            p.body = seq({leaf(0), leaf(1), leaf(3), leaf(2)});
        } else if (mutation == 4) {
            p.body = seq({leaf(0), {Region::Choice, {leaf(1), seq({})}}, leaf(2)});
        } else {
            p.body = seq({leaf(0), leaf(1), leaf(2)});
        }
        auto imported = addStructuredBoundaryCuts(p);
        require(imported.success, imported.reason);
        Constructor c(imported.program);
        auto site = [&](unsigned operation) {
            for (Cut cut = 0; cut < c.control.graph.operations.size(); ++cut) {
                if (c.control.graph.operations[cut] == operation) { return cut; }
            }
            return NoAnalysisId;
        };
        const auto seed = site(1), deadline = site(2);
        require(seed != NoAnalysisId && deadline != NoAnalysisId, "ordinary support fixture lost sites");
        ProducerSupportScope scope;
        scope.seeds[unsigned(P)].push_back(c.control.canonicalCut[seed]);
        const auto access = (Id(1) * PipeCount + unsigned(P)) * 2 + 1;
        std::vector<Cut> seeds{c.control.canonicalCut[seed]};
        if (mutation == 7) { seeds.push_back(c.control.canonicalCut[site(3)]); }
        scope.ordinary.push_back({deadline, access, P,
            std::make_shared<const std::vector<Cut>>(seeds)});
        OrderedPacket packet;
        if (mutation != 1) {
            const auto position = mutation == 2 ? deadline : seed;
            packet.push_back({position, {Command::Barrier, P, Pipe::S, 0}, EndpointPurpose::LocalFence});
        }
        const auto prepared = c.ledger.preparePacket(packet);
        require(prepared.valid(), prepared.reason());
        const auto view = c.ledger.packetView(prepared);
        require(bool(view), "ordinary support packet view missing");
        const auto proved = c.ordinarySupportGuarantee(scope, *view);
        require(bool(proved) == (mutation == 0), "ordinary support accepted an unproved original path");
        if (mutation == 7) {
            require(c.ordinaryFenceSites.empty(), "failed seed interface was cached partially");
            require(!c.ordinarySupportGuarantee(scope, *view),
                    "second invalid seed query reused a partial interface");
        }
        require(c.ledger.records().empty(), "ordinary support probe selected a command");
    }
    static RecurringCertificate proof(const Program& p, unsigned mutation = 0)
    {
        Constructor c(p);
        c.recurringFrontiers = qualifyCyclicFrontiers(p, c.control, c.requirements);
        require(!c.recurringFrontiers.families.empty(), "fixture lost semantic family");
        const auto release = c.recurringFrontiers.families.front().roles.back();
        if (mutation == 1) { c.recurringFrontiers.roles[release].acquisitions.clear(); }
        if (mutation == 2) { c.recurringFrontiers.roles[release].publications.clear(); }
        const auto result = c.recurringCertificate(0);
        const auto sites = c.result.work.recurringInterfaceSites;
        const auto& repeated = c.recurringCertificate(0);
        require(repeated.complete == result.complete && c.result.work.recurringInterfaceSites == sites,
                "immutable proof repeated a graph solve");
        require(c.ledger.records().empty(), "analysis selected a packet or reserved an event");
        return result;
    }
    static void incidenceCost(const Program& p)
    {
        Constructor c(p);
        c.recurringFrontiers = qualifyCyclicFrontiers(p, c.control, c.requirements);
        for (Id family = 0; family < c.recurringFrontiers.families.size(); ++family) {
            require(c.recurringCertificate(family).complete, "subdivided family certificate failed");
        }
        const auto incidences = p.operations.front().accesses.size() + p.operations.back().accesses.size();
        require(c.result.work.recurringInterfaceAccesses == 2 * incidences,
                "family projection rescanned complete operation access vectors");
    }
    static void outsideKeyUse(const Program& p)
    {
        Constructor c(p);
        const auto plan = c.run({}, true);
        require(plan.success && c.activeRoles.size() == 2, "shared-role setup failed");
        c.result = plan; // run() exported the construction record; restore test-only inspection.
        const auto& channel = c.result.channels[c.activeRoles.begin()->second];
        const auto exit = c.control.graph.exit;
        c.ledger.append(exit, {Command::Publish, channel.source, channel.observer, channel.key},
                        EndpointPurpose::RecurringCompletion);
        RecurringPacket packet;
        const auto empty = c.prepareOwnedPacket({});
        require(bool(empty), "empty staged embedding setup failed");
        packet.packet = *empty;
        require(!c.qualifyRecurringInterface({0}, packet, {}),
                "extra active role-key use escaped the embedding check");
    }
};
}
namespace {
o::Program periodic(unsigned unrelated = 0)
{
    auto p = base(2, 3);
    for (unsigned i = 0; i < unrelated; ++i) { p.operations.push_back(op(o::Pipe::MTE3, {{1, false, true}})); }
    p.operations.push_back(op(o::Pipe::MTE2, {{0, false, true}}));
    p.operations.push_back(op(o::Pipe::V, {{0, true, false}}));
    const auto input = o::makePeriodicLoop(p, 1, {});
    require(input.success, input.reason);
    return input.program;
}
void outsideHistory()
{
    auto p = periodic();
    auto& graph = *p.observed;
    const auto originalEntry = graph.entry;
    const auto operation = p.operations.size();
    p.operations.push_back(op(o::Pipe::MTE1, {{0, false, true}}));
    const auto observation = graph.observations.size();
    graph.observations.push_back({100000, {}, true});
    graph.entry = graph.sites.size();
    graph.sites.push_back({operation, observation, {originalEntry}, {}, 0});
    const auto proof = o::selected::ReplayTestAccess::proof(p);
    require(proof.complete, "outside history must not erase independent event certificate");
    const auto outside = (std::size_t(0) * o::PipeCount + unsigned(o::Pipe::MTE1)) * 2 + 1;
    for (const auto& at : proof.guaranteed) {
        require(!at.second.count(outside), "partial write erased an older outside writer");
    }
    accepted(p); // Ordinary repair supplies the separate obligation.
}
void initialWriter()
{
    auto p = periodic();
    auto& graph = *p.observed;
    const auto entry = graph.entry, operation = p.operations.size();
    p.operations.push_back(op(o::Pipe::MTE2, {{0, false, true}}));
    const auto observation = graph.observations.size();
    graph.observations.push_back({100001, {}, true});
    graph.entry = graph.sites.size();
    graph.sites.push_back({operation, observation, {entry}, {}, 0});
    const auto proof = o::selected::ReplayTestAccess::proof(p);
    require(proof.complete, "initial writer lost event interface");
    bool checked = false;
    for (std::size_t site = 0; site < graph.sites.size(); ++site) {
        const auto op = graph.sites[site].operation;
        if (op == o::NoControlId || op == operation || p.operations[op].pipe != o::Pipe::MTE2) { continue; }
        const auto mode = o::selected::occurrenceMode(p, site);
        if (!mode.valid || mode.previous) { continue; }
        const auto found = proof.guaranteed.find(site);
        const auto write = unsigned(o::Pipe::MTE2) * 2 + 1;
        require(found == proof.guaranteed.end() || !found->second.count(write),
                "first-use capacity was confused with older writer completion");
        checked = true;
    }
    require(checked, "initial WAW test had no first occurrence");
    accepted(p);
}
void subdivision()
{
    for (unsigned cells : {1u, 8u, 32u}) {
        auto p = base(cells);
        p.operations = {op(o::Pipe::MTE2, {}), op(o::Pipe::V, {})};
        for (unsigned cell = 0; cell < cells; ++cell) {
            p.operations[0].accesses.push_back({cell, false, true});
            p.operations[1].accesses.push_back({cell, true, false});
        }
        const auto refined = o::makePeriodicLoop(p, 1, {});
        require(refined.success, refined.reason);
        o::selected::ReplayTestAccess::incidenceCost(refined.program);
        const auto plan = accepted(refined.program);
        require(plan.channels.size() == 2 && plan.activations.size() == 1,
                "uniform physical subdivision multiplied selected protocols");
    }
}
void inclusion(const o::Program& p)
{
    const auto proof = o::selected::ReplayTestAccess::proof(p);
    require(proof.complete, "paired recurring interface not proved: " + proof.reason);
    const auto plan = accepted(p);
    require(plan.work.recurringLocalPackets != 0 && plan.work.recurringAnalysisSites == 0 &&
            plan.work.recurringReplaySites == 0, "normal caller still tried complete candidate ledgers");
    for (const auto& at : proof.guaranteed) {
        const auto operation = o::operationAtCut(p, at.first);
        const auto observer = unsigned(p.operations[operation].pipe);
        const auto& state = plan.certificate.cuts[at.first].beforeIssue;
        for (auto access : at.second) {
            const auto* row = state.facts()->history.find(access);
            require(row && o::frontierContains(*row, observer), "conditional coverage exceeded actual selected credit");
        }
    }
}
}
int main()
{
    for (unsigned mutation = 0; mutation != 8; ++mutation) {
        o::selected::ReplayTestAccess::ordinaryFenceProof(mutation);
    }
    const auto p = periodic();
    inclusion(p);
    inclusion(periodic(8)); // Original operation IDs and unrelated storage both change.
    require(!o::selected::ReplayTestAccess::proof(p, 1).complete, "missing return consumption accepted");
    require(!o::selected::ReplayTestAccess::proof(p, 2).complete, "unmatched return receipt accepted");
    outsideHistory();
    initialWriter();
    subdivision();
    o::selected::ReplayTestAccess::outsideKeyUse(p);
}
