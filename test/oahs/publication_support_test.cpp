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
namespace s = mlir::pto::oahs::selected;
const auto P = o::Pipe::MTE2, Q = o::Pipe::V, R = o::Pipe::MTE1;

void exactGapsAndPropagation()
{
    auto p = base(3);
    p.operations = {op(Q, {{0, false, true}}), op(P, {{1, false, true}}),
                    op(P, {{2, false, true}}), op(R, {{2, true, false}})};
    s::Control control(p);
    o::CausalFrontier frontier(p);
    s::Ledger ledger(p, control.canonicalCut);
    s::PublicationSupport support(p, control, frontier);
    const auto early = ledger.append(2, {o::Command::Publish, P, R, 0}, o::EndpointPurpose::Completion);
    ledger.append(3, {o::Command::Acquire, P, R, 0}, o::EndpointPurpose::Completion);
    support.refresh(ledger);
    const auto baseline = support.records().front().admittedRoot;
    const auto revision = ledger.version();
    auto inspect = [&](const s::OrderedPacket& packet) {
        const auto prepared = ledger.preparePacket(packet);
        const auto view = ledger.packetView(prepared);
        require(bool(view), prepared.reason());
        return support.inspect(ledger, &*view, view->changedCuts());
    };
    const s::OrderedPacket after{
        {1, {o::Command::Publish, Q, P, 0}, o::EndpointPurpose::Completion},
        {2, {o::Command::Acquire, Q, P, 0}, o::EndpointPurpose::Completion}};
    require(inspect(after).preserved, "receipt after release must preserve its signature");
    auto before = after;
    before.back().gap = s::WordGap{2, o::NoAnalysisId, early};
    require(!inspect(before).preserved, "same-word receipt before release was ignored");
    const auto nodes = support.nodes;
    require(!inspect(before).preserved && support.nodes == nodes, "identical rejected probe grew its dependency DAG");
    before.back().cut = 1;
    before.back().gap.reset();
    require(!inspect(before).preserved, "earlier word failed to invalidate the publication");
    require(ledger.version() == revision && support.records().front().admittedRoot == baseline,
            "private probe mutated the ledger or admission root");
    const auto preparedProbe = inspect(before);
    const auto evaluatedSites = support.sites;
    const auto applied = ledger.appendPacket(ledger.preparePacket(before));
    require(applied.size() == before.size(), "commit fixture packet");
    support.accept(ledger, preparedProbe);
    require(support.sites == evaluatedSites, "commit re-evaluated its prepared symbolic delta");
    require(!support.records().front().preserved && support.records().front().admittedRoot == baseline,
            "changed source silently renewed its admission contract");
    // Erasure narrows the source back to the exact admitted expression.
    ledger.erase(applied[0]); ledger.erase(applied[1]);
    support.refresh(ledger);
    require(support.records().front().preserved, "erasure did not restore the original signature");
}

void unchangedReceipt()
{
    auto p = base(3);
    p.operations = {op(Q, {{0, false, true}}), op(P, {{1, false, true}}),
                    op(R, {{1, true, false}}), op(R, {{2, false, true}})};
    s::Control control(p);
    o::CausalFrontier frontier(p);
    s::Ledger ledger(p, control.canonicalCut);
    s::PublicationSupport support(p, control, frontier);
    ledger.append(2, {o::Command::Publish, P, R, 0}, o::EndpointPurpose::Completion);
    ledger.append(2, {o::Command::Acquire, P, R, 0}, o::EndpointPurpose::Completion);
    const auto downstream = ledger.append(4, {o::Command::Publish, R, Q, 0}, o::EndpointPurpose::Completion);
    support.refresh(ledger);
    ledger.append(1, {o::Command::Publish, Q, P, 0}, o::EndpointPurpose::Completion);
    ledger.append(1, {o::Command::Acquire, Q, P, 0}, o::EndpointPurpose::Completion);
    support.refresh(ledger);
    bool checked = false;
    for (const auto& contract : support.records()) {
        if (contract.publication == downstream) {
            checked = true;
            require(!contract.preserved, "changed SET did not invalidate unchanged downstream WAIT/source");
        }
    }
    require(checked, "downstream publication was not registered");
}

void restorationAndGenerations()
{
    auto p = base(2);
    p.operations = {op(Q, {{0, false, true}}), op(P, {{1, false, true}})};
    s::Control control(p);
    o::CausalFrontier frontier(p);
    s::Ledger ledger(p, control.canonicalCut);
    s::PublicationSupport support(p, control, frontier);
    const auto source = ledger.append(1, {o::Command::Publish, Q, P, 0}, o::EndpointPurpose::Completion);
    const auto receipt = ledger.append(1, {o::Command::Acquire, Q, P, 0}, o::EndpointPurpose::Completion);
    const auto publication = ledger.append(2, {o::Command::Publish, P, R, 0}, o::EndpointPurpose::Completion);
    support.refresh(ledger);
    const auto admitted = support.records().back().admittedRoot;
    ledger.erase(source); ledger.erase(receipt);
    support.refresh(ledger);
    const auto restoreSource = ledger.restoration(source, ledger.tail(1));
    const auto restoreReceipt = ledger.restoration(receipt, ledger.tail(1));
    require(bool(restoreSource) && bool(restoreReceipt), "restoration fixture identities");
    ledger.appendPacket(ledger.preparePacket({*restoreSource, *restoreReceipt}));
    support.refresh(ledger);
    require(support.records().back().preserved && support.records().back().admittedRoot == admitted,
            "exact restoration lost admission-rooted support");
    ledger.erase(source); ledger.erase(receipt);
    ledger.append(1, {o::Command::Publish, Q, P, 0}, o::EndpointPurpose::Completion);
    ledger.append(1, {o::Command::Acquire, Q, P, 0}, o::EndpointPurpose::Completion);
    support.refresh(ledger);
    for (const auto& contract : support.records()) {
        if (contract.publication == publication) {
            require(!contract.preserved, "new event generation inherited an old dependency identity");
        }
    }
}

void cyclicUnknown()
{
    auto p = base(1);
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}})};
    const auto observed = o::makePeriodicLoop(p, 1, {});
    require(observed.success, observed.reason);
    const auto plan = accepted(observed.program);
    s::Control control(observed.program);
    o::CausalFrontier frontier(observed.program);
    s::Ledger ledger(observed.program, control.canonicalCut);
    s::PublicationSupport support(observed.program, control, frontier);
    std::string reason;
    require(ledger.initialize(plan.commands, reason), reason);
    support.refresh(ledger);
    for (o::Cut site = 0; site < control.graph.sites.size(); ++site) {
        if (!control.reachable[site] || !control.components[control.component[site]].cyclic ||
            !o::legalCommandCut(observed.program, site)) { continue; }
        const auto packet = ledger.preparePacket({{site, {o::Command::Barrier, P, P, 0},
            o::EndpointPurpose::LocalFence}});
        const auto view = ledger.packetView(packet);
        require(bool(view) && !support.inspect(ledger, &*view, view->changedCuts()).preserved,
                "Unknown cyclic input compared equal and certified preservation");
        break;
    }
    require(!plan.channels.empty(), "recurring client did not activate");
    require(std::any_of(plan.publicationSupport.begin(), plan.publicationSupport.end(),
        [](const auto& c) { return c.admittedRoot == o::NoAnalysisId && !c.preserved; }),
        "unqualified recurrence obtained an exact publication certificate");
}

void choiceAndDirectFence()
{
    auto p = base(3);
    p.operations = {op(P, {{0, false, true}}), op(P, {{1, false, true}}), op(Q, {{2, false, true}})};
    p.body = seq({{o::Region::Choice, {leaf(0), leaf(1)}}, leaf(2)});
    s::Control control(p);
    o::CausalFrontier frontier(p);
    s::Ledger ledger(p, control.canonicalCut);
    s::PublicationSupport support(p, control, frontier);
    o::Cut join = o::NoAnalysisId;
    for (o::Cut site = 0; site < control.graph.sites.size(); ++site) {
        if (control.graph.operations[site] == 2) { join = site; }
    }
    require(join != o::NoAnalysisId, "choice continuation missing");
    ledger.append(join, {o::Command::Publish, P, Q, 0}, o::EndpointPurpose::Completion);
    support.refresh(ledger);
    require(support.records().front().preserved, "fixed choice dependency was not represented");
    const auto old = support.records().front().admittedRoot;
    // A P fence inserted before the existing source is represented as a local
    // dependency change. Direct append must invalidate downstream signatures.
    o::Cut before = o::NoAnalysisId;
    for (o::Cut site = 0; site < control.graph.sites.size(); ++site) {
        if (control.graph.operations[site] == 0) { before = site; }
    }
    require(before != o::NoAnalysisId, "choice payload missing");
    ledger.append(before, {o::Command::Barrier, P, P, 0}, o::EndpointPurpose::LocalFence);
    support.refresh(ledger);
    require(support.records().front().admittedRoot == old, "choice contract was renewed after direct mutation");
    // Stale selected interfaces cannot support a private positive answer.
    ledger.append(before, {o::Command::Barrier, Q, Q, 0}, o::EndpointPurpose::LocalFence);
    const auto prepared = ledger.preparePacket({{join, {o::Command::Barrier, P, P, 0},
        o::EndpointPurpose::LocalFence}});
    const auto view = ledger.packetView(prepared);
    require(bool(view) && !support.inspect(ledger, &*view, view->changedCuts()).preserved,
            "stale interface certified a private packet");
}

void normalConstructor()
{
    auto p = base(4, 2);
    p.target.keys[unsigned(P)][unsigned(Q)] = {0};
    p.target.keys[unsigned(Q)][unsigned(P)] = {0};
    p.operations = {op(Q, {{0, false, true}}), op(P, {{1, false, true, true}}),
        op(Q, {{1, true, false}}), op(P, {{2, false, true, true}}),
        op(P, {{3, false, true, true}}), op(R, {{3, true, false}}), op(Q, {{2, true, false}})};
    const auto plan = accepted(p);
    bool detected = false;
    for (const auto& contract : plan.publicationSupport) {
        const auto& command = plan.ledger[contract.publication].command;
        if (command.source == P && command.observer == R) { detected |= !contract.preserved; }
    }
    require(detected, "ordinary acknowledgment contamination lacks persistent diagnostic");
    require(std::any_of(plan.decisions.begin(), plan.decisions.end(),
        [](const auto& d) { return !d.existingPublicationsPreserved; }),
        "complete packet did not retain its preservation outcome");
}

void partitionAndFence()
{
    auto p = base(1);
    p.operations = {op(P, {{0, false, true}}), op(Q, {{0, true, false}})};
    const auto original = accepted(p);
    p.cells.push_back(p.cells.front());
    p.cells.back().ranges = {{16, 16}};
    p.operations[0].accesses.push_back({1, false, true});
    p.operations[1].accesses.push_back({1, true, false});
    const auto divided = accepted(p);
    require(original.decisions.size() == divided.decisions.size() &&
            original.publicationSupport.size() == divided.publicationSupport.size() &&
            original.work.publicationSupportNodes == divided.work.publicationSupportNodes,
            "uniform physical witnesses multiplied publication decisions/signatures");
    p.operations = {op(P, {}), op(P, {})};
    s::Control control(p);
    o::CausalFrontier frontier(p);
    s::Ledger ledger(p, control.canonicalCut);
    s::PublicationSupport support(p, control, frontier);
    ledger.append(2, {o::Command::Publish, P, Q, 0}, o::EndpointPurpose::Completion);
    support.refresh(ledger);
    const auto baseline = support.records().front().admittedRoot;
    ledger.append(1, {o::Command::Barrier, P, P, 0}, o::EndpointPurpose::LocalFence);
    support.refresh(ledger);
    require(!support.records().front().preserved && support.records().front().admittedRoot == baseline,
            "direct fence changed an issue gate without invalidating its source signature");
}

void boundedWork()
{
    for (unsigned length : {16U, 64U, 256U}) {
        auto p = base(1);
        for (unsigned i = 0; i < length; ++i) { p.operations.push_back(op(P, {})); }
        s::Control control(p);
        o::CausalFrontier frontier(p);
        s::Ledger ledger(p, control.canonicalCut);
        s::PublicationSupport support(p, control, frontier);
        support.refresh(ledger);
        ledger.clearChanges();
        const auto before = support.sites;
        ledger.append(length, {o::Command::Publish, P, Q, 0}, o::EndpointPurpose::Completion);
        support.refresh(ledger);
        require(support.sites - before == 1, "tail edit replayed an unchanged prefix");
        require(support.nodes < 5 * length + 10, "linear syntax expanded symbolic dependencies");
    }
}
int main()
{
    exactGapsAndPropagation();
    unchangedReceipt();
    restorationAndGenerations();
    cyclicUnknown();
    choiceAndDirectFence();
    normalConstructor();
    partitionAndFence();
    boundedWork();
    std::cout << "publication support passed\n";
}
