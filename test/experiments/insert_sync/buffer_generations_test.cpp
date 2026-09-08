// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/InsertSync/BufferGenerationAnalysis.h"
#include "PTO/Transforms/InsertSync/LifecycleCompletion.h"
#include <iostream>
#include <cstdlib>
using namespace mlir::pto::insert_sync_frontier;
static unsigned checks = 0;
static void check(bool yes, const char* message)
{
    ++checks;
    if (!yes) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
static Node issue(unsigned phase, unsigned lane, std::vector<unsigned> next)
{
    return {Node::Kind::Issue, lane, phase, kInvalid, std::move(next)};
}
static Node pass(std::vector<unsigned> next)
{
    Node n;
    n.next = std::move(next);
    return n;
}
static Node finish()
{
    Node n;
    n.kind = Node::Kind::Exit;
    return n;
}
static BufferGenerationAnalysis analyze(Program p, LifecycleSpec s, Bits boundaries = Bits())
{
    if (!boundaries.size())
        boundaries = Bits(p.nodes.size());
    Budget b;
    return analyzeBufferGenerations(p, s, boundaries, b);
}
static bool complete(const BufferGenerationAnalysis& a)
{
    return a.certificate.status == LifecycleCertificate::Status::Complete;
}
int main()
{
    LifecycleSpec pair{0, 1, 1, {{1, 0}, {0, 1}}};
    Program loop{2, {0, 1}, {}, {issue(0, 0, {1}), issue(1, 1, {0, 2}), finish()}};
    auto a = analyze(loop, pair);
    check(complete(a) && a.certificate.mayReuse, "loop-carried generations converge");
    check(
        a.reads.size() == 1 && a.reads[0].reachingWrites.test(0) && !a.reads[0].reachingWrites.test(2),
        "read binds to its reaching writer, not live-in");
    check(a.certificate.nodeRoles[1].publishFree, "release after last reader");
    LogicalLifecycle logical;
    logical.spec = pair;
    logical.certificate = a.certificate;
    Budget budget;
    auto projection = projectLifecycleCompletion(loop, {logical}, {}, budget);
    check(projection.status == LifecycleCompletionProjection::Status::Complete, "generation events project");
    auto proof = completion(projection.program, Bits(projection.program.nodes.size()), budget);
    check(proof.eventsProved, "constructed loop events satisfy reuse proof");

    Program uninitialized{2, {0, 1}, {}, {pass({1, 2}), issue(0, 0, {2}), issue(1, 1, {3}), finish()}};
    a = analyze(uninitialized, pair);
    check(
        !complete(a) && a.certificate.reason.find("uninitialized") != std::string::npos,
        "zero-work predecessor cannot invent a reaching generation");

    LifecycleSpec diamondSpec{0, 1, 1, {{1, 0}, {1, 0}, {0, 1}}};
    Program diamond{2, {0, 0, 1}, {}, {pass({1, 2}), issue(0, 0, {3}), issue(1, 0, {3}), issue(2, 1, {4}), finish()}};
    a = analyze(diamond, diamondSpec);
    check(
        complete(a) && a.reads[0].reachingWrites.test(0) && a.reads[0].reachingWrites.test(1),
        "branch join preserves alternative reaching writers");

    LifecycleSpec optionalSpec{0, 1, 1, {{1, 0}, {0, 1}, {0, 1}, {1, 0}, {0, 1}}};
    Program optional{
        2,
        {0, 1, 1, 0, 1},
        {},
        {issue(0, 0, {1}), issue(1, 1, {2}), pass({3, 4}), issue(2, 1, {5}), pass({5}), issue(3, 0, {6}),
         issue(4, 1, {7}), finish()}};
    Bits boundaries(optional.nodes.size());
    boundaries.set(4);
    a = analyze(optional, optionalSpec, boundaries);
    check(complete(a), "conditional final use summarized at branch boundary");
    check(
        a.certificate.nodeRoles[4].releaseBefore && a.certificate.nodeRoles[3].publishFree,
        "each alternative releases at its own final-use boundary");
    check(
        a.possibleFinalReaders.test(1) && a.possibleFinalReaders.test(3),
        "final-reader frontier retains both alternatives");

    LifecycleSpec unusedSpec{0, 1, 1, {{1, 0}, {1, 0}, {0, 1}}};
    Program unused{2, {0, 0, 1}, {}, {issue(0, 0, {1}), issue(1, 0, {2}), issue(2, 1, {3}), finish()}};
    a = analyze(unused, unusedSpec);
    check(complete(a) && a.certificate.nodeRoles[1].bypassReady, "unused generation returned before overwrite");
    check(
        a.reads[0].reachingWrites.test(1) && !a.reads[0].reachingWrites.test(0),
        "overwritten definition does not reach later reader");

    LifecycleSpec updateSpec{0, 1, 1, {{1, 0}, {1, 0, 1}, {0, 1}}};
    Program updates{2, {0, 0, 1}, {}, {issue(0, 0, {1}), issue(1, 0, {2}), issue(2, 1, {3}), finish()}};
    a = analyze(updates, updateSpec);
    check(
        complete(a) && !a.certificate.nodeRoles[0].publishReady && a.certificate.nodeRoles[1].publishReady,
        "read/modify/write chain publishes its final generation");
    check(
        a.reads[0].update && a.reads[0].reachingWrites.test(0) && a.reads[1].reachingWrites.test(1),
        "update reads old content and defines the content consumed afterward");
    check(
        a.certificate.supplies(updateSpec, 0, 1, 0) == LifecycleCertificate::Supply::None,
        "protocol does not discharge internal update ordering");
    updates.nodes[0] = pass({1});
    updates.allowUnrepresentedPhases = true;
    check(!complete(analyze(updates, updateSpec)), "uninitialized update rejected");

    LifecycleSpec bundle{0, 1, 2, {{1, 0}, {2, 0}, {0, 3}}};
    Program bundled{2, {0, 0, 1}, {}, {issue(0, 0, {1}), issue(1, 0, {2}), issue(2, 1, {0, 3}), finish()}};
    a = analyze(bundled, bundle);
    check(
        complete(a) && a.reads.size() == 2 && !a.certificate.nodeRoles[0].publishReady &&
            a.certificate.nodeRoles[1].publishReady,
        "bundle published only when all members are ready");
    bundle.phases[2].reads = 1;
    check(!complete(analyze(bundled, bundle)), "partial bundle consumer prevents merging");

    for (unsigned depth : {1u, 2u, 3u, 5u}) {
        Program p;
        p.lanes = 2;
        for (unsigned slot = 0; slot < depth; ++slot) {
            p.phaseLane.push_back(0);
            p.phaseLane.push_back(1);
            p.nodes.push_back(issue(2 * slot, 0, {2 * slot + 1}));
            p.nodes.push_back(issue(2 * slot + 1, 1, {2 * slot + 2}));
        }
        p.nodes.back().next = {0, 2 * depth};
        p.nodes.push_back(finish());
        for (unsigned slot = 0; slot < depth; ++slot) {
            LifecycleSpec s{0, 1, 1, std::vector<LifecycleTouch>(2 * depth)};
            s.phases[2 * slot].writes = 1;
            s.phases[2 * slot + 1].reads = 1;
            a = analyze(p, s);
            check(
                complete(a) && a.reads.size() == 1 && a.reads[0].reachingWrites.test(2 * slot),
                "slot depth does not alter another slot's generation");
        }
    }
    std::vector<ReconstructedLifecycleNode> reconstructed = {
        {LifecycleAction::FreeSignal, 0, {1}},   {LifecycleAction::FreeWait, 0, {2}},
        {LifecycleAction::Write, 1, {3}},        {LifecycleAction::Update, 1, {4}},
        {LifecycleAction::PublishReady, 0, {5}}, {LifecycleAction::AcquireReady, 0, {6}},
        {LifecycleAction::Read, 1, {7}},         {LifecycleAction::FreeSignal, 0, {8}},
        {LifecycleAction::FreeWait, 0, {}}};
    Budget b;
    check(verifyReconstructedLifecycle(reconstructed, 1, b), "emitted update protocol independently reconstructs");
    reconstructed[2].action = LifecycleAction::None;
    check(!verifyReconstructedLifecycle(reconstructed, 1, b), "reconstruction catches missing initializer");
    Budget exhausted;
    exhausted.left = 0;
    a = analyzeBufferGenerations(loop, pair, Bits(loop.nodes.size()), exhausted);
    check(a.certificate.status == LifecycleCertificate::Status::AnalysisLimit, "budget exhaustion never grants supply");
    // Flow remains useful when a two-lane consumer set has no single-lane
    // ready/free recipe, and when an entry value has no local definition.
    Program multiple{3, {0, 1, 2}, {},
        {issue(0, 0, {1}), issue(1, 1, {2}), issue(2, 2, {0, 3}), finish()}};
    Atom storage{Bits(3), Bits(3)};
    storage.writes.set(0); storage.reads.set(1); storage.reads.set(2);
    Budget flowBudget;
    auto flow = analyzeBufferGenerationFlow(multiple, {storage}, {storage.writes}, flowBudget);
    check(flow.status == BufferGenerationFlow::Status::Complete && flow.reads.size() == 2,
          "open flow admits multiple consumer lanes");
    check(flow.frontiers.generations.back().finalByLane[1].size() == 1 &&
          flow.frontiers.generations.back().finalByLane[2].size() == 1,
          "each lane retains its final reader independently");
    check(!flow.ordering.requirements.empty(), "open flow retains repair obligations without event assignment");
    Program choice{3, {0, 1, 2}, {},
        {pass({1, 2}), issue(0, 0, {3}), issue(1, 1, {3}), issue(2, 2, {4}), finish()}};
    flow = analyzeBufferGenerationFlow(choice, {storage}, {storage.writes}, flowBudget);
    check(!flow.orderedPhases[0].test(1) && flow.orderedPhases[0].test(2),
          "mutually exclusive accesses distinguished from join readers");
    check(flow.status == BufferGenerationFlow::Status::Complete && flow.reads[0].reachingWrites.test(3),
          "live-in preserves open facts instead of rejecting analysis");
    flow = analyzeBufferGenerationFlow(choice, {storage}, {Bits(3)}, flowBudget);
    check(flow.reads.back().reachingWrites.test(3), "may write must not kill live-in content");

    Program drained{2, {0}, {{0, 1}},
        {issue(0, 0, {1}), {Node::Kind::Signal, 0, kInvalid, 0, {2}},
         {Node::Kind::Wait, 1, kInvalid, 0, {3}}, finish()}};
    auto exits = [&](const Program& program) {
        Budget remaining;
        auto facts = completion(program, Bits(program.nodes.size()), remaining);
        return completionAtExits(program, facts).proved;
    };
    check(exits(drained), "explicit final wait proves return completion without ALL");
    auto undrained = drained;
    undrained.phaseLane.push_back(1);
    undrained.nodes[3] = issue(1, 1, {4}); undrained.nodes.push_back(finish());
    check(!exits(undrained), "unrelated final store keeps exit drain");
    auto skipped = drained;
    skipped.nodes[1].next = {2, 3};
    check(!exits(skipped), "skipped consumption cannot justify deleting ALL");
    auto reissued = drained;
    reissued.nodes[3] = issue(0, 0, {4}); reissued.nodes.push_back(finish());
    check(!exits(reissued), "old drain cannot complete a reissued generation");
    Program empty{2, {}, {}, {finish()}};
    check(exits(empty), "empty physical scope has no outstanding work");
    auto circular = drained;
    circular.nodes.insert(circular.nodes.end() - 1, {Node::Kind::All, 0, kInvalid, kInvalid, {4}});
    circular.nodes[2] = pass({3});
    check(!exits(circular), "ALL does not repair an unconsumed event token");

    Program regional{2, {0, 0, 1}, {},
        {issue(0, 0, {1}), pass({2, 3}), issue(1, 0, {3}), issue(2, 1, {4}), pass({1, 5}), finish()},
        {{1, 4, RegionScope::Kind::Loop}}};
    Atom regionAtom{Bits(3), Bits(3)};
    regionAtom.reads.set(2); regionAtom.writes.set(0); regionAtom.writes.set(1);
    Budget regionBudget;
    auto summary = summarizeBufferRegion(regional, regional.regions[0], regionAtom, regionAtom.writes, regionBudget);
    check(bool(summary) && summary->afterExit.test(3) && summary->afterExit.test(1),
          "region exports incoming and newly produced alternatives");
    Bits incoming(4); incoming.set(0);
    auto substituted = summary->apply(summary->afterExit, incoming);
    check(substituted.test(0) && substituted.test(1) && !substituted.test(3),
          "region substitutes its actual incoming generation");
    auto summarizedFlow = analyzeBufferGenerationFlow(regional, {regionAtom}, {regionAtom.writes}, regionBudget);
    regional.regions.clear();
    auto directFlow = analyzeBufferGenerationFlow(regional, {regionAtom}, {regionAtom.writes}, regionBudget);
    check(summarizedFlow.regionTransfersUsed == 1 && summarizedFlow.reads.size() == directFlow.reads.size() &&
          summarizedFlow.reads[0].reachingWrites == directFlow.reads[0].reachingWrites,
          "composed recurrence agrees with unsummarized fixed point");

    for (unsigned depth : {2u, 3u, 5u}) {
        AccessSlice x, y;
        x.known = y.known = true; x.write = true; y.read = true;
        x.extent = y.extent = 128;
        SlotMap slots;
        slots.modulus = depth; slots.extent = 128; slots.selector = {0, {{0, 1}}, true};
        for (unsigned i = 0; i < depth; ++i) slots.addresses.push_back(i * 256);
        x.slots = y.slots = slots;
        OccurrenceRelation relation;
        relation.kind = OccurrenceRelation::Kind::KnownDistance;
        relation.deltas = {{0, 1}};
        check(compareAccesses(x, y, relation, false, regionBudget).kind == AccessRelationResult::Kind::Disjoint,
              "next iteration uses a different slot");
        relation.deltas = {{0, depth}};
        check(compareAccesses(x, y, relation, false, regionBudget).kind == AccessRelationResult::Kind::Conflict,
              "wraparound retains the storage-reuse obligation");
    }

    Program streams{2, {0, 1}, {{0, 1}, {1, 0}, {0, 1}},
        {issue(0, 0, {1}), {Node::Kind::Signal, 0, kInvalid, 0, {2}},
         {Node::Kind::Wait, 1, kInvalid, 0, {3}}, issue(1, 1, {4}),
         {Node::Kind::Signal, 1, kInvalid, 1, {5}}, {Node::Kind::Wait, 0, kInvalid, 1, {6}},
         issue(0, 0, {7}), {Node::Kind::Signal, 0, kInvalid, 2, {8}},
         {Node::Kind::Wait, 1, kInvalid, 2, {9}}, finish()}};
    Bits shareable(3); shareable.set(0); shareable.set(2);
    Budget sharingBudget;
    auto assigned = shareEventKeys(streams, shareable, sharingBudget);
    check(assigned.merged == 1 && assigned.representative[2] == 0,
          "acknowledged disjoint stream lifetimes can share one physical key");
    auto constrained = allocateLifecycles({logical}, {{{0, 1}, 3}}, 2);
    check(constrained.status == LifecycleAllocation::Status::ResourceUnresolved &&
          constrained.failedIdentity == logical.identity && constrained.failedSource == 0 &&
          constrained.failedTarget == 1 && constrained.conflictingKeyMask == 3,
          "failed assignment retains the actual candidate, domain and reserved keys");
    auto compacted = allocateLifecycles({logical}, {{{0, 1}, 1}}, 2);
    check(assigned.merged == 1 && compacted.status == LifecycleAllocation::Status::Complete &&
          compacted.ready[0] == 1,
          "proved residual sharing makes room for a lifecycle without any extra ordering");
    streams.nodes[4] = pass({5}); streams.nodes[5] = pass({6});
    assigned = shareEventKeys(streams, shareable, sharingBudget);
    check(assigned.merged == 0, "missing causal acknowledgement prevents sharing despite balanced counts");

    Program initialTokens{2, {}, {{1, 0}}, {
        {Node::Kind::Signal, 1, kInvalid, 0, {1}},
        {Node::Kind::Wait, 0, kInvalid, 0, {2}},
        {Node::Kind::Signal, 1, kInvalid, 0, {3}},
        {Node::Kind::Wait, 0, kInvalid, 0, {2, 4}}, finish()}};
    Budget initialBudget;
    auto firstWaits = initialEventAcquisitions(initialTokens, 0, initialBudget);
    check(firstWaits && firstWaits->test(1) && !firstWaits->test(3),
          "initial token ownership excludes recurring publications");
    initialTokens.nodes[2].next = {1};
    firstWaits = initialEventAcquisitions(initialTokens, 0, initialBudget);
    check(firstWaits && !firstWaits->test(1),
          "shared first/recurring static acquisition cannot be deleted");
    initialTokens.nodes[0] = pass({1, 2});
    check(!initialEventAcquisitions(initialTokens, 0, initialBudget),
          "a non-publication cannot seed initial-token ownership");
    initialTokens.nodes[0] = {Node::Kind::Signal, 1, kInvalid, 0, {1}};
    Budget tokenBudget; tokenBudget.left = 0;
    check(!initialEventAcquisitions(initialTokens, 0, tokenBudget), "token-origin budget returns unknown");

    Budget cachedBudget, freshBudget;
    auto cachedFlow = analyzeBufferGenerationFlow(regional, {regionAtom}, {regionAtom.writes},
                                                   cachedBudget, directFlow.control);
    auto freshFlow = analyzeBufferGenerationFlow(regional, {regionAtom}, {regionAtom.writes}, freshBudget);
    check(cachedFlow.status == BufferGenerationFlow::Status::Complete &&
          cachedFlow.control == directFlow.control && cachedBudget.left > freshBudget.left &&
          cachedFlow.orderedPhases == freshFlow.orderedPhases &&
          cachedFlow.precedingOnLane == freshFlow.precedingOnLane,
          "slice projections reuse exact immutable control facts with less analysis work");
    auto changedControl = regional; changedControl.nodes[4].next = {5};
    auto invalidatedFlow = analyzeBufferGenerationFlow(changedControl, {regionAtom}, {regionAtom.writes},
                                                        cachedBudget, directFlow.control);
    check(invalidatedFlow.status == BufferGenerationFlow::Status::Complete &&
          invalidatedFlow.control != directFlow.control && !invalidatedFlow.orderedPhases[2].test(1),
          "changed backedge invalidates cached occurrence relationships");

    std::cout << checks << " buffer-generation checks passed\n";
}
