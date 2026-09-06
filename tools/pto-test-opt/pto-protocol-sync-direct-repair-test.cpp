// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- pto-protocol-sync-direct-repair-test.cpp ------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/ProtocolSync/DirectRepair.h"
#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"
#include "PTO/Transforms/ProtocolSync/EventLifetime.h"
#include "PTO/Transforms/ProtocolSync/ResidualObligation.h"
#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

bool checkIndependentReadinessInterleavings(
    const StructuredSyncIR& schedule, SyncPhaseId first, SyncPhaseId second, bool& overlapWitness);
bool checkStructuredFrontierInterleavings(
    const StructuredSyncIR& schedule, unsigned trips, std::uint64_t choices, bool* unsafeWitness);

namespace {

constexpr StringLiteral kSharedFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @shared_frontiers(
      %input0: !pto.partition_tensor_view<16x16xf16>,
      %input1: !pto.partition_tensor_view<16x16xf16>,
      %output0: !pto.partition_tensor_view<16x16xf16>,
      %output1: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>, pto.gm_alias = "assume-disjoint-arguments"} {
    %c0 = arith.constant 0 : i64
    %c512 = arith.constant 512 : i64
    %c1024 = arith.constant 1024 : i64
    %c1536 = arith.constant 1536 : i64
    %load0 = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    %load1 = pto.alloc_tile addr = %c512 : !pto.tile_buf<vec, 16x16xf16>
    %compute0 = pto.alloc_tile addr = %c1024 : !pto.tile_buf<vec, 16x16xf16>
    %compute1 = pto.alloc_tile addr = %c1536 : !pto.tile_buf<vec, 16x16xf16>
    pto.tload ins(%input0 : !pto.partition_tensor_view<16x16xf16>)
              outs(%load0 : !pto.tile_buf<vec, 16x16xf16>)
    pto.tload ins(%input1 : !pto.partition_tensor_view<16x16xf16>)
              outs(%load1 : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%load0 : !pto.tile_buf<vec, 16x16xf16>)
             outs(%compute0 : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%load1 : !pto.tile_buf<vec, 16x16xf16>)
             outs(%compute1 : !pto.tile_buf<vec, 16x16xf16>)
    pto.tstore ins(%compute0 : !pto.tile_buf<vec, 16x16xf16>)
               outs(%output0 : !pto.partition_tensor_view<16x16xf16>)
    pto.tstore ins(%compute1 : !pto.tile_buf<vec, 16x16xf16>)
               outs(%output1 : !pto.partition_tensor_view<16x16xf16>)
    return
  }
}
)mlir";

constexpr StringLiteral kLoopFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @recurring_direct_rejected(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %output: !pto.partition_tensor_view<16x16xf16>, %count: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tile = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
    scf.for %iteration = %c0 to %count step %c1 {
      pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
                outs(%tile : !pto.tile_buf<vec, 16x16xf16>)
      pto.tstore ins(%tile : !pto.tile_buf<vec, 16x16xf16>)
                 outs(%output : !pto.partition_tensor_view<16x16xf16>)
    }
    return
  }
}
)mlir";

constexpr StringLiteral kNonAdjacentFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @nonadjacent_same_pipe(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %output: !pto.partition_tensor_view<16x16xf16>,
      %unrelated_output: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %c512 = arith.constant 512 : i64
    %c1024 = arith.constant 1024 : i64
    %c1536 = arith.constant 1536 : i64
    %source = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    %middle = pto.alloc_tile addr = %c512 : !pto.tile_buf<vec, 16x16xf16>
    %unrelated = pto.alloc_tile addr = %c1024 : !pto.tile_buf<vec, 16x16xf16>
    %result = pto.alloc_tile addr = %c1536 : !pto.tile_buf<vec, 16x16xf16>
    pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
              outs(%source : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%source : !pto.tile_buf<vec, 16x16xf16>)
             outs(%middle : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%source : !pto.tile_buf<vec, 16x16xf16>)
             outs(%unrelated : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%middle : !pto.tile_buf<vec, 16x16xf16>)
             outs(%result : !pto.tile_buf<vec, 16x16xf16>)
    pto.tstore ins(%result : !pto.tile_buf<vec, 16x16xf16>)
               outs(%output : !pto.partition_tensor_view<16x16xf16>)
    pto.tstore ins(%unrelated : !pto.tile_buf<vec, 16x16xf16>)
               outs(%unrelated_output : !pto.partition_tensor_view<16x16xf16>)
    return
  }
}
)mlir";

constexpr StringLiteral kReservationFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @reserved_event() attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %c64 = arith.constant 64 : i64
    %c128 = arith.constant 128 : i64
    %src = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 1x32xf16>
    %idx = pto.alloc_tile addr = %c64 : !pto.tile_buf<vec, 1x32xi16>
    %dst = pto.alloc_tile addr = %c128 : !pto.tile_buf<vec, 1x32xf16>
    pto.tscatter ins(%src, %idx : !pto.tile_buf<vec, 1x32xf16>, !pto.tile_buf<vec, 1x32xi16>)
                 outs(%dst : !pto.tile_buf<vec, 1x32xf16>)
    return
  }
}
)mlir";

constexpr StringLiteral kAtomicFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @supported_first(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %output: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %source = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
    %target = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
    pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
              outs(%source : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%source : !pto.tile_buf<vec, 16x16xf16>)
             outs(%target : !pto.tile_buf<vec, 16x16xf16>)
    pto.tstore ins(%target : !pto.tile_buf<vec, 16x16xf16>)
               outs(%output : !pto.partition_tensor_view<16x16xf16>)
    return
  }
  func.func @unsupported_second(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %output: !pto.partition_tensor_view<16x16xf16>, %count: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tile = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
    scf.for %iteration = %c0 to %count step %c1 {
      pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
                outs(%tile : !pto.tile_buf<vec, 16x16xf16>)
      pto.tstore ins(%tile : !pto.tile_buf<vec, 16x16xf16>)
                 outs(%output : !pto.partition_tensor_view<16x16xf16>)
    }
    return
  }
}
)mlir";

bool check(bool condition, const Twine& message)
{
    if (condition) {
        return true;
    }
    llvm::errs() << "FAIL: " << message << '\n';
    return false;
}

OwningOpRef<ModuleOp> parseFixture(MLIRContext& context, StringRef source)
{
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
    if (!module || failed(verify(*module))) {
        return {};
    }
    return module;
}

std::string printFunction(func::FuncOp function)
{
    std::string text;
    llvm::raw_string_ostream output(text);
    function.print(output);
    return text;
}

std::string printModule(ModuleOp module)
{
    std::string text;
    llvm::raw_string_ostream output(text);
    module.print(output);
    return text;
}

struct AnalysisFixture {
    LegacySyncIRAdapter adapter;
    LegacySyncSnapshot legacy;
    SyncSemanticContext semanticContext;
    StructuredSyncIR schedule;
    FailureOr<PipelineStageAnalysisResult> stages;
    StorageTimelineAnalysisResult timelines;
    ChannelAnalysisResult channels;

    explicit AnalysisFixture(func::FuncOp function) : schedule(function), stages(failure()) {}
};

bool buildAnalysis(AnalysisFixture& fixture)
{
    func::FuncOp function = fixture.schedule.getFunction();
    if (failed(fixture.adapter.buildSnapshot(function, fixture.legacy))) {
        return false;
    }
    fixture.semanticContext = fixture.adapter.buildSemanticContext(fixture.legacy);
    StructuredSyncIRBuilder builder(fixture.semanticContext);
    if (failed(builder.build(function, fixture.schedule))) {
        return false;
    }
    fixture.stages = analyzePipelineStages(fixture.schedule);
    if (failed(fixture.stages)) {
        return false;
    }
    fixture.timelines = analyzeStorageTimelines(fixture.schedule, *fixture.stages);
    fixture.channels = analyzeChannels(fixture.schedule, *fixture.stages, fixture.timelines);
    compareWithLegacyDemandOracle(
        fixture.legacy, fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    return true;
}

FailureOr<SyncInterpretationResult> interpretEmpty(const AnalysisFixture& fixture)
{
    SyncSelectedWorld world;
    return interpretSelectedWorld(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, world);
}

bool testReadinessOverlapAndMutations(func::FuncOp function)
{
    bool passed = true;
    AnalysisFixture emitted(function);
    if (!check(
            buildAnalysis(emitted) && emitted.schedule.getPhases().size() == 6,
            "cannot reconstruct emitted independent-frontier fixture")) {
        return false;
    }
    bool overlap = false;
    passed &= check(
        checkIndependentReadinessInterleavings(emitted.schedule, 1, 2, overlap) && overlap,
        "independent L2 and C1 cannot remain outstanding concurrently");
    Operation* load1 = emitted.schedule.findPhase(1)->operation;
    Operation* earlySet = emitted.schedule.findPhase(0)->operation->getNextNode();
    if (!check(isa<SetFlagOp>(earlySet), "first producer has no immediate publication")) {
        return false;
    }
    // This mutation remains race-free but removes the required overlap witness.
    earlySet->moveAfter(load1);
    AnalysisFixture delayed(function);
    if (!check(buildAnalysis(delayed), "cannot reconstruct delayed publication")) {
        return false;
    }
    passed &= check(
        checkIndependentReadinessInterleavings(delayed.schedule, 1, 2, overlap) && !overlap,
        "oracle did not distinguish safe serialization from independent readiness");
    earlySet->moveAfter(emitted.schedule.findPhase(0)->operation);
    Operation* secondWait = emitted.schedule.findPhase(3)->operation->getPrevNode();
    if (!check(isa<WaitFlagOp>(secondWait), "second consumer has no immediate acquisition")) {
        return false;
    }
    secondWait->moveBefore(emitted.schedule.findPhase(2)->operation);
    AnalysisFixture advanced(function);
    if (!check(buildAnalysis(advanced), "cannot reconstruct advanced acquisition")) {
        return false;
    }
    passed &= check(
        checkIndependentReadinessInterleavings(advanced.schedule, 1, 2, overlap) && !overlap,
        "oracle did not detect blocking from an unnecessarily early acquisition");
    return passed;
}

bool testSharedRepairAndMaterialization(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kSharedFixture);
    if (!check(static_cast<bool>(module), "cannot parse shared-frontier fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture), "cannot analyze shared-frontier fixture")) {
        return false;
    }
    FailureOr<SyncInterpretationResult> residuals = interpretEmpty(fixture);
    const bool hasResiduals = succeeded(residuals) && !residuals->isComplete();
    if (!check(hasResiduals, "shared fixture has no residuals")) {
        return false;
    }
    FailureOr<SyncDirectRepairPlan> plan =
        buildDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations);
    bool passed = check(
        succeeded(plan) && plan->status == SyncDirectRepairPlanStatus::Ready, "shared residual plan is not ready");
    if (failed(plan)) {
        return false;
    }
    passed &= check(
        succeeded(verifyDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations, *plan)),
        "independent logical verifier rejected shared plan");
    const unsigned independentEvents = llvm::count_if(plan->candidates, [](const SyncDirectRepairCandidate& candidate) {
        return candidate.kind == SyncDirectRepairKind::DirectedEvent && candidate.obligations.size() == 1;
    });
    const unsigned exitCandidates = llvm::count_if(plan->candidates, [](const SyncDirectRepairCandidate& candidate) {
        return candidate.kind == SyncDirectRepairKind::ExitBarrier && candidate.obligations.size() >= 2;
    });
    passed &= check(independentEvents == 4, "independent readiness frontiers were merged");
    passed &= check(exitCandidates == 1, "terminal phases did not share one exit drain");

    auto duplicateObligations = residuals->obligations;
    auto duplicate = duplicateObligations.front();
    duplicate.id = duplicateObligations.size();
    duplicateObligations.push_back(duplicate);
    auto deduplicated = buildDirectRepairPlan(fixture.schedule, *fixture.stages, duplicateObligations);
    passed &= check(
        succeeded(deduplicated) && deduplicated->candidates.size() == plan->candidates.size() &&
            succeeded(verifyDirectRepairPlan(fixture.schedule, *fixture.stages, duplicateObligations, *deduplicated)),
        "identical endpoints must share one normal repair without losing obligations");

    SyncDirectRepairPlan merged = *plan;
    const bool hasTwoFrontiers = merged.candidates.size() >= 2;
    if (!check(hasTwoFrontiers, "cannot construct merged-frontier negative")) {
        return false;
    }
    auto& first = merged.candidates[0];
    const auto& second = merged.candidates[1];
    first.sourcePhase = second.sourcePhase;
    first.sourceOperation = second.sourceOperation;
    first.obligations.append(second.obligations.begin(), second.obligations.end());
    merged.candidates.erase(std::next(merged.candidates.begin()));
    for (auto [id, candidate] : llvm::enumerate(merged.candidates)) {
        candidate.id = id;
    }
    passed &= check(
        failed(verifyDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations, merged)),
        "normal-placement verifier accepted a delayed merged publication");

    passed &= check(
        succeeded(allocateDirectRepairEvents(fixture.schedule, *plan)) &&
            succeeded(verifyDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations, *plan)),
        "event allocation produced an invalid direct plan");
    SyncSelectedWorld selectedWorld;
    SmallVector<SyncDirectCandidateId, 8> selected;
    for (const SyncDirectRepairCandidate& candidate : plan->candidates) {
        selected.push_back(candidate.id);
    }
    passed &= check(
        succeeded(applyDirectRepairCandidates(*plan, residuals->obligations, selected, selectedWorld)),
        "cannot adapt direct candidates to selected-world effects");
    FailureOr<SyncInterpretationResult> complete =
        interpretSelectedWorld(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, selectedWorld);
    passed &= check(succeeded(complete) && complete->isComplete(), "direct repairs left residual obligations");

    passed &= check(
        succeeded(
            materializeAndVerifyDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations, *plan)),
        "staged direct materialization failed independent verification");
    unsigned sets = 0;
    unsigned waits = 0;
    unsigned tails = 0;
    const unsigned directedCandidates =
        llvm::count_if(plan->candidates, [](const SyncDirectRepairCandidate& candidate) {
            return candidate.kind == SyncDirectRepairKind::DirectedEvent;
        });
    function.walk([&](Operation* operation) {
        sets += isa<SetFlagOp>(operation) ? 1 : 0;
        waits += isa<WaitFlagOp>(operation) ? 1 : 0;
        auto barrier = dyn_cast<BarrierOp>(operation);
        tails += barrier && barrier.getPipe().getPipe() == PIPE::PIPE_ALL ? 1 : 0;
    });
    passed &= check(
        sets == directedCandidates && waits == directedCandidates && tails == 1,
        "materialized shared recipe has wrong action counts");
    return testReadinessOverlapAndMutations(function) && passed;
}

std::string acknowledgedReuseFixture(bool insideLoop = false)
{
    std::string text = R"mlir(module attributes {pto.target_arch = "a3"} {
      func.func @acknowledged_reuse(%input: !pto.partition_tensor_view<16x16xf16>)
          attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
        %zero = arith.constant 0 : i64
        %offset = arith.constant 512 : i64
        %a = pto.alloc_tile addr = %zero : !pto.tile_buf<vec, 16x16xf16>
        %b = pto.alloc_tile addr = %offset : !pto.tile_buf<vec, 16x16xf16>
    )mlir";
    if (insideLoop) {
        text += "%lo = arith.constant 0 : index\n%hi = arith.constant 3 : index\n"
                "%step = arith.constant 1 : index\nscf.for %i = %lo to %hi step %step {\n";
    }
    for (unsigned index = 0; index < 7; ++index) {
        text += R"mlir(
          pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)
          pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf16>) outs(%b : !pto.tile_buf<vec, 16x16xf16>)
        )mlir";
    }
    return text + (insideLoop ? "}\nreturn\n}\n}\n" : "return\n}\n}\n");
}

bool testEventLoopAncestry(MLIRContext& context)
{
    auto module = parseFixture(context, acknowledgedReuseFixture(true));
    if (!module) {
        return false;
    }
    AnalysisFixture fixture(*module->getOps<func::FuncOp>().begin());
    const bool analyzed = buildAnalysis(fixture) && fixture.schedule.getPhases().size() == 14;
    if (!analyzed) {
        return false;
    }
    SmallVector<SyncEventGeneration, 3> events;
    for (unsigned index = 0; index < 3; ++index) {
        const auto* source = fixture.schedule.findPhase(index);
        const auto* target = fixture.schedule.findPhase(index + 1);
        // Deliberately false metadata: actual ancestry must prevent reuse.
        events.push_back(
            {index,
             SyncEventGenerationKind::DirectRepair,
             source->core,
             source->pipe,
             target->pipe,
             source->operation,
             target->operation,
             {},
             kInvalidSyncId,
             false,
             0});
    }
    return check(
        !buildEventConsumptionOrder(events).provesConsumedBeforeSet(0, 2),
        "once-only event proof trusted metadata instead of actual loop ancestry");
}

bool testConsumptionProofBoundaries(const StructuredSyncIR& schedule)
{
    const bool expectedPhaseCount = schedule.getPhases().size() == 14;
    if (!expectedPhaseCount) {
        return false;
    }
    SmallVector<SyncEventGeneration, 16> events;
    for (unsigned index = 0; index < 13; ++index) {
        const auto* source = schedule.findPhase(index);
        const auto* target = schedule.findPhase(index + 1);
        events.push_back(
            {index,
             SyncEventGenerationKind::DirectRepair,
             source->core,
             source->pipe,
             target->pipe,
             source->operation,
             target->operation,
             {},
             kInvalidSyncId,
             false,
             0});
    }
    const auto model = ProtocolSyncTarget::resolve(schedule.getFunction());
    if (!check(
            buildEventConsumptionOrder(events).provesConsumedBeforeSet(0, 2) &&
                succeeded(verifySyncEventGenerationAssignment(model, {}, events, false)),
            "acknowledgement did not prove consumption before reuse")) {
        return false;
    }
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
        auto changed = events;
        if (mutation == 0) {
            changed.erase(std::next(changed.begin())); // Remove the acknowledgement.
            for (auto [id, event] : llvm::enumerate(changed)) {
                event.id = id;
            }
        } else if (mutation == 1) {
            changed.front().recurring = true;
        } else if (mutation == 2) {
            changed.front().setAnchor = nullptr;
        } else {
            changed.front().guard.push_back({0, 0});
        }
        if (!check(
                failed(verifySyncEventGenerationAssignment(model, {}, changed, false)),
                "unproven event consumption must retain interference")) {
            return false;
        }
    }
    events.resize(129, events.front());
    for (auto [id, event] : llvm::enumerate(events)) {
        event.id = id;
    }
    return check(
        !buildEventConsumptionOrder(events).provesConsumedBeforeSet(0, 2),
        "bounded proof exhaustion must not invent consumption order");
}

bool testAcknowledgedEventReuse(MLIRContext& context)
{
    auto module = parseFixture(context, acknowledgedReuseFixture());
    if (!module) {
        return false;
    }
    auto function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    const bool validFixture = buildAnalysis(fixture) && testConsumptionProofBoundaries(fixture.schedule);
    if (!validFixture) {
        return false;
    }
    auto residuals = interpretEmpty(fixture);
    if (failed(residuals)) {
        return false;
    }
    auto plan = buildDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations);
    ProtocolSyncStatistics statistics;
    const bool allocated = succeeded(plan) && plan->isComplete() &&
                           succeeded(allocateDirectRepairEvents(fixture.schedule, *plan, &statistics));
    if (!check(
            allocated && statistics.maxEventDomainPressure == 1,
            "seven acknowledged generations must reuse one ID per direction")) {
        return false;
    }
    if (failed(
            materializeAndVerifyDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations, *plan))) {
        return false;
    }
    AnalysisFixture emitted(function);
    bool overlap = false;
    if (!check(
            buildAnalysis(emitted) && checkIndependentReadinessInterleavings(emitted.schedule, 0, 1, overlap),
            "reused IDs must pass fresh concrete and independent asynchronous verification")) {
        return false;
    }
    Operation* acknowledgement = nullptr;
    function.walk([&](SetFlagOp set) {
        const bool firstAcknowledgement = !acknowledgement && set.getSrcPipe().getPipe() == PIPE::PIPE_V;
        if (firstAcknowledgement) {
            acknowledgement = set;
        }
    });
    WaitFlagOp consumption;
    function.walk([&](WaitFlagOp wait) {
        const bool firstConsumption = !consumption && wait.getSrcPipe().getPipe() == PIPE::PIPE_V;
        if (firstConsumption) {
            consumption = wait;
        }
    });
    if (!check(acknowledgement && consumption, "missing acknowledgement actions")) {
        return false;
    }
    consumption->erase();
    acknowledgement->erase();
    AnalysisFixture broken(function);
    bool unsafeWitness = false;
    return check(
        buildAnalysis(broken) && failed(verifyFreshConcreteSyncSemantics(function)) &&
            !checkStructuredFrontierInterleavings(broken.schedule, 0, 0, &unsafeWitness) && unsafeWitness,
        "removed acknowledgement must invalidate concrete reuse and expose an unsafe execution");
}

bool testVerifierRollbackAndRecurrence(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kSharedFixture);
    if (!check(static_cast<bool>(module), "cannot parse verifier fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture), "cannot analyze verifier fixture")) {
        return false;
    }
    FailureOr<SyncInterpretationResult> residuals = interpretEmpty(fixture);
    if (!check(succeeded(residuals), "cannot interpret verifier fixture")) {
        return false;
    }
    FailureOr<SyncDirectRepairPlan> plan =
        buildDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations);
    if (!check(
            succeeded(plan) && succeeded(allocateDirectRepairEvents(fixture.schedule, *plan)),
            "cannot build verifier plan")) {
        return false;
    }
    auto event = llvm::find_if(plan->candidates, [](const SyncDirectRepairCandidate& candidate) {
        return candidate.kind == SyncDirectRepairKind::DirectedEvent;
    });
    if (!check(event != plan->candidates.end(), "verifier fixture has no directed event")) {
        return false;
    }
    event->eventId = 6;
    bool passed = check(
        failed(verifyDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations, *plan)),
        "logical verifier accepted compiler-forbidden event ID 6");
    const std::string before = printFunction(function);
    {
        ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
        passed &= check(
            failed(
                materializeAndVerifyDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations, *plan)),
            "malformed direct materialization was accepted");
    }
    passed &= check(printFunction(function) == before, "failed direct materialization changed the input function");

    ArrayRef<SyncPhase> phases = fixture.schedule.getPhases();
    for (SyncObligationKind kind :
         {SyncObligationKind::OrderedMemory, SyncObligationKind::AccConflict, SyncObligationKind::Visibility,
          SyncObligationKind::UnknownAlias}) {
        const SyncResidualObligation unsupported{
            0,
            kind,
            phases[0].id,
            phases[1].id,
            std::nullopt,
            std::nullopt,
            SyncControlRelation::MustExecute,
            {SyncIterationRelationKind::SameIteration, 0},
            "test unsupported direct residual"};
        FailureOr<SyncDirectRepairPlan> unsupportedPlan =
            buildDirectRepairPlan(fixture.schedule, *fixture.stages, ArrayRef<SyncResidualObligation>(unsupported));
        const bool rejected =
            succeeded(unsupportedPlan) && unsupportedPlan->status == SyncDirectRepairPlanStatus::Partial &&
            unsupportedPlan->rejections.size() == 1 &&
            unsupportedPlan->rejections.front().reason == SyncDirectRepairRejection::UnsupportedObligation;
        passed &= check(rejected, "protocol-shaped residual did not fail closed in direct repair");
    }

    OwningOpRef<ModuleOp> loopModule = parseFixture(context, kLoopFixture);
    if (!check(static_cast<bool>(loopModule), "cannot parse recurrence fixture")) {
        return false;
    }
    AnalysisFixture loopFixture(*loopModule->getOps<func::FuncOp>().begin());
    if (!check(buildAnalysis(loopFixture), "cannot analyze recurrence fixture")) {
        return false;
    }
    ArrayRef<SyncPhase> loopPhases = loopFixture.schedule.getPhases();
    if (!check(
            loopPhases.size() == 2 && !loopPhases.front().iterationDomain.loops.empty(),
            "recurrence fixture has an unexpected phase shape")) {
        return false;
    }
    const SyncResidualObligation recurring{
        0,
        SyncObligationKind::Completion,
        loopPhases[0].id,
        loopPhases[1].id,
        std::nullopt,
        std::nullopt,
        SyncControlRelation::MustExecute,
        {SyncIterationRelationKind::LoopCarried, 1, loopPhases[0].iterationDomain.loops.front()},
        "test recurring completion"};
    FailureOr<SyncDirectRepairPlan> loopPlan =
        buildDirectRepairPlan(loopFixture.schedule, *loopFixture.stages, ArrayRef<SyncResidualObligation>(recurring));
    const bool rejectedRecurrence =
        succeeded(loopPlan) && llvm::any_of(loopPlan->rejections, [](const auto& rejection) {
            return rejection.reason == SyncDirectRepairRejection::UnsupportedRecurrence;
        });
    passed &= check(
        succeeded(loopPlan) && loopPlan->status == SyncDirectRepairPlanStatus::Partial && rejectedRecurrence,
        "recurring direct event reuse did not fail closed");
    return passed;
}

bool testNonAdjacentSamePipe(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kNonAdjacentFixture);
    if (!check(static_cast<bool>(module), "cannot parse nonadjacent same-pipe fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture), "cannot analyze nonadjacent same-pipe fixture")) {
        return false;
    }
    FailureOr<SyncInterpretationResult> residuals = interpretEmpty(fixture);
    if (!check(succeeded(residuals), "cannot interpret nonadjacent same-pipe fixture")) {
        return false;
    }
    FailureOr<SyncDirectRepairPlan> plan =
        buildDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations);
    const bool ready = succeeded(plan) && plan->status == SyncDirectRepairPlanStatus::Ready &&
                       succeeded(allocateDirectRepairEvents(fixture.schedule, *plan));
    if (!check(ready, "cannot plan nonadjacent same-pipe repair")) {
        return false;
    }
    if (!check(
            succeeded(
                materializeAndVerifyDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations, *plan)),
            "nonadjacent same-pipe repair failed concrete verification")) {
        return false;
    }
    SmallVector<TAbsOp, 3> computes;
    BarrierOp vectorBarrier;
    function.walk([&](Operation* operation) {
        if (auto compute = dyn_cast<TAbsOp>(operation)) {
            computes.push_back(compute);
        }
        auto barrier = dyn_cast<BarrierOp>(operation);
        const bool isVectorBarrier = barrier && barrier.getPipe().getPipe() == PIPE::PIPE_V;
        if (isVectorBarrier) {
            vectorBarrier = barrier;
        }
    });
    Operation* previous = vectorBarrier ? vectorBarrier->getPrevNode() : nullptr;
    Operation* next = vectorBarrier ? vectorBarrier->getNextNode() : nullptr;
    while (isa_and_nonnull<SetFlagOp, WaitFlagOp>(previous)) {
        previous = previous->getPrevNode();
    }
    while (isa_and_nonnull<SetFlagOp, WaitFlagOp>(next)) {
        next = next->getNextNode();
    }
    const bool exactPlacement = computes.size() == 3 && vectorBarrier && previous == computes[1] && next == computes[2];
    return check(exactPlacement, "same-pipe barrier was not placed at the nonadjacent target frontier");
}

bool testEventCapacity(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kSharedFixture);
    if (!check(static_cast<bool>(module), "cannot parse event-capacity fixture")) {
        return false;
    }
    AnalysisFixture fixture(*module->getOps<func::FuncOp>().begin());
    if (!check(buildAnalysis(fixture), "cannot analyze event-capacity fixture")) {
        return false;
    }
    FailureOr<SyncInterpretationResult> residuals = interpretEmpty(fixture);
    if (!check(succeeded(residuals), "cannot interpret event-capacity fixture")) {
        return false;
    }
    FailureOr<SyncDirectRepairPlan> basePlan =
        buildDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations);
    if (!check(succeeded(basePlan), "cannot build event-capacity base plan")) {
        return false;
    }
    auto event = llvm::find_if(basePlan->candidates, [](const SyncDirectRepairCandidate& candidate) {
        return candidate.kind == SyncDirectRepairKind::DirectedEvent;
    });
    if (!check(event != basePlan->candidates.end(), "event-capacity fixture has no directed event")) {
        return false;
    }

    SyncDirectRepairPlan capacityPlan;
    capacityPlan.status = SyncDirectRepairPlanStatus::Ready;
    for (unsigned id = 0; id < 6; ++id) {
        SyncDirectRepairCandidate candidate = *event;
        candidate.id = id;
        candidate.eventId.reset();
        capacityPlan.candidates.push_back(std::move(candidate));
    }
    ProtocolSyncStatistics allocationStatistics;
    bool passed = check(
        succeeded(allocateDirectRepairEvents(fixture.schedule, capacityPlan, &allocationStatistics)) &&
            capacityPlan.status == SyncDirectRepairPlanStatus::Ready,
        "six direct events did not fit the compiler event pool");
    llvm::SmallVector<unsigned, 6> eventIds;
    for (const SyncDirectRepairCandidate& candidate : capacityPlan.candidates) {
        if (candidate.eventId) {
            eventIds.push_back(*candidate.eventId);
        }
    }
    llvm::sort(eventIds);
    passed &= check(
        eventIds == llvm::SmallVector<unsigned, 6>({0, 1, 2, 3, 4, 5}),
        "six direct events did not receive exactly compiler IDs 0..5");
    passed &= check(
        allocationStatistics.allocationGraphVertices == 6 && allocationStatistics.allocationGraphEdges == 15 &&
            allocationStatistics.maxEventDomainPressure == 6,
        "overlapping direct generations produced incorrect interference statistics");

    capacityPlan.status = SyncDirectRepairPlanStatus::Ready;
    for (SyncDirectRepairCandidate& candidate : capacityPlan.candidates) {
        candidate.eventId.reset();
    }
    SyncDirectRepairCandidate seventh = *event;
    seventh.id = capacityPlan.candidates.size();
    capacityPlan.candidates.push_back(std::move(seventh));
    passed &= check(
        succeeded(allocateDirectRepairEvents(fixture.schedule, capacityPlan)) &&
            capacityPlan.status == SyncDirectRepairPlanStatus::ResourceInfeasible &&
            capacityPlan.allocationFailure == SyncEventAllocationFailure::ConservativeInterference,
        "seventh direct event did not fail closed");
    passed &= check(
        llvm::none_of(
            capacityPlan.candidates,
            [](const SyncDirectRepairCandidate& candidate) { return candidate.eventId.has_value(); }),
        "resource-infeasible allocation retained a concrete event ID");
    passed &= check(
        capacityPlan.rejections.size() == 1 &&
            capacityPlan.rejections.front().reason == SyncDirectRepairRejection::EventCapacity,
        "resource-infeasible allocation has no canonical capacity rejection");

    capacityPlan.status = SyncDirectRepairPlanStatus::Ready;
    capacityPlan.rejections.clear();
    capacityPlan.allocationFailure = SyncEventAllocationFailure::None;
    capacityPlan.candidates.resize(129, *event);
    for (auto [id, candidate] : llvm::enumerate(capacityPlan.candidates)) {
        candidate.id = id;
        candidate.eventId.reset();
    }
    passed &= check(
        succeeded(allocateDirectRepairEvents(fixture.schedule, capacityPlan)) &&
            capacityPlan.status == SyncDirectRepairPlanStatus::AllocationAnalysisLimit &&
            capacityPlan.allocationFailure == SyncEventAllocationFailure::AnalysisLimit,
        "consumption-proof limit was not preserved as a handled direct allocation outcome");

    OwningOpRef<ModuleOp> reservationModule = parseFixture(context, kReservationFixture);
    if (!check(static_cast<bool>(reservationModule), "cannot parse reservation fixture")) {
        return false;
    }
    AnalysisFixture reservationFixture(*reservationModule->getOps<func::FuncOp>().begin());
    if (!check(buildAnalysis(reservationFixture), "cannot analyze reservation fixture")) {
        return false;
    }
    SyncDirectRepairPlan reservationPlan;
    reservationPlan.status = SyncDirectRepairPlanStatus::Ready;
    SyncDirectRepairCandidate reservedCandidate = *event;
    reservedCandidate.id = 0;
    reservedCandidate.sourcePipe = PIPE::PIPE_V;
    reservedCandidate.targetPipe = PIPE::PIPE_S;
    reservedCandidate.eventId.reset();
    reservationPlan.candidates.push_back(std::move(reservedCandidate));
    const bool reservationAllocated =
        succeeded(allocateDirectRepairEvents(reservationFixture.schedule, reservationPlan)) &&
        reservationPlan.candidates.front().eventId == 1;
    passed &= check(reservationAllocated, "direct allocator reused hidden event reservation zero");
    return passed;
}

bool testModuleAtomicity(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kAtomicFixture);
    if (!check(static_cast<bool>(module), "cannot parse module-atomicity fixture")) {
        return false;
    }
    const std::string before = printModule(*module);
    PTOProtocolSyncOptions options;
    options.executionMode = "direct-repair";
    options.fallbackMode = "fail";
    PassManager manager(&context);
    manager.addPass(createPTOProtocolSyncPass(options));
    LogicalResult result = failure();
    {
        ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
        result = manager.run(*module);
    }
    return check(failed(result), "mixed supported/unsupported module did not fail closed") &&
           check(printModule(*module) == before, "failed direct-repair module transaction changed the input");
}

} // namespace

int main()
{
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect, PTODialect, scf::SCFDialect>();
    MLIRContext context(registry);
    bool passed = testSharedRepairAndMaterialization(context);
    passed &= testVerifierRollbackAndRecurrence(context);
    passed &= testNonAdjacentSamePipe(context);
    passed &= testEventCapacity(context);
    passed &= testModuleAtomicity(context);
    passed &= testAcknowledgedEventReuse(context);
    passed &= testEventLoopAncestry(context);
    if (passed) {
        llvm::outs() << "protocol-sync direct independent readiness and overlap witness: pass\n";
        llvm::outs() << "protocol-sync direct full-world interpretation: pass\n";
        llvm::outs() << "protocol-sync direct verifier rollback: pass\n";
        llvm::outs() << "protocol-sync direct recurrence rejection: pass\n";
        llvm::outs() << "protocol-sync direct nonadjacent same-pipe: pass\n";
        llvm::outs() << "protocol-sync direct event capacity: pass\n";
        llvm::outs() << "protocol-sync direct reservation allocation: pass\n";
        llvm::outs() << "protocol-sync direct module atomicity: pass\n";
        llvm::outs() << "protocol-sync acknowledged event reuse and independent mutations: pass\n";
    }
    return passed ? 0 : 1;
}
