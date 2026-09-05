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

namespace {

constexpr StringLiteral kSharedFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @shared_frontiers(
      %input0: !pto.partition_tensor_view<16x16xf16>,
      %input1: !pto.partition_tensor_view<16x16xf16>,
      %output0: !pto.partition_tensor_view<16x16xf16>,
      %output1: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
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
    const unsigned sharedEvents = llvm::count_if(plan->candidates, [](const SyncDirectRepairCandidate& candidate) {
        return candidate.kind == SyncDirectRepairKind::DirectedEvent && candidate.obligations.size() == 2;
    });
    const unsigned exitCandidates = llvm::count_if(plan->candidates, [](const SyncDirectRepairCandidate& candidate) {
        return candidate.kind == SyncDirectRepairKind::ExitBarrier && candidate.obligations.size() >= 2;
    });
    passed &= check(sharedEvents == 2, "overlapping intervals did not share both directed frontiers");
    passed &= check(exitCandidates == 1, "terminal phases did not share one exit drain");

    SyncDirectRepairPlan inflated = *plan;
    auto shared = llvm::find_if(inflated.candidates, [](const SyncDirectRepairCandidate& candidate) {
        return candidate.kind == SyncDirectRepairKind::DirectedEvent && candidate.obligations.size() == 2;
    });
    if (!check(shared != inflated.candidates.end(), "cannot construct inflated-frontier negative")) {
        return false;
    }
    const auto setSingleObligation = [&](SyncDirectRepairCandidate& candidate, SyncObligationId id) {
        const SyncResidualObligation& obligation = residuals->obligations[id];
        const SyncPhase* source = fixture.schedule.findPhase(obligation.source);
        const SyncPhase* target = fixture.schedule.findPhase(obligation.target);
        candidate.sourcePhase = source->id;
        candidate.targetPhase = target->id;
        candidate.sourceOperation = source->operation;
        candidate.targetOperation = target->operation;
        candidate.obligations.assign(1, id);
    };
    SyncDirectRepairCandidate split = *shared;
    setSingleObligation(*shared, shared->obligations.front());
    setSingleObligation(split, split.obligations.back());
    inflated.candidates.insert(std::next(shared), std::move(split));
    for (auto [id, candidate] : llvm::enumerate(inflated.candidates)) {
        candidate.id = id;
    }
    passed &= check(
        failed(verifyDirectRepairPlan(fixture.schedule, *fixture.stages, residuals->obligations, inflated)),
        "independent verifier accepted two mergeable direct frontiers");

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
    return passed;
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
    const bool exactPlacement = computes.size() == 3 && vectorBarrier && vectorBarrier->getPrevNode() == computes[1] &&
                                vectorBarrier->getNextNode() == computes[2];
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
            capacityPlan.status == SyncDirectRepairPlanStatus::ResourceInfeasible,
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
    if (passed) {
        llvm::outs() << "protocol-sync direct shared frontiers: pass\n";
        llvm::outs() << "protocol-sync direct full-world interpretation: pass\n";
        llvm::outs() << "protocol-sync direct verifier rollback: pass\n";
        llvm::outs() << "protocol-sync direct recurrence rejection: pass\n";
        llvm::outs() << "protocol-sync direct nonadjacent same-pipe: pass\n";
        llvm::outs() << "protocol-sync direct event capacity: pass\n";
        llvm::outs() << "protocol-sync direct reservation allocation: pass\n";
        llvm::outs() << "protocol-sync direct module atomicity: pass\n";
    }
    return passed ? 0 : 1;
}
