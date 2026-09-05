// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- pto-protocol-sync-mixed-test.cpp -------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"
#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

constexpr StringLiteral kFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @mixed(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %output: !pto.partition_tensor_view<16x16xf16>,
      %flat_input: !pto.partition_tensor_view<16x16xf16>,
      %flat_output: !pto.partition_tensor_view<16x16xf16>,
      %trip_count: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %addr0 = arith.constant 0 : i64
    %addr1 = arith.constant 1024 : i64
    %tile = pto.alloc_tile addr = %addr0 : !pto.tile_buf<vec, 16x16xf16>
    %flat_tile = pto.alloc_tile addr = %addr1 : !pto.tile_buf<vec, 16x16xf16>
    pto.tload ins(%flat_input : !pto.partition_tensor_view<16x16xf16>)
              outs(%flat_tile : !pto.tile_buf<vec, 16x16xf16>)
    pto.tstore ins(%flat_tile : !pto.tile_buf<vec, 16x16xf16>)
               outs(%flat_output : !pto.partition_tensor_view<16x16xf16>)
    scf.for %iteration = %c0 to %trip_count step %c1 {
      pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
                outs(%tile : !pto.tile_buf<vec, 16x16xf16>)
      pto.tstore ins(%tile : !pto.tile_buf<vec, 16x16xf16>)
                 outs(%output : !pto.partition_tensor_view<16x16xf16>)
    }
    return
  }
}
)mlir";

constexpr StringLiteral kAmbiguousGMFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @ambiguous_gm(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %output: !pto.partition_tensor_view<16x16xf16>,
      %trip_count: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0_index = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : i64
    %tile = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    scf.for %index = %c0_index to %trip_count step %c1 {
      pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
                outs(%tile : !pto.tile_buf<vec, 16x16xf16>)
      pto.tstore ins(%tile : !pto.tile_buf<vec, 16x16xf16>)
                 outs(%output : !pto.partition_tensor_view<16x16xf16>)
    }
    return
  }
}
)mlir";

constexpr StringLiteral kSharedFrontierFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func private @produce(
      %first: !pto.tile_buf<vec, 16x16xf16>,
      %second: !pto.tile_buf<vec, 16x16xf16>)
      attributes {pto.tileop.effects = ["write", "write"],
                  pto.tileop.helper, pto.tileop.kind = "vector"} {
    return
  }
  func.func private @consume(
      %first: !pto.tile_buf<vec, 16x16xf16>,
      %second: !pto.tile_buf<vec, 16x16xf16>)
      attributes {pto.tileop.effects = ["read", "read"],
                  pto.tileop.helper, pto.tileop.kind = "vector"} {
    return
  }
  func.func private @noise(%tile: !pto.tile_buf<vec, 16x16xf16>)
      attributes {pto.tileop.effects = ["none"],
                  pto.tileop.helper, pto.tileop.kind = "vector"} {
    return
  }
  func.func @shared_frontier()
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %c512 = arith.constant 512 : i64
    %c1024 = arith.constant 1024 : i64
    %first = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    %second = pto.alloc_tile addr = %c512 : !pto.tile_buf<vec, 16x16xf16>
    %temporary = pto.alloc_tile addr = %c1024 : !pto.tile_buf<vec, 16x16xf16>
    func.call @produce(%first, %second) :
      (!pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 16x16xf16>) -> ()
    func.call @noise(%temporary) : (!pto.tile_buf<vec, 16x16xf16>) -> ()
    func.call @consume(%first, %second) :
      (!pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 16x16xf16>) -> ()
    return
  }
}
)mlir";

constexpr StringLiteral kCandidateSubsetFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func private @consume(%tile: !pto.tile_buf<vec, 16x16xf16>)
      attributes {pto.tileop.effects = ["read"],
                  pto.tileop.helper, pto.tileop.kind = "vector"} {
    return
  }
  func.func @candidate_subset(%input: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %c512 = arith.constant 512 : i64
    %c1024 = arith.constant 1024 : i64
    %first = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    %second = pto.alloc_tile addr = %c512 : !pto.tile_buf<vec, 16x16xf16>
    %third = pto.alloc_tile addr = %c1024 : !pto.tile_buf<vec, 16x16xf16>
    pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
              outs(%first : !pto.tile_buf<vec, 16x16xf16>)
    func.call @consume(%first) : (!pto.tile_buf<vec, 16x16xf16>) -> ()
    pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
              outs(%second : !pto.tile_buf<vec, 16x16xf16>)
    pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
              outs(%third : !pto.tile_buf<vec, 16x16xf16>)
    func.call @consume(%second) : (!pto.tile_buf<vec, 16x16xf16>) -> ()
    func.call @consume(%third) : (!pto.tile_buf<vec, 16x16xf16>) -> ()
    return
  }
}
)mlir";

constexpr StringLiteral kDepthTwoFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func private @consume(%tile: !pto.tile_buf<vec, 16x16xf16>)
      attributes {pto.tileop.effects = ["read"],
                  pto.tileop.helper, pto.tileop.kind = "vector"} {
    return
  }
  func.func @ready_release_two(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %trip_count: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0_index = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : i64
    %buffers = pto.alloc_multi_tile addr = %c0
        : !pto.multi_tile_buf<vec, 16x16xf16, count=2>
    scf.for %index = %c0_index to %trip_count step %c1 {
      %selector = arith.remui %index, %c2 : index
      %slot = pto.multi_tile_get %buffers[%selector]
          : !pto.multi_tile_buf<vec, 16x16xf16, count=2>
         -> !pto.tile_buf<vec, 16x16xf16>
      pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
                outs(%slot : !pto.tile_buf<vec, 16x16xf16>)
      func.call @consume(%slot) : (!pto.tile_buf<vec, 16x16xf16>) -> ()
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

bool buildAnalysis(AnalysisFixture& fixture, bool proveDisjointGlobalArguments = true)
{
    func::FuncOp function = fixture.schedule.getFunction();
    if (failed(fixture.adapter.buildSnapshot(function, fixture.legacy))) {
        return false;
    }
    SyncSemanticContext sourceContext = fixture.adapter.buildSemanticContext(fixture.legacy);
    if (!proveDisjointGlobalArguments) {
        fixture.semanticContext = std::move(sourceContext);
    } else {
        llvm::DenseSet<Value> copied;
        auto copyStorage = [&](Value value) {
            if (!copied.insert(value).second) {
                return;
            }
            for (SyncStorageProvenance provenance : sourceContext.lookupStorage(value)) {
                auto rootArgument = dyn_cast<BlockArgument>(provenance.root);
                const bool isGlobalArgument = provenance.space == AddressSpace::GM && rootArgument &&
                                              rootArgument.getOwner() == &function.getBody().front();
                if (isGlobalArgument) {
                    provenance.physical = true;
                    provenance.unknownRange = false;
                    provenance.aliasesUnknownRange = false;
                    provenance.intervals = {{4096ULL * (rootArgument.getArgNumber() + 1), 512ULL}};
                }
                fixture.semanticContext.addStorage(value, std::move(provenance));
            }
        };
        for (Value argument : function.getArguments()) {
            copyStorage(argument);
        }
        function.walk([&](Operation* operation) {
            for (Value operand : operation->getOperands()) {
                copyStorage(operand);
            }
            for (Value result : operation->getResults()) {
                copyStorage(result);
            }
        });
    }
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

void buildIdentityMapping(func::FuncOp function, IRMapping& mapping)
{
    function.walk([&](Operation* operation) {
        mapping.map(operation, operation);
        for (Value result : operation->getResults()) {
            mapping.map(result, result);
        }
        for (Region& region : operation->getRegions()) {
            for (Block& block : region) {
                for (BlockArgument argument : block.getArguments()) {
                    mapping.map(argument, argument);
                }
            }
        }
    });
}

bool materializeMixed(AnalysisFixture& fixture, SyncMixedProtocolPlan& plan)
{
    FailureOr<SyncMixedProtocolPlan> selected =
        buildMixedProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    const bool invalidPlan = failed(selected) || selected->status != SyncMixedPlanStatus::Ready ||
                             failed(allocateMixedProtocolEvents(fixture.schedule, *selected));
    if (invalidPlan) {
        return false;
    }
    plan = std::move(*selected);
    func::FuncOp function = fixture.schedule.getFunction();
    IRMapping mapping;
    buildIdentityMapping(function, mapping);
    if (plan.oneShot && failed(materializeOneShotPublishPlan(function, mapping, *plan.oneShot))) {
        return false;
    }
    if (plan.readyRelease && failed(materializeReadyReleaseProtocolPlan(function, mapping, *plan.readyRelease))) {
        return false;
    }
    if (plan.directRepair.status == SyncDirectRepairPlanStatus::Ready &&
        failed(materializeDirectRepairPlan(function, mapping, plan.directRepair))) {
        return false;
    }
    return true;
}

bool runConcreteFault(
    MLIRContext& context, StringRef source, StringRef functionName,
    const std::function<bool(func::FuncOp, SyncMixedProtocolPlan&)>& mutation, bool expectedValid)
{
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
    if (!module || failed(verify(*module))) {
        return false;
    }
    AnalysisFixture fixture(module->lookupSymbol<func::FuncOp>(functionName));
    SyncMixedProtocolPlan plan;
    const bool invalidFixture = !buildAnalysis(fixture) || !materializeMixed(fixture, plan);
    if (invalidFixture) {
        return false;
    }
    func::FuncOp function = fixture.schedule.getFunction();
    if (!mutation(function, plan)) {
        return false;
    }
    const bool valid = succeeded(verifyConcreteSyncSemantics(fixture.semanticContext, function));
    return valid == expectedValid;
}

bool testAmbiguousGlobalArgumentsFailClosed(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kAmbiguousGMFixture, &context);
    if (!check(module && succeeded(verify(*module)), "cannot parse ambiguous-GM fixture")) {
        return false;
    }
    AnalysisFixture fixture(module->lookupSymbol<func::FuncOp>("ambiguous_gm"));
    if (!check(buildAnalysis(fixture, false), "cannot build ambiguous-GM analysis")) {
        return false;
    }
    FailureOr<SyncMixedProtocolPlan> plan =
        buildMixedProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    return check(
        succeeded(plan) && plan->status == SyncMixedPlanStatus::Unsupported,
        "mixed planning accepted potentially aliased GM arguments without visibility proof");
}

SyncDirectRepairCandidate* findExitRepair(SyncMixedProtocolPlan& plan)
{
    auto found = llvm::find_if(plan.directRepair.candidates, [](const SyncDirectRepairCandidate& candidate) {
        return candidate.kind == SyncDirectRepairKind::ExitBarrier;
    });
    return found == plan.directRepair.candidates.end() ? nullptr : &*found;
}

bool replaceOnce(std::string& text, StringRef from, StringRef to)
{
    const std::size_t position = text.find(from.str());
    if (position == std::string::npos) {
        return false;
    }
    text.replace(position, from.size(), to.str());
    return true;
}

bool allEventIdsClear(const SyncMixedProtocolPlan& plan)
{
    const bool oneShotClear = !plan.oneShot || llvm::all_of(plan.oneShot->candidates, [](const auto& candidate) {
        return !candidate.eventId;
    });
    const bool readyReleaseClear = !plan.readyRelease || llvm::all_of(plan.readyRelease->lanes, [](const auto& lane) {
        return !lane.readyEventId && !lane.releaseEventId;
    });
    const bool directClear =
        llvm::all_of(plan.directRepair.candidates, [](const auto& candidate) { return !candidate.eventId; });
    return oneShotClear && readyReleaseClear && directClear;
}

bool testMixedSelectionAndAllocation(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kFixture, &context);
    if (!check(module && succeeded(verify(*module)), "cannot parse mixed fixture")) {
        return false;
    }
    AnalysisFixture fixture(module->lookupSymbol<func::FuncOp>("mixed"));
    if (!check(buildAnalysis(fixture), "cannot build mixed analysis")) {
        return false;
    }
    FailureOr<SyncMixedProtocolPlan> plan =
        buildMixedProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    const bool selected = succeeded(plan) && plan->status == SyncMixedPlanStatus::Ready && plan->readyRelease &&
                          plan->oneShot && plan->oneShot->candidates.size() == 1 &&
                          plan->selectedWorldKind == SyncMixedWorldKind::CombinedProtocols &&
                          plan->initialResidualCount == 2 && plan->directRepair.candidates.size() == 1 &&
                          plan->candidateCountBeforeDeletion == 3 && plan->completeWorldsAttempted == 5 &&
                          plan->completeWorldsFeasible == 3 && plan->selectedCost.generatedEventPairs == 3 &&
                          plan->selectedCost.targetedBarriers == 0 && plan->selectedCost.fixedExitDrains == 1 &&
                          plan->selectedCost.staticActions == 8 && plan->reverseDeletionAttempts == 3 &&
                          plan->reverseDeletionRemoved == 0 &&
                          succeeded(verifyMixedProtocolPlan(
                              fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan));
    if (!check(selected, "mixed protocol/direct selection is not exact")) {
        if (succeeded(plan)) {
            printMixedProtocolPlan(fixture.schedule.getFunction(), *plan, llvm::errs());
        }
        return false;
    }

    SyncMixedProtocolPlan unallocated = *plan;
    SyncMixedProtocolPlan forgedScarcity = unallocated;
    forgedScarcity.status = SyncMixedPlanStatus::ResourceInfeasible;
    forgedScarcity.completeWorldsFeasible = 0;
    forgedScarcity.failures = {{SyncMixedPlanRejection::EventCapacity, "injected false event scarcity"}};
    if (!check(
            failed(verifyMixedProtocolPlan(
                fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, forgedScarcity)),
            "mixed verifier accepted feasible allocation relabeled as event scarcity")) {
        return false;
    }
    SyncMixedProtocolPlan disguisedInternalFailure = unallocated;
    const SyncObligationId uncovered = disguisedInternalFailure.directObligations.front().id;
    disguisedInternalFailure.directRepair.rejections = {
        {uncovered, SyncDirectRepairRejection::InternalInvariant, "injected internal invariant"}};
    disguisedInternalFailure.status = SyncMixedPlanStatus::Unsupported;
    disguisedInternalFailure.failures = {
        {SyncMixedPlanRejection::IncompleteDirectRepair, "disguised internal invariant"}};
    const std::string beforeInternalFailure = printModule(*module);
    const bool internalFailureRejected = failed(verifyMixedProtocolPlan(
        fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, disguisedInternalFailure));
    const bool internalFailureAtomic = failed(materializeAndVerifyMixedProtocolPlanInDisposableModule(
        fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, disguisedInternalFailure,
        fixture.semanticContext));
    if (!check(
            internalFailureRejected && internalFailureAtomic && beforeInternalFailure == printModule(*module),
            "mixed verifier accepted a fallback-eligible disguised internal invariant")) {
        return false;
    }

    const bool allocated = succeeded(allocateMixedProtocolEvents(fixture.schedule, *plan)) &&
                           plan->status == SyncMixedPlanStatus::Ready && plan->readyRelease->lanes.size() == 1 &&
                           plan->readyRelease->lanes.front().readyEventId == 1 &&
                           plan->readyRelease->lanes.front().releaseEventId == 0;
    const bool collisionAvoided = plan->oneShot->candidates.front().eventId == 0;
    if (!check(
            allocated && collisionAvoided &&
                succeeded(verifyMixedProtocolPlan(
                    fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan)),
            "combined allocation did not avoid the protocol-owned event")) {
        return false;
    }

    SyncMixedProtocolPlan exhausted = unallocated;
    if (!check(exhausted.oneShot && !exhausted.oneShot->candidates.empty(), "mixed fixture has no one-shot event")) {
        return false;
    }
    const SyncOneShotPublishCandidate baseEvent = exhausted.oneShot->candidates.front();
    for (unsigned index = 1; index < 6; ++index) {
        SyncOneShotPublishCandidate extra = baseEvent;
        extra.id = exhausted.oneShot->candidates.size();
        exhausted.oneShot->candidates.push_back(std::move(extra));
    }
    const bool rejectedAtomically = succeeded(allocateMixedProtocolEvents(fixture.schedule, exhausted)) &&
                                    exhausted.status == SyncMixedPlanStatus::ResourceInfeasible &&
                                    allEventIdsClear(exhausted);
    if (!check(rejectedAtomically, "combined event exhaustion retained a partial allocation")) {
        return false;
    }

    SyncMixedProtocolPlan analysisLimited = unallocated;
    if (!check(
            analysisLimited.oneShot && !analysisLimited.oneShot->candidates.empty(),
            "mixed fixture has no analysis-limit event candidate")) {
        return false;
    }
    const SyncOneShotPublishCandidate limitedBaseEvent = analysisLimited.oneShot->candidates.front();
    for (unsigned index = 1; index < 1025; ++index) {
        SyncOneShotPublishCandidate extra = limitedBaseEvent;
        extra.id = analysisLimited.oneShot->candidates.size();
        analysisLimited.oneShot->candidates.push_back(std::move(extra));
    }
    ProtocolSyncStatistics limitStatistics;
    const bool analysisLimitPreserved =
        failed(allocateMixedProtocolEvents(fixture.schedule, analysisLimited, &limitStatistics)) &&
        analysisLimited.status == SyncMixedPlanStatus::Ready && allEventIdsClear(analysisLimited) &&
        limitStatistics.allocationGraphVertices >= 1025 && limitStatistics.allocationSearchLimitHits == 1;
    if (!check(analysisLimitPreserved, "mixed analysis limit lost atomicity or allocation statistics")) {
        return false;
    }

    SyncMixedProtocolPlan malformed = *plan;
    malformed.oneShot->candidates.front().eventId = malformed.readyRelease->lanes.front().readyEventId;
    const std::string before = printModule(*module);
    const bool rejectedCollision = failed(
        verifyMixedProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, malformed));
    const bool rejectedBeforeMutation = failed(materializeAndVerifyMixedProtocolPlanInDisposableModule(
        fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, malformed, fixture.semanticContext));
    if (!check(
            rejectedCollision && rejectedBeforeMutation && before == printModule(*module),
            "mixed verifier fault injection changed the source module")) {
        return false;
    }

    if (!check(
            succeeded(materializeAndVerifyMixedProtocolPlanInDisposableModule(
                fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan,
                fixture.semanticContext)) &&
                succeeded(verify(*module)),
            "valid mixed materialization failed independent verification")) {
        return false;
    }
    unsigned generated = 0;
    unsigned tails = 0;
    fixture.schedule.getFunction().walk([&](Operation* operation) {
        generated += operation->hasAttrOfType<UnitAttr>("pto.protocol_sync.generated") ? 1U : 0U;
        if (auto barrier = dyn_cast<BarrierOp>(operation)) {
            tails += barrier.getPipe().getPipe() == PIPE::PIPE_ALL ? 1U : 0U;
        }
    });
    return check(generated == 9 && tails == 1, "mixed materialization has an unexpected action set");
}

bool testSuccessfulReverseDeletion(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kFixture, &context);
    if (!check(module && succeeded(verify(*module)), "cannot parse reverse-deletion fixture")) {
        return false;
    }
    AnalysisFixture fixture(module->lookupSymbol<func::FuncOp>("mixed"));
    if (!check(buildAnalysis(fixture), "cannot build reverse-deletion analysis")) {
        return false;
    }
    FailureOr<SyncMixedProtocolPlan> plan =
        buildMixedProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    const bool basePlanComplete = succeeded(plan) && plan->isComplete();
    if (!check(basePlanComplete, "cannot build reverse-deletion base plan")) {
        return false;
    }

    SyncDirectRepairCandidate* exit = findExitRepair(*plan);
    const bool sharedExitRepair = exit && exit->obligations.size() == 2;
    if (!check(sharedExitRepair, "reverse-deletion fixture needs one shared exit recipe")) {
        return false;
    }
    SyncDirectRepairCandidate completeExit = *exit;
    exit->obligations = {completeExit.obligations.front()};
    completeExit.id = plan->directRepair.candidates.size();
    plan->directRepair.candidates.push_back(std::move(completeExit));

    const bool selected = succeeded(selectMixedProtocolCandidates(
                              fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan)) &&
                          plan->candidateCountBeforeDeletion == 4 && plan->reverseDeletionAttempts == 4 &&
                          plan->reverseDeletionRemoved == 1 && plan->directRepair.candidates.size() == 1;
    SyncDirectRepairCandidate* retainedExit = findExitRepair(*plan);
    const bool remapped = retainedExit && retainedExit->id == 0 && retainedExit->obligations.size() == 2 &&
                          retainedExit->obligations[0] == 0 && retainedExit->obligations[1] == 1;
    FailureOr<SyncInterpretationResult> exact = interpretSelectedWorld(
        fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, plan->selectedWorld);
    if (!check(
            selected && remapped && succeeded(exact) && exact->isComplete() &&
                succeeded(verifyMixedProtocolPlan(
                    fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan)),
            "reverse deletion did not retain a canonical complete candidate set")) {
        return false;
    }
    if (!check(
            succeeded(allocateMixedProtocolEvents(fixture.schedule, *plan)) &&
                succeeded(verifyMixedProtocolPlan(
                    fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan)) &&
                succeeded(materializeAndVerifyMixedProtocolPlanInDisposableModule(
                    fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan,
                    fixture.semanticContext)),
            "reverse-deleted plan did not allocate and materialize")) {
        return false;
    }
    unsigned generated = 0;
    fixture.schedule.getFunction().walk([&](Operation* operation) {
        generated += operation->hasAttrOfType<UnitAttr>("pto.protocol_sync.generated") ? 1U : 0U;
    });
    return check(generated == 9, "reverse-deleted plan materialized an unexpected action set");
}

bool testReadyReleaseFamilyIsolation(MLIRContext& context)
{
    std::string unrelatedDepthThreeFamily(kFixture);
    if (!check(
            replaceOnce(
                unrelatedDepthThreeFamily, "%flat_tile = pto.alloc_tile addr = %addr1 : !pto.tile_buf<vec, 16x16xf16>",
                "%flat_tiles = pto.alloc_multi_tile addr = %addr1\n"
                "        : !pto.multi_tile_buf<vec, 16x16xf16, count=3>\n"
                "    %flat_tile = pto.multi_tile_get %flat_tiles[%c0]\n"
                "        : !pto.multi_tile_buf<vec, 16x16xf16, count=3>\n"
                "       -> !pto.tile_buf<vec, 16x16xf16>"),
            "cannot prepare unrelated-family fixture")) {
        return false;
    }
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(unrelatedDepthThreeFamily, &context);
    if (!check(module && succeeded(verify(*module)), "cannot parse unrelated-family fixture")) {
        return false;
    }
    AnalysisFixture fixture(module->lookupSymbol<func::FuncOp>("mixed"));
    if (!check(buildAnalysis(fixture), "cannot build unrelated-family analysis")) {
        return false;
    }
    FailureOr<SyncReadyReleasePlan> plan = buildReadyReleaseProtocolPlan(
        fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, nullptr,
        SyncReadyReleasePlanningScope::MixedCandidate);
    if (!check(
            succeeded(plan) && plan->status == SyncReadyReleasePlanStatus::Ready &&
                succeeded(allocateReadyReleaseProtocolEvents(fixture.schedule, *plan)),
            "unrelated outside-loop family prevented ReadyRelease selection")) {
        return false;
    }
    func::FuncOp function = fixture.schedule.getFunction();
    IRMapping mapping;
    OwningOpRef<ModuleOp> stagingModule = ModuleOp::create(function.getLoc());
    func::FuncOp clone = cast<func::FuncOp>(function->clone(mapping));
    stagingModule->push_back(clone);
    if (!check(
            succeeded(materializeReadyReleaseProtocolPlan(clone, mapping, *plan)) &&
                succeeded(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
            "ReadyRelease verifier rejected an unrelated outside-loop family")) {
        return false;
    }

    std::string sharedFamily(kFixture);
    const bool replacedOutput = replaceOnce(sharedFamily, "outs(%flat_tile", "outs(%tile");
    const bool replacedInput = replaceOnce(sharedFamily, "ins(%flat_tile", "ins(%tile");
    if (!check(replacedOutput && replacedInput, "cannot prepare shared-family fixture")) {
        return false;
    }
    module = parseSourceString<ModuleOp>(sharedFamily, &context);
    if (!check(module && succeeded(verify(*module)), "cannot parse shared-family fixture")) {
        return false;
    }
    AnalysisFixture sharedFixture(module->lookupSymbol<func::FuncOp>("mixed"));
    if (!check(buildAnalysis(sharedFixture), "cannot build shared-family analysis")) {
        return false;
    }
    plan = buildReadyReleaseProtocolPlan(
        sharedFixture.schedule, *sharedFixture.stages, sharedFixture.timelines, sharedFixture.channels, nullptr,
        SyncReadyReleasePlanningScope::MixedCandidate);
    const bool rejectedSharedFamily =
        succeeded(plan) && plan->status == SyncReadyReleasePlanStatus::Unsupported && !plan->rejections.empty() &&
        llvm::none_of(plan->rejections, [](const SyncReadyReleasePlanRejection& rejection) {
            return rejection.reason == SyncReadyReleaseRejection::InternalInvariant;
        });
    return check(rejectedSharedFamily, "planner admitted a ReadyRelease family used outside its carrier loop");
}

bool testCompleteWorldCompetitionAndSharedFrontier(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kSharedFrontierFixture, &context);
    if (!check(module && succeeded(verify(*module)), "cannot parse shared-frontier fixture")) {
        return false;
    }
    AnalysisFixture fixture(module->lookupSymbol<func::FuncOp>("shared_frontier"));
    if (!check(buildAnalysis(fixture), "cannot build shared-frontier analysis")) {
        return false;
    }
    FailureOr<SyncOneShotPublishPlan> candidates =
        buildOneShotPublishCandidates(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    const bool grouped = succeeded(candidates) && candidates->candidates.size() == 1 &&
                         candidates->candidates.front().channels.size() == 2 &&
                         candidates->candidates.front().generations.size() == 2 &&
                         candidates->candidates.front().kind == SyncOneShotPublishKind::PipeBarrier;
    if (!check(grouped, "shared publication/acquisition frontiers were not grouped")) {
        return false;
    }

    FailureOr<SyncMixedProtocolPlan> protocol =
        buildMixedProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, true);
    FailureOr<SyncMixedProtocolPlan> direct =
        buildMixedProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, false);
    const bool worldsCompete =
        succeeded(protocol) && protocol->status == SyncMixedPlanStatus::Ready && protocol->oneShot &&
        protocol->selectedWorldKind == SyncMixedWorldKind::OneShotPublish && protocol->completeWorldsAttempted == 2 &&
        protocol->completeWorldsFeasible == 2 && protocol->directRepair.candidates.size() == 1 &&
        protocol->selectedCost.generatedEventPairs == 0 && protocol->selectedCost.targetedBarriers == 1 &&
        protocol->selectedCost.fixedExitDrains == 1 && protocol->selectedCost.staticActions == 1 && succeeded(direct) &&
        direct->status == SyncMixedPlanStatus::Ready && !direct->hasProtocol() &&
        direct->selectedWorldKind == SyncMixedWorldKind::DirectOnly && direct->completeWorldsAttempted == 1 &&
        direct->completeWorldsFeasible == 1 && direct->directRepair.candidates.size() == 2;
    if (!check(worldsCompete, "complete protocol and freshly repaired direct worlds did not compete")) {
        return false;
    }
    if (!check(
            succeeded(verifyMixedProtocolPlan(
                fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *protocol)) &&
                succeeded(verifyMixedProtocolPlan(
                    fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *direct)),
            "complete-world verifier did not reproduce the selected alternatives")) {
        return false;
    }
    SyncMixedProtocolPlan corruptedCost = *protocol;
    ++corruptedCost.selectedCost.fixedExitDrains;
    SyncMixedProtocolPlan corruptedWorldCount = *protocol;
    ++corruptedWorldCount.completeWorldsAttempted;
    if (!check(
            failed(verifyMixedProtocolPlan(
                fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, corruptedCost)) &&
                failed(verifyMixedProtocolPlan(
                    fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, corruptedWorldCount)),
            "complete-world verifier accepted corrupted selection evidence")) {
        return false;
    }
    if (!check(
            succeeded(allocateMixedProtocolEvents(fixture.schedule, *protocol)) &&
                succeeded(materializeAndVerifyMixedProtocolPlanInDisposableModule(
                    fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *protocol,
                    fixture.semanticContext)),
            "shared-frontier world did not allocate and materialize")) {
        return false;
    }
    unsigned generated = 0;
    fixture.schedule.getFunction().walk([&](Operation* operation) {
        generated += operation->hasAttrOfType<UnitAttr>("pto.protocol_sync.generated") ? 1U : 0U;
    });
    return check(generated == 2, "shared-frontier world emitted more than one barrier plus its fixed drain");
}

bool testCandidateGranularitySelection(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kCandidateSubsetFixture, &context);
    if (!check(module && succeeded(verify(*module)), "cannot parse candidate-subset fixture")) {
        return false;
    }
    AnalysisFixture fixture(module->lookupSymbol<func::FuncOp>("candidate_subset"));
    if (!check(buildAnalysis(fixture), "cannot build candidate-subset analysis")) {
        return false;
    }
    FailureOr<SyncOneShotPublishPlan> candidates =
        buildOneShotPublishCandidates(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    if (!check(
            succeeded(candidates) && candidates->candidates.size() == 3,
            "candidate-subset fixture did not expose three atomic protocols")) {
        return false;
    }
    FailureOr<SyncMixedProtocolPlan> plan =
        buildMixedProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    const SyncDirectRepairCandidate* event = nullptr;
    if (succeeded(plan)) {
        auto found = llvm::find_if(plan->directRepair.candidates, [](const SyncDirectRepairCandidate& candidate) {
            return candidate.kind == SyncDirectRepairKind::DirectedEvent;
        });
        event = found == plan->directRepair.candidates.end() ? nullptr : &*found;
    }
    const bool selectedSubset =
        succeeded(plan) && plan->status == SyncMixedPlanStatus::Ready && plan->oneShot &&
        plan->oneShot->candidates.size() == 1 && plan->oneShot->candidates.front().sourcePhase == 0 &&
        plan->oneShot->candidates.front().targetPhase == 1 && event && event->obligations.size() == 2 &&
        plan->selectedCost.generatedEventPairs == 2 && plan->selectedCost.fixedExitDrains == 1 &&
        plan->completeWorldsAttempted == 6 && plan->completeWorldsFeasible == 6;
    if (!check(selectedSubset, "selector did not retain one protocol and directly repair the other channels")) {
        return false;
    }
    return check(
        succeeded(
            verifyMixedProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan)),
        "candidate-granularity selection was not reproducible");
}

bool testConcreteVerifierFaultInjection(MLIRContext& context)
{
    const bool tagsAreDiagnosticOnly = runConcreteFault(
        context, kFixture, "mixed",
        [](func::FuncOp function, SyncMixedProtocolPlan&) {
            function.walk([&](Operation* operation) {
                operation->removeAttr("pto.protocol_sync.generated");
                operation->removeAttr("pto.protocol_sync.protocol_id");
                operation->removeAttr("pto.protocol_sync.protocol_kind");
                operation->removeAttr("pto.protocol_sync.direct_candidate_id");
                operation->removeAttr("pto.protocol_sync.role");
                operation->removeAttr("pto.protocol_sync.logical_lane");
                operation->removeAttr("pto.protocol_sync.logical_lanes");
            });
            return true;
        },
        true);
    if (!check(tagsAreDiagnosticOnly, "concrete verifier treated diagnostic tags as semantic authority")) {
        return false;
    }

    const bool adjacentProtocolsCompose = runConcreteFault(
        context, kFixture, "mixed",
        [](func::FuncOp function, SyncMixedProtocolPlan& plan) {
            if (!plan.oneShot || !plan.readyRelease) {
                return false;
            }
            auto found = llvm::find_if(plan.oneShot->candidates, [](const SyncOneShotPublishCandidate& candidate) {
                return candidate.kind == SyncOneShotPublishKind::DirectedEvent;
            });
            if (found == plan.oneShot->candidates.end()) {
                return false;
            }
            Operation* wait = found->targetOperation->getPrevNode();
            BarrierOp tail;
            function.walk([&](BarrierOp barrier) {
                const bool isTail = barrier.getPipe().getPipe() == PIPE::PIPE_ALL;
                if (isTail) {
                    tail = barrier;
                }
            });
            const bool invalidBoundary = !isa_and_nonnull<WaitFlagOp>(wait) || !tail;
            if (invalidBoundary) {
                return false;
            }
            found->targetOperation->moveBefore(tail);
            wait->moveBefore(found->targetOperation);
            return true;
        },
        true);
    if (!check(
            adjacentProtocolsCompose,
            "concrete verifier conflated an adjacent loop-spanning event with ReadyRelease boundary actions")) {
        return false;
    }

    const bool earlySetRejected = runConcreteFault(
        context, kFixture, "mixed",
        [](func::FuncOp, SyncMixedProtocolPlan& plan) {
            if (!plan.oneShot) {
                return false;
            }
            auto found = llvm::find_if(plan.oneShot->candidates, [](const SyncOneShotPublishCandidate& candidate) {
                return candidate.kind == SyncOneShotPublishKind::DirectedEvent;
            });
            if (found == plan.oneShot->candidates.end()) {
                return false;
            }
            Operation* set = found->sourceOperation->getNextNode();
            if (!isa_and_nonnull<SetFlagOp>(set)) {
                return false;
            }
            set->moveBefore(found->sourceOperation);
            return true;
        },
        false);
    if (!check(earlySetRejected, "concrete verifier accepted a set before the final producer")) {
        return false;
    }

    const bool lateWaitRejected = runConcreteFault(
        context, kFixture, "mixed",
        [](func::FuncOp, SyncMixedProtocolPlan& plan) {
            if (!plan.oneShot) {
                return false;
            }
            auto found = llvm::find_if(plan.oneShot->candidates, [](const SyncOneShotPublishCandidate& candidate) {
                return candidate.kind == SyncOneShotPublishKind::DirectedEvent;
            });
            if (found == plan.oneShot->candidates.end()) {
                return false;
            }
            Operation* wait = found->targetOperation->getPrevNode();
            if (!isa_and_nonnull<WaitFlagOp>(wait)) {
                return false;
            }
            wait->moveAfter(found->targetOperation);
            return true;
        },
        false);
    if (!check(lateWaitRejected, "concrete verifier accepted a wait after the first consumer")) {
        return false;
    }

    const bool directionRejected = runConcreteFault(
        context, kFixture, "mixed",
        [](func::FuncOp function, SyncMixedProtocolPlan& plan) {
            if (!plan.oneShot) {
                return false;
            }
            auto found = llvm::find_if(plan.oneShot->candidates, [](const SyncOneShotPublishCandidate& candidate) {
                return candidate.kind == SyncOneShotPublishKind::DirectedEvent;
            });
            if (found == plan.oneShot->candidates.end()) {
                return false;
            }
            auto set = dyn_cast_or_null<SetFlagOp>(found->sourceOperation->getNextNode());
            if (!set) {
                return false;
            }
            set.setSrcPipeAttr(PipeAttr::get(function.getContext(), PIPE::PIPE_V));
            return true;
        },
        false);
    if (!check(directionRejected, "concrete verifier accepted a mismatched event direction")) {
        return false;
    }

    const bool overlappingIdRejected = runConcreteFault(
        context, kFixture, "mixed",
        [](func::FuncOp, SyncMixedProtocolPlan& plan) {
            if (!plan.oneShot) {
                return false;
            }
            auto found = llvm::find_if(plan.oneShot->candidates, [](const SyncOneShotPublishCandidate& candidate) {
                return candidate.kind == SyncOneShotPublishKind::DirectedEvent;
            });
            if (found == plan.oneShot->candidates.end()) {
                return false;
            }
            Operation* set = found->sourceOperation->getNextNode();
            Operation* wait = found->targetOperation->getPrevNode();
            const bool invalidPair = !isa_and_nonnull<SetFlagOp>(set) || !isa_and_nonnull<WaitFlagOp>(wait);
            if (invalidPair) {
                return false;
            }
            Operation* overlapping = set->clone();
            OpBuilder builder(wait);
            builder.insert(overlapping);
            return true;
        },
        false);
    if (!check(overlappingIdRejected, "concrete verifier accepted overlapping same-ID generations")) {
        return false;
    }

    const bool changedAccessRejected = runConcreteFault(
        context, kFixture, "mixed",
        [](func::FuncOp function, SyncMixedProtocolPlan&) {
            SmallVector<TStoreOp, 2> stores;
            function.walk([&](TStoreOp store) { stores.push_back(store); });
            const bool invalidStoreSet = stores.size() != 2;
            if (invalidStoreSet) {
                return false;
            }
            stores[1]->setOperand(0, stores[0]->getOperand(0));
            return true;
        },
        false);
    if (!check(changedAccessRejected, "concrete verifier trusted protocol tags after a storage-access change")) {
        return false;
    }

    const bool missingPrimeRejected = runConcreteFault(
        context, kFixture, "mixed",
        [](func::FuncOp function, SyncMixedProtocolPlan&) {
            scf::ForOp loop;
            function.walk([&](scf::ForOp candidate) { loop = candidate; });
            Operation* prime = loop ? loop->getPrevNode() : nullptr;
            if (!isa_and_nonnull<SetFlagOp>(prime)) {
                return false;
            }
            prime->erase();
            return true;
        },
        false);
    if (!check(missingPrimeRejected, "concrete verifier accepted an incomplete prime/body/drain protocol")) {
        return false;
    }

    OwningOpRef<ModuleOp> depthTwoModule = parseSourceString<ModuleOp>(kDepthTwoFixture, &context);
    if (!check(depthTwoModule && succeeded(verify(*depthTwoModule)), "cannot parse depth-two verifier fixture")) {
        return false;
    }
    AnalysisFixture depthTwo(depthTwoModule->lookupSymbol<func::FuncOp>("ready_release_two"));
    if (!check(buildAnalysis(depthTwo), "cannot build depth-two verifier analysis")) {
        return false;
    }
    FailureOr<SyncReadyReleasePlan> readyRelease =
        buildReadyReleaseProtocolPlan(depthTwo.schedule, *depthTwo.stages, depthTwo.timelines, depthTwo.channels);
    if (!check(
            succeeded(readyRelease) && readyRelease->status == SyncReadyReleasePlanStatus::Ready &&
                succeeded(allocateReadyReleaseProtocolEvents(depthTwo.schedule, *readyRelease)),
            "cannot prepare depth-two concrete verifier protocol")) {
        return false;
    }
    func::FuncOp depthTwoFunction = depthTwo.schedule.getFunction();
    IRMapping depthTwoMapping;
    buildIdentityMapping(depthTwoFunction, depthTwoMapping);
    if (!check(
            succeeded(materializeReadyReleaseProtocolPlan(depthTwoFunction, depthTwoMapping, *readyRelease)) &&
                succeeded(verifyConcreteSyncSemantics(depthTwo.semanticContext, depthTwoFunction)),
            "concrete verifier rejected valid depth-two protocol")) {
        return false;
    }
    scf::ForOp loop;
    depthTwoFunction.walk([&](scf::ForOp candidate) { loop = candidate; });
    SmallVector<Operation*, 4> bodyEvents;
    for (Operation& operation : loop.getBody()->without_terminator()) {
        if (isa<SetFlagDynOp, WaitFlagDynOp>(&operation)) {
            bodyEvents.push_back(&operation);
        }
    }
    const bool invalidBodyProtocol = bodyEvents.size() != 4;
    if (!check(!invalidBodyProtocol, "depth-two verifier fixture has no dynamic body protocol")) {
        return false;
    }
    auto releaseWait = cast<WaitFlagDynOp>(bodyEvents[0]);
    auto releaseSet = cast<SetFlagDynOp>(bodyEvents[3]);
    auto selector = releaseWait.getEventId().getDefiningOp<arith::SelectOp>();
    if (!check(selector, "depth-two verifier fixture has no release event selector")) {
        return false;
    }
    OpBuilder builder(releaseSet);
    Value swapped = builder.create<arith::SelectOp>(
        releaseSet.getLoc(), selector.getCondition(), selector.getFalseValue(), selector.getTrueValue());
    releaseSet->setOperand(0, swapped);
    const bool swappedLanesRejected = failed(verifyConcreteSyncSemantics(depthTwo.semanticContext, depthTwoFunction));
    return check(swappedLanesRejected, "concrete verifier accepted swapped ReadyRelease lane IDs");
}

} // namespace

int main()
{
    DialectRegistry registry;
    registry.insert<PTODialect, arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
    MLIRContext context(registry);
    context.loadAllAvailableDialects();
    if (!testAmbiguousGlobalArgumentsFailClosed(context)) {
        return 1;
    }
    if (!testMixedSelectionAndAllocation(context)) {
        return 1;
    }
    if (!testSuccessfulReverseDeletion(context)) {
        return 1;
    }
    if (!testReadyReleaseFamilyIsolation(context)) {
        return 1;
    }
    if (!testCompleteWorldCompetitionAndSharedFrontier(context)) {
        return 1;
    }
    if (!testCandidateGranularitySelection(context)) {
        return 1;
    }
    if (!testConcreteVerifierFaultInjection(context)) {
        return 1;
    }
    llvm::outs() << "protocol-sync mixed ambiguous GM fail-closed: pass\n";
    llvm::outs() << "protocol-sync mixed selection and reverse deletion: pass\n";
    llvm::outs() << "protocol-sync successful whole-candidate deletion: pass\n";
    llvm::outs() << "protocol-sync mixed combined event allocation: pass\n";
    llvm::outs() << "protocol-sync mixed independent verification: pass\n";
    llvm::outs() << "protocol-sync ReadyRelease family isolation: pass\n";
    llvm::outs() << "protocol-sync complete-world competition and shared frontier: pass\n";
    llvm::outs() << "protocol-sync candidate-granularity world selection: pass\n";
    llvm::outs() << "protocol-sync concrete emitted-IR verifier negatives: pass\n";
    llvm::outs() << "protocol-sync mixed atomic materialization: pass\n";
    return 0;
}
