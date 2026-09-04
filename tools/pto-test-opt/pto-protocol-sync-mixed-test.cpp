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
#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

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

SyncDirectRepairCandidate* findDirectedRepair(SyncMixedProtocolPlan& plan)
{
    auto found = llvm::find_if(plan.directRepair.candidates, [](const SyncDirectRepairCandidate& candidate) {
        return candidate.kind == SyncDirectRepairKind::DirectedEvent;
    });
    return found == plan.directRepair.candidates.end() ? nullptr : &*found;
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
    const bool readyReleaseClear = !plan.readyRelease || llvm::all_of(plan.readyRelease->lanes, [](const auto& lane) {
        return !lane.readyEventId && !lane.releaseEventId;
    });
    const bool directClear =
        llvm::all_of(plan.directRepair.candidates, [](const auto& candidate) { return !candidate.eventId; });
    return readyReleaseClear && directClear;
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
                          !plan->oneShot && plan->initialResidualCount == 3 &&
                          plan->directRepair.candidates.size() == 2 && plan->candidateCountBeforeDeletion == 3 &&
                          plan->reverseDeletionAttempts == 3 && plan->reverseDeletionRemoved == 0 &&
                          succeeded(verifyMixedProtocolPlan(
                              fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan));
    if (!check(selected, "mixed protocol/direct selection is not exact")) {
        return false;
    }

    SyncMixedProtocolPlan unallocated = *plan;
    SyncMixedProtocolPlan disguisedInternalFailure = unallocated;
    SyncDirectRepairCandidate* removed = findDirectedRepair(disguisedInternalFailure);
    const bool singularDirectRepair = removed && removed->obligations.size() == 1;
    if (!check(singularDirectRepair, "mixed fixture has no singular direct event repair")) {
        return false;
    }
    const SyncObligationId uncovered = removed->obligations.front();
    const std::size_t removedIndex =
        static_cast<std::size_t>(removed - disguisedInternalFailure.directRepair.candidates.data());
    disguisedInternalFailure.directRepair.candidates.erase(
        disguisedInternalFailure.directRepair.candidates.begin() + removedIndex);
    for (auto [index, candidate] : llvm::enumerate(disguisedInternalFailure.directRepair.candidates)) {
        candidate.id = index;
    }
    disguisedInternalFailure.directRepair.uncoveredObligations = {uncovered};
    disguisedInternalFailure.directRepair.rejections = {
        {uncovered, SyncDirectRepairRejection::InternalInvariant, "injected internal invariant"}};
    disguisedInternalFailure.directRepair.status = SyncDirectRepairPlanStatus::Partial;
    disguisedInternalFailure.status = SyncMixedPlanStatus::Unsupported;
    disguisedInternalFailure.failures = {
        {SyncMixedPlanRejection::IncompleteDirectRepair, "disguised internal invariant"}};
    const std::string beforeInternalFailure = printModule(*module);
    const bool internalFailureRejected = failed(verifyMixedProtocolPlan(
        fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, disguisedInternalFailure));
    const bool internalFailureAtomic = failed(materializeAndVerifyMixedProtocolPlanInDisposableModule(
        fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, disguisedInternalFailure));
    if (!check(
            internalFailureRejected && internalFailureAtomic && beforeInternalFailure == printModule(*module),
            "mixed verifier accepted a fallback-eligible disguised internal invariant")) {
        return false;
    }

    const bool allocated = succeeded(allocateMixedProtocolEvents(fixture.schedule, *plan)) &&
                           plan->status == SyncMixedPlanStatus::Ready && plan->readyRelease->lanes.size() == 1 &&
                           plan->readyRelease->lanes.front().readyEventId == 0 &&
                           plan->readyRelease->lanes.front().releaseEventId == 0;
    SyncDirectRepairCandidate* directEvent = findDirectedRepair(*plan);
    const bool collisionAvoided = directEvent && directEvent->eventId == 1;
    if (!check(
            allocated && collisionAvoided &&
                succeeded(verifyMixedProtocolPlan(
                    fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan)),
            "combined allocation did not avoid the protocol-owned event")) {
        return false;
    }

    SyncMixedProtocolPlan exhausted = unallocated;
    SyncDirectRepairCandidate* baseEvent = findDirectedRepair(exhausted);
    if (!check(baseEvent, "mixed fixture has no direct event candidate")) {
        return false;
    }
    for (unsigned index = 1; index < 6; ++index) {
        SyncDirectRepairCandidate extra = *baseEvent;
        extra.id = exhausted.directRepair.candidates.size();
        exhausted.directRepair.candidates.push_back(std::move(extra));
    }
    const bool rejectedAtomically = succeeded(allocateMixedProtocolEvents(fixture.schedule, exhausted)) &&
                                    exhausted.status == SyncMixedPlanStatus::ResourceInfeasible &&
                                    allEventIdsClear(exhausted);
    if (!check(rejectedAtomically, "combined event exhaustion retained a partial allocation")) {
        return false;
    }

    SyncMixedProtocolPlan malformed = *plan;
    findDirectedRepair(malformed)->eventId = malformed.readyRelease->lanes.front().readyEventId;
    const std::string before = printModule(*module);
    const bool rejectedCollision = failed(
        verifyMixedProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, malformed));
    const bool rejectedBeforeMutation = failed(materializeAndVerifyMixedProtocolPlanInDisposableModule(
        fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, malformed));
    if (!check(
            rejectedCollision && rejectedBeforeMutation && before == printModule(*module),
            "mixed verifier fault injection changed the source module")) {
        return false;
    }

    if (!check(
            succeeded(materializeAndVerifyMixedProtocolPlanInDisposableModule(
                fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan)) &&
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
                          plan->reverseDeletionRemoved == 1 && plan->directRepair.candidates.size() == 2;
    SyncDirectRepairCandidate* retainedExit = findExitRepair(*plan);
    const bool remapped = retainedExit && retainedExit->id == 1 && retainedExit->obligations.size() == 2 &&
                          retainedExit->obligations[0] == 1 && retainedExit->obligations[1] == 2;
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
                    fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *plan)),
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

} // namespace

int main()
{
    DialectRegistry registry;
    registry.insert<PTODialect, arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
    MLIRContext context(registry);
    context.loadAllAvailableDialects();
    if (!testMixedSelectionAndAllocation(context)) {
        return 1;
    }
    if (!testSuccessfulReverseDeletion(context)) {
        return 1;
    }
    if (!testReadyReleaseFamilyIsolation(context)) {
        return 1;
    }
    llvm::outs() << "protocol-sync mixed selection and reverse deletion: pass\n";
    llvm::outs() << "protocol-sync successful whole-candidate deletion: pass\n";
    llvm::outs() << "protocol-sync mixed combined event allocation: pass\n";
    llvm::outs() << "protocol-sync mixed independent verification: pass\n";
    llvm::outs() << "protocol-sync ReadyRelease family isolation: pass\n";
    llvm::outs() << "protocol-sync mixed atomic materialization: pass\n";
    return 0;
}
