// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- pto-protocol-sync-ready-release-test.cpp -------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

constexpr StringLiteral kFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @ready_release_one(
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

constexpr StringLiteral kDepthTwoFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @ready_release_two(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %output: !pto.partition_tensor_view<16x16xf16>,
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
      pto.tstore ins(%slot : !pto.tile_buf<vec, 16x16xf16>)
                 outs(%output : !pto.partition_tensor_view<16x16xf16>)
    }
    return
  }
}
)mlir";

constexpr StringLiteral kUnknownCapacityFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @unknown_capacity(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %output: !pto.partition_tensor_view<16x16xf16>,
      %buffers: !pto.multi_tile_buf<vec, 16x16xf16, count=2>,
      %trip_count: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0_index = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %index = %c0_index to %trip_count step %c1 {
      %selector = arith.remui %index, %c2 : index
      %tile = pto.multi_tile_get %buffers[%selector]
          : !pto.multi_tile_buf<vec, 16x16xf16, count=2>
         -> !pto.tile_buf<vec, 16x16xf16>
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

OwningOpRef<ModuleOp> parseFixture(MLIRContext& context, StringRef source = kFixture)
{
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, &context);
    if (!module || failed(verify(*module))) {
        return {};
    }
    return module;
}

std::string printFunction(func::FuncOp function)
{
    std::string result;
    llvm::raw_string_ostream output(result);
    function.print(output);
    return result;
}

struct AnalysisFixture {
    LegacySyncIRAdapter adapter;
    LegacySyncSnapshot legacy;
    SyncSemanticContext semanticContext;
    StructuredSyncIR schedule;
    FailureOr<PipelineStageAnalysisResult> stages;
    StorageTimelineAnalysisResult timelines;
    ChannelAnalysisResult channels;
    FailureOr<SyncReadyReleasePlan> plan;

    explicit AnalysisFixture(func::FuncOp function)
        : semanticContext(), schedule(function), stages(failure()), plan(failure())
    {}
};

bool buildAnalysis(AnalysisFixture& fixture, bool allocate)
{
    func::FuncOp function = fixture.schedule.getFunction();
    if (failed(fixture.adapter.buildSnapshot(function, fixture.legacy))) {
        return false;
    }
    SyncSemanticContext sourceContext = fixture.adapter.buildSemanticContext(fixture.legacy);
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
    fixture.plan =
        buildReadyReleaseProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    const bool planReady = succeeded(fixture.plan) && fixture.plan->status == SyncReadyReleasePlanStatus::Ready;
    if (!planReady) {
        return false;
    }
    return !allocate || (succeeded(allocateReadyReleaseProtocolEvents(fixture.schedule, *fixture.plan)) &&
                         fixture.plan->status == SyncReadyReleasePlanStatus::Ready);
}

bool testLogicalPlanAndAllocation(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context);
    if (!check(static_cast<bool>(module), "cannot parse ReadyRelease fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture, false), "cannot build ReadyRelease logical plan")) {
        return false;
    }
    bool passed = true;
    passed &= check(
        fixture.plan->capacity == 1 && fixture.plan->lanes.size() == 1, "logical plan did not select exactly one lane");
    passed &= check(
        !fixture.plan->lanes.front().readyEventId && !fixture.plan->lanes.front().releaseEventId,
        "logical plan assigned physical IDs before allocation");
    const SyncReadyReleaseTokenCertificate& certificate = fixture.plan->tokenCertificate;
    passed &= check(
        certificate.zeroTripSafe && certificate.oneTripSafe && certificate.oddEvenSafe &&
            certificate.steadyStateStable && certificate.transitionApplications == 16,
        "token certificate did not cover zero/one/odd/even steady state");
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(function);
    passed &= check(
        target.supportsEvent({SyncPhysicalCore::Cube, PIPE::PIPE_FIX}, {SyncPhysicalCore::Cube, PIPE::PIPE_MTE3}) &&
            !target.supportsEvent(
                {SyncPhysicalCore::Cube, PIPE::PIPE_MTE3}, {SyncPhysicalCore::Cube, PIPE::PIPE_FIX}) &&
            !target.supportsReadyRelease(SyncPhysicalCore::Cube, PIPE::PIPE_FIX, PIPE::PIPE_MTE3),
        "ReadyRelease target admission accepted an unsupported reverse direction");
    passed &= check(
        succeeded(allocateReadyReleaseProtocolEvents(fixture.schedule, *fixture.plan)),
        "ReadyRelease allocation failed");
    passed &= check(
        fixture.plan->lanes.front().readyEventId == 0 && fixture.plan->lanes.front().releaseEventId == 0,
        "independent event domains did not select the first compiler ID");
    passed &= check(
        failed(allocateReadyReleaseProtocolEvents(fixture.schedule, *fixture.plan)),
        "allocator accepted a second physical-ID assignment");
    return passed;
}

Operation* findGenerated(func::FuncOp function, StringRef role)
{
    Operation* result = nullptr;
    function.walk([&](Operation* operation) {
        auto candidate = operation->getAttrOfType<StringAttr>("pto.protocol_sync.role");
        const bool roleMatches = candidate && candidate.getValue() == role;
        if (roleMatches) {
            result = operation;
        }
    });
    return result;
}

bool testDepthTwoPlanAndVerifier(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kDepthTwoFixture);
    if (!check(static_cast<bool>(module), "cannot parse ReadyRelease<2> fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture, false), "cannot build ReadyRelease<2> logical plan")) {
        return false;
    }
    bool passed = true;
    const SyncReadyReleaseTokenCertificate& certificate = fixture.plan->tokenCertificate;
    passed &= check(
        fixture.plan->capacity == 2 && fixture.plan->lanes.size() == 2 && fixture.plan->slot &&
            fixture.plan->slot->kind == SyncSlotExpressionKind::AffineModulo && fixture.plan->slot->offset == 0,
        "ReadyRelease<2> did not retain its two-lane selector");
    passed &= check(
        llvm::all_of(
            fixture.plan->lanes,
            [](const SyncReadyReleaseLane& lane) { return !lane.readyEventId && !lane.releaseEventId; }),
        "ReadyRelease<2> assigned physical IDs before logical selection");
    passed &= check(
        certificate.zeroTripSafe && certificate.oneTripSafe && certificate.oddEvenSafe &&
            certificate.steadyStateStable && certificate.transitionApplications == 17 &&
            certificate.slotWitness == llvm::SmallVector<unsigned, 8>{0, 1, 0, 1, 0, 1},
        "ReadyRelease<2> token certificate did not cover its complete selector period");

    SyncReadyReleasePlan exhausted = *fixture.plan;
    SyncEventReservation fullReadyDomain;
    fullReadyDomain.source = PIPE::PIPE_MTE2;
    fullReadyDomain.target = PIPE::PIPE_MTE3;
    fullReadyDomain.eventIds = {0, 1, 2, 3, 4, 5};
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(function);
    passed &= check(
        succeeded(
            allocateReadyReleaseProtocolEvents(target, ArrayRef<SyncEventReservation>{fullReadyDomain}, exhausted)) &&
            exhausted.status == SyncReadyReleasePlanStatus::Unsupported && exhausted.rejections.size() == 1 &&
            exhausted.rejections.front().reason == SyncReadyReleaseRejection::EventCapacity &&
            llvm::all_of(
                exhausted.lanes,
                [](const SyncReadyReleaseLane& lane) { return !lane.readyEventId && !lane.releaseEventId; }),
        "ReadyRelease<2> exhaustion did not remove the complete unallocated candidate");

    SyncEventReservation sparseReady;
    sparseReady.source = PIPE::PIPE_MTE2;
    sparseReady.target = PIPE::PIPE_MTE3;
    sparseReady.eventIds = {0, 2};
    SyncEventReservation sparseRelease;
    sparseRelease.source = PIPE::PIPE_MTE3;
    sparseRelease.target = PIPE::PIPE_MTE2;
    sparseRelease.eventIds = {0, 3};
    const SyncEventReservation sparseReservations[] = {sparseReady, sparseRelease};
    passed &= check(
        succeeded(allocateReadyReleaseProtocolEvents(target, sparseReservations, *fixture.plan)) &&
            fixture.plan->status == SyncReadyReleasePlanStatus::Ready && fixture.plan->lanes[0].readyEventId == 1 &&
            fixture.plan->lanes[1].readyEventId == 3 && fixture.plan->lanes[0].releaseEventId == 1 &&
            fixture.plan->lanes[1].releaseEventId == 2,
        "ReadyRelease<2> did not allocate independent non-contiguous domains");

    IRMapping mapping;
    OwningOpRef<ModuleOp> stagingModule = ModuleOp::create(function.getLoc());
    func::FuncOp clone = cast<func::FuncOp>(function->clone(mapping));
    stagingModule->push_back(clone);
    passed &= check(
        succeeded(materializeReadyReleaseProtocolPlan(clone, mapping, *fixture.plan)) &&
            succeeded(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        "independent verifier rejected valid non-contiguous ReadyRelease<2> emission");

    LegacySyncIRAdapter ambiguousAdapter;
    LegacySyncSnapshot ambiguousLegacy;
    StructuredSyncIR ambiguousSchedule(function);
    FailureOr<PipelineStageAnalysisResult> ambiguousStages = failure();
    bool builtAmbiguousSchedule = succeeded(ambiguousAdapter.buildSnapshot(function, ambiguousLegacy));
    if (builtAmbiguousSchedule) {
        SyncSemanticContext ambiguousContext = ambiguousAdapter.buildSemanticContext(ambiguousLegacy);
        StructuredSyncIRBuilder ambiguousBuilder(ambiguousContext);
        builtAmbiguousSchedule = succeeded(ambiguousBuilder.build(function, ambiguousSchedule));
        if (builtAmbiguousSchedule) {
            ambiguousStages = analyzePipelineStages(ambiguousSchedule);
            builtAmbiguousSchedule = succeeded(ambiguousStages);
        }
    }
    passed &= check(
        builtAmbiguousSchedule &&
            failed(verifyReadyReleaseProtocolMaterialization(ambiguousSchedule, *ambiguousStages, clone, mapping)),
        "independent verifier accepted ReadyRelease with ambiguous GM arguments");

    const SyncGenerationTimeline& timeline = fixture.timelines.getTimelines()[fixture.plan->generation];
    const SyncStorageFamily* family = fixture.schedule.findStorageFamily(timeline.family);
    if (!check(family != nullptr, "ReadyRelease<2> has no storage family")) {
        return false;
    }
    auto allocation = family->root.getDefiningOp<AllocMultiTileOp>();
    if (!check(static_cast<bool>(allocation), "ReadyRelease<2> root is not alloc_multi_tile")) {
        return false;
    }
    MultiTileBufType originalType = allocation.getResult().getType();
    allocation.getResult().setType(MultiTileBufType::get(&context, originalType.getSlotType(), 3));
    passed &= check(
        failed(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        "independent verifier trusted cached capacity instead of the allocation root type");
    allocation.getResult().setType(originalType);

    Operation* laneOnePrime = findGenerated(clone, "release-prime-set");
    Attribute originalLane = laneOnePrime->getAttr("pto.protocol_sync.logical_lane");
    laneOnePrime->setAttr(
        "pto.protocol_sync.logical_lane", IntegerAttr::get(IntegerType::get(clone.getContext(), 64), 4294967297LL));
    passed &= check(
        failed(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        "independent verifier accepted oversized logical-lane metadata");
    laneOnePrime->setAttr("pto.protocol_sync.logical_lane", originalLane);

    auto releaseWait = cast<WaitFlagDynOp>(findGenerated(clone, "release-body-wait"));
    auto releaseSelector = releaseWait.getEventId().getDefiningOp<arith::SelectOp>();
    auto laneCompare = releaseSelector.getCondition().getDefiningOp<arith::CmpIOp>();
    Value originalLaneOne = laneCompare.getRhs();
    OpBuilder corruptBuilder(laneCompare);
    Value oversizedLaneOne = corruptBuilder.create<arith::ConstantIndexOp>(laneCompare.getLoc(), 4294967297LL);
    laneCompare->setOperand(1, oversizedLaneOne);
    passed &= check(
        failed(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        "independent verifier accepted an oversized selector constant");
    laneCompare->setOperand(1, originalLaneOne);
    oversizedLaneOne.getDefiningOp()->erase();

    Value laneOne = releaseSelector.getTrueValue();
    Value laneZero = releaseSelector.getFalseValue();
    releaseSelector->setOperand(1, laneZero);
    releaseSelector->setOperand(2, laneOne);
    passed &= check(
        failed(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        "independent verifier accepted a swapped release-lane selector");
    releaseSelector->setOperand(1, laneOne);
    releaseSelector->setOperand(2, laneZero);

    auto clonedAllocation = mapping.lookupOrNull(allocation.getResult()).getDefiningOp<AllocMultiTileOp>();
    const std::int64_t shiftedAddresses[] = {512, 1024};
    clonedAllocation->setAttr(kPtoMultiBufferAddrsAttrName, DenseI64ArrayAttr::get(&context, shiftedAddresses));
    passed &= check(
        failed(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        "independent verifier accepted shifted physical slot ranges");
    clonedAllocation->removeAttr(kPtoMultiBufferAddrsAttrName);

    Value clonedSlot = mapping.lookupOrNull(fixture.plan->slot->selector);
    auto clonedRemainder = clonedSlot.getDefiningOp<arith::RemUIOp>();
    Value originalModulus = clonedRemainder.getRhs();
    OpBuilder modulusBuilder(clonedRemainder);
    Value wrongModulus = modulusBuilder.create<arith::ConstantIndexOp>(clonedRemainder.getLoc(), 1);
    clonedRemainder->setOperand(1, wrongModulus);
    passed &= check(
        failed(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        "independent verifier accepted a changed slot modulus");
    clonedRemainder->setOperand(1, originalModulus);
    wrongModulus.getDefiningOp()->erase();

    findGenerated(clone, "release-prime-set")->erase();
    passed &= check(
        failed(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        "independent verifier accepted a missing ReadyRelease<2> prime lane");
    return passed;
}

bool testUnknownPhysicalCapacityRejected(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kUnknownCapacityFixture);
    if (!check(static_cast<bool>(module), "cannot parse unknown-capacity fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    LegacySyncIRAdapter adapter;
    LegacySyncSnapshot legacy;
    if (!check(succeeded(adapter.buildSnapshot(function, legacy)), "cannot build unknown-capacity snapshot")) {
        return false;
    }
    SyncSemanticContext sourceContext = adapter.buildSemanticContext(legacy);
    SyncSemanticContext semanticContext;
    for (unsigned argument : {0U, 1U}) {
        Value value = function.getArgument(argument);
        for (const SyncStorageProvenance& provenance : sourceContext.lookupStorage(value)) {
            semanticContext.addStorage(value, provenance);
        }
    }
    MultiTileGetOp getSlot;
    function.walk([&](MultiTileGetOp operation) { getSlot = operation; });
    if (!check(static_cast<bool>(getSlot), "unknown-capacity fixture has no slot selection")) {
        return false;
    }
    Value tile = getSlot.getResult();
    ArrayRef<SyncStorageProvenance> tileStorage = sourceContext.lookupStorage(tile);
    const bool hasUnambiguousStorage = tileStorage.size() == 1;
    if (!check(hasUnambiguousStorage, "unknown-capacity tile has ambiguous provenance")) {
        return false;
    }
    SyncStorageProvenance physicalStorage = tileStorage.front();
    physicalStorage.physical = true;
    semanticContext.addStorage(tile, std::move(physicalStorage));

    StructuredSyncIR schedule(function);
    StructuredSyncIRBuilder builder(semanticContext);
    if (!check(succeeded(builder.build(function, schedule)), "cannot build unknown-capacity schedule")) {
        return false;
    }
    FailureOr<PipelineStageAnalysisResult> stages = analyzePipelineStages(schedule);
    if (!check(succeeded(stages), "cannot build unknown-capacity stages")) {
        return false;
    }
    StorageTimelineAnalysisResult timelines = analyzeStorageTimelines(schedule, *stages);
    const SyncStorageFamily* localFamily = nullptr;
    for (const SyncStorageFamily& family : schedule.getStorageFamilies()) {
        if (family.root == function.getArgument(2)) {
            localFamily = &family;
            break;
        }
    }
    const bool preservedDepthTwoSelector =
        localFamily && llvm::any_of(schedule.getAccesses(), [&](const SyncAccess& access) {
            return access.family == localFamily->id && access.slot && access.slot->depth == 2;
        });
    const bool rejectedAsUnknownCapacity =
        localFamily && localFamily->physical && localFamily->role == SyncStorageRole::LocalBuffer &&
        !localFamily->slotCount && preservedDepthTwoSelector && timelines.getTimelines().size() == 1 &&
        timelines.getTimelines().front().rejection == SyncTimelineRejection::UnknownCapacity;
    return check(
        rejectedAsUnknownCapacity,
        "physical local storage without an allocation descriptor was not rejected as unknown-capacity");
}

using CorruptMaterialization = std::function<void(func::FuncOp)>;

bool testMalformedMaterialization(MLIRContext& context, StringRef name, const CorruptMaterialization& corrupt)
{
    OwningOpRef<ModuleOp> module = parseFixture(context);
    const bool parsed = static_cast<bool>(module);
    const std::string parseFailure = (Twine(name) + ": cannot parse fixture").str();
    if (!check(parsed, parseFailure)) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    const bool analysisReady = buildAnalysis(fixture, true);
    const std::string analysisFailure = (Twine(name) + ": cannot build allocated plan").str();
    if (!check(analysisReady, analysisFailure)) {
        return false;
    }

    IRMapping mapping;
    OwningOpRef<ModuleOp> stagingModule = ModuleOp::create(function.getLoc());
    func::FuncOp clone = cast<func::FuncOp>(function->clone(mapping));
    stagingModule->push_back(clone);
    const bool valid =
        succeeded(materializeReadyReleaseProtocolPlan(clone, mapping, *fixture.plan)) &&
        succeeded(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping));
    const std::string verificationFailure = (Twine(name) + ": verifier rejected the valid protocol").str();
    if (!check(valid, verificationFailure)) {
        return false;
    }
    corrupt(clone);
    return check(
        failed(verifyReadyReleaseProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        Twine(name) + ": verifier accepted the corrupted protocol");
}

bool testVerifierNegatives(MLIRContext& context)
{
    bool passed = true;
    passed &= testMalformedMaterialization(context, "missing-drain", [](func::FuncOp function) {
        findGenerated(function, "release-drain-wait")->erase();
    });
    passed &= testMalformedMaterialization(context, "ready-event-id", [](func::FuncOp function) {
        auto set = cast<SetFlagOp>(findGenerated(function, "ready-body-set"));
        set.setEventIdAttr(EventAttr::get(function.getContext(), EVENT::EVENT_ID5));
    });
    passed &= testMalformedMaterialization(context, "release-direction", [](func::FuncOp function) {
        auto wait = cast<WaitFlagOp>(findGenerated(function, "release-body-wait"));
        wait.setSrcPipeAttr(PipeAttr::get(function.getContext(), PIPE::PIPE_MTE2));
    });
    passed &= testMalformedMaterialization(context, "prime-placement", [](func::FuncOp function) {
        Operation* prime = findGenerated(function, "release-prime-set");
        Operation* drain = findGenerated(function, "release-drain-wait");
        prime->moveAfter(drain);
    });
    passed &= testMalformedMaterialization(context, "ready-signal-placement", [](func::FuncOp function) {
        Operation* signal = findGenerated(function, "ready-body-set");
        signal->moveBefore(findGenerated(function, "release-body-wait"));
    });
    passed &= testMalformedMaterialization(context, "ready-wait-placement", [](func::FuncOp function) {
        Operation* wait = findGenerated(function, "ready-body-wait");
        wait->moveAfter(findGenerated(function, "release-body-set"));
    });
    passed &= testMalformedMaterialization(context, "guarded-ready-signal", [](func::FuncOp function) {
        Operation* signal = findGenerated(function, "ready-body-set");
        OpBuilder builder(signal);
        Value condition = builder.create<arith::ConstantIntOp>(signal->getLoc(), 1, 1);
        auto choice = builder.create<scf::IfOp>(signal->getLoc(), TypeRange{}, condition, false);
        signal->moveBefore(choice.thenYield());
    });
    passed &= testMalformedMaterialization(context, "logical-lane", [](func::FuncOp function) {
        Operation* prime = findGenerated(function, "release-prime-set");
        prime->setAttr(
            "pto.protocol_sync.logical_lane", IntegerAttr::get(IntegerType::get(function.getContext(), 32), 1));
    });
    passed &= testMalformedMaterialization(context, "body-pipe-all", [](func::FuncOp function) {
        scf::ForOp loop;
        function.walk([&](scf::ForOp operation) { loop = operation; });
        OpBuilder builder(loop.getBody()->getTerminator());
        builder.create<BarrierOp>(loop.getLoc(), PipeAttr::get(function.getContext(), PIPE::PIPE_ALL));
    });
    return passed;
}

bool testAtomicMalformedPlan(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context);
    if (!check(static_cast<bool>(module), "cannot parse atomicity fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture, true), "cannot build atomicity plan")) {
        return false;
    }
    fixture.plan->producerOperation = fixture.plan->consumerOperation;
    const std::string before = printFunction(function);
    ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
    const bool rejected =
        failed(materializeAndVerifyReadyReleaseProtocolPlan(fixture.schedule, *fixture.stages, *fixture.plan));
    return check(rejected, "malformed ReadyRelease plan was accepted") &&
           check(printFunction(function) == before, "malformed plan changed the source function");
}

bool testReservedEventAllocation(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kReservationFixture);
    if (!check(static_cast<bool>(module), "cannot parse reservation fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    LegacySyncIRAdapter adapter;
    LegacySyncSnapshot legacy;
    if (!check(succeeded(adapter.buildSnapshot(function, legacy)), "cannot build reservation snapshot")) {
        return false;
    }
    SyncSemanticContext semanticContext = adapter.buildSemanticContext(legacy);
    StructuredSyncIR schedule(function);
    StructuredSyncIRBuilder builder(semanticContext);
    if (!check(succeeded(builder.build(function, schedule)), "cannot build reservation schedule")) {
        return false;
    }

    SyncReadyReleasePlan plan;
    plan.status = SyncReadyReleasePlanStatus::Ready;
    plan.capacity = 1;
    plan.core = SyncPhysicalCore::Vector;
    plan.producerPipe = PIPE::PIPE_V;
    plan.consumerPipe = PIPE::PIPE_S;
    plan.lanes.push_back({0, std::nullopt, std::nullopt});
    const bool allocated = succeeded(allocateReadyReleaseProtocolEvents(schedule, plan));
    bool passed =
        check(allocated && plan.status == SyncReadyReleasePlanStatus::Ready, "reservation-aware allocation failed") &&
        check(plan.lanes.front().readyEventId == 1, "allocator reused a hidden ready-domain reservation") &&
        check(plan.lanes.front().releaseEventId == 0, "reverse domain did not allocate independently");

    SyncReadyReleasePlan exhausted;
    exhausted.status = SyncReadyReleasePlanStatus::Ready;
    exhausted.capacity = 1;
    exhausted.core = SyncPhysicalCore::Vector;
    exhausted.producerPipe = PIPE::PIPE_V;
    exhausted.consumerPipe = PIPE::PIPE_S;
    exhausted.lanes.push_back({0, std::nullopt, std::nullopt});
    SyncEventReservation fullDomain;
    fullDomain.source = PIPE::PIPE_V;
    fullDomain.target = PIPE::PIPE_S;
    fullDomain.eventIds = {0, 1, 2, 3, 4, 5};
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(function);
    const bool exhaustionHandled =
        succeeded(allocateReadyReleaseProtocolEvents(target, ArrayRef<SyncEventReservation>{fullDomain}, exhausted));
    passed &= check(exhaustionHandled, "event exhaustion was reported as an internal allocation error");
    passed &= check(
        exhausted.status == SyncReadyReleasePlanStatus::Unsupported && exhausted.rejections.size() == 1 &&
            exhausted.rejections.front().reason == SyncReadyReleaseRejection::EventCapacity,
        "event exhaustion did not reject the complete candidate with event-capacity attribution");
    passed &= check(
        !exhausted.lanes.front().readyEventId && !exhausted.lanes.front().releaseEventId,
        "event exhaustion partially assigned physical IDs");
    return passed;
}

bool testSelectedWorldInterpreter(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context);
    if (!check(static_cast<bool>(module), "cannot parse selected-world fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture, false), "cannot build selected-world plan")) {
        return false;
    }
    FailureOr<SyncSelectedWorld> world = buildSelectedWorld(*fixture.plan);
    if (!check(succeeded(world), "cannot adapt ReadyRelease selected world")) {
        return false;
    }
    FailureOr<SyncInterpretationResult> selected =
        interpretSelectedWorld(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *world);
    bool passed = check(succeeded(selected) && selected->isComplete(), "complete selected world left residuals");

    SyncSelectedWorld incomplete = *world;
    incomplete.completions.pop_back();
    passed &= check(
        failed(
            interpretSelectedWorld(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, incomplete)),
        "missing release transfer was reported as a repairable residual");
    SyncReadyReleasePlan malformed = *fixture.plan;
    malformed.lanes.front().logicalLane = malformed.capacity;
    passed &= check(failed(buildSelectedWorld(malformed)), "out-of-range logical lane was accepted");
    return passed;
}

bool testA2SharedCapability(MLIRContext& context)
{
    bool passed = true;
    for (StringRef source : {StringRef(kFixture), StringRef(kDepthTwoFixture)}) {
        OwningOpRef<ModuleOp> module = parseFixture(context, source);
        if (!check(static_cast<bool>(module), "cannot parse A2 shared-capability fixture")) {
            return false;
        }
        module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, "a2"));
        func::FuncOp function = *module->getOps<func::FuncOp>().begin();
        AnalysisFixture fixture(function);
        const bool ready = buildAnalysis(fixture, true);
        passed &= check(ready, "A2 did not admit a shared ReadyRelease capability");
        if (!ready) {
            continue;
        }
        passed &= check(
            succeeded(materializeAndVerifyReadyReleaseProtocolPlan(
                fixture.schedule, *fixture.stages, *fixture.plan)),
            "A2 shared ReadyRelease capability failed materialization or verification");
    }
    return passed;
}

} // namespace

int main()
{
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect, PTODialect, scf::SCFDialect>();
    MLIRContext context(registry);
    bool passed = testLogicalPlanAndAllocation(context);
    passed &= testDepthTwoPlanAndVerifier(context);
    passed &= testUnknownPhysicalCapacityRejected(context);
    passed &= testVerifierNegatives(context);
    passed &= testAtomicMalformedPlan(context);
    passed &= testReservedEventAllocation(context);
    passed &= testSelectedWorldInterpreter(context);
    passed &= testA2SharedCapability(context);
    if (passed) {
        llvm::outs() << "protocol-sync ReadyRelease logical planning: pass\n";
        llvm::outs() << "protocol-sync ReadyRelease depth-two planning: pass\n";
        llvm::outs() << "protocol-sync ReadyRelease authoritative capacity: pass\n";
        llvm::outs() << "protocol-sync ReadyRelease emitted-IR verification: pass\n";
        llvm::outs() << "protocol-sync ReadyRelease atomic rejection: pass\n";
        llvm::outs() << "protocol-sync ReadyRelease reservation allocation: pass\n";
        llvm::outs() << "protocol-sync ReadyRelease selected-world interpretation: pass\n";
    }
    return passed ? 0 : 1;
}
