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
#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/ReadyReleaseProtocol.h"
#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
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

} // namespace

int main()
{
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect, PTODialect, scf::SCFDialect>();
    MLIRContext context(registry);
    bool passed = testLogicalPlanAndAllocation(context);
    passed &= testVerifierNegatives(context);
    passed &= testAtomicMalformedPlan(context);
    passed &= testReservedEventAllocation(context);
    if (passed) {
        llvm::outs() << "protocol-sync ReadyRelease logical planning: pass\n";
        llvm::outs() << "protocol-sync ReadyRelease emitted-IR verification: pass\n";
        llvm::outs() << "protocol-sync ReadyRelease atomic rejection: pass\n";
        llvm::outs() << "protocol-sync ReadyRelease reservation allocation: pass\n";
    }
    return passed ? 0 : 1;
}
