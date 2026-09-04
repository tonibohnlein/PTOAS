// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.
//===- pto-protocol-sync-one-shot-test.cpp -------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/OneShotProtocol.h"
#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

constexpr StringLiteral kFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @one_shot_verifier(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %output: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %c512 = arith.constant 512 : i64
    %c1024 = arith.constant 1024 : i64
    %source = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    %middle = pto.alloc_tile addr = %c512 : !pto.tile_buf<vec, 16x16xf16>
    %result = pto.alloc_tile addr = %c1024 : !pto.tile_buf<vec, 16x16xf16>
    pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
              outs(%source : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%source : !pto.tile_buf<vec, 16x16xf16>)
             outs(%middle : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%middle : !pto.tile_buf<vec, 16x16xf16>)
             outs(%result : !pto.tile_buf<vec, 16x16xf16>)
    pto.tstore ins(%result : !pto.tile_buf<vec, 16x16xf16>)
               outs(%output : !pto.partition_tensor_view<16x16xf16>)
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

constexpr StringLiteral kRepeatedDomainFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @repeated_domain(
      %input: !pto.partition_tensor_view<16x16xf16>,
      %output: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %c512 = arith.constant 512 : i64
    %c1024 = arith.constant 1024 : i64
    %c1536 = arith.constant 1536 : i64
    %sourceA = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    %resultA = pto.alloc_tile addr = %c512 : !pto.tile_buf<vec, 16x16xf16>
    %sourceB = pto.alloc_tile addr = %c1024 : !pto.tile_buf<vec, 16x16xf16>
    %resultB = pto.alloc_tile addr = %c1536 : !pto.tile_buf<vec, 16x16xf16>
    pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
              outs(%sourceA : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%sourceA : !pto.tile_buf<vec, 16x16xf16>)
             outs(%resultA : !pto.tile_buf<vec, 16x16xf16>)
    pto.tload ins(%input : !pto.partition_tensor_view<16x16xf16>)
              outs(%sourceB : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%sourceB : !pto.tile_buf<vec, 16x16xf16>)
             outs(%resultB : !pto.tile_buf<vec, 16x16xf16>)
    pto.tstore ins(%resultA : !pto.tile_buf<vec, 16x16xf16>)
               outs(%output : !pto.partition_tensor_view<16x16xf16>)
    pto.tstore ins(%resultB : !pto.tile_buf<vec, 16x16xf16>)
               outs(%output : !pto.partition_tensor_view<16x16xf16>)
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
    FailureOr<SyncOneShotPlan> plan;

    explicit AnalysisFixture(func::FuncOp function)
        : semanticContext(), schedule(function), stages(failure()), plan(failure())
    {}
};

bool buildAnalysis(AnalysisFixture& fixture, bool allocateEvents = true)
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
    fixture.plan = buildOneShotProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    if (failed(fixture.plan) || fixture.plan->status != SyncOneShotPlanStatus::Ready) {
        return false;
    }
    if (!allocateEvents) {
        return true;
    }
    return succeeded(allocateOneShotProtocolEvents(fixture.schedule, *fixture.plan)) &&
           fixture.plan->status == SyncOneShotPlanStatus::Ready;
}

bool testTargetContract(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context);
    if (!check(static_cast<bool>(module), "cannot parse target fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    ProtocolSyncTarget a3Target = ProtocolSyncTarget::resolve(function);
    bool passed = check(a3Target.isSupported(), "explicit A3 target was rejected");
    passed &= check(a3Target.getKind() == ProtocolSyncTargetKind::Npu2201A3, "A3 target kind changed");
    passed &= check(a3Target.getName() == "npu2201-a3-protocol-sync-v1", "A3 target name changed");
    passed &= check(a3Target.supportsReadyReleaseEmission(), "A3 ReadyRelease qualification disappeared");
    const auto cube = [](PIPE pipe) { return ProtocolSyncResource{SyncPhysicalCore::Cube, pipe}; };
    passed &= check(!a3Target.supportsEvent(cube(PIPE::PIPE_MTE1), cube(PIPE::PIPE_FIX)), "MTE1_FIX admitted");
    passed &= check(!a3Target.supportsEvent(cube(PIPE::PIPE_MTE3), cube(PIPE::PIPE_FIX)), "MTE3_FIX admitted");
    passed &= check(!a3Target.supportsEvent(cube(PIPE::PIPE_FIX), cube(PIPE::PIPE_MTE1)), "FIX_MTE1 admitted");
    passed &= check(a3Target.supportsEvent(cube(PIPE::PIPE_M), cube(PIPE::PIPE_FIX)), "M_FIX rejected");
    passed &= check(a3Target.supportsEvent(cube(PIPE::PIPE_MTE2), cube(PIPE::PIPE_FIX)), "MTE2_FIX rejected");
    passed &= check(a3Target.supportsEvent(cube(PIPE::PIPE_FIX), cube(PIPE::PIPE_MTE3)), "FIX_MTE3 rejected");
    passed &= check(a3Target.getCompilerEventIds() == ArrayRef<unsigned>({0, 1, 2, 3, 4, 5}), "event pool changed");
    passed &= check(
        stringifyProtocolSyncProducer(ProtocolSyncProducer::ProtocolPlan) == "protocol-plan",
        "protocol-plan producer spelling changed");
    passed &= check(
        stringifyProtocolSyncProducer(ProtocolSyncProducer::LegacyFallbackUnsupported) == "legacy-fallback-unsupported",
        "unsupported fallback producer spelling changed");
    passed &= check(
        stringifyProtocolSyncProducer(ProtocolSyncProducer::LegacyFallbackResourceInfeasible) ==
            "legacy-fallback-resource-infeasible",
        "resource fallback producer spelling changed");
    passed &= check(
        stringifyProtocolSyncProducer(ProtocolSyncProducer::FailClosedPolicy) == "fail-closed-policy",
        "fail-closed producer spelling changed");
    passed &= check(
        stringifyProtocolSyncProducer(ProtocolSyncProducer::InternalError) == "internal-error",
        "internal-error producer spelling changed");

    module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, "a2"));
    ProtocolSyncTarget a2Target = ProtocolSyncTarget::resolve(function);
    passed &= check(a2Target.isSupported(), "explicit A2 target was rejected");
    passed &= check(a2Target.getKind() == ProtocolSyncTargetKind::Npu2201A2, "A2 target kind changed");
    passed &= check(a2Target.getName() == "npu2201-a2-protocol-sync-v1", "A2 target name changed");
    passed &= check(!a2Target.supportsReadyReleaseEmission(), "A2 ReadyRelease was admitted without qualification");
    passed &= check(a2Target.getCompilerEventIds() == a3Target.getCompilerEventIds(), "A2/A3 event pools diverged");

    constexpr PIPE cubeBarriers[] = {PIPE::PIPE_M, PIPE::PIPE_MTE1, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3, PIPE::PIPE_FIX};
    constexpr PIPE vectorBarriers[] = {PIPE::PIPE_V, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3};
    constexpr std::pair<PIPE, PIPE> cubeEvents[] = {
        {PIPE::PIPE_M, PIPE::PIPE_MTE1},   {PIPE::PIPE_M, PIPE::PIPE_MTE2},    {PIPE::PIPE_M, PIPE::PIPE_FIX},
        {PIPE::PIPE_MTE1, PIPE::PIPE_M},   {PIPE::PIPE_MTE1, PIPE::PIPE_MTE2}, {PIPE::PIPE_MTE1, PIPE::PIPE_MTE3},
        {PIPE::PIPE_MTE2, PIPE::PIPE_M},   {PIPE::PIPE_MTE2, PIPE::PIPE_MTE1}, {PIPE::PIPE_MTE2, PIPE::PIPE_MTE3},
        {PIPE::PIPE_MTE2, PIPE::PIPE_FIX}, {PIPE::PIPE_MTE3, PIPE::PIPE_MTE1}, {PIPE::PIPE_MTE3, PIPE::PIPE_MTE2},
        {PIPE::PIPE_FIX, PIPE::PIPE_M},    {PIPE::PIPE_FIX, PIPE::PIPE_MTE2},  {PIPE::PIPE_FIX, PIPE::PIPE_MTE3}};
    constexpr PIPE vectorEventPipes[] = {PIPE::PIPE_S, PIPE::PIPE_V, PIPE::PIPE_MTE2, PIPE::PIPE_MTE3};
    constexpr SyncPhysicalCore cores[] = {SyncPhysicalCore::Cube, SyncPhysicalCore::Vector};
    constexpr PIPE pipes[] = {PIPE::PIPE_S,    PIPE::PIPE_V,   PIPE::PIPE_M,   PIPE::PIPE_MTE1,      PIPE::PIPE_MTE2,
                              PIPE::PIPE_MTE3, PIPE::PIPE_FIX, PIPE::PIPE_ALL, PIPE::PIPE_UNASSIGNED};
    for (SyncPhysicalCore core : cores) {
        for (PIPE pipe : pipes) {
            const ProtocolSyncResource resource{core, pipe};
            const bool expectedBarrier = core == SyncPhysicalCore::Cube ? llvm::is_contained(cubeBarriers, pipe) :
                                                                          llvm::is_contained(vectorBarriers, pipe);
            passed &= check(
                a2Target.supportsPipeBarrier(resource) == expectedBarrier, "A2 barrier table differs from fixture");
            passed &= check(
                a3Target.supportsPipeBarrier(resource) == expectedBarrier, "A3 barrier table differs from fixture");
            passed &= check(
                a2Target.supportsPipeBarrier(resource) == a3Target.supportsPipeBarrier(resource),
                "A2/A3 barrier tables diverged");
            for (PIPE destination : pipes) {
                const bool expectedEvent = core == SyncPhysicalCore::Cube ?
                                               llvm::is_contained(cubeEvents, std::make_pair(pipe, destination)) :
                                               pipe != destination && llvm::is_contained(vectorEventPipes, pipe) &&
                                                   llvm::is_contained(vectorEventPipes, destination);
                passed &= check(
                    a2Target.supportsEvent(resource, {core, destination}) == expectedEvent,
                    "A2 event table differs from fixture");
                passed &= check(
                    a3Target.supportsEvent(resource, {core, destination}) == expectedEvent,
                    "A3 event table differs from fixture");
                passed &= check(
                    a2Target.supportsEvent(resource, {core, destination}) ==
                        a3Target.supportsEvent(resource, {core, destination}),
                    "A2/A3 event tables diverged");
                const SyncPhysicalCore otherCore =
                    core == SyncPhysicalCore::Cube ? SyncPhysicalCore::Vector : SyncPhysicalCore::Cube;
                passed &= check(
                    !a2Target.supportsEvent(resource, {otherCore, destination}) &&
                        !a3Target.supportsEvent(resource, {otherCore, destination}),
                    "cross-core pair entered the local event table");
            }
        }
    }

    for (StringRef arch : {"a2", "a3"}) {
        module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, arch));
        for (StringRef deviceSpec : {"Ascend910", "Ascend910A", "Ascend910B1", "Ascend910B3", "Ascend910_1234"}) {
            module->getOperation()->setAttr("pto.device-spec", StringAttr::get(&context, deviceSpec));
            passed &= check(
                !ProtocolSyncTarget::resolve(function).isSupported(),
                Twine("unqualified device profile admitted: ") + deviceSpec);
        }
    }
    module->getOperation()->setAttr("pto.device-spec", UnitAttr::get(&context));
    passed &= check(!ProtocolSyncTarget::resolve(function).isSupported(), "non-string device profile was admitted");
    module->getOperation()->removeAttr("pto.device-spec");
    module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, "a5"));
    passed &= check(!ProtocolSyncTarget::resolve(function).isSupported(), "A5 target was admitted");
    module->getOperation()->removeAttr("pto.target_arch");
    passed &= check(!ProtocolSyncTarget::resolve(function).isSupported(), "attr-less target was admitted");
    return passed;
}

bool testPlanTargetIdentity(MLIRContext& context, StringRef plannedArch, StringRef replayArch)
{
    OwningOpRef<ModuleOp> module = parseFixture(context);
    if (!check(static_cast<bool>(module), "cannot parse target-identity fixture")) {
        return false;
    }
    module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, plannedArch));
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    const bool fixtureBuilt =
        check(buildAnalysis(fixture, false), Twine("cannot build ") + plannedArch + " target-identity fixture");
    if (fixtureBuilt == false) {
        return false;
    }
    std::string planBefore;
    llvm::raw_string_ostream planBeforeOutput(planBefore);
    printOneShotProtocolPlan(function, *fixture.plan, planBeforeOutput);
    planBeforeOutput.flush();
    module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, replayArch));
    bool passed = check(
        failed(allocateOneShotProtocolEvents(fixture.schedule, *fixture.plan)),
        Twine(plannedArch) + " plan was allocated by the " + replayArch + " target");
    std::string planAfter;
    llvm::raw_string_ostream planAfterOutput(planAfter);
    printOneShotProtocolPlan(function, *fixture.plan, planAfterOutput);
    planAfterOutput.flush();
    passed &= check(planAfter == planBefore, "target-identity allocation rejection mutated the plan");

    module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, plannedArch));
    passed &= check(
        succeeded(allocateOneShotProtocolEvents(fixture.schedule, *fixture.plan)),
        Twine("cannot allocate ") + plannedArch + " target-identity fixture");
    const std::string before = printFunction(function);
    module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, replayArch));
    passed &= check(
        failed(materializeAndVerifyOneShotProtocolPlan(fixture.schedule, *fixture.stages, *fixture.plan)),
        Twine(plannedArch) + " plan was accepted by the " + replayArch + " target");
    passed &= check(printFunction(function) == before, "staged target-identity rejection mutated the function");
    passed &= check(
        failed(materializeAndVerifyOneShotProtocolPlanInPlace(fixture.schedule, *fixture.stages, *fixture.plan)),
        Twine(plannedArch) + " plan was accepted in place by the " + replayArch + " target");
    passed &= check(printFunction(function) == before, "in-place target-identity rejection mutated the function");
    return passed;
}

bool testMalformedPlan(MLIRContext& context, StringRef name, const std::function<void(SyncOneShotPlan&)>& corrupt)
{
    OwningOpRef<ModuleOp> module = parseFixture(context);
    if (!check(static_cast<bool>(module), Twine(name) + ": cannot parse fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture), Twine(name) + ": cannot build ready plan")) {
        return false;
    }
    SyncOneShotPlan malformed = *fixture.plan;
    corrupt(malformed);
    const std::string before = printFunction(function);
    ScopedDiagnosticHandler silence(&context, [](Diagnostic&) { return success(); });
    const bool rejected = failed(materializeAndVerifyOneShotProtocolPlan(fixture.schedule, *fixture.stages, malformed));
    return check(rejected, Twine(name) + ": malformed plan was accepted") &&
           check(printFunction(function) == before, Twine(name) + ": rejection mutated the source function");
}

bool testVerifierRejectsMalformedPlans(MLIRContext& context)
{
    bool passed = true;
    passed &= testMalformedPlan(context, "phase-order", [](SyncOneShotPlan& plan) { plan.phaseOrder.pop_back(); });
    passed &= testMalformedPlan(context, "source-operation", [](SyncOneShotPlan& plan) {
        plan.protocols.front().sourceOperation = plan.protocols.front().targetOperation;
    });
    passed &= testMalformedPlan(
        context, "source-pipe", [](SyncOneShotPlan& plan) { plan.protocols.front().sourcePipe = PIPE::PIPE_M; });
    passed &= testMalformedPlan(context, "tail", [](SyncOneShotPlan& plan) {
        plan.tailSectionOperation = plan.protocols.front().sourceOperation;
    });
    return passed;
}

Operation* findGenerated(func::FuncOp function, StringRef role, std::optional<unsigned> protocol = std::nullopt)
{
    Operation* result = nullptr;
    function.walk([&](Operation* operation) {
        auto candidateRole = operation->getAttrOfType<StringAttr>("pto.protocol_sync.role");
        const bool roleMatches = candidateRole && candidateRole.getValue() == role;
        if (!roleMatches) {
            return;
        }
        if (protocol) {
            auto candidateProtocol = operation->getAttrOfType<IntegerAttr>("pto.protocol_sync.protocol_id");
            const bool protocolMatches = candidateProtocol && candidateProtocol.getInt() == *protocol;
            if (!protocolMatches) {
                return;
            }
        }
        result = operation;
    });
    return result;
}

using CorruptMaterialization = std::function<void(func::FuncOp, const IRMapping&)>;

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
    const bool analysisReady = buildAnalysis(fixture);
    const std::string analysisFailure = (Twine(name) + ": cannot build ready plan").str();
    if (!check(analysisReady, analysisFailure)) {
        return false;
    }

    IRMapping mapping;
    OwningOpRef<ModuleOp> stagingModule = ModuleOp::create(function.getLoc());
    func::FuncOp clone = cast<func::FuncOp>(function->clone(mapping));
    stagingModule->push_back(clone);
    if (!check(
            succeeded(materializeOneShotProtocolPlan(clone, mapping, *fixture.plan)),
            Twine(name) + ": cannot materialize valid plan") ||
        !check(
            succeeded(verifyOneShotProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
            Twine(name) + ": verifier rejected valid materialization")) {
        return false;
    }
    corrupt(clone, mapping);
    return check(
        failed(verifyOneShotProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        Twine(name) + ": corrupted materialization was accepted");
}

bool testVerifierRejectsMalformedMaterializations(MLIRContext& context)
{
    bool passed = true;
    passed &= testMalformedMaterialization(context, "event-direction", [](func::FuncOp function, const IRMapping&) {
        auto set = cast<SetFlagOp>(findGenerated(function, "event-set", 0));
        set.setDstPipeAttr(PipeAttr::get(function.getContext(), PIPE::PIPE_MTE3));
    });
    passed &= testMalformedMaterialization(context, "event-id", [](func::FuncOp function, const IRMapping&) {
        auto set = cast<SetFlagOp>(findGenerated(function, "event-set", 0));
        set.setEventIdAttr(EventAttr::get(function.getContext(), EVENT::EVENT_ID5));
    });
    passed &= testMalformedMaterialization(context, "missing-wait", [](func::FuncOp function, const IRMapping&) {
        findGenerated(function, "event-wait", 0)->erase();
    });
    passed &= testMalformedMaterialization(context, "set-placement", [](func::FuncOp function, const IRMapping&) {
        Operation* set = findGenerated(function, "event-set", 0);
        Operation* wait = findGenerated(function, "event-wait", 0);
        set->moveAfter(wait);
    });
    passed &= testMalformedMaterialization(context, "omitted-boundary", [](func::FuncOp function, const IRMapping&) {
        findGenerated(function, "event-set", 0)->erase();
        findGenerated(function, "event-wait", 0)->erase();
    });
    passed &= testMalformedMaterialization(context, "missing-tail", [](func::FuncOp function, const IRMapping&) {
        findGenerated(function, "tail-drain")->erase();
    });
    passed &= testMalformedMaterialization(context, "misplaced-tail", [](func::FuncOp function, const IRMapping&) {
        Operation* tail = findGenerated(function, "tail-drain");
        tail->moveBefore(&function.getBody().front().front());
    });
    passed &= testMalformedMaterialization(context, "wrong-boundary-id", [](func::FuncOp function, const IRMapping&) {
        Operation* set = findGenerated(function, "event-set", 0);
        set->setAttr("pto.protocol_sync.protocol_id", IntegerAttr::get(IntegerType::get(function.getContext(), 32), 2));
    });
    passed &= testMalformedMaterialization(context, "untagged-sync", [](func::FuncOp function, const IRMapping&) {
        findGenerated(function, "event-set", 0)->removeAttr("pto.protocol_sync.generated");
    });
    passed &= testMalformedMaterialization(context, "extra-sync", [](func::FuncOp function, const IRMapping&) {
        func::ReturnOp returnOp;
        function.walk([&](func::ReturnOp operation) { returnOp = operation; });
        OpBuilder builder(returnOp);
        builder.create<BarrierOp>(returnOp.getLoc(), PipeAttr::get(function.getContext(), PIPE::PIPE_V));
    });
    return passed;
}

bool testVerifierRejectsDuplicateEventKey(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kRepeatedDomainFixture);
    if (!check(static_cast<bool>(module), "cannot parse repeated-domain fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture), "cannot build repeated-domain plan")) {
        return false;
    }

    IRMapping mapping;
    OwningOpRef<ModuleOp> stagingModule = ModuleOp::create(function.getLoc());
    func::FuncOp clone = cast<func::FuncOp>(function->clone(mapping));
    stagingModule->push_back(clone);
    if (!check(
            succeeded(materializeOneShotProtocolPlan(clone, mapping, *fixture.plan)),
            "cannot materialize repeated-domain plan") ||
        !check(
            succeeded(verifyOneShotProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
            "verifier rejected valid repeated-domain materialization")) {
        return false;
    }
    auto secondSet = cast<SetFlagOp>(findGenerated(clone, "event-set", 2));
    auto secondWait = cast<WaitFlagOp>(findGenerated(clone, "event-wait", 2));
    secondSet.setEventIdAttr(EventAttr::get(&context, EVENT::EVENT_ID0));
    secondWait.setEventIdAttr(EventAttr::get(&context, EVENT::EVENT_ID0));
    return check(
        failed(verifyOneShotProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        "verifier accepted a duplicate same-domain event key");
}

SyncOneShotPlan makeAllocationPlan(unsigned count, PIPE source, PIPE target)
{
    SyncOneShotPlan plan;
    plan.status = SyncOneShotPlanStatus::Ready;
    plan.targetKind = ProtocolSyncTargetKind::Npu2201A3;
    for (unsigned id = 0; id < count; ++id) {
        SyncOneShotProtocol protocol;
        protocol.id = id;
        protocol.kind = SyncOneShotProtocolKind::DirectedEvent;
        protocol.core = SyncPhysicalCore::Vector;
        protocol.sourcePipe = source;
        protocol.targetPipe = target;
        plan.protocols.push_back(protocol);
    }
    return plan;
}

bool testEventAllocation(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context);
    if (!check(static_cast<bool>(module), "cannot parse allocation fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture), "cannot build allocation fixture")) {
        return false;
    }

    bool passed = true;
    SyncOneShotPlan six = makeAllocationPlan(6, PIPE::PIPE_MTE2, PIPE::PIPE_V);
    ProtocolSyncStatistics statistics;
    passed &=
        check(succeeded(allocateOneShotProtocolEvents(fixture.schedule, six, &statistics)), "six allocation failed");
    passed &= check(six.status == SyncOneShotPlanStatus::Ready, "six generations exhausted the pool");
    for (auto [expected, protocol] : llvm::enumerate(six.protocols)) {
        passed &= check(protocol.eventId == expected, "same-domain allocation did not use a distinct event ID");
    }
    passed &= check(statistics.eventDomains == 1, "same-domain pressure reported the wrong domain count");
    passed &= check(statistics.maxEventDomainPressure == 6, "same-domain pressure was not reported");
    passed &= check(statistics.maximumEventIdPlusOne == 6, "maximum allocated event ID was not reported");

    SyncOneShotPlan seven = makeAllocationPlan(7, PIPE::PIPE_MTE2, PIPE::PIPE_V);
    passed &=
        check(succeeded(allocateOneShotProtocolEvents(fixture.schedule, seven)), "exhaustion was an internal error");
    passed &= check(seven.status == SyncOneShotPlanStatus::Unsupported, "seventh generation did not fail closed");
    passed &= check(
        !seven.rejections.empty() && seven.rejections.back().reason == SyncOneShotRejection::EventCapacity,
        "seventh generation did not report event capacity");

    SyncOneShotPlan domains = makeAllocationPlan(1, PIPE::PIPE_MTE2, PIPE::PIPE_V);
    SyncOneShotProtocol reverse = domains.protocols.front();
    reverse.id = 1;
    reverse.sourcePipe = PIPE::PIPE_V;
    reverse.targetPipe = PIPE::PIPE_MTE3;
    domains.protocols.push_back(reverse);
    passed &=
        check(succeeded(allocateOneShotProtocolEvents(fixture.schedule, domains)), "distinct-domain allocation failed");
    passed &= check(
        domains.protocols[0].eventId == 0 && domains.protocols[1].eventId == 0,
        "distinct domains did not independently reuse event ID zero");

    OwningOpRef<ModuleOp> reservationModule = parseSourceString<ModuleOp>(kReservationFixture, &context);
    if (!check(
            static_cast<bool>(reservationModule) && succeeded(verify(*reservationModule)),
            "cannot parse reservation fixture")) {
        return false;
    }
    func::FuncOp reservationFunction = *reservationModule->getOps<func::FuncOp>().begin();
    LegacySyncIRAdapter adapter;
    LegacySyncSnapshot legacy;
    passed &= check(succeeded(adapter.buildSnapshot(reservationFunction, legacy)), "cannot build reservation snapshot");
    SyncSemanticContext semanticContext = adapter.buildSemanticContext(legacy);
    StructuredSyncIR reservationSchedule(reservationFunction);
    StructuredSyncIRBuilder builder(semanticContext);
    passed &=
        check(succeeded(builder.build(reservationFunction, reservationSchedule)), "cannot build reservation schedule");
    SyncOneShotPlan reserved = makeAllocationPlan(1, PIPE::PIPE_V, PIPE::PIPE_S);
    passed &=
        check(succeeded(allocateOneShotProtocolEvents(reservationSchedule, reserved)), "reserved allocation failed");
    passed &= check(reserved.protocols.front().eventId == 1, "allocator reused hidden reserved event ID zero");
    return passed;
}

} // namespace

int main()
{
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect, PTODialect, scf::SCFDialect>();
    MLIRContext context(registry);
    bool passed = testTargetContract(context);
    passed &= testPlanTargetIdentity(context, "a3", "a2");
    passed &= testPlanTargetIdentity(context, "a2", "a3");
    passed &= testVerifierRejectsMalformedPlans(context);
    passed &= testVerifierRejectsMalformedMaterializations(context);
    passed &= testVerifierRejectsDuplicateEventKey(context);
    passed &= testEventAllocation(context);
    if (passed) {
        llvm::outs() << "protocol-sync target contract: pass\n";
        llvm::outs() << "protocol-sync malformed-plan rejection: pass\n";
        llvm::outs() << "protocol-sync emitted-IR verifier rejection: pass\n";
        llvm::outs() << "protocol-sync event allocation: pass\n";
    }
    return passed ? 0 : 1;
}
