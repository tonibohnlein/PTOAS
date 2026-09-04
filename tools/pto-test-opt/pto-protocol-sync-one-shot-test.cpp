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
#include "PTO/Transforms/ProtocolSync/EventAllocation.h"
#include "PTO/Transforms/ProtocolSync/OneShotProtocol.h"
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace mlir::pto::protocol_sync {

class StructuredSyncIRTestPeer {
public:
    static bool markGlobalReadOrdered(StructuredSyncIR& schedule, SyncPhaseId phase)
    {
        for (SyncAccess& access : schedule.accesses) {
            const bool isTargetRead = access.phase == phase && access.mode == SyncAccessMode::Read &&
                                      access.visibility == SyncVisibilityClass::Global;
            if (isTargetRead) {
                access.mode = SyncAccessMode::Ordered;
                return true;
            }
        }
        return false;
    }

    static bool markGlobalWriteOrdered(StructuredSyncIR& schedule, SyncPhaseId phase)
    {
        for (SyncAccess& access : schedule.accesses) {
            const bool isSourceWrite = access.phase == phase && access.mode == SyncAccessMode::Write &&
                                       access.visibility == SyncVisibilityClass::Global;
            if (isSourceWrite) {
                access.mode = SyncAccessMode::Ordered;
                return true;
            }
        }
        return false;
    }
};

} // namespace mlir::pto::protocol_sync

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

constexpr StringLiteral kSSACompletionFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @ssa_completion(
      %input: !pto.ptr<i32, gm>, %output: !pto.ptr<i32, gm>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %value = pto.load_scalar %input[%c0] : !pto.ptr<i32, gm> -> i32
    pto.store_scalar %value, %output[%c0] : !pto.ptr<i32, gm>, i32
    return
  }
}
)mlir";

constexpr StringLiteral kSSAPureForwardingFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @ssa_pure_forwarding(
      %input: !pto.ptr<i32, gm>, %output: !pto.ptr<i32, gm>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : i32
    %value = pto.load_scalar %input[%c0] : !pto.ptr<i32, gm> -> i32
    %forwarded = arith.addi %value, %c1 : i32
    pto.store_scalar %forwarded, %output[%c0] : !pto.ptr<i32, gm>, i32
    return
  }
}
)mlir";

constexpr StringLiteral kSSAChoiceFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @ssa_choice(
      %condition: i1, %then_input: !pto.ptr<i32, gm>,
      %else_input: !pto.ptr<i32, gm>, %output: !pto.ptr<i32, gm>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %value = scf.if %condition -> (i32) {
      %then_value = pto.load_scalar %then_input[%c0]
          : !pto.ptr<i32, gm> -> i32
      scf.yield %then_value : i32
    } else {
      %else_value = pto.load_scalar %else_input[%c0]
          : !pto.ptr<i32, gm> -> i32
      scf.yield %else_value : i32
    }
    pto.store_scalar %value, %output[%c0] : !pto.ptr<i32, gm>, i32
    return
  }
}
)mlir";

constexpr StringLiteral kSSALoopCarriedFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @ssa_loop_carried(
      %initial: i32, %input: !pto.ptr<i32, gm>,
      %output: !pto.ptr<i32, gm>, %trip_count: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %index = %c0 to %trip_count step %c1
        iter_args(%carried = %initial) -> (i32) {
      pto.store_scalar %carried, %output[%c0] : !pto.ptr<i32, gm>, i32
      %next = pto.load_scalar %input[%c0] : !pto.ptr<i32, gm> -> i32
      scf.yield %next : i32
    }
    pto.store_scalar %result, %output[%c0] : !pto.ptr<i32, gm>, i32
    return
  }
}
)mlir";

constexpr StringLiteral kNestedMemoryFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @nested_memory(
      %buffer: !pto.partition_tensor_view<16x16xf16>,
      %outer_count: index, %inner_count: index)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tile = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
    scf.for %outer = %c0 to %outer_count step %c1 {
      scf.for %inner = %c0 to %inner_count step %c1 {
        pto.tload ins(%buffer : !pto.partition_tensor_view<16x16xf16>)
                  outs(%tile : !pto.tile_buf<vec, 16x16xf16>)
        pto.tstore ins(%tile : !pto.tile_buf<vec, 16x16xf16>)
                   outs(%buffer : !pto.partition_tensor_view<16x16xf16>)
      }
    }
    return
  }
}
)mlir";

constexpr StringLiteral kVisibilityFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @gm_visibility(%buffer: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %c512 = arith.constant 512 : i64
    %source = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    %target = pto.alloc_tile addr = %c512 : !pto.tile_buf<vec, 16x16xf16>
    pto.tstore ins(%source : !pto.tile_buf<vec, 16x16xf16>)
               outs(%buffer : !pto.partition_tensor_view<16x16xf16>)
    pto.tload ins(%buffer : !pto.partition_tensor_view<16x16xf16>)
              outs(%target : !pto.tile_buf<vec, 16x16xf16>)
    return
  }
}
)mlir";

constexpr StringLiteral kOrderedUnknownAliasFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @ordered_unknown_alias(
      %first: !pto.partition_tensor_view<16x16xf16>,
      %second: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %c512 = arith.constant 512 : i64
    %source = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    %target = pto.alloc_tile addr = %c512 : !pto.tile_buf<vec, 16x16xf16>
    pto.tstore ins(%source : !pto.tile_buf<vec, 16x16xf16>)
               outs(%first : !pto.partition_tensor_view<16x16xf16>)
    pto.tstore ins(%target : !pto.tile_buf<vec, 16x16xf16>)
               outs(%second : !pto.partition_tensor_view<16x16xf16>)
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
    return true;
}

bool buildOneShotAnalysis(AnalysisFixture& fixture)
{
    if (!buildAnalysis(fixture)) {
        return false;
    }
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
    const bool analysisBuilt = buildOneShotAnalysis(fixture);
    const bool fixtureReady = check(analysisBuilt, Twine(name) + ": cannot build ready plan");
    if (!fixtureReady) {
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
    const bool analysisReady = buildOneShotAnalysis(fixture);
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

bool testVerifierRequiresDistinctCoexecutingEvents(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kRepeatedDomainFixture);
    if (!check(static_cast<bool>(module), "cannot parse repeated-domain fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildOneShotAnalysis(fixture), "cannot build repeated-domain plan")) {
        return false;
    }
    SmallVector<const SyncOneShotProtocol*, 2> repeatedDomain;
    for (const SyncOneShotProtocol& protocol : fixture.plan->protocols) {
        if (protocol.kind == SyncOneShotProtocolKind::DirectedEvent && protocol.sourcePipe == PIPE::PIPE_MTE2 &&
            protocol.targetPipe == PIPE::PIPE_V) {
            repeatedDomain.push_back(&protocol);
        }
    }
    if (!check(
            repeatedDomain.size() == 2 && repeatedDomain[0]->eventId == 0 && repeatedDomain[1]->eventId == 1,
            "coexecuting same-domain generations did not receive distinct event IDs")) {
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
    return check(
        succeeded(verifyOneShotProtocolMaterialization(fixture.schedule, *fixture.stages, clone, mapping)),
        "verifier rejected distinct coexecuting same-domain event IDs");
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
    if (!check(buildOneShotAnalysis(fixture), "cannot build allocation fixture")) {
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
    passed &= check(
        statistics.allocationGraphVertices == 6 && statistics.allocationGraphEdges == 15,
        "overlapping one-shot generations produced the wrong interference graph");
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

bool testControlExclusiveEventAllocation(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseFixture(context, kSSAChoiceFixture);
    if (!check(static_cast<bool>(module), "cannot parse control-exclusive allocation fixture")) {
        return false;
    }
    func::FuncOp function = *module->getOps<func::FuncOp>().begin();
    AnalysisFixture fixture(function);
    if (!check(buildAnalysis(fixture), "cannot analyze control-exclusive allocation fixture")) {
        return false;
    }
    SmallVector<const SyncPhase*, 2> guardedPhases;
    for (const SyncPhase& phase : fixture.schedule.getPhases()) {
        if (!phase.guard.empty()) {
            guardedPhases.push_back(&phase);
        }
    }
    const bool hasTwoGuardedPhases =
        check(guardedPhases.size() == 2, "choice fixture does not have two guarded phases");
    if (!hasTwoGuardedPhases) {
        return false;
    }
    SmallVector<SyncEventGeneration, 2> exclusive;
    for (const SyncPhase* phase : guardedPhases) {
        SyncEventGeneration generation;
        generation.id = exclusive.size();
        generation.kind = SyncEventGenerationKind::DirectRepair;
        generation.core = SyncPhysicalCore::Vector;
        generation.sourcePipe = PIPE::PIPE_MTE2;
        generation.targetPipe = PIPE::PIPE_V;
        generation.setAnchor = phase->operation;
        generation.waitAnchor = phase->operation->getBlock()->getTerminator();
        generation.guard.assign(phase->guard.begin(), phase->guard.end());
        exclusive.push_back(std::move(generation));
    }
    const ProtocolSyncTarget target = ProtocolSyncTarget::resolve(function);
    FailureOr<SyncEventAllocationResult> exclusiveAllocation = allocateSyncEventGenerations(target, {}, exclusive);
    bool passed = check(
        succeeded(exclusiveAllocation) && exclusiveAllocation->status == SyncEventAllocationStatus::Allocated &&
            exclusiveAllocation->graphEdges == 0 && exclusiveAllocation->eventIds.size() == 2 &&
            exclusiveAllocation->eventIds[0] == 0 && exclusiveAllocation->eventIds[1] == 0,
        "mutually exclusive event generations did not share one ID");

    SmallVector<SyncEventGeneration, 2> coexecuting = exclusive;
    for (SyncEventGeneration& generation : coexecuting) {
        generation.guard.clear();
    }
    FailureOr<SyncEventAllocationResult> coexecutingAllocation = allocateSyncEventGenerations(target, {}, coexecuting);
    passed &= check(
        succeeded(coexecutingAllocation) && coexecutingAllocation->status == SyncEventAllocationStatus::Allocated &&
            coexecutingAllocation->graphEdges == 1 && coexecutingAllocation->eventIds.size() == 2 &&
            coexecutingAllocation->eventIds[0] != coexecutingAllocation->eventIds[1],
        "coexecuting event generations did not interfere");
    for (SyncEventGeneration& generation : coexecuting) {
        generation.eventId = 0;
    }
    passed &= check(
        failed(verifySyncEventGenerationAssignment(target, {}, coexecuting, false)),
        "event verifier accepted one ID for coexecuting generations");

    SmallVector<SyncEventGeneration, 512> largeExclusive;
    for (unsigned arm = 0; arm < 512; ++arm) {
        SyncEventGeneration generation = exclusive.front();
        generation.id = arm;
        generation.guard.clear();
        generation.guard.push_back({0, arm});
        largeExclusive.push_back(std::move(generation));
    }
    FailureOr<SyncEventAllocationResult> largeAllocation =
        allocateSyncEventGenerations(target, {}, largeExclusive);
    passed &= check(
        succeeded(largeAllocation) && largeAllocation->status == SyncEventAllocationStatus::Allocated &&
            largeAllocation->graphEdges == 0 && largeAllocation->maximumDomainPressure == 1 &&
            llvm::all_equal(largeAllocation->eventIds),
        "large structurally exclusive allocation did not take the bounded edgeless fast path");
    const bool largeWasAllocated =
        succeeded(largeAllocation) && largeAllocation->status == SyncEventAllocationStatus::Allocated;
    if (largeWasAllocated) {
        for (auto [generation, eventId] : llvm::zip_equal(largeExclusive, largeAllocation->eventIds)) {
            generation.eventId = eventId;
        }
    }
    passed &= check(
        succeeded(verifySyncEventGenerationAssignment(target, {}, largeExclusive, false)),
        "allocator and verifier disagreed at the shared default domain limit");

    SmallVector<SyncEventGeneration, 1025> overLimit;
    overLimit.append(largeExclusive.begin(), largeExclusive.end());
    for (SyncEventGeneration& generation : overLimit) {
        generation.eventId.reset();
    }
    for (unsigned arm = overLimit.size(); arm < 1025; ++arm) {
        SyncEventGeneration generation = exclusive.front();
        generation.id = arm;
        generation.guard.clear();
        generation.guard.push_back({0, arm});
        overLimit.push_back(std::move(generation));
    }
    FailureOr<SyncEventAllocationResult> limitedAllocation =
        allocateSyncEventGenerations(target, {}, overLimit);
    passed &= check(
        succeeded(limitedAllocation) && limitedAllocation->status == SyncEventAllocationStatus::AnalysisLimit &&
            limitedAllocation->eventIds.empty() && limitedAllocation->searchLimitHits == 1,
        "generation-domain input limit was not reported as an analysis limit");
    SyncEventAllocationOptions invalidRaisedLimit;
    invalidRaisedLimit.maximumBacktrackingNodes = kHardMaximumEventBacktrackingNodes + 1;
    invalidRaisedLimit.maximumExactVertices = kHardMaximumExactEventVertices + 1;
    invalidRaisedLimit.maximumGenerationsPerDomain = kHardMaximumEventGenerationsPerDomain + 1;
    passed &= check(
        failed(allocateSyncEventGenerations(target, {}, largeExclusive, invalidRaisedLimit)) &&
            failed(verifySyncEventGenerationAssignment(target, {}, largeExclusive, false, invalidRaisedLimit)),
        "allocator or verifier accepted options above immutable safety caps");

    SmallVector<SyncEventGeneration, 3> boundedSearch;
    for (unsigned index = 0; index < 3; ++index) {
        SyncEventGeneration generation = exclusive.front();
        generation.id = index;
        generation.guard.clear();
        if (index != 1) {
            generation.guard.push_back({0, index / 2});
        }
        boundedSearch.push_back(std::move(generation));
    }
    SyncEventAllocationOptions lowBudget;
    lowBudget.maximumBacktrackingNodes = 1;
    FailureOr<SyncEventAllocationResult> boundedAllocation =
        allocateSyncEventGenerations(target, {}, boundedSearch, lowBudget);
    passed &= check(
        succeeded(boundedAllocation) && boundedAllocation->status == SyncEventAllocationStatus::Allocated &&
            boundedAllocation->eventIds.size() == 3 &&
            boundedAllocation->eventIds[0] == boundedAllocation->eventIds[2] &&
            boundedAllocation->eventIds[0] != boundedAllocation->eventIds[1] &&
            boundedAllocation->searchLimitHits == 1 && boundedAllocation->backtrackingNodes == 1,
        "bounded minimization discarded or misclassified a known feasible assignment");

    SyncEventAllocationOptions loweredExactLimit;
    loweredExactLimit.maximumExactVertices = 2;
    FailureOr<SyncEventAllocationResult> loweredExactAllocation =
        allocateSyncEventGenerations(target, {}, boundedSearch, loweredExactLimit);
    passed &= check(
        succeeded(loweredExactAllocation) &&
            loweredExactAllocation->status == SyncEventAllocationStatus::Allocated &&
            loweredExactAllocation->maximumDomainPressure == 2 && loweredExactAllocation->searchLimitHits == 1 &&
            loweredExactAllocation->backtrackingNodes == 0,
        "lowered exact-search cap did not skip recursive minimization and retain feasibility");

    SmallVector<SyncEventGeneration, 129> aboveExactLimit;
    for (unsigned arm = 0; arm < 128; ++arm) {
        SyncEventGeneration generation = exclusive.front();
        generation.id = arm;
        generation.guard.clear();
        generation.guard.push_back({0, arm});
        aboveExactLimit.push_back(std::move(generation));
    }
    SyncEventGeneration central = exclusive.front();
    central.id = aboveExactLimit.size();
    central.guard.clear();
    aboveExactLimit.push_back(std::move(central));
    FailureOr<SyncEventAllocationResult> aboveExactAllocation =
        allocateSyncEventGenerations(target, {}, aboveExactLimit);
    passed &= check(
        succeeded(aboveExactAllocation) && aboveExactAllocation->status == SyncEventAllocationStatus::Allocated &&
            aboveExactAllocation->graphEdges == 128 && aboveExactAllocation->maximumDomainPressure == 2 &&
            aboveExactAllocation->searchLimitHits == 1 && aboveExactAllocation->backtrackingNodes == 0,
        "above-cap nontrivial graph entered recursive minimization or lost its feasible coloring");
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
    if (!check(buildOneShotAnalysis(fixture), "cannot build selected-world plan")) {
        return false;
    }
    FailureOr<SyncSelectedWorld> world = buildSelectedWorld(*fixture.plan, fixture.channels);
    if (!check(succeeded(world), "cannot adapt one-shot selected world")) {
        return false;
    }
    FailureOr<SyncInterpretationResult> selected =
        interpretSelectedWorld(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *world);
    bool passed = check(succeeded(selected) && selected->isComplete(), "complete selected world left residuals");

    SyncSelectedWorld empty;
    FailureOr<SyncInterpretationResult> residuals =
        interpretSelectedWorld(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, empty);
    passed &= check(succeeded(residuals) && !residuals->isComplete(), "empty selected world did not expose residuals");
    if (world->completions.empty()) {
        return false;
    }
    world->completions.erase(world->completions.begin());
    ProtocolSyncStatistics partialStatistics;
    passed &= check(
        failed(interpretSelectedWorld(
            fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *world, &partialStatistics)),
        "incomplete selected protocol was reported as a repairable residual");
    passed &= check(
        partialStatistics.interpreterTransitions > 0,
        "failed interpretation discarded completed transition statistics");

    FailureOr<SyncSelectedWorld> malformed = buildSelectedWorld(*fixture.plan, fixture.channels);
    if (!check(succeeded(malformed), "cannot rebuild one-shot selected world")) {
        return false;
    }
    SyncSelectedCompletion reverse = malformed->completions.front();
    std::swap(reverse.source, reverse.target);
    malformed->completions.push_back(reverse);
    passed &= check(
        failed(
            interpretSelectedWorld(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, *malformed)),
        "backward same-iteration completion was accepted");
    return passed;
}

bool hasObligation(const SyncInterpretationResult& result, SyncObligationKind kind)
{
    return llvm::any_of(
        result.obligations, [&](const SyncResidualObligation& obligation) { return obligation.kind == kind; });
}

bool hasObligationBetween(
    const SyncInterpretationResult& result, SyncObligationKind kind, SyncPhaseId source, SyncPhaseId target,
    SyncIterationRelationKind iteration)
{
    return llvm::any_of(result.obligations, [&](const SyncResidualObligation& obligation) {
        return obligation.kind == kind && obligation.source == source && obligation.target == target &&
               obligation.iteration.kind == iteration;
    });
}

bool hasObligationWithRelation(
    const SyncInterpretationResult& result, SyncObligationKind kind, SyncPhaseId source, SyncPhaseId target,
    const SyncIterationRelation& relation)
{
    return llvm::any_of(result.obligations, [&](const SyncResidualObligation& obligation) {
        return obligation.kind == kind && obligation.source == source && obligation.target == target &&
               obligation.iteration.kind == relation.kind && obligation.iteration.distance == relation.distance &&
               obligation.iteration.carrier == relation.carrier;
    });
}

bool testIndependentResidualEffects(MLIRContext& context)
{
    OwningOpRef<ModuleOp> ssaModule = parseFixture(context, kSSACompletionFixture);
    if (!check(static_cast<bool>(ssaModule), "cannot parse SSA-completion fixture")) {
        return false;
    }
    AnalysisFixture ssaFixture(*ssaModule->getOps<func::FuncOp>().begin());
    if (!check(buildAnalysis(ssaFixture), "cannot analyze SSA-completion fixture")) {
        return false;
    }
    SyncSelectedWorld empty;
    FailureOr<SyncInterpretationResult> uncovered = interpretSelectedWorld(
        ssaFixture.schedule, *ssaFixture.stages, ssaFixture.timelines, ssaFixture.channels, empty);
    bool passed = check(
        succeeded(uncovered) && !hasObligation(*uncovered, SyncObligationKind::SSACompletion),
        "intrinsically ordered scalar SSA dependency produced a residual");

    SyncSelectedWorld ordered;
    ordered.completions.push_back(
        {0, 1, SyncControlRelation::MustExecute, {SyncIterationRelationKind::SameIteration, 0}});
    ordered.exitCompletedPhases.push_back(1);
    FailureOr<SyncInterpretationResult> covered = interpretSelectedWorld(
        ssaFixture.schedule, *ssaFixture.stages, ssaFixture.timelines, ssaFixture.channels, ordered);
    passed &= check(
        succeeded(covered) && !hasObligation(*covered, SyncObligationKind::SSACompletion),
        "selected completion did not discharge the SSA dependency");

    OwningOpRef<ModuleOp> pureModule = parseFixture(context, kSSAPureForwardingFixture);
    if (!check(static_cast<bool>(pureModule), "cannot parse pure SSA-forwarding fixture")) {
        return false;
    }
    AnalysisFixture pureFixture(*pureModule->getOps<func::FuncOp>().begin());
    if (!check(buildAnalysis(pureFixture), "cannot analyze pure SSA-forwarding fixture")) {
        return false;
    }
    FailureOr<SyncInterpretationResult> pure = interpretSelectedWorld(
        pureFixture.schedule, *pureFixture.stages, pureFixture.timelines, pureFixture.channels, empty);
    passed &= check(
        succeeded(pure) &&
            !hasObligationBetween(
                *pure, SyncObligationKind::SSACompletion, 0, 1, SyncIterationRelationKind::SameIteration),
        "pure scalar SSA forwarding produced a completion residual");

    OwningOpRef<ModuleOp> choiceModule = parseFixture(context, kSSAChoiceFixture);
    if (!check(static_cast<bool>(choiceModule), "cannot parse conditional SSA fixture")) {
        return false;
    }
    AnalysisFixture choiceFixture(*choiceModule->getOps<func::FuncOp>().begin());
    if (!check(buildAnalysis(choiceFixture), "cannot analyze conditional SSA fixture")) {
        return false;
    }
    FailureOr<SyncInterpretationResult> choice = interpretSelectedWorld(
        choiceFixture.schedule, *choiceFixture.stages, choiceFixture.timelines, choiceFixture.channels, empty);
    const bool thenOrdered = succeeded(choice) && !hasObligationBetween(
                                                      *choice, SyncObligationKind::SSACompletion, 0, 2,
                                                      SyncIterationRelationKind::SameIteration);
    const bool elseOrdered = succeeded(choice) && !hasObligationBetween(
                                                      *choice, SyncObligationKind::SSACompletion, 1, 2,
                                                      SyncIterationRelationKind::SameIteration);
    passed &= check(thenOrdered && elseOrdered, "conditional scalar SSA source produced a completion residual");

    OwningOpRef<ModuleOp> loopModule = parseFixture(context, kSSALoopCarriedFixture);
    if (!check(static_cast<bool>(loopModule), "cannot parse loop-carried SSA fixture")) {
        return false;
    }
    AnalysisFixture loopFixture(*loopModule->getOps<func::FuncOp>().begin());
    if (!check(buildAnalysis(loopFixture), "cannot analyze loop-carried SSA fixture")) {
        return false;
    }
    FailureOr<SyncInterpretationResult> loop = interpretSelectedWorld(
        loopFixture.schedule, *loopFixture.stages, loopFixture.timelines, loopFixture.channels, empty);
    const bool loopOrdered =
        succeeded(loop) &&
        !hasObligationBetween(*loop, SyncObligationKind::SSACompletion, 1, 0, SyncIterationRelationKind::LoopCarried);
    passed &= check(loopOrdered, "loop-carried scalar SSA source produced a completion residual");

    OwningOpRef<ModuleOp> nestedModule = parseFixture(context, kNestedMemoryFixture);
    if (!check(static_cast<bool>(nestedModule), "cannot parse nested memory fixture")) {
        return false;
    }
    AnalysisFixture nestedFixture(*nestedModule->getOps<func::FuncOp>().begin());
    if (!check(buildAnalysis(nestedFixture), "cannot analyze nested memory fixture")) {
        return false;
    }
    ArrayRef<SyncPhase> nestedPhases = nestedFixture.schedule.getPhases();
    const bool validNestedShape = nestedPhases.size() == 2 && nestedPhases[0].iterationDomain.loops.size() == 2;
    if (!check(validNestedShape, "nested memory fixture has an unexpected phase shape")) {
        return false;
    }
    const SyncRegionId outerCarrier = nestedPhases[0].iterationDomain.loops.front();
    const SyncRegionId innerCarrier = nestedPhases[0].iterationDomain.loops.back();
    SyncSelectedWorld innerOnly;
    innerOnly.visibility.push_back(
        {1, 0, SyncControlRelation::MustExecute, {SyncIterationRelationKind::LoopCarried, 1, innerCarrier}});
    innerOnly.exitCompletedPhases.append({0, 1});
    FailureOr<SyncInterpretationResult> nested = interpretSelectedWorld(
        nestedFixture.schedule, *nestedFixture.stages, nestedFixture.timelines, nestedFixture.channels, innerOnly);
    const SyncIterationRelation outerRelation{SyncIterationRelationKind::Unknown, 0, outerCarrier};
    const SyncIterationRelation innerRelation{SyncIterationRelationKind::LoopCarried, 1, innerCarrier};
    const bool outerResidual =
        succeeded(nested) && hasObligationWithRelation(*nested, SyncObligationKind::Visibility, 1, 0, outerRelation);
    const bool innerDischarged =
        succeeded(nested) && !hasObligationWithRelation(*nested, SyncObligationKind::Visibility, 1, 0, innerRelation);
    passed &= check(
        outerResidual && innerDischarged, "inner-loop visibility incorrectly discharged the enclosing-loop boundary");

    OwningOpRef<ModuleOp> visibilityModule = parseFixture(context, kVisibilityFixture);
    if (!check(static_cast<bool>(visibilityModule), "cannot parse GM-visibility fixture")) {
        return false;
    }
    AnalysisFixture visibilityFixture(*visibilityModule->getOps<func::FuncOp>().begin());
    if (!check(buildAnalysis(visibilityFixture), "cannot analyze GM-visibility fixture")) {
        return false;
    }
    SyncSelectedWorld completionOnly;
    completionOnly.completions.push_back(
        {0, 1, SyncControlRelation::MustExecute, {SyncIterationRelationKind::SameIteration, 0}});
    completionOnly.exitCompletedPhases.append({0, 1});
    FailureOr<SyncInterpretationResult> visibility = interpretSelectedWorld(
        visibilityFixture.schedule, *visibilityFixture.stages, visibilityFixture.timelines, visibilityFixture.channels,
        completionOnly);
    passed &= check(
        succeeded(visibility) && hasObligation(*visibility, SyncObligationKind::Visibility),
        "generic phase completion incorrectly discharged GM visibility");

    completionOnly.visibility.push_back(
        {0, 1, SyncControlRelation::MustExecute, {SyncIterationRelationKind::SameIteration, 0}});
    FailureOr<SyncInterpretationResult> published = interpretSelectedWorld(
        visibilityFixture.schedule, *visibilityFixture.stages, visibilityFixture.timelines, visibilityFixture.channels,
        completionOnly);
    passed &= check(
        succeeded(published) && !hasObligation(*published, SyncObligationKind::Visibility),
        "qualified GM visibility did not discharge the publication obligation");

    // Operation providers do not yet produce Ordered accesses, so use the
    // schedule's test peer to exercise the interpreter's forward contract.
    const bool markedOrdered = StructuredSyncIRTestPeer::markGlobalReadOrdered(visibilityFixture.schedule, 1);
    FailureOr<SyncInterpretationResult> orderedTargetResult = interpretSelectedWorld(
        visibilityFixture.schedule, *visibilityFixture.stages, visibilityFixture.timelines, visibilityFixture.channels,
        completionOnly);
    passed &= check(
        markedOrdered && succeeded(orderedTargetResult) &&
            hasObligation(*orderedTargetResult, SyncObligationKind::OrderedMemory),
        "selected completion or visibility discharged an ordered target access");

    OwningOpRef<ModuleOp> orderedUnknownModule = parseFixture(context, kOrderedUnknownAliasFixture);
    if (!check(static_cast<bool>(orderedUnknownModule), "cannot parse ordered unknown-alias fixture")) {
        return false;
    }
    AnalysisFixture orderedUnknownFixture(*orderedUnknownModule->getOps<func::FuncOp>().begin());
    if (!check(buildAnalysis(orderedUnknownFixture), "cannot analyze ordered unknown-alias fixture")) {
        return false;
    }
    const bool markedSourceOrdered =
        StructuredSyncIRTestPeer::markGlobalWriteOrdered(orderedUnknownFixture.schedule, 0);
    SyncSelectedWorld selectedOrder;
    selectedOrder.completions.push_back(
        {0, 1, SyncControlRelation::MustExecute, {SyncIterationRelationKind::SameIteration, 0}});
    selectedOrder.exitCompletedPhases.append({0, 1});
    FailureOr<SyncInterpretationResult> orderedUnknown = interpretSelectedWorld(
        orderedUnknownFixture.schedule, *orderedUnknownFixture.stages, orderedUnknownFixture.timelines,
        orderedUnknownFixture.channels, selectedOrder);
    const bool preservesOrderedEffect =
        succeeded(orderedUnknown) &&
        hasObligationBetween(
            *orderedUnknown, SyncObligationKind::OrderedMemory, 0, 1, SyncIterationRelationKind::SameIteration) &&
        !hasObligationBetween(
            *orderedUnknown, SyncObligationKind::Completion, 0, 1, SyncIterationRelationKind::SameIteration);
    passed &= check(
        markedSourceOrdered && preservesOrderedEffect,
        "unknown aliasing converted an ordered endpoint into a repairable completion");
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
    passed &= testVerifierRequiresDistinctCoexecutingEvents(context);
    passed &= testEventAllocation(context);
    passed &= testControlExclusiveEventAllocation(context);
    passed &= testSelectedWorldInterpreter(context);
    passed &= testIndependentResidualEffects(context);
    if (passed) {
        llvm::outs() << "protocol-sync target contract: pass\n";
        llvm::outs() << "protocol-sync malformed-plan rejection: pass\n";
        llvm::outs() << "protocol-sync emitted-IR verifier rejection: pass\n";
        llvm::outs() << "protocol-sync event allocation: pass\n";
        llvm::outs() << "protocol-sync selected-world interpretation: pass\n";
    }
    return passed ? 0 : 1;
}
