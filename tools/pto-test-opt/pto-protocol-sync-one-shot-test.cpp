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
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>

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

bool check(bool condition, const Twine& message)
{
    if (condition) {
        return true;
    }
    llvm::errs() << "FAIL: " << message << '\n';
    return false;
}

OwningOpRef<ModuleOp> parseFixture(MLIRContext& context)
{
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(kFixture, &context);
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
    fixture.plan = buildOneShotProtocolPlan(fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels);
    if (failed(fixture.plan) || fixture.plan->status != SyncOneShotPlanStatus::Ready) {
        return false;
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
    ProtocolSyncTarget target = ProtocolSyncTarget::resolve(function);
    bool passed = check(target.isSupported(), "explicit A3 target was rejected");
    const auto cube = [](PIPE pipe) { return SyncOneShotResource{SyncPhysicalCore::Cube, pipe}; };
    passed &= check(!target.supportsEvent(cube(PIPE::PIPE_MTE1), cube(PIPE::PIPE_FIX)), "MTE1_FIX admitted");
    passed &= check(!target.supportsEvent(cube(PIPE::PIPE_MTE3), cube(PIPE::PIPE_FIX)), "MTE3_FIX admitted");
    passed &= check(!target.supportsEvent(cube(PIPE::PIPE_FIX), cube(PIPE::PIPE_MTE1)), "FIX_MTE1 admitted");
    passed &= check(target.supportsEvent(cube(PIPE::PIPE_M), cube(PIPE::PIPE_FIX)), "M_FIX rejected");
    passed &= check(target.supportsEvent(cube(PIPE::PIPE_MTE2), cube(PIPE::PIPE_FIX)), "MTE2_FIX rejected");
    passed &= check(target.supportsEvent(cube(PIPE::PIPE_FIX), cube(PIPE::PIPE_MTE3)), "FIX_MTE3 rejected");
    passed &= check(target.getCompilerEventIds() == ArrayRef<unsigned>({0, 1, 2, 3, 4, 5}), "event pool changed");

    module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, "a2"));
    passed &= check(!ProtocolSyncTarget::resolve(function).isSupported(), "unqualified A2 target was admitted");
    module->getOperation()->setAttr("pto.target_arch", StringAttr::get(&context, "a3"));
    for (StringRef deviceSpec : {"Ascend910", "Ascend910A", "Ascend910B1", "Ascend910B3", "Ascend910_1234"}) {
        module->getOperation()->setAttr("pto.device-spec", StringAttr::get(&context, deviceSpec));
        passed &= check(
            !ProtocolSyncTarget::resolve(function).isSupported(),
            Twine("unqualified device profile admitted: ") + deviceSpec);
    }
    module->getOperation()->setAttr("pto.device-spec", UnitAttr::get(&context));
    passed &= check(!ProtocolSyncTarget::resolve(function).isSupported(), "non-string device profile was admitted");
    module->getOperation()->removeAttr("pto.device-spec");
    module->getOperation()->removeAttr("pto.target_arch");
    passed &= check(!ProtocolSyncTarget::resolve(function).isSupported(), "attr-less target was admitted");
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
    const bool rejected = failed(materializeAndVerifyOneShotProtocolPlan(
        fixture.schedule, *fixture.stages, fixture.timelines, fixture.channels, malformed));
    return check(rejected, Twine(name) + ": malformed plan was accepted") &&
           check(printFunction(function) == before, Twine(name) + ": rejection mutated the source function");
}

bool testVerifierRejectsMalformedPlans(MLIRContext& context)
{
    bool passed = true;
    passed &= testMalformedPlan(context, "phase-order", [](SyncOneShotPlan& plan) { plan.phaseOrder.pop_back(); });
    passed &= testMalformedPlan(context, "source-phase", [](SyncOneShotPlan& plan) {
        plan.protocols.front().sourcePhase = plan.protocols.front().targetPhase;
    });
    passed &= testMalformedPlan(context, "source-operation", [](SyncOneShotPlan& plan) {
        plan.protocols.front().sourceOperation = plan.protocols.front().targetOperation;
    });
    passed &= testMalformedPlan(
        context, "source-pipe", [](SyncOneShotPlan& plan) { plan.protocols.front().sourcePipe = PIPE::PIPE_M; });
    passed &= testMalformedPlan(
        context, "core", [](SyncOneShotPlan& plan) { plan.protocols.front().core = SyncPhysicalCore::Cube; });
    passed &=
        testMalformedPlan(context, "channel", [](SyncOneShotPlan& plan) { plan.protocols.front().channels.clear(); });
    passed &= testMalformedPlan(context, "tail", [](SyncOneShotPlan& plan) {
        plan.tailSectionOperation = plan.protocols.front().sourceOperation;
    });
    return passed;
}

} // namespace

int main()
{
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect, PTODialect, scf::SCFDialect>();
    MLIRContext context(registry);
    bool passed = testTargetContract(context);
    passed &= testVerifierRejectsMalformedPlans(context);
    if (passed) {
        llvm::outs() << "protocol-sync target contract: pass\n";
        llvm::outs() << "protocol-sync malformed-plan rejection: pass\n";
    }
    return passed ? 0 : 1;
}
