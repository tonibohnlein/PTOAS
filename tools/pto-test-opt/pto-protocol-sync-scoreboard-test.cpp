// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- pto-protocol-sync-scoreboard-test.cpp - Concrete negative controls ---===//
// Reduced hardware-graph checks: pipe-local barriers, directed token payloads,
// outstanding RAW/WAR/WAW effects, and completion learned through event chains.

#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"
#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

const std::string kLoad = "pto.tload ins(%in : !pto.partition_tensor_view<16x16xf16>) "
                          "outs(%a : !pto.tile_buf<vec, 16x16xf16>)\n";
const std::string kCompute = "pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf16>) "
                             "outs(%b : !pto.tile_buf<vec, 16x16xf16>)\n";
const std::string kStore = "pto.tstore ins(%b : !pto.tile_buf<vec, 16x16xf16>) "
                           "outs(%out : !pto.partition_tensor_view<16x16xf16>)\n";
const std::string kSet = "pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]\n";
const std::string kWait = "pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]\n";
const std::string kReady = kSet + kWait;
const std::string kPublish = "pto.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]\n"
                             "pto.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]\n";
const std::string kRelease = "pto.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]\n"
                             "pto.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]\n";
const std::string kVectorBarrier = "pto.barrier <PIPE_V>\n";
const std::string kLoadBarrier = "pto.barrier <PIPE_MTE2>\n";

enum class ExitDrain { Present, Missing };

struct Case {
    const char* name;
    std::string body;
    bool scoreboard;
    std::optional<bool> fullVerifier;
    bool needsWitness = false;
    ExitDrain exitDrain = ExitDrain::Present;
    StringRef expectedFailureStage = {};
};

std::string program(StringRef arch, StringRef body, ExitDrain exitDrain)
{
    const StringRef drain = exitDrain == ExitDrain::Present ? "pto.barrier <PIPE_ALL>\n" : "";
    return "module attributes {pto.target_arch = \"" + arch.str() + R"mlir("} {
      func.func @scoreboard(%in: !pto.partition_tensor_view<16x16xf16>,
                           %out: !pto.partition_tensor_view<16x16xf16>)
          attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
        %c0 = arith.constant 0 : i64
        %c512 = arith.constant 512 : i64
        %a = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
        %b = pto.alloc_tile addr = %c512 : !pto.tile_buf<vec, 16x16xf16>
    )mlir" +
           body.str() + drain.str() + "return\n}\n}\n";
}

bool fabricatedGraphCovers(const StructuredSyncIR& schedule)
{
    SyncSelectedWorld fabricated;
    for (const SyncPhase& first : schedule.getPhases()) {
        for (const SyncPhase& second : schedule.getPhases()) {
            if (first.id < second.id) {
                fabricated.completions.push_back(
                    {first.id,
                     second.id,
                     SyncControlRelation::MustExecute,
                     {SyncIterationRelationKind::SameIteration, 0}});
            }
        }
    }
    return succeeded(verifyLocalMemoryCoverage(schedule, fabricated));
}

bool runCase(MLIRContext& context, StringRef arch, const Case& test)
{
    auto module = parseSourceString<ModuleOp>(program(arch, test.body, test.exitDrain), &context);
    if (!module) {
        return false;
    }
    auto function = *module->getOps<func::FuncOp>().begin();
    LegacySyncIRAdapter adapter;
    LegacySyncSnapshot snapshot;
    if (failed(adapter.buildSnapshot(function, snapshot))) {
        return false;
    }
    SyncSemanticContext semantics = adapter.buildSemanticContext(snapshot);
    StructuredSyncIR schedule(function);
    const bool invalidSchedule =
        failed(StructuredSyncIRBuilder(semantics).build(function, schedule)) || !schedule.getFailures().empty();
    if (invalidSchedule) {
        return false;
    }
    SyncAccessId source = kInvalidSyncId;
    SyncAccessId target = kInvalidSyncId;
    const bool result = succeeded(verifyConcreteLocalScoreboard(schedule, &source, &target));
    const bool witness = !test.needsWitness || (schedule.findAccess(source) && schedule.findAccess(target));
    // An overstrong completion graph covers these hazards, but the scoreboard
    // must still reject: it consumes actual synchronization, not this graph.
    const bool independent = !test.needsWitness || fabricatedGraphCovers(schedule);
    StringRef firstFailedStage;
    bool full = true;
    if (test.fullVerifier) {
        const bool accepted = succeeded(verifyConcreteSyncSemantics(semantics, function, nullptr, &firstFailedStage));
        full = accepted == *test.fullVerifier;
    }
    const bool expectedStage = test.expectedFailureStage.empty() || firstFailedStage == test.expectedFailureStage;
    if (result != test.scoreboard || !witness || !independent || !full || !expectedStage) {
        llvm::errs() << "FAIL " << arch << ':' << test.name << " scoreboard=" << result << " witness=" << witness
                     << " independent=" << independent << " full-verdict-matches=" << full
                     << " first-failed-stage=" << firstFailedStage << '\n';
        return false;
    }
    return true;
}

bool runCases(MLIRContext& context)
{
    const std::string unrelated = "pto.set_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]\n"
                                  "pto.wait_flag[<PIPE_MTE2>, <PIPE_MTE3>, <EVENT_ID0>]\n";
    const Case cases[] = {
        {"direct-chain", kLoad + kReady + kCompute + kPublish + kStore, true, true},
        {"named-barrier-not-cross-pipe", kLoad + kReady + kCompute + kVectorBarrier + kStore, false, false, true},
        {"unrelated-token-not-publication", kLoad + kReady + kCompute + kVectorBarrier + unrelated + kStore, false,
         false, true},
        {"overlapping-loads", kLoad + kLoad, false, false, true},
        {"overlapping-loads-barrier", kLoad + kLoadBarrier + kLoad, true, true},
        // Local hazard coverage alone cannot certify completion at return.
        {"named-barrier-missing-exit-drain", kLoad + kLoadBarrier, true, false, false, ExitDrain::Missing,
         "residual-obligations"},
        {"named-barrier-restored-exit-drain", kLoad + kLoadBarrier, true, true},
        {"redundant-terminal-drains", kLoad + kLoadBarrier + kLoadBarrier, true, true},
        {"wrong-barrier", kLoad + kVectorBarrier + kLoad, false, false, true},
        {"reclamation-missing", kLoad + kReady + kCompute + kLoad, false, false, true},
        {"reclamation-proven", kLoad + kReady + kCompute + kRelease + kLoad, true, true},
        {"same-pipe-waw", kLoad + kReady + kCompute + kCompute, false, false, true},
        {"same-pipe-barrier", kLoad + kReady + kCompute + kVectorBarrier + kCompute, true, true},
        {"set-before-producer", kSet + kLoad + kWait + kCompute, false, false, true},
        {"wait-after-consumer", kLoad + kSet + kCompute + kWait, false, false, true},
        {"rearm-live-token", kLoad + kSet + kSet + kWait + kCompute, false, false},
        {"missing-set", kLoad + kWait + kCompute, false, false},
        {"missing-wait", kLoad + kSet + kCompute, false, false, true},
        {"unconsumed-token", kLoad + kSet, false, false},
        // Payload forwarding without an intermediate V operation is useful
        // scoreboard evidence, not an admission expansion: F's current event
        // extractor still requires physical endpoints for each direct pair.
        {"empty-lane-relay",
         kLoad + kReady + kPublish +
             "pto.tstore ins(%a : !pto.tile_buf<vec, 16x16xf16>) "
             "outs(%out : !pto.partition_tensor_view<16x16xf16>)\n",
         true, std::nullopt},
    };
    for (StringRef arch : {"a2", "a3"}) {
        for (const Case& test : cases) {
            if (!runCase(context, arch, test)) {
                return false;
            }
        }
    }
    llvm::outs() << "protocol-sync concrete scoreboard: 40 A2/A3 cases pass\n";
    return true;
}

} // namespace

int main()
{
    DialectRegistry registry;
    registry.insert<PTODialect, arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
    MLIRContext context(registry);
    context.disableMultithreading();
    return runCases(context) ? 0 : 1;
}
