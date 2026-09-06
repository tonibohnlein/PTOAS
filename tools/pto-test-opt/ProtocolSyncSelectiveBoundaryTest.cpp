// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Entry/exit/bypass coverage and independent zero-trip overlap execution.
#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/SelectiveLoopRepair.h"
#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"
#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

bool checkSelectiveBoundaryInterleavings(const StructuredSyncIR&, unsigned, bool*);

namespace {
constexpr StringLiteral kFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
 func.func @boundary(%n: index, %x: !pto.partition_tensor_view<16x16xf16>,
                     %y: !pto.partition_tensor_view<16x16xf16>, %z: !pto.partition_tensor_view<16x16xf16>)
 attributes {pto.kernel_kind = #pto.kernel_kind<vector>, pto.gm_alias = "assume-disjoint-arguments"} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %base = arith.constant 0 : i64
  %other = arith.constant 512 : i64
  %output = arith.constant 1024 : i64
  %a = pto.alloc_tile addr = %base : !pto.tile_buf<vec, 16x16xf16>
  %b = pto.alloc_tile addr = %other : !pto.tile_buf<vec, 16x16xf16>
  %c = pto.alloc_tile addr = %output : !pto.tile_buf<vec, 16x16xf16>
  pto.tload ins(%y : !pto.partition_tensor_view<16x16xf16>) outs(%b : !pto.tile_buf<vec, 16x16xf16>)
  pto.tload ins(%x : !pto.partition_tensor_view<16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)
  scf.for %i = %zero to %n step %one {
   pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf16>) outs(%c : !pto.tile_buf<vec, 16x16xf16>)
   pto.tstore ins(%c : !pto.tile_buf<vec, 16x16xf16>) outs(%z : !pto.partition_tensor_view<16x16xf16>)
  }
  pto.tabs ins(%b : !pto.tile_buf<vec, 16x16xf16>) outs(%b : !pto.tile_buf<vec, 16x16xf16>)
  return
 }
})mlir";

bool check(bool condition, StringRef detail)
{
    if (!condition) {
        llvm::errs() << "FAIL selective boundary: " << detail << '\n';
    }
    return condition;
}

std::unique_ptr<StructuredSyncIR> extract(ModuleOp module)
{
    auto function = *module.getOps<func::FuncOp>().begin();
    LegacySyncIRAdapter adapter;
    LegacySyncSnapshot snapshot;
    if (failed(adapter.buildSnapshot(function, snapshot))) {
        return {};
    }
    auto schedule = std::make_unique<StructuredSyncIR>(function);
    if (failed(StructuredSyncIRBuilder(adapter.buildSemanticContext(snapshot)).build(function, *schedule))) {
        return {};
    }
    return schedule;
}

bool prepare(ModuleOp module)
{
    auto schedule = extract(module);
    if (!schedule) {
        return false;
    }
    auto stages = analyzePipelineStages(*schedule);
    if (failed(stages)) {
        return false;
    }
    auto timelines = analyzeStorageTimelines(*schedule, *stages);
    auto channels = analyzeChannels(*schedule, *stages, timelines);
    auto plan = buildMixedProtocolPlan(*schedule, *stages, timelines, channels, false);
    const bool ready = succeeded(plan) && plan->isComplete() && plan->selectiveLoop;
    if (!check(ready, "complete general-only boundary plan")) {
        return false;
    }
    const bool allocated = succeeded(allocateMixedProtocolEvents(*schedule, *plan));
    const bool verified =
        allocated && succeeded(verifyMixedProtocolPlan(*schedule, *stages, timelines, channels, *plan));
    if (!verified) {
        return false;
    }
    // The verifier must reject malformed logical endpoints without entering
    // allocation with a null anchor or an empty recurring source-loop list.
    for (unsigned mutation = 0; mutation < 3; ++mutation) {
        auto broken = *plan;
        auto& edge = broken.selectiveLoop->edges.front();
        if (mutation == 0) {
            edge.placement.source = nullptr;
        } else if (mutation == 1) {
            edge.completion.source = kInvalidSyncId;
        } else {
            for (auto& recurring : broken.selectiveLoop->edges) {
                if (recurring.completion.iteration.kind == SyncIterationRelationKind::LoopCarried) {
                    recurring.completion.source = 0;
                }
            }
        }
        const bool rejected = failed(verifyMixedProtocolPlan(*schedule, *stages, timelines, channels, broken));
        if (!check(rejected, "malformed logical endpoint")) {
            return false;
        }
    }
    return succeeded(materializeSelectiveLoopRepair(schedule->getFunction(), *plan->selectiveLoop));
}

bool mutations(ModuleOp module)
{
    SmallVector<Operation*, 8> boundaryActions;
    module.walk([&](Operation* operation) {
        const bool boundary = isa<SetFlagOp, WaitFlagOp>(operation) && isa<scf::IfOp>(operation->getParentOp());
        if (boundary) {
            boundaryActions.push_back(operation);
        }
    });
    for (Operation* action : boundaryActions) {
        Operation* following = action->getNextNode();
        Block detached;
        action->moveBefore(&detached, detached.end());
        const bool rejected = failed(verifyFreshConcreteSyncSemantics(*module.getOps<func::FuncOp>().begin()));
        action->moveBefore(following);
        if (!check(rejected, "missing boundary action")) {
            return false;
        }
    }
    SmallVector<arith::CmpIOp, 8> guards;
    module.walk([&](scf::IfOp choice) { guards.push_back(choice.getCondition().getDefiningOp<arith::CmpIOp>()); });
    for (auto comparison : guards) {
        const auto saved = comparison.getPredicate();
        comparison.setPredicate(arith::CmpIPredicate::ne);
        auto broken = extract(module);
        const bool rejected = broken && failed(reconstructSelectiveLoop(*broken)) &&
                              failed(verifyFreshConcreteSyncSemantics(broken->getFunction()));
        comparison.setPredicate(saved);
        if (!check(rejected, "changed boundary participation")) {
            return false;
        }
    }
    // Boundary and recurring channels in the same directed domain must not
    // share a concrete key, even though each recipe is valid in isolation.
    SetFlagOp bodySignal;
    SetFlagOp exitSignal;
    module.walk([&](SetFlagOp set) {
        const bool release =
            set.getSrcPipe().getPipe() == PIPE::PIPE_MTE3 && set.getDstPipe().getPipe() == PIPE::PIPE_V;
        if (!release) {
            return;
        }
        if (isa<scf::IfOp>(set->getParentOp())) {
            exitSignal = set;
        } else if (isa<scf::ForOp>(set->getParentOp())) {
            bodySignal = set;
        }
    });
    if (exitSignal && bodySignal) {
        const auto saved = exitSignal.getEventId();
        WaitFlagOp exitWait;
        module.walk([&](WaitFlagOp wait) {
            const bool matching = wait.getSrcPipe() == exitSignal.getSrcPipe() &&
                                  wait.getDstPipe() == exitSignal.getDstPipe() && wait.getEventId() == saved;
            if (matching) {
                exitWait = wait;
            }
        });
        if (!exitWait) {
            return false;
        }
        exitSignal.setEventIdAttr(bodySignal.getEventId());
        exitWait.setEventIdAttr(bodySignal.getEventId());
        const bool rejected = failed(verifyFreshConcreteSyncSemantics(*module.getOps<func::FuncOp>().begin()));
        exitSignal.setEventIdAttr(saved);
        exitWait.setEventIdAttr(saved);
        if (!check(rejected, "boundary key collides with a live recurring key")) {
            return false;
        }
    }
    auto function = *module.getOps<func::FuncOp>().begin();
    auto loop = *function.getOps<scf::ForOp>().begin();
    auto prime = *function.getOps<SetFlagOp>().begin();
    Operation* next = prime->getNextNode();
    prime->moveBefore(loop);
    auto movedPrime = extract(module);
    const bool rejectsPrime = movedPrime && failed(reconstructSelectiveLoop(*movedPrime)) &&
                              failed(verifyFreshConcreteSyncSemantics(function));
    prime->moveBefore(next);
    // Last top-level wait is recurring cleanup; moving it to the loop exit
    // would delay unrelated suffix work, even if memory remains race-free.
    WaitFlagOp drain;
    for (auto wait : function.getOps<WaitFlagOp>()) {
        drain = wait;
    }
    Operation* following = drain->getNextNode();
    drain->moveAfter(loop);
    auto movedDrain = extract(module);
    const bool rejectsDrain = movedDrain && failed(reconstructSelectiveLoop(*movedDrain)) &&
                              failed(verifyFreshConcreteSyncSemantics(function));
    drain->moveBefore(following);
    return check(rejectsPrime && rejectsDrain, "nonempty prefix publication or premature cleanup mutation");
}
} // namespace

bool testSelectiveLoopBoundaries(MLIRContext& context)
{
    for (unsigned variant = 0; variant < 16; ++variant) {
        const bool reuse = (variant & 1U) != 0;
        const bool mayAlias = (variant & 2U) != 0;
        const bool a2 = (variant & 4U) != 0;
        const bool nonunit = (variant & 8U) != 0;
        std::string source = kFixture.str();
        if (mayAlias) {
            source.replace(
                source.find("assume-disjoint-arguments"), std::string("assume-disjoint-arguments").size(), "may-alias");
        }
        if (a2) {
            source.replace(source.find("\"a3\""), 4, "\"a2\"");
        }
        if (nonunit) {
            source.replace(
                source.find("arith.constant 0 : index"), std::string("arith.constant 0 : index").size(),
                "arith.constant -5 : index");
            source.replace(
                source.find("arith.constant 1 : index"), std::string("arith.constant 1 : index").size(),
                "arith.constant 3 : index");
        }
        if (reuse) {
            source.insert(
                source.find("  return"), "  pto.tabs ins(%b : !pto.tile_buf<vec, 16x16xf16>) outs(%c : "
                                         "!pto.tile_buf<vec, 16x16xf16>)\n");
        }
        auto module = parseSourceString<ModuleOp>(source, &context);
        if (!module || !prepare(*module)) {
            return false;
        }
        auto concrete = extract(*module);
        const bool verified = concrete && succeeded(reconstructSelectiveLoop(*concrete)) &&
                              succeeded(verifyFreshConcreteSyncSemantics(concrete->getFunction()));
        if (!check(verified, "concrete boundary memory and event verification")) {
            return false;
        }
        for (unsigned trips : {0U, 1U, 2U, 3U, 4U, 7U}) {
            bool overlap = false;
            const bool safe = checkSelectiveBoundaryInterleavings(*concrete, trips, reuse ? nullptr : &overlap);
            if (!check(safe && (reuse || trips != 0 || overlap), "boundary safety and zero-trip independent suffix")) {
                return false;
            }
        }
        if (!mutations(*module)) {
            return false;
        }
    }
    llvm::outs() << "protocol-sync selective boundaries: entry/exit/bypass, guards and zero-trip overlap pass\n";
    return true;
}
