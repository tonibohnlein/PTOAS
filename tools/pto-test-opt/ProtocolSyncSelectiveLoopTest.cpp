// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/SelectiveLoopRepair.h"
#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"
#include "PTO/Transforms/ProtocolSync/MixedProtocolPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include <iterator>
#include <memory>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

bool checkSelectiveLoopInterleavings(const StructuredSyncIR&, unsigned, bool*);

namespace {
constexpr StringLiteral kFixture = R"mlir(
module attributes {pto.target_arch = "a3"} {
 func.func @selective(%n: index, %x: !pto.partition_tensor_view<16x16xf16>,
                     %y: !pto.partition_tensor_view<16x16xf16>, %z: !pto.partition_tensor_view<16x16xf16>)
 attributes {pto.kernel_kind = #pto.kernel_kind<vector>, pto.gm_alias = "assume-disjoint-arguments"} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %base = arith.constant 0 : i64
  %other = arith.constant 512 : i64
  %a = pto.alloc_tile addr = %base : !pto.tile_buf<vec, 16x16xf16>
  %b = pto.alloc_tile addr = %other : !pto.tile_buf<vec, 16x16xf16>
  scf.for %i = %zero to %n step %one {
   pto.tload ins(%x : !pto.partition_tensor_view<16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)
   pto.tload ins(%y : !pto.partition_tensor_view<16x16xf16>) outs(%b : !pto.tile_buf<vec, 16x16xf16>)
   pto.tadd ins(%a, %b : !pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 16x16xf16>)
            outs(%a : !pto.tile_buf<vec, 16x16xf16>)
   pto.tstore ins(%a : !pto.tile_buf<vec, 16x16xf16>) outs(%z : !pto.partition_tensor_view<16x16xf16>)
  }
  return
 }
})mlir";

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

bool check(bool result, StringRef detail)
{
    if (!result) {
        llvm::errs() << "FAIL selective loop: " << detail << '\n';
    }
    return result;
}

bool allocatePlan(const StructuredSyncIR& schedule, SyncSelectiveLoopPlan& plan)
{
    SmallVector<SyncEventGeneration, 16> generations;
    if (failed(appendSelectiveLoopEventGenerations(schedule, plan, generations))) {
        return false;
    }
    auto allocation =
        allocateSyncEventGenerations(ProtocolSyncTarget::resolve(schedule.getFunction()), {}, generations);
    const bool allocated = succeeded(allocation) && allocation->status == SyncEventAllocationStatus::Allocated;
    if (!allocated) {
        return false;
    }
    unsigned index = 0;
    for (auto& edge : plan.edges) {
        if (edge.isEvent()) {
            edge.placement.eventId = allocation->eventIds[index++];
        }
    }
    return true;
}

bool materializePlan(const StructuredSyncIR& schedule, SyncSelectiveLoopPlan& plan)
{
    return allocatePlan(schedule, plan) && succeeded(materializeSelectiveLoopRepair(schedule.getFunction(), plan));
}

bool testIndependentLoopReadiness(MLIRContext& context)
{
    std::string text = kFixture.str();
    const auto start = text.find("   pto.tadd");
    const auto end = text.find("  }", start);
    text.replace(
        start, end - start,
        "   pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)\n"
        "   pto.tabs ins(%b : !pto.tile_buf<vec, 16x16xf16>) outs(%b : !pto.tile_buf<vec, 16x16xf16>)\n");
    auto module = parseSourceString<ModuleOp>(text, &context);
    auto schedule = module ? extract(*module) : nullptr;
    if (!schedule) {
        return false;
    }
    auto plan = buildSelectiveLoopRepair(*schedule);
    const bool planned = succeeded(plan) && plan->plan;
    if (!check(planned, "independent loop plan")) {
        return false;
    }
    if (!materializePlan(*schedule, *plan->plan)) {
        return false;
    }
    auto concrete = extract(*module);
    if (!check(concrete && succeeded(reconstructSelectiveLoop(*concrete)), "independent loop concrete supply")) {
        return false;
    }
    for (unsigned trips : {1U, 2U, 3U, 4U}) {
        bool overlap = false;
        const bool safe = checkSelectiveLoopInterleavings(*concrete, trips, &overlap);
        if (!check(safe && overlap, "consumer A overlaps independent load B")) {
            return false;
        }
    }
    return true;
}

bool testSinglePhaseBarrier(MLIRContext& context)
{
    std::string text = kFixture.str();
    const auto first = text.find("   pto.tload");
    const auto second = text.find("   pto.tload", first + 1);
    text.erase(second, text.find("  }", second) - second);
    auto module = parseSourceString<ModuleOp>(text, &context);
    auto schedule = module ? extract(*module) : nullptr;
    if (!schedule) {
        return false;
    }
    auto plan = buildSelectiveLoopRepair(*schedule);
    const bool planned = succeeded(plan) && plan->plan && plan->plan->edges.size() == 1 &&
                         !plan->plan->edges.front().isEvent() &&
                         plan->plan->edges.front().completion.iteration.distance == 1;
    if (!check(planned, "one-phase carried barrier")) {
        return false;
    }
    if (!materializePlan(*schedule, *plan->plan)) {
        return false;
    }
    auto concrete = extract(*module);
    if (!concrete || failed(reconstructSelectiveLoop(*concrete))) {
        return false;
    }
    auto loop = *concrete->getFunction().getOps<scf::ForOp>().begin();
    auto barrier = *loop.getBody()->getOps<BarrierOp>().begin();
    barrier.erase();
    auto removed = extract(*module);
    return check(removed && failed(reconstructSelectiveLoop(*removed)), "deleted self-recurrence barrier");
}

bool testPlanMutations(const StructuredSyncIR& schedule)
{
    auto stages = analyzePipelineStages(schedule);
    if (failed(stages)) {
        return false;
    }
    auto timelines = analyzeStorageTimelines(schedule, *stages);
    auto channels = analyzeChannels(schedule, *stages, timelines);
    auto plan = buildMixedSelectiveLoopPlan(schedule, *stages, timelines, channels);
    const bool ready = succeeded(plan) && *plan;
    if (!check(ready, "complete native loop world")) {
        return false;
    }
    auto& selected = **plan;
    ++selected.selectedCost.generatedEventPairs;
    const bool cost = failed(verifyMixedSelectiveLoopPlan(schedule, *stages, timelines, channels, selected));
    --selected.selectedCost.generatedEventPairs;
    selected.allocationFailure = SyncEventAllocationFailure::AnalysisLimit;
    const bool failureKind = failed(verifyMixedSelectiveLoopPlan(schedule, *stages, timelines, channels, selected));
    selected.allocationFailure = SyncEventAllocationFailure::None;
    const bool checked = check(cost && failureKind, "forged native cost/allocation result") &&
                         succeeded(allocateMixedProtocolEvents(schedule, selected));
    if (!checked) {
        return false;
    }
    if (!check(
            succeeded(verifyMixedProtocolPlan(schedule, *stages, timelines, channels, selected)),
            "allocated loop uses shared verification")) {
        return false;
    }
    auto& edges = selected.selectiveLoop->edges;
    const auto saved = edges.front().placement.eventId;
    edges.front().placement.eventId.reset();
    const bool partial = failed(verifyMixedProtocolPlan(schedule, *stages, timelines, channels, selected));
    edges.front().placement.eventId = saved;
    const auto second = edges[1].placement.eventId;
    edges[1].placement.eventId = saved;
    const bool collision = failed(verifyMixedProtocolPlan(schedule, *stages, timelines, channels, selected));
    edges[1].placement.eventId = second;
    return check(partial && collision, "partial and colliding recurring assignments");
}

bool checkMixedRejection(const StructuredSyncIR& schedule, SyncMixedPlanRejection reason)
{
    auto stages = analyzePipelineStages(schedule);
    if (failed(stages)) {
        return false;
    }
    auto timelines = analyzeStorageTimelines(schedule, *stages);
    auto channels = analyzeChannels(schedule, *stages, timelines);
    auto plan = buildMixedProtocolPlan(schedule, *stages, timelines, channels, false);
    const bool rejected = succeeded(plan) && !plan->isComplete() &&
                          llvm::any_of(plan->failures, [&](const auto& item) { return item.reason == reason; });
    return check(
        rejected && succeeded(verifyMixedProtocolPlan(schedule, *stages, timelines, channels, *plan)),
        "native refusal remains a verified refusal, not an internal error");
}

bool testResourceRefusal(MLIRContext& context)
{
    std::string text = kFixture.str();
    std::string producers;
    std::string consumers;
    for (unsigned id = 0; id < 7; ++id) {
        const auto suffix = std::to_string(id);
        producers += "   %addr" + suffix + " = arith.constant " + std::to_string(512 * id) + " : i64\n";
        producers +=
            "   %tile" + suffix + " = pto.alloc_tile addr = %addr" + suffix + " : !pto.tile_buf<vec, 16x16xf16>\n";
        producers += "   pto.tload ins(%x : !pto.partition_tensor_view<16x16xf16>) outs(%tile" + suffix +
                     " : !pto.tile_buf<vec, 16x16xf16>)\n";
        consumers += "   pto.tabs ins(%tile" + suffix + " : !pto.tile_buf<vec, 16x16xf16>) outs(%tile" + suffix +
                     " : !pto.tile_buf<vec, 16x16xf16>)\n";
    }
    const auto start = text.find("   pto.tload");
    text.replace(start, text.find("  }", start) - start, producers + consumers);
    auto module = parseSourceString<ModuleOp>(text, &context);
    auto schedule = module ? extract(*module) : nullptr;
    if (!schedule) {
        return false;
    }
    auto logical = buildSelectiveLoopRepair(*schedule);
    return check(succeeded(logical) && logical->plan.has_value(), "logical plan survives seven-channel domain") &&
           checkMixedRejection(*schedule, SyncMixedPlanRejection::EventCapacity);
}

bool testConcreteBudget(MLIRContext& context)
{
    std::string text = kFixture.str();
    std::string body;
    for (unsigned id = 0; id < 65; ++id) {
        const auto suffix = std::to_string(id);
        body += "   %addr" + suffix + " = arith.constant " + std::to_string(512 * id) + " : i64\n";
        body += "   %tile" + suffix + " = pto.alloc_tile addr = %addr" + suffix + " : !pto.tile_buf<vec, 16x16xf16>\n";
        body += "   pto.tload ins(%x : !pto.partition_tensor_view<16x16xf16>) outs(%tile" + suffix +
                " : !pto.tile_buf<vec, 16x16xf16>)\n";
    }
    const auto start = text.find("   pto.tload");
    text.replace(start, text.find("  }", start) - start, body);
    auto module = parseSourceString<ModuleOp>(text, &context);
    auto schedule = module ? extract(*module) : nullptr;
    if (!schedule) {
        return false;
    }
    auto plan = buildSelectiveLoopRepair(*schedule);
    const bool unsupported = succeeded(plan) && !plan->plan && plan->status == SyncSelectiveLoopStatus::AnalysisLimit;
    return check(unsupported, "concrete budget rejected before materialization") &&
           checkMixedRejection(*schedule, SyncMixedPlanRejection::SelectiveAnalysisLimit);
}
} // namespace

bool testSelectiveLoopRepair(MLIRContext& context)
{
    const bool initial = testIndependentLoopReadiness(context) && testSinglePhaseBarrier(context) &&
                         testConcreteBudget(context) && testResourceRefusal(context);
    if (!initial) {
        return false;
    }
    auto module = parseSourceString<ModuleOp>(kFixture, &context);
    auto schedule = module ? extract(*module) : nullptr;
    if (!schedule) {
        return false;
    }
    if (!testPlanMutations(*schedule)) {
        return false;
    }
    auto plan = buildSelectiveLoopRepair(*schedule);
    const bool planned = succeeded(plan) && plan->plan;
    if (!check(planned, "hazard-derived plan")) {
        return false;
    }
    const bool logical = llvm::none_of(plan->plan->edges, [](const auto& edge) { return edge.placement.eventId; });
    if (!check(
            logical && plan->plan->candidateCountBeforeDeletion == 6 && plan->plan->edges.size() == 5 &&
                plan->plan->deletionRemoved == 1,
            "six-to-five complete channel deletion before allocation")) {
        return false;
    }
    auto broken = plan->plan->world;
    llvm::erase_if(broken.completions, [](const auto& edge) { return edge.iteration.distance == 1; });
    if (!check(failed(verifySelectiveLoopMemory(*schedule, broken)), "missing carried memory coverage")) {
        return false;
    }
    if (!check(
            allocatePlan(*schedule, *plan->plan) &&
                succeeded(materializeSelectiveLoopRepair(schedule->getFunction(), *plan->plan)),
            "materialization")) {
        return false;
    }
    auto concrete = extract(*module);
    if (!check(concrete && succeeded(reconstructSelectiveLoop(*concrete)), "concrete reconstruction")) {
        return false;
    }
    for (unsigned trips : {0U, 1U, 2U, 3U, 4U, 7U, 11U}) {
        const bool safe = checkSelectiveLoopInterleavings(*concrete, trips, nullptr);
        if (!check(safe, "independent memory/token execution")) {
            return false;
        }
    }
    // Shift disjoint storage into overlap without changing synchronization.
    auto function = concrete->getFunction();
    auto allocations = function.getOps<AllocTileOp>();
    auto second = *std::next(allocations.begin());
    auto address = second.getAddr().getDefiningOp<arith::ConstantOp>();
    const auto saved = address.getValue();
    address.setValueAttr(IntegerAttr::get(saved.getType(), 256));
    auto shifted = extract(*module);
    const bool detectsOverlap = shifted && failed(reconstructSelectiveLoop(*shifted));
    address.setValueAttr(saved);
    if (!check(detectsOverlap, "shifted physical overlap")) {
        return false;
    }
    auto prime = *function.getOps<SetFlagOp>().begin();
    prime.erase();
    auto missing = extract(*module);
    if (!check(missing && failed(reconstructSelectiveLoop(*missing)), "deleted prime")) {
        return false;
    }
    llvm::outs() << "protocol-sync selective loop: native recipe, occurrence coverage and overlap oracle pass\n";
    return true;
}
