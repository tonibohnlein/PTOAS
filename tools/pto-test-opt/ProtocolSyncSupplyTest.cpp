// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- ProtocolSyncSupplyTest.cpp - Exhaustive logical supply differential --===//
// Independent Floyd closure over every five-node forward graph. No hardware
// claims: these are logical selected effects, including deliberately incomplete
// worlds. The concrete instruction scoreboard remains a separate test oracle.

#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/CompletionSupply.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {
constexpr StringLiteral kProgram = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @supply(%in: !pto.partition_tensor_view<16x16xf16>,
                    %out: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %a = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    pto.tload ins(%in : !pto.partition_tensor_view<16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)
    pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)
    pto.tstore ins(%a : !pto.tile_buf<vec, 16x16xf16>) outs(%out : !pto.partition_tensor_view<16x16xf16>)
    pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)
    pto.tstore ins(%a : !pto.tile_buf<vec, 16x16xf16>) outs(%out : !pto.partition_tensor_view<16x16xf16>)
    return
  }
}
)mlir";

bool witnessMatches(
    const SyncCompletionSupply& supply, const SyncSelectedWorld& world, unsigned source, unsigned target, bool reaches)
{
    const auto path = supply.witness(source, target);
    if (path.empty()) {
        return !reaches;
    }
    unsigned current = source;
    for (unsigned index : path) {
        const bool invalidEdge = index >= world.completions.size() || world.completions[index].source != current;
        if (invalidEdge) {
            return false;
        }
        current = world.completions[index].target;
    }
    return reaches && current == target;
}

bool graphCase(const StructuredSyncIR& schedule, unsigned mask)
{
    constexpr unsigned count = 5;
    bool closure[count][count] = {};
    SyncSelectedWorld world;
    unsigned bit = 0;
    for (unsigned source = 0; source < count; ++source) {
        for (unsigned target = source + 1; target < count; ++target) {
            if (mask & (1u << bit)) {
                closure[source][target] = true;
                world.completions.push_back({source, target});
            }
            ++bit;
        }
    }
    // Input ordering must not affect propagation or witness validity.
    std::reverse(world.completions.begin(), world.completions.end());
    for (unsigned middle = 0; middle < count; ++middle) {
        for (unsigned source = 0; source < count; ++source) {
            for (unsigned target = 0; target < count; ++target) {
                closure[source][target] |= closure[source][middle] && closure[middle][target];
            }
        }
    }
    auto supply = buildCompletionSupply(schedule, world);
    const bool complete = succeeded(supply) && supply->getStatus() == SyncCompletionSupplyStatus::Complete;
    if (!complete) {
        return false;
    }
    for (unsigned target = 0; target < count; ++target) {
        const auto* frontier = supply->getBefore(target);
        const auto* phase = schedule.findPhase(target);
        if (!frontier || !phase || frontier->before != phase->before || frontier->lane != phase->pipe ||
            frontier->region != phase->region || frontier->core != phase->core) {
            return false;
        }
        for (unsigned source = 0; source < count; ++source) {
            const bool mismatch = supply->covers(source, target, {SyncIterationRelationKind::SameIteration, 0}) !=
                                      closure[source][target] ||
                                  !witnessMatches(*supply, world, source, target, closure[source][target]);
            if (mismatch) {
                return false;
            }
        }
    }
    return true;
}

bool limitsAndEffects(const StructuredSyncIR& schedule)
{
    SyncSelectedWorld world;
    world.visibility.push_back({0, 4});
    world.exitCompletedPhases.push_back(0);
    auto empty = buildCompletionSupply(schedule, world);
    const bool invalidEmpty = failed(empty) || empty->covers(0, 4, {SyncIterationRelationKind::SameIteration, 0}) ||
                              empty->getBefore(kInvalidSyncId) || !empty->witness(kInvalidSyncId, 4).empty();
    if (invalidEmpty) {
        return false; // Visibility and exit supply cannot become body completion.
    }
    world.completions.push_back({0, 4});
    auto limited = buildCompletionSupply(schedule, world, 24);
    auto exactBudget = buildCompletionSupply(schedule, world, 25);
    const bool invalidLimit = failed(limited) || limited->getStatus() != SyncCompletionSupplyStatus::LimitExceeded ||
                              limited->getBefore(4) || failed(exactBudget) ||
                              !exactBudget->covers(0, 4, {SyncIterationRelationKind::SameIteration, 0}) ||
                              exactBudget->covers(0, 4, {SyncIterationRelationKind::LoopCarried, 1, 0}) ||
                              exactBudget->covers(0, 4, {SyncIterationRelationKind::SameIteration, 0, 0});
    if (invalidLimit) {
        return false;
    }
    const SyncSelectedCompletion invalid[] = {
        {4, 0},
        {0, 0},
        {0, kInvalidSyncId},
        {0, 4, SyncControlRelation::Unknown},
        {0, 4, SyncControlRelation::SameGuard},
        {0, 4, SyncControlRelation::MustExecute, {SyncIterationRelationKind::SameIteration, 1}},
        {0, 4, SyncControlRelation::MustExecute, {SyncIterationRelationKind::LoopCarried, 1, 0}},
    };
    for (const auto& edge : invalid) {
        world.completions.assign(1, edge);
        if (succeeded(buildCompletionSupply(schedule, world))) {
            return false;
        }
    }
    world.completions.assign(2, SyncSelectedCompletion{0, 4});
    if (succeeded(buildCompletionSupply(schedule, world))) {
        return false;
    }
    world.orderedLoop = 0;
    auto unsupported = buildCompletionSupply(schedule, world);
    return succeeded(unsupported) && unsupported->getStatus() == SyncCompletionSupplyStatus::Unsupported &&
           !unsupported->getBefore(0);
}
} // namespace

bool runCompletionSupplyTests(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(kProgram, &context);
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
    const bool validSchedule =
        succeeded(StructuredSyncIRBuilder(semantics).build(function, schedule)) && schedule.getPhases().size() == 5;
    if (!validSchedule) {
        return false;
    }
    for (unsigned mask = 0; mask < 1024; ++mask) {
        if (!graphCase(schedule, mask)) {
            llvm::errs() << "FAIL: completion supply graph=" << mask << '\n';
            return false;
        }
    }
    if (!limitsAndEffects(schedule)) {
        llvm::errs() << "FAIL: completion supply bounds/effect separation\n";
        return false;
    }
    llvm::outs() << "protocol-sync completion supply: 1024 graphs, witnesses and negative controls pass\n";
    return true;
}
