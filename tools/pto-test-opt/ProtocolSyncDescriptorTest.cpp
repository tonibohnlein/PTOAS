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
#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace mlir::pto::protocol_sync {
class StructuredSyncIRTestPeer {
public:
    static auto& actions(StructuredSyncIR& schedule) { return schedule.semanticActions; }
    static auto& accesses(StructuredSyncIR& schedule) { return schedule.accesses; }
    static auto& states(StructuredSyncIR& schedule) { return schedule.descriptorStates; }
};
} // namespace mlir::pto::protocol_sync

namespace {

constexpr StringLiteral kType = R"mlir(
!tile = !pto.tile_buf<loc=vec, dtype=f16, rows=4, cols=16, v_row=?, v_col=?,
                     blayout=row_major, slayout=none_box, fractal=512, pad=0>
)mlir";

std::string program(unsigned initial, unsigned update, StringRef extra = {})
{
    return kType.str() + R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @descriptor(%input: !pto.partition_tensor_view<4x16xf16>, %dynamic: index, %cond: i1,
                        %scalar: !pto.ptr<i32, gm>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %base = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %initial = arith.constant )mlir" +
           std::to_string(initial) + R"mlir( : index
    %update = arith.constant )mlir" +
           std::to_string(update) + R"mlir( : index
    %a = pto.alloc_tile addr = %base valid_row = %initial valid_col = %c16 : !tile
    %b = pto.alloc_tile addr = %base valid_row = %c1 valid_col = %c16 : !tile
    pto.tload ins(%input : !pto.partition_tensor_view<4x16xf16>) outs(%a : !tile)
    %r0, %s0 = pto.get_validshape %a : !tile
    pto.set_validshape %a, %update, %c16 : !tile
    pto.tabs ins(%a : !tile) outs(%a : !tile)
    %r1, %s1 = pto.get_validshape %a : !tile
    pto.set_validshape %a, %c1, %c16 : !tile
    pto.tabs ins(%a : !tile) outs(%a : !tile)
    %r2, %s2 = pto.get_validshape %a : !tile
    %rb, %sb = pto.get_validshape %b : !tile
)mlir" + extra.str() +
           "\npto.barrier <PIPE_ALL>\nreturn } }";
}

bool check(bool condition, StringRef detail)
{
    if (!condition) {
        llvm::errs() << "FAIL descriptor: " << detail << '\n';
    }
    return condition;
}

std::unique_ptr<StructuredSyncIR> extract(ModuleOp module)
{
    auto function = *module.getOps<func::FuncOp>().begin();
    LegacySyncSnapshot snapshot;
    LegacySyncIRAdapter adapter;
    if (failed(adapter.buildSnapshot(function, snapshot))) {
        return {};
    }
    auto context = adapter.buildSemanticContext(snapshot);
    auto schedule = std::make_unique<StructuredSyncIR>(function);
    if (failed(StructuredSyncIRBuilder(context).build(function, *schedule))) {
        return {};
    }
    return schedule;
}

std::uint64_t literal(Value value)
{
    return cast<IntegerAttr>(value.getDefiningOp<arith::ConstantOp>().getValue()).getValue().getZExtValue();
}

bool compareLexicalOracle(const StructuredSyncIR& schedule)
{
    // Independent test interpreter: raw operations, per-handle integer state,
    // no production transfer function, version IDs, or footprint helpers.
    llvm::DenseMap<Value, std::pair<std::uint64_t, std::uint64_t>> dimensions;
    for (Operation& operation : schedule.getFunction().getBody().front()) {
        if (auto allocation = dyn_cast<AllocTileOp>(operation)) {
            dimensions[allocation.getResult()] = {literal(allocation.getValidRow()), literal(allocation.getValidCol())};
        } else if (auto update = dyn_cast<SetValidShapeOp>(operation)) {
            dimensions[update.getSource()] = {literal(update.getValidRow()), literal(update.getValidCol())};
        }
        for (const SyncSemanticAction& action : schedule.getSemanticActions()) {
            if (action.operation != &operation || !action.descriptorState) {
                continue;
            }
            const auto& state = schedule.getDescriptorStates()[*action.descriptorState];
            const bool mismatch = !dimensions.contains(state.handle) ||
                                  dimensions.lookup(state.handle) != std::make_pair(state.rows, state.columns);
            if (mismatch) {
                return false;
            }
        }
        for (const SyncAccess& access : schedule.getAccesses()) {
            if (schedule.findPhase(access.phase)->operation != &operation ||
                access.storage.space != AddressSpace::VEC) {
                continue;
            }
            if (!access.descriptorState) {
                return false;
            }
            const auto& state = schedule.getDescriptorStates()[*access.descriptorState];
            const auto region = recoverLocalAccessRegion(access);
            const bool mismatch = !dimensions.contains(access.value) ||
                                  dimensions.lookup(access.value) != std::make_pair(state.rows, state.columns) ||
                                  region.interval.size != 128 || region.precision != SyncRegionPrecision::Conservative;
            if (mismatch) {
                return false;
            }
        }
    }
    return true;
}

bool bindingMutations(StructuredSyncIR& schedule)
{
    for (SyncAccess& access : StructuredSyncIRTestPeer::accesses(schedule)) {
        if (!access.descriptorState) {
            continue;
        }
        const auto saved = access.descriptorState;
        access.descriptorState = 1; // b, same physical address, different handle.
        const bool rejectsWrongHandle = failed(verifySyncDescriptorBindings(schedule));
        access.descriptorState.reset();
        const bool rejectsMissing = failed(verifySyncDescriptorBindings(schedule));
        access.descriptorState = 3;
        const bool rejectsStale = saved == access.descriptorState || failed(verifySyncDescriptorBindings(schedule));
        access.descriptorState = saved;
        if (!rejectsWrongHandle || !rejectsMissing || !rejectsStale) {
            return false;
        }
    }
    for (SyncSemanticAction& action : StructuredSyncIRTestPeer::actions(schedule)) {
        const auto saved = action.descriptorState;
        const auto& effect = schedule.getSummaries()[action.summary].descriptor;
        if (effect && effect->role != SyncDescriptorRole::Read) {
            continue;
        }
        action.descriptorState = effect ? std::optional<std::uint32_t>(3) : std::optional<std::uint32_t>(0);
        // Skip a read that genuinely observes version 3.
        const bool rejects = action.descriptorState == saved || failed(verifySyncDescriptorBindings(schedule));
        action.descriptorState = saved;
        if (!rejects) {
            return false;
        }
    }
    auto& state = StructuredSyncIRTestPeer::states(schedule).front();
    const auto rows = state.rows;
    state.rows = 5;
    const bool rejectsBounds = failed(verifySyncDescriptorBindings(schedule));
    state.rows = rows;
    return rejectsBounds && succeeded(verifySyncDescriptorBindings(schedule));
}

bool testVersions(MLIRContext& context)
{
    for (unsigned initial = 0; initial <= 4; ++initial) {
        for (unsigned update = 0; update <= 4; ++update) {
            auto module = parseSourceString<ModuleOp>(program(initial, update), &context);
            auto schedule = module ? extract(*module) : nullptr;
            const bool complete =
                schedule && schedule->getFailures().empty() && schedule->getDescriptorStates().size() == 4;
            const bool verified = complete && compareLexicalOracle(*schedule) && bindingMutations(*schedule);
            if (!check(verified, "version oracle")) {
                return false;
            }
            auto function = schedule->getFunction();
            auto read = *function.getOps<GetValidShapeOp>().begin();
            auto updateOp = *function.getOps<SetValidShapeOp>().begin();
            read->moveAfter(updateOp);
            auto moved = extract(*module);
            if (!check(
                    moved && moved->getFailures().empty() && compareLexicalOracle(*moved),
                    "moved read observes new state")) {
                return false;
            }
        }
    }
    llvm::outs() << "protocol-sync descriptor versions: 25 programs and binding mutations pass\n";
    return true;
}

bool testRejected(MLIRContext& context)
{
    const char* additions[] = {
        "pto.set_validshape %a, %dynamic, %c16 : !tile",
        "pto.set_validshape %a, %r0, %s0 : !tile",
        "%loaded = pto.load_scalar %scalar[%c0] : !pto.ptr<i32, gm> -> i32\n"
        "%bound = arith.index_cast %loaded : i32 to index\npto.set_validshape %a, %bound, %c16 : !tile",
        "scf.if %cond { pto.set_validshape %a, %c1, %c16 : !tile }",
        "%alias = pto.subview %a[%c0, %c0] sizes [4, 16] : !tile -> !tile",
        "%bad = arith.constant 5 : index\npto.set_validshape %a, %bad, %c16 : !tile",
        "%bad = arith.constant -1 : index\npto.set_validshape %a, %bad, %c16 : !tile",
    };
    for (const char* extra : additions) {
        auto module = parseSourceString<ModuleOp>(program(4, 2, extra), ParserConfig(&context, false));
        auto schedule = module ? extract(*module) : nullptr;
        const bool rejected = schedule && llvm::any_of(schedule->getFailures(), [](const SyncFailure& failure) {
                                  return failure.reason == SyncFailureReason::UnsupportedDescriptorState;
                              });
        if (!check(rejected, "unsupported descriptor must retain semantic failure")) {
            return false;
        }
    }
    std::string unknownInitialization = program(4, 2);
    const std::string original = "valid_row = %initial";
    unknownInitialization.replace(unknownInitialization.find(original), original.size(), "valid_row = %dynamic");
    auto module = parseSourceString<ModuleOp>(unknownInitialization, &context);
    auto schedule = module ? extract(*module) : nullptr;
    if (!check(
            schedule && !schedule->getFailures().empty(),
            "unresolved initialization cannot be hidden by later update")) {
        return false;
    }
    llvm::outs() << "protocol-sync unsupported descriptor state: pass\n";
    return true;
}

} // namespace

bool testProtocolSyncDescriptorState(MLIRContext& context) { return testVersions(context) && testRejected(context); }
