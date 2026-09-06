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
#include "mlir/Dialect/SCF/IR/SCF.h"
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
            const bool mismatch = !dimensions.contains(state.handle) || !state.rows || !state.columns ||
                                  dimensions.lookup(state.handle) != std::make_pair(*state.rows, *state.columns);
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
            const bool mismatch = !dimensions.contains(access.value) || !state.rows || !state.columns ||
                                  dimensions.lookup(access.value) != std::make_pair(*state.rows, *state.columns) ||
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
    const Value handle = state.handle;
    state.handle = {};
    const bool rejectsNull = failed(verifySyncDescriptorBindings(schedule));
    state.handle = handle;
    const auto rows = state.rows;
    state.rows = 5;
    const bool rejectsBounds = failed(verifySyncDescriptorBindings(schedule));
    state.rows = rows;
    return rejectsNull && rejectsBounds && succeeded(verifySyncDescriptorBindings(schedule));
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
        "pto.set_validshape %a, %r0, %s0 : !tile",
        "%loaded = pto.load_scalar %scalar[%c0] : !pto.ptr<i32, gm> -> i32\n"
        "%bound = arith.index_cast %loaded : i32 to index\npto.set_validshape %a, %bound, %c16 : !tile",
        "scf.if %cond { pto.set_validshape %a, %c1, %c16 : !tile }",
        "scf.for %i = %c0 to %c4 step %c1 { pto.set_validshape %a, %c1, %c16 : !tile }",
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
    llvm::outs() << "protocol-sync unsupported descriptor state: pass\n";
    return true;
}

bool testSymbolicDimensions(MLIRContext& context)
{
    std::string source = program(
        4, 2,
        "%difference = arith.subi %dynamic, %c1 : index\n"
        "%bounded = arith.minsi %difference, %c4 : index\n"
        "pto.set_validshape %a, %bounded, %c16 : !tile\n"
        "pto.tabs ins(%a : !tile) outs(%a : !tile)");
    const std::string original = "valid_row = %initial";
    source.replace(source.find(original), original.size(), "valid_row = %dynamic");
    auto module = parseSourceString<ModuleOp>(source, &context);
    auto schedule = module ? extract(*module) : nullptr;
    if (!check(
            schedule && schedule->getFailures().empty() && succeeded(verifySyncDescriptorBindings(*schedule)),
            "symbolic descriptor provenance")) {
        return false;
    }
    unsigned symbolic = 0;
    for (auto& state : StructuredSyncIRTestPeer::states(*schedule)) {
        if (state.scalarProvenance != SyncDescriptorScalarProvenance::NonphysicalExpression) {
            continue;
        }
        ++symbolic;
        if (!check(
                !state.rows && state.columns == 16 && state.rowSource,
                "symbolic range remains unknown, including a possibly negative min")) {
            return false;
        }
        state.rows = 0;
        const bool fabricatedBound = failed(verifySyncDescriptorBindings(*schedule));
        state.rows.reset();
        state.scalarProvenance = SyncDescriptorScalarProvenance::Constant;
        const bool fabricatedProvenance = failed(verifySyncDescriptorBindings(*schedule));
        state.scalarProvenance = SyncDescriptorScalarProvenance::NonphysicalExpression;
        Value savedSource = state.rowSource;
        state.rowSource = state.columnSource;
        const bool changedSource = failed(verifySyncDescriptorBindings(*schedule));
        state.rowSource = savedSource;
        if (!check(fabricatedBound && fabricatedProvenance && changedSource, "symbolic binding mutations rejected")) {
            return false;
        }
    }
    for (const auto& access : schedule->getAccesses()) {
        if (access.storage.space != AddressSpace::VEC) {
            continue;
        }
        const auto region = recoverLocalAccessRegion(access);
        if (!check(
                region.interval.size == 128 && region.precision == SyncRegionPrecision::Conservative,
                "symbolic metadata never shrinks physical bounds")) {
            return false;
        }
    }
    auto setter = *schedule->getFunction().getOps<SetValidShapeOp>().begin();
    Value savedRow = setter.getValidRow();
    setter.getValidRowMutable().assign(setter.getValidCol());
    const bool liveSourceRejected = failed(verifySyncDescriptorBindings(*schedule));
    setter.getValidRowMutable().assign(savedRow);
    auto getter = *schedule->getFunction().getOps<GetValidShapeOp>().begin();
    Value savedHandle = getter.getSource();
    getter.getSourceMutable().assign(schedule->getDescriptorStates()[1].handle);
    const bool liveHandleRejected = failed(verifySyncDescriptorBindings(*schedule));
    getter.getSourceMutable().assign(savedHandle);
    return check(
        symbolic == 2 && liveSourceRejected && liveHandleRejected && succeeded(verifySyncDescriptorBindings(*schedule)),
        "symbolic initialization and update preserve exact SSA sources");
}

bool testScalarProvenanceLimits(MLIRContext& context)
{
    for (unsigned count : {4u, 257u}) {
        std::string extra;
        std::string previous = "%dynamic";
        for (unsigned i = 0; i < count; ++i) {
            const std::string name = "%chain" + std::to_string(i);
            extra += name + " = arith.minsi " + previous + ", %c4 : index\n";
            previous = name;
        }
        extra += "pto.set_validshape %a, " + previous + ", %c16 : !tile";
        auto module = parseSourceString<ModuleOp>(program(4, 2, extra), &context);
        auto schedule = module ? extract(*module) : nullptr;
        if (!check(
                schedule && schedule->getFailures().empty() == (count == 4), "bounded scalar provenance traversal")) {
            return false;
        }
    }
    auto module = parseSourceString<ModuleOp>(
        program(4, 2, R"mlir(
      %loaded = pto.load_scalar %scalar[%c0] : !pto.ptr<i32, gm> -> i32
      %bound = arith.index_cast %loaded : i32 to index
      scf.for %i = %c0 to %bound step %c1 {
        %local = pto.alloc_tile addr = %base valid_row = %i valid_col = %c16 : !tile
        %lr, %lc = pto.get_validshape %local : !tile
      }
    )mlir"),
        &context);
    auto schedule = module ? extract(*module) : nullptr;
    return check(
        schedule && llvm::any_of(
                        schedule->getFailures(),
                        [](const SyncFailure& failure) {
                            return failure.reason == SyncFailureReason::UnsupportedDescriptorState;
                        }),
        "induction variable cannot hide a physical scalar bound");
}

bool testNestedReads(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(
        program(4, 2, R"mlir(
      scf.for %i = %c0 to %dynamic step %c1 {
        scf.if %cond {
          %rn, %sn = pto.get_validshape %a : !tile
          pto.tabs ins(%a : !tile) outs(%a : !tile)
        }
      }
      pto.set_validshape %a, %c4, %c16 : !tile
      %rf, %sf = pto.get_validshape %a : !tile
    )mlir"),
        &context);
    auto schedule = module ? extract(*module) : nullptr;
    const bool valid =
        schedule && schedule->getFailures().empty() && succeeded(verifySyncDescriptorBindings(*schedule));
    if (!check(valid, "outer versions reach nested reads")) {
        return false;
    }
    auto function = schedule->getFunction();
    auto loop = *function.getOps<scf::ForOp>().begin();
    unsigned nestedAccesses = 0;
    for (const auto& access : schedule->getAccesses()) {
        if (!loop->isAncestor(schedule->findPhase(access.phase)->operation)) {
            continue;
        }
        const bool bound = access.descriptorState && schedule->getDescriptorStates()[*access.descriptorState].rows == 1;
        const auto region = recoverLocalAccessRegion(access);
        if (!check(
                bound && region.interval.size == 128 && region.precision == SyncRegionPrecision::Conservative,
                "nested payload keeps version and conservative allocation bounds")) {
            return false;
        }
        ++nestedAccesses;
    }
    bool sawNestedRead = false;
    for (const auto& action : schedule->getSemanticActions()) {
        const bool nestedRead = isa<GetValidShapeOp>(action.operation) && loop->isAncestor(action.operation);
        if (!nestedRead) {
            continue;
        }
        const auto& state = schedule->getDescriptorStates()[*action.descriptorState];
        // Every executing arm observes the last outer update (one row).
        // A zero-trip path performs no metadata transfer at all.
        if (!check(state.rows == 1 && state.columns == 16, "nested version is unchanged by participation")) {
            return false;
        }
        sawNestedRead = true;
    }
    auto update = *function.getOps<SetValidShapeOp>().begin();
    Operation* next = update->getNextNode();
    update->moveBefore(loop.getBody(), loop.getBody()->begin());
    const bool rejectsMovedUpdate = failed(verifySyncDescriptorBindings(*schedule));
    update->moveBefore(next);
    // Keep all cached points unchanged while moving the suffix update before
    // the loop. The raw dominance check must reject the stale nested binding.
    SetValidShapeOp suffix;
    for (auto candidate : function.getOps<SetValidShapeOp>()) {
        suffix = candidate;
    }
    Operation* suffixNext = suffix->getNextNode();
    suffix->moveBefore(loop);
    const bool rejectsChangedVersion = failed(verifySyncDescriptorBindings(*schedule));
    suffix->moveBefore(suffixNext);
    return check(
        nestedAccesses != 0 && sawNestedRead && rejectsMovedUpdate && rejectsChangedVersion &&
            succeeded(verifySyncDescriptorBindings(*schedule)),
        "live region and version mutations");
}

bool testDescriptorDomains(MLIRContext& context)
{
    for (StringRef space : {"vec", "mat", "left", "right", "acc"}) {
        for (bool addressed : {false, true}) {
            const std::string layout =
                space == "vec" ? "blayout=row_major, slayout=none_box" : "blayout=col_major, slayout=row_major";
            const std::string type = "!pto.tile_buf<loc=" + space.str() +
                                     ", dtype=f16, rows=16, cols=16, v_row=?, v_col=?, " + layout +
                                     ", fractal=512, pad=0>";
            const std::string source = "!tile = " + type + R"mlir(
              module attributes {pto.target_arch = "a3"} {
                func.func @metadata(%n: index, %b: i1) {
                  %base = arith.constant 0 : i64
                  %c0 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  %c16 = arith.constant 16 : index
                  %a = pto.alloc_tile )mlir" +
                                       (addressed ? "addr = %base " : "") +
                                       R"mlir(valid_row = %c16 valid_col = %c16 : !tile
                  pto.set_validshape %a, %c1, %c16 : !tile
                  scf.for %i = %c0 to %n step %c1 {
                    scf.if %b {
                      %r, %s = pto.get_validshape %a : !tile
                    }
                  }
                  return
                }
              }
            )mlir";
            auto module = parseSourceString<ModuleOp>(source, &context);
            auto schedule = module ? extract(*module) : nullptr;
            const bool valid = schedule && schedule->getFailures().empty() &&
                               succeeded(verifySyncDescriptorBindings(*schedule)) &&
                               schedule->getDescriptorStates().size() == 2;
            if (!check(valid, "metadata validity is independent of storage domain and physical address")) {
                return false;
            }
            if (!addressed) {
                SyncAccess access;
                access.value = schedule->getDescriptorStates().front().handle;
                access.storage.space =
                    cast<AddressSpaceAttr>(cast<TileBufType>(access.value.getType()).getMemorySpace())
                        .getAddressSpace();
                if (!check(
                        recoverLocalAccessRegion(access).precision == SyncRegionPrecision::Unknown,
                        "known metadata does not establish an unknown physical footprint")) {
                    return false;
                }
            }
        }
    }
    llvm::outs() << "protocol-sync nested descriptor reads: five domains, addressed/unaddressed and mutations pass\n";
    return true;
}

bool testScopedDefinitions(MLIRContext& context)
{
    for (StringRef space : {"vec", "mat", "left", "right", "acc"}) {
        for (bool choice : {false, true}) {
            const std::string layout =
                space == "vec" ? "blayout=row_major, slayout=none_box" : "blayout=col_major, slayout=row_major";
            const std::string source = "!tile = !pto.tile_buf<loc=" + space.str() +
                                       ", dtype=f16, rows=16, cols=16, v_row=?, v_col=?, " + layout +
                                       R"mlir(, fractal=512, pad=0>
                module {
                  func.func @scoped(%n: index, %b: i1) {
                    %c0 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    %c16 = arith.constant 16 : index
                    scf.for %i = %c0 to %n step %c1 {
                )mlir" + (choice ? "scf.if %b {\n" : "") +
                                       R"mlir(
                      %a = pto.alloc_tile valid_row = %c16 valid_col = %c16 : !tile
                      %before_r, %before_c = pto.get_validshape %a : !tile
                      pto.set_validshape %a, %c1, %c16 : !tile
                      scf.if %b {
                        %after_r, %after_c = pto.get_validshape %a : !tile
                      }
                )mlir" + (choice ? "} else {}\n" : "") +
                                       "}\nreturn } }";
            auto module = parseSourceString<ModuleOp>(source, &context);
            auto schedule = module ? extract(*module) : nullptr;
            const bool valid =
                schedule && schedule->getFailures().empty() && succeeded(verifySyncDescriptorBindings(*schedule));
            if (!check(valid, "scope-local descriptor initialization and update")) {
                return false;
            }
            unsigned reads = 0;
            SetValidShapeOp update;
            GetValidShapeOp before;
            for (const auto& action : schedule->getSemanticActions()) {
                if (auto read = dyn_cast<GetValidShapeOp>(action.operation)) {
                    const auto& state = schedule->getDescriptorStates()[*action.descriptorState];
                    if (!check(state.rows == (reads == 0 ? 16 : 1), "fresh initialization then scoped update")) {
                        return false;
                    }
                    if (reads == 0) {
                        before = read;
                    }
                    ++reads;
                }
                if (auto candidate = dyn_cast<SetValidShapeOp>(action.operation)) {
                    update = candidate;
                }
            }
            if (!check(reads == 2 && update && before, "both scoped observations present")) {
                return false;
            }
            Operation* next = update->getNextNode();
            update->moveBefore(before);
            const bool staleRejected = failed(verifySyncDescriptorBindings(*schedule));
            update->moveBefore(next);
            if (!check(staleRejected && succeeded(verifySyncDescriptorBindings(*schedule)), "scoped order mutation")) {
                return false;
            }
            auto allocation = schedule->getDescriptorStates().front().handle.getDefiningOp();
            Operation* allocationNext = allocation->getNextNode();
            auto loop = *schedule->getFunction().getOps<scf::ForOp>().begin();
            allocation->moveBefore(loop);
            const bool ownerRejected = failed(verifySyncDescriptorBindings(*schedule));
            allocation->moveBefore(allocationNext);
            if (!check(ownerRejected, "moving allocation must invalidate owner-bound updates")) {
                return false;
            }
            if (choice) {
                auto branch = cast<scf::IfOp>(allocation->getParentOp());
                Operation* readNext = before->getNextNode();
                before->moveBefore(branch.elseBlock()->getTerminator());
                const bool oppositeRejected = failed(verifySyncDescriptorBindings(*schedule));
                before->moveBefore(readNext);
                if (!check(oppositeRejected, "opposite arms are not the same owner block")) {
                    return false;
                }
            }
            if (!check(succeeded(verifySyncDescriptorBindings(*schedule)), "restored descriptor scope")) {
                return false;
            }
        }
    }
    llvm::outs() << "protocol-sync scoped descriptor definitions: five domains and order mutations pass\n";
    return true;
}

} // namespace

bool testProtocolSyncDescriptorState(MLIRContext& context)
{
    return testVersions(context) && testRejected(context) && testSymbolicDimensions(context) &&
           testScalarProvenanceLimits(context) && testNestedReads(context) && testDescriptorDomains(context) &&
           testScopedDefinitions(context);
}
