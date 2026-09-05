// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- pto-protocol-sync-local-memory-test.cpp - Byte and hazard oracles -----===//

#include "PTO/Transforms/InsertSync/LegacySyncIRAdapter.h"
#include "PTO/Transforms/ProtocolSync/ConcreteSyncVerifier.h"
#include "PTO/Transforms/ProtocolSync/DirectRepair.h"
#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

constexpr StringLiteral kPrelude = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @oracle(%in: !pto.partition_tensor_view<16x16xf16>,
                    %out: !pto.partition_tensor_view<16x16xf16>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : i64
    %c256 = arith.constant 256 : i64
    %c512 = arith.constant 512 : i64
    %a = pto.alloc_tile addr = %c0 : !pto.tile_buf<vec, 16x16xf16>
    %b = pto.alloc_tile addr = %c256 : !pto.tile_buf<vec, 16x16xf16>
    %c = pto.alloc_tile addr = %c512 : !pto.tile_buf<vec, 16x16xf16>
)mlir";

bool check(bool condition, StringRef message)
{
    if (!condition) {
        llvm::errs() << "FAIL: " << message << '\n';
    }
    return condition;
}

std::string makeProgram(unsigned encoding)
{
    std::string text = kPrelude.str();
    constexpr StringLiteral tileType = "!pto.tile_buf<vec, 16x16xf16>";
    const char* buffers[] = {"%a", "%b", "%c", "%a", "%b"};
    for (const char* buffer : buffers) {
        const unsigned mode = encoding % 3;
        encoding /= 3;
        const std::string operand = std::string(buffer) + " : " + tileType.str();
        if (mode == 0) {
            text += "pto.tload ins(%in : !pto.partition_tensor_view<16x16xf16>) outs(" + operand + ")\n";
        } else if (mode == 1) {
            text += "pto.tstore ins(" + operand + ") outs(%out : !pto.partition_tensor_view<16x16xf16>)\n";
        } else {
            text += "pto.tabs ins(" + operand + ") outs(" + operand + ")\n";
        }
    }
    return text + "return\n}\n}\n";
}

struct Fixture {
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    LegacySyncSnapshot snapshot;
    SyncSemanticContext context;
    StructuredSyncIR schedule;

    explicit Fixture(OwningOpRef<ModuleOp> input)
        : module(std::move(input)), function(*module->getOps<func::FuncOp>().begin()), schedule(function)
    {}

    bool build()
    {
        LegacySyncIRAdapter adapter;
        if (failed(adapter.buildSnapshot(function, snapshot))) {
            return false;
        }
        context = adapter.buildSemanticContext(snapshot);
        return succeeded(StructuredSyncIRBuilder(context).build(function, schedule));
    }
};

bool checkByteOracle(const Fixture& fixture, const SyncLocalMemoryAnalysis& analysis)
{
    for (const SyncLocalAccessRegion& region : analysis.regions) {
        const SyncAccess* access = fixture.schedule.findAccess(region.access);
        auto allocation = access->value.getDefiningOp<AllocTileOp>();
        auto address = allocation.getAddr().getDefiningOp<arith::ConstantOp>();
        const auto begin = cast<IntegerAttr>(address.getValue()).getValue().getZExtValue();
        std::set<std::uint64_t> expected;
        std::set<std::uint64_t> actual;
        // All fixture tiles contain exactly 256 f16 values; do not reuse the
        // production size or partition calculation for this reference set.
        for (std::uint64_t byte = begin; byte < begin + 512; ++byte) {
            expected.insert(byte);
        }
        for (const SyncLocalStorageAtom& atom : analysis.atoms) {
            if (!llvm::is_contained(atom.accesses, region.access)) {
                continue;
            }
            for (std::uint64_t byte = atom.interval.begin; byte < atom.interval.begin + atom.interval.size; ++byte) {
                if (!actual.insert(byte).second) {
                    return check(false, "atom membership must occur exactly once");
                }
            }
        }
        if (!check(actual == expected, "atom union must reconstruct the independent byte set")) {
            return false;
        }
    }
    return true;
}

bool testSparseChains(MLIRContext& context)
{
    constexpr unsigned programCount = 243; // Every R/W/RW sequence of length five.
    for (unsigned encoding = 0; encoding < programCount; ++encoding) {
        auto module = parseSourceString<ModuleOp>(makeProgram(encoding), &context);
        if (!module || failed(verify(*module))) {
            return check(false, "generated fixture must parse and verify");
        }
        Fixture fixture(std::move(module));
        if (!fixture.build()) {
            return check(false, "generated fixture extraction");
        }
        auto analysis = analyzeLocalMemory(fixture.schedule);
        const bool validAnalysis =
            succeeded(analysis) && analysis->boundary.empty() && checkByteOracle(fixture, *analysis);
        if (!validAnalysis) {
            return check(false, "generated local analysis");
        }
        SyncSelectedWorld world;
        for (const SyncResidualObligation& requirement : analysis->requirements) {
            const bool duplicate = llvm::any_of(world.completions, [&](const SyncSelectedCompletion& completion) {
                return completion.source == requirement.source && completion.target == requirement.target;
            });
            if (!duplicate) {
                world.completions.push_back(
                    {requirement.source, requirement.target, requirement.control, requirement.iteration});
            }
        }
        if (!check(
                succeeded(verifyLocalMemoryCoverage(fixture.schedule, world)),
                "sparse chain must cover every raw hazard")) {
            return false;
        }
        if (!analysis->requirements.empty()) {
            SyncAccessId source;
            SyncAccessId target;
            const bool rejects = failed(verifyLocalMemoryCoverage(fixture.schedule, {}, &source, &target));
            if (!check(
                    rejects && source != kInvalidSyncId && target != kInvalidSyncId, "missing order needs a witness")) {
                return false;
            }
        }
    }
    llvm::outs() << "protocol-sync local byte and sparse-pair oracle: 243 programs pass\n";
    return true;
}

bool testConcreteMutations(MLIRContext& context)
{
    // No GM write before a later GM read: the default may-alias contract needs
    // only completion here, not an unqualified global publication mechanism.
    std::string text = kPrelude.str();
    text += R"mlir(
      pto.tload ins(%in : !pto.partition_tensor_view<16x16xf16>) outs(%a : !pto.tile_buf<vec, 16x16xf16>)
      pto.tabs ins(%a : !pto.tile_buf<vec, 16x16xf16>) outs(%b : !pto.tile_buf<vec, 16x16xf16>)
      pto.tabs ins(%b : !pto.tile_buf<vec, 16x16xf16>) outs(%c : !pto.tile_buf<vec, 16x16xf16>)
      pto.tstore ins(%c : !pto.tile_buf<vec, 16x16xf16>) outs(%out : !pto.partition_tensor_view<16x16xf16>)
      return
    }
  }
    )mlir";
    auto module = parseSourceString<ModuleOp>(text, &context);
    if (!module) {
        return false;
    }
    Fixture fixture(std::move(module));
    if (!fixture.build()) {
        return false;
    }
    auto stages = analyzePipelineStages(fixture.schedule);
    if (failed(stages)) {
        return false;
    }
    auto timelines = analyzeStorageTimelines(fixture.schedule, *stages);
    auto channels = analyzeChannels(fixture.schedule, *stages, timelines);
    auto residuals = interpretSelectedWorld(fixture.schedule, *stages, timelines, channels, {});
    if (failed(residuals)) {
        return false;
    }
    auto plan = buildDirectRepairPlan(fixture.schedule, *stages, residuals->obligations);
    const bool allocated =
        succeeded(plan) && plan->isComplete() && succeeded(allocateDirectRepairEvents(fixture.schedule, *plan));
    if (!allocated) {
        return check(false, "local plan must be complete and allocate");
    }
    if (failed(materializeAndVerifyDirectRepairPlan(fixture.schedule, *stages, residuals->obligations, *plan))) {
        return check(false, "local plan must pass concrete verification");
    }
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
        OwningOpRef<ModuleOp> clone = cast<ModuleOp>(fixture.module->clone());
        auto function = *clone->getOps<func::FuncOp>().begin();
        auto set = *function.getOps<SetFlagOp>().begin();
        auto wait = *function.getOps<WaitFlagOp>().begin();
        auto load = *function.getOps<TLoadOp>().begin();
        auto compute = *function.getOps<TAbsOp>().begin();
        if (mutation == 0) {
            set.erase();
        } else if (mutation == 1) {
            set->moveBefore(load);
        } else if (mutation == 2) {
            wait->moveAfter(compute);
        } else {
            auto barrier = *function.getOps<BarrierOp>().begin();
            barrier.erase();
        }
        if (!check(
                failed(verifyFreshConcreteSyncSemantics(function)), "concrete synchronization mutation must reject")) {
            return false;
        }
    }
    llvm::outs() << "protocol-sync local concrete mutations: pass\n";
    return true;
}

bool testUnknownAndOverflow(MLIRContext& context)
{
    constexpr StringLiteral source = R"mlir(
module {
  func.func @bounds(%address: i64) {
    %negative = arith.constant -1 : i64
    %eight = arith.constant 8 : i64
    %dynamic = pto.alloc_tile addr = %address : !pto.tile_buf<vec, 16x16xf16>
    %invalid = pto.alloc_tile addr = %negative : !pto.tile_buf<vec, 16x16xf16>
    %multiply_overflow = pto.alloc_tile addr = %eight : !pto.tile_buf<vec, 1x4611686018427387904xf32>
    %addition_overflow = pto.alloc_tile addr = %eight : !pto.tile_buf<vec, 1x4611686018427387903xf32>
    return
  }
}
)mlir";
    // Exercise footprint recovery on malformed bounds without asking unrelated
    // operation verifiers to accept negative addresses or overflowing shapes.
    auto module = parseSourceString<ModuleOp>(source, ParserConfig(&context, /*verifyAfterParse=*/false));
    if (!module) {
        return false;
    }
    auto function = *module->getOps<func::FuncOp>().begin();
    for (auto allocation : function.getOps<AllocTileOp>()) {
        SyncAccess access;
        access.value = allocation.getResult();
        access.storage.space = AddressSpace::VEC;
        const SyncLocalAccessRegion region = recoverLocalAccessRegion(access);
        if (!check(region.precision == SyncRegionPrecision::Unknown, "invalid footprint must stay unknown")) {
            return false;
        }
    }
    llvm::outs() << "protocol-sync local unknown and overflow bounds: pass\n";
    return true;
}

std::string branchEffect(unsigned mode, StringRef buffer)
{
    const std::string operand = buffer.str() + " : !pto.tile_buf<vec, 16x16xf16>";
    if (mode == 0) {
        return "pto.tload ins(%in : !pto.partition_tensor_view<16x16xf16>) outs(" + operand + ")\n";
    }
    if (mode == 1) {
        return "pto.tstore ins(" + operand + ") outs(%out : !pto.partition_tensor_view<16x16xf16>)\n";
    }
    return "pto.tabs ins(" + operand + ") outs(" + operand + ")\n";
}

std::string makeBranchProgram(unsigned encoding, bool emptyElse)
{
    std::string text = kPrelude.str();
    const auto position = text.find("@oracle(");
    text.insert(position + StringRef("@oracle(").size(), "%condition: i1, ");
    text += branchEffect(0, "%a") + "scf.if %condition {\n";
    for (unsigned arm = 0; arm < 2; ++arm) {
        for (StringRef buffer : {StringRef("%a"), StringRef("%b")}) {
            if (arm == 0 || !emptyElse) {
                text += branchEffect(encoding % 3, buffer);
            }
            encoding /= 3;
        }
        text += arm == 0 ? "} else {\n" : "}\n";
    }
    return text + branchEffect(1, "%a") + branchEffect(0, "%b") + "return\n}\n}\n";
}

bool onPath(const SyncPhase& phase, unsigned arm) { return phase.guard.empty() || phase.guard.front().arm == arm; }

bool checkMayDefinitions(const Fixture& fixture, const SyncLocalMemoryAnalysis& analysis)
{
    const auto phases = fixture.schedule.getPhases();
    for (const SyncLocalMemoryState& state : analysis.states) {
        std::set<std::uint32_t> expected;
        for (unsigned arm = 0; arm < 2; ++arm) {
            if (!onPath(phases[state.phase], arm)) {
                continue;
            }
            for (const SyncLocalMemoryState& prior : analysis.states) {
                const bool possibleWriter = prior.atom == state.atom && prior.writes && prior.phase < state.phase &&
                                            onPath(phases[prior.phase], arm);
                if (possibleWriter) {
                    expected.insert(prior.id);
                }
            }
        }
        const std::set<std::uint32_t> actual(state.mayDefinitions.begin(), state.mayDefinitions.end());
        if (!check(actual == expected, "may-definitions must equal possible prior conservative writes")) {
            return false;
        }
    }
    return true;
}

bool checkBranchPaths(const Fixture& fixture, const SyncLocalMemoryAnalysis& analysis)
{
    const auto phases = fixture.schedule.getPhases();
    for (unsigned arm = 0; arm < 2; ++arm) {
        SmallVector<llvm::BitVector> reach(phases.size(), llvm::BitVector(phases.size()));
        for (const SyncResidualObligation& requirement : analysis.requirements) {
            const auto& source = phases[requirement.source];
            const auto& target = phases[requirement.target];
            const bool oppositeArms =
                !source.guard.empty() && !target.guard.empty() && source.guard.front().arm != target.guard.front().arm;
            if (oppositeArms) {
                return check(false, "must not order mutually exclusive arms");
            }
            const bool bothExecute = onPath(source, arm) && onPath(target, arm);
            if (bothExecute) {
                reach[source.id].set(target.id);
            }
        }
        for (unsigned middle = 0; middle < phases.size(); ++middle) {
            for (auto& successors : reach) {
                if (successors.test(middle)) {
                    successors |= reach[middle];
                }
            }
        }
        // Reference intervals come from the fixture's three known byte sets,
        // not from production atoms or region summaries.
        for (const SyncAccess& source : fixture.schedule.getAccesses()) {
            if (source.storage.space != AddressSpace::VEC || !onPath(phases[source.phase], arm)) {
                continue;
            }
            auto first = source.value.getDefiningOp<AllocTileOp>().getAddr().getDefiningOp<arith::ConstantOp>();
            const auto begin = cast<IntegerAttr>(first.getValue()).getInt();
            for (const SyncAccess& target : fixture.schedule.getAccesses()) {
                if (target.storage.space != AddressSpace::VEC || source.phase >= target.phase ||
                    !onPath(phases[target.phase], arm) ||
                    (source.mode == SyncAccessMode::Read && target.mode == SyncAccessMode::Read)) {
                    continue;
                }
                auto second = target.value.getDefiningOp<AllocTileOp>().getAddr().getDefiningOp<arith::ConstantOp>();
                const auto endBegin = cast<IntegerAttr>(second.getValue()).getInt();
                const bool overlaps = begin < endBegin + 512 && endBegin < begin + 512;
                if (overlaps && !reach[source.phase].test(target.phase)) {
                    return check(false, "branch transfer dropped a feasible cross-boundary hazard");
                }
            }
        }
    }
    return true;
}

bool testCompositionalRegions(MLIRContext& context)
{
    for (unsigned encoding = 0; encoding < 162; ++encoding) {
        auto module = parseSourceString<ModuleOp>(makeBranchProgram(encoding % 81, encoding >= 81), &context);
        if (!module) {
            return false;
        }
        Fixture fixture(std::move(module));
        if (!fixture.build()) {
            return false;
        }
        auto baseline = analyzeLocalMemory(fixture.schedule);
        const bool protectedBoundary = succeeded(baseline) && !baseline->boundary.empty() &&
                                       baseline->coveredAccesses.none() && baseline->states.empty() &&
                                       baseline->requirements.empty() && baseline->regionSummaries.empty() &&
                                       baseline->expandedStateStatus == SyncLocalExpandedStateStatus::NotRequested;
        if (!check(protectedBoundary, "production must not run unused cross-region expansion")) {
            return false;
        }
        auto analysis = analyzeLocalMemory(fixture.schedule, {/*expandState=*/true});
        const bool validAnalysis = succeeded(analysis) &&
                                   analysis->expandedStateStatus == SyncLocalExpandedStateStatus::Complete &&
                                   !analysis->regionSummaries.empty() && checkByteOracle(fixture, *analysis) &&
                                   checkBranchPaths(fixture, *analysis) && checkMayDefinitions(fixture, *analysis);
        if (!validAnalysis) {
            return false;
        }
        if (!check(failed(analyzeLocalRegionFlow(fixture.schedule, *analysis)), "must not append a second transfer")) {
            return false;
        }
        if (!check(analysis->coveredAccesses.none(), "unverified branch boundaries must retain old protection")) {
            return false;
        }
        for (const SyncLocalRegionSummary& summary : analysis->regionSummaries) {
            if (!check(
                    summary.complete && summary.outgoing.mustDefinitions.empty() && summary.outgoing.mayHaveLiveIn,
                    "conservative writes are not kills")) {
                return false;
            }
            for (std::uint32_t definition : summary.incoming.mayDefinitions) {
                if (!check(llvm::is_contained(summary.outgoing.mayDefinitions, definition), "incoming may-def lost")) {
                    return false;
                }
            }
        }
    }
    llvm::outs() << "protocol-sync compositional branch oracle: 162 programs, 324 paths pass\n";
    return true;
}

bool testExpansionLimits(MLIRContext& context)
{
    constexpr unsigned writeCount = 1600;
    std::string text = kPrelude.str();
    for (unsigned i = 0; i < writeCount; ++i) {
        text += branchEffect(0, "%a");
    }
    auto module = parseSourceString<ModuleOp>(text + "return\n}\n}\n", &context);
    if (!module) {
        return false;
    }
    Fixture fixture(std::move(module));
    if (!fixture.build()) {
        return false;
    }
    auto baseline = analyzeLocalMemory(fixture.schedule);
    const bool sparse = succeeded(baseline) && baseline->boundary.empty() && baseline->atoms.size() == 1 &&
                        baseline->states.size() == writeCount && baseline->requirements.size() == writeCount - 1 &&
                        baseline->coveredAccesses.count() == writeCount && baseline->regionSummaries.empty() &&
                        baseline->expandedStateStatus == SyncLocalExpandedStateStatus::NotRequested;
    if (!check(sparse, "long straight-line production chain must remain sparse and covered")) {
        return false;
    }
    for (const SyncLocalMemoryState& state : baseline->states) {
        const auto predecessor = state.id == 0 ? kInvalidSyncId : state.id - 1;
        if (!check(
                state.mayDefinitions.empty() && state.previousDefinition == predecessor,
                "production uses predecessor links, not expanded may-definition snapshots")) {
            return false;
        }
    }
    auto expanded = analyzeLocalMemory(fixture.schedule, {/*expandState=*/true});
    const bool preserved = succeeded(expanded) &&
                           expanded->expandedStateStatus == SyncLocalExpandedStateStatus::LimitExceeded &&
                           expanded->coveredAccesses == baseline->coveredAccesses &&
                           expanded->requirements.size() == baseline->requirements.size() &&
                           expanded->states.size() == baseline->states.size() && expanded->regionSummaries.empty();
    if (!check(preserved, "diagnostic exhaustion must preserve the complete production baseline")) {
        return false;
    }
    for (unsigned id = 0; id < baseline->requirements.size(); ++id) {
        const auto& expected = baseline->requirements[id];
        const auto& actual = expanded->requirements[id];
        if (!check(
                actual.id == expected.id && actual.kind == expected.kind &&
                    actual.sourceAccess == expected.sourceAccess && actual.targetAccess == expected.targetAccess &&
                    actual.atoms == expected.atoms,
                "expansion limit must not change requirement identity or endpoints")) {
            return false;
        }
    }
    auto branch = parseSourceString<ModuleOp>(makeBranchProgram(0, false), &context);
    if (!branch) {
        return false;
    }
    Fixture branchFixture(std::move(branch));
    if (!branchFixture.build()) {
        return false;
    }
    auto limited = analyzeLocalMemory(branchFixture.schedule, {/*expandState=*/true, /*maximumExpandedEntries=*/0});
    const bool boundaryPreserved = succeeded(limited) && !limited->boundary.empty() &&
                                   limited->expandedStateStatus == SyncLocalExpandedStateStatus::LimitExceeded &&
                                   limited->coveredAccesses.none() && limited->states.empty() &&
                                   limited->requirements.empty() && limited->regionSummaries.empty();
    if (!check(boundaryPreserved, "cross-region limit must retain old protection without partial results")) {
        return false;
    }
    llvm::outs() << "protocol-sync expansion limits: 1600-write baseline and branch budget pass\n";
    return true;
}

} // namespace

int main()
{
    DialectRegistry registry;
    registry.insert<PTODialect, arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
    MLIRContext context(registry);
    context.disableMultithreading();
    return testSparseChains(context) && testConcreteMutations(context) && testUnknownAndOverflow(context) &&
                   testCompositionalRegions(context) && testExpansionLimits(context) ?
               0 :
               1;
}
