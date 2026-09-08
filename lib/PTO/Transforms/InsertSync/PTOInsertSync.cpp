// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOInsertSync.cpp - PTO Insert Synchronization for PTO Pipeline ----===//
//===----------------------------------------------------------------------===//
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "PTO/Transforms/InsertSync/PTOIRTranslator.h"
#include "PTO/Transforms/InsertSync/InsertSyncAnalysis.h"
#include "PTO/Transforms/InsertSync/InsertSyncDebug.h"
#include "PTO/Transforms/InsertSync/MoveSyncState.h"
#include "PTO/Transforms/InsertSync/RemoveRedundantSync.h"
#include "PTO/Transforms/InsertSync/SyncEventIdAllocation.h"
#include "PTO/Transforms/InsertSync/SyncCodegen.h"
#include "PTO/Transforms/InsertSync/SyncAudit.h"
#include "PTO/Transforms/InsertSync/SyncEffectCoverage.h"
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include "PTO/Transforms/InsertSync/InsertSyncOptions.h"
#include "PTO/Transforms/InsertSync/PruneCompletedBarriers.h"
#include "PTO/Transforms/InsertSync/StorageFrontierAnalysis.h"
#include "PTO/Transforms/InsertSync/LifecycleSynthesis.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h" // [FIX] 确保 FuncOp 定义可见

// [CRITICAL FIX] 必须在包含 .inc 之前设置好命名空间环境
// 将 Passes.h.inc 生成的声明包裹在 namespace mlir 中
// 此外，为了确保 Passes.h.inc 中生成的 func::FuncOp 能被解析，
// 我们需要在 pto 命名空间内给 func 做一个别名。

namespace mlir {
namespace pto {
  // [FIX] 给 mlir::func 起别名为 func，这样 .inc 文件里的 func::FuncOp 就能找到了
  namespace func = ::mlir::func;

  #define GEN_PASS_DEF_PTOINSERTSYNC
  #include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

// PTOMaterializeTileOpSections establishes an atomic compute-helper ABI:
// one V/M lane, no DMA, no pipe events and complete argument effects.
// InsertSync orders callers using that ABI; it must not independently run
// tile-level synchronization inside the helper's lower-level instruction body.
static bool hasAtomicTileOpContract(func::FuncOp function)
{
    if (!function->hasAttr("pto.tileop.helper") || function.getNumResults()) {
        return false;
    }
    auto kind = function->getAttrOfType<StringAttr>("pto.tileop.kind");
    auto effects = function->getAttrOfType<ArrayAttr>("pto.tileop.effects");
    if (!kind || (kind.getValue() != "vector" && kind.getValue() != "cube") || !effects ||
        effects.size() != function.getNumArguments()) {
        return false;
    }
    if (!llvm::all_of(effects, [](Attribute attribute) {
            auto effect = dyn_cast<StringAttr>(attribute);
            return effect && (effect.getValue() == "none" || effect.getValue() == "read" ||
                              effect.getValue() == "write" || effect.getValue() == "readwrite");
        })) {
        return false;
    }
    // Require the materialized compute section as well as the ABI. Bare
    // attributes on an arbitrary function are not a reason to skip analysis.
    bool section = false;
    bool compatible = true;
    function.getBody().walk([&](Operation* op) {
        if (isa<SectionVectorOp>(op)) {
            section = true;
            compatible &= kind.getValue() == "vector";
        } else if (isa<SectionCubeOp>(op)) {
            section = true;
            compatible &= kind.getValue() == "cube";
        } else if (auto pipe = dyn_cast<OpPipeInterface>(op)) {
            compatible &= pipe.getPipe() == (kind.getValue() == "vector" ? PIPE::PIPE_V : PIPE::PIPE_M);
        } else if (isa<func::CallOp, SetFlagOp, WaitFlagOp, RecordEventOp, WaitEventOp, BarrierOp>(op)) {
            compatible = false;
        }
    });
    for (auto [type, effect] : llvm::zip(function.getArgumentTypes(), effects)) {
        if (!isa<TileBufType>(type) && cast<StringAttr>(effect).getValue() != "none") {
            compatible = false;
        }
    }
    return section && compatible;
}

// ==============================================================================
// Main Pass Implementation
// ==============================================================================

static bool hasGatherScatterLikeOps(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<pto::TGatherOp, pto::TGatherBOp, pto::TScatterOp, pto::MGatherOp,
            pto::MScatterOp>(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

struct PTOInsertSyncPass : public mlir::pto::impl::PTOInsertSyncBase<PTOInsertSyncPass> {
    PTOInsertSyncPass() = default;
    explicit PTOInsertSyncPass(const InsertSyncOptions& options)
    {
        deferSamePipe = options.deferSamePipe;
        gmAlias = options.gmAlias;
        audit = options.audit;
        effectCoverage = options.effectCoverage;
        pruneCompletedBarriers = options.pruneCompletedBarriers;
        mmadChains = options.mmadChains;
        frontierRefinement = options.frontierRefinement;
        frontierPlacement = options.frontierPlacement;
        lifecycleSynthesis = options.lifecycleSynthesis;
        bufferGenerations = options.bufferGenerations;
    }
    PTOInsertSyncPass(const PTOInsertSyncPass& other) : PTOInsertSyncBase(other)
    {
        deferSamePipe = other.deferSamePipe;
        gmAlias = other.gmAlias;
        audit = other.audit;
        effectCoverage = other.effectCoverage;
        pruneCompletedBarriers = other.pruneCompletedBarriers;
        mmadChains = other.mmadChains;
        frontierRefinement = other.frontierRefinement;
        frontierPlacement = other.frontierPlacement;
        lifecycleSynthesis = other.lifecycleSynthesis;
        bufferGenerations = other.bufferGenerations;
    }

    Option<bool> deferSamePipe{
        *this, "defer-same-pipe", llvm::cl::init(false),
        llvm::cl::desc("Establish cross-pipe supply before same-pipe repair")};
    Option<std::string> gmAlias{
        *this, "gm-alias", llvm::cl::init(""), llvm::cl::desc("GM contract: may-alias or assume-disjoint-arguments")};
    Option<std::string> audit{
        *this, "audit", llvm::cl::init("off"), llvm::cl::desc("Independent local audit: off, report, or strict")};
    Option<std::string> effectCoverage{
        *this, "effect-coverage", llvm::cl::init("report"),
        llvm::cl::desc("Translator completeness rollout: report (default) or strict")};
    Option<bool> pruneCompletedBarriers{
        *this, "prune-completed-barriers", llvm::cl::init(false),
        llvm::cl::desc("Remove only named barriers whose whole source prefix is already complete")};
    Option<bool> mmadChains{
        *this, "mmad-chains", llvm::cl::init(false),
        llvm::cl::desc("Use qualified A2/A3 structured accumulator ordering; experimental")};
    Option<bool> frontierRefinement{
        *this, "frontier-refinement", llvm::cl::init(false),
        llvm::cl::desc("Refine generated named barriers using storage/lane frontiers; experimental")};
    Option<bool> frontierPlacement{
        *this, "frontier-placement", llvm::cl::init(false),
        llvm::cl::desc("Place generated handoffs at proved storage frontiers; includes refinement; experimental")};

    Option<bool> lifecycleSynthesis{
        *this, "lifecycle-synthesis", llvm::cl::init(false),
        llvm::cl::desc("Construct exact-slot lifecycle protocols before residual insertion; experimental")};

    Option<bool> bufferGenerations{
        *this, "buffer-generations", llvm::cl::init(false),
        llvm::cl::desc("Construct synchronization from per-buffer reaching generations; experimental")};

    void auditOutput(func::FuncOp function)
    {
        if (audit == "off") {
            return;
        }
        auto result = auditInsertSyncLocal(function);
        function->setAttr(
            "pto.insert_sync.audit", StringAttr::get(&getContext(), stringifyInsertSyncAuditStatus(result.status)));
        function->setAttr("pto.insert_sync.audit_reason", StringAttr::get(&getContext(), result.reason));
        function.emitRemark("InsertSync audit: ")
            << stringifyInsertSyncAuditStatus(result.status) << ": " << result.reason;
        if (audit == "strict" && result.status != InsertSyncAuditStatus::VerifiedLocal) {
            function.emitError("InsertSync strict local audit did not establish safety");
            signalPassFailure();
        }
    }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    // Backend-partitioned PTODSL containers carry private func declarations
    // in the outer child module to model cross-child calls. Those declaration
    // funcs have a function type but no entry block arguments, so the
    // translator's argument walk must not run on them.
    if (func.isDeclaration()) {
      return;
    }
    if (audit != "off" && audit != "report" && audit != "strict") {
        func.emitError("InsertSync audit must be off, report, or strict");
        signalPassFailure();
        return;
    }
    if (effectCoverage != "report" && effectCoverage != "strict") {
        func.emitError("InsertSync effect-coverage must be report or strict");
        signalPassFailure();
        return;
    }
    auto contract = resolveInsertSyncGMAlias(func, gmAlias);
    if (failed(contract)) {
        signalPassFailure();
        return;
    }
    func->setAttr(
        "pto.gm_alias",
        StringAttr::get(
            &getContext(), *contract == InsertSyncGMAliasMode::MayAlias ? "may-alias" : "assume-disjoint-arguments"));

    if (hasAtomicTileOpContract(func)) {
        func->setAttr("pto.insert_sync.status", StringAttr::get(&getContext(), "atomic-helper-contract"));
        auditOutput(func);
        return;
    }

    // If the function already contains explicit synchronization ops (either
    // low-level pipe flags or the higher-level record/wait events), do not run
    // the automatic insertion pass again. Re-inserting on top of manual sync
    // can introduce duplicated/mismatched event dependencies that may lead to
    // runtime failures on NPU.
    //
    bool hasExplicitSync = false;
    func.walk([&](Operation *op) {
      if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::RecordEventOp,
              pto::WaitEventOp>(op)) {
        hasExplicitSync = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasExplicitSync) {
        func->setAttr("pto.insert_sync.status", StringAttr::get(&getContext(), "explicit-sync-bypass"));
        auditOutput(func);
        return;
    }

    if (lifecycleSynthesis || bufferGenerations) {
        InsertSyncOptions lifecycleOptions;
        lifecycleOptions.deferSamePipe = deferSamePipe;
        lifecycleOptions.mmadChains = mmadChains;
        lifecycleOptions.effectCoverage = effectCoverage;
        lifecycleOptions.bufferGenerations = bufferGenerations;
        lifecycleOptions.frontierRefinement = frontierRefinement;
        lifecycleOptions.frontierPlacement = frontierPlacement;
        auto result = tryInsertSyncLifecycleSynthesis(func, lifecycleOptions);
        func.emitRemark("InsertSync lifecycle synthesis: ")
            << result.reason << "; attempted=" << result.attempted
            << ", proposed=" << result.selected << ", streams=" << result.logicalStreams
            << ", supplied-access-pairs=" << result.suppliedPairs
            << ", combined-event-audit=" << (result.combinedEventAuditProved ? "proved" : "unproved")
            << ", planning-attempts=" << result.planningAttempts << ", retries=" << result.retries
            << ", consumer-regions=" << result.consumerRegions << ", guarded-actions=" << result.guardedActions;
        for (const auto& diagnostic : result.diagnostics) {
            func.emitRemark("InsertSync lifecycle candidate: ")
                << diagnostic.identity << "; stage=" << diagnostic.stage
                << "; accepted=" << diagnostic.accepted << "; " << diagnostic.reason;
        }
        if (result.status == InsertSyncLifecycleResult::Status::InternalError ||
            result.status == InsertSyncLifecycleResult::Status::InputError) {
            func.emitError(result.reason);
            signalPassFailure();
            return;
        }
        if (result.status == InsertSyncLifecycleResult::Status::Applied) {
            func->setAttr("pto.insert_sync.status", StringAttr::get(&getContext(), "lifecycle-plus-residuals"));
            if (bufferGenerations)
                func->setAttr("pto.insert_sync.buffer_generations", UnitAttr::get(&getContext()));
            // Counts are output diagnostics, never input semantic promises.
            auto type = IntegerType::get(&getContext(), 64);
            func->setAttr("pto.insert_sync.lifecycle_channels", IntegerAttr::get(type, result.selected));
            func->setAttr("pto.insert_sync.lifecycle_retries", IntegerAttr::get(type, result.retries));
            func->setAttr("pto.insert_sync.lifecycle_consumer_regions", IntegerAttr::get(type, result.consumerRegions));
            func->setAttr("pto.insert_sync.lifecycle_guarded_actions", IntegerAttr::get(type, result.guardedActions));
            func->setAttr("pto.insert_sync.lifecycle_supplied_pairs", IntegerAttr::get(type, result.suppliedPairs));
            auditOutput(func);
            return;
        }
        // Optional construction failed before mutating func. Run the existing
        // path, including its configured R4/R5 refinement and scarcity policy.
    }

    // 0. 数据结构准备
    MemoryDependentAnalyzer memAnalyzer;
    memAnalyzer.setGMContract(func, *contract);
    SyncIRs syncIR;
    SyncOperations syncOpsStorage;
    Buffer2MemInfoMap buffer2MemInfoMap;

    // 1. Translator: 构建 SyncIR
    PTOIRTranslator translator(syncIR, memAnalyzer, buffer2MemInfoMap, func, SyncAnalysisMode::NORMALSYNC);
    translator.enableGenerationFlow(bufferGenerations);
    translator.Build();
    auto coverage = inspectInsertSyncEffectCoverage(func, syncIR, effectCoverage == "strict");
    if (failed(coverage)) {
        signalPassFailure();
        return;
    }
    func->setAttr("pto.insert_sync.effect_coverage",
                  StringAttr::get(&getContext(), *coverage ? "complete" : "gap-legacy-retained"));
    func->setAttr("pto.insert_sync.status", StringAttr::get(&getContext(), "analyzed"));

    // 如果 IR 太简单，直接跳过
    if (syncIR.size() <= 1) {
        auditOutput(func);
        return;
    }

    dumpInsertSyncPhase("After Translator", syncIR, syncOpsStorage,
                        func.getOperation());

    // 2. Analyzer: 依赖分析与插入逻辑 Sync
    InsertSyncAnalysis analyzer(syncIR, memAnalyzer, syncOpsStorage, func,
                                SyncAnalysisMode::NORMALSYNC);
    std::optional<InsertSyncLifecycleStructure> generationStructure;
    if (bufferGenerations) {
        insert_sync_frontier::Budget budget;
        generationStructure = buildInsertSyncLifecycleStructure(func, syncIR, budget, true);
        if (generationStructure->status == StorageFrontierSnapshot::Status::Complete)
            analyzer.setStorageFlow(&generationStructure->storageFlow);
    }
    analyzer.Run(/*insertBarAllAtLast=*/true,
                 /*deferSamePipeRepair=*/deferSamePipe,
                 /*useMmadChains=*/mmadChains);

    dumpInsertSyncPhase("After Analysis", syncIR, syncOpsStorage,
                        func.getOperation());

    // [NEW] 3. Optimization: Sync Motion
    // 将不必要的 Wait 提至 Loop 外，将不必要的 Set 沉降到 Loop 后
    MoveSyncState syncMove(syncIR, syncOpsStorage);
    syncMove.Run(); // 执行优化

    dumpInsertSyncPhase("After Sync Motion", syncIR, syncOpsStorage,
                        func.getOperation());

    // 4. [NEW] Optimization 2: Remove Redundant Sync
    // 消除由于 Motion 或 Analysis 产生的冗余同步对。
    //
    // NOTE:
    // Current redundancy matching is pipe-pair based and may over-remove
    // set/wait around gather/scatter-like ops on A5, causing runtime mismatch
    // or vector exceptions. Keep correctness-first behavior here by skipping
    // this optimization for those kernels until dependency-aware matching is
    // added.
    if (!hasGatherScatterLikeOps(func)) {
      RemoveRedundantSync removeRedundant(syncIR, syncOpsStorage,
                                          SyncAnalysisMode::NORMALSYNC);
      removeRedundant.Run();
    }

    dumpInsertSyncPhase("After Remove Redundant Sync", syncIR, syncOpsStorage,
                        func.getOperation());

    SyncEventIdAllocation eventIdAllocation(syncIR, syncOpsStorage);
    eventIdAllocation.Allocate();

    dumpInsertSyncPhase("After EventId Allocation", syncIR, syncOpsStorage,
                        func.getOperation());

    // Existing fixed barriers are never owned by this optional refinement.
    // Capture identities before code generation; do not infer ownership from
    // the absence of a user tag after emission.
    llvm::SmallPtrSet<Operation *, 32> fixedInputBarriers;
    if (
        (frontierRefinement || frontierPlacement || bufferGenerations) && *coverage) {
        func.walk([&](BarrierOp barrier) { fixedInputBarriers.insert(barrier.getOperation()); });
    }
    SyncCodegen codegen(syncIR, func, SyncAnalysisMode::NORMALSYNC);
    codegen.Run();
    if (!recheckInsertSyncGenerationRequirements(func, analyzer.getGenerationMmadRequirements(), analyzer.getSlotRequirements())) {
        func.emitError("emitted generation-qualified requirements could not be reconstructed");
        signalPassFailure();
        return;
    }
    if (bufferGenerations)
        func->setAttr("pto.insert_sync.generation_mmad_witnesses",
                      IntegerAttr::get(IntegerType::get(&getContext(), 64), analyzer.getGenerationMmadRequirements().size()));
    if (bufferGenerations && *coverage) {
        func->setAttr("pto.insert_sync.generation_slot_witnesses",
                      IntegerAttr::get(IntegerType::get(&getContext(), 64), analyzer.getSlotRequirements().size()));
        SmallVector<Operation*> candidates;
        func.walk([&](BarrierOp barrier) {
            if (!fixedInputBarriers.contains(barrier.getOperation())) candidates.push_back(barrier.getOperation());
        });
        insert_sync_frontier::Budget budget;
        auto cleanup = refineInsertSyncCompletion(func, syncIR, candidates, budget);
        func->setAttr("pto.insert_sync.generation_barriers_removed",
                      IntegerAttr::get(IntegerType::get(&getContext(), 64), cleanup.removed));
    }
    if (pruneCompletedBarriers && *coverage) {
        // Optional refinement never becomes an admission gate. It does not
        // remove events, move endpoints, or alter allocator/scarcity decisions.
        auto result = pruneProvenCompletedBarriers(func);
        func->setAttr("pto.insert_sync.completed_barriers_removed",
                      IntegerAttr::get(IntegerType::get(&getContext(), 64), result.removed));
        func.emitRemark("InsertSync completed-prefix pruning: ")
            << result.removed << " barriers removed; " << result.reason;
    }
    if (
        (frontierRefinement || frontierPlacement) && *coverage) {
        SmallVector<Operation *> candidates;
        func.walk([&](BarrierOp barrier) {
            if (
                barrier.getPipe().getPipe() != PIPE::PIPE_ALL &&
                !fixedInputBarriers.contains(barrier.getOperation())) {
                candidates.push_back(barrier.getOperation());
            }
        });
        SmallVector<Operation *> ownedEvents;
        if (frontierPlacement) {
            func.walk([&](Operation *op) {
                // Production has already bypassed authored local sync before
                // insertion. Only the newly emitted static events reach this path.
                if (isa<SetFlagOp, WaitFlagOp>(op)) {
                    ownedEvents.push_back(op);
                }
            });
        }
        auto result = refineInsertSyncStorageFrontiers(
            func, syncIR, candidates, mmadChains, frontierPlacement, ownedEvents);
        auto i64 = IntegerType::get(&getContext(), 64);
        func->setAttr("pto.insert_sync.frontier_barriers_removed", IntegerAttr::get(i64, result.removed));
        func->setAttr("pto.insert_sync.frontier_barriers_guarded", IntegerAttr::get(i64, result.guarded));
        func->setAttr("pto.insert_sync.frontier_requirements", IntegerAttr::get(i64, result.requirements));
        func->setAttr("pto.insert_sync.frontier_work", IntegerAttr::get(i64, result.work));
        func->setAttr(
            "pto.insert_sync.frontier_signals_advanced", IntegerAttr::get(i64, result.signalsAdvanced));
        func->setAttr("pto.insert_sync.frontier_waits_delayed", IntegerAttr::get(i64, result.waitsDelayed));
        func->setAttr(
            "pto.insert_sync.frontier_boundary_handoffs", IntegerAttr::get(i64, result.boundaryHandoffs));
        func->setAttr("pto.insert_sync.frontier_generations", IntegerAttr::get(i64, result.generations));
        func.emitRemark("InsertSync frontier refinement: ")
            << result.removed << " named barriers removed, " << result.guarded
            << " overflow-guarded; " << result.signalsAdvanced << " signals advanced, "
            << result.waitsDelayed << " waits delayed, " << result.boundaryHandoffs
            << " boundary constructions; " << result.reason;
        if (result.internalError) {
            func.emitError(result.reason);
            signalPassFailure();
            return;
        }
        // On a successful clone commit, old SyncIR operation pointers are
        // intentionally stale. No later stage may dereference them.
    }
    auditOutput(func);
  }
};

struct PTOAuditInsertSyncPass : PassWrapper<PTOAuditInsertSyncPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOAuditInsertSyncPass)
    StringRef getArgument() const final { return "pto-audit-insert-sync"; }
    StringRef getDescription() const final
    {
        return "Independently check concrete local synchronization without insertion";
    }
    void runOnOperation() override
    {
        auto function = getOperation();
        if (function.isDeclaration()) {
            return;
        }
        auto result = auditInsertSyncLocal(function);
        function.emitRemark("InsertSync audit: ")
            << stringifyInsertSyncAuditStatus(result.status) << ": " << result.reason;
        if (result.status != InsertSyncAuditStatus::VerifiedLocal) {
            if (result.target) {
                result.target->emitRemark("audit target");
            }
            if (result.source) {
                result.source->emitRemark("audit outstanding source");
            }
            signalPassFailure();
        }
    }
};

// Keep registration in the same linked TU as InsertSync. This is registry
// metadata, not mutable global configuration; all options belong to a pass.
PassRegistration<PTOAuditInsertSyncPass> registerAuditPass;

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInsertSyncPass(const InsertSyncOptions& options)
{
    return std::make_unique<PTOInsertSyncPass>(options);
}

std::unique_ptr<Pass> mlir::pto::createPTOInsertSyncPass() { return createPTOInsertSyncPass(InsertSyncOptions{}); }
