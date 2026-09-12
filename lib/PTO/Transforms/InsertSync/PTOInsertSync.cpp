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
#include "PTO/Transforms/InsertSync/InsertSyncOptions.h"
#include "PTO/Transforms/InsertSync/SyncConstruction.h"
#include "PTO/Transforms/InsertSync/StructuredSyncPlan.h"
#include "PTO/Transforms/InsertSync/SyncAddressAnalysis.h"
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
  explicit PTOInsertSyncPass(const InsertSyncOptions &options) {
    planner = options.planner;
    logicalWorkBudget = options.logicalWorkBudget;
    gmAlias = options.gmAlias;
    hardwareContract = options.hardwareContract;
    structuredPrecision = options.structuredPrecision;
  }
  PTOInsertSyncPass(const PTOInsertSyncPass &other) : PTOInsertSyncBase(other) {
    planner = other.planner;
    logicalWorkBudget = other.logicalWorkBudget;
    gmAlias = other.gmAlias;
    hardwareContract = other.hardwareContract;
    structuredPrecision = other.structuredPrecision;
  }
  Option<std::string> planner{*this, "planner", llvm::cl::init("existing"),
      llvm::cl::desc("Planning engine: existing, logical (reference), logical-or-existing, structured")};
  Option<uint64_t> logicalWorkBudget{*this, "logical-work-budget", llvm::cl::init(kDefaultLogicalSyncWorkBudget),
      llvm::cl::desc("Reference logical engine work allowance (not used by structured)")};
  Option<std::string> gmAlias{*this, "gm-alias", llvm::cl::init(""),
      llvm::cl::desc("Logical constructor GM caller contract")};

  Option<std::string> hardwareContract{*this, "hardware-contract", llvm::cl::init("conservative"),
      llvm::cl::desc("Structured hardware premises: conservative or a2a3-mmad-acc-v1 (experimental)")};
  Option<bool> structuredPrecision{*this, "structured-precision", llvm::cl::init(true),
      llvm::cl::desc("Enable periodic precision (false exercises general conservative composition during migration)")};

  void initializeLogicalMetadata(func::FuncOp func) {
    SmallVector<StringAttr> stale;
    for (auto attr : func->getAttrs())
      if (attr.getName().strref().starts_with("pto.insert_sync.logical_") ||
          attr.getName().strref() == "pto.insert_sync.producer") stale.push_back(attr.getName());
    for (auto name : stale) func->removeAttr(name);
    func->setAttr("pto.insert_sync.requested_planner", StringAttr::get(&getContext(), planner));
    func->setAttr("pto.insert_sync.producer", StringAttr::get(&getContext(), "none"));
  }

  // True means construction completed or strict construction failed. False
  // invokes upstream construction only after authored-event admission.
  // Name the engine that actually ran. Shared status attributes keep their
  // historical `logical_` spelling; only user-facing diagnostics change.
  StringRef engineLabel() const { return planner == "structured" ? "structured" : "logical"; }
  bool tryLogical(func::FuncOp func, bool hasBarrier) {
    if (planner == "existing") return false;
    initializeLogicalMetadata(func);
    auto contract = resolveInsertSyncGMAlias(func, gmAlias);
    if (failed(contract)) { signalPassFailure(); return true; }
    using Result = logical_sync::ConstructionResult;
    Result result;
    if (hasBarrier) result.reason = "explicit synchronization summary not established";
    else if (planner == "structured")
      result = structured_sync::constructStructuredSync(func, *contract,
          hardwareContract == "a2a3-mmad-acc-v1"
              ? structured_sync::HardwareContract::A2A3MmadAccV1
              : structured_sync::HardwareContract::Conservative, structuredPrecision);
    else result = logical_sync::constructLogicalSync(func, *contract, false, logicalWorkBudget);
    StringRef status;
    switch (result.status) {
    case Result::Applied: status = "applied"; break;
    case Result::Unsupported: status = "unsupported"; break;
    case Result::AnalysisLimit: status = "analysis-limit"; break;
    case Result::Unproved: status = "unproved"; break;
    case Result::AllocationFailure: status = "allocation-failure"; break;
    case Result::InternalError: status = "internal-error"; break;
    }
    auto i64 = IntegerType::get(&getContext(), 64);
    func->setAttr("pto.insert_sync.hardware_contract", StringAttr::get(&getContext(), hardwareContract));
    func->setAttr("pto.insert_sync.logical_status", StringAttr::get(&getContext(), status));
    func->setAttr("pto.insert_sync.logical_reason", StringAttr::get(&getContext(), result.reason));
    func->setAttr("pto.insert_sync.logical_work", IntegerAttr::get(i64, result.work));
    func->setAttr("pto.insert_sync.logical_requirements", IntegerAttr::get(i64, result.requirements));
    func->setAttr("pto.insert_sync.logical_streams", IntegerAttr::get(i64, result.handoffs));
    func.emitRemark("InsertSync ") << engineLabel() << " construction: " << status << "; " << result.reason
                                   << "; work=" << result.work;
    if (result.status == Result::Applied) {
      func->setAttr("pto.gm_alias", StringAttr::get(&getContext(),
          *contract == InsertSyncGMAliasMode::MayAlias ? "may-alias" : "assume-disjoint-arguments"));
      func->setAttr("pto.insert_sync.producer", StringAttr::get(&getContext(), planner == "structured" ? "structured" : "logical"));
      return true;
    }
    if (planner == "logical" || planner == "structured" || result.status == Result::InternalError) {
      func.emitError(engineLabel()) << " synchronization construction failed: " << status << "; " << result.reason;
      signalPassFailure(); return true;
    }
    func->setAttr("pto.insert_sync.producer", StringAttr::get(&getContext(), "existing-fallback"));
    return false;
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

    if (planner != "existing" && planner != "logical" && planner != "logical-or-existing" && planner != "structured") {
      func.emitError("InsertSync planner must be existing, logical, logical-or-existing, or structured");
      signalPassFailure(); return;
    }
    if ((hardwareContract != "conservative" && hardwareContract != "a2a3-mmad-acc-v1") ||
        (hardwareContract != "conservative" && planner != "structured")) {
      func.emitError("hardware-contract requires structured and must be conservative or a2a3-mmad-acc-v1");
      signalPassFailure(); return;
    }
    // Neither constructor imports authored event ownership. Classify once,
    // before selection: unsupported logical input is not necessarily safe for
    // legacy insertion. A standalone barrier does not imply a complete manual
    // protocol and retains the existing barrier/fallback policy.
    bool hasExplicitSync = false;
    bool hasBarrier = false;
    func.walk([&](Operation *op) {
      hasBarrier |= isa<pto::BarrierOp>(op);
      if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp,
              pto::WaitFlagDynOp, pto::RecordEventOp, pto::WaitEventOp>(op)) {
        hasExplicitSync = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasExplicitSync) {
      if (planner != "existing") {
        initializeLogicalMetadata(func);
        func->setAttr("pto.insert_sync.logical_status", StringAttr::get(&getContext(), "unsupported"));
        func->setAttr("pto.insert_sync.logical_reason",
                      StringAttr::get(&getContext(), "authored event protocol has no qualified summary"));
        if (planner == "logical" || planner == "structured") {
          func.emitError(engineLabel())
              << " synchronization construction failed: unsupported; authored event protocol has no qualified summary";
          signalPassFailure();
        } else {
          func->setAttr("pto.insert_sync.producer", StringAttr::get(&getContext(), "authored"));
        }
      }
      return;
    }

    auto admission = qualifySyncPhysicalAddresses(func);
    if (admission.status == SyncAddressAdmission::Rejected) {
      func.emitError("InsertSync physical-address admission failed: ") << admission.reason;
      signalPassFailure();
      return;
    }
    if (tryLogical(func, hasBarrier)) return;

    // 0. 数据结构准备
    MemoryDependentAnalyzer memAnalyzer;
    SyncIRs syncIR;
    SyncOperations syncOpsStorage;
    Buffer2MemInfoMap buffer2MemInfoMap;

    // 1. Translator: 构建 SyncIR
    PTOIRTranslator translator(syncIR, memAnalyzer, buffer2MemInfoMap, func, SyncAnalysisMode::NORMALSYNC);
    if (failed(translator.Build())) { signalPassFailure(); return; }

    // 如果 IR 太简单，直接跳过
    if (syncIR.size() <= 1) {
      return;
    }

    dumpInsertSyncPhase("After Translator", syncIR, syncOpsStorage,
                        func.getOperation());

    // 2. Analyzer: 依赖分析与插入逻辑 Sync
    InsertSyncAnalysis analyzer(syncIR, memAnalyzer, syncOpsStorage, func,
                                SyncAnalysisMode::NORMALSYNC);
    analyzer.Run(/*insertBarAllAtLast=*/true);

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

    SyncCodegen codegen(syncIR, func, SyncAnalysisMode::NORMALSYNC);
    codegen.Run();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInsertSyncPass() {
  return std::make_unique<PTOInsertSyncPass>();
}

std::unique_ptr<Pass> mlir::pto::createPTOInsertSyncPass(const InsertSyncOptions &options) {
  return std::make_unique<PTOInsertSyncPass>(options);
}
