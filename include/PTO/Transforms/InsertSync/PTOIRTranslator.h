// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_PTOIRTRANSLATOR_H
#define MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_PTOIRTRANSLATOR_H
 
#include "PTO/IR/PTO.h"
#include "PTO/IR/SyncProtocolModel.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/Transforms/InsertSync/SyncSlotMapping.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
 
namespace mlir {
namespace pto {
 
// Shared semantic accounting, independent of the synchronization constructor.
// Phase pointers borrow the translator output and expire with that output.
// Ordinary phases use production's OpPipeInterface + MemoryEffectOpInterface
// contract. Other mechanisms are represented explicitly rather than silently
// treated as ordinary memory accesses.
struct SyncSemanticRecord {
  enum Kind {
    Ordinary, Storage, Descriptor, Control, Pure, Macro, Authored, Visibility, Protocol,
    Unmodeled
  };
  Operation *operation = nullptr;
  Kind kind = Unmodeled;
  SmallVector<CompoundInstanceElement *> phases;
  std::string gap;
  std::optional<SyncMacroModel> macro;
  std::optional<SyncProtocolModel> protocol;
};
struct SyncSemanticReport {
  SmallVector<SyncSemanticRecord, 0> operations;
  bool complete() const {
    return llvm::all_of(operations, [](const auto &record) {
      return record.gap.empty();
    });
  }
};

class PTOIRTranslator {
public:
  PTOIRTranslator(SyncIRs &syncIR,
                  MemoryDependentAnalyzer &memDepAnalyzer,
                  Buffer2MemInfoMap &buffer2MemInfoMap,
                  func::FuncOp func,
                  SyncAnalysisMode syncAnalysisMode)
    : func_(func), 
      index(0),
      syncIR_(syncIR), 
      buffer2MemInfoMap_(buffer2MemInfoMap),
      memAnalyzer_(memDepAnalyzer),
      mode_(syncAnalysisMode) {
    (void)memAnalyzer_;
    (void)mode_;
  };
 
  // 核心入口：执行 IR 分析和转换
  // On failure, both output collections are empty; callers must stop.
  LogicalResult Build();

  // Recompute after an analysis (such as structured-origin closure) refreshes
  // physical effects. This is read-only and never inserts synchronization.
  SyncSemanticReport describeSemantics() const;
 
  // 获取生成的 SyncIR (指令序列)
  SyncIRs &getSyncIR() { return syncIR_; }
 
  // 获取 Buffer 分析结果 (别名映射)
  Buffer2MemInfoMap &getBuffer2MemInfoMap() { return buffer2MemInfoMap_; }
 
  // 打印调试信息 (Buffer Map 和 SyncIR)
  void print();
 
private:
  func::FuncOp func_;
  bool translated_ = false;
  // Unseeded scalar facts for this immutable import only. Never share these
  // with the per-residue maps used to qualify carried slots.
  SyncSlotMapping::ConstantCache constantAddresses_;
  std::optional<uint64_t> getKnownPhysicalAddress(Value value);
  unsigned index; // 当前 SyncIR 节点的索引计数器
  
  // 核心数据结构 (定义在 SyncCommon.h 中)
  SyncIRs &syncIR_;
  Buffer2MemInfoMap &buffer2MemInfoMap_;
  MemoryDependentAnalyzer &memAnalyzer_;
  SyncAnalysisMode mode_;
 
  // --- 递归遍历逻辑 ---
  LogicalResult RecursionIR(Region *region);
  // RecursionIR 的按类别分发器：返回 nullopt 表示 op 不属于该类别，
  // 需继续尝试后续类别；返回 WalkResult 表示已匹配处理完毕。
  std::optional<WalkResult> dispatchAllocOp(Operation *op);
  std::optional<WalkResult> dispatchAliasViewOp(Operation *op);
  std::optional<WalkResult> dispatchControlAndComputeOp(Operation *op);
 
  // --- 内存/Alias 分析 ---
  void UpdateKernelArgMemInfo();
  LogicalResult UpdateAllocTileOpMemInfo(pto::AllocTileOp op);
  LogicalResult UpdateAllocMultiTileOpMemInfo(pto::AllocMultiTileOp op);
  LogicalResult UpdateDeclareTileOpMemInfo(pto::DeclareTileOp op);
  LogicalResult UpdateDeclareGlobalOpMemInfo(pto::DeclareGlobalOp op);
  
  // 处理 View/Alias (MakeTensorView, Subview, Mov)
  void UpdateAliasBufferInfo(Value result, Value source);
  void UpdateConservativeAliasBufferInfo(Value result, Value source);
  void UpdateTileSubViewAliasBufferInfo(pto::SubViewOp op);
  LogicalResult UpdateIntegerToPtrCastMemInfo(pto::CastPtrOp op);
  void UpdateMultiTileGetAliasBufferInfo(pto::MultiTileGetOp op);
  void UpdateSlotSelectedAliasBufferInfo(Value result, Value source,
                                         Value slot);
 
  // --- 控制流处理 (SCF) ---
  LogicalResult UpdateForOpInfo(scf::ForOp forOp);
  LogicalResult UpdateWhileOpInfo(scf::WhileOp whileOp);
  LogicalResult UpdateIfOpInfo(scf::IfOp ifOp);
  void UpdateYieldOpInfo(scf::YieldOp yieldOp);

  // --- 核心：处理计算/搬运指令 (生成 Compound 节点) ---
  void UpdatePTOOpInfo(Operation *op);
  void UpdatePTOOpInfoWithPipeline(Operation *op, PipelineType pipe,
                                   bool skipIfNoMemInfo = false);
  void UpdateMacroOpInfo(Operation *op);
  void MakeMacroCompound(Operation *op, PipelineType pipe, ValueRange defValues,
                         ValueRange useValues, int macroPhaseId);
  void UpdateHelperCallInfo(func::CallOp callOp);

  // --- 辅助函数 ---

  // 获取 PTO Op 对应的硬件流水线类型
  PipelineType getOpPipeline(Operation *op) const;

  // 根据 Values 填充 Def/Use 列表
  void UpdateDefUseVec(ValueRange values, SmallVector<const BaseMemInfo *> &vec);

  // 调试辅助
  std::string getPipelineName(PipelineType pipe) const;
  void printMemInfoList(llvm::raw_ostream &os,
                        const SmallVector<const BaseMemInfo *> &list,
                        AsmState &state) const;
};
 
} // namespace pto
} // namespace mlir
 
#endif // MLIR_DIALECT_PTO_TRANSFORMS_INJECTSYNC_PTOIRTRANSLATOR_H
