// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/SyncPhysicalFacts.h"
#include "PTO/Transforms/InsertSync/SyncEffectCoverage.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include <limits>
#include <optional>
using namespace mlir;
using namespace mlir::pto;
namespace {
std::optional<unsigned> laneFor(PIPE pipe, bool cube)
{
    if (
        pipe == PIPE::PIPE_MTE2) {
        return 0;
    }
    if (
        cube) {
        if (
            pipe == PIPE::PIPE_MTE1) {
            return 1;
        }
        if (
            pipe == PIPE::PIPE_M) {
            return 2;
        }
        if (
            pipe == PIPE::PIPE_FIX) {
            return 3;
        }
    } else {
        if (
            pipe == PIPE::PIPE_V) {
            return 1;
        }
        if (
            pipe == PIPE::PIPE_MTE3) {
            return 2;
        }
    }
    return std::nullopt;
}

// Validate the existing translator's payload mapping, not a second opcode
// semantics table. Explicit synchronization is interpreted separately below.
// This also lets the development pass inspect manually synchronized fixtures
// without sending flag-resource effects through the payload-only coverage gate.
bool mappedPayloadEffects(Operation* op, const CompoundInstanceElement* compound)
{
    // Some op interfaces omit by-value scalar operands from memory effects.
    // Their prerequisites still have to be synchronous scalar computations.
    for (Value value : op->getOperands())
        if (isa<IntegerType, IndexType, FloatType>(value.getType()) &&
            !isInsertSyncScalarPrerequisite(value)) return false;
    auto interface = dyn_cast<MemoryEffectOpInterface>(op);
    if (
        !interface) {
        return false;
    }
    SmallVector<MemoryEffects::EffectInstance> effects;
    interface.getEffects(effects);
    for (
        const auto& effect : effects) {
        Value value = effect.getValue();
        bool write = isa<MemoryEffects::Write>(effect.getEffect());
        if (
            (!write && !isa<MemoryEffects::Read>(effect.getEffect())) || !value) {
            return false;
        }
        if (
            !isa<TileBufType, TensorViewType, PartitionTensorViewType, PtrType, BaseMemRefType>(value.getType())) {
            // By-value scalar operands have no payload region. An asynchronous
            // producer of that value is still visited and rejected by this adapter
            // if it is not one of the modeled physical phases.
            if (
                write || !isa<IntegerType, IndexType, FloatType>(value.getType())) {
                return false;
            }
            continue;
        }
        const auto& entries = write ? compound->defVec : compound->useVec;
        if (
            !llvm::any_of(entries, [&](const BaseMemInfo* entry) { return entry && entry->baseBuffer == value; })) {
            return false;
        }
    }
    return true;
}
bool memoryType(Type t) {
  return isa<TileBufType, MultiTileBufType, PtrType, TensorViewType,
             PartitionTensorViewType, BaseMemRefType>(t);
}
class Importer {
  func::FuncOp function;
  uint64_t budget;
  unsigned fragments = 0;
  llvm::DenseMap<Operation *, const CompoundInstanceElement *> compounds;
  SyncPhysicalFacts result;
  bool fail(StringRef reason, bool limit = false) {
    result.reason = reason.str();
    result.status = limit ? SyncPhysicalFacts::Status::AnalysisLimit : SyncPhysicalFacts::Status::Unsupported;
    return false;
  }
  bool access(const BaseMemInfo *info) {
    if (!info || info->scope == AddressSpace::Zero || info->aliasesUnknownRange) return false;
    fragments += info->baseAddresses.size();
    if (fragments > 2048) return fail("physical fragment budget", true);
    if (info->scope == AddressSpace::GM) return true;
    if (!info->hasKnownPhysicalAddresses || !info->allocateSize ||
        info->baseAddresses.empty() || info->baseAddresses.size() > 64) return false;
    return llvm::all_of(info->baseAddresses, [&](uint64_t address) {
      return address <= std::numeric_limits<uint64_t>::max() - info->allocateSize;
    });
  }
  bool region(Region &r, unsigned depth) {
    if (depth > 24) return fail("region nesting budget", true);
    if (!llvm::hasSingleElement(r)) return fail("unsupported multi-block region");
    for (Operation &op : r.front()) {
      if (++result.work > budget || result.work > 8192) return fail("IR visitation budget", true);
      if (isa<scf::ForOp, scf::IfOp, SectionCubeOp, SectionVectorOp>(op)) {
        for (Region &child : op.getRegions())
          if (!child.empty() && !region(child, depth + 1)) return false;
        continue;
      }
      if (op.getNumRegions() || getSyncMacroModel(&op)) return fail("unmodeled physical region or macro");
      if (isa<SetFlagOp, WaitFlagOp, BarrierOp, RecordEventOp, WaitEventOp>(op))
        return fail("explicit synchronization is not an unsynchronized input");
      if (auto alloc = dyn_cast<AllocTileOp>(op)) {
        if (cast<TileBufType>(alloc.getResult().getType()).hasDynamicValid() &&
            (!isInsertSyncScalarPrerequisite(alloc.getValidRow()) ||
             !isInsertSyncScalarPrerequisite(alloc.getValidCol())))
          return fail("unqualified descriptor allocation prerequisite");
        continue;
      }
      if (auto update = dyn_cast<SetValidShapeOp>(op)) {
        if (!update.getSource().getDefiningOp<AllocTileOp>() ||
            !isInsertSyncScalarPrerequisite(update.getValidRow()) ||
            !isInsertSyncScalarPrerequisite(update.getValidCol()))
          return fail("unqualified descriptor handle or scalar prerequisite");
        continue;
      }
      if (auto read = dyn_cast<GetValidShapeOp>(op)) {
        if (!read.getSource().getDefiningOp<AllocTileOp>()) return fail("unqualified descriptor read handle");
        continue;
      }
      if (auto physical = dyn_cast<OpPipeInterface>(op)) {
        bool qualified = isa<TLoadOp, TStoreOp, TAbsOp, TAddOp, TExtractOp, TMovOp,
          TMatmulOp, TMatmulAccOp, TSort32Op, TMrgSortOp, TGatherOp,
          TAddSOp, TMulSOp, TMulOp, TSubOp, TExpOp, TExpandsOp, TCvtOp, TRowSumOp,
          TSqrtOp, TRecipOp, TNegOp, TMaxOp, TRowMaxOp, TRowExpandDivOp,
          TRowExpandMulOp, TRowExpandSubOp, TColExpandMulOp>(op) ||
          (isa<TFillPadOp>(op) && physical.getPipe() == PIPE::PIPE_V);
        auto found = compounds.find(&op);
        if (!qualified || !laneFor(physical.getPipe(), result.cube) || found == compounds.end())
          return fail("physical phase has no qualified adapter");
        auto *phase = found->second;
        if (phase->kPipeValue != static_cast<PipelineType>(physical.getPipe()) ||
            !mappedPayloadEffects(&op, phase)) return fail("translated payload effect is incomplete");
        unsigned count = phase->useVec.size() + phase->defVec.size();
        if (!count) return fail("empty physical summary");
        if (count > 16 || result.phases.size() >= 256) return fail("phase/access budget", true);
        for (auto *info : phase->useVec) if (!access(info)) {
          if (result.reason.empty()) fail("unproved read footprint");
          return false;
        }
        for (auto *info : phase->defVec) if (!access(info)) {
          if (result.reason.empty()) fail("unproved write footprint");
          return false;
        }
        result.phases.push_back(phase);
      } else if (!isa<AllocMultiTileOp>(op) && !isMemoryEffectFree(&op)) {
        return fail("unmodeled non-payload effect");
      }
    }
    return true;
  }
public:
  Importer(func::FuncOp f, uint64_t b) : function(f), budget(b) {}
  SyncPhysicalFacts run(const SyncIRs &ir) {
    auto module = function->getParentOfType<ModuleOp>();
    auto arch = module ? module->getAttrOfType<StringAttr>("pto.target_arch") : StringAttr();
    auto core = function->getAttrOfType<FunctionKernelKindAttr>("pto.kernel_kind");
    bool explicitCore = core && (core.getKernelKind() == FunctionKernelKind::Cube ||
                                core.getKernelKind() == FunctionKernelKind::Vector);
    result.cube = explicitCore && core.getKernelKind() == FunctionKernelKind::Cube;
    Operation *section = nullptr;
    bool invalid = false;
    function.walk([&](Operation *op) {
      if (!isa<SectionCubeOp, SectionVectorOp>(op)) return;
      if (section || op->getParentOp() != function.getOperation()) { invalid = true; return; }
      section = op;
      bool cube = isa<SectionCubeOp>(op);
      invalid |= explicitCore && result.cube != cube;
      result.cube = cube;
    });
    if (section) function.walk([&](Operation *op) {
      if (op == function || op == section || section->isAncestor(op)) return;
      if (isa<OpPipeInterface, SetFlagOp, WaitFlagOp, BarrierOp>(op)) invalid = true;
    });
    if (!arch || (arch.getValue() != "a2" && arch.getValue() != "a3") ||
        (!explicitCore && !section) || invalid) {
      fail("requires one qualified A2/A3 physical context"); return result;
    }
    result.lifetimeScope = section ? section : function.getOperation();
    for (const auto &element : ir)
      if (auto *phase = dyn_cast<CompoundInstanceElement>(element.get()))
        if (!phase->elementOp || !compounds.try_emplace(phase->elementOp, phase).second) {
          fail("multi-phase operation requires its complete semantic adapter"); return result;
        }
    if (region(function.getBody(), 0)) result.status = SyncPhysicalFacts::Status::Complete;
    return result;
  }
};
} // namespace

SyncPhysicalFacts mlir::pto::importSyncPhysicalFacts(func::FuncOp f, const SyncIRs &ir, uint64_t budget) {
  return Importer(f, budget).run(ir);
}
bool mlir::pto::supportsLogicalSyncTranslation(func::FuncOp function) {
  bool supported = true;
  function.walk([&](Operation *op) {
    if (isa<scf::WhileOp>(op)) supported = false;
    if (auto loop = dyn_cast<scf::ForOp>(op))
      for (Value value : loop.getInitArgs()) if (memoryType(value.getType())) supported = false;
  });
  return supported;
}
bool mlir::pto::logicalSyncMayAlias(const BaseMemInfo *a, const BaseMemInfo *b,
                                  func::FuncOp function, InsertSyncGMAliasMode gm) {
  if (!a || !b) return true;
  if (a->scope != b->scope) return false;
  if (a->aliasesUnknownRange || b->aliasesUnknownRange) return true;
  if (a->scope == AddressSpace::GM && a->rootBuffer != b->rootBuffer) {
    return gm != InsertSyncGMAliasMode::DisjointArguments ||
           !disjointInsertSyncGMRoots(function, a->rootBuffer, b->rootBuffer);
  }
  if (a->scope != AddressSpace::GM &&
      (!a->hasKnownPhysicalAddresses || !b->hasKnownPhysicalAddresses)) return true;
  if (a->baseAddresses.empty() || b->baseAddresses.empty() || !a->allocateSize || !b->allocateSize) return true;
  for (uint64_t x : a->baseAddresses) for (uint64_t y : b->baseAddresses) {
    if (x > std::numeric_limits<uint64_t>::max() - a->allocateSize ||
        y > std::numeric_limits<uint64_t>::max() - b->allocateSize) return true;
    if (x < y + b->allocateSize && y < x + a->allocateSize) return true;
  }
  return false;
}
