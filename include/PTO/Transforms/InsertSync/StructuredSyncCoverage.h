// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCCOVERAGE_H
#define PTO_TRANSFORMS_INSERTSYNC_STRUCTUREDSYNCCOVERAGE_H

#include "PTO/Transforms/InsertSync/SyncPhysicalFacts.h"
#include "PTO/Transforms/InsertSync/SyncEffectCoverage.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include <limits>
#include <map>

namespace mlir::pto {
namespace structured_sync_coverage_detail {
using namespace mlir;
using namespace mlir::pto;


inline bool memoryType(Type type) {
  return isa<TileBufType, MultiTileBufType, TensorViewType,
             PartitionTensorViewType, PtrType, BaseMemRefType>(type);
}

inline bool supportedLane(PIPE pipe, bool cube) {
  // PIPE_S is synchronous on both modeled targets; target modeling, rather
  // than an operation-name list, supplies that property.
  if (pipe == PIPE::PIPE_S || pipe == PIPE::PIPE_MTE2 ||
      pipe == PIPE::PIPE_MTE3)
    return true;
  if (cube)
    return pipe == PIPE::PIPE_MTE1 || pipe == PIPE::PIPE_M ||
           pipe == PIPE::PIPE_FIX;
  return pipe == PIPE::PIPE_V;
}

inline std::string diagnostic(StringRef kind, Operation *op, StringRef detail = {}) {
  std::string result = kind.str();
  if (op) {
    result += ":";
    result += op->getName().getStringRef().str();
  }
  if (!detail.empty()) {
    result += ":";
    result += detail.str();
  }
  return result;
}

inline bool mappedPayloadEffects(Operation *op,
                          const CompoundInstanceElement *compound) {
  if (!compound || compound->elementOp != op)
    return false;
  for (Value value : op->getOperands())
    if (isa<IntegerType, IndexType, FloatType>(value.getType()) &&
        !isInsertSyncScalarPrerequisite(value))
      return false;
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  for (const auto &effect : effects) {
    Value value = effect.getValue();
    bool read = isa<MemoryEffects::Read>(effect.getEffect());
    bool write = isa<MemoryEffects::Write>(effect.getEffect());
    if ((!read && !write) || !value)
      return false;
    if (!memoryType(value.getType())) {
      // TMrgSort format2's `executed` operand is an in-place vector-register
      // result on PIPE_V. Admit it only when this operation is its sole use;
      // observing the mutation needs an explicit register-result dependency.
      auto merge = dyn_cast<TMrgSortOp>(op);
      if (write && merge &&
          (merge.isFormat2() || merge.isFormat2WithoutTmp()) &&
          merge.getExcuted() == value && value.hasOneUse()) {
        auto type = dyn_cast<VectorType>(value.getType());
        if (type && type.getRank() == 1 && type.getNumElements() == 4 &&
            type.getElementType().isInteger(16))
          continue;
      }
      if (write || !isa<IntegerType, IndexType, FloatType>(value.getType()) ||
          !isInsertSyncScalarPrerequisite(value))
        return false;
      continue;
    }
    const auto &entries = write ? compound->defVec : compound->useVec;
    if (!llvm::any_of(entries, [&](const BaseMemInfo *entry) {
          return entry && entry->baseBuffer == value;
        }))
      return false;
  }
  return true;
}

inline std::optional<PipelineType> helperPipe(func::FuncOp callee) {
  if (!callee)
    return {};
  if (auto role = callee->getAttrOfType<StringAttr>(
          "pto.ptodsl.subkernel_helper"))
    return llvm::StringSwitch<std::optional<PipelineType>>(role.getValue())
        .Case("cube", PipelineType::PIPE_M)
        .Case("simd", PipelineType::PIPE_V)
        .Default(std::nullopt);
  if (auto kind = callee->getAttrOfType<StringAttr>("pto.tileop.kind"))
    return llvm::StringSwitch<std::optional<PipelineType>>(kind.getValue())
        .Case("cube", PipelineType::PIPE_M)
        .Case("vector", PipelineType::PIPE_V)
        .Default(std::nullopt);
  return {};
}

inline bool visiblePureHelper(func::FuncOp callee) {
  if (!callee || callee.isDeclaration())
    return false;
  bool pure = true;
  callee.getBody().walk([&](Operation *nested) {
    if (!isa<func::ReturnOp, scf::YieldOp>(nested) &&
        (isa<OpPipeInterface>(nested) || !isMemoryEffectFree(nested)))
      pure = false;
  });
  return pure;
}

inline bool helperEffectsMatch(func::CallOp call, func::FuncOp callee,
                        const CompoundInstanceElement *phase) {
  auto effects = callee
      ? callee->getAttrOfType<ArrayAttr>("pto.tileop.effects")
      : ArrayAttr();
  if (!effects || effects.size() != call.getNumOperands() || !phase)
    return false;
  for (auto [operand, attribute] : llvm::zip(call.getOperands(), effects)) {
    auto text = dyn_cast<StringAttr>(attribute);
    if (!text)
      return false;
    StringRef effect = text.getValue();
    if (effect != "none" && effect != "read" && effect != "write" &&
        effect != "readwrite")
      return false;
    if (!memoryType(operand.getType())) {
      if (effect != "none")
        return false;
      continue;
    }
    auto represented = [&](const auto &entries) {
      return llvm::any_of(entries, [&](const BaseMemInfo *info) {
        return info && info->baseBuffer == operand;
      });
    };
    if ((effect == "read" || effect == "readwrite") &&
        !represented(phase->useVec))
      return false;
    if ((effect == "write" || effect == "readwrite") &&
        !represented(phase->defVec))
      return false;
  }
  return true;
}

class Importer {
  func::FuncOp function;
  llvm::DenseMap<Operation *, SmallVector<const CompoundInstanceElement *, 1>>
      compounds;
  SyncPhysicalFacts result;
  uint64_t fragments = 0;
  bool compositional = false;
  bool emitted = false;
  bool authoredWholeFunction = false;

  bool fail(StringRef kind, Operation *op = nullptr, StringRef detail = {}) {
    result.status = SyncPhysicalFacts::Status::Unsupported;
    result.reason = diagnostic(kind, op, detail);
    return false;
  }

  bool qualifyAccess(const BaseMemInfo *info, Operation *op, bool write) {
    // Complete space/effect information suffices for conservative composition.
    // Unknown intervals become a whole-space cell, never an absent access.
    if (compositional && info && info->scope != AddressSpace::Zero)
      return true;
    if (!info || info->scope == AddressSpace::Zero || info->aliasesUnknownRange)
      return fail(write ? "unqualified-write-footprint"
                        : "unqualified-read-footprint",
                  op, "unknown-or-zero-range");
    fragments += info->baseAddresses.size();
    if (fragments > 8192) {
      result.status = SyncPhysicalFacts::Status::AnalysisLimit;
      result.reason = diagnostic("physical-fragment-limit", op);
      return false;
    }
    if (info->scope == AddressSpace::GM)
      return true;
    if (!info->hasKnownPhysicalAddresses || !info->allocateSize ||
        info->baseAddresses.empty())
      return fail(write ? "unqualified-write-footprint"
                        : "unqualified-read-footprint",
                  op, "local-address-unavailable");
    for (uint64_t base : info->baseAddresses)
      if (base > std::numeric_limits<uint64_t>::max() - info->allocateSize)
        return fail(write ? "unqualified-write-footprint"
                          : "unqualified-read-footprint",
                    op, "interval-overflow");
    return true;
  }

  bool addPhase(Operation *op, PipelineType pipe,
                const CompoundInstanceElement *phase,
                bool useDeclaredEffects) {
    if (!phase)
      return fail("missing-translated-phase", op);
    if (phase->kPipeValue != pipe)
      return fail("translated-pipeline-mismatch", op);
    if (useDeclaredEffects && !mappedPayloadEffects(op, phase))
      return fail("incomplete-translated-effects", op);
    if (phase->useVec.empty() && phase->defVec.empty())
      return fail("empty-physical-summary", op);
    for (auto *info : phase->useVec)
      if (!qualifyAccess(info, op, false))
        return false;
    for (auto *info : phase->defVec)
      if (!qualifyAccess(info, op, true))
        return false;
    result.phases.push_back(phase);
    return true;
  }

  bool allocation(AllocTileOp alloc) {
    auto type = cast<TileBufType>(alloc.getResult().getType());
    return !type.hasDynamicValid() ||
           (isInsertSyncScalarPrerequisite(alloc.getValidRow()) &&
            isInsertSyncScalarPrerequisite(alloc.getValidCol())) ||
           fail("invalid-descriptor-contract", alloc,
                "allocation-valid-shape");
  }

  bool allocation(AllocMultiTileOp alloc) {
    auto layout = getPTOStaticMultiTileSlotLayout(
        alloc.getResult().getType().getSlotType());
    return (succeeded(layout) && layout->footprintBytes) ||
           fail("invalid-descriptor-contract", alloc, "multi-tile-layout");
  }

  bool update(SetValidShapeOp update) {
    return (update.getSource().getDefiningOp<AllocTileOp>() &&
            isInsertSyncScalarPrerequisite(update.getValidRow()) &&
            isInsertSyncScalarPrerequisite(update.getValidCol())) ||
           fail("invalid-descriptor-contract", update,
                "asynchronous-valid-shape-source");
  }

  bool read(GetValidShapeOp read) {
    return bool(read.getSource().getDefiningOp<AllocTileOp>()) ||
           fail("invalid-descriptor-contract", read,
                "descriptor-read-handle");
  }

  // A physical section narrows lifetime, not semantic admission. Audit the
  // surrounding function so captured descriptors cannot hide asynchronous
  // producers, configuration resources, physical phases, or authored sync.
  bool auditOutside(Operation *section) {
    auto walked = function.walk([&](Operation *op) {
      const bool isSelectedSection = op == function.getOperation() ||
                                     op == section ||
                                     section->isAncestor(op);
      if (isSelectedSection) {
        return WalkResult::advance();
      }
      ++result.work;
      // Test-only whole-function reconstruction owns these exact commands.
      // Dynamic/frontend events and outside physical effects still decline.
      if (authoredWholeFunction && isa<SetFlagOp, WaitFlagOp, BarrierOp>(op))
        return WalkResult::advance();
      if (isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, BarrierOp,
              RecordEventOp, WaitEventOp>(op)) {
        fail("explicit-synchronization-outside-physical-section", op);
        return WalkResult::interrupt();
      }
      const auto semantics = getSyncOperationSemantics(op);
      const bool unsupportedResource =
          semantics.kind == SyncOperationSemanticKind::Macro ||
          semantics.kind == SyncOperationSemanticKind::ImplicitResource ||
          semantics.kind == SyncOperationSemanticKind::TypedEffect;
      if (unsupportedResource) {
        fail("unsupported-resource-outside-physical-section", op);
        return WalkResult::interrupt();
      }
      if (auto alloc = dyn_cast<AllocTileOp>(op)) {
        return allocation(alloc) ? WalkResult::advance()
                                 : WalkResult::interrupt();
      }
      if (auto alloc = dyn_cast<AllocMultiTileOp>(op)) {
        return allocation(alloc) ? WalkResult::advance()
                                 : WalkResult::interrupt();
      }
      if (auto descriptor = dyn_cast<SetValidShapeOp>(op)) {
        return update(descriptor) ? WalkResult::advance()
                                  : WalkResult::interrupt();
      }
      if (auto descriptor = dyn_cast<GetValidShapeOp>(op)) {
        return read(descriptor) ? WalkResult::advance()
                                : WalkResult::interrupt();
      }
      const bool physicalOperation =
          compounds.count(op) || isa<OpPipeInterface>(op);
      if (physicalOperation) {
        fail("physical-operation-outside-selected-section", op);
        return WalkResult::interrupt();
      }
      if (auto call = dyn_cast<func::CallOp>(op)) {
        if (!visiblePureHelper(SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                call, call.getCalleeAttr()))) {
          fail("unsupported-helper-outside-physical-section", call);
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      const bool supportedContainer =
          op->hasTrait<OpTrait::IsTerminator>() ||
          op->hasTrait<OpTrait::HasRecursiveMemoryEffects>() ||
          isMemoryEffectFree(op);
      if (supportedContainer) {
        return WalkResult::advance();
      }
      fail("unsupported-effect-outside-physical-section", op);
      return WalkResult::interrupt();
    });
    return !walked.wasInterrupted();
  }

  bool helper(func::CallOp call) {
    auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, call.getCalleeAttr());
    auto found = compounds.find(call.getOperation());
    if (found == compounds.end()) {
      if (visiblePureHelper(callee))
        return true;
      return fail("unsupported-helper-contract", call,
                  "missing-translated-phase-or-pure-body");
    }
    if (found->second.size() != 1)
      return fail("multi-phase-needs-endpoints", call);
    auto pipe = helperPipe(callee);
    if (!pipe)
      return fail("unsupported-helper-contract", call,
                  "missing-helper-pipeline");
    bool cubePipe = *pipe == PipelineType::PIPE_M;
    if (!supportedLane(static_cast<PIPE>(*pipe), cubePipe))
      return fail("unsupported-lane", call);
    if (!helperEffectsMatch(call, callee, found->second.front()))
      return fail("unsupported-helper-contract", call,
                  "effect-contract-does-not-match-translation");
    return addPhase(call, *pipe, found->second.front(), false);
  }

  bool macro(Operation *op, const SyncMacroModel &model) {
    // Composition keeps a macro atomic but imports every lowering-owned phase
    // and explicit internal completion transfer. No internal insertion point
    // or completion fact is inferred from the operation name or hidden keys.
    TCoreType core = result.cube ? TCoreType::CUBE : TCoreType::VECTOR;
    if (!compositional || model.phases.empty() ||
        !supportsSyncMacroCore(model, core))
      return fail("multi-phase-needs-endpoints", op);
    auto found = compounds.find(op);
    if (found == compounds.end() || found->second.size() != model.phases.size())
      return fail("missing-translated-phase", op);
    SmallVector<const CompoundInstanceElement *, 2> translated(
        found->second.begin(), found->second.end());
    llvm::sort(translated, [](const auto *a, const auto *b) {
      return a->macroOpInstanceId < b->macroOpInstanceId;
    });
    for (unsigned i = 0; i < model.phases.size(); ++i) {
      if (model.phases[i].phaseId != i ||
          !supportedLane(static_cast<PIPE>(model.phases[i].pipe),
                         result.cube) ||
          translated[i]->macroOpInstanceId != int(i) ||
          !addPhase(op, model.phases[i].pipe, translated[i], false))
        return false;
    }
    for (const auto &transfer : model.completionTransfers)
      if (transfer.sourcePhaseId >= model.phases.size() ||
          transfer.targetPhaseId >= model.phases.size() ||
          transfer.sourcePhaseId >= transfer.targetPhaseId)
        return fail("unsupported-macro-contract", op,
                    "invalid-ordered-completion-transfer");
    return true;
  }

  // The parameter must not be named `region`: it would shadow this member and
  // make the recursive call below resolve to the Region itself.
  bool region(Region &body, unsigned depth) {
    if (!compositional && depth > 32)
      return fail("region-depth-limit");
    if (!llvm::hasSingleElement(body))
      return fail("unsupported-multi-block-region");
    for (Operation &op : body.front()) {
      ++result.work;
      if (isa<scf::ForOp, scf::IfOp, SectionCubeOp, SectionVectorOp>(op) ||
          (compositional && isa<scf::WhileOp>(op))) {
        for (Region &child : op.getRegions())
          if (!child.empty() && !region(child, depth + 1))
            return false;
        continue;
      }
      if (isa<scf::WhileOp>(op))
        return fail("unsupported-control-region", &op, "scf.while");
      if (compositional && !emitted &&
          isa<BarrierOp, CmoCacheInvalidOp, FenceBarrierAllOp, TNotifyOp,
              TWaitOp>(op))
        continue; // imported as immutable state transitions by the composer
      if (emitted &&
          isa<SetFlagOp, WaitFlagOp, BarrierOp, CmoCacheInvalidOp,
              FenceBarrierAllOp, TNotifyOp, TWaitOp>(op))
        continue; // independently parsed and verified from the actual IR
      if (isa<SetFlagOp, WaitFlagOp, BarrierOp, RecordEventOp, WaitEventOp>(op))
        return fail("explicit-synchronization-input", &op);
      auto semantics = getSyncOperationSemantics(&op);
      if (semantics.kind == SyncOperationSemanticKind::Macro) {
        if (!macro(&op, *semantics.macro))
          return false;
        continue;
      }
      if (op.getNumRegions())
        return fail("unsupported-control-region", &op);

      if (auto alloc = dyn_cast<AllocTileOp>(op)) {
        if (!allocation(alloc)) {
          return false;
        }
        continue;
      }
      if (auto alloc = dyn_cast<AllocMultiTileOp>(op)) {
        if (!allocation(alloc)) {
          return false;
        }
        continue;
      }
      if (auto update = dyn_cast<SetValidShapeOp>(op)) {
        if (!this->update(update)) {
          return false;
        }
        continue;
      }
      if (auto read = dyn_cast<GetValidShapeOp>(op)) {
        if (!this->read(read)) {
          return false;
        }
        continue;
      }
      if (auto call = dyn_cast<func::CallOp>(op)) {
        if (!helper(call))
          return false;
        continue;
      }
      // TReshape is an SSA view: it aliases its source storage and performs no
      // physical work. PTOIRTranslator has already forwarded that alias into
      // the phase footprints. Its PIPE_S interface is an ISA modeling detail,
      // not a scalar payload that needs completion or an event protocol.
      if (isa<TReshapeOp>(op))
        continue;
      if (auto physical = dyn_cast<OpPipeInterface>(op)) {
        PIPE p = physical.getPipe();
        if (!supportedLane(p, result.cube))
          return fail("unsupported-lane", &op);
        if (semantics.kind == SyncOperationSemanticKind::ImplicitResource ||
            semantics.kind == SyncOperationSemanticKind::TypedEffect)
          return fail("unsupported-implicit-resource", &op);
        if (semantics.kind != SyncOperationSemanticKind::SinglePhaseOrdinary) {
          return fail("missing-positive-single-phase-contract", &op);
        }
        if (semantics.pipe != static_cast<PipelineType>(p))
          return fail("translated-pipeline-mismatch", &op);
        const bool unsupportedFillPad =
            isa<TFillPadOp>(op) && p != PIPE::PIPE_V;
        if (unsupportedFillPad) {
          return fail("unsupported-single-phase-variant", &op,
                      "tfillpad-non-vector");
        }
        auto found = compounds.find(&op);
        if (found == compounds.end())
          return fail("missing-translated-phase", &op);
        if (found->second.size() != 1)
          return fail("multi-phase-needs-endpoints", &op);
        if (!addPhase(&op, static_cast<PipelineType>(p),
                      found->second.front(), true))
          return false;
        continue;
      }
      if (!isMemoryEffectFree(&op))
        return fail("unsupported-implicit-resource", &op);
    }
    return true;
  }

public:
  Importer(func::FuncOp f, const SyncIRs &ir, bool composition = false,
           bool actual = false, bool authored = false)
      : function(f), compositional(composition), emitted(actual), authoredWholeFunction(authored) {
    for (const auto &element : ir)
      if (auto *phase = dyn_cast<CompoundInstanceElement>(element.get()))
        if (phase->elementOp)
          compounds[phase->elementOp].push_back(phase);
  }

  SyncPhysicalFacts run() {
    auto module = function->getParentOfType<ModuleOp>();
    auto arch = module
        ? module->getAttrOfType<StringAttr>("pto.target_arch")
        : StringAttr();
    auto core = function->getAttrOfType<FunctionKernelKindAttr>(
        "pto.kernel_kind");
    bool explicitCore = core &&
        (core.getKernelKind() == FunctionKernelKind::Cube ||
         core.getKernelKind() == FunctionKernelKind::Vector);
    result.cube = explicitCore &&
                  core.getKernelKind() == FunctionKernelKind::Cube;
    Operation *section = nullptr;
    bool invalidSection = false;
    function.walk([&](Operation *op) {
      if (!isa<SectionCubeOp, SectionVectorOp>(op))
        return;
      if (section || op->getParentOp() != function.getOperation()) {
        invalidSection = true;
        return;
      }
      section = op;
      bool cube = isa<SectionCubeOp>(op);
      invalidSection |= explicitCore && result.cube != cube;
      result.cube = cube;
    });
    if (!arch || (arch.getValue() != "a2" && arch.getValue() != "a3") ||
        (!explicitCore && !section) || invalidSection) {
      fail("requires-one-qualified-a2a3-physical-context");
      return result;
    }
    result.lifetimeScope = section ? section : function.getOperation();
    Region &physicalRegion = section ? section->getRegion(0) : function.getBody();
    if (section && !auditOutside(section)) {
      return result;
    }
    if (region(physicalRegion, 0))
      result.status = SyncPhysicalFacts::Status::Complete;
    return result;
  }
};

} // namespace structured_sync_coverage_detail

inline SyncPhysicalFacts importCoverageStructuredSyncPhysicalFacts(
    func::FuncOp function, const SyncIRs &ir, bool compositional = false,
    bool emitted = false, bool authoredWholeFunction = false) {
  return structured_sync_coverage_detail::Importer(function, ir, compositional, emitted, authoredWholeFunction).run();
}


} // namespace mlir::pto
#endif
