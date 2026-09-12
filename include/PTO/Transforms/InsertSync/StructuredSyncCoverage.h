// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
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
  // Do not admit arbitrary PIPE_S payloads: A2/A3 documentation does not
  // establish a generic scalar-completes-at-issue rule, and PIPE_S has no
  // same-pipe barrier. Descriptor/control operations are handled explicitly.
  if (pipe == PIPE::PIPE_MTE2 || pipe == PIPE::PIPE_MTE3)
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

inline bool implicitResourceOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  if (name.starts_with("pto.comm.") || name.starts_with("pto.sync.") ||
      name.starts_with("pto.cmo.") || name.starts_with("pto.fence.") ||
      name == "pto.syncall" || name == "pto.tsync" ||
      name == "pto.tprefetch_async" || name == "pto.make_prefetch_async_context" ||
      name == "pto.get_prefetch_async_session" ||
      name == "pto.aic_initialize_pipe" || name == "pto.aiv_initialize_pipe" ||
      name == "pto.initialize_l2g2l_pipe" || name == "pto.initialize_l2l_pipe" ||
      name == "pto.talloc" || name == "pto.tpush" || name == "pto.tpop" ||
      name == "pto.tfree" || name == "pto.set_quant_scalar" ||
      name == "pto.set_quant_vector" || name == "pto.tprint" ||
      name == "pto.print" || name == "pto.set_ffts")
    return true;
  return false;
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

  bool fail(StringRef kind, Operation *op = nullptr, StringRef detail = {}) {
    result.status = SyncPhysicalFacts::Status::Unsupported;
    result.reason = diagnostic(kind, op, detail);
    return false;
  }

  bool qualifyAccess(const BaseMemInfo *info, Operation *op, bool write) {
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

  // The parameter must not be named `region`: it would shadow this member and
  // make the recursive call below resolve to the Region itself.
  bool region(Region &body, unsigned depth) {
    if (depth > 32)
      return fail("region-depth-limit");
    if (!llvm::hasSingleElement(body))
      return fail("unsupported-multi-block-region");
    for (Operation &op : body.front()) {
      ++result.work;
      if (isa<scf::ForOp, scf::IfOp, SectionCubeOp, SectionVectorOp>(op)) {
        for (Region &child : op.getRegions())
          if (!child.empty() && !region(child, depth + 1))
            return false;
        continue;
      }
      if (isa<scf::WhileOp>(op))
        return fail("unsupported-control-region", &op, "scf.while");
      if (isa<SetFlagOp, WaitFlagOp, BarrierOp, RecordEventOp, WaitEventOp>(op))
        return fail("explicit-synchronization-input", &op);
      if (getSyncMacroModel(&op))
        return fail("multi-phase-needs-endpoints", &op);
      if (op.getNumRegions())
        return fail("unsupported-control-region", &op);

      if (auto alloc = dyn_cast<AllocTileOp>(op)) {
        auto type = cast<TileBufType>(alloc.getResult().getType());
        if (type.hasDynamicValid() &&
            (!isInsertSyncScalarPrerequisite(alloc.getValidRow()) ||
             !isInsertSyncScalarPrerequisite(alloc.getValidCol())))
          return fail("invalid-descriptor-contract", &op,
                      "allocation-valid-shape");
        continue;
      }
      if (auto alloc = dyn_cast<AllocMultiTileOp>(op)) {
        auto layout = getPTOStaticMultiTileSlotLayout(
            alloc.getResult().getType().getSlotType());
        if (failed(layout) || !layout->footprintBytes)
          return fail("invalid-descriptor-contract", &op,
                      "multi-tile-layout");
        continue;
      }
      if (auto update = dyn_cast<SetValidShapeOp>(op)) {
        if (!isInsertSyncScalarPrerequisite(update.getValidRow()) ||
            !isInsertSyncScalarPrerequisite(update.getValidCol()))
          return fail("invalid-descriptor-contract", &op,
                      "asynchronous-valid-shape-source");
        continue;
      }
      if (isa<GetValidShapeOp>(op))
        continue;
      if (auto call = dyn_cast<func::CallOp>(op)) {
        if (!helper(call))
          return false;
        continue;
      }
      if (auto physical = dyn_cast<OpPipeInterface>(op)) {
        PIPE p = physical.getPipe();
        if (!supportedLane(p, result.cube))
          return fail("unsupported-lane", &op);
        if (implicitResourceOperation(&op))
          return fail("unsupported-implicit-resource", &op);
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
  Importer(func::FuncOp f, const SyncIRs &ir) : function(f) {
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
    if (region(physicalRegion, 0))
      result.status = SyncPhysicalFacts::Status::Complete;
    return result;
  }
};

} // namespace structured_sync_coverage_detail

inline SyncPhysicalFacts importCoverageStructuredSyncPhysicalFacts(
    func::FuncOp function, const SyncIRs &ir) {
  return structured_sync_coverage_detail::Importer(function, ir).run();
}


} // namespace mlir::pto
#endif
