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
#include "PTO/Transforms/SlotAffineAnalysis.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/IR/PTOMultiBuffer.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <map>
#include <set>
#include <tuple>
using namespace mlir;
using namespace mlir::pto;

uint64_t mlir::pto::estimateSyncPhysicalSlotQualificationWork(
    const BaseMemInfo *memory, const Buffer2MemInfoMap &buffers) {
  constexpr uint64_t fragmentLimit = 2048;
  uint64_t accessSlots = memory ? std::min<uint64_t>(memory->baseAddresses.size(), fragmentLimit) : 0;
  uint64_t rootSlots = 0;
  if (memory && memory->rootBuffer && memory->rootBuffer.getDefiningOp<AllocMultiTileOp>()) {
    auto found = buffers.find(memory->rootBuffer);
    if (found != buffers.end() && found->second.size() == 1 && found->second.front())
      rootSlots = std::min<uint64_t>(found->second.front()->baseAddresses.size(), fragmentLimit);
  }
  uint64_t levels = 1;
  for (uint64_t remaining = rootSlots; remaining > 1; remaining = (remaining + 1) / 2) ++levels;
  // Fixed provenance/view work, original-access validation, root validation,
  // copy/sort/nonoverlap checks, and returned original-order mapping storage.
  return 160 + 2 * accessSlots + rootSlots * (8 + 3 * levels);
}

std::optional<SyncPhysicalSlotMapping> mlir::pto::qualifySyncPhysicalSlots(
    const BaseMemInfo *memory, const Buffer2MemInfoMap &buffers) {
  auto physical = [](const BaseMemInfo *info) {
    if (!info || !info->hasKnownPhysicalAddresses || info->aliasesUnknownRange ||
        info->scope == AddressSpace::GM || info->scope == AddressSpace::Zero ||
        !info->allocateSize || info->baseAddresses.empty() || info->baseAddresses.size() > 2048)
      return false;
    return llvm::all_of(info->baseAddresses, [&](uint64_t base) {
      return base <= std::numeric_limits<uint64_t>::max() - info->allocateSize;
    });
  };
  if (!physical(memory)) return {};
  auto root = memory->rootBuffer ? memory->rootBuffer.getDefiningOp<AllocMultiTileOp>() : AllocMultiTileOp();
  if (!root) {
    // Ordinary translated ranges are conservative effect footprints. No
    // dynamic slot interpretation is added to them.
    return SyncPhysicalSlotMapping{{}, memory->baseAddresses, memory->allocateSize, memory->scope};
  }
  Value selected = memory->baseBuffer;
  SmallVector<TileBufType> views;
  MultiTileGetOp get;
  for (unsigned hop = 0; selected && hop < 32; ++hop) {
    Operation *op = selected.getDefiningOp();
    if (!op) return {};
    if ((get = dyn_cast<MultiTileGetOp>(op))) break;
    auto viewType = dyn_cast<TileBufType>(selected.getType());
    if (!viewType) return {};
    views.push_back(viewType);
    if (auto reshape = dyn_cast<TReshapeOp>(op)) selected = reshape.getSrc();
    else if (auto bitcast = dyn_cast<BitcastOp>(op)) selected = bitcast.getSrc();
    else return {}; // In particular, no dynamic subview offset is guessed.
  }
  if (!get || get.getSource() != root.getResult() ||
      findMultiTileSlotExpr(memory->baseBuffer) != get.getSlot()) return {};
  auto rootEntries = buffers.find(root.getResult());
  if (rootEntries == buffers.end() || rootEntries->second.size() != 1) return {};
  const BaseMemInfo *rootMemory = rootEntries->second.front().get();
  auto type = root.getResult().getType();
  auto slot = type.getSlotType();
  auto layout = getPTOStaticMultiTileSlotLayout(slot);
  if (failed(layout) || layout->footprintBytes > uint64_t(INT64_MAX) ||
      slot.getCompactModeI32() == int32_t(CompactMode::RowPlusOne) ||
      !physical(rootMemory) || rootMemory->rootBuffer != root.getResult() ||
      rootMemory->scope != memory->scope || rootMemory->baseAddresses.size() != type.getCount() ||
      rootMemory->allocateSize != layout->footprintBytes || memory->allocateSize != layout->footprintBytes)
    return {};
  auto scope = dyn_cast_or_null<AddressSpaceAttr>(slot.getMemorySpace());
  if (!scope || scope.getAddressSpace() != memory->scope) return {};
  for (TileBufType view : views) {
    auto shape = getPTOStaticMultiTileSlotLayout(view);
    if (failed(shape) || shape->footprintBytes > layout->footprintBytes ||
        view.getMemorySpace() != slot.getMemorySpace() ||
        view.getCompactModeI32() == int32_t(CompactMode::RowPlusOne)) return {};
  }
  const auto &bases = rootMemory->baseAddresses;
  if (auto planned = root->getAttrOfType<DenseI64ArrayAttr>(kPtoMultiBufferAddrsAttrName)) {
    if (root.getAddr() || uint64_t(planned.size()) != bases.size()) return {};
    for (unsigned i = 0; i < bases.size(); ++i)
      if (planned[i] < 0 || uint64_t(planned[i]) != bases[i]) return {};
  } else {
    // Translation and lowering share checked aligned-stride addressing. The
    // effect remains the unpadded footprint at each resulting slot base.
    if (!root.getAddr()) return {};
    for (unsigned i = 0; i < bases.size(); ++i) {
      auto address = getPTOStaticMultiTileSlotAddress(*layout, bases.front(), i);
      if (failed(address) || bases[i] != *address) return {};
    }
  }
  SmallVector<uint64_t> sorted(bases);
  llvm::sort(sorted);
  for (unsigned i = 0; i < sorted.size(); ++i) {
    if (sorted[i] % layout->alignmentBytes ||
        failed(getPTOStaticMultiTileSlotAddress(*layout, sorted[i], 0)) ||
        (i && sorted[i - 1] + layout->footprintBytes > sorted[i])) return {};
  }
  IntegerAttr literal;
  if (matchPattern(get.getSlot(), m_Constant(&literal))) {
    if (!literal.getValue().isSignedIntN(64)) return {};
    int64_t index = literal.getValue().getSExtValue();
    if (index < 0 || uint64_t(index) >= bases.size() || memory->baseAddresses.size() != 1 ||
        memory->baseAddresses.front() != bases[index]) return {};
  } else if (ArrayRef<uint64_t>(memory->baseAddresses) != ArrayRef<uint64_t>(bases)) return {};
  return SyncPhysicalSlotMapping{get.getSlot(), bases, layout->footprintBytes, memory->scope};
}

std::optional<std::vector<std::pair<unsigned, unsigned>>> mlir::pto::overlappingSyncPhysicalSlots(
    const SyncPhysicalSlotMapping &a, const SyncPhysicalSlotMapping &b,
    llvm::function_ref<bool(uint64_t)> spend) {
  std::vector<std::pair<unsigned, unsigned>> result;
  if (!spend(1)) return {};
  if (a.scope != b.scope) return result;
  const uint64_t n = a.bases.size() + b.bases.size();
  uint64_t levels = 1;
  for (uint64_t remaining = n; remaining > 1; remaining = (remaining + 1) / 2) ++levels;
  // Collect/sort and balanced active/expiration indexes. No Cartesian charge
  // is paid for disjoint intervals; actual output is billed separately below.
  if (!spend(n * (4 + 3 * levels))) return {};
  struct Span { uint64_t begin, end; unsigned index; bool right; };
  SmallVector<Span> spans;
  for (unsigned i = 0; i < a.bases.size(); ++i) spans.push_back({a.bases[i], a.bases[i] + a.bytes, i, false});
  for (unsigned i = 0; i < b.bases.size(); ++i) spans.push_back({b.bases[i], b.bases[i] + b.bytes, i, true});
  llvm::sort(spans, [](const Span &x, const Span &y) {
    return std::tie(x.begin, x.end, x.right, x.index) < std::tie(y.begin, y.end, y.right, y.index);
  });
  std::set<unsigned> activeLeft, activeRight;
  std::multimap<uint64_t, std::pair<bool, unsigned>> expiration;
  for (const auto &span : spans) {
    while (!expiration.empty() && expiration.begin()->first <= span.begin) {
      auto [right, index] = expiration.begin()->second;
      (right ? activeRight : activeLeft).erase(index);
      expiration.erase(expiration.begin());
    }
    if (span.right) {
      for (unsigned left : activeLeft) {
        if (!spend(1)) return {};
        result.emplace_back(left, span.index);
      }
      activeRight.insert(span.index);
    } else {
      for (unsigned right : activeRight) {
        if (!spend(1)) return {};
        result.emplace_back(span.index, right);
      }
      activeLeft.insert(span.index);
    }
    expiration.emplace(span.end, std::make_pair(span.right, span.index));
  }
  return result;
}

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
  bool bounded;
  uint64_t fragments = 0;
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
    if (bounded && fragments > 2048) return fail("physical fragment budget", true);
    if (info->scope == AddressSpace::GM) return true;
    if (!info->hasKnownPhysicalAddresses || !info->allocateSize ||
        info->baseAddresses.empty() || (bounded && info->baseAddresses.size() > 64)) return false;
    if (info->rootBuffer && info->rootBuffer.getDefiningOp<AllocMultiTileOp>() &&
        (info->allocateSize > uint64_t(INT64_MAX) ||
         llvm::any_of(info->baseAddresses, [&](uint64_t base) {
           return base > uint64_t(INT64_MAX) - info->allocateSize;
         }))) return false;
    return llvm::all_of(info->baseAddresses, [&](uint64_t address) {
      return address <= std::numeric_limits<uint64_t>::max() - info->allocateSize;
    });
  }
  bool region(Region &r, unsigned depth) {
    if (bounded && depth > 24) return fail("region nesting budget", true);
    if (!llvm::hasSingleElement(r)) return fail("unsupported multi-block region");
    for (Operation &op : r.front()) {
      ++result.work;
      if (bounded && (result.work > budget || result.work > 8192)) return fail("IR visitation budget", true);
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
      if (auto alloc = dyn_cast<AllocMultiTileOp>(op)) {
        auto type = alloc.getResult().getType().getSlotType();
        auto layout = getPTOStaticMultiTileSlotLayout(type);
        if (failed(layout) || layout->footprintBytes > uint64_t(INT64_MAX) ||
            type.getCompactModeI32() == int32_t(CompactMode::RowPlusOne))
          return fail("unqualified multi-tile physical layout");
        // This admission precedes all interval/alias candidate pruning. Merely
        // declining optional selector refinement would be too late if the
        // original translated range had already missed a real lowered overlap.
        if (auto planned = alloc->getAttrOfType<DenseI64ArrayAttr>(kPtoMultiBufferAddrsAttrName))
          for (int64_t base : planned.asArrayRef())
            if (base < 0 || uint64_t(base) % layout->alignmentBytes ||
                failed(getPTOStaticMultiTileSlotAddress(*layout, uint64_t(base), 0)))
              return fail("unqualified planned multi-tile physical interval");
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
        if (bounded && (count > 16 || result.phases.size() >= 256)) return fail("phase/access budget", true);
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
  Importer(func::FuncOp f, uint64_t b, bool bounded = true) : function(f), budget(b), bounded(bounded) {}
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
SyncPhysicalFacts mlir::pto::importStructuredSyncPhysicalFacts(func::FuncOp f, const SyncIRs &ir) {
  return Importer(f, 0, false).run(ir);
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

SyncAccessCandidates mlir::pto::enumerateSyncAccessCandidates(
    ArrayRef<SyncPhysicalAccess> accesses, llvm::function_ref<bool(uint64_t)> spend) {
  SyncAccessCandidates result;
  auto charge = [&](uint64_t amount) {
    if (!spend(amount)) {
      result.status = SyncAccessCandidates::Status::AnalysisLimit;
      return false;
    }
    result.work += amount;
    return true;
  };
  struct Interval { uint64_t begin, end; unsigned access; };
  std::map<AddressSpace, SmallVector<unsigned>> reads, writes, unknown;
  std::map<std::pair<AddressSpace, PipelineType>, SmallVector<unsigned>> readersByLane;
  std::map<AddressSpace, std::vector<Interval>> intervals;
  for (unsigned id = 0; id < accesses.size(); ++id) {
    const auto &access = accesses[id];
    const auto *mem = access.memory;
    if (!mem) return result;
    if (!charge(mem->baseAddresses.size() + 1)) return result;
    (access.write ? writes : reads)[mem->scope].push_back(id);
    if (!access.write) readersByLane[{mem->scope, access.lane}].push_back(id);
    bool known = mem->scope != AddressSpace::GM && mem->hasKnownPhysicalAddresses &&
        !mem->aliasesUnknownRange && mem->allocateSize && !mem->baseAddresses.empty();
    for (uint64_t base : mem->baseAddresses)
      known &= base <= std::numeric_limits<uint64_t>::max() - mem->allocateSize;
    if (!known) { unknown[mem->scope].push_back(id); continue; }
    for (uint64_t base : mem->baseAddresses)
      intervals[mem->scope].push_back({base, base + mem->allocateSize, id});
  }
  std::set<std::pair<unsigned, unsigned>> candidates;
  auto candidate = [&](unsigned x, unsigned y) {
    if (!charge(1)) return false;
    ++result.candidateVisits;
    const auto &a = accesses[x]; const auto &b = accesses[y];
    if (!a.write && !b.write &&
        (a.memory->scope != AddressSpace::ACC || a.lane == b.lane)) return true;
    candidates.emplace(std::min(x, y), std::max(x, y));
    return true;
  };
  for (auto &[scope, spans] : intervals) {
    llvm::sort(spans, [](const Interval &a, const Interval &b) {
      return std::tie(a.begin, a.end, a.access) < std::tie(b.begin, b.end, b.access);
    });
    std::map<PipelineType, std::set<unsigned>> activeReads;
    std::set<unsigned> activeWrites;
    std::multimap<uint64_t, unsigned> expiration;
    for (unsigned i = 0; i < spans.size(); ++i) {
      ++result.intervalVisits;
      const auto &current = spans[i];
      while (!expiration.empty() && expiration.begin()->first <= current.begin) {
        unsigned old = expiration.begin()->second;
        const auto &access = accesses[spans[old].access];
        if (access.write) activeWrites.erase(old);
        else activeReads[access.lane].erase(old);
        expiration.erase(expiration.begin());
      }
      if (!candidate(current.access, current.access)) return result;
      for (unsigned previous : activeWrites)
        if (!candidate(current.access, spans[previous].access)) return result;
      const auto &access = accesses[current.access];
      if (access.write || scope == AddressSpace::ACC)
        for (const auto &[lane, readers] : activeReads) {
          if (!access.write && lane == access.lane) continue;
          for (unsigned previous : readers)
            if (!candidate(current.access, spans[previous].access)) return result;
        }
      if (access.write) activeWrites.insert(i);
      else activeReads[access.lane].insert(i);
      expiration.emplace(current.end, i);
    }
  }
  for (const auto &[scope, uncertain] : unknown)
    for (unsigned x : uncertain) {
      for (unsigned y : writes[scope]) if (!candidate(x, y)) return result;
      if (accesses[x].write) {
        for (unsigned y : reads[scope]) if (!candidate(x, y)) return result;
      } else if (scope == AddressSpace::ACC) {
        for (const auto &[key, readers] : readersByLane)
          if (key.first == scope && key.second != accesses[x].lane)
            for (unsigned y : readers) if (!candidate(x, y)) return result;
      }
    }
  result.pairs.assign(candidates.begin(), candidates.end());
  result.status = SyncAccessCandidates::Status::Complete;
  return result;
}
