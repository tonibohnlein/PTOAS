// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License. Ports the original region and physical-footprint extraction from OAHS Native.cpp. The
// adapter borrows SyncInput effects; all precision/widening is private to Phase A. It builds no
// selected commands and changes no source IR.
#include "PTO/Transforms/FrontierSynch/OriginalStructure.h"
#include "PTO/Transforms/FrontierSynch/StorageWitnesses.h"
#include "StorageOrigins.h"
#include <functional>

namespace mlir::pto::frontiersynch {
namespace {
LogicalResult importControl(func::FuncOp function, const SyncInput &input,
                            SyncSlotMapping::AnalysisContext &scalarFacts,
                            OriginalStructure &result) {
  if (!function.getBody().hasOneBlock()) {
    return function.emitError("frontier-synch: function requires one CFG entry block");
  }
  DenseMap<mlir::Operation *, std::size_t> originalAnchors;
  function.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (op != function.getOperation()) {
      originalAnchors.try_emplace(op, result.originalSites.size());
      result.originalSites.push_back(op);
    }
  });
  DenseMap<mlir::Operation *, SmallVector<std::size_t>> operationIds;
  for (const auto *phase : input.instructions()) {
    operationIds[phase->elementOp].push_back(result.operations.size());
    result.operations.push_back({phase, originalAnchors.lookup(phase->elementOp), {}});
  }
  std::vector<bool> represented(result.operations.size());
  bool invalidControl = false;
  std::function<Region(mlir::Region &)> importRegion = [&](mlir::Region &region) -> Region {
    if (!region.empty() && !region.hasOneBlock()) {
      invalidControl = true;
      return {};
    }
    Region sequence;
    sequence.kind = Region::Sequence;
    for (Block &block : region) {
      for (mlir::Operation &op : block) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          Region choice;
          choice.kind = Region::Choice;
          choice.originalOwner = originalAnchors.lookup(&op);
          choice.children.push_back(importRegion(ifOp.getThenRegion()));
          choice.children.push_back(importRegion(ifOp.getElseRegion()));
          sequence.children.push_back(std::move(choice));
        } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          Region loop;
          loop.kind = Region::For;
          loop.originalOwner = originalAnchors.lookup(&op);
          const auto &domain = scalarFacts.domain(forOp);
          loop.qualifiedCounted = bool(domain);
          loop.zeroTripPossible = !domain || !domain->atLeastOnce;
          loop.children.push_back(importRegion(forOp.getRegion()));
          sequence.children.push_back(std::move(loop));
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
          Region loop;
          loop.kind = Region::While;
          loop.originalOwner = originalAnchors.lookup(&op);
          loop.children.push_back(importRegion(whileOp.getBefore()));
          loop.children.push_back(importRegion(whileOp.getAfter()));
          sequence.children.push_back(std::move(loop));
        } else if (auto found = operationIds.find(&op); found != operationIds.end()) {
          for (auto id : found->second) {
            Region phase;
            phase.kind = Region::Operation;
            phase.originalOwner = originalAnchors.lookup(&op);
            phase.operation = id;
            represented[id] = true;
            sequence.children.push_back(std::move(phase));
          }
        }
      }
    }
    return sequence;
  };
  result.body = importRegion(function.getBody());
  if (invalidControl) {
    return function.emitError("frontier-synch: original control requires single-block structured regions");
  }
  for (auto [i, phase] : llvm::enumerate(input.instructions())) {
    if (!represented[i]) {
      return phase->elementOp->emitError(
          "frontier-synch: translated phase lies outside supported structured control");
    }
  }
  return success();
}

// Native.cpp's dependency-sliced address specialization, retaining one record
// per selector/owner instead of its later operation-copy materialization.
std::optional<PhysicalAddressRelation>
deriveAddress(const BaseMemInfo &memory, SyncSlotMapping::AnalysisContext &scalarFacts,
              const DenseMap<mlir::Operation *, std::size_t> &owners) {
  if (memory.scope == AddressSpace::GM || memory.scope == AddressSpace::Zero ||
      !memory.allocateSize || memory.aliasesUnknownRange || memory.baseAddresses.empty() ||
      memory.hasKnownPhysicalAddresses) {
    return {};
  }
  if (!memory.rootBuffer) {
    return {};
  }
  auto alloc = memory.rootBuffer.getDefiningOp<AllocTileOp>();
  if (!alloc || !alloc.getAddr()) {
    return {};
  }
  for (auto loop = alloc->getParentOfType<scf::ForOp>(); loop;
       loop = loop->getParentOfType<scf::ForOp>()) {
    auto mapping = SyncSlotMapping::derive(loop, alloc.getAddr(), scalarFacts);
    if (!mapping) {
      continue;
    }
    PhysicalAddressRelation relation;
    relation.owner = owners.lookup(loop.getOperation());
    relation.address = alloc.getAddr();
    relation.memory = &memory;
    for (auto &values : mapping->values) {
      const auto address = SyncSlotMapping::evaluate(alloc.getAddr(), values);
      SmallVector<uint64_t> footprint;
      for (auto offset : memory.baseAddresses) {
        if (!address || offset > std::numeric_limits<uint64_t>::max() - *address ||
            memory.allocateSize > std::numeric_limits<uint64_t>::max() - (*address + offset)) {
          break;
        }
        footprint.push_back(*address + offset);
      }
      if (footprint.size() != memory.baseAddresses.size()) {
        break;
      }
      relation.addresses.push_back(std::move(footprint));
    }
    if (relation.addresses.size() == mapping->period) {
      return relation;
    }
  }
  return {};
}

// Coordinate qualification ported from MemoryDependentAnalyzer::storageCoordinates.
// A bounding interval is not proof of a definite full-cell write.
void setCoordinates(const BaseMemInfo &memory, std::size_t root, Cell &cell) {
  if (memory.aliasesUnknownRange || memory.baseAddresses.size() != 1 || !memory.allocateSize ||
      memory.scope == AddressSpace::Zero) {
    return;
  }
  const bool absolute = memory.scope != AddressSpace::GM;
  if ((absolute && !memory.hasKnownPhysicalAddresses) || (!absolute && !memory.rootBuffer)) {
    return;
  }
  const auto begin = memory.baseAddresses.front();
  if (memory.allocateSize > std::numeric_limits<uint64_t>::max() - begin) {
    return;
  }
  cell.coordinateSpace = absolute ? "physical-local" : "root:" + std::to_string(root);
  cell.ranges = {{begin, memory.allocateSize}};
}

void importStorage(func::FuncOp function, const SyncInput &input,
                   SyncSlotMapping::AnalysisContext &scalarFacts, OriginalStructure &result) {
  const auto origins = origin_detail::collectOrigins(function, input.buffers());
  DenseMap<mlir::Operation *, std::size_t> owners;
  for (auto [i, op] : llvm::enumerate(result.originalSites)) {
    owners[op] = i;
  }
  DenseMap<Value, std::size_t> storageRoots;
  auto rootId = [&](Value value) {
    auto inserted = storageRoots.try_emplace(value, result.storageRoots.size());
    if (inserted.second) {
      result.storageRoots.push_back(value);
    }
    return inserted.first->second;
  };
  DenseMap<const BaseMemInfo *, std::size_t> groupIds;
  std::vector<std::unique_ptr<BaseMemInfo>> memories;
  std::vector<FootprintGroup> groups;
  auto groupFor = [&](const BaseMemInfo *original) {
    auto found = groupIds.find(original);
    if (found != groupIds.end()) {
      return found->second;
    }
    const auto group = groups.size();
    groupIds[original] = group;
    auto memory = original->clone();
    auto origin = origins.find(original->baseBuffer);
    // The root-set closure gives possible provenance, not geometry. Keep carried
    // views conservative until an offset/occurrence proof replaces widening.
    const bool carried = origin != origins.end() && origin->second.carried;
    if (carried || (origin != origins.end() && origin->second.unknown)) {
      memory->aliasesUnknownRange = true;
      memory->hasKnownPhysicalAddresses = false;
    } else if (auto relation = deriveAddress(*original, scalarFacts, owners)) {
      memory->baseAddresses.clear();
      for (const auto &footprint : relation->addresses) {
        memory->baseAddresses.append(footprint.begin(), footprint.end());
      }
      llvm::sort(memory->baseAddresses);
      memory->baseAddresses.erase(
          std::unique(memory->baseAddresses.begin(), memory->baseAddresses.end()),
          memory->baseAddresses.end());
      memory->hasKnownPhysicalAddresses = true;
      memory->aliasesUnknownRange = false;
      result.physicalAddresses.push_back(std::move(*relation));
    }
    const bool local = memory->scope != AddressSpace::GM && memory->scope != AddressSpace::Zero;
    const bool overflow = llvm::any_of(memory->baseAddresses, [&](uint64_t base) {
      return memory->allocateSize > std::numeric_limits<uint64_t>::max() - base;
    });
    if (overflow || !memory->allocateSize || memory->baseAddresses.empty() ||
        (local && !memory->hasKnownPhysicalAddresses)) {
      memory->aliasesUnknownRange = true;
      memory->hasKnownPhysicalAddresses = false;
    }
    FootprintGroup entry;
    auto &cell = entry.description;
    cell.addressSpace = std::to_string(static_cast<unsigned>(memory->scope));
    cell.provenance = "original translated footprint";
    cell.unknownRange = memory->aliasesUnknownRange;
    std::size_t root = NoControlId;
    if (memory->rootBuffer) {
      root = rootId(memory->rootBuffer);
      cell.storageOrigins.push_back(root);
    }
    if (origin != origins.end()) {
      for (auto value : origin->second.roots) {
        cell.storageOrigins.push_back(rootId(value));
      }
    }
    for (uint64_t base : memory->baseAddresses) {
      cell.ranges.push_back({base, memory->allocateSize});
    }
    setCoordinates(*memory, root, cell);
    groups.push_back(std::move(entry));
    memories.push_back(std::move(memory));
    return group;
  };
  for (auto [i, operation] : llvm::enumerate(result.operations)) {
    for (const auto *memory : operation.instruction->defVec) {
      const auto group = groupFor(memory);
      groups[group].uses.push_back({i, false, true});
    }
    for (const auto *memory : operation.instruction->useVec) {
      const auto group = groupFor(memory);
      groups[group].uses.push_back({i, true, false});
    }
  }
  appendCanonicalStorage(result, groups, [&](std::size_t a, std::size_t b) {
    return input.memory().MemAlias(memories[a].get(), memories[b].get());
  });
}
} // namespace

LogicalResult importOriginalStructure(func::FuncOp function, const SyncInput &input,
                                      OriginalStructure &result) {
  OriginalStructure candidate;
  SyncSlotMapping::AnalysisContext scalarFacts;
  if (failed(importControl(function, input, scalarFacts, candidate))) {
    return failure();
  }
  candidate.descriptors = std::make_unique<SyncTileDescriptorState>(function);
  importStorage(function, input, scalarFacts, candidate);
  result = std::move(candidate);
  return success();
}
} // namespace mlir::pto::frontiersynch
