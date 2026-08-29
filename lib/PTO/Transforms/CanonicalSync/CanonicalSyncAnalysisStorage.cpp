// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncAnalysisInternal.h"

#include "PTO/Transforms/SlotAffineAnalysis.h"

#include "mlir/IR/Matchers.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

SyncCoverStorageDomainRole getStorageDomainRole(AddressSpace space) {
  switch (space) {
  case AddressSpace::LEFT:
    return SyncCoverStorageDomainRole::L0Left;
  case AddressSpace::RIGHT:
    return SyncCoverStorageDomainRole::L0Right;
  case AddressSpace::ACC:
    return SyncCoverStorageDomainRole::Accumulator;
  default:
    return SyncCoverStorageDomainRole::Other;
  }
}

bool checkedIntervalEnd(std::uint64_t begin, std::uint64_t size,
                        std::uint64_t &end) {
  const bool invalid =
      size == 0 || begin > std::numeric_limits<std::uint64_t>::max() - size;
  if (invalid) {
    return false;
  }
  end = begin + size;
  return begin < end;
}

bool sameAccessIdentity(const ExtractedAccess &first,
                        const BaseMemInfo &second) {
  const bool sameAddresses =
      first.addresses.size() == second.baseAddresses.size() &&
      std::equal(first.addresses.begin(), first.addresses.end(),
                 second.baseAddresses.begin());
  return first.base == second.baseBuffer && first.root == second.rootBuffer &&
         first.space == second.scope && sameAddresses &&
         first.size == second.allocateSize &&
         first.knownPhysical == second.hasKnownPhysicalAddresses &&
         first.unknownRange == second.aliasesUnknownRange &&
         first.rootRelativeOffset == second.rootRelativeOffset;
}

std::optional<unsigned> getFunctionArgument(Value value,
                                            func::FuncOp function) {
  auto argument = dyn_cast<BlockArgument>(value);
  const bool invalid =
      !argument || argument.getOwner() != &function.getBody().front();
  if (invalid) {
    return std::nullopt;
  }
  return argument.getArgNumber();
}

struct ActiveStorageAccess {
  std::uint64_t end = 0;
  SyncCoverStorageAccessId access = 0;

  bool operator>(const ActiveStorageAccess &other) const {
    return std::tie(end, access) > std::tie(other.end, other.access);
  }
};

using ActiveStorageHeap =
    std::priority_queue<ActiveStorageAccess, std::vector<ActiveStorageAccess>,
                        std::greater<ActiveStorageAccess>>;

void expireStorageAccesses(std::uint64_t begin, ActiveStorageHeap &expiry,
                           std::set<SyncCoverStorageAccessId> &active) {
  while (!expiry.empty()) {
    const bool expired = expiry.top().end <= begin;
    if (!expired) {
      break;
    }
    active.erase(expiry.top().access);
    expiry.pop();
  }
}

} // namespace

void ProgramBuilder::appendAccesses(SyncCoverNodeId node,
                                    ArrayRef<const BaseMemInfo *> memoryInfos,
                                    bool writes) {
  for (const BaseMemInfo *memory : memoryInfos) {
    if (!memory) {
      continue;
    }
    auto existingIndex = std::find_if(
        nodeAccessIndices_[node].begin(), nodeAccessIndices_[node].end(),
        [&](std::size_t index) {
          return sameAccessIdentity(extractedAccesses_[index], *memory);
        });
    const SyncCoverStorageAccessMode incoming =
        writes ? SyncCoverStorageAccessMode::Write
               : SyncCoverStorageAccessMode::Read;
    if (existingIndex != nodeAccessIndices_[node].end()) {
      ExtractedAccess &existing = extractedAccesses_[*existingIndex];
      const unsigned merged = static_cast<unsigned>(existing.mode) |
                              static_cast<unsigned>(incoming);
      existing.mode = static_cast<SyncCoverStorageAccessMode>(merged);
      continue;
    }
    ExtractedAccess access;
    access.node = node;
    access.base = memory->baseBuffer;
    access.root = memory->rootBuffer;
    access.space = memory->scope;
    access.addresses.assign(memory->baseAddresses.begin(),
                            memory->baseAddresses.end());
    access.size = memory->allocateSize;
    access.knownPhysical = memory->hasKnownPhysicalAddresses;
    access.unknownRange = memory->aliasesUnknownRange;
    access.rootRelativeOffset = memory->rootRelativeOffset;
    access.mode = incoming;
    extractedAccesses_.push_back(std::move(access));
    nodeAccessIndices_[node].push_back(extractedAccesses_.size() - 1);
  }
}

LogicalResult ProgramBuilder::materializeNodeAccesses(SyncCoverNodeId node) {
  for (std::size_t accessIndex : nodeAccessIndices_[node]) {
    ExtractedAccess &access = extractedAccesses_[accessIndex];
    auto domainPosition = storageDomains_.find(access.space);
    if (domainPosition == storageDomains_.end()) {
      const SyncCoverGraphResult added =
          graph_.addStorageDomain(getStorageDomainRole(access.space));
      if (!added) {
        return function_.emitError(
            "cannot construct canonical sync storage domain");
      }
      domainPosition =
          storageDomains_.emplace(access.space, *added.index).first;
    }
    const SyncCoverStorageDomainId domain = domainPosition->second;

    auto familyPosition = storageFamilies_.find(access.root);
    if (familyPosition == storageFamilies_.end()) {
      familyPosition =
          storageFamilies_.insert({access.root, nextStorageFamily_++}).first;
    }
    const SyncCoverStorageAccessFamilyId family = familyPosition->second;
    const bool exactLocal = access.space != AddressSpace::GM &&
                            access.space != AddressSpace::Zero &&
                            access.knownPhysical && !access.unknownRange &&
                            access.size != 0 && !access.addresses.empty();
    const bool hasRootRelativeGmRange =
        access.space == AddressSpace::GM && access.rootRelativeOffset;
    const bool hasKnownRange =
        (access.knownPhysical || hasRootRelativeGmRange) &&
        !access.unknownRange && access.size != 0 && !access.addresses.empty();
    if (!hasKnownRange) {
      if (failed(addConservativeAccess(access, domain, family))) {
        return failure();
      }
      continue;
    }

    std::vector<std::uint64_t> ends;
    ends.reserve(access.addresses.size());
    for (std::uint64_t address : access.addresses) {
      std::uint64_t end = 0;
      if (!checkedIntervalEnd(address, access.size, end)) {
        if (failed(addConservativeAccess(access, domain, family))) {
          return failure();
        }
        ends.clear();
        break;
      }
      ends.push_back(end);
    }
    if (ends.empty()) {
      continue;
    }
    for (std::size_t ordinal = 0; ordinal < access.addresses.size();
         ++ordinal) {
      const bool storageLimitReached =
          graph_.getStorageAccesses().size() == options_.maximumStorageAccesses;
      if (storageLimitReached) {
        return function_.emitError(
            "canonical sync storage-access limit exceeded");
      }
      const SyncCoverGraphResult added = graph_.addStorageAccess(
          node, domain, family, {access.addresses[ordinal], ends[ordinal]},
          access.mode, static_cast<unsigned>(ordinal), exactLocal);
      if (!added) {
        return function_.emitError(
            "cannot construct canonical sync storage access");
      }
      access.graphAccesses.push_back(*added.index);
    }
  }
  return success();
}

LogicalResult
ProgramBuilder::addConservativeAccess(ExtractedAccess &access,
                                      SyncCoverStorageDomainId domain,
                                      SyncCoverStorageAccessFamilyId family) {
  const bool storageLimitReached =
      graph_.getStorageAccesses().size() == options_.maximumStorageAccesses;
  if (storageLimitReached) {
    return function_.emitError("canonical sync storage-access limit exceeded");
  }
  const SyncCoverGraphResult added = graph_.addStorageAccess(
      access.node, domain, family,
      {0, std::numeric_limits<std::uint64_t>::max()}, access.mode);
  if (!added) {
    return function_.emitError(
        "cannot construct conservative canonical sync storage access");
  }
  access.graphAccesses = {*added.index};
  return success();
}

LogicalResult ProgramBuilder::buildStorageConflictIndex() {
  const auto &graphAccesses = graph_.getStorageAccesses();
  const std::size_t missing = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> extractedByGraphAccess(graphAccesses.size(),
                                                  missing);
  for (std::size_t extracted = 0; extracted < extractedAccesses_.size();
       ++extracted) {
    for (SyncCoverStorageAccessId access :
         extractedAccesses_[extracted].graphAccesses) {
      const bool invalid = access >= extractedByGraphAccess.size() ||
                           extractedByGraphAccess[access] != missing;
      if (invalid) {
        return function_.emitError(
            "canonical sync storage index has invalid access ownership");
      }
      extractedByGraphAccess[access] = extracted;
    }
  }
  if (llvm::is_contained(extractedByGraphAccess, missing)) {
    return function_.emitError(
        "canonical sync storage index lost access ownership");
  }

  using StorageConflictGroup =
      std::pair<SyncCoverStorageDomainId, SyncCoverStorageAccessFamilyId>;
  std::map<StorageConflictGroup, std::vector<SyncCoverStorageAccessId>>
      accessesByConflictGroup;
  std::vector<SyncCoverStorageAccessId> gmAccesses;
  for (const SyncCoverStorageAccess &access : graphAccesses) {
    if (access.domain >= graph_.getStorageDomains().size()) {
      return function_.emitError(
          "canonical sync storage index has invalid domain ownership");
    }
    const ExtractedAccess &extracted =
        extractedAccesses_[extractedByGraphAccess[access.id]];
    const bool isGm = extracted.space == AddressSpace::GM;
    accessesByConflictGroup[{access.domain, isGm ? access.family : 0}]
        .push_back(access.id);
    if (isGm) {
      gmAccesses.push_back(access.id);
    }
  }

  std::vector<bool> nodeIsInLoop(nodeBindings_.size(), false);
  for (const auto &[loop, nodes] : loopNodes_) {
    (void)loop;
    for (SyncCoverNodeId node : nodes) {
      nodeIsInLoop[node] = true;
    }
  }
  std::set<std::pair<SyncCoverNodeId, SyncCoverNodeId>> nodePairs;
  const auto retainNodePair = [&](SyncCoverNodeId first,
                                  SyncCoverNodeId second) -> LogicalResult {
    const std::pair<SyncCoverNodeId, SyncCoverNodeId> pair =
        std::minmax(first, second);
    if (pair.first == pair.second && !nodeIsInLoop[pair.first]) {
      return success();
    }
    const bool retained = nodePairs.find(pair) != nodePairs.end();
    if (retained) {
      return success();
    }
    const bool edgeLimitReached =
        nodePairs.size() == options_.maximumStorageConflictEdges;
    if (edgeLimitReached) {
      return function_.emitError(
          "canonical sync storage-conflict edge limit exceeded");
    }
    nodePairs.insert(pair);
    return success();
  };
  for (auto &[group, domainAccesses] : accessesByConflictGroup) {
    (void)group;
    llvm::sort(domainAccesses, [&](SyncCoverStorageAccessId first,
                                   SyncCoverStorageAccessId second) {
      const SyncCoverStorageAccess &firstAccess = graphAccesses[first];
      const SyncCoverStorageAccess &secondAccess = graphAccesses[second];
      return std::tie(firstAccess.extent.begin, firstAccess.extent.end,
                      firstAccess.id) < std::tie(secondAccess.extent.begin,
                                                 secondAccess.extent.end,
                                                 secondAccess.id);
    });
    ActiveStorageHeap writerExpiry;
    ActiveStorageHeap readerExpiry;
    std::set<SyncCoverStorageAccessId> activeWriters;
    std::set<SyncCoverStorageAccessId> activeReaders;
    for (SyncCoverStorageAccessId currentId : domainAccesses) {
      const SyncCoverStorageAccess &current = graphAccesses[currentId];
      expireStorageAccesses(current.extent.begin, writerExpiry, activeWriters);
      expireStorageAccesses(current.extent.begin, readerExpiry, activeReaders);
      const bool currentWrites = syncCoverStorageModeWrites(current.mode);
      const bool currentReads = syncCoverStorageModeReads(current.mode);
      const auto joinActive = [&](const auto &active) -> LogicalResult {
        for (SyncCoverStorageAccessId previousId : active) {
          if (!consumePairInspection()) {
            return function_.emitError(
                "canonical sync pair-inspection limit exceeded");
          }
          const ExtractedAccess &previousExtracted =
              extractedAccesses_[extractedByGraphAccess[previousId]];
          const ExtractedAccess &currentExtracted =
              extractedAccesses_[extractedByGraphAccess[currentId]];
          if (gmAccessesAreNoAlias(previousExtracted, currentExtracted)) {
            continue;
          }
          if (failed(retainNodePair(graphAccesses[previousId].node,
                                    current.node))) {
            return failure();
          }
        }
        return success();
      };
      const bool failedJoin =
          (currentWrites && (failed(joinActive(activeReaders)) ||
                             failed(joinActive(activeWriters)))) ||
          (!currentWrites && currentReads && failed(joinActive(activeWriters)));
      if (failedJoin) {
        return failure();
      }
      const ExtractedAccess &currentExtracted =
          extractedAccesses_[extractedByGraphAccess[currentId]];
      if (currentWrites && nodeIsInLoop[current.node] &&
          !gmAccessesAreNoAlias(currentExtracted, currentExtracted) &&
          failed(retainNodePair(current.node, current.node))) {
        return failure();
      }
      if (currentWrites) {
        activeWriters.insert(currentId);
        writerExpiry.push({current.extent.end, currentId});
      } else if (currentReads) {
        activeReaders.insert(currentId);
        readerExpiry.push({current.extent.end, currentId});
      }
    }
  }

  for (std::size_t currentIndex = 0; currentIndex < gmAccesses.size();
       ++currentIndex) {
    const SyncCoverStorageAccessId currentId = gmAccesses[currentIndex];
    const SyncCoverStorageAccess &current = graphAccesses[currentId];
    for (std::size_t previousIndex = 0; previousIndex < currentIndex;
         ++previousIndex) {
      const SyncCoverStorageAccessId previousId = gmAccesses[previousIndex];
      const SyncCoverStorageAccess &previous = graphAccesses[previousId];
      if (current.family == previous.family ||
          (!syncCoverStorageModeWrites(current.mode) &&
           !syncCoverStorageModeWrites(previous.mode))) {
        continue;
      }
      if (!consumePairInspection()) {
        return function_.emitError(
            "canonical sync pair-inspection limit exceeded");
      }
      const ExtractedAccess &currentExtracted =
          extractedAccesses_[extractedByGraphAccess[currentId]];
      const ExtractedAccess &previousExtracted =
          extractedAccesses_[extractedByGraphAccess[previousId]];
      if (gmAccessesAreNoAlias(previousExtracted, currentExtracted)) {
        continue;
      }
      if (failed(retainNodePair(previous.node, current.node))) {
        return failure();
      }
    }
  }

  storageConflictPeers_.assign(nodeBindings_.size(), {});
  for (const auto &[first, second] : nodePairs) {
    storageConflictPeers_[first].push_back(second);
    if (first != second) {
      storageConflictPeers_[second].push_back(first);
    }
  }
  for (std::vector<SyncCoverNodeId> &peers : storageConflictPeers_) {
    llvm::sort(peers);
  }
  return success();
}

bool ProgramBuilder::gmAccessesAreNoAlias(const ExtractedAccess &first,
                                          const ExtractedAccess &second) const {
  if (first.space != AddressSpace::GM || second.space != AddressSpace::GM) {
    return false;
  }
  if (options_.gmAliasPolicy ==
      CanonicalSyncGmAliasPolicy::AllAccessesNoAlias) {
    return true;
  }
  const std::optional<unsigned> firstArgument =
      getFunctionArgument(first.root, function_);
  const std::optional<unsigned> secondArgument =
      getFunctionArgument(second.root, function_);
  if (!firstArgument || !secondArgument || *firstArgument == *secondArgument) {
    return false;
  }
  const std::pair<unsigned, unsigned> pair =
      std::minmax(*firstArgument, *secondArgument);
  return options_.gmAliasPolicy ==
             CanonicalSyncGmAliasPolicy::DistinctArgumentsNoAlias ||
         noAliasArguments_.count(pair) != 0;
}

FailureOr<std::vector<std::pair<unsigned, unsigned>>>
ProgramBuilder::getOrdinalPairs(const ExtractedAccess &first,
                                const ExtractedAccess &second, Operation *loop,
                                unsigned distance) {
  const std::size_t firstCount = first.graphAccesses.size();
  const std::size_t secondCount = second.graphAccesses.size();
  if (firstCount == 1 && secondCount == 1) {
    return std::vector<std::pair<unsigned, unsigned>>{{0, 0}};
  }
  if (firstCount == secondCount && firstCount <= kMaximumSlotCount &&
      firstCount <= std::numeric_limits<std::uint32_t>::max()) {
    Value firstSlot = findMultiTileSlotExpr(first.base);
    Value secondSlot = findMultiTileSlotExpr(second.base);
    std::optional<std::int64_t> offset = getIterationOffset(loop, distance);
    if (firstSlot && secondSlot && offset) {
      Value shiftedSymbol;
      if (distance != 0) {
        shiftedSymbol = cast<scf::ForOp>(loop).getInductionVar();
      }
      auto exact = enumerateSlotSSAOrdinalPairs(
          firstSlot, secondSlot, static_cast<std::uint32_t>(firstCount),
          shiftedSymbol, *offset);
      if (exact) {
        std::vector<std::pair<unsigned, unsigned>> pairs;
        pairs.reserve(exact->size());
        for (const SlotOrdinalPair &pair : *exact) {
          pairs.push_back({pair.first, pair.second});
        }
        return pairs;
      }
    }
  }
  const bool inspectionOverflow =
      firstCount > std::numeric_limits<unsigned>::max() ||
      secondCount > std::numeric_limits<unsigned>::max() ||
      (firstCount != 0 &&
       secondCount >
           (options_.maximumPairInspections - pairInspections_) / firstCount);
  if (inspectionOverflow) {
    function_.emitError("canonical sync pair-inspection limit exceeded");
    return failure();
  }
  std::vector<std::pair<unsigned, unsigned>> conservative;
  conservative.reserve(firstCount * secondCount);
  for (std::size_t firstOrdinal = 0; firstOrdinal < firstCount;
       ++firstOrdinal) {
    for (std::size_t secondOrdinal = 0; secondOrdinal < secondCount;
         ++secondOrdinal) {
      conservative.push_back({static_cast<unsigned>(firstOrdinal),
                              static_cast<unsigned>(secondOrdinal)});
    }
  }
  return conservative;
}

std::optional<std::int64_t>
ProgramBuilder::getIterationOffset(Operation *loop, unsigned distance) const {
  if (distance == 0) {
    return 0;
  }
  auto forOp = dyn_cast_or_null<scf::ForOp>(loop);
  APInt step;
  const bool invalidLoop =
      !forOp || !matchPattern(forOp.getStep(), m_ConstantInt(&step)) ||
      !step.isStrictlyPositive() || step.getActiveBits() > 63;
  if (invalidLoop) {
    return std::nullopt;
  }
  const std::uint64_t value = step.getZExtValue();
  const bool offsetOverflows =
      value >
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
          distance;
  if (offsetOverflows) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(value * distance);
}
