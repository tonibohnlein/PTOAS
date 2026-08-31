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
#include <numeric>
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
  case AddressSpace::MAT:
    return SyncCoverStorageDomainRole::L1Tile;
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

bool checkedAddSize(std::size_t &value, std::size_t amount) {
  const bool overflows =
      amount > std::numeric_limits<std::size_t>::max() - value;
  if (overflows) {
    return false;
  }
  value += amount;
  return true;
}

bool checkedMultiplySize(std::size_t first, std::size_t second,
                         std::size_t &result) {
  const bool overflows =
      first != 0 && second > std::numeric_limits<std::size_t>::max() / first;
  if (overflows) {
    return false;
  }
  result = first * second;
  return true;
}

SyncCoverStorageAccessPath getAccessPath(const SyncCoverGraph &graph,
                                         SyncCoverNodeId node,
                                         AddressSpace space) {
  const bool scalarGlobalMemory =
      space == AddressSpace::GM &&
      graph.getNodes()[node].resource ==
          static_cast<std::uint32_t>(PipelineType::PIPE_S);
  return scalarGlobalMemory ? SyncCoverStorageAccessPath::ScalarDCache
                            : SyncCoverStorageAccessPath::PhysicalPipeline;
}

std::size_t logarithmicWorkBound(std::size_t count) {
  std::size_t result = 1;
  while (count > 1) {
    count = count / 2 + count % 2;
    ++result;
  }
  return result;
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

std::optional<std::vector<std::vector<std::uint32_t>>>
getReachableSymbolResiduesByPhase(
    Operation *loop, std::uint32_t modulus,
    const std::vector<std::size_t> *reachableSourcePhases,
    std::size_t phasePeriod) {
  if (!reachableSourcePhases) {
    return std::nullopt;
  }
  if (modulus == 0 || phasePeriod == 0) {
    return std::vector<std::vector<std::uint32_t>>{};
  }
  auto forOp = dyn_cast_or_null<scf::ForOp>(loop);
  APInt lower;
  APInt step;
  const bool unavailable =
      !forOp || !matchPattern(forOp.getLowerBound(), m_ConstantInt(&lower)) ||
      !matchPattern(forOp.getStep(), m_ConstantInt(&step)) ||
      lower.isNegative() || !step.isStrictlyPositive() ||
      lower.getActiveBits() > 64 || step.getActiveBits() > 64;
  if (unavailable) {
    return std::nullopt;
  }
  const std::size_t divisor =
      std::gcd(phasePeriod, static_cast<std::size_t>(modulus));
  const std::size_t reduced = phasePeriod / divisor;
  const bool horizonOverflows =
      reduced > std::numeric_limits<std::size_t>::max() / modulus;
  if (horizonOverflows) {
    return std::nullopt;
  }
  const std::size_t horizon = reduced * modulus;
  const std::uint64_t lowerResidue = lower.getZExtValue() % modulus;
  const std::uint64_t stepResidue = step.getZExtValue() % modulus;
  std::vector<std::vector<std::uint32_t>> residuesByPhase(phasePeriod);
  for (std::size_t iteration = 0; iteration < horizon; ++iteration) {
    const std::size_t phase = iteration % phasePeriod;
    if (!std::binary_search(reachableSourcePhases->begin(),
                            reachableSourcePhases->end(), phase)) {
      continue;
    }
    const std::uint64_t iterationResidue = iteration % modulus;
    const std::uint64_t residue =
        (lowerResidue + stepResidue * iterationResidue) % modulus;
    residuesByPhase[phase].push_back(static_cast<std::uint32_t>(residue));
  }
  for (std::vector<std::uint32_t> &residues : residuesByPhase) {
    llvm::sort(residues);
    residues.erase(std::unique(residues.begin(), residues.end()),
                   residues.end());
  }
  return residuesByPhase;
}

std::optional<std::size_t>
getRestrictedOrdinalWorkBound(std::uint32_t modulus, std::size_t phasePeriod,
                              std::size_t reachablePhases) {
  if (modulus == 0 || phasePeriod == 0) {
    return std::nullopt;
  }
  const std::size_t divisor =
      std::gcd(phasePeriod, static_cast<std::size_t>(modulus));
  const std::size_t reduced = phasePeriod / divisor;
  const bool horizonOverflows =
      reduced > std::numeric_limits<std::size_t>::max() / modulus;
  if (horizonOverflows) {
    return std::nullopt;
  }
  const std::size_t horizon = reduced * modulus;
  std::size_t taggedPairs = 0;
  if (!checkedMultiplySize(modulus, reachablePhases, taggedPairs)) {
    return std::nullopt;
  }
  std::size_t workItems = horizon;
  const bool workItemsUnavailable =
      !checkedAddSize(workItems, taggedPairs) ||
      !checkedAddSize(workItems, reachablePhases) ||
      !checkedAddSize(workItems, 1);
  if (workItemsUnavailable) {
    return std::nullopt;
  }
  std::size_t searchUnits = reachablePhases;
  std::size_t phaseSortUnits = horizon;
  std::size_t pairSortUnits = taggedPairs;
  const bool sortUnitsUnavailable = !checkedAddSize(searchUnits, 1) ||
                                    !checkedAddSize(phaseSortUnits, 1) ||
                                    !checkedAddSize(pairSortUnits, 1);
  if (sortUnitsUnavailable) {
    return std::nullopt;
  }
  std::size_t perItemWork = logarithmicWorkBound(searchUnits);
  const bool perItemWorkUnavailable =
      !checkedAddSize(perItemWork, logarithmicWorkBound(phaseSortUnits)) ||
      !checkedAddSize(perItemWork, logarithmicWorkBound(pairSortUnits)) ||
      !checkedAddSize(perItemWork, 16);
  if (perItemWorkUnavailable) {
    return std::nullopt;
  }
  std::size_t result = 0;
  if (!checkedMultiplySize(workItems, perItemWork, result)) {
    return std::nullopt;
  }
  return result;
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
          graph_.addStorageDomain(getStorageDomainRole(access.space),
                                  static_cast<std::uint32_t>(access.space));
      if (!added) {
        return function_.emitError(
            "cannot construct canonical sync storage domain");
      }
      domainPosition =
          storageDomains_.emplace(access.space, *added.index).first;
    }
    const SyncCoverStorageDomainId domain = domainPosition->second;
    const SyncCoverStorageAccessPath path =
        getAccessPath(graph_, node, access.space);

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
          access.mode, static_cast<unsigned>(ordinal), exactLocal, path);
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
      {0, std::numeric_limits<std::uint64_t>::max()}, access.mode, std::nullopt,
      false, getAccessPath(graph_, access.node, access.space));
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

FailureOr<std::vector<OrdinalPairPhaseState>> ProgramBuilder::getOrdinalPairs(
    const ExtractedAccess &first, const ExtractedAccess &second,
    Operation *loop, unsigned distance,
    const std::vector<std::size_t> *reachableSourcePhases,
    std::size_t phasePeriod) {
  const std::size_t firstCount = first.graphAccesses.size();
  const std::size_t secondCount = second.graphAccesses.size();
  std::uint16_t sourcePhaseMask = 1;
  if (reachableSourcePhases) {
    sourcePhaseMask = 0;
    for (std::size_t phase : *reachableSourcePhases) {
      if (phase >= kCanonicalSyncMaximumPeriodicRecurrenceStates) {
        return failure();
      }
      sourcePhaseMask |= std::uint16_t{1} << phase;
    }
  }
  if (firstCount == 1 && secondCount == 1) {
    return std::vector<OrdinalPairPhaseState>{{0, 0, sourcePhaseMask}};
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
      const std::uint32_t slotCount = static_cast<std::uint32_t>(firstCount);
      if (reachableSourcePhases) {
        const std::optional<std::size_t> work = getRestrictedOrdinalWorkBound(
            slotCount, phasePeriod, reachableSourcePhases->size());
        if (!work || !consumePairInspections(*work)) {
          function_.emitError("canonical sync pair-inspection limit exceeded");
          return failure();
        }
      }
      const std::optional<std::vector<std::vector<std::uint32_t>>>
          residuesByPhase = getReachableSymbolResiduesByPhase(
              loop, slotCount, reachableSourcePhases, phasePeriod);
      if (residuesByPhase) {
        std::vector<OrdinalPairPhaseState> pairs;
        bool exactAvailable = true;
        for (std::size_t phase : *reachableSourcePhases) {
          if (phase >= residuesByPhase->size()) {
            exactAvailable = false;
            break;
          }
          const auto exact = enumerateSlotSSAOrdinalPairsForResidues(
              firstSlot, secondSlot, slotCount, (*residuesByPhase)[phase],
              shiftedSymbol, *offset);
          if (!exact) {
            exactAvailable = false;
            break;
          }
          for (const SlotOrdinalPair &pair : *exact) {
            pairs.push_back(
                {pair.first, pair.second,
                 static_cast<std::uint16_t>(std::uint16_t{1} << phase)});
          }
        }
        if (exactAvailable) {
          llvm::sort(pairs, [](const OrdinalPairPhaseState &lhs,
                               const OrdinalPairPhaseState &rhs) {
            return std::tie(lhs.first, lhs.second) <
                   std::tie(rhs.first, rhs.second);
          });
          std::vector<OrdinalPairPhaseState> merged;
          merged.reserve(pairs.size());
          for (const OrdinalPairPhaseState &pair : pairs) {
            const bool newPair = merged.empty() ||
                                 merged.back().first != pair.first ||
                                 merged.back().second != pair.second;
            if (newPair) {
              merged.push_back(pair);
            } else {
              merged.back().sourcePhases |= pair.sourcePhases;
            }
          }
          return merged;
        }
      } else {
        const auto exact = enumerateSlotSSAOrdinalPairs(
            firstSlot, secondSlot, slotCount, shiftedSymbol, *offset);
        if (exact) {
          std::vector<OrdinalPairPhaseState> pairs;
          pairs.reserve(exact->size());
          for (const SlotOrdinalPair &pair : *exact) {
            pairs.push_back({pair.first, pair.second, sourcePhaseMask});
          }
          return pairs;
        }
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
  std::vector<OrdinalPairPhaseState> conservative;
  conservative.reserve(firstCount * secondCount);
  for (std::size_t firstOrdinal = 0; firstOrdinal < firstCount;
       ++firstOrdinal) {
    for (std::size_t secondOrdinal = 0; secondOrdinal < secondCount;
         ++secondOrdinal) {
      conservative.push_back({static_cast<unsigned>(firstOrdinal),
                              static_cast<unsigned>(secondOrdinal),
                              sourcePhaseMask});
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
