// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/CanonicalSyncModel.h"

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;

bool CanonicalPhysicalResource::operator<(
    const CanonicalPhysicalResource &other) const {
  return std::tie(core, pipe) < std::tie(other.core, other.pipe);
}

std::optional<std::uint64_t> CanonicalByteInterval::end() const {
  const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
  if (size > maximum - begin) {
    return std::nullopt;
  }
  return begin + size;
}

template <typename Record>
static std::uint32_t appendRecord(llvm::SmallVectorImpl<Record> &records,
                                  Record record, bool frozen) {
  const bool idExhausted = records.size() >= kInvalidCanonicalSyncId;
  if (frozen || idExhausted) {
    llvm_unreachable("cannot append to a frozen or exhausted sync model");
  }
  const auto id = static_cast<std::uint32_t>(records.size());
  record.id = id;
  records.push_back(std::move(record));
  return id;
}

CanonicalRegionId CanonicalSyncProgram::appendRegion(CanonicalRegion region) {
  const CanonicalRegionId parent = region.parent;
  const CanonicalRegionId id =
      appendRecord(regions, std::move(region), graphFrozen);
  regionChildren.emplace_back();
  if (parent != kInvalidCanonicalSyncId) {
    if (parent >= id) {
      llvm_unreachable("canonical region parent must precede its child");
    }
    regionChildren[parent].push_back(id);
  }
  return id;
}

CanonicalPhaseId CanonicalSyncProgram::appendPhase(CanonicalPhase phase) {
  return appendRecord(phases, std::move(phase), graphFrozen);
}

CanonicalAccessId CanonicalSyncProgram::appendAccess(CanonicalAccess access) {
  return appendRecord(accesses, std::move(access), graphFrozen);
}

CanonicalFenceEffectId
CanonicalSyncProgram::appendFenceEffect(CanonicalFenceEffect effect) {
  return appendRecord(fenceEffects, std::move(effect), graphFrozen);
}

CanonicalDemandId CanonicalSyncProgram::appendDemand(CanonicalDemand demand) {
  return appendRecord(demands, std::move(demand), graphFrozen);
}

void CanonicalSyncProgram::retainDemands(const llvm::BitVector &retained) {
  const bool invalidState = graphFrozen || frozen ||
                            retained.size() != demands.size() ||
                            !mechanisms.empty() || !directMechanisms.empty();
  if (invalidState) {
    llvm_unreachable("cannot compact a frozen canonical demand graph");
  }
  SmallVector<CanonicalDemand, 0> remaining;
  remaining.reserve(retained.count());
  for (const CanonicalDemand &demand : demands) {
    if (!retained.test(demand.id)) {
      continue;
    }
    CanonicalDemand kept = demand;
    kept.id = remaining.size();
    remaining.push_back(std::move(kept));
  }
  demands = std::move(remaining);
}

void CanonicalSyncProgram::appendDemandCause(CanonicalDemandId demand,
                                             CanonicalDemandCause cause) {
  if (graphFrozen || demand >= demands.size()) {
    llvm_unreachable("cannot extend an invalid or frozen demand");
  }
  demands[demand].causes.push_back(std::move(cause));
}

CanonicalMechanismId
CanonicalSyncProgram::appendMechanism(CanonicalMechanism mechanism) {
  const CanonicalMechanismId id =
      appendRecord(mechanisms, std::move(mechanism),
                   frozen || !buildingMechanisms || mechanismCatalogComplete ||
                       setCoverInstance.has_value());
  const CanonicalMechanism &stored = mechanisms[id];
  SmallVector<CanonicalRegionId, 2> loops;
  for (CanonicalRegionId region = stored.actionRegion;
       region != kInvalidCanonicalSyncId; region = regions[region].parent) {
    if (regions[region].kind == CanonicalRegionKind::Loop) {
      loops.push_back(region);
    }
  }
  llvm::sort(loops);
  mechanismExecutionLoops.push_back(std::move(loops));

  SmallVector<CanonicalPhaseId, 8> prefix;
  for (const CanonicalPhase &phase : phases) {
    if (statistics) {
      ++statistics->mechanismPrefixPhaseTests;
    }
    bool matchingResource = phase.resource == stored.source;
    if (stored.kind == CanonicalMechanismKind::FixedFence &&
        stored.fenceEffect) {
      matchingResource = llvm::is_contained(
          fenceEffects[*stored.fenceEffect].drainedResources, phase.resource);
    }
    if (matchingResource && stored.sourcePoint.operation &&
        canonical_sync_detail::phaseMayPrecedePoint(phase,
                                                    stored.sourcePoint) &&
        canonical_sync_detail::controlsCanCoexecute(phase.controlPath,
                                                    stored.guard)) {
      prefix.push_back(phase.id);
    }
  }
  if (statistics) {
    statistics->precomputedPrefixEntries += prefix.size();
  }
  mechanismSourcePrefixes.push_back(std::move(prefix));
  return id;
}

void CanonicalSyncProgram::appendMechanismOrigin(CanonicalMechanismId mechanism,
                                                 CanonicalDemandId demand) {
  const bool invalidMechanism = mechanism >= mechanisms.size();
  const bool invalidDemand = demand >= demands.size();
  if (frozen || !buildingMechanisms || mechanismCatalogComplete ||
      setCoverInstance || invalidMechanism || invalidDemand) {
    llvm_unreachable("cannot extend an invalid or frozen mechanism");
  }
  if (!llvm::is_contained(mechanisms[mechanism].origins, demand)) {
    mechanisms[mechanism].origins.push_back(demand);
  }
}

void CanonicalSyncProgram::appendMechanismCacheMaintenance(
    CanonicalMechanismId mechanism, ArrayRef<Operation *> actions) {
  if (frozen || !buildingMechanisms || mechanismCatalogComplete ||
      setCoverInstance || mechanism >= mechanisms.size()) {
    llvm_unreachable("cannot extend an invalid or frozen mechanism");
  }
  for (Operation *action : actions) {
    if (!llvm::is_contained(mechanisms[mechanism].cacheMaintenance, action)) {
      mechanisms[mechanism].cacheMaintenance.push_back(action);
    }
  }
}

void CanonicalSyncProgram::setMechanismEventId(CanonicalMechanismId mechanism,
                                               unsigned eventId) {
  if (frozen || mechanism >= mechanisms.size()) {
    llvm_unreachable("cannot allocate an invalid or frozen mechanism");
  }
  mechanisms[mechanism].eventId = eventId;
}

void CanonicalSyncProgram::setMechanismReleaseEventId(
    CanonicalMechanismId mechanism, unsigned eventId) {
  if (frozen || mechanism >= mechanisms.size()) {
    llvm_unreachable("cannot allocate an invalid or frozen mechanism");
  }
  mechanisms[mechanism].releaseEventId = eventId;
}

void CanonicalSyncProgram::setScarcityEventGroups(
    SmallVector<CanonicalScarcityEventGroup, 2> groups) {
  if (frozen || !setCoverSolution ||
      !setCoverSolution->scarcityEventGroups.empty()) {
    llvm_unreachable("cannot replace a frozen event-scarcity plan");
  }
  setCoverSolution->scarcityEventGroups = std::move(groups);
}

void CanonicalSyncProgram::setDirectMechanism(CanonicalDemandId demand,
                                              CanonicalMechanismId mechanism) {
  const bool invalidDemand = demand >= demands.size();
  const bool invalidMechanism = mechanism >= mechanisms.size();
  if (frozen || !buildingMechanisms || mechanismCatalogComplete ||
      setCoverInstance || invalidDemand || invalidMechanism) {
    llvm_unreachable("cannot map an invalid or frozen direct mechanism");
  }
  if (directMechanisms.empty()) {
    directMechanisms.assign(demands.size(), kInvalidCanonicalSyncId);
  }
  directMechanisms[demand] = mechanism;
}

void CanonicalSyncProgram::appendCoverageWorld(CanonicalCoverageWorld world) {
  if (frozen || !mechanismCatalogComplete || coverageCatalogComplete ||
      setCoverInstance) {
    llvm_unreachable("cannot append to a frozen sync model");
  }
  coverageWorlds.push_back(std::move(world));
}

void CanonicalSyncProgram::setSetCoverInstance(
    CanonicalSetCoverInstance instance) {
  if (frozen || setCoverInstance || setCoverSolution) {
    llvm_unreachable("cannot replace a frozen canonical set-cover instance");
  }
  setCoverInstance = std::move(instance);
}

void CanonicalSyncProgram::setSetCoverSolution(
    CanonicalSetCoverSolution solution) {
  if (frozen || !setCoverInstance || setCoverSolution) {
    llvm_unreachable("cannot replace a frozen canonical set-cover solution");
  }
  setCoverSolution = std::move(solution);
}

static bool validControlPath(llvm::ArrayRef<CanonicalControlAtom> path,
                             size_t regionCount) {
  return llvm::all_of(path, [regionCount](const CanonicalControlAtom &atom) {
    return atom.choice < regionCount;
  });
}

LogicalResult CanonicalSyncProgram::freezeGraph() {
  if (graphFrozen) {
    return success();
  }
  const auto fail = [this](llvm::Twine message) {
    function.emitError("invalid canonical synchronization model: ") << message;
    return failure();
  };
  const bool invalidRegionIndexSize = regionChildren.size() != regions.size();
  if (invalidRegionIndexSize) {
    return fail("region hierarchy index has an invalid size");
  }
  for (const CanonicalRegion &region : regions) {
    if (region.id != 0 && region.parent >= regions.size()) {
      return fail("region has an invalid parent");
    }
    const ArrayRef<CanonicalRegionId> children = regionChildren[region.id];
    const bool invalidChildren =
        !llvm::is_sorted(children) ||
        std::adjacent_find(children.begin(), children.end()) != children.end() ||
        llvm::any_of(children, [&](CanonicalRegionId child) {
          return child >= regions.size() || child <= region.id ||
                 regions[child].parent != region.id;
        });
    const bool missingFromParent =
        region.id != 0 &&
        !llvm::is_contained(regionChildren[region.parent], region.id);
    if (invalidChildren || missingFromParent) {
      return fail("region hierarchy index is inconsistent");
    }
  }
  for (const CanonicalPhase &phase : phases) {
    const bool invalidRegion = phase.region >= regions.size();
    const bool invalidControl =
        !validControlPath(phase.controlPath, regions.size());
    if (!phase.operation || invalidRegion || invalidControl) {
      return fail("phase has an invalid operation, region, or control path");
    }
  }
  for (const CanonicalAccess &access : accesses) {
    if (access.phase >= phases.size()) {
      return fail("access has an invalid phase");
    }
    for (const CanonicalByteInterval &interval : access.intervals) {
      if (!interval.end()) {
        return fail("access byte interval overflows uint64_t");
      }
    }
  }
  for (const CanonicalFenceEffect &effect : fenceEffects) {
    const bool invalidRegion = effect.region >= regions.size();
    const bool invalidGuard = !validControlPath(effect.guard, regions.size());
    const bool invalidLoop =
        llvm::any_of(effect.loopPath, [this](CanonicalRegionId id) {
          return id >= regions.size() ||
                 regions[id].kind != CanonicalRegionKind::Loop;
        });
    if (!effect.operation || invalidRegion || invalidGuard || invalidLoop ||
        effect.drainedResources.empty()) {
      return fail("fence effect has an invalid operation, region, or scope");
    }
  }
  for (const CanonicalDemand &demand : demands) {
    const bool validTarget = demand.kind == CanonicalDemandKind::ExitCompletion
                                 ? demand.target == kInvalidCanonicalSyncId
                                 : demand.target < phases.size();
    const bool validSource = demand.source < phases.size();
    const bool validOwner = demand.owner < regions.size();
    const bool validGuards =
        validControlPath(demand.sourceGuard, regions.size()) &&
        validControlPath(demand.targetGuard, regions.size());
    const bool validDistance = llvm::all_of(
        demand.iterationDistance,
        [this](const CanonicalLoopDistance &distance) {
          return distance.loop < regions.size() &&
                 regions[distance.loop].kind == CanonicalRegionKind::Loop;
        });
    const bool validVisibility =
        (demand.requirement == CanonicalRequirement::Visibility) ==
        demand.visibility.has_value();
    if (!validSource || !validTarget || !validOwner || !validGuards ||
        !validDistance || !validVisibility) {
      return fail("demand has an invalid endpoint, owner, guard, or distance");
    }
  }
  graphFrozen = true;
  return success();
}

LogicalResult CanonicalSyncProgram::freeze() {
  if (frozen) {
    return success();
  }
  if (failed(freezeGraph())) {
    return failure();
  }
  const auto fail = [this](llvm::Twine message) {
    function.emitError("invalid canonical synchronization plan: ") << message;
    return failure();
  };
  const bool invalidReachabilityCacheSize =
      mechanismExecutionLoops.size() != mechanisms.size() ||
      mechanismSourcePrefixes.size() != mechanisms.size();
  if (invalidReachabilityCacheSize) {
    return fail("mechanism reachability cache has an invalid size");
  }
  for (const CanonicalMechanism &mechanism : mechanisms) {
    const bool tail = mechanism.kind == CanonicalMechanismKind::TailBarrier;
    const bool hasBothPoints =
        mechanism.sourcePoint.operation && mechanism.targetPoint.operation;
    const bool hasNeitherPoint =
        !mechanism.sourcePoint.operation && !mechanism.targetPoint.operation;
    const bool validPoints =
        tail ? hasBothPoints || hasNeitherPoint : hasBothPoints;
    const bool validOrigins =
        llvm::all_of(mechanism.origins, [this](CanonicalDemandId demand) {
          return demand < demands.size();
        });
    const bool validFence =
        mechanism.kind == CanonicalMechanismKind::FixedFence
            ? mechanism.fenceEffect &&
                  *mechanism.fenceEffect < fenceEffects.size()
            : !mechanism.fenceEffect;
    const bool validGeneratedCacheMaintenance =
        (mechanism.kind == CanonicalMechanismKind::VisibilityFence) ==
        mechanism.generatedCacheMaintenance.has_value();
    const bool recurring =
        mechanism.kind == CanonicalMechanismKind::RecurringEvent;
    const bool validRecurrence =
        recurring == mechanism.recurrenceLoop.has_value() &&
        (!mechanism.boundaryRecurring || recurring) &&
        (!mechanism.recurrenceLoop ||
         (*mechanism.recurrenceLoop < regions.size() &&
          regions[*mechanism.recurrenceLoop].kind ==
              CanonicalRegionKind::Loop));
    const ArrayRef<CanonicalRegionId> cachedLoops =
        mechanismExecutionLoops[mechanism.id];
    const ArrayRef<CanonicalPhaseId> cachedPrefix =
        mechanismSourcePrefixes[mechanism.id];
    const bool validCachedLoops =
        llvm::is_sorted(cachedLoops) &&
        std::adjacent_find(cachedLoops.begin(), cachedLoops.end()) ==
            cachedLoops.end() &&
        llvm::all_of(cachedLoops, [this](CanonicalRegionId id) {
          return id < regions.size() &&
                 regions[id].kind == CanonicalRegionKind::Loop;
        });
    const bool validCachedPrefix =
        llvm::is_sorted(cachedPrefix) &&
        std::adjacent_find(cachedPrefix.begin(), cachedPrefix.end()) ==
            cachedPrefix.end() &&
        llvm::all_of(cachedPrefix, [this](CanonicalPhaseId id) {
          return id < phases.size();
        });
    if (!validPoints || !validOrigins || !validFence ||
        !validGeneratedCacheMaintenance || !validRecurrence ||
        !validCachedLoops || !validCachedPrefix ||
        mechanism.actionRegion >= regions.size()) {
      return fail("mechanism has an invalid action point, origin, or region");
    }
  }
  for (const CanonicalCoverageWorld &world : coverageWorlds) {
    const bool invalidMechanism =
        llvm::any_of(world.mechanisms, [this](CanonicalMechanismId id) {
          return id >= mechanisms.size();
        });
    const bool invalidDemand = llvm::any_of(world.covered,
                                            [this](CanonicalDemandId id) {
                                              return id >= demands.size();
                                            }) ||
                               llvm::any_of(world.differentialDisagreements,
                                            [this](CanonicalDemandId id) {
                                              return id >= demands.size();
                                            });
    const bool invalidSummary = llvm::any_of(
        world.summaries, [this](const CanonicalRegionSummary &summary) {
          const bool invalidRegion = summary.region >= regions.size();
          const bool invalidChild =
              llvm::any_of(summary.children, [this](CanonicalRegionId id) {
                return id >= regions.size();
              });
          if (invalidRegion || invalidChild) {
            return true;
          }
          const bool invalidCompletion = llvm::any_of(
              summary.completions,
              [this](const CanonicalCompletionTransfer &completion) {
                return completion.phase >= phases.size() ||
                       llvm::any_of(completion.requiredLoops,
                                    [this](CanonicalRegionId id) {
                                      return id >= regions.size() ||
                                             regions[id].kind !=
                                                 CanonicalRegionKind::Loop;
                                    });
              });
          const bool invalidTransfer = llvm::any_of(
              summary.transfers,
              [this](const CanonicalBoundaryTransfer &transfer) {
                return llvm::any_of(
                    transfer.requiredLoops, [this](CanonicalRegionId id) {
                      return id >= regions.size() ||
                             regions[id].kind != CanonicalRegionKind::Loop;
                    });
              });
          return invalidCompletion || invalidTransfer;
        });
    if (invalidMechanism || invalidDemand || invalidSummary ||
        !world.differentialDisagreements.empty() ||
        !world.flattenedOracleMatched || !world.unrolledOracleAvailable ||
        (world.unrolledOracleExhaustive && !world.unrolledOracleMatched)) {
      return fail("coverage world references an invalid ID");
    }
  }
  if (!setCoverInstance) {
    return fail("set-cover instance is missing");
  }
  if (buildingMechanisms || !mechanismCatalogComplete ||
      !coverageCatalogComplete) {
    return fail("mechanism or coverage catalog is incomplete");
  }
  const bool invalidBaseline =
      llvm::any_of(setCoverInstance->baseline, [this](CanonicalMechanismId id) {
        return id >= mechanisms.size();
      });
  const bool invalidUniverse =
      llvm::any_of(
          setCoverInstance->universe,
          [this](CanonicalDemandId id) { return id >= demands.size(); }) ||
      setCoverInstance->universe.size() != demands.size() ||
      !llvm::is_sorted(setCoverInstance->universe) ||
      std::adjacent_find(setCoverInstance->universe.begin(),
                         setCoverInstance->universe.end()) !=
          setCoverInstance->universe.end() ||
      setCoverInstance->providersByDemand.size() != demands.size();
  SmallVector<uint8_t, 8> providerCoverage(demands.size(), 0U);
  bool invalidCandidate = false;
  bool invalidProviders = false;
  for (const CanonicalSetCoverCandidate &candidate :
       setCoverInstance->candidates) {
    const bool wrongId =
        candidate.id >= setCoverInstance->candidates.size() ||
        &candidate != &setCoverInstance->candidates[candidate.id];
    const bool invalidMechanism =
        candidate.mechanisms.size() != 1U ||
        candidate.mechanisms.front() >= mechanisms.size();
    const bool invalidDemand =
        llvm::any_of(
            candidate.directOrigins,
            [this](CanonicalDemandId id) { return id >= demands.size(); }) ||
        llvm::any_of(
            candidate.additionalCoverage,
            [this](CanonicalDemandId id) { return id >= demands.size(); }) ||
        llvm::any_of(candidate.coveredDemands, [this](CanonicalDemandId id) {
          return id >= demands.size();
        });
    const bool overlap = llvm::any_of(
        candidate.directOrigins, [&candidate](CanonicalDemandId id) {
          return llvm::is_contained(candidate.additionalCoverage, id);
        });
    const bool invalidCoverageOrder =
        !llvm::is_sorted(candidate.directOrigins) ||
        std::adjacent_find(candidate.directOrigins.begin(),
                           candidate.directOrigins.end()) !=
            candidate.directOrigins.end() ||
        !llvm::is_sorted(candidate.additionalCoverage) ||
        std::adjacent_find(candidate.additionalCoverage.begin(),
                           candidate.additionalCoverage.end()) !=
            candidate.additionalCoverage.end() ||
        !llvm::is_sorted(candidate.coveredDemands) ||
        std::adjacent_find(candidate.coveredDemands.begin(),
                           candidate.coveredDemands.end()) !=
            candidate.coveredDemands.end();
    SmallVector<CanonicalDemandId, 8> listedCoverage(
        candidate.directOrigins.begin(), candidate.directOrigins.end());
    llvm::append_range(listedCoverage, candidate.additionalCoverage);
    llvm::sort(listedCoverage);
    const bool inconsistentIncidence =
        ArrayRef<CanonicalDemandId>(listedCoverage) !=
        ArrayRef<CanonicalDemandId>(candidate.coveredDemands);
    if (!invalidDemand) {
      for (CanonicalDemandId demand : candidate.coveredDemands) {
        providerCoverage[demand] = 1U;
        const bool providersAvailable =
            setCoverInstance->providersByDemand.size() == demands.size();
        if (providersAvailable) {
          invalidProviders |= !llvm::is_contained(
              setCoverInstance->providersByDemand[demand], candidate.id);
        }
      }
    }
    invalidCandidate |= wrongId || invalidMechanism || invalidDemand ||
                        inconsistentIncidence || invalidCoverageOrder ||
                        overlap ||
                        candidate.weight == 0;
  }
  const bool providersAvailable =
      setCoverInstance->providersByDemand.size() == demands.size();
  if (providersAvailable) {
    for (CanonicalDemandId demand = 0; demand < demands.size(); ++demand) {
      const ArrayRef<CanonicalSetCoverCandidateId> providers =
          setCoverInstance->providersByDemand[demand];
      invalidProviders |=
          !llvm::is_sorted(providers) ||
          std::adjacent_find(providers.begin(), providers.end()) !=
              providers.end();
      for (CanonicalSetCoverCandidateId candidate : providers) {
        invalidProviders |=
            candidate >= setCoverInstance->candidates.size() ||
            (candidate < setCoverInstance->candidates.size() &&
             !llvm::is_contained(
                 setCoverInstance->candidates[candidate].coveredDemands,
                 demand));
      }
      invalidProviders |= setCoverInstance->providersByDemand[demand].empty();
    }
  }
  if (invalidBaseline || invalidUniverse || invalidCandidate ||
      invalidProviders || llvm::is_contained(providerCoverage, 0U)) {
    return fail("set-cover instance references an invalid ID or incidence");
  }
  if (!setCoverSolution) {
    return fail("set-cover solution is missing");
  }
  {
    const bool invalidGreedyCandidate =
        llvm::any_of(setCoverSolution->greedyCandidates,
                     [this](CanonicalSetCoverCandidateId id) {
                       return id >= setCoverInstance->candidates.size();
                     });
    const bool invalidMechanism = llvm::any_of(
        setCoverSolution->mechanisms,
        [this](CanonicalMechanismId id) { return id >= mechanisms.size(); });
    const bool invalidOrder =
        !llvm::is_sorted(setCoverSolution->mechanisms) ||
        std::adjacent_find(setCoverSolution->mechanisms.begin(),
                           setCoverSolution->mechanisms.end()) !=
            setCoverSolution->mechanisms.end();
    const bool missingBaseline = llvm::any_of(
        setCoverInstance->baseline, [this](CanonicalMechanismId id) {
          return !llvm::is_contained(setCoverSolution->mechanisms, id);
        });
    const bool invalidDeletion = llvm::any_of(
        setCoverSolution->reverseDeleted, [this](CanonicalMechanismId id) {
          return id >= mechanisms.size() ||
                 llvm::is_contained(setCoverInstance->baseline, id) ||
                 llvm::is_contained(setCoverSolution->mechanisms, id);
        });
    SmallVector<uint8_t, 8> selectedCoverage(demands.size(), 0U);
    for (const CanonicalSetCoverCandidate &candidate :
         setCoverInstance->candidates) {
      if (!llvm::is_contained(setCoverSolution->mechanisms,
                              candidate.mechanisms.front())) {
        continue;
      }
      for (CanonicalDemandId demand : candidate.coveredDemands) {
        selectedCoverage[demand] = 1U;
      }
    }
    const bool incompleteCoverage = llvm::is_contained(selectedCoverage, 0U);
    const std::uint64_t expectedWeight =
        static_cast<std::uint64_t>(llvm::count_if(
            setCoverSolution->mechanisms, [this](CanonicalMechanismId id) {
              return !llvm::is_contained(setCoverInstance->baseline, id);
            }));
    llvm::BitVector scarcityMembers(mechanisms.size());
    bool invalidScarcityGroup = false;
    for (const CanonicalScarcityEventGroup &group :
         setCoverSolution->scarcityEventGroups) {
      const bool invalidSize = group.members.size() < 2;
      for (CanonicalMechanismId id : group.members) {
        const bool invalidMember =
            id >= mechanisms.size() || scarcityMembers.test(id) ||
            !llvm::is_contained(setCoverSolution->mechanisms, id) ||
            (id < mechanisms.size() &&
             mechanisms[id].kind != CanonicalMechanismKind::Event &&
             mechanisms[id].kind != CanonicalMechanismKind::RecurringEvent);
        invalidScarcityGroup |= invalidMember;
        if (id < mechanisms.size()) {
          scarcityMembers.set(id);
          const CanonicalMechanism &member = mechanisms[id];
          if (group.kind == CanonicalScarcityEventKind::Serialized) {
            invalidScarcityGroup |=
                member.kind != CanonicalMechanismKind::Event;
          } else {
            invalidScarcityGroup |=
                member.recurrenceLoop != group.recurrenceLoop ||
                !canonical_sync_detail::programPointMustPrecede(
                    member.sourcePoint, group.sourcePoint) ||
                !canonical_sync_detail::programPointMustPrecede(
                    group.targetPoint, member.targetPoint);
          }
        }
      }
      const bool hasRepresentative =
          !group.members.empty() && group.members.front() < mechanisms.size();
      if (hasRepresentative) {
        const CanonicalMechanism &representative =
            mechanisms[group.members.front()];
        invalidScarcityGroup |= group.source != representative.source ||
                                group.target != representative.target ||
                                group.guard != representative.guard;
      }
      const bool releaseRequired =
          group.kind == CanonicalScarcityEventKind::Serialized ||
          group.recurrenceLoop.has_value();
      invalidScarcityGroup |=
          invalidSize || group.releaseEventId.has_value() != releaseRequired;
      invalidScarcityGroup |=
          group.kind == CanonicalScarcityEventKind::Serialized
              ? (group.sourcePoint.operation || group.targetPoint.operation ||
                 group.recurrenceLoop.has_value())
              : (!group.sourcePoint.operation || !group.targetPoint.operation ||
                 !canonical_sync_detail::programPointMustPrecede(
                     group.sourcePoint, group.targetPoint));
    }
    bool invalidEventAssignment = false;
    for (const CanonicalMechanism &mechanism : mechanisms) {
      const bool selected =
          llvm::is_contained(setCoverSolution->mechanisms, mechanism.id);
      const bool mustHaveEvent =
          selected &&
          ((mechanism.kind == CanonicalMechanismKind::Event &&
            !scarcityMembers.test(mechanism.id)) ||
           mechanism.kind == CanonicalMechanismKind::CrossCoreEvent ||
           (mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
            !scarcityMembers.test(mechanism.id)));
      const bool assignmentMatches =
          mechanism.eventId.has_value() == mustHaveEvent &&
          mechanism.releaseEventId.has_value() ==
              (selected &&
               mechanism.kind == CanonicalMechanismKind::RecurringEvent &&
               !scarcityMembers.test(mechanism.id));
      if (!assignmentMatches) {
        invalidEventAssignment = true;
        break;
      }
    }
    if (invalidGreedyCandidate || invalidMechanism || invalidOrder ||
        missingBaseline || invalidDeletion || incompleteCoverage ||
        setCoverSolution->weight != expectedWeight || invalidScarcityGroup ||
        invalidEventAssignment || !setCoverSolution->coverageVerified) {
      return fail("set-cover solution references an invalid ID or lacks a "
                  "checked coverage proof or event allocation");
    }
  }
  const bool wrongDirectCount = directMechanisms.size() != demands.size();
  if (wrongDirectCount ||
      llvm::any_of(directMechanisms, [this](CanonicalMechanismId id) {
        return id >= mechanisms.size();
      })) {
    return fail("every demand must name a valid direct mechanism");
  }
  frozen = true;
  return success();
}

const CanonicalRegion &
CanonicalSyncProgram::getRegion(CanonicalRegionId id) const {
  return regions[id];
}

const CanonicalPhase &
CanonicalSyncProgram::getPhase(CanonicalPhaseId id) const {
  return phases[id];
}

const CanonicalAccess &
CanonicalSyncProgram::getAccess(CanonicalAccessId id) const {
  return accesses[id];
}

const CanonicalFenceEffect &
CanonicalSyncProgram::getFenceEffect(CanonicalFenceEffectId id) const {
  return fenceEffects[id];
}

const CanonicalDemand &
CanonicalSyncProgram::getDemand(CanonicalDemandId id) const {
  return demands[id];
}

const CanonicalMechanism &
CanonicalSyncProgram::getMechanism(CanonicalMechanismId id) const {
  return mechanisms[id];
}

StringRef mlir::pto::stringifyCanonicalCore(CanonicalCore core) {
  return core == CanonicalCore::AIC ? "AIC" : "AIV";
}

StringRef mlir::pto::stringifyCanonicalRegionKind(CanonicalRegionKind kind) {
  switch (kind) {
  case CanonicalRegionKind::Function:
    return "function";
  case CanonicalRegionKind::Sequence:
    return "sequence";
  case CanonicalRegionKind::Choice:
    return "choice";
  case CanonicalRegionKind::Loop:
    return "loop";
  case CanonicalRegionKind::Transparent:
    return "transparent";
  }
  llvm_unreachable("unknown canonical region kind");
}

StringRef mlir::pto::stringifyCanonicalAccessMode(CanonicalAccessMode mode) {
  switch (mode) {
  case CanonicalAccessMode::Read:
    return "R";
  case CanonicalAccessMode::Write:
    return "W";
  case CanonicalAccessMode::ReadWrite:
    return "RW";
  }
  llvm_unreachable("unknown canonical access mode");
}

StringRef mlir::pto::stringifyCanonicalDemandKind(CanonicalDemandKind kind) {
  switch (kind) {
  case CanonicalDemandKind::Raw:
    return "RAW";
  case CanonicalDemandKind::War:
    return "WAR";
  case CanonicalDemandKind::Waw:
    return "WAW";
  case CanonicalDemandKind::OrderedMemory:
    return "ORDERED";
  case CanonicalDemandKind::HardwareAccReadConflict:
    return "ACC_RAR";
  case CanonicalDemandKind::SsaCompletion:
    return "SSA";
  case CanonicalDemandKind::ExitCompletion:
    return "EXIT";
  case CanonicalDemandKind::Visibility:
    return "VISIBILITY";
  }
  llvm_unreachable("unknown canonical demand kind");
}

StringRef mlir::pto::stringifyCanonicalVisibilityDirection(
    CanonicalVisibilityDirection direction) {
  switch (direction) {
  case CanonicalVisibilityDirection::ScalarToNonScalar:
    return "scalar-to-nonscalar";
  case CanonicalVisibilityDirection::NonScalarToScalar:
    return "nonscalar-to-scalar";
  case CanonicalVisibilityDirection::Mte3ToMte2Gm:
    return "mte3-to-mte2-gm";
  }
  llvm_unreachable("unknown canonical visibility direction");
}

StringRef
mlir::pto::stringifyCanonicalGmAliasPolicy(CanonicalGmAliasPolicy policy) {
  switch (policy) {
  case CanonicalGmAliasPolicy::Conservative:
    return "conservative";
  case CanonicalGmAliasPolicy::DistinctRootsUnsafe:
    return "distinct-roots-unsafe";
  }
  llvm_unreachable("unknown canonical GM alias policy");
}

StringRef mlir::pto::stringifyCanonicalCacheMaintenance(
    CanonicalCacheMaintenance maintenance) {
  switch (maintenance) {
  case CanonicalCacheMaintenance::None:
    return "none";
  case CanonicalCacheMaintenance::CleanSource:
    return "clean-source";
  case CanonicalCacheMaintenance::InvalidateTarget:
    return "invalidate-target";
  }
  llvm_unreachable("unknown canonical cache maintenance requirement");
}

StringRef
mlir::pto::stringifyCanonicalMechanismKind(CanonicalMechanismKind kind) {
  switch (kind) {
  case CanonicalMechanismKind::IntrinsicOrder:
    return "intrinsic";
  case CanonicalMechanismKind::PipeBarrier:
    return "barrier";
  case CanonicalMechanismKind::Event:
    return "event";
  case CanonicalMechanismKind::CrossCoreEvent:
    return "cross-core-event";
  case CanonicalMechanismKind::RecurringEvent:
    return "recurring-event";
  case CanonicalMechanismKind::VisibilityFence:
    return "visibility-fence";
  case CanonicalMechanismKind::FixedFence:
    return "fixed-fence";
  case CanonicalMechanismKind::TailBarrier:
    return "tail";
  }
  llvm_unreachable("unknown canonical mechanism kind");
}
