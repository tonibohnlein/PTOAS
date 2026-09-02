// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

// Structural analysis proposes a bounded number of sets of existing direct
// mechanisms. It never claims synchronization coverage. Coverage.cpp grounds
// every proposal with the summary, flat, and bounded-unrolled oracles before a
// useful proposal may enter the set-cover instance.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr std::size_t kMaximumStructuralProposals = 256;
constexpr std::size_t kMaximumMechanismsPerProposal = 64;
constexpr std::size_t kMaximumOmissionsPerFamily = 16;
constexpr std::size_t kMaximumTransitiveEdgeInspections = 16384;

template <typename T> void canonicalize(SmallVectorImpl<T> &items) {
  llvm::sort(items);
  items.erase(std::unique(items.begin(), items.end()), items.end());
}

bool isSameIteration(const CanonicalDemand &demand) {
  return llvm::all_of(
      demand.iterationDistance, [](const CanonicalLoopDistance &distance) {
        return distance.relation == CanonicalIterationRelation::Same;
      });
}

bool isLevelDemand(const CanonicalSyncProgram &program,
                   const CanonicalDemand &demand) {
  return demand.requirement == CanonicalRequirement::Completion &&
         demand.target < program.getPhases().size() &&
         demand.source != demand.target && isSameIteration(demand);
}

std::string resourceKey(CanonicalPhysicalResource resource) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << stringifyCanonicalCore(resource.core) << ':'
     << stringifyPIPE(resource.pipe);
  return os.str();
}

std::string stageKey(const CanonicalSyncProgram &program,
                     const CanonicalDemand &demand) {
  const CanonicalPhase &source = program.getPhase(demand.source);
  const CanonicalPhase &target = program.getPhase(demand.target);
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "stage=" << resourceKey(source.resource) << '/'
     << source.operation->getName() << "->" << resourceKey(target.resource)
     << '/' << target.operation->getName()
     << ";kind=" << stringifyCanonicalDemandKind(demand.kind);
  if (!demand.causes.empty() &&
      demand.causes.front().sourceAccess < program.getAccesses().size()) {
    const CanonicalAccess &access =
        program.getAccess(demand.causes.front().sourceAccess);
    os << ";space="
       << (access.unknownSpace ? StringRef("unknown")
                               : stringifyAddressSpace(access.space));
  }
  return os.str();
}

struct ProposalBuilder {
  CanonicalSyncProgram &program;

  void append(CanonicalStructuralProposalKind kind, CanonicalRegionId owner,
              unsigned level, StringRef semanticKey,
              ArrayRef<CanonicalMechanismId> mechanisms,
              ArrayRef<CanonicalDemandId> crossingDemands) {
    SmallVector<CanonicalMechanismId, 8> canonicalMechanisms(mechanisms);
    SmallVector<CanonicalDemandId, 8> canonicalDemands(crossingDemands);
    canonicalize(canonicalMechanisms);
    canonicalize(canonicalDemands);
    if (canonicalMechanisms.size() < 2U ||
        canonicalMechanisms.size() > kMaximumMechanismsPerProposal) {
      return;
    }
    if (program.getStructuralProposals().size() >=
        kMaximumStructuralProposals) {
      return;
    }
    CanonicalStructuralProposal proposal;
    proposal.kind = kind;
    proposal.owner = owner;
    proposal.level = level;
    proposal.semanticKey = semanticKey.str();
    proposal.mechanisms = std::move(canonicalMechanisms);
    proposal.crossingDemands = std::move(canonicalDemands);
    program.appendStructuralProposal(std::move(proposal));
  }

  void appendFamily(CanonicalStructuralProposalKind fullKind,
                    CanonicalStructuralProposalKind omissionKind,
                    CanonicalRegionId owner, unsigned level,
                    StringRef semanticKey,
                    ArrayRef<CanonicalMechanismId> mechanisms,
                    ArrayRef<CanonicalDemandId> demands) {
    append(fullKind, owner, level, semanticKey, mechanisms, demands);
    const std::size_t omissionCount =
        std::min(mechanisms.size(), kMaximumOmissionsPerFamily);
    for (std::size_t omitted = 0; omitted < omissionCount; ++omitted) {
      SmallVector<CanonicalMechanismId, 8> subset;
      for (std::size_t index = 0; index < mechanisms.size(); ++index) {
        if (index != omitted) {
          subset.push_back(mechanisms[index]);
        }
      }
      std::string omissionKey =
          (Twine(semanticKey) + ";omit=m" + Twine(mechanisms[omitted])).str();
      append(omissionKind, owner, level, omissionKey, subset, demands);
    }
  }
};

struct OwnerLevelGraph {
  CanonicalRegionId owner = kInvalidCanonicalSyncId;
  SmallVector<CanonicalDemandId, 16> demands;
  SmallVector<unsigned, 0> levels;
  unsigned maximumLevel = 0;
  bool acyclic = true;
};

OwnerLevelGraph buildOwnerLevels(const CanonicalSyncProgram &program,
                                 CanonicalRegionId owner) {
  OwnerLevelGraph graph;
  graph.owner = owner;
  graph.levels.resize(program.getPhases().size(), 0U);
  SmallVector<CanonicalPhaseId, 16> phases;
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (demand.owner != owner || !isLevelDemand(program, demand)) {
      continue;
    }
    graph.demands.push_back(demand.id);
    phases.push_back(demand.source);
    phases.push_back(demand.target);
  }
  canonicalize(phases);
  llvm::sort(phases, [&](CanonicalPhaseId first, CanonicalPhaseId second) {
    const CanonicalPhase &lhs = program.getPhase(first);
    const CanonicalPhase &rhs = program.getPhase(second);
    return std::tie(lhs.sourceOrder, lhs.id) <
           std::tie(rhs.sourceOrder, rhs.id);
  });
  for (CanonicalPhaseId phase : phases) {
    for (CanonicalDemandId demandId : graph.demands) {
      const CanonicalDemand &demand = program.getDemand(demandId);
      if (demand.source != phase) {
        continue;
      }
      const CanonicalPhase &source = program.getPhase(demand.source);
      const CanonicalPhase &target = program.getPhase(demand.target);
      if (std::tie(target.sourceOrder, target.id) <=
          std::tie(source.sourceOrder, source.id)) {
        graph.acyclic = false;
        return graph;
      }
      graph.levels[target.id] =
          std::max(graph.levels[target.id], graph.levels[source.id] + 1U);
      graph.maximumLevel =
          std::max(graph.maximumLevel, graph.levels[target.id]);
    }
  }
  return graph;
}

SmallVector<CanonicalMechanismId, 8>
directMechanisms(const CanonicalSyncProgram &program,
                 ArrayRef<CanonicalDemandId> demands) {
  SmallVector<CanonicalMechanismId, 8> mechanisms;
  for (CanonicalDemandId demand : demands) {
    const CanonicalMechanismId id = program.getDirectMechanisms()[demand];
    const CanonicalMechanismKind kind = program.getMechanism(id).kind;
    const bool baseline = kind == CanonicalMechanismKind::IntrinsicOrder ||
                          kind == CanonicalMechanismKind::FixedFence ||
                          kind == CanonicalMechanismKind::TailBarrier;
    if (!baseline) {
      mechanisms.push_back(id);
    }
  }
  canonicalize(mechanisms);
  return mechanisms;
}

bool hasAlternatePath(const CanonicalSyncProgram &program,
                      const OwnerLevelGraph &graph, CanonicalDemandId omitted) {
  const CanonicalDemand &edge = program.getDemand(omitted);
  SmallVector<CanonicalPhaseId, 16> worklist{edge.source};
  llvm::BitVector visited(program.getPhases().size());
  visited.set(edge.source);
  for (std::size_t next = 0; next < worklist.size(); ++next) {
    const CanonicalPhaseId phase = worklist[next];
    for (CanonicalDemandId demandId : graph.demands) {
      if (demandId == omitted) {
        continue;
      }
      const CanonicalDemand &candidate = program.getDemand(demandId);
      if (candidate.source != phase || visited.test(candidate.target)) {
        continue;
      }
      if (candidate.target == edge.target) {
        return true;
      }
      visited.set(candidate.target);
      worklist.push_back(candidate.target);
    }
  }
  return false;
}

void proposeTransitiveBasis(const CanonicalSyncProgram &program,
                            const OwnerLevelGraph &graph,
                            ProposalBuilder &builder) {
  const std::size_t work = graph.demands.size() * graph.demands.size();
  if (work > kMaximumTransitiveEdgeInspections) {
    return;
  }
  SmallVector<CanonicalDemandId, 16> basis;
  for (CanonicalDemandId demand : graph.demands) {
    if (!hasAlternatePath(program, graph, demand)) {
      basis.push_back(demand);
    }
  }
  const SmallVector<CanonicalMechanismId, 8> members =
      directMechanisms(program, basis);
  if (members.size() < directMechanisms(program, graph.demands).size()) {
    builder.append(CanonicalStructuralProposalKind::RegionTransitiveBasis,
                   graph.owner, 0U, "graph=region-transitive-basis", members,
                   graph.demands);
  }
}

void proposeLevelBoundaries(const CanonicalSyncProgram &program,
                            const OwnerLevelGraph &graph,
                            ProposalBuilder &builder) {
  for (unsigned level = 0; level < graph.maximumLevel; ++level) {
    SmallVector<CanonicalDemandId, 16> crossing;
    for (CanonicalDemandId demandId : graph.demands) {
      const CanonicalDemand &demand = program.getDemand(demandId);
      if (graph.levels[demand.source] <= level &&
          graph.levels[demand.target] > level) {
        crossing.push_back(demandId);
      }
    }
    const SmallVector<CanonicalMechanismId, 8> members =
        directMechanisms(program, crossing);
    const std::string key =
        (Twine("graph=level-boundary;level=") + Twine(level)).str();
    builder.appendFamily(CanonicalStructuralProposalKind::LevelBoundary,
                         CanonicalStructuralProposalKind::LevelBoundaryMinusOne,
                         graph.owner, level, key, members, crossing);

    for (CanonicalCore core : {CanonicalCore::AIC, CanonicalCore::AIV}) {
      SmallVector<CanonicalDemandId, 16> coreDemands;
      for (CanonicalDemandId demandId : crossing) {
        const CanonicalDemand &demand = program.getDemand(demandId);
        const CanonicalPhase &source = program.getPhase(demand.source);
        const CanonicalPhase &target = program.getPhase(demand.target);
        if (source.resource.core == core || target.resource.core == core) {
          coreDemands.push_back(demandId);
        }
      }
      const SmallVector<CanonicalMechanismId, 8> coreMembers =
          directMechanisms(program, coreDemands);
      const std::string coreKey =
          (Twine("semantic=hardware-unit;") + stringifyCanonicalCore(core) +
           ";level=" + Twine(level))
              .str();
      builder.append(CanonicalStructuralProposalKind::SemanticLevelBoundary,
                     graph.owner, level, coreKey, coreMembers, coreDemands);
    }

    for (CanonicalDemandId seedId : crossing) {
      const std::string seedKey = stageKey(program, program.getDemand(seedId));
      SmallVector<CanonicalDemandId, 8> stageDemands;
      for (CanonicalDemandId demandId : crossing) {
        if (stageKey(program, program.getDemand(demandId)) == seedKey) {
          stageDemands.push_back(demandId);
        }
      }
      builder.append(CanonicalStructuralProposalKind::SemanticLevelBoundary,
                     graph.owner, level, seedKey,
                     directMechanisms(program, stageDemands), stageDemands);
    }
  }
}

struct StorageFamily {
  CanonicalRegionId owner = kInvalidCanonicalSyncId;
  Value root;
  AddressSpace space = AddressSpace::Zero;
  bool unknownSpace = false;
  CanonicalAccessId rootOrdinal = kInvalidCanonicalSyncId;
  SmallVector<CanonicalDemandId, 16> demands;
};

void proposeStorageLifecycles(const CanonicalSyncProgram &program,
                              ProposalBuilder &builder) {
  SmallVector<StorageFamily, 8> families;
  for (const CanonicalDemand &demand : program.getDemands()) {
    for (const CanonicalDemandCause &cause : demand.causes) {
      if (cause.sourceAccess >= program.getAccesses().size()) {
        continue;
      }
      const CanonicalAccess &access = program.getAccess(cause.sourceAccess);
      if (!access.aliasRoot) {
        continue;
      }
      auto family = llvm::find_if(families, [&](const StorageFamily &item) {
        return item.owner == demand.owner && item.root == access.aliasRoot &&
               item.space == access.space &&
               item.unknownSpace == access.unknownSpace;
      });
      if (family == families.end()) {
        StorageFamily item;
        item.owner = demand.owner;
        item.root = access.aliasRoot;
        item.space = access.space;
        item.unknownSpace = access.unknownSpace;
        item.rootOrdinal = access.id;
        families.push_back(std::move(item));
        family = std::prev(families.end());
      }
      if (!llvm::is_contained(family->demands, demand.id)) {
        family->demands.push_back(demand.id);
      }
    }
  }
  for (const StorageFamily &family : families) {
    const SmallVector<CanonicalMechanismId, 8> members =
        directMechanisms(program, family.demands);
    std::string storage;
    llvm::raw_string_ostream os(storage);
    os << "semantic=storage-lifecycle;root=a" << family.rootOrdinal << ";space="
       << (family.unknownSpace ? StringRef("unknown")
                               : stringifyAddressSpace(family.space));
    builder.appendFamily(
        CanonicalStructuralProposalKind::StorageLifecycle,
        CanonicalStructuralProposalKind::StorageLifecycleMinusOne, family.owner,
        0U, os.str(), members, family.demands);
  }
}

} // namespace

LogicalResult
mlir::pto::proposeCanonicalSyncStructuralGroups(CanonicalSyncProgram &program,
                                                bool enabled) {
  const bool invalidState = !program.isGraphFrozen() || program.isFrozen() ||
                            !program.mechanismCatalogComplete ||
                            program.coverageCatalogComplete ||
                            program.structuralProposalCatalogComplete;
  if (invalidState) {
    return program.getFunction().emitError(
        "canonical sync structural proposals require a complete direct "
        "mechanism catalog");
  }
  if (enabled) {
    ProposalBuilder builder{program};
    for (const CanonicalRegion &region : program.getRegions()) {
      OwnerLevelGraph graph = buildOwnerLevels(program, region.id);
      if (!graph.acyclic || graph.demands.size() < 2U) {
        continue;
      }
      proposeLevelBoundaries(program, graph, builder);
      proposeTransitiveBasis(program, graph, builder);
    }
    proposeStorageLifecycles(program, builder);
  }
  program.structuralProposalCatalogComplete = true;
  return success();
}
