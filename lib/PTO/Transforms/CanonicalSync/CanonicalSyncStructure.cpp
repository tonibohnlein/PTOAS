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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::canonical_sync_detail;

namespace {

constexpr std::size_t kMaximumStructuralProposals = 256;
constexpr std::size_t kMaximumMechanismsPerProposal = 64;
constexpr std::size_t kMaximumOmissionsPerFamily = 16;
constexpr std::size_t kMaximumTransitiveClosureWords = 1U << 20;
constexpr std::size_t kMaximumLifecyclePhases = 128;
constexpr std::size_t kMaximumLifecycleDemands = 512;

enum class ProposalFamily : std::uint8_t {
  Level,
  Transitive,
  Connector,
  Semantic,
  Storage,
  Count,
};

constexpr std::array<std::size_t,
                     static_cast<std::size_t>(ProposalFamily::Count)>
    kMaximumProposalsByFamily = {64U, 32U, 64U, 64U, 32U};

ProposalFamily proposalFamily(CanonicalStructuralProposalKind kind) {
  switch (kind) {
  case CanonicalStructuralProposalKind::LevelBoundary:
  case CanonicalStructuralProposalKind::LevelBoundaryMinusOne:
    return ProposalFamily::Level;
  case CanonicalStructuralProposalKind::RegionTransitiveBasis:
    return ProposalFamily::Transitive;
  case CanonicalStructuralProposalKind::ConnectorNeighborhood:
    return ProposalFamily::Connector;
  case CanonicalStructuralProposalKind::SemanticLevelBoundary:
    return ProposalFamily::Semantic;
  case CanonicalStructuralProposalKind::StorageLifecycle:
  case CanonicalStructuralProposalKind::StorageLifecycleMinusOne:
  case CanonicalStructuralProposalKind::StorageOwnershipProtocol:
    return ProposalFamily::Storage;
  }
  llvm_unreachable("unknown canonical structural proposal kind");
}

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
  const StringRef sourceName =
      source.operation ? source.operation->getName().getStringRef()
                       : StringRef("unknown");
  const StringRef targetName =
      target.operation ? target.operation->getName().getStringRef()
                       : StringRef("unknown");
  os << "stage=" << resourceKey(source.resource) << '/' << sourceName << "->"
     << resourceKey(target.resource) << '/' << targetName
     << ";kind=" << stringifyCanonicalDemandKind(demand.kind);
  const bool hasKnownSourceAccess =
      !demand.causes.empty() &&
      demand.causes.front().sourceAccess < program.getAccesses().size();
  if (hasKnownSourceAccess) {
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
  std::array<std::size_t, static_cast<std::size_t>(ProposalFamily::Count)>
      familyCounts{};

  void append(CanonicalStructuralProposalKind kind, CanonicalRegionId owner,
              unsigned level, StringRef semanticKey,
              ArrayRef<CanonicalMechanismId> mechanisms,
              ArrayRef<CanonicalDemandId> crossingDemands) {
    SmallVector<CanonicalMechanismId, 8> canonicalMechanisms(mechanisms);
    SmallVector<CanonicalDemandId, 8> canonicalDemands(crossingDemands);
    canonicalize(canonicalMechanisms);
    canonicalize(canonicalDemands);
    const bool invalidMemberCount =
        canonicalMechanisms.size() < 2U ||
        canonicalMechanisms.size() > kMaximumMechanismsPerProposal;
    if (invalidMemberCount) {
      return;
    }
    const bool proposalBudgetExhausted =
        program.getStructuralProposals().size() >=
        kMaximumStructuralProposals;
    const ProposalFamily family = proposalFamily(kind);
    const std::size_t familyIndex = static_cast<std::size_t>(family);
    const bool familyBudgetExhausted =
        familyCounts[familyIndex] >= kMaximumProposalsByFamily[familyIndex];
    if (proposalBudgetExhausted || familyBudgetExhausted) {
      return;
    }
    CanonicalStructuralProposal proposal;
    proposal.kind = kind;
    proposal.owner = owner;
    proposal.level = level;
    proposal.semanticKey = semanticKey.str();
    proposal.mechanisms = std::move(canonicalMechanisms);
    proposal.crossingDemands = std::move(canonicalDemands);
    const std::size_t oldProposalCount =
        program.getStructuralProposals().size();
    program.appendStructuralProposal(std::move(proposal));
    const bool proposalAdded =
        program.getStructuralProposals().size() != oldProposalCount;
    if (proposalAdded) {
      ++familyCounts[familyIndex];
    }
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
      const bool notForwardInIssueOrder =
          std::tie(target.sourceOrder, target.id) <=
          std::tie(source.sourceOrder, source.id);
      if (notForwardInIssueOrder) {
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

void proposeTransitiveBasis(const CanonicalSyncProgram &program,
                            const OwnerLevelGraph &graph,
                            ProposalBuilder &builder) {
  struct Edge {
    CanonicalPhaseId source = kInvalidCanonicalSyncId;
    CanonicalPhaseId target = kInvalidCanonicalSyncId;
    SmallVector<CanonicalDemandId, 2> demands;
  };
  SmallVector<CanonicalPhaseId, 16> phases;
  SmallVector<Edge, 16> edges;
  for (CanonicalDemandId demandId : graph.demands) {
    const CanonicalDemand &demand = program.getDemand(demandId);
    phases.push_back(demand.source);
    phases.push_back(demand.target);
    auto edge = llvm::find_if(edges, [&](const Edge &candidate) {
      return candidate.source == demand.source &&
             candidate.target == demand.target;
    });
    if (edge == edges.end()) {
      edges.push_back({demand.source, demand.target, {demandId}});
      continue;
    }
    edge->demands.push_back(demandId);
  }
  canonicalize(phases);
  llvm::sort(phases, [&](CanonicalPhaseId first, CanonicalPhaseId second) {
    const CanonicalPhase &lhs = program.getPhase(first);
    const CanonicalPhase &rhs = program.getPhase(second);
    return std::tie(lhs.sourceOrder, lhs.id) <
           std::tie(rhs.sourceOrder, rhs.id);
  });
  const std::size_t wordsPerRow = (phases.size() + 63U) / 64U;
  const bool closureBudgetExceeded =
      phases.empty() ||
      phases.size() * wordsPerRow > kMaximumTransitiveClosureWords;
  if (closureBudgetExceeded) {
    return;
  }
  SmallVector<unsigned, 16> localPhase(program.getPhases().size(),
                                        kInvalidCanonicalSyncId);
  for (unsigned index = 0; index < phases.size(); ++index) {
    localPhase[phases[index]] = index;
  }
  SmallVector<SmallVector<unsigned, 4>, 0> outgoing(phases.size());
  for (unsigned edgeId = 0; edgeId < edges.size(); ++edgeId) {
    outgoing[localPhase[edges[edgeId].source]].push_back(edgeId);
  }
  SmallVector<BitVector, 0> reachable;
  reachable.reserve(phases.size());
  for (std::size_t index = 0; index < phases.size(); ++index) {
    reachable.emplace_back(phases.size());
  }
  for (unsigned source : llvm::reverse(llvm::seq<unsigned>(0, phases.size()))) {
    for (unsigned edgeId : outgoing[source]) {
      const unsigned target = localPhase[edges[edgeId].target];
      reachable[source].set(target);
      reachable[source] |= reachable[target];
    }
  }
  SmallVector<CanonicalDemandId, 16> basis;
  for (const Edge &edge : edges) {
    const unsigned source = localPhase[edge.source];
    const unsigned target = localPhase[edge.target];
    const bool alternate = llvm::any_of(outgoing[source], [&](unsigned edgeId) {
      const unsigned successor = localPhase[edges[edgeId].target];
      return successor != target && reachable[successor].test(target);
    });
    if (!alternate) {
      llvm::append_range(basis, edge.demands);
    }
  }
  const SmallVector<CanonicalMechanismId, 8> members =
      directMechanisms(program, basis);
  const bool reducesDirectMechanisms =
      members.size() < directMechanisms(program, graph.demands).size();
  if (reducesDirectMechanisms) {
    builder.append(CanonicalStructuralProposalKind::RegionTransitiveBasis,
                   graph.owner, 0U, "graph=region-transitive-basis", members,
                   graph.demands);
  }
}

void proposeLevelBoundaries(const CanonicalSyncProgram &program,
                            const OwnerLevelGraph &graph,
                            ProposalBuilder &builder, bool includeLevel,
                            bool includeSemantic) {
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
    if (includeLevel) {
      builder.appendFamily(
          CanonicalStructuralProposalKind::LevelBoundary,
          CanonicalStructuralProposalKind::LevelBoundaryMinusOne, graph.owner,
          level, key, members, crossing);
    }

    if (!includeSemantic) {
      continue;
    }

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
        const bool sameStage =
            stageKey(program, program.getDemand(demandId)) == seedKey;
        if (sameStage) {
          stageDemands.push_back(demandId);
        }
      }
      builder.append(CanonicalStructuralProposalKind::SemanticLevelBoundary,
                     graph.owner, level, seedKey,
                     directMechanisms(program, stageDemands), stageDemands);
    }
  }
}

void proposeConnectorNeighborhoods(const CanonicalSyncProgram &program,
                                   const OwnerLevelGraph &graph,
                                   ProposalBuilder &builder) {
  SmallVector<CanonicalPhaseId, 16> phases;
  for (CanonicalDemandId demandId : graph.demands) {
    const CanonicalDemand &demand = program.getDemand(demandId);
    phases.push_back(demand.source);
    phases.push_back(demand.target);
  }
  canonicalize(phases);
  for (CanonicalPhaseId connector : phases) {
    SmallVector<CanonicalDemandId, 8> incoming;
    SmallVector<CanonicalDemandId, 8> outgoing;
    for (CanonicalDemandId demandId : graph.demands) {
      const CanonicalDemand &demand = program.getDemand(demandId);
      if (demand.target == connector) {
        incoming.push_back(demandId);
      }
      if (demand.source == connector) {
        outgoing.push_back(demandId);
      }
    }
    const bool isNotConnector = incoming.empty() || outgoing.empty();
    if (isNotConnector) {
      continue;
    }
    SmallVector<CanonicalDemandId, 16> neighborhood(incoming.begin(),
                                                    incoming.end());
    llvm::append_range(neighborhood, outgoing);
    canonicalize(neighborhood);
    const CanonicalPhase &phase = program.getPhase(connector);
    const StringRef operationName =
        phase.operation ? phase.operation->getName().getStringRef()
                        : StringRef("unknown");
    const std::string key =
        (Twine("graph=connector;phase=p") + Twine(connector) + ";resource=" +
         resourceKey(phase.resource) + ";operation=" + operationName)
            .str();
    builder.append(CanonicalStructuralProposalKind::ConnectorNeighborhood,
                   graph.owner, graph.levels[connector], key,
                   directMechanisms(program, neighborhood), neighborhood);
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

bool sameInterval(const CanonicalByteInterval &first,
                  const CanonicalByteInterval &second) {
  return first.begin == second.begin && first.size == second.size;
}

bool sameGuard(ArrayRef<CanonicalControlAtom> first,
               ArrayRef<CanonicalControlAtom> second) {
  return first == second;
}

bool accessHasExactSlot(const CanonicalAccess &access) {
  return access.physical && !access.unknownRange &&
         access.intervals.size() == 1U && access.intervals.front().size != 0U;
}

struct StorageRootUsage {
  Value root;
  AddressSpace space = AddressSpace::Zero;
  CanonicalAccessId rootOrdinal = kInvalidCanonicalSyncId;
  CanonicalPhysicalResource producer;
  CanonicalPhysicalResource consumer;
  Type type;
  SmallVector<CanonicalAccessId, 16> accesses;
};

struct StorageLoopUsage {
  const StorageRootUsage *root = nullptr;
  CanonicalRegionId loop = kInvalidCanonicalSyncId;
  SmallVector<CanonicalAccessId, 16> accesses;
};

struct StorageProtocolFamily {
  CanonicalRegionId loop = kInvalidCanonicalSyncId;
  std::string key;
  SmallVector<const StorageLoopUsage *, 4> lanes;
};

std::string normalizedStorageFamilyKey(const StorageRootUsage &usage) {
  std::string result;
  llvm::raw_string_ostream os(result);
  os << "space=" << stringifyAddressSpace(usage.space) << ";type="
     << usage.type << ";stage=" << resourceKey(usage.producer) << "->"
     << resourceKey(usage.consumer);
  return os.str();
}

SmallVector<StorageRootUsage, 8>
buildNormalizedStorageRoots(const CanonicalSyncProgram &program) {
  SmallVector<StorageRootUsage, 8> roots;
  for (const CanonicalAccess &access : program.getAccesses()) {
    if (!access.aliasRoot || !accessHasExactSlot(access)) {
      continue;
    }
    auto existing = llvm::find_if(roots, [&](const StorageRootUsage &usage) {
      return usage.root == access.aliasRoot;
    });
    if (existing == roots.end()) {
      StorageRootUsage usage;
      usage.root = access.aliasRoot;
      usage.space = access.space;
      usage.rootOrdinal = access.id;
      usage.type = access.aliasRoot.getType();
      roots.push_back(std::move(usage));
      existing = std::prev(roots.end());
    }
    existing->accesses.push_back(access.id);
  }

  llvm::erase_if(roots, [&](StorageRootUsage &usage) {
    SmallVector<CanonicalPhysicalResource, 2> writers;
    for (CanonicalAccessId accessId : usage.accesses) {
      const CanonicalAccess &access = program.getAccess(accessId);
      const CanonicalPhysicalResource resource =
          program.getPhase(access.phase).resource;
      if (accessWrites(access.mode) && !llvm::is_contained(writers, resource)) {
        writers.push_back(resource);
      }
    }
    if (writers.size() != 1U) {
      return true;
    }
    SmallVector<CanonicalPhysicalResource, 2> externalReaders;
    for (CanonicalAccessId accessId : usage.accesses) {
      const CanonicalAccess &access = program.getAccess(accessId);
      const CanonicalPhysicalResource resource =
          program.getPhase(access.phase).resource;
      if (accessReads(access.mode) && resource != writers.front() &&
          !llvm::is_contained(externalReaders, resource)) {
        externalReaders.push_back(resource);
      }
    }
    // Read-modify-write phases keep ownership on the producing pipeline.  A
    // stationary accumulator is consequently one writer plus one *external*
    // reader, even though the writer also reads the same physical interval.
    const bool unsupported = externalReaders.size() != 1U;
    if (!unsupported) {
      usage.producer = writers.front();
      usage.consumer = externalReaders.front();
    }
    return unsupported;
  });
  return roots;
}

SmallVector<StorageLoopUsage, 8>
buildStorageLoopUsages(const CanonicalSyncProgram &program,
                       ArrayRef<StorageRootUsage> roots) {
  SmallVector<StorageLoopUsage, 8> usages;
  for (const StorageRootUsage &root : roots) {
    for (CanonicalAccessId accessId : root.accesses) {
      const CanonicalAccess &access = program.getAccess(accessId);
      const CanonicalPhase &phase = program.getPhase(access.phase);
      if (phase.loopPath.empty()) {
        continue;
      }
      // A storage lifetime can be owned by an enclosing loop even when its
      // producing stage is a nested reduction.  Record the access at every
      // enclosing loop and let the exact cycle/witness checks below reject
      // levels that do not form a complete ready/release lifecycle.
      for (CanonicalRegionId loop : phase.loopPath) {
        auto usage = llvm::find_if(usages, [&](const StorageLoopUsage &item) {
          return item.root == &root && item.loop == loop;
        });
        if (usage == usages.end()) {
          usages.push_back({&root, loop, {}});
          usage = std::prev(usages.end());
        }
        usage->accesses.push_back(accessId);
      }
    }
  }
  return usages;
}

SmallVector<StorageProtocolFamily, 8> buildStorageProtocolFamilies(
    ArrayRef<StorageLoopUsage> usages) {
  SmallVector<StorageProtocolFamily, 8> families;
  for (const StorageLoopUsage &usage : usages) {
    const std::string key = normalizedStorageFamilyKey(*usage.root);
    auto family = llvm::find_if(
        families, [&](const StorageProtocolFamily &candidate) {
          return candidate.loop == usage.loop && candidate.key == key;
        });
    if (family == families.end()) {
      families.push_back({usage.loop, key, {}});
      family = std::prev(families.end());
    }
    family->lanes.push_back(&usage);
  }
  const std::size_t groupedFamilyCount = families.size();
  for (std::size_t index = 0; index < groupedFamilyCount; ++index) {
    const CanonicalRegionId loop = families[index].loop;
    const std::string key = families[index].key;
    const SmallVector<const StorageLoopUsage *, 4> lanes =
        families[index].lanes;
    if (lanes.size() < 2U) {
      continue;
    }
    for (const StorageLoopUsage *lane : lanes) {
      families.push_back({loop, key, {lane}});
    }
  }
  return families;
}

struct GuardedLaneAccesses {
  unsigned lane = 0;
  SmallVector<CanonicalControlAtom, 2> guard;
  SmallVector<CanonicalAccessId, 16> accesses;
};

std::optional<unsigned> unsignedConstant(Value value) {
  APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)) || constant.isNegative() ||
      constant.getActiveBits() > std::numeric_limits<unsigned>::digits) {
    return std::nullopt;
  }
  return static_cast<unsigned>(constant.getZExtValue());
}

bool loopHasUnitStep(const CanonicalSyncProgram &program,
                     CanonicalRegionId loopId) {
  if (loopId >= program.getRegions().size()) {
    return false;
  }
  auto loop = dyn_cast_or_null<scf::ForOp>(
      program.getRegion(loopId).operation);
  return loop && unsignedConstant(loop.getStep()) == 1U;
}

bool loopHasKnownNonnegativeOrigin(const CanonicalSyncProgram &program,
                                   CanonicalRegionId loopId) {
  if (loopId >= program.getRegions().size()) {
    return false;
  }
  auto loop = dyn_cast_or_null<scf::ForOp>(
      program.getRegion(loopId).operation);
  return loop && unsignedConstant(loop.getLowerBound()).has_value();
}

struct ModuloGuardConstraint {
  unsigned modulus = 0;
  unsigned residue = 0;
  bool equal = false;
};

std::optional<ModuloGuardConstraint> getModuloGuardConstraint(
    const CanonicalSyncProgram &program, CanonicalRegionId loopId,
    const CanonicalControlAtom &atom) {
  if (atom.choice >= program.getRegions().size()) {
    return std::nullopt;
  }
  auto choice = dyn_cast_or_null<scf::IfOp>(
      program.getRegion(atom.choice).operation);
  auto loop = dyn_cast_or_null<scf::ForOp>(
      program.getRegion(loopId).operation);
  auto compare = choice ? choice.getCondition().getDefiningOp<arith::CmpIOp>()
                        : nullptr;
  if (!loop || !compare || compare.getPredicate() != arith::CmpIPredicate::eq) {
    return std::nullopt;
  }
  Value remainderValue = compare.getLhs();
  Value residueValue = compare.getRhs();
  if (!remainderValue.getDefiningOp<arith::RemSIOp>() &&
      !remainderValue.getDefiningOp<arith::RemUIOp>()) {
    std::swap(remainderValue, residueValue);
  }
  Value dividend;
  Value divisor;
  if (auto remainder = remainderValue.getDefiningOp<arith::RemSIOp>()) {
    dividend = remainder.getLhs();
    divisor = remainder.getRhs();
  } else if (auto remainder =
                 remainderValue.getDefiningOp<arith::RemUIOp>()) {
    dividend = remainder.getLhs();
    divisor = remainder.getRhs();
  } else {
    return std::nullopt;
  }
  const std::optional<unsigned> modulus = unsignedConstant(divisor);
  const std::optional<unsigned> residue = unsignedConstant(residueValue);
  if (dividend != loop.getInductionVar() || !modulus || *modulus == 0U ||
      !residue || *residue >= *modulus || atom.arm > 1U) {
    return std::nullopt;
  }
  return ModuloGuardConstraint{*modulus, *residue, atom.arm == 0U};
}

std::optional<unsigned> inferOwnershipPeriod(
    const CanonicalSyncProgram &program, CanonicalRegionId loop,
    ArrayRef<CanonicalOwnershipStage> stages) {
  std::optional<unsigned> period;
  for (const CanonicalOwnershipStage &stage : stages) {
    for (ArrayRef<CanonicalControlAtom> guard :
         {ArrayRef<CanonicalControlAtom>(stage.producerGuard),
          ArrayRef<CanonicalControlAtom>(stage.consumerGuard)}) {
      for (const CanonicalControlAtom &atom : guard) {
        const std::optional<ModuloGuardConstraint> constraint =
            getModuloGuardConstraint(program, loop, atom);
        if (!constraint) {
          continue;
        }
        if (period && *period != constraint->modulus) {
          return std::nullopt;
        }
        period = constraint->modulus;
      }
    }
  }
  return period.value_or(1U);
}

BitVector ownershipStageResidues(const CanonicalSyncProgram &program,
                                 CanonicalRegionId loop, unsigned period,
                                 ArrayRef<CanonicalControlAtom> guard) {
  BitVector possible(period, true);
  for (const CanonicalControlAtom &atom : guard) {
    const std::optional<ModuloGuardConstraint> constraint =
        getModuloGuardConstraint(program, loop, atom);
    if (!constraint || constraint->modulus != period) {
      continue;
    }
    if (constraint->equal) {
      possible.reset();
      possible.set(constraint->residue);
    } else {
      possible.reset(constraint->residue);
    }
  }
  return possible;
}

BitVector ownershipSlotResidues(const CanonicalSyncProgram &program,
                                CanonicalRegionId loop, unsigned period,
                                ArrayRef<CanonicalOwnershipStage> stages) {
  BitVector result(period);
  for (const CanonicalOwnershipStage &stage : stages) {
    if (!stage.initialProducer) {
      result |= ownershipStageResidues(program, loop, period,
                                       stage.producerGuard);
    }
  }
  return result;
}

unsigned maximumCyclicReuseDistance(const BitVector &active) {
  if (active.empty() || active.none()) {
    return 0U;
  }
  const unsigned period = active.size();
  unsigned maximum = 0U;
  for (int current = active.find_first(); current >= 0;
       current = active.find_next(current)) {
    unsigned distance = 1U;
    while (distance <= period &&
           !active.test((static_cast<unsigned>(current) + distance) %
                        period)) {
      ++distance;
    }
    maximum = std::max(maximum, distance);
  }
  return maximum;
}

std::optional<unsigned> uniqueCyclicDistance(const BitVector &source,
                                             const BitVector &target,
                                             bool requirePositive) {
  if (source.count() != 1U || target.count() != 1U ||
      source.size() != target.size() || source.empty()) {
    return std::nullopt;
  }
  const unsigned period = source.size();
  const unsigned sourceResidue = source.find_first();
  const unsigned targetResidue = target.find_first();
  unsigned distance =
      (targetResidue + period - sourceResidue) % period;
  if (requirePositive && distance == 0U) {
    distance = period;
  }
  if (!requirePositive && distance != 0U) {
    return std::nullopt;
  }
  return distance;
}

bool phaseOrderLess(const CanonicalSyncProgram &program,
                    CanonicalAccessId first, CanonicalAccessId second) {
  const CanonicalPhase &left = program.getPhase(program.getAccess(first).phase);
  const CanonicalPhase &right =
      program.getPhase(program.getAccess(second).phase);
  return std::tie(left.sourceOrder, first) <
         std::tie(right.sourceOrder, second);
}

Operation *liftOperationToBlock(Operation *operation, Block *block) {
  while (operation && operation->getBlock() != block) {
    operation = operation->getParentOp();
  }
  return operation;
}

Operation *liftOperationToRegion(Operation *operation, Region *region) {
  while (operation && operation->getParentRegion() != region) {
    operation = operation->getParentOp();
  }
  return operation;
}

bool operationIsInsideRegion(Operation *operation, Region *region) {
  return liftOperationToRegion(operation, region) != nullptr;
}

bool consumerAnchorIsTotal(
    Operation *anchor, ArrayRef<CanonicalPhaseId> consumers,
    const CanonicalSyncProgram &program) {
  auto choice = dyn_cast<scf::IfOp>(anchor);
  if (!choice) {
    return consumers.size() == 1U &&
           program.getPhase(consumers.front()).operation == anchor;
  }
  if (choice.getElseRegion().empty()) {
    return false;
  }
  const auto armHasConsumer = [&](Region &arm) {
    return llvm::any_of(consumers, [&](CanonicalPhaseId phase) {
      return operationIsInsideRegion(program.getPhase(phase).operation, &arm);
    });
  };
  return armHasConsumer(choice.getThenRegion()) &&
         armHasConsumer(choice.getElseRegion());
}

bool appendOwnershipStagesInRegion(const CanonicalSyncProgram &program,
                                   const StorageRootUsage &root,
                                   const StorageLoopUsage &usage,
                                   Region *path,
                                   CanonicalOwnershipProtocol &protocol) {
  struct Item {
    Operation *anchor = nullptr;
    SmallVector<CanonicalPhaseId, 2> producers;
    SmallVector<CanonicalPhaseId, 2> consumers;
    unsigned order = 0;
  };
  SmallVector<Item, 16> items;
  SmallVector<CanonicalControlAtom, 2> pathGuard;
  bool initializedGuard = false;
  for (CanonicalAccessId accessId : usage.accesses) {
    const CanonicalAccess &access = program.getAccess(accessId);
    const CanonicalPhase &phase = program.getPhase(access.phase);
    if (!operationIsInsideRegion(phase.operation, path)) {
      continue;
    }
    const bool producerAccess =
        accessWrites(access.mode) && phase.resource == root.producer;
    const bool consumerAccess =
        accessReads(access.mode) && phase.resource == root.consumer;
    if (!producerAccess && !consumerAccess) {
      continue;
    }
    if (!initializedGuard) {
      pathGuard.assign(phase.controlPath.begin(), phase.controlPath.end());
      initializedGuard = true;
    } else {
      llvm::erase_if(pathGuard, [&](const CanonicalControlAtom &atom) {
        return !llvm::is_contained(phase.controlPath, atom);
      });
    }
    Operation *anchor = liftOperationToRegion(phase.operation, path);
    if (!anchor) {
      return false;
    }
    auto item = llvm::find_if(items, [&](const Item &candidate) {
      return candidate.anchor == anchor;
    });
    if (item == items.end()) {
      items.push_back({anchor, {}, {}, phase.sourceOrder});
      item = std::prev(items.end());
    }
    item->order = std::min(item->order, phase.sourceOrder);
    SmallVectorImpl<CanonicalPhaseId> &phases =
        producerAccess ? item->producers : item->consumers;
    if (!llvm::is_contained(phases, phase.id)) {
      phases.push_back(phase.id);
    }
  }
  if (items.empty()) {
    return true;
  }
  llvm::stable_sort(items,
                    [](const Item &first, const Item &second) {
                      return std::tie(first.order, first.anchor) <
                             std::tie(second.order, second.anchor);
                    });

  CanonicalOwnershipStage stage;
  stage.producerGuard = pathGuard;
  stage.consumerGuard = pathGuard;
  Operation *firstProducer = nullptr;
  Operation *lastProducer = nullptr;
  Operation *firstConsumer = nullptr;
  Operation *lastConsumer = nullptr;
  const auto finish = [&]() -> bool {
    if (stage.producers.empty() || stage.consumers.empty() ||
        !firstProducer || !lastProducer || !firstConsumer || !lastConsumer) {
      return false;
    }
    stage.writeAcquire = {firstProducer,
                          CanonicalProgramPointPosition::Before};
    stage.ready = {lastProducer, CanonicalProgramPointPosition::After};
    stage.readAcquire = {firstConsumer,
                         CanonicalProgramPointPosition::Before};
    stage.release = {lastConsumer, CanonicalProgramPointPosition::After};
    protocol.stages.push_back(stage);
    stage = CanonicalOwnershipStage{};
    stage.producerGuard = pathGuard;
    stage.consumerGuard = pathGuard;
    firstProducer = nullptr;
    lastProducer = nullptr;
    firstConsumer = nullptr;
    lastConsumer = nullptr;
    return true;
  };

  for (const Item &item : items) {
    if (!item.producers.empty()) {
      if (!stage.consumers.empty() && !finish()) {
        return false;
      }
      if (!item.consumers.empty()) {
        return false;
      }
      firstProducer = firstProducer ? firstProducer : item.anchor;
      lastProducer = item.anchor;
      llvm::append_range(stage.producers, item.producers);
      continue;
    }
    if (item.consumers.empty() || stage.producers.empty() ||
        !consumerAnchorIsTotal(item.anchor, item.consumers, program)) {
      return false;
    }
    firstConsumer = firstConsumer ? firstConsumer : item.anchor;
    lastConsumer = item.anchor;
    llvm::append_range(stage.consumers, item.consumers);
  }
  return stage.producers.empty() ? true : finish();
}

bool appendStructuredOwnershipStages(const CanonicalSyncProgram &program,
                                     const StorageRootUsage &root,
                                     const StorageLoopUsage &usage,
                                     CanonicalOwnershipProtocol &protocol) {
  if (usage.loop >= program.getRegions().size()) {
    return false;
  }
  auto loop = dyn_cast_or_null<scf::ForOp>(program.getRegion(usage.loop).operation);
  if (!loop) {
    return false;
  }
  Region *base = &loop.getRegion();
  SmallVector<Operation *, 4> anchors;
  for (CanonicalAccessId accessId : usage.accesses) {
    const CanonicalPhase &phase =
        program.getPhase(program.getAccess(accessId).phase);
    Operation *anchor = liftOperationToRegion(phase.operation, base);
    if (anchor && !llvm::is_contained(anchors, anchor)) {
      anchors.push_back(anchor);
    }
  }
  if (anchors.size() == 1U) {
    if (auto choice = dyn_cast<scf::IfOp>(anchors.front())) {
      if (choice.getElseRegion().empty()) {
        return false;
      }
      const std::size_t beforeThen = protocol.stages.size();
      if (!appendOwnershipStagesInRegion(program, root, usage,
                                         &choice.getThenRegion(), protocol) ||
          protocol.stages.size() == beforeThen) {
        return false;
      }
      const std::size_t beforeElse = protocol.stages.size();
      return appendOwnershipStagesInRegion(program, root, usage,
                                           &choice.getElseRegion(), protocol) &&
             protocol.stages.size() != beforeElse;
    }
  }
  return appendOwnershipStagesInRegion(program, root, usage, base, protocol) &&
         !protocol.stages.empty();
}

bool appendStationaryOwnershipStage(const CanonicalSyncProgram &program,
                                    const StorageRootUsage &root,
                                    const StorageLoopUsage &usage,
                                    CanonicalOwnershipProtocol &protocol) {
  if (usage.loop >= program.getRegions().size()) {
    return false;
  }
  auto ownerLoop = dyn_cast_or_null<scf::ForOp>(
      program.getRegion(usage.loop).operation);
  if (!ownerLoop) {
    return false;
  }

  Operation *producerAnchor = nullptr;
  Operation *firstConsumerAnchor = nullptr;
  Operation *lastConsumerAnchor = nullptr;
  SmallVector<CanonicalPhaseId, 8> producers;
  SmallVector<CanonicalPhaseId, 4> consumers;
  SmallVector<CanonicalControlAtom, 2> commonGuard;
  bool initializedGuard = false;
  for (CanonicalAccessId accessId : usage.accesses) {
    const CanonicalAccess &access = program.getAccess(accessId);
    const CanonicalPhase &phase = program.getPhase(access.phase);
    const bool producerAccess =
        accessWrites(access.mode) && phase.resource == root.producer;
    const bool consumerAccess =
        accessReads(access.mode) && phase.resource == root.consumer;
    if (!producerAccess && !consumerAccess) {
      continue;
    }

    if (!initializedGuard) {
      commonGuard.assign(phase.controlPath.begin(), phase.controlPath.end());
      initializedGuard = true;
    } else {
      llvm::erase_if(commonGuard, [&](const CanonicalControlAtom &atom) {
        return !llvm::is_contained(phase.controlPath, atom);
      });
    }

    Operation *anchor =
        liftOperationToBlock(phase.operation, ownerLoop.getBody());
    if (!anchor) {
      return false;
    }
    if (producerAccess) {
      // A stationary stage summarizes one nested producer region.  Flat
      // producer/consumer chains remain handled by appendOwnershipStages.
      if (anchor == phase.operation || !isa<scf::ForOp>(anchor) ||
          (producerAnchor && producerAnchor != anchor)) {
        return false;
      }
      producerAnchor = anchor;
      if (!llvm::is_contained(producers, phase.id)) {
        producers.push_back(phase.id);
      }
      continue;
    }
    // The consumer frontier may contain several ordinary operations, but it
    // must live directly in the owner loop after the nested producer region.
    if (anchor != phase.operation) {
      return false;
    }
    if (!firstConsumerAnchor ||
        anchor->isBeforeInBlock(firstConsumerAnchor)) {
      firstConsumerAnchor = anchor;
    }
    if (!lastConsumerAnchor || lastConsumerAnchor->isBeforeInBlock(anchor)) {
      lastConsumerAnchor = anchor;
    }
    if (!llvm::is_contained(consumers, phase.id)) {
      consumers.push_back(phase.id);
    }
  }

  const bool ordered = producerAnchor && firstConsumerAnchor &&
                       lastConsumerAnchor && !producers.empty() &&
                       !consumers.empty() &&
                       producerAnchor->isBeforeInBlock(firstConsumerAnchor);
  if (!ordered) {
    return false;
  }
  llvm::sort(producers, [&](CanonicalPhaseId first, CanonicalPhaseId second) {
    return program.getPhase(first).sourceOrder <
           program.getPhase(second).sourceOrder;
  });
  llvm::sort(consumers, [&](CanonicalPhaseId first, CanonicalPhaseId second) {
    return program.getPhase(first).sourceOrder <
           program.getPhase(second).sourceOrder;
  });

  CanonicalOwnershipStage stage;
  stage.writeAcquire = {producerAnchor,
                        CanonicalProgramPointPosition::Before};
  stage.ready = {producerAnchor, CanonicalProgramPointPosition::After};
  stage.readAcquire = {firstConsumerAnchor,
                       CanonicalProgramPointPosition::Before};
  stage.release = {lastConsumerAnchor,
                   CanonicalProgramPointPosition::After};
  stage.producers = std::move(producers);
  stage.consumers = std::move(consumers);
  stage.producerGuard = commonGuard;
  stage.consumerGuard = std::move(commonGuard);
  protocol.stages.push_back(std::move(stage));
  return true;
}

bool appendOwnershipStages(const CanonicalSyncProgram &program,
                           const StorageRootUsage &root,
                           GuardedLaneAccesses &path,
                           CanonicalOwnershipProtocol &protocol) {
  llvm::stable_sort(path.accesses, [&](CanonicalAccessId first,
                                      CanonicalAccessId second) {
    return phaseOrderLess(program, first, second);
  });
  CanonicalOwnershipStage current;
  bool hasProducer = false;
  const auto finish = [&]() -> bool {
    if (!hasProducer || current.consumers.empty()) {
      return false;
    }
    Operation *producer = current.ready.operation;
    Operation *firstConsumer = current.readAcquire.operation;
    Operation *lastConsumer = current.release.operation;
    const bool ordered = producer && firstConsumer && lastConsumer &&
                         producer->getBlock() == firstConsumer->getBlock() &&
                         producer->getBlock() == lastConsumer->getBlock() &&
                         producer->isBeforeInBlock(firstConsumer) &&
                         (firstConsumer == lastConsumer ||
                          firstConsumer->isBeforeInBlock(lastConsumer));
    if (!ordered) {
      return false;
    }
    protocol.stages.push_back(current);
    current = CanonicalOwnershipStage{};
    hasProducer = false;
    return true;
  };

  for (CanonicalAccessId accessId : path.accesses) {
    const CanonicalAccess &access = program.getAccess(accessId);
    const CanonicalPhase &phase = program.getPhase(access.phase);
    const bool producerAccess =
        accessWrites(access.mode) && phase.resource == root.producer;
    const bool consumerAccess =
        accessReads(access.mode) && phase.resource == root.consumer;
    if (producerAccess) {
      if (hasProducer && !finish()) {
        return false;
      }
      current.lane = path.lane;
      current.writeAcquire = {phase.operation,
                              CanonicalProgramPointPosition::Before};
      current.ready = {phase.operation, CanonicalProgramPointPosition::After};
      current.producers.push_back(phase.id);
      current.producerGuard = path.guard;
      current.consumerGuard = path.guard;
      hasProducer = true;
      continue;
    }
    if (!consumerAccess || !hasProducer) {
      continue;
    }
    if (current.consumers.empty()) {
      current.readAcquire = {phase.operation,
                             CanonicalProgramPointPosition::Before};
    }
    if (!llvm::is_contained(current.consumers, phase.id)) {
      current.consumers.push_back(phase.id);
    }
    current.release = {phase.operation, CanonicalProgramPointPosition::After};
  }
  return !hasProducer || finish();
}

bool distanceIsLocalToLoop(const CanonicalDemand &demand,
                           CanonicalRegionId loop,
                           CanonicalIterationRelation &relation) {
  relation = CanonicalIterationRelation::Same;
  for (const CanonicalLoopDistance &distance : demand.iterationDistance) {
    if (distance.loop == loop) {
      if (distance.relation == CanonicalIterationRelation::Any) {
        return false;
      }
      relation = distance.relation;
      continue;
    }
    if (distance.relation != CanonicalIterationRelation::Same) {
      return false;
    }
  }
  return true;
}

bool demandUsesExactSlot(const CanonicalSyncProgram &program,
                         const CanonicalOwnershipSlot &slot,
                         const CanonicalDemand &demand) {
  return llvm::any_of(demand.causes, [&](const CanonicalDemandCause &cause) {
    if (cause.sourceAccess >= program.getAccesses().size() ||
        cause.targetAccess >= program.getAccesses().size()) {
      return false;
    }
    const CanonicalAccess &source = program.getAccess(cause.sourceAccess);
    const CanonicalAccess &target = program.getAccess(cause.targetAccess);
    return accessHasExactSlot(source) && accessHasExactSlot(target) &&
           source.aliasRoot == slot.root && target.aliasRoot == slot.root &&
           sameInterval(source.intervals.front(), slot.interval) &&
           sameInterval(target.intervals.front(), slot.interval);
  });
}

bool appendPhaseShiftedOwnershipStages(
    const CanonicalSyncProgram &program, const StorageRootUsage &root,
    const StorageLoopUsage &usage, const CanonicalOwnershipSlot &slot,
    CanonicalOwnershipProtocol &protocol) {
  if (usage.loop >= program.getRegions().size()) {
    return false;
  }
  auto loop = dyn_cast_or_null<scf::ForOp>(
      program.getRegion(usage.loop).operation);
  if (!loop) {
    return false;
  }

  SmallVector<CanonicalPhaseId, 4> steadyProducers;
  SmallVector<CanonicalPhaseId, 2> initialProducers;
  SmallVector<CanonicalPhaseId, 16> consumers;
  for (CanonicalAccessId accessId : root.accesses) {
    const CanonicalAccess &access = program.getAccess(accessId);
    const CanonicalPhase &phase = program.getPhase(access.phase);
    const bool producer =
        accessWrites(access.mode) && phase.resource == root.producer;
    const bool consumer =
        accessReads(access.mode) && phase.resource == root.consumer;
    if (consumer && llvm::is_contained(usage.accesses, accessId) &&
        !llvm::is_contained(consumers, phase.id)) {
      consumers.push_back(phase.id);
    }
    if (!producer) {
      continue;
    }
    if (llvm::is_contained(usage.accesses, accessId)) {
      if (!llvm::is_contained(steadyProducers, phase.id)) {
        steadyProducers.push_back(phase.id);
      }
      continue;
    }
    Operation *operation = phase.operation;
    const bool immediatePreheader =
        operation->getBlock() == loop->getBlock() &&
        operation->isBeforeInBlock(loop);
    if (immediatePreheader &&
        !llvm::is_contained(initialProducers, phase.id)) {
      initialProducers.push_back(phase.id);
    }
  }
  if (steadyProducers.empty() || consumers.empty() ||
      initialProducers.size() > 1U) {
    return false;
  }

  const auto appendForProducer = [&](CanonicalPhaseId producer,
                                     bool initial) -> bool {
    SmallVector<CanonicalPhaseId, 8> stageConsumers;
    CanonicalIterationRelation readyRelation =
        CanonicalIterationRelation::Same;
    bool foundRelation = false;
    for (const CanonicalDemand &demand : program.getDemands()) {
      CanonicalIterationRelation relation;
      const bool matching =
          demand.requirement == CanonicalRequirement::Completion &&
          demand.source == producer &&
          llvm::is_contained(consumers, demand.target) &&
          demandUsesExactSlot(program, slot, demand) &&
          distanceIsLocalToLoop(demand, usage.loop, relation) &&
          (initial ? relation == CanonicalIterationRelation::Same
                   : relation == CanonicalIterationRelation::AnyPositive);
      if (!matching) {
        continue;
      }
      if (foundRelation && readyRelation != relation) {
        return false;
      }
      foundRelation = true;
      readyRelation = relation;
      if (!llvm::is_contained(stageConsumers, demand.target)) {
        stageConsumers.push_back(demand.target);
      }
    }
    if (!foundRelation || stageConsumers.empty()) {
      return false;
    }
    llvm::sort(stageConsumers,
               [&](CanonicalPhaseId first, CanonicalPhaseId second) {
                 return program.getPhase(first).sourceOrder <
                        program.getPhase(second).sourceOrder;
               });
    const CanonicalPhase &firstConsumer =
        program.getPhase(stageConsumers.front());
    const CanonicalPhase &lastConsumer =
        program.getPhase(stageConsumers.back());
    if (firstConsumer.operation->getBlock() !=
            lastConsumer.operation->getBlock() ||
        firstConsumer.controlPath != lastConsumer.controlPath) {
      return false;
    }
    if (!initial) {
      const bool hasRelease = llvm::any_of(
          program.getDemands(), [&](const CanonicalDemand &demand) {
            CanonicalIterationRelation relation;
            return demand.requirement == CanonicalRequirement::Completion &&
                   demand.source == lastConsumer.id &&
                   demand.target == producer &&
                   demandUsesExactSlot(program, slot, demand) &&
                   distanceIsLocalToLoop(demand, usage.loop, relation) &&
                   relation == CanonicalIterationRelation::AnyPositive;
          });
      if (!hasRelease) {
        return false;
      }
    }

    const CanonicalPhase &producerPhase = program.getPhase(producer);
    CanonicalOwnershipStage stage;
    stage.initialProducer = initial;
    stage.readyRelation = readyRelation;
    stage.writeAcquire = {producerPhase.operation,
                          CanonicalProgramPointPosition::Before};
    stage.ready = {producerPhase.operation,
                   CanonicalProgramPointPosition::After};
    stage.readAcquire = {firstConsumer.operation,
                         CanonicalProgramPointPosition::Before};
    stage.release = {lastConsumer.operation,
                     CanonicalProgramPointPosition::After};
    stage.producers.push_back(producer);
    stage.consumers = std::move(stageConsumers);
    stage.producerGuard = producerPhase.controlPath;
    stage.consumerGuard = firstConsumer.controlPath;
    protocol.stages.push_back(std::move(stage));
    return true;
  };

  for (CanonicalPhaseId producer : initialProducers) {
    if (!appendForProducer(producer, true)) {
      return false;
    }
  }
  for (CanonicalPhaseId producer : steadyProducers) {
    if (!appendForProducer(producer, false)) {
      return false;
    }
  }
  return !protocol.stages.empty();
}

bool causeUsesOwnershipSlot(const CanonicalSyncProgram &program,
                            const StorageLoopUsage &usage,
                            const CanonicalOwnershipProtocol &protocol,
                            const CanonicalDemandCause &cause,
                            unsigned slot) {
  const bool validAccesses =
      cause.sourceAccess < program.getAccesses().size() &&
      cause.targetAccess < program.getAccesses().size() &&
      llvm::is_contained(usage.accesses, cause.sourceAccess) &&
      llvm::is_contained(usage.accesses, cause.targetAccess) &&
      slot < protocol.slots.size();
  if (!validAccesses) {
    return false;
  }
  const CanonicalAccess &source = program.getAccess(cause.sourceAccess);
  const CanonicalAccess &target = program.getAccess(cause.targetAccess);
  const CanonicalOwnershipSlot &candidate = protocol.slots[slot];
  const bool exactSameSlot = accessHasExactSlot(source) &&
                             accessHasExactSlot(target) &&
                             source.aliasRoot == candidate.root &&
                             target.aliasRoot == candidate.root &&
                             sameInterval(source.intervals.front(),
                                          candidate.interval) &&
                             sameInterval(target.intervals.front(),
                                          candidate.interval);
  return exactSameSlot;
}

std::optional<unsigned>
ownershipDemandSlot(const CanonicalSyncProgram &program,
                    const StorageLoopUsage &usage,
                    const CanonicalOwnershipProtocol &protocol,
                    const CanonicalDemand &demand) {
  for (unsigned slot = 0; slot < protocol.slots.size(); ++slot) {
    const bool matchingCause = llvm::any_of(
        demand.causes, [&](const CanonicalDemandCause &cause) {
          return causeUsesOwnershipSlot(program, usage, protocol, cause,
                                        slot);
        });
    if (matchingCause) {
      return slot;
    }
  }
  return std::nullopt;
}

bool phaseBelongsToOwnershipStage(const CanonicalOwnershipProtocol &protocol,
                                  unsigned slot, CanonicalPhaseId phase) {
  return llvm::any_of(
      protocol.stages, [&](const CanonicalOwnershipStage &stage) {
        return stage.slot == slot &&
               (llvm::is_contained(stage.producers, phase) ||
                llvm::is_contained(stage.consumers, phase));
      });
}

bool slotHasOwnershipCycle(const CanonicalSyncProgram &program,
                           const StorageLoopUsage &usage,
                           const CanonicalOwnershipProtocol &protocol,
                           unsigned slot) {
  SmallVector<CanonicalPhaseId, 16> phases;
  struct Edge {
    CanonicalPhaseId source = kInvalidCanonicalSyncId;
    CanonicalPhaseId target = kInvalidCanonicalSyncId;
    CanonicalIterationRelation relation = CanonicalIterationRelation::Same;
  };
  SmallVector<Edge, 16> edges;
  bool hasRecurrence = false;
  for (const CanonicalDemand &demand : program.getDemands()) {
    CanonicalIterationRelation relation;
    const std::optional<unsigned> demandSlot =
        ownershipDemandSlot(program, usage, protocol, demand);
    const bool relevant = demand.requirement == CanonicalRequirement::Completion &&
                          demandSlot == slot &&
                          distanceIsLocalToLoop(
                              demand, protocol.recurrenceLoop, relation) &&
                          phaseBelongsToOwnershipStage(protocol, slot,
                                                       demand.source) &&
                          phaseBelongsToOwnershipStage(protocol, slot,
                                                       demand.target);
    if (!relevant) {
      continue;
    }
    phases.push_back(demand.source);
    phases.push_back(demand.target);
    edges.push_back({demand.source, demand.target, relation});
    hasRecurrence |= relation == CanonicalIterationRelation::AnyPositive;
  }
  canonicalize(phases);
  if (!hasRecurrence || phases.size() < 2U ||
      phases.size() > kMaximumLifecyclePhases) {
    return false;
  }
  SmallVector<unsigned, 16> local(program.getPhases().size(),
                                  kInvalidCanonicalSyncId);
  for (unsigned index = 0; index < phases.size(); ++index) {
    local[phases[index]] = index;
  }
  SmallVector<BitVector, 16> reachable;
  reachable.reserve(phases.size());
  for (unsigned index = 0; index < phases.size(); ++index) {
    reachable.emplace_back(phases.size());
    reachable.back().set(index);
  }
  for (const Edge &edge : edges) {
    reachable[local[edge.source]].set(local[edge.target]);
  }
  for (unsigned connector = 0; connector < phases.size(); ++connector) {
    for (unsigned source = 0; source < phases.size(); ++source) {
      if (reachable[source].test(connector)) {
        reachable[source] |= reachable[connector];
      }
    }
  }
  return llvm::any_of(edges, [&](const Edge &edge) {
    const unsigned source = local[edge.source];
    const unsigned target = local[edge.target];
    return source != target && reachable[source].test(target) &&
           reachable[target].test(source);
  });
}

std::optional<CanonicalDemandId> findOwnershipWitnessDemand(
    const CanonicalSyncProgram &program, ArrayRef<const StorageLoopUsage *> usages,
    const CanonicalOwnershipProtocol &protocol, unsigned slot,
    CanonicalPhaseId source, CanonicalPhaseId target,
    CanonicalIterationRelation requiredRelation) {
  if (slot >= usages.size() || !usages[slot]) {
    return std::nullopt;
  }
  for (const CanonicalDemand &demand : program.getDemands()) {
    CanonicalIterationRelation relation;
    const bool matches =
        demand.requirement == CanonicalRequirement::Completion &&
        demand.source == source && demand.target == target &&
        demandUsesExactSlot(program, protocol.slots[slot], demand) &&
        distanceIsLocalToLoop(demand, protocol.recurrenceLoop, relation) &&
        relation == requiredRelation;
    if (matches) {
      return demand.id;
    }
  }
  return std::nullopt;
}

bool buildOwnershipWitness(const CanonicalSyncProgram &program,
                           ArrayRef<const StorageLoopUsage *> usages,
                           CanonicalOwnershipProtocol &protocol) {
  if (protocol.period == 0U || protocol.reuseDistance == 0U ||
      protocol.reuseDistance == std::numeric_limits<unsigned>::max()) {
    return false;
  }
  // Observe every residue once and carry its release edge through the next
  // reuse of that physical slot.  max(period, reuse) + 1 is insufficient for
  // the final residue of a modulo schedule: for period two it observes 0->2,
  // but not the equally important 1->3 ownership transfer.
  protocol.witnessHorizon = protocol.period + protocol.reuseDistance;
  for (const CanonicalOwnershipStage &stage : protocol.stages) {
    if (stage.slot >= protocol.slots.size()) {
      return false;
    }
    const CanonicalPhaseId producer = stage.producers.front();
    const CanonicalPhaseId firstConsumer = stage.consumers.front();
    const CanonicalPhaseId lastConsumer = stage.consumers.back();
    const std::optional<CanonicalDemandId> ready = findOwnershipWitnessDemand(
        program, usages, protocol, stage.slot, producer, firstConsumer,
        stage.readyRelation);
    const std::optional<CanonicalDemandId> release = stage.initialProducer
        ? std::optional<CanonicalDemandId>()
        : findOwnershipWitnessDemand(
              program, usages, protocol, stage.slot, lastConsumer, producer,
              CanonicalIterationRelation::AnyPositive);
    if (!ready || (!stage.initialProducer && !release)) {
      protocol.witnessEdges.clear();
      protocol.witnessHorizon = 0U;
      return false;
    }
    if (stage.initialProducer) {
      protocol.witnessEdges.push_back(
          {CanonicalOwnershipWitnessKind::Ready, stage.lane, producer,
           firstConsumer, 0U, 0U});
      continue;
    }
    const BitVector producerActive = ownershipStageResidues(
        program, protocol.recurrenceLoop, protocol.period,
        stage.producerGuard);
    const BitVector consumerActive = ownershipStageResidues(
        program, protocol.recurrenceLoop, protocol.period,
        stage.consumerGuard);
    for (unsigned iteration = 0; iteration < protocol.witnessHorizon;
         ++iteration) {
      if (!producerActive.test(iteration % protocol.period)) {
        continue;
      }
      const unsigned consumerIteration = iteration + stage.readyDistance;
      const unsigned nextProducerIteration =
          consumerIteration + stage.releaseDistance;
      if (consumerIteration >= protocol.witnessHorizon ||
          nextProducerIteration >= protocol.witnessHorizon ||
          !consumerActive.test(consumerIteration % protocol.period)) {
        continue;
      }
      protocol.witnessEdges.push_back(
          {CanonicalOwnershipWitnessKind::Ready, stage.lane, producer,
           firstConsumer, iteration, consumerIteration});
      protocol.witnessEdges.push_back(
          {CanonicalOwnershipWitnessKind::Release, stage.lane, lastConsumer,
           producer, consumerIteration, nextProducerIteration});
    }
  }
  return true;
}

void populateOwnershipWitnesses(const CanonicalSyncProgram &program,
                                CanonicalOwnershipProtocol &protocol) {
  for (const CanonicalDemand &demand : program.getDemands()) {
    if (!canonicalOwnershipProtocolCoversDemand(program, protocol, demand)) {
      continue;
    }
    protocol.witnessDemands.push_back(demand.id);
    const CanonicalMechanismId parent =
        program.getDirectMechanisms()[demand.id];
    if (!llvm::is_contained(protocol.parentMechanisms, parent)) {
      protocol.parentMechanisms.push_back(parent);
    }
  }
  canonicalize(protocol.parentMechanisms);
  canonicalize(protocol.witnessDemands);
}

bool getExactUsageSlot(const CanonicalSyncProgram &program,
                       const StorageLoopUsage &usage,
                       CanonicalOwnershipSlot &slot) {
  if (usage.accesses.empty()) {
    return false;
  }
  const CanonicalAccess &first = program.getAccess(usage.accesses.front());
  if (!accessHasExactSlot(first)) {
    return false;
  }
  slot.root = usage.root->root;
  slot.interval = first.intervals.front();
  slot.slotExpression = first.slotExpression;
  return llvm::all_of(usage.accesses, [&](CanonicalAccessId accessId) {
    const CanonicalAccess &access = program.getAccess(accessId);
    return accessHasExactSlot(access) && access.aliasRoot == slot.root &&
           sameInterval(access.intervals.front(), slot.interval);
  });
}

bool ownershipSlotsAreDisjoint(
    ArrayRef<CanonicalOwnershipSlot> slots) {
  for (std::size_t first = 0; first < slots.size(); ++first) {
    for (std::size_t second = first + 1U; second < slots.size(); ++second) {
      const CanonicalByteInterval &left = slots[first].interval;
      const CanonicalByteInterval &right = slots[second].interval;
      if (*left.end() > right.begin && *right.end() > left.begin) {
        return false;
      }
    }
  }
  return true;
}

bool buildOwnershipProtocolFamily(
    const CanonicalSyncProgram &program, const StorageProtocolFamily &family,
    CanonicalOwnershipProtocol &protocol,
    SmallVectorImpl<const StorageLoopUsage *> &slotUsages) {
  const std::size_t depth = family.lanes.size();
  if (depth == 0U || depth > 4U) {
    return false;
  }
  protocol.owner = family.loop;
  protocol.recurrenceLoop = family.loop;
  protocol.producer = family.lanes.front()->root->producer;
  protocol.consumer = family.lanes.front()->root->consumer;
  protocol.familyKey = family.key;
  protocol.depth = static_cast<unsigned>(depth);

  struct ParsedSlot {
    CanonicalOwnershipSlot slot;
    const StorageLoopUsage *usage = nullptr;
    SmallVector<CanonicalOwnershipStage, 8> stages;
    BitVector residues;
  };
  SmallVector<ParsedSlot, 4> parsedSlots;
  for (const StorageLoopUsage *usage : family.lanes) {
    ParsedSlot parsed;
    if (!usage || !getExactUsageSlot(program, *usage, parsed.slot)) {
      return false;
    }
    CanonicalOwnershipProtocol temporary;
    const bool stationary =
        depth == 1U && appendStationaryOwnershipStage(
                           program, *usage->root, *usage, temporary);
    if (!stationary) {
      SmallVector<GuardedLaneAccesses, 2> paths;
      for (CanonicalAccessId accessId : usage->accesses) {
        const CanonicalPhase &phase =
            program.getPhase(program.getAccess(accessId).phase);
        auto path = llvm::find_if(paths, [&](const GuardedLaneAccesses &item) {
          return sameGuard(item.guard, phase.controlPath);
        });
        if (path == paths.end()) {
          paths.push_back({0U, phase.controlPath, {}});
          path = std::prev(paths.end());
        }
        path->accesses.push_back(accessId);
      }
      const bool flat = paths.size() == 1U &&
                        appendOwnershipStages(program, *usage->root,
                                              paths.front(), temporary);
      if (!flat) {
        temporary.stages.clear();
        if (!appendStructuredOwnershipStages(program, *usage->root, *usage,
                                             temporary)) {
          temporary.stages.clear();
          if (!appendPhaseShiftedOwnershipStages(
                  program, *usage->root, *usage, parsed.slot, temporary)) {
            return false;
          }
        }
      }
    }
    if (temporary.stages.empty()) {
      return false;
    }
    parsed.usage = usage;
    parsed.stages = std::move(temporary.stages);
    parsedSlots.push_back(std::move(parsed));
  }

  SmallVector<CanonicalOwnershipSlot, 4> physicalSlots;
  for (const ParsedSlot &parsed : parsedSlots) {
    physicalSlots.push_back(parsed.slot);
  }
  if (!ownershipSlotsAreDisjoint(physicalSlots)) {
    return false;
  }

  SmallVector<CanonicalOwnershipStage, 16> allStages;
  for (const ParsedSlot &parsed : parsedSlots) {
    llvm::append_range(allStages, parsed.stages);
  }
  const std::optional<unsigned> period =
      inferOwnershipPeriod(program, family.loop, allStages);
  if (!period || *period == 0U || *period > 4U ||
      (*period > 1U &&
       (!loopHasUnitStep(program, family.loop) ||
        !loopHasKnownNonnegativeOrigin(program, family.loop)))) {
    return false;
  }
  protocol.period = *period;

  BitVector assignedResidues(*period);
  bool uniqueResiduePartition = depth == *period;
  bool everySlotIsAlwaysActive = true;
  bool hasPhaseShiftedStage = false;
  for (ParsedSlot &parsed : parsedSlots) {
    for (CanonicalOwnershipStage &stage : parsed.stages) {
      const BitVector producerResidues = ownershipStageResidues(
          program, family.loop, *period, stage.producerGuard);
      const BitVector consumerResidues = ownershipStageResidues(
          program, family.loop, *period, stage.consumerGuard);
      if (stage.initialProducer) {
        stage.readyDistance = 0U;
        stage.releaseDistance = 0U;
        continue;
      }
      const bool positiveReady =
          stage.readyRelation == CanonicalIterationRelation::AnyPositive;
      const std::optional<unsigned> readyDistance = uniqueCyclicDistance(
          producerResidues, consumerResidues, positiveReady);
      const std::optional<unsigned> releaseDistance = uniqueCyclicDistance(
          consumerResidues, producerResidues, true);
      if (!readyDistance || !releaseDistance) {
        return false;
      }
      stage.readyDistance = *readyDistance;
      stage.releaseDistance = *releaseDistance;
      hasPhaseShiftedStage |= positiveReady;
    }
    parsed.residues = ownershipSlotResidues(
        program, family.loop, *period, parsed.stages);
    parsed.slot.reuseDistance =
        maximumCyclicReuseDistance(parsed.residues);
    for (const CanonicalOwnershipStage &stage : parsed.stages) {
      if (!stage.initialProducer) {
        parsed.slot.reuseDistance =
            std::max(parsed.slot.reuseDistance,
                     stage.readyDistance + stage.releaseDistance);
      }
    }
    if (parsed.slot.reuseDistance == 0U) {
      return false;
    }
    everySlotIsAlwaysActive &= parsed.residues.all();
    if (parsed.residues.count() != 1U) {
      uniqueResiduePartition = false;
      continue;
    }
    const unsigned residue = parsed.residues.find_first();
    if (assignedResidues.test(residue)) {
      uniqueResiduePartition = false;
    }
    assignedResidues.set(residue);
  }
  uniqueResiduePartition &= assignedResidues.all();
  if (depth > 1U && !uniqueResiduePartition && !everySlotIsAlwaysActive) {
    return false;
  }

  protocol.lanes.resize(uniqueResiduePartition && !hasPhaseShiftedStage
                            ? depth
                            : 1U);
  slotUsages.resize(depth);
  for (unsigned slotIndex = 0; slotIndex < depth; ++slotIndex) {
    ParsedSlot &parsed = parsedSlots[slotIndex];
    const unsigned lane = uniqueResiduePartition && !hasPhaseShiftedStage
                              ? parsed.residues.find_first()
                              : 0U;
    parsed.slot.lane = lane;
    protocol.reuseDistance =
        std::max(protocol.reuseDistance, parsed.slot.reuseDistance);
    slotUsages[slotIndex] = parsed.usage;
    protocol.roots.push_back(parsed.slot.root);
    protocol.slots.push_back(parsed.slot);
    for (CanonicalOwnershipStage &stage : parsed.stages) {
      stage.slot = slotIndex;
      stage.lane = lane;
      protocol.stages.push_back(std::move(stage));
    }
  }
  llvm::stable_sort(
      protocol.stages,
      [&](const CanonicalOwnershipStage &first,
          const CanonicalOwnershipStage &second) {
        const unsigned firstOrder =
            program.getPhase(first.producers.front()).sourceOrder;
        const unsigned secondOrder =
            program.getPhase(second.producers.front()).sourceOrder;
        return std::tie(firstOrder, first.slot, first.lane) <
               std::tie(secondOrder, second.slot, second.lane);
      });
  return true;
}

void synthesizeStorageOwnershipProtocols(CanonicalSyncProgram &program) {
  const SmallVector<StorageRootUsage, 8> roots =
      buildNormalizedStorageRoots(program);
  const SmallVector<StorageLoopUsage, 8> usages =
      buildStorageLoopUsages(program, roots);
  const SmallVector<StorageProtocolFamily, 8> families =
      buildStorageProtocolFamilies(usages);
  constexpr std::size_t kMaximumOwnershipProtocols = 32U;
  constexpr std::size_t kMaximumOwnershipStages = 64U;
  SmallVector<std::pair<CanonicalRegionId, std::string>, 8>
      successfulGroupedFamilies;
  for (const StorageProtocolFamily &family : families) {
    if (program.getOwnershipProtocols().size() >=
        kMaximumOwnershipProtocols) {
      return;
    }
    const bool coveredByGroupedFamily =
        family.lanes.size() == 1U &&
        llvm::is_contained(successfulGroupedFamilies,
                           std::make_pair(family.loop, family.key));
    if (coveredByGroupedFamily) {
      continue;
    }
    CanonicalOwnershipProtocol protocol;
    SmallVector<const StorageLoopUsage *, 4> slotUsages;
    if (!buildOwnershipProtocolFamily(program, family, protocol,
                                      slotUsages) ||
        protocol.stages.empty() ||
        protocol.stages.size() > kMaximumOwnershipStages) {
      continue;
    }
    const bool allSlotsCyclic = llvm::all_of(
        llvm::seq<unsigned>(0, protocol.slots.size()), [&](unsigned slot) {
          return slotHasOwnershipCycle(program, *slotUsages[slot], protocol,
                                       slot);
        });
    if (!allSlotsCyclic) {
      continue;
    }
    if (!buildOwnershipWitness(program, slotUsages, protocol)) {
      continue;
    }
    populateOwnershipWitnesses(program, protocol);
    const bool hasSame = llvm::any_of(
        protocol.witnessDemands, [&](CanonicalDemandId demand) {
          CanonicalIterationRelation relation;
          return distanceIsLocalToLoop(program.getDemand(demand),
                                       protocol.recurrenceLoop, relation) &&
                 relation == CanonicalIterationRelation::Same;
        });
    const bool hasRecurrence = llvm::any_of(
        protocol.witnessDemands, [&](CanonicalDemandId demand) {
          CanonicalIterationRelation relation;
          return distanceIsLocalToLoop(program.getDemand(demand),
                                       protocol.recurrenceLoop, relation) &&
                 relation == CanonicalIterationRelation::AnyPositive;
        });
    // A direct recurrence mechanism can already intern every accumulator
    // ready/release demand into one descriptor.  Keep the independently
    // discovered ownership protocol when it has distinct same-iteration and
    // recurrence witnesses even if those witnesses share one direct parent;
    // later protocol composition needs the explicit storage lifetime.
    if (!hasSame || !hasRecurrence || protocol.witnessDemands.size() < 2U) {
      continue;
    }
    const CanonicalOwnershipStage &representative = protocol.stages.front();
    CanonicalMechanism mechanism;
    mechanism.kind = CanonicalMechanismKind::PeriodicOwnership;
    mechanism.source = protocol.producer;
    mechanism.target = protocol.consumer;
    mechanism.sourcePoint = representative.ready;
    mechanism.targetPoint = representative.readAcquire;
    llvm::append_range(mechanism.origins, protocol.witnessDemands);
    mechanism.actionRegion = protocol.recurrenceLoop;
    mechanism.recurrenceLoop = protocol.recurrenceLoop;
    program.appendOwnershipProtocol(std::move(protocol), std::move(mechanism));
    if (family.lanes.size() > 1U) {
      successfulGroupedFamilies.emplace_back(family.loop, family.key);
    }
  }
}

void proposeStorageLifecycles(CanonicalSyncProgram &program,
                              ProposalBuilder &builder) {
  synthesizeStorageOwnershipProtocols(program);
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
    SmallVector<CanonicalPhaseId, 16> phases;
    for (CanonicalDemandId demandId : family.demands) {
      const CanonicalDemand &demand = program.getDemand(demandId);
      phases.push_back(demand.source);
      phases.push_back(demand.target);
    }
    canonicalize(phases);
    const bool lifecycleBudgetExceeded =
        phases.size() > kMaximumLifecyclePhases ||
        family.demands.size() > kMaximumLifecycleDemands;
    if (lifecycleBudgetExceeded) {
      continue;
    }
    SmallVector<unsigned, 16> localPhase(program.getPhases().size(),
                                         kInvalidCanonicalSyncId);
    for (unsigned index = 0; index < phases.size(); ++index) {
      localPhase[phases[index]] = index;
    }
    SmallVector<SmallVector<unsigned, 4>, 0> outgoing(phases.size());
    for (CanonicalDemandId demandId : family.demands) {
      const CanonicalDemand &demand = program.getDemand(demandId);
      const unsigned source = localPhase[demand.source];
      const unsigned target = localPhase[demand.target];
      if (!llvm::is_contained(outgoing[source], target)) {
        outgoing[source].push_back(target);
      }
    }
    SmallVector<BitVector, 0> reachable;
    reachable.reserve(phases.size());
    for (unsigned source = 0; source < phases.size(); ++source) {
      reachable.emplace_back(phases.size());
      SmallVector<unsigned, 16> worklist{source};
      while (!worklist.empty()) {
        const unsigned current = worklist.pop_back_val();
        for (unsigned target : outgoing[current]) {
          if (reachable[source].test(target)) {
            continue;
          }
          reachable[source].set(target);
          worklist.push_back(target);
        }
      }
    }
    BitVector assigned(phases.size());
    for (unsigned seed = 0; seed < phases.size(); ++seed) {
      if (assigned.test(seed)) {
        continue;
      }
      BitVector component(phases.size());
      for (unsigned candidate = 0; candidate < phases.size(); ++candidate) {
        const bool mutuallyReachable =
            candidate == seed ||
            (reachable[seed].test(candidate) &&
             reachable[candidate].test(seed));
        if (mutuallyReachable) {
          component.set(candidate);
          assigned.set(candidate);
        }
      }
      const bool selfCycle = reachable[seed].test(seed);
      const bool isAcyclicSingleton = component.count() < 2U && !selfCycle;
      if (isAcyclicSingleton) {
        continue;
      }
      SmallVector<CanonicalDemandId, 16> componentDemands;
      bool hasPositiveDistance = false;
      for (CanonicalDemandId demandId : family.demands) {
        const CanonicalDemand &demand = program.getDemand(demandId);
        const bool outsideComponent =
            !component.test(localPhase[demand.source]) ||
            !component.test(localPhase[demand.target]);
        if (outsideComponent) {
          continue;
        }
        componentDemands.push_back(demandId);
        hasPositiveDistance |= llvm::any_of(
            demand.iterationDistance,
            [](const CanonicalLoopDistance &distance) {
              return distance.relation ==
                     CanonicalIterationRelation::AnyPositive;
            });
      }
      if (!hasPositiveDistance) {
        continue;
      }
      std::string storage;
      llvm::raw_string_ostream os(storage);
      os << "semantic=storage-lifecycle-scc;root=a" << family.rootOrdinal
         << ";space="
         << (family.unknownSpace ? StringRef("unknown")
                                 : stringifyAddressSpace(family.space))
         << ";phases=" << component.count();
      builder.append(CanonicalStructuralProposalKind::StorageLifecycle,
                     family.owner, 0U, os.str(),
                     directMechanisms(program, componentDemands),
                     componentDemands);
    }
  }
}

} // namespace

LogicalResult
mlir::pto::proposeCanonicalSyncStructuralGroups(CanonicalSyncProgram &program,
                                                CanonicalStructuralCoverFamilies
                                                    enabledFamilies) {
  const bool invalidState = !program.isGraphFrozen() || program.isFrozen() ||
                            !program.mechanismCatalogComplete ||
                            program.coverageCatalogComplete ||
                            program.structuralProposalCatalogComplete;
  if (invalidState) {
    return program.getFunction().emitError(
        "canonical sync structural proposals require a complete direct "
        "mechanism catalog");
  }
  const bool invalidFamilyMask =
      (enabledFamilies & ~kAllCanonicalStructuralCoverFamilies) != 0U;
  if (invalidFamilyMask) {
    return program.getFunction().emitError(
        "canonical sync structural proposal family mask is invalid");
  }
  if (enabledFamilies != 0U) {
    ProposalBuilder builder{program};
    SmallVector<OwnerLevelGraph, 8> ownerGraphs;
    const bool needsOwnerGraphs =
        hasCanonicalStructuralCoverFamily(
            enabledFamilies, CanonicalStructuralCoverFamily::Level) ||
        hasCanonicalStructuralCoverFamily(
            enabledFamilies, CanonicalStructuralCoverFamily::Transitive) ||
        hasCanonicalStructuralCoverFamily(
            enabledFamilies, CanonicalStructuralCoverFamily::Connector) ||
        hasCanonicalStructuralCoverFamily(
            enabledFamilies, CanonicalStructuralCoverFamily::Semantic);
    if (needsOwnerGraphs) {
      for (const CanonicalRegion &region : program.getRegions()) {
        OwnerLevelGraph graph = buildOwnerLevels(program, region.id);
        const bool unusableGraph = !graph.acyclic || graph.demands.size() < 2U;
        if (unusableGraph) {
          continue;
        }
        ownerGraphs.push_back(std::move(graph));
      }
    }
    if (hasCanonicalStructuralCoverFamily(
            enabledFamilies, CanonicalStructuralCoverFamily::Transitive)) {
      for (const OwnerLevelGraph &graph : ownerGraphs) {
        proposeTransitiveBasis(program, graph, builder);
      }
    }
    if (hasCanonicalStructuralCoverFamily(
            enabledFamilies, CanonicalStructuralCoverFamily::Connector)) {
      for (const OwnerLevelGraph &graph : ownerGraphs) {
        proposeConnectorNeighborhoods(program, graph, builder);
      }
    }
    const bool includeLevel = hasCanonicalStructuralCoverFamily(
        enabledFamilies, CanonicalStructuralCoverFamily::Level);
    const bool includeSemantic = hasCanonicalStructuralCoverFamily(
        enabledFamilies, CanonicalStructuralCoverFamily::Semantic);
    for (const OwnerLevelGraph &graph : ownerGraphs) {
      proposeLevelBoundaries(program, graph, builder, includeLevel,
                             includeSemantic);
    }
    if (hasCanonicalStructuralCoverFamily(
            enabledFamilies, CanonicalStructuralCoverFamily::Storage)) {
      proposeStorageLifecycles(program, builder);
    }
  }
  program.structuralProposalCatalogComplete = true;
  return success();
}
