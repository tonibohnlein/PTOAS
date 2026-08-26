// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "PTO/Transforms/CanonicalSync/SyncCoverColumnGeneration.h"

#include "PTO/Transforms/CanonicalSync/SyncCoverDescriptorBuilder.h"

#include <algorithm>
#include <map>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::pto;

namespace {

constexpr std::uint64_t kCanonicalTag = 0x43414e4fULL;
constexpr std::uint64_t kMergedTag = 0x4d455247ULL;
constexpr std::uint64_t kBarrierTag = 0x42415252ULL;

std::uint64_t makeProviderIdentity(std::uint64_t tag, std::size_t ordinal) {
  constexpr unsigned kTagShift = 32;
  return (tag << kTagShift) ^ static_cast<std::uint64_t>(ordinal);
}

bool chargeInspections(SyncCoverColumnGeneratorReport &report,
                       const SyncCoverColumnGenerationOptions &options,
                       std::size_t count = 1) {
  if (count > options.maximumInspections - report.inspections) {
    report.truncated = true;
    return false;
  }
  report.inspections += count;
  return true;
}

bool chargeCandidate(SyncCoverColumnGeneratorReport &report,
                     const SyncCoverColumnGenerationOptions &options,
                     std::size_t supplyEdges) {
  if (report.candidates == options.maximumCandidates ||
      supplyEdges > options.maximumSupplyEdges - report.supplyEdges) {
    report.truncated = true;
    return false;
  }
  ++report.candidates;
  report.supplyEdges += supplyEdges;
  return true;
}

const SyncCoverResourceDomain *
findDomain(const SyncCoverMechanismUniverse &universe,
           SyncCoverResourceKind kind, std::uint32_t source,
           std::uint32_t target, std::uint64_t poolIdentity = 0) {
  const auto &domains = universe.getResourceDomains();
  auto found = std::find_if(domains.begin(), domains.end(),
                            [&](const SyncCoverResourceDomain &domain) {
                              return domain.kind == kind &&
                                     domain.sourceResource == source &&
                                     domain.targetResource == target &&
                                     domain.poolIdentity == poolIdentity;
                            });
  return found == domains.end() ? nullptr : &*found;
}

bool hasEmptyGuard(const SyncCoverGuard &guard) {
  return guard.literals.empty();
}

struct LinearDemand {
  SyncCoverDemandId demand = 0;
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  SyncCoverScopeId scope = 0;
  std::uint32_t sourceResource = 0;
  std::uint32_t targetResource = 0;
  std::size_t sourceOrder = 0;
  std::size_t targetOrder = 0;
};

using LinearDomainKey =
    std::tuple<SyncCoverScopeId, std::uint32_t, std::uint32_t>;

std::optional<LinearDemand>
getLinearDemand(const SyncCoverColumnGenerationContext &context,
                const SyncCoverGraph &graph, SyncCoverDemandId demandId) {
  const auto &demands = graph.getDemands();
  const auto &nodes = graph.getNodes();
  const auto &scopes = graph.getScopes();
  if (demandId >= demands.size()) {
    return std::nullopt;
  }
  const SyncCoverDemand &demand = demands[demandId];
  if (demand.distance != 0 || demand.source >= nodes.size() ||
      demand.target >= nodes.size() || demand.scope >= scopes.size()) {
    return std::nullopt;
  }
  const SyncCoverNode &source = nodes[demand.source];
  const SyncCoverNode &target = nodes[demand.target];
  const bool isUnconditionalLinearContext =
      source.scope == demand.scope && target.scope == demand.scope &&
      hasEmptyGuard(source.guard) && hasEmptyGuard(target.guard) &&
      hasEmptyGuard(demand.sourceGuard) &&
      hasEmptyGuard(demand.targetGuard) && source.order < target.order &&
      scopes[demand.scope].timeline.has_value();
  if (!isUnconditionalLinearContext) {
    return std::nullopt;
  }
  return LinearDemand{demandId,
                      demand.source,
                      demand.target,
                      demand.scope,
                      source.resource,
                      target.resource,
                      source.order,
                      target.order};
}

std::optional<std::map<LinearDomainKey, std::vector<LinearDemand>>>
groupLinearCrossPipeDemands(
    const SyncCoverGraph &graph,
    const SyncCoverColumnGenerationContext &context,
    SyncCoverColumnGeneratorReport &report) {
  std::map<LinearDomainKey, std::vector<LinearDemand>> groups;
  for (SyncCoverDemandId demandId : context.activeDemands) {
    if (!chargeInspections(report, context.options)) {
      return std::nullopt;
    }
    const std::optional<LinearDemand> demand =
        getLinearDemand(context, graph, demandId);
    if (!demand) {
      continue;
    }
    if (demand->sourceResource == demand->targetResource) {
      continue;
    }
    groups[{demand->scope, demand->sourceResource, demand->targetResource}]
        .push_back(*demand);
  }
  for (auto &entry : groups) {
    std::sort(entry.second.begin(), entry.second.end(),
              [](const LinearDemand &first, const LinearDemand &second) {
                return std::tie(first.sourceOrder, first.targetOrder,
                                first.demand) <
                       std::tie(second.sourceOrder, second.targetOrder,
                                second.demand);
              });
  }
  return groups;
}

std::optional<SyncCoverMechanismDescriptor>
makeMergedDescriptor(const SyncCoverResourceDomain &domain,
                     const std::vector<LinearDemand> &members,
                     SyncCoverNodeId setAfter, SyncCoverNodeId waitBefore,
                     SyncCoverScopeId scope,
                     std::uint64_t providerIdentity) {
  SyncCoverMechanismDescriptorBuilder builder(
      SyncCoverMechanismKind::VerifiedProtocol, providerIdentity);
  const SyncCoverDescriptorActionRef produce = builder.addAction(
      SyncCoverResourceActionKind::Produce, domain.sourceResource,
      {SyncCoverAnchorKind::AfterNode, setAfter, 0, 0});
  const SyncCoverDescriptorActionRef consume = builder.addAction(
      SyncCoverResourceActionKind::Consume, domain.targetResource,
      {SyncCoverAnchorKind::BeforeNode, waitBefore, 0, 0});
  std::vector<SyncCoverProtocolSupply> supplies;
  supplies.reserve(members.size());
  for (const LinearDemand &member : members) {
    SyncCoverEdge edge;
    edge.source = member.source;
    edge.target = member.target;
    edge.kind = SyncCoverEdgeKind::CompletionSupply;
    edge.scope = scope;
    supplies.push_back({edge, produce, consume});
  }
  if (!builder.addProtocolLane(domain, scope, 0, 1, {produce, consume},
                               std::move(supplies))) {
    return std::nullopt;
  }
  return std::move(builder).takeDescriptor();
}

class CanonicalEventGenerator final : public SyncCoverColumnGenerator {
public:
  const char *name() const override { return "canonical"; }

  SyncCoverColumnGeneratorReport
  generate(const SyncCoverColumnGenerationContext &context,
           SyncCoverMechanismUniverse &universe) const override {
    SyncCoverColumnGeneratorReport report;
    report.generator = name();
    const SyncCoverGraph &graph = universe.getGraph();
    const auto groups = groupLinearCrossPipeDemands(graph, context, report);
    if (!groups) {
      return report;
    }
    for (const auto &entry : *groups) {
      const auto [scope, source, target] = entry.first;
      const SyncCoverResourceDomain *domain = findDomain(
          universe, SyncCoverResourceKind::EventId, source, target);
      if (!domain) {
        continue;
      }
      for (const LinearDemand &demand : entry.second) {
        if (!chargeCandidate(report, context.options, 1)) {
          return report;
        }
        const auto descriptor = makeSyncCoverCanonicalEvent(
            *domain, demand.source, demand.target, scope, 1,
            makeProviderIdentity(kCanonicalTag, demand.demand));
        const SyncCoverMechanismResult added =
            descriptor ? universe.addMechanism(*descriptor)
                       : SyncCoverMechanismResult{
                             SyncCoverMechanismError::InvalidMechanism,
                             std::nullopt};
        if (added && added.index) {
          ++report.admitted;
        } else {
          ++report.rejectedByVerifier;
        }
      }
    }
    return report;
  }
};

class MergedPrefixEventGenerator final : public SyncCoverColumnGenerator {
public:
  const char *name() const override { return "merged-prefix"; }

  SyncCoverColumnGeneratorReport
  generate(const SyncCoverColumnGenerationContext &context,
           SyncCoverMechanismUniverse &universe) const override {
    SyncCoverColumnGeneratorReport report;
    report.generator = name();
    const SyncCoverGraph &graph = universe.getGraph();
    const auto groups = groupLinearCrossPipeDemands(graph, context, report);
    if (!groups) {
      return report;
    }
    std::size_t ordinal = 0;
    for (const auto &entry : *groups) {
      const auto [scope, source, target] = entry.first;
      if (!context.target.hasPrefixSetSemantics(source)) {
        report.skippedByCapability += entry.second.size();
        continue;
      }
      const SyncCoverResourceDomain *domain = findDomain(
          universe, SyncCoverResourceKind::EventId, source, target);
      if (!domain) {
        continue;
      }
      const std::vector<LinearDemand> &points = entry.second;
      if (points.size() < 2) {
        continue;
      }
      std::size_t minimumTargetOrder = points.front().targetOrder;
      SyncCoverNodeId minimumTarget = points.front().target;
      std::vector<LinearDemand> prefix;
      for (std::size_t index = 0; index < points.size(); ++index) {
        prefix.push_back(points[index]);
        if (points[index].targetOrder < minimumTargetOrder) {
          minimumTargetOrder = points[index].targetOrder;
          minimumTarget = points[index].target;
        }
        const bool sourceBoundary =
            index + 1 == points.size() ||
            points[index + 1].sourceOrder != points[index].sourceOrder;
        if (!sourceBoundary || prefix.size() < 2) {
          continue;
        }
        if (points[index].sourceOrder >= minimumTargetOrder) {
          continue;
        }
        if (!chargeCandidate(report, context.options, prefix.size())) {
          return report;
        }
        const auto descriptor = makeMergedDescriptor(
            *domain, prefix, points[index].source, minimumTarget, scope,
            makeProviderIdentity(kMergedTag, ordinal++));
        const SyncCoverMechanismResult added =
            descriptor
                ? universe.addVerifiedProtocol(
                      *descriptor,
                      [&](const SyncCoverMechanismDescriptor &actual) {
                        return verifySyncCoverMergedPrefixEvent(
                            graph, *domain, actual, context.target);
                      })
                : SyncCoverMechanismResult{
                      SyncCoverMechanismError::InvalidMechanism,
                      std::nullopt};
        if (added && added.index) {
          ++report.admitted;
        } else {
          ++report.rejectedByVerifier;
        }
      }
    }
    return report;
  }
};

class PiercedBarrierGenerator final : public SyncCoverColumnGenerator {
public:
  const char *name() const override { return "pierce-barrier"; }

  SyncCoverColumnGeneratorReport
  generate(const SyncCoverColumnGenerationContext &context,
           SyncCoverMechanismUniverse &universe) const override {
    SyncCoverColumnGeneratorReport report;
    report.generator = name();
    const SyncCoverGraph &graph = universe.getGraph();
    using PipeScopeKey = std::pair<SyncCoverScopeId, std::uint32_t>;
    std::map<PipeScopeKey, std::vector<LinearDemand>> groups;
    for (SyncCoverDemandId demandId : context.activeDemands) {
      if (!chargeInspections(report, context.options)) {
        return report;
      }
      const std::optional<LinearDemand> demand =
          getLinearDemand(context, graph, demandId);
      if (!demand) {
        continue;
      }
      if (demand->sourceResource != demand->targetResource) {
        continue;
      }
      if (context.target.hasHardwareCompletion(demand->sourceResource)) {
        ++report.skippedByCapability;
        continue;
      }
      groups[{demand->scope, demand->sourceResource}].push_back(*demand);
    }

    std::size_t ordinal = 0;
    for (auto &entry : groups) {
      std::vector<LinearDemand> remaining = std::move(entry.second);
      std::sort(remaining.begin(), remaining.end(),
                [](const LinearDemand &first, const LinearDemand &second) {
                  return std::tie(first.targetOrder, first.sourceOrder,
                                  first.demand) <
                         std::tie(second.targetOrder, second.sourceOrder,
                                  second.demand);
                });
      while (!remaining.empty()) {
        const SyncCoverNodeId anchor = remaining.front().target;
        const std::size_t anchorOrder = remaining.front().targetOrder;
        std::vector<LinearDemand> covered;
        std::vector<LinearDemand> next;
        for (const LinearDemand &candidate : remaining) {
          if (!chargeInspections(report, context.options)) {
            return report;
          }
          if (candidate.sourceOrder < anchorOrder &&
              anchorOrder <= candidate.targetOrder) {
            covered.push_back(candidate);
          } else {
            next.push_back(candidate);
          }
        }
        if (!chargeCandidate(report, context.options, covered.size())) {
          return report;
        }
        SyncCoverMechanismDescriptor descriptor;
        descriptor.kind = SyncCoverMechanismKind::Barrier;
        descriptor.providerIdentity =
            makeProviderIdentity(kBarrierTag, ordinal++);
        descriptor.barrier = SyncCoverBarrierPlacement{
            entry.first.second,
            {SyncCoverAnchorKind::BeforeNode, anchor, 0, 0},
            entry.first.first};
        for (const LinearDemand &member : covered) {
          SyncCoverEdge edge;
          edge.source = member.source;
          edge.target = member.target;
          edge.kind = SyncCoverEdgeKind::CompletionSupply;
          edge.scope = member.scope;
          descriptor.supplyEdges.push_back(edge);
        }
        const SyncCoverMechanismResult added =
            universe.addMechanism(descriptor);
        if (added && added.index) {
          ++report.admitted;
        } else {
          ++report.rejectedByVerifier;
        }
        remaining = std::move(next);
      }
    }
    return report;
  }
};

} // namespace

bool mlir::pto::verifySyncCoverMergedPrefixEvent(
    const SyncCoverGraph &graph, const SyncCoverResourceDomain &domain,
    const SyncCoverMechanismDescriptor &descriptor,
    const SyncCoverTargetCapabilities &target) {
  if (!target.hasPrefixSetSemantics(domain.sourceResource) ||
      descriptor.kind != SyncCoverMechanismKind::VerifiedProtocol ||
      descriptor.actions.size() != 2 || descriptor.supplyEdges.size() < 2 ||
      descriptor.resourceUses.size() != 1) {
    return false;
  }
  const SyncCoverResourceAction &produce = descriptor.actions[0];
  const SyncCoverResourceAction &consume = descriptor.actions[1];
  const auto &nodes = graph.getNodes();
  const bool actionShape =
      produce.kind == SyncCoverResourceActionKind::Produce &&
      consume.kind == SyncCoverResourceActionKind::Consume &&
      produce.anchor.kind == SyncCoverAnchorKind::AfterNode &&
      consume.anchor.kind == SyncCoverAnchorKind::BeforeNode &&
      produce.anchor.node < nodes.size() &&
      consume.anchor.node < nodes.size() &&
      produce.resource == domain.sourceResource &&
      consume.resource == domain.targetResource &&
      domain.kind == SyncCoverResourceKind::EventId &&
      domain.sourceResource != domain.targetResource;
  if (!actionShape) {
    return false;
  }
  const SyncCoverNode &setNode = nodes[produce.anchor.node];
  const SyncCoverNode &waitNode = nodes[consume.anchor.node];
  if (setNode.resource != domain.sourceResource ||
      waitNode.resource != domain.targetResource ||
      setNode.order >= waitNode.order ||
      setNode.scope != waitNode.scope || !hasEmptyGuard(setNode.guard) ||
      !hasEmptyGuard(waitNode.guard) ||
      !syncCoverNodeCanProduceCompletion(graph, produce.anchor.node,
                                         domain.targetResource)) {
    return false;
  }
  bool anchorIsProducer = false;
  bool anchorIsConsumer = false;
  for (const SyncCoverEdge &edge : descriptor.supplyEdges) {
    if (edge.kind != SyncCoverEdgeKind::CompletionSupply ||
        edge.distance != 0 || edge.source >= nodes.size() ||
        edge.target >= nodes.size() || edge.scope != setNode.scope) {
      return false;
    }
    const SyncCoverNode &source = nodes[edge.source];
    const SyncCoverNode &targetNode = nodes[edge.target];
    const bool dominated =
        source.resource == domain.sourceResource &&
        targetNode.resource == domain.targetResource &&
        source.scope == setNode.scope && targetNode.scope == setNode.scope &&
        hasEmptyGuard(source.guard) && hasEmptyGuard(targetNode.guard) &&
        source.order <= setNode.order && waitNode.order <= targetNode.order;
    if (!dominated) {
      return false;
    }
    anchorIsProducer |= edge.source == produce.anchor.node;
    anchorIsConsumer |= edge.target == consume.anchor.node;
  }
  return anchorIsProducer && anchorIsConsumer;
}

SyncCoverColumnGenerationResult mlir::pto::runSyncCoverColumnGenerators(
    const SyncCoverColumnGenerationContext &context,
    SyncCoverMechanismUniverse &universe,
    const std::vector<std::unique_ptr<SyncCoverColumnGenerator>> &generators) {
  SyncCoverColumnGenerationResult result;
  for (const std::unique_ptr<SyncCoverColumnGenerator> &generator : generators) {
    SyncCoverColumnGeneratorReport report =
        generator->generate(context, universe);
    result.totalAdmitted += report.admitted;
    result.reports.push_back(std::move(report));
  }
  return result;
}

std::unique_ptr<SyncCoverColumnGenerator>
mlir::pto::makeSyncCoverCanonicalEventGenerator() {
  return std::make_unique<CanonicalEventGenerator>();
}

std::unique_ptr<SyncCoverColumnGenerator>
mlir::pto::makeSyncCoverMergedPrefixEventGenerator() {
  return std::make_unique<MergedPrefixEventGenerator>();
}

std::unique_ptr<SyncCoverColumnGenerator>
mlir::pto::makeSyncCoverPiercedBarrierGenerator() {
  return std::make_unique<PiercedBarrierGenerator>();
}
