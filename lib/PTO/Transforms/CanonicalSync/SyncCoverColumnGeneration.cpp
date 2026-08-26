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
constexpr std::uint64_t kUnitRecurrenceTag = 0x554e4954ULL;
constexpr std::uint64_t kRingTag = 0x52494e47ULL;
constexpr std::uint64_t kTokenRingTag = 0x544f4b45ULL;

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
      ++report.skippedByContext;
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

struct RecurrenceDemand {
  SyncCoverDemandId demand = 0;
  SyncCoverNodeId source = 0;
  SyncCoverNodeId target = 0;
  SyncCoverScopeId scope = 0;
  unsigned distance = 0;
};

std::optional<std::vector<RecurrenceDemand>> collectRecurrenceDemands(
    const SyncCoverGraph &graph,
    const SyncCoverColumnGenerationContext &context,
    SyncCoverColumnGeneratorReport &report, unsigned minimumDistance,
    unsigned maximumDistance) {
  std::vector<RecurrenceDemand> result;
  const auto &demands = graph.getDemands();
  const auto &nodes = graph.getNodes();
  const auto &scopes = graph.getScopes();
  for (SyncCoverDemandId demandId : context.activeDemands) {
    if (!chargeInspections(report, context.options)) {
      return std::nullopt;
    }
    if (demandId >= demands.size()) {
      ++report.skippedByContext;
      continue;
    }
    const SyncCoverDemand &demand = demands[demandId];
    const bool eligible =
        demand.distance >= minimumDistance &&
        demand.distance <= maximumDistance && demand.scope < scopes.size() &&
        scopes[demand.scope].isLoop && scopes[demand.scope].timeline &&
        demand.source < nodes.size() && demand.target < nodes.size() &&
        hasEmptyGuard(demand.sourceGuard) &&
        hasEmptyGuard(demand.targetGuard) &&
        hasEmptyGuard(nodes[demand.source].guard) &&
        hasEmptyGuard(nodes[demand.target].guard) &&
        nodes[demand.source].resource != nodes[demand.target].resource;
    if (eligible) {
      result.push_back({demandId, demand.source, demand.target, demand.scope,
                        demand.distance});
    } else {
      ++report.skippedByContext;
    }
  }
  return result;
}

bool hasExactMechanismSupply(const SyncCoverGraph &graph,
                             const RecurrenceDemand &demand) {
  return std::any_of(graph.getEdges().begin(), graph.getEdges().end(),
                     [&](const SyncCoverEdge &edge) {
                       return edge.kind == SyncCoverEdgeKind::CompletionSupply &&
                              edge.mechanism && edge.source == demand.source &&
                              edge.target == demand.target &&
                              edge.scope == demand.scope &&
                              edge.distance == demand.distance &&
                              hasEmptyGuard(edge.sourceGuard) &&
                              hasEmptyGuard(edge.targetGuard);
                     });
}

std::optional<SyncCoverMechanismDescriptor>
makeRingDescriptor(const SyncCoverResourceDomain &domain,
                   const RecurrenceDemand &demand,
                   std::uint64_t providerIdentity) {
  SyncCoverMechanismDescriptorBuilder builder(
      SyncCoverMechanismKind::VerifiedProtocol, providerIdentity);
  std::vector<SyncCoverDescriptorActionRef> actions;
  actions.reserve(2 * static_cast<std::size_t>(demand.distance) + 2);
  for (unsigned lane = 0; lane < demand.distance; ++lane) {
    actions.push_back(builder.addAction(
        SyncCoverResourceActionKind::Produce, domain.sourceResource,
        {SyncCoverAnchorKind::ScopeEntry, 0, demand.scope}));
  }
  const SyncCoverDescriptorActionRef consume = builder.addAction(
      SyncCoverResourceActionKind::Consume, domain.targetResource,
      {SyncCoverAnchorKind::BeforeNode, demand.target, 0});
  actions.push_back(consume);
  const SyncCoverDescriptorActionRef produce = builder.addAction(
      SyncCoverResourceActionKind::Produce, domain.sourceResource,
      {SyncCoverAnchorKind::AfterNode, demand.source, 0});
  actions.push_back(produce);
  for (unsigned lane = 0; lane < demand.distance; ++lane) {
    actions.push_back(builder.addAction(
        SyncCoverResourceActionKind::Consume, domain.targetResource,
        {SyncCoverAnchorKind::ScopeExit, 0, demand.scope}));
  }
  SyncCoverEdge edge;
  edge.source = demand.source;
  edge.target = demand.target;
  edge.kind = SyncCoverEdgeKind::CompletionSupply;
  edge.scope = demand.scope;
  edge.distance = demand.distance;
  if (!builder.addProtocolLane(domain, demand.scope, demand.distance,
                               demand.distance, actions,
                               {{edge, produce, consume}})) {
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
        report.skippedByMissingDomain += entry.second.size();
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
          report.admittedIds.push_back(*added.index);
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
        report.skippedByMissingDomain += entry.second.size();
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
          ++report.skippedByContext;
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
          report.admittedIds.push_back(*added.index);
        } else {
          ++report.rejectedByVerifier;
        }
      }
    }
    return report;
  }
};

class UnitRecurrenceEventGenerator final
    : public SyncCoverColumnGenerator {
public:
  const char *name() const override { return "unit-recurrence"; }

  SyncCoverColumnGeneratorReport
  generate(const SyncCoverColumnGenerationContext &context,
           SyncCoverMechanismUniverse &universe) const override {
    SyncCoverColumnGeneratorReport report;
    report.generator = name();
    const SyncCoverGraph &graph = universe.getGraph();
    const auto demands =
        collectRecurrenceDemands(graph, context, report, 1, 1);
    if (!demands) {
      return report;
    }
    for (const RecurrenceDemand &demand : *demands) {
      if (hasExactMechanismSupply(graph, demand)) {
        ++report.skippedByContext;
        continue;
      }
      if (!chargeCandidate(report, context.options, 1)) {
        return report;
      }
      const SyncCoverNode &source = graph.getNodes()[demand.source];
      const SyncCoverNode &target = graph.getNodes()[demand.target];
      const SyncCoverMechanismResult domainResult =
          universe.addResourceDomain(SyncCoverResourceKind::EventId,
                                     source.resource, target.resource,
                                     context.target.eventIdBudget);
      if (!domainResult || !domainResult.index) {
        ++report.rejectedByVerifier;
        continue;
      }
      const SyncCoverResourceDomain &domain =
          universe.getResourceDomains()[*domainResult.index];
      const auto descriptor = makeSyncCoverUnitRecurrenceEvent(
          domain, demand.source, demand.target, demand.scope,
          makeProviderIdentity(kUnitRecurrenceTag, demand.demand));
      const SyncCoverMechanismResult added =
          descriptor
              ? universe.addVerifiedProtocol(
                    *descriptor,
                    [&](const SyncCoverMechanismDescriptor &actual) {
                      return verifySyncCoverUnitRecurrenceEvent(universe,
                                                                actual);
                    })
              : SyncCoverMechanismResult{
                    SyncCoverMechanismError::InvalidMechanism, std::nullopt};
      if (added && added.index) {
        ++report.admitted;
        report.admittedIds.push_back(*added.index);
      } else {
        ++report.rejectedByVerifier;
      }
    }
    return report;
  }
};

class RingReleaseEventGenerator final : public SyncCoverColumnGenerator {
public:
  const char *name() const override { return "ring-release"; }

  SyncCoverColumnGeneratorReport
  generate(const SyncCoverColumnGenerationContext &context,
           SyncCoverMechanismUniverse &universe) const override {
    SyncCoverColumnGeneratorReport report;
    report.generator = name();
    const SyncCoverGraph &graph = universe.getGraph();
    const auto demands = collectRecurrenceDemands(
        graph, context, report, 2, context.target.eventIdBudget);
    if (!demands) {
      return report;
    }
    for (const RecurrenceDemand &demand : *demands) {
      if (hasExactMechanismSupply(graph, demand)) {
        ++report.skippedByContext;
        continue;
      }
      if (!chargeCandidate(report, context.options, 1)) {
        return report;
      }
      const SyncCoverNode &source = graph.getNodes()[demand.source];
      const SyncCoverNode &target = graph.getNodes()[demand.target];
      const SyncCoverMechanismResult domainResult =
          universe.addResourceDomain(SyncCoverResourceKind::EventId,
                                     source.resource, target.resource,
                                     context.target.eventIdBudget);
      if (!domainResult || !domainResult.index) {
        ++report.rejectedByVerifier;
        continue;
      }
      const SyncCoverResourceDomain &domain =
          universe.getResourceDomains()[*domainResult.index];
      const auto descriptor = makeRingDescriptor(
          domain, demand,
          makeProviderIdentity(kRingTag, demand.demand));
      const SyncCoverMechanismResult added =
          descriptor
              ? universe.addVerifiedProtocol(
                    *descriptor,
                    [&](const SyncCoverMechanismDescriptor &actual) {
                      return verifySyncCoverRingReleaseProtocol(graph, domain,
                                                                actual);
                    })
              : SyncCoverMechanismResult{
                    SyncCoverMechanismError::InvalidMechanism, std::nullopt};
      if (added && added.index) {
        ++report.admitted;
        report.admittedIds.push_back(*added.index);
      } else {
        ++report.rejectedByVerifier;
      }
    }
    return report;
  }
};

class BufferTokenRingGenerator final : public SyncCoverColumnGenerator {
public:
  const char *name() const override { return "token-ring"; }

  SyncCoverColumnGeneratorReport
  generate(const SyncCoverColumnGenerationContext &context,
           SyncCoverMechanismUniverse &universe) const override {
    SyncCoverColumnGeneratorReport report;
    report.generator = name();
    const SyncCoverGraph &graph = universe.getGraph();
    constexpr unsigned kMaximumReportedDistance = 32;
    const auto demands = collectRecurrenceDemands(
        graph, context, report, 1, kMaximumReportedDistance);
    if (!demands) {
      return report;
    }
    if (!context.target.supportsBufferTokens ||
        context.target.bufferTokenBudget == 0) {
      report.skippedByCapability = demands->size();
      return report;
    }
    for (const RecurrenceDemand &demand : *demands) {
      if (demand.distance > context.target.bufferTokenBudget) {
        ++report.skippedByCapability;
        continue;
      }
      if (!chargeCandidate(report, context.options, 1)) {
        return report;
      }
      const SyncCoverNode &source = graph.getNodes()[demand.source];
      const SyncCoverNode &target = graph.getNodes()[demand.target];
      const std::uint64_t poolIdentity = demand.demand + 1;
      const SyncCoverMechanismResult domainResult = universe.addResourceDomain(
          SyncCoverResourceKind::BufferToken, source.resource,
          target.resource, context.target.bufferTokenBudget, poolIdentity);
      if (!domainResult || !domainResult.index) {
        ++report.rejectedByVerifier;
        continue;
      }
      const SyncCoverResourceDomain &domain =
          universe.getResourceDomains()[*domainResult.index];
      const auto descriptor = makeRingDescriptor(
          domain, demand,
          makeProviderIdentity(kTokenRingTag, demand.demand));
      const SyncCoverMechanismResult added =
          descriptor
              ? universe.addVerifiedProtocol(
                    *descriptor,
                    [&](const SyncCoverMechanismDescriptor &actual) {
                      return verifySyncCoverRingReleaseProtocol(graph, domain,
                                                                actual);
                    })
              : SyncCoverMechanismResult{
                    SyncCoverMechanismError::InvalidMechanism, std::nullopt};
      if (added && added.index) {
        ++report.admitted;
        report.admittedIds.push_back(*added.index);
      } else {
        ++report.rejectedByVerifier;
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
        ++report.skippedByContext;
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
          report.admittedIds.push_back(*added.index);
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

bool mlir::pto::verifySyncCoverRingReleaseProtocol(
    const SyncCoverGraph &graph, const SyncCoverResourceDomain &domain,
    const SyncCoverMechanismDescriptor &descriptor) {
  if (descriptor.kind != SyncCoverMechanismKind::VerifiedProtocol ||
      descriptor.supplyEdges.size() != 1 ||
      descriptor.resourceUses.size() != 1) {
    return false;
  }
  const SyncCoverEdge &edge = descriptor.supplyEdges.front();
  const SyncCoverResourceUse &use = descriptor.resourceUses.front();
  const auto &nodes = graph.getNodes();
  const auto &scopes = graph.getScopes();
  const bool validLane =
      use.distance != 0 && use.width == use.distance &&
      edge.distance == use.distance && edge.scope == use.scope &&
      use.domain == domain.id && use.scope < scopes.size() &&
      scopes[use.scope].isLoop && scopes[use.scope].timeline &&
      edge.kind == SyncCoverEdgeKind::CompletionSupply &&
      edge.source < nodes.size() && edge.target < nodes.size();
  if (!validLane) {
    return false;
  }
  const SyncCoverNode &source = nodes[edge.source];
  const SyncCoverNode &target = nodes[edge.target];
  const bool validEndpoints =
      source.resource == domain.sourceResource &&
      target.resource == domain.targetResource && hasEmptyGuard(source.guard) &&
      hasEmptyGuard(target.guard) && hasEmptyGuard(edge.sourceGuard) &&
      hasEmptyGuard(edge.targetGuard) &&
      graph.scopeMustExecuteWithin(use.scope, source.scope) &&
      graph.scopeMustExecuteWithin(use.scope, target.scope);
  if (!validEndpoints) {
    return false;
  }
  if (domain.kind == SyncCoverResourceKind::EventId &&
      !syncCoverNodeCanProduceCompletion(graph, edge.source,
                                         domain.targetResource)) {
    return false;
  }
  if (domain.kind != SyncCoverResourceKind::EventId &&
      domain.kind != SyncCoverResourceKind::BufferToken) {
    return false;
  }

  const unsigned distance = use.distance;
  if (descriptor.actions.size() !=
      2 * static_cast<std::size_t>(distance) + 2) {
    return false;
  }
  std::size_t primes = 0;
  std::size_t drains = 0;
  std::size_t bodyProduces = 0;
  std::size_t bodyConsumes = 0;
  for (const SyncCoverResourceAction &action : descriptor.actions) {
    const bool produce =
        action.kind == SyncCoverResourceActionKind::Produce;
    if (action.resource !=
        (produce ? domain.sourceResource : domain.targetResource)) {
      return false;
    }
    switch (action.anchor.kind) {
    case SyncCoverAnchorKind::ScopeEntry:
      if (!produce || action.anchor.scope != use.scope) {
        return false;
      }
      ++primes;
      break;
    case SyncCoverAnchorKind::ScopeExit:
      if (produce || action.anchor.scope != use.scope) {
        return false;
      }
      ++drains;
      break;
    case SyncCoverAnchorKind::AfterNode:
      if (!produce || action.anchor.node != edge.source) {
        return false;
      }
      ++bodyProduces;
      break;
    case SyncCoverAnchorKind::BeforeNode:
      if (produce || action.anchor.node != edge.target) {
        return false;
      }
      ++bodyConsumes;
      break;
    case SyncCoverAnchorKind::TimelinePoint:
      return false;
    }
  }
  return primes == distance && drains == distance && bodyProduces == 1 &&
         bodyConsumes == 1;
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

std::vector<std::unique_ptr<SyncCoverColumnGenerator>>
mlir::pto::makeDefaultSyncCoverColumnGenerators() {
  std::vector<std::unique_ptr<SyncCoverColumnGenerator>> generators;
  generators.push_back(makeSyncCoverCanonicalEventGenerator());
  generators.push_back(makeSyncCoverMergedPrefixEventGenerator());
  generators.push_back(makeSyncCoverUnitRecurrenceEventGenerator());
  generators.push_back(makeSyncCoverRingReleaseEventGenerator());
  generators.push_back(makeSyncCoverBufferTokenRingGenerator());
  generators.push_back(makeSyncCoverPiercedBarrierGenerator());
  return generators;
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
mlir::pto::makeSyncCoverUnitRecurrenceEventGenerator() {
  return std::make_unique<UnitRecurrenceEventGenerator>();
}

std::unique_ptr<SyncCoverColumnGenerator>
mlir::pto::makeSyncCoverRingReleaseEventGenerator() {
  return std::make_unique<RingReleaseEventGenerator>();
}

std::unique_ptr<SyncCoverColumnGenerator>
mlir::pto::makeSyncCoverBufferTokenRingGenerator() {
  return std::make_unique<BufferTokenRingGenerator>();
}

std::unique_ptr<SyncCoverColumnGenerator>
mlir::pto::makeSyncCoverPiercedBarrierGenerator() {
  return std::make_unique<PiercedBarrierGenerator>();
}
