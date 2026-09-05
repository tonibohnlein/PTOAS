// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- EventAllocation.cpp - Shared event lifetime allocation ----------===//

#include "PTO/Transforms/ProtocolSync/EventAllocation.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

using EventDomain = std::tuple<std::uint8_t, std::uint8_t, std::uint8_t>;

EventDomain eventDomain(const SyncEventGeneration& generation)
{
    return {
        static_cast<std::uint8_t>(generation.core), static_cast<std::uint8_t>(generation.sourcePipe),
        static_cast<std::uint8_t>(generation.targetPipe)};
}

bool controlsAreMutuallyExclusive(ArrayRef<SyncControlAtom> first, ArrayRef<SyncControlAtom> second)
{
    return llvm::any_of(first, [&](const SyncControlAtom& left) {
        return llvm::any_of(
            second, [&](const SyncControlAtom& right) { return left.choice == right.choice && left.arm != right.arm; });
    });
}

bool generationsInterfere(const SyncEventGeneration& first, const SyncEventGeneration& second)
{
    const bool differentDomain = eventDomain(first) != eventDomain(second);
    if (differentDomain) {
        return false;
    }
    if (first.recurring || second.recurring) {
        return true;
    }
    if (controlsAreMutuallyExclusive(first.guard, second.guard)) {
        return false;
    }
    // Lexical wait-before-set order does not prove that an asynchronous target
    // pipeline consumed the first event before the source issues the next set.
    // Reuse between coexecuting generations needs a future certified target
    // happens-before relation; until then they conservatively interfere.
    return true;
}

bool isReserved(ArrayRef<SyncEventReservation> reservations, const SyncEventGeneration& generation, unsigned eventId)
{
    return llvm::any_of(reservations, [&](const SyncEventReservation& reservation) {
        return reservation.source == generation.sourcePipe && reservation.target == generation.targetPipe &&
               llvm::is_contained(reservation.eventIds, eventId);
    });
}

bool hasValidShape(const ProtocolSyncTarget& target, ArrayRef<SyncEventGeneration> generations)
{
    for (auto [index, generation] : llvm::enumerate(generations)) {
        const bool validDomain =
            generation.id == index && generation.core != SyncPhysicalCore::Unknown &&
            generation.sourcePipe != PIPE::PIPE_UNASSIGNED && generation.targetPipe != PIPE::PIPE_UNASSIGNED &&
            generation.sourcePipe != generation.targetPipe &&
            target.supportsEvent({generation.core, generation.sourcePipe}, {generation.core, generation.targetPipe});
        if (!validDomain) {
            return false;
        }
    }
    return true;
}

class DomainColoring {
public:
    DomainColoring(
        ArrayRef<unsigned> colors, ArrayRef<llvm::BitVector> adjacency, std::uint64_t maximumNodes,
        std::uint64_t& exploredNodes)
        : colors(colors),
          adjacency(adjacency),
          maximumNodes(maximumNodes),
          exploredNodes(exploredNodes),
          assigned(adjacency.size(), kUnassigned),
          degrees(adjacency.size()),
          saturation(adjacency.size()),
          neighborColorCounts(adjacency.size(), llvm::SmallVector<unsigned, 6>(colors.size(), 0))
    {
        for (unsigned vertex = 0; vertex < adjacency.size(); ++vertex) {
            degrees[vertex] = adjacency[vertex].count();
        }
    }

    bool solveGreedy()
    {
        for (unsigned colored = 0; colored < adjacency.size(); ++colored) {
            const unsigned vertex = selectVertex();
            bool assignedColor = false;
            for (unsigned color = 0; color < colors.size(); ++color) {
                if (neighborColorCounts[vertex][color] != 0) {
                    continue;
                }
                assign(vertex, color);
                assignedColor = true;
                break;
            }
            if (!assignedColor) {
                clear();
                return false;
            }
        }
        return true;
    }

    bool solveExact()
    {
        clear();
        return search(0);
    }

    unsigned getColor(unsigned vertex) const { return colors[assigned[vertex]]; }
    bool exhaustedBudget() const { return budgetExhausted; }
    unsigned usedColorCount() const
    {
        llvm::SmallVector<bool, 6> used(colors.size(), false);
        for (unsigned color : assigned) {
            if (color != kUnassigned) {
                used[color] = true;
            }
        }
        return llvm::count(used, true);
    }

private:
    static constexpr unsigned kUnassigned = ~0U;

    unsigned selectVertex() const
    {
        unsigned selected = kUnassigned;
        unsigned selectedSaturation = 0;
        unsigned selectedDegree = 0;
        for (unsigned vertex = 0; vertex < adjacency.size(); ++vertex) {
            if (assigned[vertex] != kUnassigned) {
                continue;
            }
            const unsigned candidateSaturation = saturation[vertex];
            const unsigned candidateDegree = degrees[vertex];
            const bool preferable =
                selected == kUnassigned || candidateSaturation > selectedSaturation ||
                (candidateSaturation == selectedSaturation && candidateDegree > selectedDegree) ||
                (candidateSaturation == selectedSaturation && candidateDegree == selectedDegree && vertex < selected);
            if (preferable) {
                selected = vertex;
                selectedSaturation = candidateSaturation;
                selectedDegree = candidateDegree;
            }
        }
        return selected;
    }

    void assign(unsigned vertex, unsigned color)
    {
        assigned[vertex] = color;
        for (int neighbor = adjacency[vertex].find_first(); neighbor >= 0;
             neighbor = adjacency[vertex].find_next(neighbor)) {
            if (assigned[neighbor] != kUnassigned) {
                continue;
            }
            if (neighborColorCounts[neighbor][color]++ == 0) {
                ++saturation[neighbor];
            }
        }
    }

    void unassign(unsigned vertex, unsigned color)
    {
        for (int neighbor = adjacency[vertex].find_first(); neighbor >= 0;
             neighbor = adjacency[vertex].find_next(neighbor)) {
            if (assigned[neighbor] != kUnassigned) {
                continue;
            }
            if (--neighborColorCounts[neighbor][color] == 0) {
                --saturation[neighbor];
            }
        }
        assigned[vertex] = kUnassigned;
    }

    void clear()
    {
        std::fill(assigned.begin(), assigned.end(), kUnassigned);
        std::fill(saturation.begin(), saturation.end(), 0);
        for (auto& counts : neighborColorCounts) {
            std::fill(counts.begin(), counts.end(), 0);
        }
    }

    bool search(unsigned colored)
    {
        if (exploredNodes >= maximumNodes) {
            budgetExhausted = true;
            return false;
        }
        ++exploredNodes;
        if (colored == adjacency.size()) {
            return true;
        }
        const unsigned vertex = selectVertex();
        if (vertex == kUnassigned) {
            return false;
        }
        for (unsigned color = 0; color < colors.size(); ++color) {
            if (neighborColorCounts[vertex][color] != 0) {
                continue;
            }
            assign(vertex, color);
            if (search(colored + 1)) {
                return true;
            }
            unassign(vertex, color);
            if (budgetExhausted) {
                return false;
            }
        }
        return false;
    }

    ArrayRef<unsigned> colors;
    ArrayRef<llvm::BitVector> adjacency;
    std::uint64_t maximumNodes;
    std::uint64_t& exploredNodes;
    llvm::SmallVector<unsigned, 16> assigned;
    llvm::SmallVector<unsigned, 16> degrees;
    llvm::SmallVector<unsigned, 16> saturation;
    llvm::SmallVector<llvm::SmallVector<unsigned, 6>, 16> neighborColorCounts;
    bool budgetExhausted = false;
};

SmallVector<unsigned, 6> availableColors(
    const ProtocolSyncTarget& target, ArrayRef<SyncEventReservation> reservations,
    const SyncEventGeneration& representative)
{
    SmallVector<unsigned, 6> available;
    llvm::copy_if(target.getCompilerEventIds(), std::back_inserter(available), [&](unsigned eventId) {
        return !isReserved(reservations, representative, eventId);
    });
    return available;
}

} // namespace

FailureOr<SyncEventAllocationResult> mlir::pto::protocol_sync::allocateSyncEventGenerations(
    const ProtocolSyncTarget& target, ArrayRef<SyncEventReservation> reservations,
    ArrayRef<SyncEventGeneration> generations, const SyncEventAllocationOptions& options)
{
    const bool alreadyAllocated =
        llvm::any_of(generations, [](const SyncEventGeneration& generation) { return generation.eventId.has_value(); });
    const bool invalidInput = !target.isSupported() || !hasValidShape(target, generations) || alreadyAllocated;
    if (invalidInput) {
        return failure();
    }

    const bool invalidOptions =
        options.maximumBacktrackingNodes == 0 ||
        options.maximumBacktrackingNodes > kHardMaximumEventBacktrackingNodes || options.maximumExactVertices == 0 ||
        options.maximumExactVertices > kHardMaximumExactEventVertices || options.maximumGenerationsPerDomain == 0 ||
        options.maximumGenerationsPerDomain > kHardMaximumEventGenerationsPerDomain;
    if (invalidOptions) {
        return failure();
    }

    SyncEventAllocationResult result;
    result.graphVertices = generations.size();
    result.eventIds.assign(generations.size(), 0);
    std::map<EventDomain, SmallVector<unsigned, 8>> domainVertices;
    for (auto [index, generation] : llvm::enumerate(generations)) {
        domainVertices[eventDomain(generation)].push_back(index);
    }
    result.eventDomains = domainVertices.size();
    for (const auto& domainEntry : domainVertices) {
        const SmallVector<unsigned, 8>& vertices = domainEntry.second;
        const bool exceedsDomainLimit = vertices.size() > options.maximumGenerationsPerDomain;
        if (exceedsDomainLimit) {
            result.status = SyncEventAllocationStatus::AnalysisLimit;
            result.eventIds.clear();
            ++result.searchLimitHits;
            return result;
        }
        llvm::SmallVector<llvm::BitVector, 16> adjacency(
            vertices.size(), llvm::BitVector(vertices.size()));
        std::uint64_t domainEdges = 0;
        for (auto [firstPosition, first] : llvm::enumerate(vertices)) {
            for (auto [secondOffset, second] : llvm::enumerate(ArrayRef(vertices).drop_front(firstPosition + 1))) {
                if (!generationsInterfere(generations[first], generations[second])) {
                    continue;
                }
                const unsigned secondPosition = firstPosition + secondOffset + 1;
                adjacency[firstPosition].set(secondPosition);
                adjacency[secondPosition].set(firstPosition);
                ++domainEdges;
            }
        }
        result.graphEdges += domainEdges;
        const SmallVector<unsigned, 6> colors = availableColors(target, reservations, generations[vertices.front()]);
        if (colors.empty()) {
            result.status = SyncEventAllocationStatus::ResourceInfeasible;
            result.eventIds.clear();
            return result;
        }
        const std::uint64_t completeEdgeCount =
            static_cast<std::uint64_t>(vertices.size()) * (vertices.size() - 1) / 2;
        if (domainEdges == 0) {
            for (unsigned vertex : vertices) {
                result.eventIds[vertex] = colors.front();
            }
            result.maximumDomainPressure = std::max<std::uint64_t>(result.maximumDomainPressure, 1);
            continue;
        }
        if (domainEdges == completeEdgeCount) {
            const bool exceedsEventPool = vertices.size() > colors.size();
            if (exceedsEventPool) {
                result.status = SyncEventAllocationStatus::ResourceInfeasible;
                result.eventIds.clear();
                return result;
            }
            for (auto [position, vertex] : llvm::enumerate(vertices)) {
                result.eventIds[vertex] = colors[position];
            }
            result.maximumDomainPressure =
                std::max<std::uint64_t>(result.maximumDomainPressure, vertices.size());
            continue;
        }

        std::uint64_t feasibilityNodes = 0;
        DomainColoring feasible(colors, adjacency, options.maximumBacktrackingNodes, feasibilityNodes);
        bool solved = feasible.solveGreedy();
        const bool exactSearchAllowed = vertices.size() <= options.maximumExactVertices;
        if (!solved && exactSearchAllowed) {
            solved = feasible.solveExact();
        }
        result.backtrackingNodes += feasibilityNodes;
        if (!solved && feasible.exhaustedBudget()) {
            result.status = SyncEventAllocationStatus::AnalysisLimit;
            result.eventIds.clear();
            ++result.searchLimitHits;
            return result;
        }
        if (!solved && !exactSearchAllowed) {
            result.status = SyncEventAllocationStatus::AnalysisLimit;
            result.eventIds.clear();
            ++result.searchLimitHits;
            return result;
        }
        if (!solved) {
            result.status = SyncEventAllocationStatus::ResourceInfeasible;
            result.eventIds.clear();
            return result;
        }
        unsigned bestColorCount = feasible.usedColorCount();
        for (unsigned vertex = 0; vertex < vertices.size(); ++vertex) {
            result.eventIds[vertices[vertex]] = feasible.getColor(vertex);
        }

        if (!exactSearchAllowed && bestColorCount > 1) {
            ++result.searchLimitHits;
            result.maximumDomainPressure = std::max<std::uint64_t>(result.maximumDomainPressure, bestColorCount);
            continue;
        }
        std::uint64_t minimizationNodes = 0;
        for (unsigned colorCount = 1; colorCount < bestColorCount; ++colorCount) {
            DomainColoring candidate(
                ArrayRef<unsigned>(colors).take_front(colorCount), adjacency, options.maximumBacktrackingNodes,
                minimizationNodes);
            if (candidate.solveExact()) {
                bestColorCount = colorCount;
                for (unsigned vertex = 0; vertex < vertices.size(); ++vertex) {
                    result.eventIds[vertices[vertex]] = candidate.getColor(vertex);
                }
                break;
            }
            if (candidate.exhaustedBudget()) {
                ++result.searchLimitHits;
                break;
            }
        }
        result.backtrackingNodes += minimizationNodes;
        result.maximumDomainPressure = std::max<std::uint64_t>(result.maximumDomainPressure, bestColorCount);
    }
    for (unsigned eventId : result.eventIds) {
        result.maximumEventIdPlusOne = std::max<std::uint64_t>(result.maximumEventIdPlusOne, eventId + 1);
    }
    return result;
}

LogicalResult mlir::pto::protocol_sync::verifySyncEventGenerationAssignment(
    const ProtocolSyncTarget& target, ArrayRef<SyncEventReservation> reservations,
    ArrayRef<SyncEventGeneration> generations, bool allowUnallocated, const SyncEventAllocationOptions& options)
{
    const bool invalidOptions =
        options.maximumBacktrackingNodes == 0 ||
        options.maximumBacktrackingNodes > kHardMaximumEventBacktrackingNodes || options.maximumExactVertices == 0 ||
        options.maximumExactVertices > kHardMaximumExactEventVertices || options.maximumGenerationsPerDomain == 0 ||
        options.maximumGenerationsPerDomain > kHardMaximumEventGenerationsPerDomain;
    const bool invalidInput = !target.isSupported() || !hasValidShape(target, generations) || invalidOptions;
    if (invalidInput) {
        return failure();
    }
    if (generations.empty()) {
        return success();
    }
    const unsigned allocated = llvm::count_if(
        generations, [](const SyncEventGeneration& generation) { return generation.eventId.has_value(); });
    if (allocated == 0) {
        return success(allowUnallocated);
    }
    if (allocated != generations.size()) {
        return failure();
    }
    for (const SyncEventGeneration& generation : generations) {
        const unsigned eventId = *generation.eventId;
        const bool invalidId =
            !llvm::is_contained(target.getCompilerEventIds(), eventId) || isReserved(reservations, generation, eventId);
        if (invalidId) {
            return failure();
        }
    }
    std::map<EventDomain, SmallVector<unsigned, 8>> domainVertices;
    for (auto [index, generation] : llvm::enumerate(generations)) {
        domainVertices[eventDomain(generation)].push_back(index);
    }
    for (const auto& domainEntry : domainVertices) {
        const SmallVector<unsigned, 8>& vertices = domainEntry.second;
        const bool exceedsDomainLimit = vertices.size() > options.maximumGenerationsPerDomain;
        if (exceedsDomainLimit) {
            return failure();
        }
        for (auto [position, first] : llvm::enumerate(vertices)) {
            for (unsigned second : ArrayRef(vertices).drop_front(position + 1)) {
                const bool collides = generations[first].eventId == generations[second].eventId &&
                                      generationsInterfere(generations[first], generations[second]);
                if (collides) {
                    return failure();
                }
            }
        }
    }
    return success();
}

void mlir::pto::protocol_sync::recordSyncEventAllocationStatistics(
    const SyncEventAllocationResult& allocation, ProtocolSyncStatistics& statistics)
{
    statistics.allocationGraphVertices += allocation.graphVertices;
    statistics.allocationGraphEdges += allocation.graphEdges;
    statistics.allocationBacktrackingNodes += allocation.backtrackingNodes;
    statistics.allocationSearchLimitHits += allocation.searchLimitHits;
    statistics.eventDomains += allocation.eventDomains;
    statistics.maxEventDomainPressure = std::max(statistics.maxEventDomainPressure, allocation.maximumDomainPressure);
    statistics.maximumEventIdPlusOne = std::max(statistics.maximumEventIdPlusOne, allocation.maximumEventIdPlusOne);
}
