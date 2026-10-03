// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "FixedBodyUpper.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/SyncStorageBounds.h"
#include "llvm/ADT/DenseMap.h"
#include <limits>
namespace mlir::pto::frontiersynch {
using Int = llvm::DynamicAPInt;
FailureOr<std::shared_ptr<const FixedBodyUpper>> FixedBodyUpper::build(
    const SyncInput& input, ArrayRef<const CompoundInstanceElement*> phases, CostLedger& costs, std::string& reason)
{
    auto result = std::shared_ptr<FixedBodyUpper>(new FixedBodyUpper());
    if (phases.size() > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) || phases.empty()) {
        reason = "fixed-body upper requires a nonempty representable phase list";
        return failure();
    }
    Operation* loop = phases.front()->elementOp->getParentOp();
    auto invariant = [&](Value value) {
        if (!value) {
            return false;
        }
        Operation* owner = value.getDefiningOp();
        if (auto argument = dyn_cast<BlockArgument>(value)) {
            owner = argument.getOwner()->getParentOp();
        }
        return owner && owner != loop && !loop->isProperAncestor(owner);
    };
    for (auto* phase : phases) {
        if (phase->elementOp->getParentOp() != loop) {
            reason = "fixed-body origins do not share an original execution body";
            return failure();
        }
        auto qualified = [&](ArrayRef<const BaseMemInfo*> memories) {
            return llvm::all_of(memories, [&](const BaseMemInfo* memory) {
                // Unknown ranges are a shared universal may-alias fact in the
                // fixed address space. Known offsets may exclude an origin
                // only when the original base and root are iteration invariant.
                return memory->aliasesUnknownRange ||
                       (memory->scope == AddressSpace::GM && memory->baseAddresses.empty()) ||
                       (invariant(memory->baseBuffer) && invariant(memory->rootBuffer));
            });
        };
        if (!qualified(phase->useVec) || !qualified(phase->defVec)) {
            reason = "fixed-body source-filtered exclusion lacks occurrence-invariant shared geometry";
            return failure();
        }
    }
    if (failed(result->recover(input, phases, costs, reason))) {
        return failure();
    }
    {
        CostScope backend(costs, CostStage::Backend);
        if (failed(result->reduce(phases, reason)) || failed(result->summarize(phases, reason))) {
            return failure();
        }
    }
    return std::shared_ptr<const FixedBodyUpper>(result);
}
LogicalResult FixedBodyUpper::recover(
    const SyncInput& input, ArrayRef<const CompoundInstanceElement*> phases, CostLedger& costs, std::string& reason)
{
    DenseMap<PipelineType, std::size_t> columns;
    for (auto* phase : phases) {
        columns.try_emplace(phase->kPipeValue, columns.size());
    }
    if (columns.size() > std::numeric_limits<std::size_t>::max() / phases.size()) {
        reason = "fixed-body choice table size is not representable";
        return failure();
    }
    SmallVector<std::optional<PeriodicPrerequisite>> latest(phases.size() * columns.size());
    {
        CostScope effects(costs, CostStage::Effects);
        // Bounding cells never certify a whole-class kill. The currently
        // recovered model therefore retains every origin through the circular
        // body. Its shortest distance is 0/1 and its maximum is infinite. These
        // are exact distances in that no-kill control abstraction, not sampled
        // storage lifetimes. Alias filtering remains source-specific.
        for (auto [a, source] : llvm::enumerate(phases)) {
            if (source->macroOpInstanceId >= 0) {
                reason = "fixed-body upper macro prerequisite association is not qualified";
                return failure();
            }
            for (auto [b, consumer] : llvm::enumerate(phases)) {
                auto observe = [&](ArrayRef<const BaseMemInfo*> from, ArrayRef<const BaseMemInfo*> to,
                                   UpperHazard hazard) {
                    for (auto* first : from) {
                        for (auto* second : to) {
                            ++queryComparisons;
                            DepBaseMemInfoPairVec witnesses;
                            if (!input.memory().DepBetween({first}, {second}, witnesses)) {
                                continue;
                            }
                            auto origin = input.storageBounds().memoryIdentity(first);
                            auto target = input.storageBounds().memoryIdentity(second);
                            if (!origin || !target) {
                                return failure();
                            }
                            Int distance(a < b ? 0 : 1);
                            origins.push_back({a, b, *origin, *target, hazard, distance, std::nullopt});
                            auto& found = latest[b * columns.size() + columns.lookup(source->kPipeValue)];
                            if (!found || distance < found->distance ||
                                (distance == found->distance && a > found->source)) {
                                found = PeriodicPrerequisite{a, b, distance};
                            }
                        }
                    }
                    return success();
                };
                if (failed(observe(source->defVec, consumer->useVec, UpperHazard::RAW)) ||
                    failed(observe(source->useVec, consumer->defVec, UpperHazard::WAR)) ||
                    failed(observe(source->defVec, consumer->defVec, UpperHazard::WAW))) {
                    reason = "fixed-body origin lacks immutable shared memory identity";
                    return failure();
                }
            }
        }
    }
    for (const auto& entry : latest) {
        if (entry) {
            candidates.push_back(*entry);
        }
    }
    return success();
}
LogicalResult FixedBodyUpper::reduce(ArrayRef<const CompoundInstanceElement*> phases, std::string& reason)
{
    PeriodicDemandReduction first;
    if (failed(first.build(phases, candidates))) {
        reason = "fixed-body consolidated quotient reduction failed";
        return failure();
    }
    DenseMap<PipelineType, std::size_t> last;
    for (auto [id, phase] : llvm::enumerate(phases)) {
        last[phase->kPipeValue] = id;
    }
    SmallVector<std::size_t> previous(phases.size());
    for (auto [id, phase] : llvm::enumerate(phases)) {
        previous[id] = last.lookup(phase->kPipeValue);
        last[phase->kPipeValue] = id;
    }
    SmallVector<PeriodicPrerequisite> adjacent;
    for (auto id : first.retained()) {
        auto edge = first.generators()[id].edge;
        if (phases[edge.source]->kPipeValue == phases[edge.consumer]->kPipeValue) {
            edge.source = previous[edge.consumer];
            edge.distance = Int(edge.source < edge.consumer ? 0 : 1);
        }
        adjacent.push_back(std::move(edge));
    }
    if (failed(upper.build(phases, adjacent))) {
        reason = "fixed-body adjacent upper quotient reduction failed";
        return failure();
    }
    return success();
}
LogicalResult FixedBodyUpper::summarize(ArrayRef<const CompoundInstanceElement*> phases, std::string& reason)
{
    // Genuine lower information is only native here: it has no C->I edges.
    // Each selected S prefix is max(0,count*j+offset), yielding a closed-form
    // upper bound on excess above the original graph, without trip expansion.
    for (std::size_t b = 0; b < phases.size(); ++b) {
        for (std::size_t pipe = 0; pipe < upper.pipes().size(); ++pipe) {
            const auto count = upper.pipePayloadCounts()[pipe];
            auto queried = upper.frontierOffset(pipe, b, PeriodicEventKind::Start);
            if (failed(queried)) {
                reason = "fixed-body frontier query failed";
                return failure();
            }
            const auto& offset = *queried;
            if (!offset) {
                continue;
            }
            Int scale(static_cast<int64_t>(count));
            Int startup(0);
            if (*offset < Int(1)) {
                startup = (Int(1) - *offset + scale - Int(1)) / scale;
            }
            terms.push_back({b, pipe, scale, *offset, startup});
            // Start queries have positive-length support; native C identity
            // never creates a diagonal support entry. Real feedback does.
            gamma += scale;
        }
    }
    return success();
}
Int FixedBodyUpper::excessBound(const Int& trips) const
{
    if (trips <= Int(0)) {
        return Int(0);
    }
    Int result(0);
    for (const auto& term : terms) {
        if (trips <= term.startup) {
            continue;
        }
        Int length = trips - term.startup;
        Int sum = (term.startup + trips - Int(1)) * length / Int(2);
        result += term.count * sum + term.offset * length;
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
