// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- DescriptorState.cpp - Constant bounded tile metadata flow -----------===//

#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

std::optional<std::uint64_t> constantDimension(Value value, std::int64_t fallback)
{
    if (!value) {
        return fallback >= 0 ? std::optional<std::uint64_t>(fallback) : std::nullopt;
    }
    IntegerAttr attribute;
    const bool known = matchPattern(value, m_Constant(&attribute));
    const bool invalid = !known || attribute.getValue().getBitWidth() > 64 || attribute.getValue().isNegative();
    if (invalid) {
        return std::nullopt;
    }
    return attribute.getValue().getZExtValue();
}

bool supportedAllocation(Value handle)
{
    if (!handle) {
        return false;
    }
    auto allocation = handle.getDefiningOp<AllocTileOp>();
    if (!allocation) {
        return false;
    }
    auto type = allocation.getResult().getType();
    // Descriptor dimensions describe the handle, not its physical footprint.
    // Cube layouts and unaddressed handles have the same metadata contract.
    return type.getRank() == 2 && type.getShape()[0] >= 0 && type.getShape()[1] >= 0;
}

bool isEntryDefinition(Operation* operation)
{
    if (!operation) {
        return false;
    }
    auto function = dyn_cast_or_null<func::FuncOp>(operation->getParentOp());
    return function && operation->getBlock() == &function.getBody().front();
}

bool isStructuredRead(Operation* operation, Operation* allocation)
{
    for (Operation* parent = operation->getParentOp(); parent; parent = parent->getParentOp()) {
        if (parent == allocation->getParentOp()) {
            return true;
        }
        if (!isa<scf::ForOp, scf::IfOp>(parent)) {
            return false;
        }
    }
    return false;
}

} // namespace

namespace mlir::pto::protocol_sync {

class SyncDescriptorStateBuilder {
public:
    explicit SyncDescriptorStateBuilder(StructuredSyncIR& schedule) : schedule(schedule) {}
    LogicalResult build();

private:
    void reject(Operation* operation, StringRef detail);
    bool validateHandle(const SyncSemanticAction& action, Value handle);
    void transfer(SyncSemanticAction& action);

    StructuredSyncIR& schedule;
    llvm::DenseMap<Value, std::uint32_t> current;
    llvm::DenseSet<Value> invalid;
    llvm::DenseSet<Value> validated;
    llvm::DenseMap<Operation*, unsigned> payloadOperations;
};

void SyncDescriptorStateBuilder::reject(Operation* operation, StringRef detail)
{
    schedule.failures.push_back({SyncFailureReason::UnsupportedDescriptorState, operation, detail.str()});
}

bool SyncDescriptorStateBuilder::validateHandle(const SyncSemanticAction& action, Value handle)
{
    if (invalid.contains(handle)) {
        return false;
    }
    const bool supported = handle && (validated.contains(handle) || supportedAllocation(handle));
    if (!supported) {
        reject(action.operation, "descriptor requires a direct tile allocation with static rank-two bounds");
        return false;
    }
    Operation* allocation = handle.getDefiningOp();
    if (!isEntryDefinition(allocation)) {
        reject(action.operation, "descriptor allocation must execute once in the function entry block");
        return false;
    }
    if (validated.contains(handle)) {
        return true;
    }
    for (Operation* user : handle.getUsers()) {
        const bool metadata = isa<SetValidShapeOp, GetValidShapeOp>(user);
        const bool payload = payloadOperations.lookup(user) == 1 && user->getNumResults() == 0;
        const bool nestedUpdate = isa<SetValidShapeOp>(user) && !isEntryDefinition(user);
        if (nestedUpdate) {
            reject(user, "descriptor updates in choices or loops require version merges");
            return false;
        }
        const bool unsupportedUser = !isStructuredRead(user, allocation) || (!metadata && !payload);
        if (unsupportedUser) {
            reject(user, "descriptor forwarding, aliases, escapes or unsupported regions are unsupported");
            return false;
        }
    }
    validated.insert(handle);
    return true;
}

void SyncDescriptorStateBuilder::transfer(SyncSemanticAction& action)
{
    const auto& effect = schedule.summaries[action.summary].descriptor;
    if (!effect || invalid.contains(effect->handle)) {
        return;
    }
    auto found = current.find(effect->handle);
    if (effect->role == SyncDescriptorRole::Read) {
        if (found == current.end()) {
            reject(action.operation, "descriptor read has no supported incoming version");
            invalid.insert(effect->handle);
        } else {
            action.descriptorState = found->second;
        }
        return;
    }
    const auto type = cast<TileBufType>(effect->handle.getType());
    const auto rows = constantDimension(effect->rows, type.getValidShape()[0]);
    const auto columns = constantDimension(effect->columns, type.getValidShape()[1]);
    const bool missingInput = effect->role == SyncDescriptorRole::Update && found == current.end();
    const bool unsupportedBounds = !rows || !columns || *rows > static_cast<std::uint64_t>(type.getShape()[0]) ||
                                   *columns > static_cast<std::uint64_t>(type.getShape()[1]);
    if (missingInput || unsupportedBounds) {
        reject(action.operation, "descriptor dimensions need bounded constants and supported initialization");
        invalid.insert(effect->handle);
        current.erase(effect->handle);
        return;
    }
    const std::uint32_t id = schedule.descriptorStates.size();
    schedule.descriptorStates.push_back(
        {id, effect->handle, action.id, *rows, *columns, effect->rows, effect->columns});
    current[effect->handle] = id;
    action.descriptorState = id;
}

LogicalResult SyncDescriptorStateBuilder::build()
{
    if (schedule.frozen || !schedule.descriptorStates.empty()) {
        return failure();
    }
    for (const SyncPhase& phase : schedule.phases) {
        if (schedule.summaries[phase.summary].provider == SyncSummaryProvider::Pipeline) {
            ++payloadOperations[phase.operation];
        }
    }
    for (const SyncSemanticAction& action : schedule.semanticActions) {
        const auto& descriptor = schedule.summaries[action.summary].descriptor;
        if (descriptor && !validateHandle(action, descriptor->handle)) {
            invalid.insert(descriptor->handle);
        }
    }
    // Definitions execute once in the entry block. Nested reads preserve the
    // incoming version, including zero-trip loops and either choice arm.
    for (const SyncProgramPoint& point : schedule.points) {
        if (point.kind == SyncProgramPointKind::SemanticActionBefore) {
            transfer(schedule.semanticActions[point.action]);
        } else if (point.kind == SyncProgramPointKind::PhaseBefore) {
            for (SyncAccessId id : schedule.phases[point.phase].accesses) {
                SyncAccess& access = schedule.accesses[id];
                auto found = current.find(access.value);
                const bool supported = found != current.end() && !invalid.contains(access.value);
                if (supported) {
                    access.descriptorState = found->second;
                }
            }
        }
    }
    return success();
}

LogicalResult buildSyncDescriptorStates(StructuredSyncIR& schedule)
{
    return SyncDescriptorStateBuilder(schedule).build();
}

LogicalResult verifySyncDescriptorBindings(const StructuredSyncIR& schedule)
{
    if (!schedule.getFailures().empty()) {
        return failure();
    }
    llvm::DenseMap<Value, std::uint32_t> latest;
    unsigned definitions = 0;
    const auto states = schedule.getDescriptorStates();
    DominanceInfo dominance(schedule.getFunction());
    // Validate the live IR, not cached program-point order or guard metadata.
    // In particular, moving an update into a loop cannot preserve a certificate.
    for (const SyncDescriptorState& state : states) {
        if (!state.handle) {
            return failure();
        }
        const auto* definition = schedule.findSemanticAction(state.definition);
        Operation* allocation = state.handle.getDefiningOp();
        const bool invalidDefinition = !definition || !supportedAllocation(state.handle) ||
                                       !isEntryDefinition(allocation) || !isEntryDefinition(definition->operation);
        if (invalidDefinition) {
            return failure();
        }
        for (Operation* user : state.handle.getUsers()) {
            const bool misplacedUpdate = isa<SetValidShapeOp>(user) && !isEntryDefinition(user);
            if (misplacedUpdate || !isStructuredRead(user, allocation)) {
                return failure();
            }
        }
    }
    auto observesLatest = [&](Value handle, std::uint32_t id, Operation* use) {
        const SyncDescriptorState* selected = nullptr;
        Operation* selectedDefinition = nullptr;
        for (const SyncDescriptorState& candidate : states) {
            if (candidate.handle != handle) {
                continue;
            }
            Operation* definition = schedule.findSemanticAction(candidate.definition)->operation;
            if (!dominance.properlyDominates(definition, use)) {
                continue;
            }
            if (!selectedDefinition || selectedDefinition->isBeforeInBlock(definition)) {
                selected = &candidate;
                selectedDefinition = definition;
            }
        }
        return selected && selected->id == id;
    };
    for (const SyncProgramPoint& point : schedule.getProgramPoints()) {
        if (point.kind == SyncProgramPointKind::SemanticActionBefore) {
            const SyncSemanticAction& action = *schedule.findSemanticAction(point.action);
            const SyncOpSummary& summary = schedule.getSummaries()[action.summary];
            const auto& effect = summary.descriptor;
            const bool wrongProvider = effect && summary.provider != SyncSummaryProvider::Descriptor;
            if (wrongProvider) {
                return failure();
            }
            if (!effect) {
                if (action.descriptorState) {
                    return failure();
                }
                continue;
            }
            if (!action.descriptorState || *action.descriptorState >= states.size()) {
                return failure();
            }
            const SyncDescriptorState& state = states[*action.descriptorState];
            if (state.id != *action.descriptorState || state.handle != effect->handle) {
                return failure();
            }
            if (effect->role == SyncDescriptorRole::Read) {
                auto found = latest.find(effect->handle);
                const bool wrongVersion = found == latest.end() || found->second != state.id;
                if (wrongVersion || !observesLatest(effect->handle, state.id, action.operation)) {
                    return failure();
                }
            } else {
                const auto type = cast<TileBufType>(effect->handle.getType());
                const auto rows = constantDimension(effect->rows, type.getValidShape()[0]);
                const auto columns = constantDimension(effect->columns, type.getValidShape()[1]);
                const bool wrongBounds = !rows || !columns || *rows > static_cast<std::uint64_t>(type.getShape()[0]) ||
                                         *columns > static_cast<std::uint64_t>(type.getShape()[1]);
                const bool wrongPredecessor = effect->role == SyncDescriptorRole::Initialize ?
                                                  latest.contains(effect->handle) :
                                                  !latest.contains(effect->handle);
                if (state.definition != action.id || wrongBounds || wrongPredecessor || state.rows != *rows ||
                    state.columns != *columns || state.rowSource != effect->rows ||
                    state.columnSource != effect->columns) {
                    return failure();
                }
                latest[effect->handle] = state.id;
                ++definitions;
            }
        } else if (point.kind == SyncProgramPointKind::PhaseBefore) {
            for (SyncAccessId id : schedule.findPhase(point.phase)->accesses) {
                const SyncAccess& access = *schedule.findAccess(id);
                auto found = latest.find(access.value);
                const bool wrongVersion = found == latest.end() ?
                                              access.descriptorState.has_value() :
                                              access.descriptorState != std::optional<std::uint32_t>(found->second);
                if (wrongVersion) {
                    return failure();
                }
                if (access.descriptorState &&
                    !observesLatest(
                        access.value, *access.descriptorState, schedule.findPhase(point.phase)->operation)) {
                    return failure();
                }
            }
        }
    }
    return success(definitions == states.size());
}

} // namespace mlir::pto::protocol_sync
