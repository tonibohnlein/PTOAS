// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- DescriptorState.cpp - Scope-local tile metadata provenance ----------===//

#include "PTO/Transforms/ProtocolSync/LocalMemoryAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

bool matchesLiveDescriptor(Operation* operation, const SyncDescriptorEffect& effect)
{
    if (auto allocation = dyn_cast<AllocTileOp>(operation)) {
        return effect.role == SyncDescriptorRole::Initialize && effect.handle == allocation.getResult() &&
               effect.rows == allocation.getValidRow() && effect.columns == allocation.getValidCol();
    }
    if (auto update = dyn_cast<SetValidShapeOp>(operation)) {
        return effect.role == SyncDescriptorRole::Update && effect.handle == update.getSource() &&
               effect.rows == update.getValidRow() && effect.columns == update.getValidCol();
    }
    if (auto read = dyn_cast<GetValidShapeOp>(operation)) {
        return effect.role == SyncDescriptorRole::Read && effect.handle == read.getSource() && !effect.rows &&
               !effect.columns;
    }
    return false;
}

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

bool nonphysicalScalar(Value value, Operation* use, DominanceInfo& dominance)
{
    SmallVector<std::pair<Value, Operation*>, 16> work{{value, use}};
    llvm::DenseSet<Value> visited;
    while (!work.empty()) {
        const auto [current, consumer] = work.pop_back_val();
        const bool unavailable =
            !current || !consumer || !current.getType().isIntOrIndex() || !dominance.dominates(current, consumer);
        if (unavailable) {
            return false;
        }
        if (!visited.insert(current).second) {
            continue;
        }
        constexpr unsigned MAX_SCALAR_PROVENANCE_VALUES = 256;
        const bool exhausted = visited.size() > MAX_SCALAR_PROVENANCE_VALUES;
        if (exhausted) {
            return false;
        }
        if (auto argument = dyn_cast<BlockArgument>(current)) {
            Operation* owner = argument.getOwner()->getParentOp();
            if (isa<func::FuncOp>(owner)) {
                continue;
            }
            auto loop = dyn_cast<scf::ForOp>(owner);
            const bool induction = loop && loop.getInductionVar() == current;
            if (!induction) {
                return false;
            }
            for (Value bound : {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}) {
                work.push_back({bound, loop});
            }
            continue;
        }
        Operation* definition = current.getDefiningOp();
        const bool arithmetic = definition && definition->getName().getDialectNamespace() == "arith" &&
                                definition->getNumRegions() == 0 && isMemoryEffectFree(definition);
        if (!arithmetic) {
            return false;
        }
        for (Value operand : definition->getOperands()) {
            work.push_back({operand, definition});
        }
    }
    return true;
}

bool supportedDimension(
    Value value, std::optional<std::uint64_t> constant, std::int64_t bound, Operation* use, DominanceInfo& dominance)
{
    if (constant) {
        return *constant <= static_cast<std::uint64_t>(bound) && (!value || dominance.dominates(value, use));
    }
    IntegerAttr literal;
    // A negative/oversized literal must not masquerade as a symbolic value.
    if (!value || matchPattern(value, m_Constant(&literal))) {
        return false;
    }
    return nonphysicalScalar(value, use, dominance);
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

bool isDescriptorScope(Operation* operation)
{
    if (!operation) {
        return false;
    }
    for (Operation* parent = operation->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto function = dyn_cast<func::FuncOp>(parent)) {
            return function.getBody().hasOneBlock();
        }
        if (!isa_and_nonnull<scf::ForOp, scf::IfOp>(parent)) {
            return false;
        }
    }
    return false;
}

bool isStructuredRead(Operation* operation, Operation* allocation)
{
    for (Operation* current = operation; current; current = current->getParentOp()) {
        const bool sameBlock = current->getBlock() == allocation->getBlock();
        if (sameBlock) {
            return true;
        }
        Operation* parent = current->getParentOp();
        if (!isa_and_nonnull<scf::ForOp, scf::IfOp>(parent)) {
            return false;
        }
    }
    return false;
}

} // namespace

namespace mlir::pto::protocol_sync {

class SyncDescriptorStateBuilder {
public:
    explicit SyncDescriptorStateBuilder(StructuredSyncIR& schedule)
        : schedule(schedule), dominance(schedule.getFunction())
    {}
    LogicalResult build();

private:
    void reject(Operation* operation, StringRef detail);
    bool validateHandle(const SyncSemanticAction& action, Value handle);
    void transfer(SyncSemanticAction& action);

    StructuredSyncIR& schedule;
    DominanceInfo dominance;
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
    if (!isDescriptorScope(allocation)) {
        reject(action.operation, "descriptor allocation needs a supported structured lifetime scope");
        return false;
    }
    if (validated.contains(handle)) {
        return true;
    }
    for (Operation* user : handle.getUsers()) {
        const bool metadata = isa<SetValidShapeOp, GetValidShapeOp>(user);
        const bool payload = payloadOperations.lookup(user) == 1 && user->getNumResults() == 0;
        const bool nestedUpdate = isa<SetValidShapeOp>(user) && user->getBlock() != allocation->getBlock();
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
    const bool unsupportedBounds =
        !supportedDimension(effect->rows, rows, type.getShape()[0], action.operation, dominance) ||
        !supportedDimension(effect->columns, columns, type.getShape()[1], action.operation, dominance);
    if (missingInput || unsupportedBounds) {
        reject(
            action.operation, "descriptor dimensions need valid constants or supported nonphysical scalar provenance");
        invalid.insert(effect->handle);
        current.erase(effect->handle);
        return;
    }
    const std::uint32_t id = schedule.descriptorStates.size();
    const auto provenance = rows && columns ? SyncDescriptorScalarProvenance::Constant :
                                              SyncDescriptorScalarProvenance::NonphysicalExpression;
    schedule.descriptorStates.push_back(
        {id, effect->handle, action.id, rows, columns, effect->rows, effect->columns, provenance});
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
    // Definitions execute in the allocation's block. Each dynamic invocation
    // of that scope initializes its own descriptor; handles cannot escape.
    // Nested reads preserve that version and empty regions do no transfer.
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
    // Moving an update away from its allocation's block cannot preserve a
    // certificate. Scope-local descriptors do not carry state across invocations.
    for (const SyncDescriptorState& state : states) {
        if (!state.handle) {
            return failure();
        }
        const auto* definition = schedule.findSemanticAction(state.definition);
        Operation* allocation = state.handle.getDefiningOp();
        const bool invalidDefinition = !definition || !definition->operation || !supportedAllocation(state.handle) ||
                                       !isDescriptorScope(allocation) ||
                                       definition->operation->getBlock() != allocation->getBlock();
        if (invalidDefinition) {
            return failure();
        }
        for (Operation* user : state.handle.getUsers()) {
            const bool misplacedUpdate = isa<SetValidShapeOp>(user) && user->getBlock() != allocation->getBlock();
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
            if (!matchesLiveDescriptor(action.operation, *effect)) {
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
                const bool wrongBounds =
                    !supportedDimension(effect->rows, rows, type.getShape()[0], action.operation, dominance) ||
                    !supportedDimension(effect->columns, columns, type.getShape()[1], action.operation, dominance);
                const auto provenance = rows && columns ? SyncDescriptorScalarProvenance::Constant :
                                                          SyncDescriptorScalarProvenance::NonphysicalExpression;
                const bool wrongPredecessor = effect->role == SyncDescriptorRole::Initialize ?
                                                  latest.contains(effect->handle) :
                                                  !latest.contains(effect->handle);
                if (state.definition != action.id || wrongBounds || wrongPredecessor || state.rows != rows ||
                    state.columns != columns || state.scalarProvenance != provenance ||
                    state.rowSource != effect->rows || state.columnSource != effect->columns) {
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
