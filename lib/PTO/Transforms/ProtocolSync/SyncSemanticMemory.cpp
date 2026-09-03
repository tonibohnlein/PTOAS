// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- SyncSemanticMemory.cpp - ProtocolSync memory summaries -----------===//

#include "SyncSemanticsInternal.h"

#include "PTO/IR/PTOTypeUtils.h"
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/SlotAffineAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <limits>
#include <numeric>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

constexpr unsigned kMaxSlotExpressionPeels = 4;

std::int64_t positiveModulo(std::int64_t value, std::uint32_t modulus)
{
    std::int64_t result = value % static_cast<std::int64_t>(modulus);
    return result < 0 ? result + modulus : result;
}

bool getConstantInteger(Value value, std::int64_t& result)
{
    IntegerAttr attribute;
    if (!matchPattern(value, m_Constant(&attribute))) {
        return false;
    }
    result = attribute.getValue().getSExtValue();
    return true;
}

bool addWithoutOverflow(std::int64_t first, std::int64_t second, std::int64_t& result)
{
    constexpr std::int64_t maximum = std::numeric_limits<std::int64_t>::max();
    constexpr std::int64_t minimum = std::numeric_limits<std::int64_t>::min();
    if (second > 0 && first > maximum - second) {
        return false;
    }
    if (second < 0 && first < minimum - second) {
        return false;
    }
    result = first + second;
    return true;
}

bool peelSlotOffset(Value value, Value& remaining, std::int64_t& offset)
{
    Operation* operation = value.getDefiningOp();
    if (!operation) {
        return false;
    }
    Value lhs;
    Value rhs;
    bool isSubtract = false;
    if (auto add = dyn_cast<arith::AddIOp>(operation)) {
        lhs = add.getLhs();
        rhs = add.getRhs();
    } else if (auto subtract = dyn_cast<arith::SubIOp>(operation)) {
        lhs = subtract.getLhs();
        rhs = subtract.getRhs();
        isSubtract = true;
    } else {
        return false;
    }

    std::int64_t constant = 0;
    std::int64_t updatedOffset = 0;
    if (getConstantInteger(rhs, constant)) {
        if (isSubtract && constant == std::numeric_limits<std::int64_t>::min()) {
            return false;
        }
        const std::int64_t delta = isSubtract ? -constant : constant;
        if (!addWithoutOverflow(offset, delta, updatedOffset)) {
            return false;
        }
        remaining = lhs;
        offset = updatedOffset;
        return true;
    }
    const bool hasLeftConstant = !isSubtract && getConstantInteger(lhs, constant);
    const bool hasValidLeftOffset = hasLeftConstant && addWithoutOverflow(offset, constant, updatedOffset);
    if (hasValidLeftOffset) {
        remaining = rhs;
        offset = updatedOffset;
        return true;
    }
    return false;
}

SyncSlotExpression canonicalizeSlotExpression(Value selector, std::uint32_t depth)
{
    SyncSlotExpression result;
    result.selector = selector;
    result.depth = depth;

    std::int64_t constant = 0;
    if (getConstantInteger(selector, constant)) {
        const bool constantOutOfRange = depth == 0 || constant < 0 || static_cast<std::uint64_t>(constant) >= depth;
        if (constantOutOfRange) {
            return result;
        }
        result.kind = SyncSlotExpressionKind::Constant;
        result.offset = constant;
        result.modulus = depth;
        return result;
    }

    Value inner;
    std::int64_t modulus = 0;
    if (auto remainder = selector.getDefiningOp<arith::RemSIOp>()) {
        inner = remainder.getLhs();
        if (!getConstantInteger(remainder.getRhs(), modulus)) {
            return result;
        }
    } else if (auto remainder = selector.getDefiningOp<arith::RemUIOp>()) {
        inner = remainder.getLhs();
        if (!getConstantInteger(remainder.getRhs(), modulus)) {
            return result;
        }
    } else {
        return result;
    }
    constexpr std::uint64_t maximumModulus = std::numeric_limits<std::uint32_t>::max();
    const bool modulusOutOfRange = modulus <= 0 || static_cast<std::uint64_t>(modulus) > maximumModulus;
    if (modulusOutOfRange) {
        return result;
    }
    const std::uint32_t canonicalModulus = static_cast<std::uint32_t>(modulus);
    if (depth == 0 || canonicalModulus != depth) {
        return result;
    }

    std::int64_t offset = 0;
    for (unsigned peel = 0; peel < kMaxSlotExpressionPeels; ++peel) {
        Value remaining;
        if (!peelSlotOffset(inner, remaining, offset)) {
            break;
        }
        inner = remaining;
    }
    if (getConstantInteger(inner, constant)) {
        std::int64_t combined = 0;
        const bool validCombined = addWithoutOverflow(constant, offset, combined) && combined >= 0;
        if (!validCombined) {
            return result;
        }
        result.kind = SyncSlotExpressionKind::Constant;
        result.offset = positiveModulo(combined, canonicalModulus);
        result.modulus = canonicalModulus;
        return result;
    }

    auto loop = dyn_cast_or_null<scf::ForOp>(inner.getParentBlock()->getParentOp());
    const bool isLoopInduction = loop && loop.getInductionVar() == inner;
    if (!isLoopInduction) {
        return result;
    }
    std::optional<std::int64_t> lowerBound = getConstantIntValue(loop.getLowerBound());
    std::optional<std::int64_t> step = getConstantIntValue(loop.getStep());
    std::int64_t minimumInner = 0;
    const bool hasMinimum = lowerBound && addWithoutOverflow(*lowerBound, offset, minimumInner);
    const bool validLoopCoordinate = hasMinimum && minimumInner >= 0 && step && *step > 0;
    if (!validLoopCoordinate) {
        return result;
    }

    result.kind = SyncSlotExpressionKind::AffineModulo;
    result.induction = inner;
    result.coefficient = positiveModulo(*step, canonicalModulus);
    result.offset = positiveModulo(minimumInner, canonicalModulus);
    result.modulus = canonicalModulus;
    return result;
}

SyncVisibilityClass visibilityFor(AddressSpace space)
{
    if (space == AddressSpace::GM) {
        return SyncVisibilityClass::Global;
    }
    return space == AddressSpace::Zero ? SyncVisibilityClass::Unknown : SyncVisibilityClass::Local;
}

std::optional<SyncSlotExpression> getSlotExpression(Value value)
{
    Value selector = findMultiTileSlotExpr(value);
    if (!selector) {
        return std::nullopt;
    }
    return canonicalizeSlotExpression(selector, getSyncSlotDepth(value).value_or(0));
}

} // namespace

SyncSlotRelation mlir::pto::protocol_sync::compareSlotsAtDistance(
    const SyncSlotExpression& first, const SyncSlotExpression& second, unsigned distance)
{
    if (first.kind == SyncSlotExpressionKind::Unknown || second.kind == SyncSlotExpressionKind::Unknown ||
        first.modulus == 0 || first.modulus != second.modulus) {
        return SyncSlotRelation::Unknown;
    }
    const std::uint32_t modulus = first.modulus;
    if (first.kind == SyncSlotExpressionKind::Constant && second.kind == SyncSlotExpressionKind::Constant) {
        const std::int64_t firstOffset = positiveModulo(first.offset, modulus);
        const std::int64_t secondOffset = positiveModulo(second.offset, modulus);
        return firstOffset == secondOffset ? SyncSlotRelation::Same : SyncSlotRelation::Different;
    }
    if (first.kind != SyncSlotExpressionKind::AffineModulo || second.kind != SyncSlotExpressionKind::AffineModulo ||
        first.loop == kInvalidSyncId || first.loop != second.loop) {
        return SyncSlotRelation::Unknown;
    }

    const std::int64_t firstCoefficient = positiveModulo(first.coefficient, modulus);
    const std::int64_t secondCoefficient = positiveModulo(second.coefficient, modulus);
    if (firstCoefficient != secondCoefficient) {
        return SyncSlotRelation::Unknown;
    }
    const std::uint64_t shifted = (static_cast<std::uint64_t>(secondCoefficient) * (distance % modulus)) % modulus;
    const std::int64_t difference = positiveModulo(
        positiveModulo(second.offset, modulus) + static_cast<std::int64_t>(shifted) -
            positiveModulo(first.offset, modulus),
        modulus);
    return difference == 0 ? SyncSlotRelation::Same : SyncSlotRelation::Different;
}

FailureOr<unsigned> mlir::pto::protocol_sync::findFirstPositiveReuseDistance(
    const SyncSlotExpression& slot, unsigned searchLimit)
{
    if (slot.kind == SyncSlotExpressionKind::Unknown || slot.modulus == 0 || searchLimit == 0) {
        return failure();
    }
    if (slot.kind == SyncSlotExpressionKind::Constant) {
        return 1U;
    }
    const std::uint32_t coefficient = static_cast<std::uint32_t>(positiveModulo(slot.coefficient, slot.modulus));
    const unsigned period = slot.modulus / std::gcd(coefficient, slot.modulus);
    if (period > searchLimit) {
        return failure();
    }
    return period;
}

std::optional<std::uint32_t> mlir::pto::protocol_sync::getSyncSlotDepth(Value value)
{
    Value current = value;
    for (unsigned hops = 0; current && hops < pto::kValue32; ++hops) {
        Operation* definition = current.getDefiningOp();
        if (!definition) {
            break;
        }
        if (auto get = dyn_cast<MultiTileGetOp>(definition)) {
            return get.getSource().getType().getCount();
        }
        if (auto subview = dyn_cast<SubViewOp>(definition)) {
            current = subview.getSource();
        } else if (auto reshape = dyn_cast<TReshapeOp>(definition)) {
            current = reshape.getSrc();
        } else if (auto bitcast = dyn_cast<BitcastOp>(definition)) {
            current = bitcast.getSrc();
        } else {
            break;
        }
    }
    return std::nullopt;
}

SyncPhysicalCore mlir::pto::protocol_sync::detail::resolveCore(Operation* operation, PIPE pipe)
{
    for (Operation* parent = operation; parent; parent = parent->getParentOp()) {
        if (isa<SectionCubeOp>(parent)) {
            return SyncPhysicalCore::Cube;
        }
        if (isa<SectionVectorOp>(parent)) {
            return SyncPhysicalCore::Vector;
        }
    }
    if (pipe == PIPE::PIPE_M || pipe == PIPE::PIPE_MTE1) {
        return SyncPhysicalCore::Cube;
    }
    if (pipe == PIPE::PIPE_FIX) {
        return SyncPhysicalCore::Cube;
    }
    if (pipe == PIPE::PIPE_V) {
        return SyncPhysicalCore::Vector;
    }
    if (pipe == PIPE::PIPE_S || pipe == PIPE::PIPE_MTE2 || pipe == PIPE::PIPE_MTE3) {
        func::FuncOp function = operation->getParentOfType<func::FuncOp>();
        auto kind = function ? function->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name) :
                               FunctionKernelKindAttr();
        if (kind) {
            return kind.getKernelKind() == FunctionKernelKind::Cube ? SyncPhysicalCore::Cube : SyncPhysicalCore::Vector;
        }
    }
    return SyncPhysicalCore::Unknown;
}

void mlir::pto::protocol_sync::detail::setFailure(SyncOpSummary& summary, SyncFailureReason reason, StringRef message)
{
    if (summary.failure == SyncFailureReason::None) {
        summary.failure = reason;
        summary.failureDetail = message.str();
    }
}

void mlir::pto::protocol_sync::detail::setPhysicalResource(
    Operation* operation, PIPE pipe, SyncPhysicalPhase& phase, SyncOpSummary& summary)
{
    phase.pipe = pipe;
    phase.core = resolveCore(operation, pipe);
    if (phase.core == SyncPhysicalCore::Unknown) {
        setFailure(summary, SyncFailureReason::MissingPhysicalCore, "operation has no physical core assignment");
    }
}

void mlir::pto::protocol_sync::detail::appendValueEffects(
    SyncPhysicalPhase& phase, Value value, SyncAccessMode mode, const SyncSemanticContext& context,
    SyncOpSummary& summary, ProtocolSyncStatistics* statistics)
{
    ArrayRef<SyncStorageProvenance> storage = context.lookupStorage(value);
    std::optional<SyncSlotExpression> slot = getSlotExpression(value);
    if (storage.empty()) {
        SyncStorageProvenance unknown;
        unknown.value = value;
        unknown.root = value;
        unknown.aliasesUnknownRange = true;
        if (auto space = getPTOAddressSpaceAttr(value.getType())) {
            unknown.space = space.getAddressSpace();
        }
        const SyncVisibilityClass visibility = visibilityFor(unknown.space);
        phase.effects.push_back({value, mode, std::move(unknown), visibility, slot});
        setFailure(summary, SyncFailureReason::MissingStorageProvenance, "memory effect has no storage provenance");
        return;
    }
    for (const SyncStorageProvenance& provenance : storage) {
        phase.effects.push_back({value, mode, provenance, visibilityFor(provenance.space), slot});
    }
}

void mlir::pto::protocol_sync::detail::addPipelinePhase(
    Operation* operation, PIPE pipe, const SyncSemanticContext& context, SyncOpSummary& summary,
    ProtocolSyncStatistics* statistics)
{
    if (pipe == PIPE::PIPE_UNASSIGNED || pipe == PIPE::PIPE_ALL) {
        setFailure(summary, SyncFailureReason::MissingPipeline, "operation has no concrete physical pipeline");
        return;
    }
    SyncPhysicalPhase phase;
    setPhysicalResource(operation, pipe, phase, summary);
    auto interface = dyn_cast<MemoryEffectOpInterface>(operation);
    if (!interface) {
        if (!isMemoryEffectFree(operation)) {
            setFailure(
                summary, SyncFailureReason::UnsupportedEffectfulOperation,
                "physical operation has no MemoryEffectOpInterface");
        }
        summary.phases.push_back(std::move(phase));
        return;
    }
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    interface.getEffects(effects);
    for (const MemoryEffects::EffectInstance& effect : effects) {
        Value value = effect.getValue();
        if (!value) {
            setFailure(
                summary, SyncFailureReason::UnscopedMemoryEffect,
                "physical operation reports a memory effect without an SSA value");
            continue;
        }
        if (isa<MemoryEffects::Read>(effect.getEffect())) {
            appendValueEffects(phase, value, SyncAccessMode::Read, context, summary, statistics);
        } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
            appendValueEffects(phase, value, SyncAccessMode::Write, context, summary, statistics);
        } else {
            setFailure(
                summary, SyncFailureReason::UnsupportedMemoryEffectKind,
                "physical operation reports an unsupported memory-effect kind");
        }
    }
    summary.phases.push_back(std::move(phase));
}
