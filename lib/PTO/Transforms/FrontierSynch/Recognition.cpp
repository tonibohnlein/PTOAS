// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Inspect original region structure and supplied effects without expanding loops.
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "RotationPattern.h"
#include "RecognitionInternal.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>
#include <tuple>

namespace mlir::pto::frontiersynch {
void RecognitionResult::note(RecognitionIssue issue, Operation* anchor, bool outsideClass)
{
    diagnostics.push_back({issue, anchor});
    if (outsideClass) {
        state = RecognitionState::NotApplicable;
    } else if (state != RecognitionState::NotApplicable) {
        state = RecognitionState::MissingPremise;
    }
}
namespace {
bool metadata(Operation& op)
{
    return isa<AllocTileOp, AllocMultiTileOp, MultiTileGetOp, SubViewOp>(op) ||
           op.hasTrait<OpTrait::IsTerminator>() || isMemoryEffectFree(&op);
}
bool fixedBody(Block& block, const PhaseIndex& index, RecognitionResult& result)
{
    bool fixed = true;
    for (Operation& op : block) {
        auto phases = index.phasesFor(&op);
        if (op.getNumRegions()) {
            result.note(RecognitionIssue::StructuredBody, &op, true);
            fixed = false;
        } else if (phases.size() > 1) {
            fixed = false;
        }
        if (!op.getNumRegions()) {
            detail::inspectLeaf(op, index, result);
        }
    }
    return fixed;
}
bool constantEquals(Value value, int64_t expected)
{
    APInt number;
    return matchPattern(value, m_ConstantInt(&number)) && number.isSignedIntN(64) &&
           number.getSExtValue() == expected;
}
struct Family {
    std::optional<uint64_t> stride;
    SmallVector<SyncStorageCell> slots;
};
using Families = DenseMap<Value, Family>;

void checkDisjoint(const Families& families, RecognitionResult& result, Operation* anchor)
{
    SmallVector<SyncStorageCell> intervals;
    for (const auto& entry : families) {
        llvm::append_range(intervals, entry.second.slots);
    }
    llvm::sort(intervals, [](const auto& a, const auto& b) {
        return std::tie(a.space, a.begin, a.end) < std::tie(b.space, b.begin, b.end);
    });
    for (std::size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i - 1].space == intervals[i].space && intervals[i].begin < intervals[i - 1].end) {
            result.note(RecognitionIssue::OverlappingFamilies, anchor, true);
            return;
        }
    }
}

void inspectAccess(std::size_t id, scf::ForOp loop, const SyncInput& input,
                   const SyncStorageEffects& effects, Families& families, RecognitionResult& result)
{
    const auto& effect = effects.effects()[id];
    const auto& memory = *effect.memory;
    Operation* anchor = effect.phase->elementOp;
    auto alias = input.buffers().find(memory.baseBuffer);
    if (alias == input.buffers().end() || alias->second.size() != 1 || !memory.rootBuffer) {
        result.note(RecognitionIssue::AliasedOperand, anchor);
        return;
    }
    auto multi = dyn_cast<MultiTileBufType>(memory.rootBuffer.getType());
    const uint64_t count = multi ? multi.getCount() : 1;
    std::optional<detail::SlotPattern> pattern;
    if (auto get = memory.baseBuffer.getDefiningOp<MultiTileGetOp>();
        get && get.getSource() == memory.rootBuffer) {
        pattern = detail::matchSlot(get.getSlot(), loop.getInductionVar(), count);
    } else if (memory.baseBuffer == memory.rootBuffer && count == 1) {
        pattern = detail::SlotPattern{};
    } else {
        result.note(RecognitionIssue::UnsupportedView, anchor);
        return;
    }
    if (!pattern) {
        result.note(RecognitionIssue::SlotExpression, anchor, true);
        return;
    }
    auto found = families.find(memory.rootBuffer);
    if (found == families.end()) {
        auto slots = mlir::pto::detail::physicalSlotRanges(memory);
        found = families.try_emplace(memory.rootBuffer, Family{std::nullopt, std::move(slots)}).first;
    }
    auto& family = found->second;
    // Distinguish an observed normalized form from proven machine arithmetic.
    if (!pattern->arithmeticProven) {
        result.note(RecognitionIssue::IndexArithmetic, anchor);
    }
    if (family.stride && *family.stride != pattern->stride) {
        result.note(RecognitionIssue::CommonStride, anchor, true);
    }
    family.stride = pattern->stride;
    std::optional<std::pair<uint64_t, uint64_t>> atom;
    if (family.slots.size() != count || memory.aliasesUnknownRange) {
        result.note(RecognitionIssue::UnknownGeometry, anchor);
    } else {
        const auto bytes = family.slots.front().end - family.slots.front().begin;
        atom = detail::scalarAtom(effect, bytes);
        if (!atom) {
            result.note(RecognitionIssue::InexactFootprint, anchor);
        }
    }
    result.accesses.push_back({id, memory.rootBuffer, count, pattern->stride, pattern->offset,
                               count / std::gcd(count, pattern->stride), atom});
}
} // namespace

RecognitionResult recognizeExplicit(Block& block, const PhaseIndex& index, const SyncStorageEffects& effects)
{
    RecognitionResult result;
    if (!fixedBody(block, index, result)) {
        return result;
    }
    auto sequence = index.explicitSequence(block);
    if (failed(sequence)) {
        result.note(RecognitionIssue::StructuredBody, block.getParentOp(), true);
        return result;
    }
    for (const auto* phase : *sequence) {
        for (auto id : effects.effectsFor(phase)) {
            if (effects.effects()[id].precision != SyncAccessPrecision::Exact) {
                result.note(RecognitionIssue::InexactFootprint, phase->elementOp);
            }
        }
    }
    return result;
}

void detail::inspectLeaf(Operation& op, const PhaseIndex& index, RecognitionResult& result)
{
    auto phases = index.phasesFor(&op);
    if (phases.size() > 1) {
        result.note(RecognitionIssue::MultiplePhases, &op);
    } else if (phases.empty() && !metadata(op)) {
        result.note(RecognitionIssue::UnmodeledOperation, &op);
    }
    for (const auto* phase : phases) {
        if (phase->kPipeValue == PipelineType::PIPE_UNASSIGNED) {
            result.note(RecognitionIssue::UnknownPipe, &op);
        }
    }
}

bool detail::checkRotatingDomain(scf::ForOp loop, RecognitionResult& result)
{
    if (!loop) {
        result.note(RecognitionIssue::LoopDomain, nullptr, true);
        return false;
    }
    if (!constantEquals(loop.getLowerBound(), 0) || !constantEquals(loop.getStep(), 1)) {
        result.note(RecognitionIssue::LoopDomain, loop, true);
    }
    if (loop.getNumRegionIterArgs()) {
        result.note(RecognitionIssue::LoopCarriedState, loop, true);
    }
    return true;
}

void detail::inspectRotatingPhases(scf::ForOp loop, ArrayRef<const CompoundInstanceElement*> phases,
                                  const SyncInput& input, const SyncStorageEffects& effects, RecognitionResult& result)
{
    Families families;
    for (const auto* phase : phases) {
        for (auto id : effects.effectsFor(phase)) {
            inspectAccess(id, loop, input, effects, families, result);
        }
    }
    checkDisjoint(families, result, loop);
}

RecognitionResult recognizeRotating(scf::ForOp loop, const PhaseIndex& index,
                                    const SyncInput& input, const SyncStorageEffects& effects)
{
    RecognitionResult result;
    if (!detail::checkRotatingDomain(loop, result) || !fixedBody(*loop.getBody(), index, result)) {
        return result;
    }
    SmallVector<const CompoundInstanceElement*> phases;
    for (Operation& op : *loop.getBody()) {
        llvm::append_range(phases, index.phasesFor(&op));
    }
    detail::inspectRotatingPhases(loop, phases, input, effects, result);
    return result;
}

StringRef recognitionName(RecognitionState state)
{
    switch (state) {
    case RecognitionState::Applicable: return "applicable";
    case RecognitionState::NotApplicable: return "not-applicable";
    case RecognitionState::MissingPremise: return "missing-premise";
    default: return "invalid";
    }
}
StringRef recognitionName(RecognitionIssue issue)
{
    switch (issue) {
    case RecognitionIssue::StructuredBody: return "structured-body";
    case RecognitionIssue::MultiplePhases: return "multiple-phases";
    case RecognitionIssue::UnmodeledOperation: return "unmodeled-operation";
    case RecognitionIssue::UnknownPipe: return "unknown-pipe";
    case RecognitionIssue::InexactFootprint: return "inexact-footprint";
    case RecognitionIssue::UnknownGeometry: return "unknown-geometry";
    case RecognitionIssue::LoopDomain: return "loop-domain";
    case RecognitionIssue::LoopCarriedState: return "loop-carried-state";
    case RecognitionIssue::SlotExpression: return "slot-expression";
    case RecognitionIssue::IndexArithmetic: return "index-arithmetic";
    case RecognitionIssue::CommonStride: return "common-stride";
    case RecognitionIssue::OverlappingFamilies: return "overlapping-families";
    case RecognitionIssue::AliasedOperand: return "aliased-operand";
    case RecognitionIssue::UnsupportedView: return "unsupported-view";
    case RecognitionIssue::GuardInvariance: return "guard-invariance";
    case RecognitionIssue::UnsupportedControl: return "unsupported-control";
    case RecognitionIssue::ArithmeticDimension: return "arithmetic-dimension";
    case RecognitionIssue::ArithmeticPeriod: return "arithmetic-period";
    case RecognitionIssue::ArithmeticPipeLimit: return "arithmetic-pipe-limit";
    case RecognitionIssue::ArithmeticConfiguration: return "arithmetic-configuration";
    case RecognitionIssue::AdditionalPrerequisite: return "additional-prerequisite";
    default: return "invalid";
    }
}
} // namespace mlir::pto::frontiersynch
