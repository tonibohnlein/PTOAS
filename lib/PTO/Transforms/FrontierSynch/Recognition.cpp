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
#include "PTO/IR/PTOMultiBuffer.h"
#include "RecognitionInternal.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "../InsertSync/SyncRegionArithmetic.h"
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
std::optional<int64_t> constant(Value value)
{
    APInt number;
    const bool known = matchPattern(value, m_ConstantInt(&number)) && number.isSignedIntN(64);
    if (!known) {
        return std::nullopt;
    }
    return number.getSExtValue();
}
struct Family {
    std::optional<uint64_t> stride;
    SmallVector<SyncStorageCell> slots;
    std::optional<SyncStorageCell> contiguous;
    uint64_t bytes = 0;
    uint64_t count = 0;
    std::optional<SyncStorageCell> firstPhysicalSlot;
    uint64_t physicalSlotStride = 0;
};
Family geometry(const SyncInput& input, const BaseMemInfo& memory, uint64_t count)
{
    Family family;
    family.count = count;
    auto root = input.buffers().find(memory.rootBuffer);
    if (root == input.buffers().end() || root->second.size() != 1) {
        return family;
    }
    const auto& storage = *root->second.front();
    family.bytes = storage.allocateSize;
    // A constant-base alloc_multi_tile lays out contiguous equal-sized slots.
    // Its syntax is a certificate; do not reconstruct and sort every slot here.
    auto alloc = memory.rootBuffer.getDefiningOp<AllocMultiTileOp>();
    const auto base = alloc && alloc.getAddr() ? constant(alloc.getAddr()) : std::nullopt;
    if (alloc && !alloc->hasAttr(kPtoMultiBufferAddrsAttrName) && base && *base >= 0 &&
        storage.hasKnownPhysicalAddresses && family.bytes &&
        count <= (UINT64_MAX - static_cast<uint64_t>(*base)) / family.bytes) {
        family.contiguous = SyncStorageCell{storage.scope, static_cast<uint64_t>(*base),
            static_cast<uint64_t>(*base) + count * family.bytes};
    } else {
        family.slots = mlir::pto::detail::physicalSlotRanges(input, memory);
    }
    return family;
}
using Families = DenseMap<Value, Family>;

void checkDisjoint(const Families& families, RecognitionResult& result, Operation* anchor)
{
    SmallVector<SyncStorageCell> intervals;
    for (const auto& entry : families) {
        if (entry.second.contiguous) {
            intervals.push_back(*entry.second.contiguous);
        } else {
            llvm::append_range(intervals, entry.second.slots);
        }
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

bool dischargeGlobal(std::size_t id, scf::ForOp loop, const SyncInput& input, const PhaseIndex& index)
{
    const auto& effects = input.accesses();
    const auto& effect = effects.effects()[id];
    if (effect.memory->scope != AddressSpace::GM || !effects.independentOfOtherPhases(id)) {
        return false;
    }
    const bool phaseWritesGM = llvm::any_of(effects.effectsFor(effect.phase), [&](std::size_t other) {
        const auto& candidate = effects.effects()[other];
        return candidate.memory->scope == AddressSpace::GM && candidate.mode == SyncAccessMode::Write;
    });
    if (!phaseWritesGM) { return true; }
    // A repeated writer must also have disjoint visits. Check every GM piece
    // of this phase together, so separate output operands cannot hide reuse.
    std::optional<int64_t> translation;
    uint64_t begin = UINT64_MAX, end = 0;
    Value base;
    AffineExpr commonOrigin;
    SmallVector<Value> commonSymbols;
    for (auto other : effects.effectsFor(effect.phase)) {
        const auto& candidate = effects.effects()[other];
        if (candidate.memory->scope != AddressSpace::GM) { continue; }
        if (!effects.independentOfOtherPhases(other) || candidate.regions.empty()) { return false; }
        for (const auto& region : candidate.regions) {
            auto iv = llvm::find(region.symbols, loop.getInductionVar());
            if (iv == region.symbols.end() || (base && base != region.base)) { return false; }
            base = region.base;
            const auto position = iv - region.symbols.begin();
            auto* context = loop.getContext();
            SmallVector<Operation*> recipe;
            for (unsigned i = 0; i < region.symbols.size(); ++i) {
                if (i != position && !detail::entryExpression(region.symbols[i], loop, index, recipe)) {
                    return false;
                }
            }
            for (auto extent : region.extents) {
                if (extent.isFunctionOfSymbol(position)) { return false; }
            }
            const auto dims = region.extents.size(), syms = region.symbols.size();
            auto parts = mlir::pto::detail::splitTranslation(region.byteOffset, dims, syms, position);
            if (!parts || (translation && *translation != parts->second)) { return false; }
            translation = parts->second;
            auto zero = parts->first;
            SmallVector<AffineExpr> zeros(dims, getAffineConstantExpr(0, context));
            auto origin = simplifyAffineExpr(zero.replaceDims(zeros), 0, syms);
            if (commonOrigin && (commonOrigin != origin || commonSymbols != region.symbols)) { return false; }
            commonOrigin = origin;
            commonSymbols = region.symbols;
            auto fixed = region;
            fixed.base = {};
            auto relative = mlir::pto::detail::checkedAdd(zero, mlir::pto::detail::checkedMul(
                origin, getAffineConstantExpr(-1, context)));
            if (!relative) { return false; }
            fixed.byteOffset = simplifyAffineExpr(relative, dims, syms);
            for (unsigned i = 0; i < syms; ++i) {
                if (fixed.byteOffset.isFunctionOfSymbol(i) || llvm::any_of(fixed.extents, [i](AffineExpr extent) {
                        return extent.isFunctionOfSymbol(i);
                    })) { return false; }
            }
            fixed.symbols.clear();
            SmallVector<SyncStorageCell> ranges;
            if (!mlir::pto::detail::materializeRegion(fixed, AddressSpace::GM, ranges)) { return false; }
            for (auto range : ranges) { begin = std::min(begin, range.begin); end = std::max(end, range.end); }
        }
    }
    auto step = constant(loop.getStep());
    if (!translation || !step || *step <= 0) { return false; }
    auto displacement = APInt(128, *translation, true) * APInt(128, *step);
    if (displacement.isNegative()) { displacement = -displacement; }
    return begin == UINT64_MAX || displacement.uge(APInt(128, end - begin));
}

void inspectAccess(std::size_t id, scf::ForOp loop, const SyncInput& input,
                   const SyncStorageEffects& effects, Families& families, RecognitionResult& result,
                   const PhaseIndex& index, bool allowParameters)
{
    const auto& effect = effects.effects()[id];
    const auto& memory = *effect.memory;
    Operation* anchor = effect.phase->elementOp;
    if (dischargeGlobal(id, loop, input, index)) {
        result.dischargedEffects.push_back(id);
        return;
    }
    // An unresolved address has an occurrence-independent alias predicate.
    // Its obligations are added beside the rotating geometric generators.
    if (!effect.rangesMaterialized && effect.regions.empty()) { return; }
    auto alias = input.buffers().find(memory.baseBuffer);
    if (alias == input.buffers().end() || alias->second.size() != 1 || !memory.rootBuffer) {
        result.note(RecognitionIssue::AliasedOperand, anchor);
        return;
    }
    auto multi = dyn_cast<MultiTileBufType>(memory.rootBuffer.getType());
    const auto physical = detail::physicalRotation(effect, input, loop, index, allowParameters);
    const uint64_t count = physical ? physical->count : (multi ? multi.getCount() : 1);
    std::optional<detail::SlotPattern> pattern;
    if (physical) {
        pattern = physical->pattern;
    } else if (effect.selection && effect.selection->family == memory.rootBuffer) {
        pattern = detail::matchRotatingSlot(effect.selection->selector, loop, count, index, allowParameters);
    } else if (!effect.selection && count == 1) {
        pattern = detail::SlotPattern{};
    } else {
        result.note(RecognitionIssue::UnsupportedView, anchor);
        return;
    }
    if (!pattern) {
        result.note(RecognitionIssue::SlotExpression, anchor, true);
        return;
    }
    // The exported pattern uses the iteration ordinal j, with iv = lower+step*j.
    // Widen products before reducing; large constants must not wrap in the host.
    auto lower = constant(loop.getLowerBound()), step = constant(loop.getStep());
    if (lower && *lower >= 0 && step && *step > 0) {
        APInt modulus(128, count), stride(128, pattern->stride);
        auto shift = (stride * APInt(128, *lower)).urem(modulus).getZExtValue();
        pattern->offset = (APInt(128, shift) + APInt(128, pattern->offset)).urem(modulus).getZExtValue();
        if (pattern->parameterOffset) {
            pattern->parameterOffset = (pattern->parameterOffset + static_cast<int64_t>(shift)) %
                                       static_cast<int64_t>(count);
        }
        pattern->stride = (stride * APInt(128, *step)).urem(modulus).getZExtValue();
    }
    Value familyValue = memory.rootBuffer;
    if (physical) {
        for (const auto& entry : families) {
            const auto& candidate = entry.second;
            if (candidate.firstPhysicalSlot && candidate.count == count &&
                candidate.physicalSlotStride == physical->strideBytes &&
                sameStorageDomain(*candidate.firstPhysicalSlot, physical->firstSlot) &&
                candidate.firstPhysicalSlot->begin == physical->firstSlot.begin &&
                candidate.firstPhysicalSlot->end == physical->firstSlot.end) {
                familyValue = entry.first;
                break;
            }
        }
    }
    auto found = families.find(familyValue);
    if (found == families.end()) {
        auto description = geometry(input, memory, count);
        if (physical) {
            description.firstPhysicalSlot = physical->firstSlot;
            description.physicalSlotStride = physical->strideBytes;
            description.bytes = physical->firstSlot.end - physical->firstSlot.begin;
            description.contiguous = SyncStorageCell{physical->firstSlot.space, physical->firstSlot.begin,
                physical->firstSlot.begin + (count - 1) * physical->strideBytes + description.bytes};
        }
        found = families.try_emplace(familyValue, std::move(description)).first;
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
    std::optional<detail::SlotRanges> atoms;
    if ((!family.contiguous && family.slots.size() != count) || (memory.aliasesUnknownRange && !physical)) {
        result.note(RecognitionIssue::UnknownGeometry, anchor);
    } else {
        const auto bytes = family.bytes;
        atoms = detail::withinSlotRanges(effect, input, bytes);
        if (!atoms) {
            result.note(RecognitionIssue::WithinSlotFootprint, anchor);
        }
    }
    RotatingAccess access{id, familyValue, count, pattern->stride, pattern->offset,
                           count / std::gcd(count, pattern->stride), std::nullopt};
    access.firstPhysicalSlot = family.firstPhysicalSlot;
    access.physicalSlotStride = family.physicalSlotStride;
    access.parameterOffset = pattern->parameterOffset;
    access.parameters = pattern->parameters;
    access.effects.push_back(id);
    access.reads = effect.mode == SyncAccessMode::Read;
    access.writes = effect.mode == SyncAccessMode::Write;
    if (!atoms) {
        result.accesses.push_back(std::move(access));
        return;
    }
    for (auto atom : *atoms) {
        access.atom = atom;
        result.accesses.push_back(access);
    }
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
    return result;
}

RecognitionResult recognizeExplicitRun(ArrayRef<Operation*> operations, const PhaseIndex& index,
                                       const SyncStorageEffects& effects)
{
    RecognitionResult result;
    Operation* previous = nullptr;
    for (auto* op : operations) {
        if (!op || op->getNumRegions() || (previous && previous->getNextNode() != op)) {
            result.note(RecognitionIssue::StructuredBody, op, true);
            continue;
        }
        previous = op;
        detail::inspectLeaf(*op, index, result);
    }
    return result;
}

void detail::inspectLeaf(Operation& op, const PhaseIndex& index, RecognitionResult& result)
{
    auto phases = index.phasesFor(&op);
    const bool extra = index.needsValuePrerequisite(&op) ||
        isa<SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp, RecordEventOp, WaitEventOp, BarrierOp>(op);
    if (extra) {
        result.note(RecognitionIssue::AdditionalPrerequisite, &op);
    }
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

bool detail::checkRotatingDomain(scf::ForOp loop, RecognitionResult& result, bool canonical)
{
    if (!loop) {
        result.note(RecognitionIssue::LoopDomain, nullptr, true);
        return false;
    }
    auto lower = constant(loop.getLowerBound()), step = constant(loop.getStep());
    const bool normalized = lower && *lower >= 0 && step && *step > 0;
    if (!normalized || (canonical && (*lower != 0 || *step != 1))) {
        result.note(RecognitionIssue::LoopDomain, loop, true);
    }
    if (loop.getNumRegionIterArgs()) {
        result.note(RecognitionIssue::LoopCarriedState, loop, true);
    }
    return true;
}

void detail::inspectRotatingPhases(scf::ForOp loop, ArrayRef<const CompoundInstanceElement*> phases,
                                  const SyncInput& input, const SyncStorageEffects& effects, RecognitionResult& result,
                                  const PhaseIndex& index, bool allowParameters)
{
    Families families;
    for (const auto* phase : phases) {
        for (auto id : effects.effectsFor(phase)) {
            inspectAccess(id, loop, input, effects, families, result, index, allowParameters);
        }
    }
    checkDisjoint(families, result, loop);
    detail::normalizeFragments(result, effects);
}

RecognitionResult recognizeRotating(scf::ForOp loop, const PhaseIndex& index,
                                    const SyncInput& input, const SyncStorageEffects& effects)
{
    RecognitionResult result;
    if (!detail::checkRotatingDomain(loop, result) || !fixedBody(*loop.getBody(), index, result)) {
        return result;
    }
    if (index.needsValuePrerequisite(loop)) {
        result.note(RecognitionIssue::AdditionalPrerequisite, loop);
    }
    SmallVector<const CompoundInstanceElement*> phases;
    for (Operation& op : *loop.getBody()) {
        llvm::append_range(phases, index.phasesFor(&op));
    }
    detail::inspectRotatingPhases(loop, phases, input, effects, result, index);
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
    case RecognitionIssue::WithinSlotFootprint: return "unsupported-slot-footprint";
    case RecognitionIssue::SymbolicGeometry: return "symbolic-storage-partition";
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
    case RecognitionIssue::TemplateExpansionLimit: return "template-expansion-limit";
    case RecognitionIssue::TemplateContext: return "template-context";
    case RecognitionIssue::GMDischarge: return "gm-discharge";
    default: return "invalid";
    }
}
} // namespace mlir::pto::frontiersynch
