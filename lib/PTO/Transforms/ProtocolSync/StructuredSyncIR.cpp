// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StructuredSyncIR.cpp - Build immutable physical schedule ---------===//

#include "PTO/Transforms/ProtocolSync/StructuredSyncIR.h"

#include "PTO/IR/PTOMultiBuffer.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <chrono>
#include <limits>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

bool hasStaticallyNonEmptyTripCount(scf::ForOp loop)
{
    std::optional<int64_t> lower = getConstantIntValue(loop.getLowerBound());
    std::optional<int64_t> upper = getConstantIntValue(loop.getUpperBound());
    std::optional<int64_t> step = getConstantIntValue(loop.getStep());
    return lower && upper && step && *step > 0 && *lower < *upper;
}

bool sameIntervals(ArrayRef<SyncByteInterval> first, ArrayRef<SyncByteInterval> second)
{
    return first.size() == second.size() && llvm::equal(first, second, [](const auto& left, const auto& right) {
               return left.begin == right.begin && left.size == right.size;
           });
}

std::optional<std::uint64_t> getStaticMultiBufferSlotBytes(TileBufType slotType)
{
    const std::uint64_t elementBytes = getPTOStorageElemByteSize(slotType.getElementType());
    if (elementBytes == 0) {
        return std::nullopt;
    }
    std::uint64_t slotBytes = elementBytes;
    for (std::int64_t dimension : slotType.getShape()) {
        if (dimension <= 0 || dimension == ShapedType::kDynamic) {
            return std::nullopt;
        }
        const std::uint64_t extent = static_cast<std::uint64_t>(dimension);
        const bool sizeOverflows = slotBytes > std::numeric_limits<std::uint64_t>::max() / extent;
        if (sizeOverflows) {
            return std::nullopt;
        }
        slotBytes *= extent;
    }
    return slotBytes;
}

std::optional<std::uint32_t> getAuthoritativeSlotCount(const SyncStorageProvenance& provenance)
{
    if (provenance.root.getDefiningOp<AllocTileOp>()) {
        return 1;
    }
    if (auto allocation = provenance.root.getDefiningOp<AllocMultiTileOp>()) {
        return allocation.getResult().getType().getCount();
    }
    return std::nullopt;
}

bool capacitiesConflict(const std::optional<std::uint32_t>& first, const std::optional<std::uint32_t>& second)
{
    if (!first || !second) {
        return false;
    }
    return *first != *second;
}

std::optional<SmallVector<SyncByteInterval, 2>> getCompletePhysicalSlots(
    const SyncStorageProvenance& provenance, std::optional<std::uint32_t> slotCount)
{
    const bool cannotRecoverSlots = !slotCount || *slotCount < 2 || !provenance.physical;
    if (cannotRecoverSlots) {
        return std::nullopt;
    }
    auto allocation = provenance.root.getDefiningOp<AllocMultiTileOp>();
    const bool slotCountMismatch = allocation && allocation.getResult().getType().getCount() != *slotCount;
    if (!allocation || slotCountMismatch) {
        return std::nullopt;
    }
    std::optional<std::uint64_t> slotBytes =
        getStaticMultiBufferSlotBytes(allocation.getResult().getType().getSlotType());
    if (!slotBytes) {
        return std::nullopt;
    }

    SmallVector<std::int64_t, 2> slotAddresses;
    auto addresses = allocation->getAttrOfType<DenseI64ArrayAttr>(kPtoMultiBufferAddrsAttrName);
    const bool plannedAddressCountMismatch = addresses && addresses.size() != *slotCount;
    if (plannedAddressCountMismatch) {
        return std::nullopt;
    }
    if (addresses) {
        slotAddresses.assign(addresses.asArrayRef().begin(), addresses.asArrayRef().end());
    } else {
        std::optional<std::int64_t> base = getConstantIntValue(allocation.getAddr());
        const bool slotSizeExceedsSignedRange =
            *slotBytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        if (!base || *base < 0 || slotSizeExceedsSignedRange) {
            return std::nullopt;
        }
        const std::int64_t stride = static_cast<std::int64_t>(*slotBytes);
        slotAddresses.reserve(*slotCount);
        for (unsigned index = 0; index < *slotCount; ++index) {
            const std::int64_t signedIndex = static_cast<std::int64_t>(index);
            const bool addressOverflows =
                index != 0 && stride > (std::numeric_limits<std::int64_t>::max() - *base) / signedIndex;
            if (addressOverflows) {
                return std::nullopt;
            }
            slotAddresses.push_back(*base + signedIndex * stride);
        }
    }

    SmallVector<SyncByteInterval, 2> intervals;
    intervals.reserve(slotAddresses.size());
    for (std::int64_t address : slotAddresses) {
        const bool invalidAddress =
            address < 0 || static_cast<std::uint64_t>(address) > std::numeric_limits<std::uint64_t>::max() - *slotBytes;
        if (invalidAddress) {
            return std::nullopt;
        }
        intervals.push_back({static_cast<std::uint64_t>(address), *slotBytes});
    }
    return intervals;
}

} // namespace

namespace mlir::pto::protocol_sync {

class StructuredSyncIRConstruction {
public:
    StructuredSyncIRConstruction(
        StructuredSyncIR& schedule, const SyncSemanticContext& context, ProtocolSyncStatistics* statistics)
        : schedule(schedule), extractor(context, statistics), statistics(statistics)
    {}

    LogicalResult build();

private:
    StructuredSyncIR& schedule;
    SyncSemanticExtractor extractor;
    ProtocolSyncStatistics* statistics;
    llvm::DenseMap<Value, llvm::DenseMap<unsigned, SyncStorageFamilyId>> familyIndex;

    SyncProgramPointId addPoint(
        SyncProgramPointKind kind, SyncRegionId region, SyncPhaseId phase = kInvalidSyncId,
        SyncSemanticActionId action = kInvalidSyncId);
    SyncRegionId addRegion(
        SyncRegionId parent, SyncRegionKind kind, SyncCardinality cardinality, Operation* operation, unsigned arm,
        ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops);
    void finishRegion(SyncRegionId region);
    void buildBlock(Block& block, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops);
    bool buildSingleBlockRegion(
        Region& body, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops,
        Operation* owner);
    void buildOperation(
        Operation* operation, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops);
    void buildLoop(
        Operation* operation, Region& body, SyncRegionId parent, SyncCardinality cardinality,
        ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops);
    void buildChoice(
        scf::IfOp operation, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops);
    void buildNestedRegion(
        Operation* operation, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops);
    void addSummary(
        Operation* operation, SyncRegionId region, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops);
    std::optional<SyncSlotExpression> bindSlotExpression(
        const std::optional<SyncSlotExpression>& slot, ArrayRef<SyncRegionId> loops) const;
    SyncStorageFamilyId getOrCreateStorageFamily(
        const SyncMemoryEffect& effect, const std::optional<SyncSlotExpression>& slot);
};

std::optional<SyncSlotExpression> StructuredSyncIRConstruction::bindSlotExpression(
    const std::optional<SyncSlotExpression>& slot, ArrayRef<SyncRegionId> loops) const
{
    if (!slot) {
        return std::nullopt;
    }
    SyncSlotExpression result = *slot;
    if (result.kind != SyncSlotExpressionKind::AffineModulo) {
        return result;
    }
    for (SyncRegionId loopId : llvm::reverse(loops)) {
        const SyncRegion& region = schedule.regions[loopId];
        if (auto loop = dyn_cast_or_null<scf::ForOp>(region.operation)) {
            const bool isSelectedLoop = loop.getInductionVar() == result.induction;
            if (isSelectedLoop) {
                result.loop = loopId;
                return result;
            }
        }
    }
    result.kind = SyncSlotExpressionKind::Unknown;
    result.loop = kInvalidSyncId;
    return result;
}

SyncStorageFamilyId StructuredSyncIRConstruction::getOrCreateStorageFamily(
    const SyncMemoryEffect& effect, const std::optional<SyncSlotExpression>& slot)
{
    const SyncStorageProvenance& provenance = effect.provenance;
    const unsigned space = static_cast<unsigned>(provenance.space);
    const std::optional<std::uint32_t> rootSlotCount = getAuthoritativeSlotCount(provenance);
    std::optional<std::uint32_t> selectorSlotCount;
    if (slot && slot->depth != 0) {
        selectorSlotCount = slot->depth;
    }
    std::optional<SmallVector<SyncByteInterval, 2>> completeSlots =
        getCompletePhysicalSlots(provenance, rootSlotCount);
    SyncStorageFamily* existingFamily = nullptr;
    auto rootEntry = familyIndex.find(provenance.root);
    if (rootEntry != familyIndex.end()) {
        auto spaceEntry = rootEntry->second.find(space);
        if (spaceEntry != rootEntry->second.end()) {
            existingFamily = &schedule.storageFamilies[spaceEntry->second];
        }
    }
    if (existingFamily != nullptr) {
        SyncStorageFamily& family = *existingFamily;
        family.physical = family.physical || provenance.physical;
        family.unknownRange = family.unknownRange || provenance.unknownRange;
        family.aliasesUnknownRange = family.aliasesUnknownRange || provenance.aliasesUnknownRange;
        if (completeSlots) {
            if (family.physicalSlotsComplete && !sameIntervals(family.intervals, *completeSlots)) {
                family.capacityConflict = true;
            } else {
                family.intervals = *completeSlots;
                family.physicalSlotsComplete = true;
            }
        } else if (!family.physicalSlotsComplete) {
            for (const SyncByteInterval& interval : provenance.intervals) {
                const bool alreadyRecorded = llvm::any_of(family.intervals, [&](const SyncByteInterval& current) {
                    return current.begin == interval.begin && current.size == interval.size;
                });
                if (!alreadyRecorded) {
                    family.intervals.push_back(interval);
                }
            }
        }
        const bool authoritativeCapacityConflict = capacitiesConflict(rootSlotCount, family.slotCount);
        const bool selectorCapacityConflict = capacitiesConflict(rootSlotCount, selectorSlotCount);
        if (authoritativeCapacityConflict || selectorCapacityConflict) {
            family.capacityConflict = true;
            family.slotCount.reset();
        } else if (rootSlotCount && !family.capacityConflict) {
            family.slotCount = *rootSlotCount;
        }
        return family.id;
    }

    SyncStorageFamily family;
    family.id = schedule.storageFamilies.size();
    family.root = provenance.root;
    family.space = provenance.space;
    family.intervals = completeSlots ? *completeSlots : provenance.intervals;
    family.physical = provenance.physical;
    family.unknownRange = provenance.unknownRange;
    family.aliasesUnknownRange = provenance.aliasesUnknownRange;
    family.physicalSlotsComplete = completeSlots.has_value();
    const bool selectorCapacityConflict = capacitiesConflict(rootSlotCount, selectorSlotCount);
    if (selectorCapacityConflict) {
        family.capacityConflict = true;
    } else if (rootSlotCount) {
        family.slotCount = *rootSlotCount;
    }
    if (effect.visibility == SyncVisibilityClass::Local) {
        family.role = SyncStorageRole::LocalBuffer;
    } else if (effect.visibility == SyncVisibilityClass::Global) {
        family.role = SyncStorageRole::GlobalBuffer;
    }
    schedule.storageFamilies.push_back(std::move(family));
    familyIndex[provenance.root][space] = schedule.storageFamilies.back().id;
    return schedule.storageFamilies.back().id;
}

SyncProgramPointId StructuredSyncIRConstruction::addPoint(
    SyncProgramPointKind kind, SyncRegionId region, SyncPhaseId phase, SyncSemanticActionId action)
{
    SyncProgramPointId id = schedule.points.size();
    schedule.points.push_back({id, kind, region, phase, action});
    return id;
}

SyncRegionId StructuredSyncIRConstruction::addRegion(
    SyncRegionId parent, SyncRegionKind kind, SyncCardinality cardinality, Operation* operation, unsigned arm,
    ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops)
{
    SyncRegionId id = schedule.regions.size();
    SyncProgramPointId entry = addPoint(SyncProgramPointKind::RegionEntry, id);
    SyncRegion region;
    region.id = id;
    region.parent = parent;
    region.kind = kind;
    region.cardinality = cardinality;
    region.operation = operation;
    region.arm = arm;
    region.guard.assign(guard.begin(), guard.end());
    region.iterationDomain.loops.assign(loops.begin(), loops.end());
    region.entry = entry;
    schedule.regions.push_back(std::move(region));
    if (parent != kInvalidSyncId) {
        SyncRegion& owner = schedule.regions[parent];
        owner.elements.push_back(
            {SyncRegionElement::Kind::ChildRegion, static_cast<unsigned>(owner.elements.size()), kInvalidSyncId, id});
    }
    return id;
}

void StructuredSyncIRConstruction::finishRegion(SyncRegionId region)
{
    schedule.regions[region].exit = addPoint(SyncProgramPointKind::RegionExit, region);
}

bool StructuredSyncIRConstruction::buildSingleBlockRegion(
    Region& body, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops, Operation* owner)
{
    if (body.empty()) {
        return true;
    }
    if (!llvm::hasSingleElement(body)) {
        schedule.failures.push_back(
            {SyncFailureReason::UnsupportedCFG, owner,
             "ProtocolSync initially requires structured single-block regions"});
        return false;
    }
    buildBlock(body.front(), parent, guard, loops);
    return true;
}

void StructuredSyncIRConstruction::buildLoop(
    Operation* operation, Region& body, SyncRegionId parent, SyncCardinality cardinality,
    ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops)
{
    SyncRegionId loop = addRegion(parent, SyncRegionKind::Loop, cardinality, operation, 0, guard, loops);
    SmallVector<SyncRegionId, 2> nestedLoops(loops.begin(), loops.end());
    nestedLoops.push_back(loop);
    (void)buildSingleBlockRegion(body, loop, guard, nestedLoops, operation);
    finishRegion(loop);
}

void StructuredSyncIRConstruction::buildChoice(
    scf::IfOp operation, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops)
{
    SyncRegionId choice =
        addRegion(parent, SyncRegionKind::Choice, SyncCardinality::ExactlyOnce, operation, 0, guard, loops);
    for (unsigned arm = 0; arm < 2; ++arm) {
        SmallVector<SyncControlAtom, 2> armGuard(guard.begin(), guard.end());
        armGuard.push_back({choice, arm});
        SyncRegionId alternative =
            addRegion(choice, SyncRegionKind::Alternative, SyncCardinality::ZeroOrOne, operation, arm, armGuard, loops);
        Region& region = arm == 0 ? operation.getThenRegion() : operation.getElseRegion();
        (void)buildSingleBlockRegion(region, alternative, armGuard, loops, operation);
        finishRegion(alternative);
    }
    finishRegion(choice);
}

void StructuredSyncIRConstruction::buildNestedRegion(
    Operation* operation, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops)
{
    SyncRegionKind kind =
        isa<SectionCubeOp, SectionVectorOp>(operation) ? SyncRegionKind::PhysicalSection : SyncRegionKind::Sequence;
    if (kind == SyncRegionKind::Sequence) {
        schedule.failures.push_back(
            {SyncFailureReason::UnsupportedRegion, operation, "region operation is preserved transparently"});
    }
    for (Region& body : operation->getRegions()) {
        SyncRegionId region = addRegion(parent, kind, SyncCardinality::ExactlyOnce, operation, 0, guard, loops);
        (void)buildSingleBlockRegion(body, region, guard, loops, operation);
        finishRegion(region);
    }
}

void StructuredSyncIRConstruction::addSummary(
    Operation* operation, SyncRegionId region, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops)
{
    const auto start = std::chrono::steady_clock::now();
    SyncOpSummary summary = extractor.summarize(operation);
    if (statistics) {
        statistics->semanticExtractionUs +=
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    }
    SyncSummaryId summaryId = schedule.summaries.size();
    if (!summary.isSupported()) {
        schedule.failures.push_back({summary.failure, operation, summary.failureDetail});
        if (statistics) {
            ++statistics->rejectedOperations;
        }
    }
    schedule.summaries.push_back(std::move(summary));
    const SyncOpSummary& stored = schedule.summaries.back();
    const bool fixedProtocolWithoutPhase = stored.phases.empty() && !stored.suppliedProtocols.empty();
    const bool needsOrderedSemanticAction =
        stored.queue.has_value() || (stored.isSupported() && fixedProtocolWithoutPhase);
    if (needsOrderedSemanticAction) {
        SyncSemanticActionId actionId = schedule.semanticActions.size();
        SyncSemanticAction action;
        action.id = actionId;
        action.summary = summaryId;
        action.region = region;
        action.operation = operation;
        action.guard.assign(guard.begin(), guard.end());
        action.iterationDomain.loops.assign(loops.begin(), loops.end());
        action.before = addPoint(
            SyncProgramPointKind::SemanticActionBefore, region, kInvalidSyncId, actionId);
        action.after = addPoint(
            SyncProgramPointKind::SemanticActionAfter, region, kInvalidSyncId, actionId);
        schedule.semanticActions.push_back(std::move(action));
        SyncRegion& owner = schedule.regions[region];
        owner.elements.push_back(
            {SyncRegionElement::Kind::SemanticAction, static_cast<unsigned>(owner.elements.size()),
             kInvalidSyncId, kInvalidSyncId, actionId});
    }
    for (const SyncPhysicalPhase& physical : stored.phases) {
        SyncPhaseId phaseId = schedule.phases.size();
        SyncPhase phase;
        phase.id = phaseId;
        phase.summary = summaryId;
        phase.region = region;
        phase.operation = operation;
        if (stored.provider == SyncSummaryProvider::Macro) {
            phase.macroPhase = physical.phaseIndex;
        }
        phase.core = physical.core;
        phase.pipe = physical.pipe;
        phase.completion = physical.completion.kind;
        phase.guard.assign(guard.begin(), guard.end());
        phase.iterationDomain.loops.assign(loops.begin(), loops.end());
        phase.before = addPoint(SyncProgramPointKind::PhaseBefore, region, phaseId);
        for (const SyncMemoryEffect& effect : physical.effects) {
            SyncAccessId accessId = schedule.accesses.size();
            std::optional<SyncSlotExpression> slot = bindSlotExpression(effect.slot, loops);
            SyncStorageFamilyId family = getOrCreateStorageFamily(effect, slot);
            schedule.accesses.push_back(
                {accessId, phaseId, family, effect.value, effect.provenance, effect.mode, slot, effect.visibility});
            phase.accesses.push_back(accessId);
        }
        phase.after = addPoint(SyncProgramPointKind::PhaseAfter, region, phaseId);
        schedule.phases.push_back(std::move(phase));
        SyncRegion& owner = schedule.regions[region];
        owner.elements.push_back(
            {SyncRegionElement::Kind::Phase, static_cast<unsigned>(owner.elements.size()), phaseId, kInvalidSyncId});
    }
}

void StructuredSyncIRConstruction::buildOperation(
    Operation* operation, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops)
{
    if (statistics) {
        ++statistics->operationsVisited;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        SyncCardinality cardinality =
            hasStaticallyNonEmptyTripCount(loop) ? SyncCardinality::OneOrMore : SyncCardinality::ZeroOrMore;
        buildLoop(operation, loop.getRegion(), parent, cardinality, guard, loops);
        return;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
        schedule.failures.push_back(
            {SyncFailureReason::UnsupportedRegion, operation,
             "scf.while requires distinct condition and body execution semantics"});
        return;
    }
    if (auto choice = dyn_cast<scf::IfOp>(operation)) {
        buildChoice(choice, parent, guard, loops);
        return;
    }
    const bool hasNestedRegions = operation->getNumRegions() != 0;
    if (hasNestedRegions) {
        buildNestedRegion(operation, parent, guard, loops);
        return;
    }
    addSummary(operation, parent, guard, loops);
}

void StructuredSyncIRConstruction::buildBlock(
    Block& block, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops)
{
    for (Operation& operation : block) {
        buildOperation(&operation, parent, guard, loops);
    }
}

LogicalResult StructuredSyncIRConstruction::build()
{
    func::FuncOp function = schedule.getFunction();
    const bool isDeclaration = function.isDeclaration();
    if (isDeclaration) {
        return schedule.freeze();
    }
    SyncRegionId root =
        addRegion(kInvalidSyncId, SyncRegionKind::Function, SyncCardinality::ExactlyOnce, function, 0, {}, {});
    if (!llvm::hasSingleElement(function.getBody())) {
        schedule.failures.push_back(
            {SyncFailureReason::UnsupportedCFG, function, "ProtocolSync initially requires single-block functions"});
    } else {
        buildBlock(function.getBody().front(), root, {}, {});
    }
    finishRegion(root);
    return schedule.freeze();
}

} // namespace mlir::pto::protocol_sync

LogicalResult StructuredSyncIRBuilder::build(func::FuncOp function, StructuredSyncIR& schedule) const
{
    const bool wrongFunction = schedule.getFunction() != function;
    const bool alreadyFrozen = schedule.isFrozen();
    if (wrongFunction || alreadyFrozen) {
        return failure();
    }
    StructuredSyncIRConstruction builder(schedule, context, statistics);
    return builder.build();
}
