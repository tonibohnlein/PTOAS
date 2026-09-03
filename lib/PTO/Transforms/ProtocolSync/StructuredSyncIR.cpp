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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"

#include <chrono>

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

    SyncProgramPointId addPoint(SyncProgramPointKind kind, SyncRegionId region, SyncPhaseId phase = kInvalidSyncId);
    SyncRegionId addRegion(
        SyncRegionId parent, SyncRegionKind kind, SyncCardinality cardinality, Operation* operation, unsigned arm,
        ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops);
    void finishRegion(SyncRegionId region);
    void buildBlock(Block& block, SyncRegionId parent, ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops);
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
};

SyncProgramPointId StructuredSyncIRConstruction::addPoint(
    SyncProgramPointKind kind, SyncRegionId region, SyncPhaseId phase)
{
    SyncProgramPointId id = schedule.points.size();
    schedule.points.push_back({id, kind, region, phase});
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

void StructuredSyncIRConstruction::buildLoop(
    Operation* operation, Region& body, SyncRegionId parent, SyncCardinality cardinality,
    ArrayRef<SyncControlAtom> guard, ArrayRef<SyncRegionId> loops)
{
    SyncRegionId loop = addRegion(parent, SyncRegionKind::Loop, cardinality, operation, 0, guard, loops);
    SmallVector<SyncRegionId, 2> nestedLoops(loops.begin(), loops.end());
    nestedLoops.push_back(loop);
    for (Block& block : body) {
        buildBlock(block, loop, guard, nestedLoops);
    }
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
        for (Block& block : region) {
            buildBlock(block, alternative, armGuard, loops);
        }
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
        for (Block& block : body) {
            buildBlock(block, region, guard, loops);
        }
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
            schedule.accesses.push_back(
                {accessId, phaseId, effect.value, effect.provenance, effect.mode, effect.slot, effect.visibility});
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
    for (Block& block : function.getBody()) {
        buildBlock(block, root, {}, {});
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
