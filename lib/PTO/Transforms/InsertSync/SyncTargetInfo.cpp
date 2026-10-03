// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Recover target/core identity once at the shared input boundary. Opcode
// effects and native order are borrowed unchanged from the shared input.
// Original barriers are captured as additional prerequisites, not admission checks.
#include "PTO/Transforms/InsertSync/SyncTargetInfo.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/MemoryDependentAnalyzer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include <limits>
#include <algorithm>
namespace mlir::pto {
namespace {
bool hasAtMostOneOccurrence(Operation *operation, func::FuncOp function)
{
    if (!llvm::hasSingleElement(function.getBody())) { return false; }
    // Joint stream cardinality is checked separately. Only single-shot if
    // regions may intervene; loops/unknown region operations are not admitted.
    for (auto *owner = operation->getParentOp(); owner && owner != function; owner = owner->getParentOp()) {
        if (!isa<scf::IfOp>(owner)) { return false; }
    }
    return function->isProperAncestor(operation);
}
FailureOr<SyncActivationWitness> recoverActivation(func::FuncOp function, scf::IfOp participation)
{
    const auto condition = participation.getCondition();
    if (auto argument = dyn_cast<BlockArgument>(condition)) {
        if (argument.getOwner() != &function.front() || !argument.getType().isInteger(1)) { return failure(); }
        SyncActivationWitness witness;
        witness.origin = SyncActivationOrigin::Argument; witness.argument = argument;
        return witness;
    }
    auto comparison = condition.getDefiningOp<arith::CmpIOp>();
    DominanceInfo dominance(function);
    if (!comparison || comparison->getBlock() != &function.front() || !isPure(comparison) ||
        !dominance.properlyDominates(condition, participation) ||
        (comparison.getPredicate() != arith::CmpIPredicate::eq &&
         comparison.getPredicate() != arith::CmpIPredicate::ne &&
         comparison.getPredicate() != arith::CmpIPredicate::slt &&
         comparison.getPredicate() != arith::CmpIPredicate::sle &&
         comparison.getPredicate() != arith::CmpIPredicate::sgt &&
         comparison.getPredicate() != arith::CmpIPredicate::sge)) {
        return failure();
    }
    auto argument = dyn_cast<BlockArgument>(comparison.getLhs());
    Value literal = comparison.getRhs();
    bool argumentFirst = true;
    if (!argument) {
        argument = dyn_cast<BlockArgument>(comparison.getRhs());
        literal = comparison.getLhs(); argumentFirst = false;
    }
    IntegerAttr constant;
    auto* source = literal.getDefiningOp();
    if (!argument || argument.getOwner() != &function.front() || !isa<IndexType>(argument.getType()) ||
        !source || source->getBlock() != &function.front() || !isPure(source) ||
        !dominance.properlyDominates(literal, comparison) || !matchPattern(literal, m_Constant(&constant)) ||
        !isa<IndexType>(constant.getType()) || !constant.getValue().isSignedIntN(64)) { return failure(); }
    SyncActivationWitness witness;
    witness.origin = SyncActivationOrigin::ComparisonResult; witness.argument = argument;
    witness.comparison = comparison; witness.constantSource = source; witness.constant = constant;
    witness.predicate = static_cast<int64_t>(comparison.getPredicate()); witness.argumentFirst = argumentFirst;
    return witness;
}
SyncParticipation recoverParticipation(func::FuncOp function)
{
    if (!llvm::hasSingleElement(function.getBody())) { return {}; }
    scf::IfOp common;
    for (auto& operation : function.front()) {
        auto candidate = dyn_cast<scf::IfOp>(operation);
        if (!candidate || !llvm::hasSingleElement(candidate.getThenRegion()) ||
            candidate.getThenRegion().front().getOps<scf::ForOp>().empty()) { continue; }
        if (common) { return {}; }
        common = candidate;
    }
    if (!common || common.getNumResults() ||
        (!common.getElseRegion().empty() && !llvm::hasSingleElement(common.getElseRegion()))) { return {}; }
    auto witness = recoverActivation(function, common);
    if (failed(witness)) { return {}; }
    return {common, common.getCondition(), *witness};
}
bool captureBarrierLexicalChain(SyncOriginalBarrierChain& result, const MemoryDependentAnalyzer& memory)
{
    const auto& computes = result.computes;
    for (unsigned index = 1; index < computes.size(); ++index) {
        SyncBarrierLexicalWitness witness;
        witness.source = computes[index - 1]; witness.consumer = computes[index];
        if (memory.DepBetween(witness.source->defVec, witness.consumer->useVec, witness.pairs)) {
            witness.kind = SyncBarrierHazard::ReadAfterWrite;
        } else if (memory.DepBetween(witness.source->useVec, witness.consumer->defVec, witness.pairs)) {
            witness.kind = SyncBarrierHazard::WriteAfterRead;
        } else if (memory.DepBetween(witness.source->defVec, witness.consumer->defVec, witness.pairs)) {
            witness.kind = SyncBarrierHazard::WriteAfterWrite;
        } else { return false; }
        result.lexical.push_back(std::move(witness));
    }
    return true;
}
SyncOriginalBarrierChain recoverOriginalBarrierChain(func::FuncOp function,
    ArrayRef<SyncPhaseTarget> records, StringRef arch, const MemoryDependentAnalyzer& memory,
    const std::shared_ptr<const SyncParticipation>& common)
{
    // All MTE2 occurrences belong to one fixed P/Q loop, optionally under a
    // common immutable guard, plus one optional singleton prefix. The source
    // barrier before P drains P_previous and gates P_current, independently of
    // any generated flags. Original shared READY/RELEASE hazards make this
    // stronger source-command edge redundant in the selected modeled closure.
    if (records.size() < 3 ||
        getSyncBarrierFact(arch, SyncPhysicalCore::AIV, PIPE::PIPE_MTE2).availability !=
            SyncMechanismAvailability::Documented) { return {}; }
    unsigned first = 0;
    while (first < records.size() &&
           !records[first].phase->elementOp->getParentOfType<scf::ForOp>()) { ++first; }
    if (first > 1 || first + 2 >= records.size()) { return {}; }
    const auto* producer = records[first].phase;
    auto loop = producer->elementOp->getParentOfType<scf::ForOp>();
    auto barrier = dyn_cast_or_null<BarrierOp>(producer->elementOp->getPrevNode());
    if (!loop || !barrier || !llvm::hasSingleElement(function.getBody())) { return {}; }
    auto frameResult = recoverSyncLoopFrames(function, loop);
    if (failed(frameResult) || frameResult->empty()) { return {}; }
    auto outer = cast<scf::ForOp>(frameResult->front());
    SmallVector<const CompoundInstanceElement*> computes;
    for (unsigned index = first + 1; index + 1 < records.size(); ++index) {
        const auto& record = records[index];
        if (record.phase->elementOp->getBlock() != loop.getBody() ||
            record.phase->kPipeValue != PipelineType::PIPE_V || record.anchorPhaseCount != 1 ||
            record.phase->macroOpInstanceId >= 0 || !record.hiddenEvents.empty() ||
            record.context.core != SyncPhysicalCore::AIV ||
            !(computes.empty() ? producer : computes.back())->elementOp->isBeforeInBlock(record.phase->elementOp)) {
            return {};
        }
        computes.push_back(record.phase);
    }
    const auto& exitRecord = records.back();
    if (computes.empty() || !hasAtMostOneOccurrence(exitRecord.phase->elementOp, function) ||
        exitRecord.anchorPhaseCount != 1 || exitRecord.phase->macroOpInstanceId >= 0 ||
        !exitRecord.hiddenEvents.empty() || exitRecord.context.core != SyncPhysicalCore::AIV) { return {}; }
    const auto* consumer = computes.front();
    const auto* terminal = records.back().phase;
    auto exitGuard = terminal->elementOp->getParentOfType<scf::IfOp>();
    if (!exitGuard || !syncFramesNonempty(exitGuard.getCondition(), *frameResult) ||
        exitGuard->getBlock() != outer->getBlock() || !outer->isBeforeInBlock(exitGuard) ||
        terminal->elementOp->getBlock() != &exitGuard.getThenRegion().front()) { return {}; }
    scf::IfOp participation;
    Operation* invocation = outer;
    if (outer->getBlock() != &function.front()) {
        participation = dyn_cast<scf::IfOp>(outer->getParentOp());
        if (!participation || participation->getBlock() != &function.front() || participation.getNumResults() ||
            !llvm::hasSingleElement(participation.getThenRegion()) ||
            outer->getBlock() != &participation.getThenRegion().front() ||
            (!participation.getElseRegion().empty() &&
             !participation.getElseRegion().front().without_terminator().empty())) { return {}; }
        if (!common || common->owner != participation || common->condition != participation.getCondition()) {
            return {};
        }
        invocation = participation;
    }
    IntegerAttr lower, step;
    if (loop.getNumRegionIterArgs() ||
        producer->elementOp->getBlock() != loop.getBody() || consumer->elementOp->getBlock() != loop.getBody() ||
        !producer->elementOp->isBeforeInBlock(consumer->elementOp) ||
        producer->kPipeValue != PipelineType::PIPE_MTE2 ||
        consumer->kPipeValue != PipelineType::PIPE_V || records.back().phase->kPipeValue != PipelineType::PIPE_MTE3 ||
        records[first].anchorPhaseCount != 1 || producer->macroOpInstanceId >= 0 ||
        !records[first].hiddenEvents.empty() ||
        barrier.getPipe().getPipe() != PIPE::PIPE_MTE2 ||
        records[first].context.core != SyncPhysicalCore::AIV ||
        records[first + 1].context.core != SyncPhysicalCore::AIV ||
        !matchPattern(loop.getLowerBound(), m_Constant(&lower)) || !lower.getValue().isZero() ||
        !matchPattern(loop.getStep(), m_Constant(&step)) || !step.getValue().isOne()) { return {}; }
    if (first) {
        const auto& prefix = records.front();
        if (prefix.phase->elementOp->getBlock() != &function.front() ||
            !prefix.phase->elementOp->isBeforeInBlock(invocation) ||
            prefix.phase->kPipeValue != PipelineType::PIPE_MTE2 ||
            prefix.anchorPhaseCount != 1 || prefix.phase->macroOpInstanceId >= 0 || !prefix.hiddenEvents.empty() ||
            prefix.context.core != SyncPhysicalCore::AIV) { return {}; }
    }
    // Only this admitted candidate pays the original-command scan and AA cost.
    unsigned originalBarriers = 0;
    function.walk([&](BarrierOp) { ++originalBarriers; });
    if (originalBarriers != 1) { return {}; }
    DepBaseMemInfoPairVec ready, release;
    if (!memory.DepBetween(producer->defVec, consumer->useVec, ready) ||
        !memory.DepBetween(computes.back()->useVec, producer->defVec, release)) { return {}; }
    SyncOriginalBarrierChain result;
    result.barrier = barrier; result.loop = loop; result.frames = *frameResult;
    result.producer = producer; result.consumer = consumer;
    result.releaseProducer = computes.back(); result.computes = computes;
    result.ready = std::move(ready); result.release = std::move(release);
    if (!memory.DepBetween(computes.back()->defVec, terminal->useVec, result.finalReads)) { return {}; }
    result.requirementDomain = SyncBarrierRequirementDomain::SharedUnion;
    if (!captureBarrierLexicalChain(result, memory)) { return {}; }
    if (first) {
        result.prefix = records.front().phase;
        if (!memory.DepBetween(result.prefix->defVec, producer->defVec, result.prefixWrites)) { return {}; }
    }
    if (participation) { result.participationContext = common; }
    return result;
}
} // namespace
FailureOr<SmallVector<Operation*>> recoverSyncLoopFrames(func::FuncOp function, Operation* innermost)
{
    SmallVector<Operation*> frames;
    auto inner = dyn_cast_or_null<scf::ForOp>(innermost);
    for (auto current = inner; current; current = current->getParentOfType<scf::ForOp>()) {
        IntegerAttr lower, step;
        if (current.getNumRegionIterArgs() || !isa<IndexType>(current.getInductionVar().getType()) ||
            !matchPattern(current.getLowerBound(), m_Constant(&lower)) || !lower.getValue().isZero() ||
            !matchPattern(current.getStep(), m_Constant(&step)) || !step.getValue().isOne()) { return failure(); }
        auto bound = dyn_cast<BlockArgument>(current.getUpperBound());
        if ((!bound || bound.getOwner() != &function.front()) &&
            !matchPattern(current.getUpperBound(), m_Constant())) { return failure(); }
        if (!frames.empty()) {
            for (auto& operation : current.getBody()->without_terminator()) {
                if (&operation != frames.back() && (operation.getNumRegions() ||
                    isa<OpPipeInterface>(operation) || !isPure(&operation))) { return failure(); }
            }
        }
        frames.push_back(current);
    }
    std::reverse(frames.begin(), frames.end());
    return frames;
}
bool syncFramesNonempty(Value condition, ArrayRef<Operation*> frames, unsigned* visitedValues)
{
    if (visitedValues) { *visitedValues = 0; }
    SmallVector<Value> leaves{condition};
    llvm::DenseSet<Value> visited;
    SmallVector<bool> covered(frames.size(), false);
    // Canonicalization may erase a true fixed-bound conjunct. Its presence
    // fact remains proved by the ORIGINAL typed bound, not a synthetic premise.
    for (unsigned axis = 0; axis < frames.size(); ++axis) {
        auto frame = dyn_cast<scf::ForOp>(frames[axis]);
        IntegerAttr literal;
        if (!frame) { return false; }
        covered[axis] = matchPattern(frame.getUpperBound(), m_Constant(&literal)) &&
                        literal.getValue().isStrictlyPositive();
    }
    while (!leaves.empty()) {
        auto value = leaves.pop_back_val();
        // A conjunction is an original SSA DAG, not a tree of operand paths.
        if (!visited.insert(value).second) { continue; }
        if (visitedValues) { ++*visitedValues; }
        IntegerAttr literal;
        if (matchPattern(value, m_Constant(&literal)) && literal.getValue().isOne()) { continue; }
        if (auto conjunction = value.getDefiningOp<arith::AndIOp>()) {
            leaves.push_back(conjunction.getLhs()); leaves.push_back(conjunction.getRhs()); continue;
        }
        auto compare = value.getDefiningOp<arith::CmpIOp>();
        if (!compare) { return false; }
        bool found = false;
        for (unsigned axis = 0; axis < frames.size(); ++axis) {
            auto frame = dyn_cast<scf::ForOp>(frames[axis]);
            if (!frame) { return false; }
            const bool forward = compare.getPredicate() == arith::CmpIPredicate::sgt &&
                                 compare.getLhs() == frame.getUpperBound();
            const bool reversed = compare.getPredicate() == arith::CmpIPredicate::slt &&
                                  compare.getRhs() == frame.getUpperBound();
            IntegerAttr zero;
            if ((forward || reversed) && matchPattern(forward ? compare.getRhs() : compare.getLhs(),
                    m_Constant(&zero)) && zero.getValue().isZero()) { covered[axis] = true; found = true; }
        }
        if (!found) { return false; }
    }
    return !frames.empty() && llvm::all_of(covered, [](bool present) { return present; });
}
std::optional<SyncParticipationArm> SyncParticipation::armFor(Operation* child) const
{
    auto choice = dyn_cast_or_null<scf::IfOp>(owner);
    if (!choice || !child || child->getParentOp() != owner) { return std::nullopt; }
    auto* region = child->getParentRegion();
    if (region == &choice.getThenRegion()) { return SyncParticipationArm{region, true}; }
    if (region == &choice.getElseRegion()) { return SyncParticipationArm{region, false}; }
    return std::nullopt;
}
bool SyncParticipation::thenOnly() const
{
    auto choice = dyn_cast_or_null<scf::IfOp>(owner);
    return choice && (choice.getElseRegion().empty() ||
        choice.getElseRegion().front().without_terminator().empty());
}
SyncCoreContext recoverSyncCoreContext(Operation* anchor)
{
    SyncCoreContext result;
    result.core = recoverSyncPhysicalCore(anchor);
    bool insideFunction = true;
    for (Operation* op = anchor; op; op = op->getParentOp()) {
        if (isa<func::FuncOp>(op)) {
            insideFunction = false;
        }
        if (insideFunction && op->getParentRegion()) {
            result.regions.push_back(op->getParentRegion());
        }
    }
    return result;
}
void SyncTargetInfo::reset()
{
    originalChain = {};
    commonParticipation.reset();
    records.clear();
    sourceIds.clear();
    function = {};
    arch.clear();
    dominance.invalidate();
}
void SyncTargetInfo::build(func::FuncOp source, ArrayRef<const CompoundInstanceElement*> phases,
                           const MemoryDependentAnalyzer& memory)
{
    function = source;
    originalChain = {};
    commonParticipation.reset();
    records.clear();
    sourceIds.clear();
    int64_t next = 0;
    function->walk<WalkOrder::PostOrder>([&](Operation* op) { sourceIds[op] = next++; });
    dominance.invalidate();
    commonParticipation = std::make_shared<const SyncParticipation>(recoverParticipation(function));
    arch = getTargetArch(function.operator->()) == PTOArch::A5 ? "a5" : "a3";
    DenseMap<Operation*, unsigned> counts;
    for (const auto* phase : phases) {
        ++counts[phase->elementOp];
    }
    for (const auto* phase : phases) {
        SyncPhaseTarget target;
        target.phase = phase;
        target.context = recoverSyncCoreContext(phase->elementOp);
        target.anchorPhaseCount = counts.lookup(phase->elementOp);
        target.before = phase->elementOp;
        target.after = phase->elementOp->getNextNode();
        if (phase->macroOpInstanceId == 0) {
            if (auto model = getSyncMacroModel(phase->elementOp)) {
                target.hiddenEvents = model->hiddenEvents;
            }
        }
        records.push_back(std::move(target));
    }
    originalChain = recoverOriginalBarrierChain(function, records, arch, memory, commonParticipation);
    // Capture only an original straight-line invocation. A local drain gates
    // its next same-pipe payload; all earlier completions follow by C order.
    originalFiniteDrains.clear();
    bool finite = llvm::hasSingleElement(function.getBody());
    Operation* priorAnchor = nullptr;
    for (const auto& record : records) {
        finite &= record.phase->elementOp->getBlock() == &function.front() &&
                  record.anchorPhaseCount == 1 && record.phase->macroOpInstanceId < 0 &&
                  record.hiddenEvents.empty() && record.context.core == SyncPhysicalCore::AIV;
        if (finite && priorAnchor) { finite &= priorAnchor->isBeforeInBlock(record.phase->elementOp); }
        priorAnchor = record.phase->elementOp;
    }
    DenseMap<PipelineType, const SyncPhaseTarget*> prior;
    if (finite) {
        for (const auto& record : records) {
            const auto pipe = record.phase->kPipeValue;
            auto previous = prior.lookup(pipe);
            if (previous) {
                BarrierOp found;
                for (auto* operation = previous->phase->elementOp->getNextNode();
                     operation && operation != record.phase->elementOp; operation = operation->getNextNode()) {
                    if (auto barrier = dyn_cast<BarrierOp>(operation)) {
                        if (barrier.getPipe().getPipe() == static_cast<PIPE>(pipe)) { found = barrier; }
                    }
                }
                if (found) {
                    originalFiniteDrains.push_back({found, previous->phase, record.phase, record.context.core});
                }
            }
            prior[pipe] = &record;
        }
    }

}
std::optional<int64_t> SyncTargetInfo::sourceIdentity(Operation* op) const
{
    auto found = sourceIds.find(op);
    return found == sourceIds.end() ? std::nullopt : std::optional<int64_t>(found->second);
}
LogicalResult SyncTargetInfo::retainSourceIds(func::FuncOp clone, const IRMapping& mapping) const
{
    for (const auto& entry : sourceIds) {
        Operation* retained = entry.first == function.operator->() ? clone.getOperation() :
                                                                      mapping.lookupOrNull(entry.first);
        if (!retained) {
            return failure();
        }
        retained->setAttr("pto.frontier.source",
                          IntegerAttr::get(IntegerType::get(clone.getContext(), 64), entry.second));
    }
    return success();
}
bool SyncTargetInfo::valueAvailable(Value value, Operation* anchor, bool after) const
{
    if (!function || !value || !anchor || !value.getParentRegion()) {
        return false;
    }
    bool inScope = function->isProperAncestor(anchor) &&
                   function->getRegion(0).isAncestor(value.getParentRegion());
    if (!inScope || !dominance.hasSSADominance(anchor->getBlock())) {
        return false;
    }
    if (after) {
        if (anchor->hasTrait<OpTrait::IsTerminator>()) {
            return false;
        }
        anchor = anchor->getNextNode();
    }
    return anchor && dominance.properlyDominates(value, anchor);
}
} // namespace mlir::pto

namespace mlir::pto {
SyncMechanismFact SyncTargetInfo::eventFact(SyncPhysicalCore core, PIPE source, PIPE target) const
{
    return getSyncEventFact(arch, core, source, target);
}
FailureOr<SyncEventPool> SyncTargetInfo::eventPool(
    SyncPhysicalCore core, PIPE source, PIPE target, std::string& reason) const
{
    auto fact = eventFact(core, source, target);
    if (fact.availability != SyncMechanismAvailability::Documented) {
        reason = "unmet directed event-pool qualification: " + fact.obligation.str();
        return failure();
    }
    // Pinned R1 classic SetFlag/WaitFlag: parameters identify directed pools;
    // IDs6/7 are reserved. Do not extrapolate to A5 or protocol-owned IDs.
    SyncEventPool pool;
    pool.core = core;
    pool.source = source;
    pool.target = target;
    pool.reservedIds = {6, 7};
    pool.namespaceSource = "classic directed event IDs0-5; IDs6/7 reserved";
    for (unsigned id = 0; id < 6; ++id) {
        bool reserved = false;
        for (const auto& record : records) {
            if (record.context.core != core) { continue; }
            for (const auto& hidden : record.hiddenEvents) {
                if (static_cast<PIPE>(hidden.srcPipe) == source && static_cast<PIPE>(hidden.dstPipe) == target &&
                    llvm::is_contained(hidden.eventIds, id)) { reserved = true; }
            }
        }
        if (reserved) {
            pool.reservedIds.push_back(id);
        } else {
            pool.eligibleIds.push_back(id);
        }
    }
    return pool;
}
SyncMechanismFact SyncTargetInfo::barrierFact(SyncPhysicalCore core, PIPE pipe) const
{
    return getSyncBarrierFact(arch, core, pipe);
}
} // namespace mlir::pto

namespace mlir::pto {
LogicalResult SyncTargetInfo::preflightMechanisms(
    func::FuncOp pending, const IRMapping& mapping, std::string& reason) const
{
    DenseMap<Operation*, Operation*> original;
    for (const auto& entry : mapping.getOperationMap()) {
        original[entry.second] = entry.first;
    }
    original[pending.getOperation()] = function.operator->();
    auto result = pending.walk([&](Operation* op) {
        if (!isa<BarrierOp, LogicalSetOp, LogicalWaitOp>(op)) {
            return WalkResult::advance();
        }
        Operation* sourceContext = nullptr;
        for (Operation* owner = op; owner; owner = owner->getParentOp()) {
            auto* source = original.lookup(owner);
            if (source && !sourceContext) {
                sourceContext = source;
            }
            if (isa<SectionCubeOp, SectionVectorOp, func::FuncOp, ModuleOp>(owner) &&
                (!source || owner->getName() != source->getName() ||
                 owner->getAttr(FunctionKernelKindAttr::name) !=
                                source->getAttr(FunctionKernelKindAttr::name))) {
                reason = "unmet placement context: speculative core context differs from original source";
                return WalkResult::interrupt();
            }
        }
        if (!sourceContext) {
            reason = "unmet placement context: no original source owner for generated command";
            return WalkResult::interrupt();
        }
        auto core = recoverSyncPhysicalCore(sourceContext);
        SyncMechanismFact fact;
        if (auto barrier = dyn_cast<BarrierOp>(op)) {
            fact = barrierFact(core, barrier.getPipe().getPipe());
        } else if (auto set = dyn_cast<LogicalSetOp>(op)) {
            fact = eventFact(core, set.getSrcPipe().getPipe(), set.getDstPipe().getPipe());
        } else if (auto wait = dyn_cast<LogicalWaitOp>(op)) {
            fact = eventFact(core, wait.getSrcPipe().getPipe(), wait.getDstPipe().getPipe());
        }
        if (fact.availability == SyncMechanismAvailability::Documented) {
            return WalkResult::advance();
        }
        reason = fact.availability == SyncMechanismAvailability::Nonexistent ?
                     "hardware-nonexistent synchronization mechanism: " : "unmet synchronization recipe: ";
        reason += fact.obligation.str();
        return WalkResult::interrupt();
    });
    return result.wasInterrupted() ? failure() : success();
}
} // namespace mlir::pto
