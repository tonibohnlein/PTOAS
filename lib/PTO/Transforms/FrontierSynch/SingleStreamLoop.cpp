// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SingleStreamLoop.h"
#include "DirectEmissionInternal.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/InsertSync/SyncStorageBounds.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include <algorithm>
#include <limits>
#include <optional>
namespace mlir::pto::frontiersynch {
namespace {
bool constant(Value value, int64_t expected)
{
    APInt integer;
    return matchPattern(value, m_ConstantInt(&integer)) && integer == expected;
}
bool lastOrdinalSlot(Value slot, Value bound, uint64_t slots)
{
    auto remainder = slot.getDefiningOp<arith::RemUIOp>();
    auto last = remainder ? remainder.getLhs().getDefiningOp<arith::SubIOp>() : arith::SubIOp{};
    if (remainder && last && last.getLhs() == bound && constant(last.getRhs(), 1) &&
        constant(remainder.getRhs(), slots)) { return true; }
    IntegerAttr literalBound, literalSlot;
    if (!slots || !matchPattern(bound, m_Constant(&literalBound)) ||
        !literalBound.getValue().isStrictlyPositive() ||
        !matchPattern(slot, m_Constant(&literalSlot))) { return false; }
    const auto width = literalBound.getValue().getBitWidth();
    if (width < 64 || literalSlot.getValue().getBitWidth() != width) { return false; }
    const auto finalOrdinal = literalBound.getValue() - APInt(width, 1);
    return literalSlot.getValue() == finalOrdinal.urem(APInt(width, slots));
}
bool nonemptyGuard(scf::IfOp guard, scf::ForOp loop)
{
    auto compare = guard.getCondition().getDefiningOp<arith::CmpIOp>();
    if (!compare) { return false; }
    return (compare.getPredicate() == arith::CmpIPredicate::sgt &&
            compare.getLhs() == loop.getUpperBound() && compare.getRhs() == loop.getLowerBound()) ||
           (compare.getPredicate() == arith::CmpIPredicate::slt &&
            compare.getRhs() == loop.getUpperBound() && compare.getLhs() == loop.getLowerBound());
}
// Rectangular re-entry uses original frame identities. Every enclosing loop
// contains exactly one child loop plus pure scalar definitions; carried state,
// foreign payloads and authored synchronization are not silently discarded.
FailureOr<SmallVector<scf::ForOp>> orderedFrames(scf::ForOp inner, func::FuncOp function)
{
    auto original = recoverSyncLoopFrames(function, inner);
    if (failed(original)) { return failure(); }
    SmallVector<scf::ForOp> result;
    for (auto* frame : *original) { result.push_back(cast<scf::ForOp>(frame)); }
    return result;
}
bool frameNonemptyGuard(Value condition, ArrayRef<scf::ForOp> frames)
{
    SmallVector<Operation*> original;
    for (auto frame : frames) { original.push_back(frame); }
    return syncFramesNonempty(condition, original);
}
bool known(const BaseMemInfo* memory)
{
    return memory && memory->hasKnownPhysicalAddresses && !memory->aliasesUnknownRange &&
           memory->allocateSize && !memory->baseAddresses.empty() &&
           llvm::all_of(memory->baseAddresses, [&](uint64_t address) {
               return address <= std::numeric_limits<uint64_t>::max() - memory->allocateSize;
           });
}
bool hazard(const MemoryDependentAnalyzer& analyzer, const CompoundInstanceElement* source,
            const CompoundInstanceElement* consumer)
{
    DepBaseMemInfoPairVec witness;
    return analyzer.DepBetween(source->defVec, consumer->useVec, witness) ||
           analyzer.DepBetween(source->useVec, consumer->defVec, witness) ||
           analyzer.DepBetween(source->defVec, consumer->defVec, witness);
}
FailureOr<DictionaryAttr> capturedHazardWitness(const SyncInput& input,
    const std::pair<const BaseMemInfo*, const BaseMemInfo*>& pair, std::size_t sourceSite,
    std::size_t consumerSite, int64_t distance, StringRef kind, MLIRContext* context)
{
    Builder builder(context);
    const auto sourceMemory = input.storageBounds().memoryIdentity(pair.first);
    const auto consumerMemory = input.storageBounds().memoryIdentity(pair.second);
    if (!sourceMemory || !consumerMemory) { return failure(); }
    return builder.getDictionaryAttr({
        builder.getNamedAttr("source", builder.getI64IntegerAttr(sourceSite)),
        builder.getNamedAttr("consumer", builder.getI64IntegerAttr(consumerSite)),
        builder.getNamedAttr("distance", builder.getI64IntegerAttr(distance)),
        builder.getNamedAttr("hazard", builder.getStringAttr(kind)),
        builder.getNamedAttr("source_memory", builder.getI64IntegerAttr(*sourceMemory)),
        builder.getNamedAttr("consumer_memory", builder.getI64IntegerAttr(*consumerMemory))});
}
// Capture a real original union-model hazard. The borrowed memory pair is
// immediately converted to immutable shared IDs, before clone publication.
FailureOr<DictionaryAttr> adjacencyWitness(const SyncInput& input,
    const CompoundInstanceElement* source, const CompoundInstanceElement* consumer,
    std::size_t sourceSite, std::size_t consumerSite, int64_t distance, MLIRContext* context)
{
    auto capture = [&](ArrayRef<const BaseMemInfo*> reads, ArrayRef<const BaseMemInfo*> writes,
                       StringRef kind) -> FailureOr<DictionaryAttr> {
        DepBaseMemInfoPairVec pairs;
        SmallVector<const BaseMemInfo*> left(reads.begin(), reads.end()), right(writes.begin(), writes.end());
        if (!input.memory().DepBetween(left, right, pairs)) { return failure(); }
        return capturedHazardWitness(input, pairs.front(), sourceSite, consumerSite, distance, kind, context);
    };
    auto raw = capture(source->defVec, consumer->useVec, "RAW");
    if (succeeded(raw)) { return raw; }
    auto war = capture(source->useVec, consumer->defVec, "WAR");
    if (succeeded(war)) { return war; }
    return capture(source->defVec, consumer->defVec, "WAW");
}
bool purePrefix(Operation* begin, Operation* end)
{
    for (auto* operation = begin; operation && operation != end; operation = operation->getNextNode()) {
        // Even a foreign acquisition can gate SET completion. No extra payload,
        // hidden phase or synchronization command may cross the delayed cut.
        if (operation->getNumRegions() || isa<OpPipeInterface>(operation) || !isPure(operation)) { return false; }
    }
    return true;
}
LogicalResult qualifyEntryCut(const SyncInput& input, func::FuncOp function,
    Operation* entry, scf::ForOp loop, const SyncParticipation& participation, std::string& reason)
{
    auto common = dyn_cast_or_null<scf::IfOp>(participation.owner);
    if (!common || !participation.thenOnly() || common->getBlock() != &function.front() ||
        loop->getBlock() != &common.getThenRegion().front() ||
        entry->getBlock() != &function.front() || !entry->isBeforeInBlock(common) ||
        !input.target().valueAvailable(participation.condition, loop, false) ||
        !input.target().valueAvailable(loop.getUpperBound(), loop, false) ||
        !purePrefix(entry->getNextNode(), common) ||
        !purePrefix(&common.getThenRegion().front().front(), loop)) {
        reason = "guarded entry publication cut has unavailable values or intervening prerequisites";
        return failure();
    }
    return success();
}
bool regularBanks(const SyncSlotSelection& selector)
{
    if (!selector.unitOrdinalModulo || selector.source < 0 || selector.loopSource < 0 ||
        selector.slots < 2 || selector.addresses.size() != selector.slots || !selector.boundingSize) {
        return false;
    }
    // Scan the original planner records, never expand a numeric slot domain.
    uint64_t priorEnd = 0;
    for (auto [slot, address] : llvm::enumerate(selector.addresses)) {
        if (address > std::numeric_limits<uint64_t>::max() - selector.boundingSize ||
            (slot && address != priorEnd)) { return false; }
        priorEnd = address + selector.boundingSize;
    }
    return true;
}
} // namespace
FailureOr<std::shared_ptr<const SingleStreamLoop>> SingleStreamLoop::build(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
    CostLedger& costs, std::string& reason, std::shared_ptr<const SingleStreamLoop> child)
{
    if (input.target().originalBarrierChain().barrier) {
        return buildRepeatedPair(function, input, trace, costs, reason);
    }
    std::optional<CostScope> qualification;
    qualification.emplace(costs, CostStage::Effects);
    reason = "single-stream boundary premises not established";
    if (trace.sites().size() < 3 ||
        !llvm::hasSingleElement(function.getBody())) { return failure(); }
    auto result = std::shared_ptr<SingleStreamLoop>(new SingleStreamLoop());
    result->exit = trace.sites().size() - 1;
    const auto* entry = trace.sites()[result->entry].phase;
    const auto* exit = trace.sites()[result->exit].phase;
    auto loop = trace.sites()[1].phase->elementOp->getParentOfType<scf::ForOp>();
    auto guard = exit->elementOp->getParentOfType<scf::IfOp>();
    auto frameResult = orderedFrames(loop, function);
    if (failed(frameResult) || frameResult->empty()) { return failure(); }
    result->frames = *frameResult;
    for (auto frame : result->frames) {
        auto identity = input.target().sourceIdentity(frame);
        if (!identity) { return failure(); }
        result->frameSources.push_back(*identity);
    }
    auto outer = result->frames.front();
    const SyncParticipation emptyParticipation;
    const auto& capturedParticipation = input.target().participation();
    const auto& participation = outer && capturedParticipation.thenOnly() && capturedParticipation.armFor(outer) &&
        capturedParticipation.armFor(outer)->outcome ?
        capturedParticipation : emptyParticipation;
    auto* scope = participation.owner ? &cast<scf::IfOp>(participation.owner).getThenRegion().front() :
                                       &function.front();
    Operation* invocation = participation.owner ? participation.owner : outer.getOperation();
    if (entry->kPipeValue != PipelineType::PIPE_MTE2 || exit->kPipeValue != PipelineType::PIPE_MTE3 ||
        !loop || !guard || outer->getBlock() != scope || guard->getBlock() != scope ||
        entry->elementOp->getBlock() != &function.front() ||
        exit->elementOp->getBlock() != &guard.getThenRegion().front() ||
        !entry->elementOp->isBeforeInBlock(invocation) || !outer->isBeforeInBlock(guard) ||
        !constant(loop.getLowerBound(), 0) || !constant(loop.getStep(), 1) ||
        loop.getNumRegionIterArgs() || !frameNonemptyGuard(guard.getCondition(), result->frames)) { return failure(); }
    if (participation.owner) {
        if (failed(qualifyEntryCut(input, function, entry->elementOp, outer, participation, reason))) {
            return failure();
        }
        result->participation = participation.owner; result->activation = participation.condition;
        result->activationBinding = input.target().activationAttribute();
        if (participation.witness.origin == SyncActivationOrigin::Argument) {
            result->activationArgument = cast<BlockArgument>(participation.condition).getArgNumber();
        }
        result->participationSource = input.target().sourceIdentity(participation.owner).value_or(-1);
        result->publicationCut = outer;
        result->publicationCutSource = input.target().sourceIdentity(outer).value_or(-1);
        if (result->participationSource < 0 || result->publicationCutSource < 0) { return failure(); }
    }
    SmallVector<const CompoundInstanceElement*> bodyPhases;
    for (std::size_t site = 1; site < result->exit; ++site) {
        const auto* phase = trace.sites()[site].phase;
        if (phase->kPipeValue != PipelineType::PIPE_V || phase->elementOp->getBlock() != loop.getBody() ||
            (!bodyPhases.empty() && !bodyPhases.back()->elementOp->isBeforeInBlock(phase->elementOp))) {
            reason = "body is not one unconditional lexical physical stream"; return failure();
        }
        bodyPhases.push_back(phase);
        result->bodies.push_back(site);
    }
    bool additionalControl = false;
    function.walk([&](Operation* operation) {
        if (operation != function && !llvm::is_contained(result->frames, dyn_cast<scf::ForOp>(operation)) &&
            operation != guard && operation != participation.owner && operation->getNumRegions()) {
            additionalControl = true;
        }
    });
    if (additionalControl) { reason = "single-stream sequence has additional participation control"; return failure(); }
    auto inductionType = loop.getInductionVar().getType();
    auto integer = dyn_cast<IntegerType>(inductionType);
    if (!isa<IndexType>(inductionType) && (!integer || integer.getWidth() < 2 || integer.getWidth() > 64)) {
        reason = "single-stream scalar type is not supported by the selected manual source ABI"; return failure();
    }
    for (auto& operation : *loop.getBody()) {
        if (operation.getNumRegions()) { return failure(); }
    }
    if (!guard.getElseRegion().empty() && !guard.getElseRegion().front().without_terminator().empty()) {
        return failure();
    }
    // Every bank selection is tied to the original loop's executed ordinal.
    // Equivalent SSA selections share their original multi-tile source family.
    SmallVector<const SyncSlotSelection*> outputs;
    for (const auto* phase : bodyPhases) {
        const SyncSlotSelection* output = nullptr;
        for (const auto* memory : phase->defVec) {
            if (auto* selector = input.slotSelection(memory->baseBuffer)) {
                if (output && output != selector) { return failure(); }
                output = selector;
            }
        }
        if (!output || output->loop != loop || !regularBanks(*output)) { return failure(); }
        outputs.push_back(output);
        result->selectorSources.push_back(output->source);
    }
    auto finalSelection = outputs.back()->selected.getDefiningOp<MultiTileGetOp>();
    bool lastRead = false;
    for (const auto* memory : exit->useVec) {
        auto final = memory->baseBuffer.getDefiningOp<MultiTileGetOp>();
        if (!final || final.getSource() != finalSelection.getSource()) { continue; }
        if (!lastOrdinalSlot(final.getSlot(), loop.getUpperBound(), outputs.back()->slots)) { return failure(); }
        lastRead = true;
    }
    if (auto argument = dyn_cast<BlockArgument>(loop.getUpperBound())) {
        if (argument.getOwner() != &function.front()) { return failure(); }
        result->boundArgument = argument.getArgNumber();
    } else {
        if (!matchPattern(loop.getUpperBound(), m_Constant(&result->boundConstant))) { return failure(); }
    }
    if (!lastRead || !hazard(input.memory(), entry, bodyPhases.front()) ||
        !hazard(input.memory(), bodyPhases.back(), exit)) { return failure(); }
    result->loop = loop; result->bound = loop.getUpperBound(); result->exitGuard = guard;
    result->guardSource = input.target().sourceIdentity(guard).value_or(-1);
    if (result->guardSource < 0) { return failure(); }
    qualification.reset();
    if (child) {
        if (!child->regionOnly || child->frames != result->frames || child->bodies != result->bodies ||
            child->activation != result->activation || child->activationOutcome != result->activationOutcome) {
            reason = "cached frame child identity/context mismatch"; return failure();
        }
        result->loopChild = std::move(child);
        result->chainWitnesses = result->loopChild->chainWitnesses;
        result->selectorSources = result->loopChild->selectorSources;
        result->selectorWitnesses = result->loopChild->selectorWitnesses;
        result->banks = result->loopChild->banks; result->loopSource = result->loopChild->loopSource;
    } else if (failed(result->buildBody(function, input, trace, costs, reason))) { return failure(); }
    reason.clear();
    return std::shared_ptr<const SingleStreamLoop>(result);
}
LogicalResult SingleStreamLoop::buildBody(func::FuncOp function, const SyncInput& input,
    const TraceDemandAnalysis& trace, CostLedger& costs, std::string& reason)
{
    std::optional<CostScope> qualification;
    qualification.emplace(costs, CostStage::Effects);
    SmallVector<const CompoundInstanceElement*> bodyPhases;
    SmallVector<const SyncSlotSelection*> outputs;
    for (auto site : bodies) {
        if (site >= trace.sites().size()) { return failure(); }
        const auto* phase = trace.sites()[site].phase;
        if (phase->kPipeValue != PipelineType::PIPE_V || phase->elementOp->getBlock() != loop.getBody()) {
            return failure();
        }
        const SyncSlotSelection* output = nullptr;
        for (const auto* memory : phase->defVec) {
            if (auto* selection = input.slotSelection(memory->baseBuffer)) {
                if (output && output != selection) { return failure(); }
                output = selection;
            }
        }
        if (!output || output->loop != loop || !regularBanks(*output)) { return failure(); }
        bodyPhases.push_back(phase); outputs.push_back(output);
    }
    if (bodyPhases.empty()) { return failure(); }
    SmallVector<RotatingFamily> families;
    SmallVector<RotatingFragment> fragments;
    DenseMap<Value, std::size_t> stationary, rotatingFamilies;
    for (auto [position, phase] : llvm::enumerate(bodyPhases)) {
        auto observe = [&](ArrayRef<const BaseMemInfo*> memories, RotatingAccessMode mode) {
            for (const auto* memory : memories) {
                if (!known(memory)) { return failure(); }
                if (auto* selector = input.slotSelection(memory->baseBuffer)) {
                    if (selector->loop != loop || !regularBanks(*selector)) { return failure(); }
                    auto selected = selector->selected.getDefiningOp<MultiTileGetOp>();
                    auto inserted = rotatingFamilies.try_emplace(selected.getSource(), families.size());
                    if (inserted.second) {
                        families.push_back({llvm::DynamicAPInt(selector->slots), llvm::DynamicAPInt(1), 1});
                    }
                    fragments.push_back({position, inserted.first->second, 0, llvm::DynamicAPInt(0), mode});
                    continue;
                }
                auto* definition = memory->baseBuffer.getDefiningOp();
                if (!definition || (frames.empty() ? loop : frames.front())->isAncestor(definition) ||
                    memory->baseAddresses.size() != 1) {
                    return failure();
                }
                auto inserted = stationary.try_emplace(memory->baseBuffer, families.size());
                if (inserted.second) { families.push_back({llvm::DynamicAPInt(1), llvm::DynamicAPInt(0), 1}); }
                fragments.push_back({position, inserted.first->second, 0, llvm::DynamicAPInt(0), mode});
            }
            return success();
        };
        if (failed(observe(phase->useVec, RotatingAccessMode::Read)) ||
            failed(observe(phase->defVec, RotatingAccessMode::Write))) { return failure(); }
    }
    // Every original union-model occurrence hazard is forward in this fixed
    // skeleton. Actual adjacent hazards generate the total body order, so they
    // dominate all those hazards. Auxiliary storage generators are a subset of
    // this order, not a claim that distinct SSA families physically partition it.
    SmallVector<PeriodicPrerequisite> required;
    SmallVector<Attribute> witnesses;
    for (auto [position, phase] : llvm::enumerate(bodyPhases)) {
        const auto next = (position + 1) % bodyPhases.size();
        const int64_t distance = next == 0 ? 1 : 0;
        auto witness = adjacencyWitness(input, phase, bodyPhases[next], bodies[position],
                                        bodies[next], distance, function.getContext());
        if (failed(witness)) {
            reason = "original shared modeled lexical or wrap hazard missing"; return failure();
        }
        witnesses.push_back(*witness);
        required.push_back({position, next, llvm::DynamicAPInt(distance)});
    }
    chainWitnesses = Builder(function.getContext()).getArrayAttr(witnesses);
    selectorSources.clear();
    for (auto* output : outputs) { selectorSources.push_back(output->source); }
    banks = outputs.back()->slots;
    selectorWitnesses = input.slotAttribute(function.getContext());
    loopSource = outputs.back()->loopSource;
    qualification.reset();
    {
        CostScope backend(costs, CostStage::Backend);
        if (failed(storage.build(bodyPhases, families, fragments)) ||
            failed(reduction.build(storage, required))) { return failure(); }
    }
    if (reduction.retained().size() != bodyPhases.size()) { return failure(); }
    SmallVector<bool> covered(bodyPhases.size(), false);
    for (auto index : reduction.retained()) {
        const auto& edge = reduction.generators()[index].edge;
        if (edge.source >= bodyPhases.size() || edge.consumer != (edge.source + 1) % bodyPhases.size() ||
            edge.distance != (edge.source + 1 == bodyPhases.size() ? 1 : 0) || covered[edge.source]) {
            reason = "rotating quotient differs from qualified lexical covers"; return failure();
        }
        covered[edge.source] = true;
    }
    return success();
}
FailureOr<std::shared_ptr<const SingleStreamLoop>> SingleStreamLoop::buildBodyRegion(
    func::FuncOp function, scf::ForOp owner, ArrayRef<std::size_t> sites,
    const SyncInput& input, const TraceDemandAnalysis& trace, CostLedger& costs, std::string& reason)
{
    reason = "fixed lexical body port premises missing";
    if (!owner || sites.empty() ||
        !constant(owner.getLowerBound(), 0) || !constant(owner.getStep(), 1) ||
        owner.getNumRegionIterArgs() || !isa<IndexType>(owner.getInductionVar().getType())) {
        return failure();
    }
    auto innermost = trace.sites()[sites.front()].phase->elementOp->getParentOfType<scf::ForOp>();
    auto frameResult = orderedFrames(innermost, function);
    if (failed(frameResult) || frameResult->empty() || frameResult->front() != owner) { return failure(); }
    for (auto& operation : *innermost.getBody()) {
        if (operation.getNumRegions()) { return failure(); }
    }
    auto result = std::shared_ptr<SingleStreamLoop>(new SingleStreamLoop());
    result->regionOnly = true; result->loop = innermost; result->bound = innermost.getUpperBound();
    if (frameResult->size() > 1) {
        result->frames = *frameResult;
        for (auto frame : result->frames) {
            auto identity = input.target().sourceIdentity(frame);
            if (!identity) { return failure(); }
            result->frameSources.push_back(*identity);
        }
    }
    result->bodies.assign(sites.begin(), sites.end());
    if (auto argument = dyn_cast<BlockArgument>(result->bound)) {
        if (argument.getOwner() != &function.front()) { return failure(); }
        result->boundArgument = argument.getArgNumber();
    } else if (!matchPattern(result->bound, m_Constant(&result->boundConstant))) { return failure(); }
    const auto& participation = input.target().participation();
    if (auto arm = participation.armFor(owner)) {
        result->armRegion = arm->region; result->activationOutcome = arm->outcome;
        result->activation = participation.condition; result->participation = participation.owner;
        result->activationBinding = input.target().activationAttribute();
        result->participationSource = input.target().sourceIdentity(participation.owner).value_or(-1);
        if (participation.witness.origin == SyncActivationOrigin::Argument) {
            result->activationArgument = cast<BlockArgument>(participation.condition).getArgNumber();
        }
    } else if (owner->getParentOp() != function) { return failure(); }
    if (failed(result->buildBody(function, input, trace, costs, reason))) { return failure(); }
    reason.clear();
    return std::shared_ptr<const SingleStreamLoop>(result);
}
FailureOr<SingleStreamPortQuery> SingleStreamLoop::portQuery(bool sourceLast, PeriodicEventKind sourceKind,
    bool targetLast, PeriodicEventKind targetKind) const
{
    if (!regionOnly || !bound || bodies.empty() || frames.size() > 1) { return failure(); }
    auto distance = reduction.threshold(sourceLast ? bodies.size() - 1 : 0, sourceKind,
                                        targetLast ? bodies.size() - 1 : 0, targetKind);
    if (failed(distance)) { return failure(); }
    if (!*distance) { return SingleStreamPortQuery{false, bound, activation, 0, 0, armRegion, activationOutcome}; }
    // Checked conversion precedes forming the affine constant. The supported
    // fixed-body generator index has finite encoded distances, never trips.
    if (**distance < 0 || **distance > std::numeric_limits<int64_t>::max() - 1) { return failure(); }
    const int64_t difference = int64_t(targetLast) - int64_t(sourceLast);
    return SingleStreamPortQuery{true, bound, activation, difference,
                                -difference - static_cast<int64_t>(**distance), armRegion, activationOutcome};
}
FailureOr<ArrayAttr> SingleStreamLoop::framePortQuery(bool sourceLast, PeriodicEventKind sourceKind,
    bool targetLast, PeriodicEventKind targetKind) const
{
    if (frames.empty() || bodies.empty()) { return failure(); }
    auto threshold = bodyIndex().threshold(sourceLast ? bodies.size() - 1 : 0, sourceKind,
                                         targetLast ? bodies.size() - 1 : 0, targetKind);
    if (failed(threshold)) { return failure(); }
    Builder builder(loop->getContext());
    if (!*threshold) { return builder.getArrayAttr({}); }
    // Witnessed total adjacency bounds every first/last threshold by one.
    // Larger thresholds require a richer product-free interface, not sampling.
    if (**threshold < 0 || **threshold > 1) { return failure(); }
    const unsigned depth = frames.size(), width = depth + unsigned(bool(activation)) + 1;
    SmallVector<int64_t> positive, active;
    auto row = [&](SmallVector<int64_t>& rows, unsigned axis, int64_t coefficient, int64_t constantValue) {
        SmallVector<int64_t> values(width, 0); values[axis] = coefficient;
        values.back() = constantValue; llvm::append_range(rows, values);
    };
    for (unsigned axis = 0; axis < depth; ++axis) { row(positive, axis, 1, -1); }
    if (activation) { row(active, depth, 1, -int64_t(activationOutcome)); }
    SmallVector<Attribute> alternatives;
    auto append = [&](ArrayRef<int64_t> equalities, ArrayRef<int64_t> inequalities) {
        alternatives.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("equalities", builder.getDenseI64ArrayAttr(equalities)),
            builder.getNamedAttr("inequalities", builder.getDenseI64ArrayAttr(inequalities))}));
    };
    if (sourceLast && !targetLast) {
        if (**threshold == 0) {
            auto equalities = active;
            for (unsigned axis = 0; axis < depth; ++axis) { row(equalities, axis, 1, -1); }
            append(equalities, positive);
        }
    } else if (!sourceLast && targetLast && **threshold == 1) {
        // Product of positive integer bounds exceeds one iff at least one
        // bound exceeds one. This is a symbolic identity for all valuations.
        for (unsigned axis = 0; axis < depth; ++axis) {
            auto inequalities = positive; row(inequalities, axis, 1, -2);
            append(active, inequalities);
        }
    } else if (**threshold == 0) { append(active, positive); }
    return builder.getArrayAttr(alternatives);
}
FailureOr<bool> SingleStreamLoop::portReaches(bool sourceLast, PeriodicEventKind sourceKind,
    bool targetLast, PeriodicEventKind targetKind, int64_t trips) const
{
    if (!regionOnly || trips <= 0) { return failure(); }
    const auto source = sourceLast ? bodies.size() - 1 : 0;
    const auto target = targetLast ? bodies.size() - 1 : 0;
    const int64_t sourceOrdinal = sourceLast ? trips - 1 : 0;
    const int64_t targetOrdinal = targetLast ? trips - 1 : 0;
    if (targetOrdinal < sourceOrdinal) { return false; }
    auto distance = reduction.threshold(source, sourceKind, target, targetKind);
    if (failed(distance)) { return failure(); }
    return *distance && **distance <= llvm::DynamicAPInt(targetOrdinal - sourceOrdinal);
}
FailureOr<DictionaryAttr> SingleStreamLoop::crossingWitness(const SyncInput& input,
    const CompoundInstanceElement* source, const CompoundInstanceElement* consumer,
    std::size_t sourceSite, std::size_t consumerSite, int64_t distance, MLIRContext* context)
{ return adjacencyWitness(input, source, consumer, sourceSite, consumerSite, distance, context); }
bool SingleStreamLoop::prefixPure(Operation* begin, Operation* end) { return purePrefix(begin, end); }
LogicalResult SingleStreamLoop::entryCut(const SyncInput& input, func::FuncOp function, Operation* entry,
    scf::ForOp loop, const SyncParticipation& participation, std::string& reason)
{ return qualifyEntryCut(input, function, entry, loop, participation, reason); }
FailureOr<std::shared_ptr<const SingleStreamLoop>> SingleStreamLoop::buildRepeatedRegion(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
    CostLedger& costs, std::string& reason)
{
    std::optional<CostScope> qualification;
    qualification.emplace(costs, CostStage::Effects);
    reason = "original-barrier repeated pair premises missing";
    const auto& original = input.target().originalBarrierChain();
    const SyncParticipation emptyParticipation;
    const auto& participation = original.participationContext ? *original.participationContext : emptyParticipation;
    if ((participation.owner && !participation.thenOnly()) || !original.barrier ||
        original.requirementDomain != SyncBarrierRequirementDomain::SharedUnion ||
        original.ready.empty() || original.release.empty() ||
        trace.sites().size() != original.computes.size() + (original.prefix ? 3 : 2)) {
        return failure();
    }
    auto loop = dyn_cast<scf::ForOp>(original.loop);
    const std::size_t first = original.prefix ? 1 : 0;
    const auto* producer = trace.sites()[first].phase;
    const auto* consumer = trace.sites()[first + 1].phase;
    if (original.computes.empty() || original.releaseProducer != original.computes.back() ||
        original.lexical.size() + 1 != original.computes.size()) { return failure(); }
    for (auto [position, phase] : llvm::enumerate(original.computes)) {
        if (trace.sites()[first + 1 + position].phase != phase) { return failure(); }
    }
    const auto lastSite = first + original.computes.size();
    if (!loop || producer != original.producer || consumer != original.consumer ||
        !isa<IndexType>(loop.getInductionVar().getType())) { return failure(); }
    auto* owner = original.frames.empty() ? original.loop : original.frames.front();
    auto arm = participation.armFor(owner);
    if (participation.owner && (!participation.thenOnly() || !arm || !arm->outcome)) { return failure(); }
    bool extra = false;
    loop.walk([&](Operation* operation) {
        if ((operation != loop && operation->getNumRegions()) ||
            (isa<BarrierOp>(operation) && operation != original.barrier)) { extra = true; }
    });
    if (extra) { return failure(); }
    // Retain union alternatives; the complete shared union model is constant
    // across occurrences. Only original ordinal-modulo selections may vary.
    SmallVector<const CompoundInstanceElement*> phases{producer};
    llvm::append_range(phases, original.computes);
    for (const auto* phase : phases) {
        for (auto memories : {ArrayRef<const BaseMemInfo*>(phase->useVec),
                              ArrayRef<const BaseMemInfo*>(phase->defVec)}) {
            for (const auto* memory : memories) {
                if (memory->scope == AddressSpace::GM) {
                    auto* definition = memory->baseBuffer.getDefiningOp();
                    if (definition && loop->isAncestor(definition)) { return failure(); }
                    continue;
                }
                if (!known(memory)) { return failure(); }
                if (auto* selector = input.slotSelection(memory->baseBuffer)) {
                    if (selector->loop != loop || !regularBanks(*selector)) { return failure(); }
                } else {
                    auto* definition = memory->baseBuffer.getDefiningOp();
                    if (!definition || loop->isAncestor(definition) || memory->baseAddresses.size() != 1) {
                        return failure();
                    }
                }
            }
        }
    }
    const SyncSlotSelection* output = nullptr;
    for (const auto* memory : original.releaseProducer->defVec) {
        if (auto* selector = input.slotSelection(memory->baseBuffer)) {
            if (output && output != selector) { return failure(); }
            output = selector;
        }
    }
    if (!output) { return failure(); }
    auto ready = capturedHazardWitness(input, original.ready.front(), first, first + 1, 0,
                                       "RAW", function.getContext());
    auto release = capturedHazardWitness(input, original.release.front(), lastSite, first, 1,
                                         "WAR", function.getContext());
    if (failed(ready) || failed(release)) { return failure(); }
    auto result = std::shared_ptr<SingleStreamLoop>(new SingleStreamLoop());
    result->protocol = SingleStreamProtocol::RepeatedReadyRelease; result->loop = loop;
    result->bound = loop.getUpperBound();
    if (original.frames.size() > 1) {
        for (auto* frame : original.frames) {
            result->frames.push_back(cast<scf::ForOp>(frame));
            auto identity = input.target().sourceIdentity(frame);
            if (!identity) { return failure(); } result->frameSources.push_back(*identity);
        }
    }
    result->regionOnly = true;
    for (std::size_t site = first; site <= lastSite; ++site) { result->bodies.push_back(site); }
    result->participation = participation.owner; result->activation = participation.condition;
    result->armRegion = arm ? arm->region : nullptr;
    result->activationOutcome = true;
    if (participation.condition) {
        result->activationBinding = input.target().activationAttribute();
        if (!result->activationBinding || participation.witness.origin == SyncActivationOrigin::None) {
            return failure();
        }
        if (participation.witness.origin == SyncActivationOrigin::Argument) {
            result->activationArgument = cast<BlockArgument>(participation.condition).getArgNumber();
        }
        result->participationSource = input.target().sourceIdentity(participation.owner).value_or(-1);
        if (result->participationSource < 0) { return failure(); }
    }
    result->originalBarrier = original.barrier;
    result->originalBarrierSource = input.target().sourceIdentity(original.barrier).value_or(-1);
    result->loopSource = output->loopSource;
    result->selectorSources = {output->source}; result->banks = output->slots;
    result->selectorWitnesses = input.slotAttribute(function.getContext());
    SmallVector<Attribute> witnesses{*ready};
    for (auto [position, lexical] : llvm::enumerate(original.lexical)) {
        if (lexical.source != original.computes[position] ||
            lexical.consumer != original.computes[position + 1] || lexical.pairs.empty()) { return failure(); }
        StringRef kind = lexical.kind == SyncBarrierHazard::ReadAfterWrite ? "RAW" :
                         lexical.kind == SyncBarrierHazard::WriteAfterRead ? "WAR" : "WAW";
        auto witness = capturedHazardWitness(input, lexical.pairs.front(), first + position + 1,
                                             first + position + 2, 0, kind, function.getContext());
        if (failed(witness)) { return failure(); }
        witnesses.push_back(*witness);
    }
    witnesses.push_back(*release);
    result->chainWitnesses = Builder(function.getContext()).getArrayAttr(witnesses);
    if (auto argument = dyn_cast<BlockArgument>(loop.getUpperBound())) {
        if (argument.getOwner() != &function.front()) { return failure(); }
        result->boundArgument = argument.getArgNumber();
    } else if (!matchPattern(loop.getUpperBound(), m_Constant(&result->boundConstant))) { return failure(); }
    if (result->originalBarrierSource < 0) { return failure(); }
    SmallVector<PeriodicPrerequisite> requirements;
    for (std::size_t position = 0; position < phases.size(); ++position) {
        requirements.push_back({position, (position + 1) % phases.size(),
                                llvm::DynamicAPInt(position + 1 == phases.size() ? 1 : 0)});
    }
    qualification.reset();
    CostScope backend(costs, CostStage::Backend);
    if (failed(result->reduction.build(phases, requirements)) || result->reduction.retained().size() != phases.size()) {
        return failure();
    }
    for (auto index : result->reduction.retained()) {
        const auto& edge = result->reduction.generators()[index].edge;
        if (edge.source >= phases.size() || edge.consumer != (edge.source + 1) % phases.size() ||
            edge.distance != (edge.source + 1 == phases.size() ? 1 : 0)) { return failure(); }
    }
    reason.clear();
    return std::shared_ptr<const SingleStreamLoop>(result);
}
FailureOr<std::shared_ptr<const SingleStreamLoop>> SingleStreamLoop::composeRepeatedRegion(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
    std::shared_ptr<const SingleStreamLoop> child, CostLedger& costs, std::string& reason)
{
    CostScope crossing(costs, CostStage::Effects);
    reason = "compact repeated-loop boundary composition premises missing";
    const auto& original = input.target().originalBarrierChain();
    const SyncParticipation emptyParticipation;
    const auto& participation = original.participationContext ? *original.participationContext : emptyParticipation;
    if (!child || !child->regionOnly || !child->repeatedPair() || child->loop != original.loop ||
        original.requirementDomain != SyncBarrierRequirementDomain::SharedUnion ||
        trace.sites().size() != original.computes.size() + (original.prefix ? 3 : 2) ||
        child->participation != participation.owner || child->activation != participation.condition) {
        return failure();
    }
    auto outer = child->frames.empty() ? child->loop : child->frames.front();
    auto arm = participation.armFor(outer);
    if (!child->activationOutcome ||
        (participation.owner && (!participation.thenOnly() || !arm || !arm->outcome ||
                                child->armRegion != arm->region)) ||
        (!participation.owner && child->armRegion)) { return failure(); }
    const auto firstPort = child->firstPort(), lastPort = child->lastPort();
    if (firstPort.site != child->first() || firstPort.coefficient != 0 || firstPort.offset != 0 ||
        lastPort.site != child->last() || lastPort.coefficient != 1 || lastPort.offset != -1 ||
        !firstPort.requiresPositiveTrips || !lastPort.requiresPositiveTrips ||
        !firstPort.activationOutcome || !lastPort.activationOutcome ||
        firstPort.armRegion != child->armRegion || lastPort.armRegion != child->armRegion ||
        firstPort.activation != participation.condition || lastPort.activation != participation.condition ||
        firstPort.participation != participation.owner || lastPort.participation != participation.owner) {
        return failure();
    }
    auto loop = child->loop;
    const auto exitSite = trace.sites().size() - 1;
    const auto* exit = trace.sites()[exitSite].phase;
    auto guard = exit->elementOp->getParentOfType<scf::IfOp>();
    auto* scope = participation.owner ? &cast<scf::IfOp>(participation.owner).getThenRegion().front() :
                                          &function.front();
    const bool present = guard && (child->frames.empty() ? nonemptyGuard(guard, loop) :
                         frameNonemptyGuard(guard.getCondition(), child->frames));
    if (!guard || !present || guard->getBlock() != scope ||
        !outer->isBeforeInBlock(guard) || exit->elementOp->getBlock() != &guard.getThenRegion().front() ||
        (!guard.getElseRegion().empty() && !guard.getElseRegion().front().without_terminator().empty())) {
        return failure();
    }
    bool extra = false;
    function.walk([&](Operation* operation) {
        if ((operation != function && operation != loop &&
             !llvm::is_contained(child->frames, dyn_cast<scf::ForOp>(operation)) && operation != guard &&
             operation != participation.owner && operation->getNumRegions()) ||
            (isa<BarrierOp>(operation) && operation != original.barrier)) { extra = true; }
    });
    if (extra) { return failure(); }
    // Resolve the actual shared last-output family, never a synthetic slot.
    const SyncSlotSelection* output = nullptr;
    for (const auto* memory : trace.sites()[child->last()].phase->defVec) {
        if (auto* selected = input.slotSelection(memory->baseBuffer)) { output = selected; }
    }
    if (!output) { return failure(); }
    auto family = output->selected.getDefiningOp<MultiTileGetOp>().getSource();
    bool lastRead = false;
    for (const auto* memory : exit->useVec) {
        auto final = memory->baseBuffer.getDefiningOp<MultiTileGetOp>();
        if (!final || final.getSource() != family) { continue; }
        if (!lastOrdinalSlot(final.getSlot(), loop.getUpperBound(), output->slots)) { return failure(); }
        lastRead = true;
    }
    if (!lastRead || !hazard(input.memory(), trace.sites()[lastPort.site].phase, exit)) { return failure(); }
    SmallVector<Attribute> witnesses(child->chainWitnesses.begin(), child->chainWitnesses.end());
    if (original.prefix) {
        if (trace.sites().front().phase != original.prefix || original.prefixWrites.empty()) { return failure(); }
        // Every complete prefix effect is retained by shared translation. With
        // all later occurrences chained, every forward crossing is implied by
        // the real prefix->first P WAW plus the child's original READY/RELEASE.
        auto prefix = capturedHazardWitness(input, original.prefixWrites.front(), 0, firstPort.site, 0,
                                           "WAW", function.getContext());
        if (failed(prefix)) { return failure(); }
        witnesses.push_back(*prefix);
    }
    auto result = std::shared_ptr<SingleStreamLoop>(new SingleStreamLoop());
    result->protocol = child->protocol; result->loop = loop; result->exitGuard = guard;
    result->bodies = child->bodies; result->exit = exitSite;
    result->frames = child->frames; result->frameSources = child->frameSources;
    result->hasPrefix = original.prefix != nullptr; result->entry = 0;
    result->loopChild = std::move(child);
    result->participation = result->loopChild->participation; result->activation = result->loopChild->activation;
    result->armRegion = result->loopChild->armRegion;
    result->activationOutcome = result->loopChild->activationOutcome;
    result->activationArgument = result->loopChild->activationArgument;
    result->activationBinding = result->loopChild->activationBinding;
    result->participationSource = result->loopChild->participationSource;
    result->originalBarrier = original.barrier;
    result->originalBarrierSource = result->loopChild->originalBarrierSource;
    result->loopSource = result->loopChild->loopSource;
    result->guardSource = input.target().sourceIdentity(guard).value_or(-1);
    result->selectorSources = result->loopChild->selectorSources; result->banks = result->loopChild->banks;
    result->selectorWitnesses = result->loopChild->selectorWitnesses;
    result->chainWitnesses = Builder(function.getContext()).getArrayAttr(witnesses);
    result->boundArgument = result->loopChild->boundArgument; result->boundConstant = result->loopChild->boundConstant;
    // The immutable child owns the periodic index; parent crossing work does
    // not copy or rerun it. No parent periodic-threshold capability is claimed.
    if (result->guardSource < 0) { return failure(); }
    reason.clear();
    return std::shared_ptr<const SingleStreamLoop>(result);
}
FailureOr<std::shared_ptr<const SingleStreamLoop>> SingleStreamLoop::buildRepeatedPair(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
    CostLedger& costs, std::string& reason)
{
    auto child = buildRepeatedRegion(function, input, trace, costs, reason);
    if (failed(child)) { return failure(); }
    return composeRepeatedRegion(function, input, trace, *child, costs, reason);
}
// Closed lexicographic frame covers. Columns are source coordinates, target
// coordinates, invariant bounds, optional original activation, and constant.
// A carry at level l fixes all deeper source coordinates at their last value
// and all deeper target coordinates at zero; no bound products are required.
DictionaryAttr SingleStreamLoop::frameEvidence(MLIRContext* context) const
{
    Builder builder(context);
    const unsigned depth = frames.size();
    SmallVector<Attribute> minimum, frameRecords;
    SmallVector<int64_t> parameters;
    for (auto frame : frames) {
        auto argument = dyn_cast<BlockArgument>(frame.getUpperBound());
        Attribute constantBound;
        matchPattern(frame.getUpperBound(), m_Constant(&constantBound));
        NamedAttrList record;
        const int64_t parameterArgument = argument ? static_cast<int64_t>(argument.getArgNumber()) : -1;
        record.set("parameter_argument", builder.getI64IntegerAttr(parameterArgument));
        if (constantBound) { record.set("parameter_constant", constantBound); }
        record.set("index_type", TypeAttr::get(frame.getInductionVar().getType()));
        parameters.push_back(parameterArgument);
        frameRecords.push_back(builder.getDictionaryAttr(record));
    }
    auto append = [&](std::size_t source, std::size_t consumer, unsigned sourceDepth,
                      unsigned targetDepth, std::optional<unsigned> carry) {
        const unsigned parameter = sourceDepth + targetDepth;
        const unsigned width = parameter + depth + unsigned(bool(activation)) + 1;
        SmallVector<int64_t> equalities, inequalities;
        auto row = [&](SmallVector<int64_t>& rows, ArrayRef<std::pair<unsigned, int64_t>> terms,
                       int64_t constantValue) {
            SmallVector<int64_t> values(width, 0);
            for (auto [column, coefficient] : terms) { values[column] = coefficient; }
            values.back() = constantValue; llvm::append_range(rows, values);
        };
        for (unsigned axis = 0; axis < depth; ++axis) {
            row(inequalities, {{parameter + axis, 1}}, -1);
            if (sourceDepth) {
                row(inequalities, {{axis, 1}}, 0);
                row(inequalities, {{axis, -1}, {parameter + axis, 1}}, -1);
            }
            if (targetDepth) {
                row(inequalities, {{sourceDepth + axis, 1}}, 0);
                row(inequalities, {{sourceDepth + axis, -1}, {parameter + axis, 1}}, -1);
            }
            if (!sourceDepth) { row(equalities, {{axis, 1}}, 0); }
            else if (!targetDepth) { row(equalities, {{axis, 1}, {parameter + axis, -1}}, 1); }
            else if (!carry || axis < *carry) {
                row(equalities, {{axis, -1}, {sourceDepth + axis, 1}}, 0);
            } else if (axis == *carry) {
                row(equalities, {{axis, -1}, {sourceDepth + axis, 1}}, -1);
            } else {
                row(equalities, {{axis, 1}, {parameter + axis, -1}}, 1);
                row(equalities, {{sourceDepth + axis, 1}}, 0);
            }
        }
        if (activation) { row(equalities, {{width - 2, 1}}, -int64_t(activationOutcome)); }
        minimum.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("source", builder.getI64IntegerAttr(source)),
            builder.getNamedAttr("consumer", builder.getI64IntegerAttr(consumer)),
            builder.getNamedAttr("source_depth", builder.getI64IntegerAttr(sourceDepth)),
            builder.getNamedAttr("consumer_depth", builder.getI64IntegerAttr(targetDepth)),
            builder.getNamedAttr("equalities", builder.getDenseI64ArrayAttr(equalities)),
            builder.getNamedAttr("inequalities", builder.getDenseI64ArrayAttr(inequalities))}));
    };
    if (!regionOnly && (!repeatedPair() || hasPrefix)) { append(entry, first(), 0, depth, {}); }
    for (unsigned site = 1; site < bodies.size(); ++site) {
        append(bodies[site - 1], bodies[site], depth, depth, {});
    }
    for (unsigned axis = 0; axis < depth; ++axis) { append(last(), first(), depth, depth, axis); }
    if (!regionOnly) { append(last(), exit, depth, 0, {}); }
    SmallVector<int64_t> bodySites(bodies.begin(), bodies.end());
    NamedAttrList result;
    result.set("representation", builder.getStringAttr("ordered-rectangular-frame-boundaries"));
    result.set("protocol", builder.getStringAttr(repeatedPair() ?
        "repeated-ready-release-with-frame-reentry" : "one-generation-with-inner-reentry"));
    result.set("minimum", builder.getArrayAttr(minimum));
    result.set("frames", builder.getArrayAttr(frameRecords));
    result.set("frame_sources", builder.getDenseI64ArrayAttr(frameSources));
    result.set("parameter_arguments", builder.getDenseI64ArrayAttr(parameters));
    result.set("body_sites", builder.getDenseI64ArrayAttr(bodySites));
    result.set("frame_depth", builder.getI64IntegerAttr(depth));
    SmallVector<Attribute> queries;
    for (bool sourceLast : {false, true}) {
        for (bool targetLast : {false, true}) {
            for (auto sourceKind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                for (auto targetKind : {PeriodicEventKind::Start, PeriodicEventKind::Completion}) {
                    auto guard = framePortQuery(sourceLast, sourceKind, targetLast, targetKind);
                    if (failed(guard)) { continue; }
                    queries.push_back(builder.getDictionaryAttr({
                        builder.getNamedAttr("source_last", builder.getBoolAttr(sourceLast)),
                        builder.getNamedAttr("target_last", builder.getBoolAttr(targetLast)),
                        builder.getNamedAttr("source_completion", builder.getBoolAttr(
                            sourceKind == PeriodicEventKind::Completion)),
                        builder.getNamedAttr("target_completion", builder.getBoolAttr(
                            targetKind == PeriodicEventKind::Completion)),
                        builder.getNamedAttr("guard", *guard)}));
                }
            }
        }
    }
    result.set("frame_port_queries", builder.getArrayAttr(queries));
    result.set("region_only", builder.getBoolAttr(regionOnly));
    result.set("activation_binding", activationBinding ? activationBinding : builder.getDictionaryAttr({}));
    result.set("activation_outcome", builder.getBoolAttr(activationOutcome));
    result.set("chain_witnesses", chainWitnesses);
    if (repeatedPair()) {
        result.set("original_barrier_source", builder.getI64IntegerAttr(originalBarrierSource));
        result.set("source_barrier_equivalence", builder.getStringAttr(
            "original MTE2 drain independently qualifies C-order; complete SharedUnion chain implies extra C/I edges"));
    }
    result.set("shared_slot_witnesses", selectorWitnesses);
    result.set("closure_certificate", builder.getStringAttr(
        "complete SharedUnion total lexical chain; frame carries preserve persistent storage"));
    return builder.getDictionaryAttr(result);
}
DictionaryAttr SingleStreamLoop::evidence(MLIRContext* context) const
{
    Builder builder(context);
    if (protocol == SingleStreamProtocol::SequentialBoundaries ||
        protocol == SingleStreamProtocol::ExclusiveArmBoundaries) {
        const bool choice = protocol == SingleStreamProtocol::ExclusiveArmBoundaries;
        SmallVector<Attribute> minimum, childEvidence;
        // Every row binds [source ordinal?, consumer ordinal?, n1,n2,active?,constant].
        auto remap = [&](ArrayRef<int64_t> rows, unsigned depth, unsigned child) {
            SmallVector<int64_t> result;
            const unsigned oldWidth = depth + 2 + unsigned(bool(activation));
            for (unsigned start = 0; start < rows.size(); start += oldWidth) {
                llvm::append_range(result, rows.slice(start, depth));
                result.push_back(child == 0 ? rows[start + depth] : 0);
                result.push_back(child == 1 ? rows[start + depth] : 0);
                if (activation) { result.push_back(rows[start + depth + 1]); }
                result.push_back(rows[start + oldWidth - 1]);
            }
            return result;
        };
        for (unsigned child = 0; child < 2; ++child) {
            auto evidence = children[child]->evidence(context); childEvidence.push_back(evidence);
            for (auto attribute : evidence.getAs<ArrayAttr>("minimum")) {
                auto original = cast<DictionaryAttr>(attribute);
                const auto depth = original.getAs<IntegerAttr>("source_depth").getInt() +
                                   original.getAs<IntegerAttr>("consumer_depth").getInt();
                NamedAttrList copied(original);
                copied.set("equalities", builder.getDenseI64ArrayAttr(remap(
                    original.getAs<DenseI64ArrayAttr>("equalities").asArrayRef(), depth, child)));
                copied.set("inequalities", builder.getDenseI64ArrayAttr(remap(
                    original.getAs<DenseI64ArrayAttr>("inequalities").asArrayRef(), depth, child)));
                minimum.push_back(builder.getDictionaryAttr(copied));
            }
        }
        for (auto attribute : crossingMinimum) {
            auto crossing = cast<DictionaryAttr>(attribute);
            auto source = crossing.getAs<IntegerAttr>("source").getInt();
            auto consumer = crossing.getAs<IntegerAttr>("consumer").getInt();
            const unsigned sourceDepth = source == int64_t(entry) ? 0 : 1;
            const unsigned targetDepth = consumer == int64_t(exit) ? 0 : 1;
            const unsigned width = sourceDepth + targetDepth + 3 + unsigned(bool(activation));
            for (auto guardAttribute : crossing.getAs<ArrayAttr>("guard")) {
                auto guard = cast<DictionaryAttr>(guardAttribute);
                SmallVector<int64_t> equalities, inequalities;
                auto appendGuard = [&](ArrayRef<int64_t> rows, SmallVector<int64_t>& output) {
                    const unsigned guardWidth = choice ? 4 : 3;
                    for (unsigned i = 0; i < rows.size(); i += guardWidth) {
                        output.append(sourceDepth + targetDepth, 0);
                        output.push_back(rows[i]); output.push_back(rows[i + 1]);
                        if (activation) { output.push_back(choice ? rows[i + 2] : 0); }
                        output.push_back(rows[i + guardWidth - 1]);
                    }
                };
                appendGuard(guard.getAs<DenseI64ArrayAttr>("equalities").asArrayRef(), equalities);
                appendGuard(guard.getAs<DenseI64ArrayAttr>("inequalities").asArrayRef(), inequalities);
                if (sourceDepth) {
                    SmallVector<int64_t> row(width, 0); row[0] = 1;
                    const unsigned child = llvm::is_contained(children[0]->bodies, source) ? 0 : 1;
                    row[sourceDepth + targetDepth + child] = -1; row.back() = 1;
                    llvm::append_range(equalities, row);
                }
                if (targetDepth) {
                    SmallVector<int64_t> row(width, 0); row[sourceDepth] = 1;
                    llvm::append_range(equalities, row);
                }
                if (activation && !choice) {
                    SmallVector<int64_t> row(width, 0); row[width - 2] = 1; row.back() = -1;
                    llvm::append_range(equalities, row);
                }
                minimum.push_back(builder.getDictionaryAttr({
                    builder.getNamedAttr("source", builder.getI64IntegerAttr(source)),
                    builder.getNamedAttr("consumer", builder.getI64IntegerAttr(consumer)),
                    builder.getNamedAttr("source_depth", builder.getI64IntegerAttr(sourceDepth)),
                    builder.getNamedAttr("consumer_depth", builder.getI64IntegerAttr(targetDepth)),
                    builder.getNamedAttr("equalities", builder.getDenseI64ArrayAttr(equalities)),
                    builder.getNamedAttr("inequalities", builder.getDenseI64ArrayAttr(inequalities))}));
            }
        }
        NamedAttrList attrs;
        attrs.set("representation", builder.getStringAttr(choice ? "exclusive-arm-shared-port-boundaries" :
                                                                 "two-child-shared-port-boundaries"));
        attrs.set("protocol", builder.getStringAttr("exclusive-once-boundary-alternatives"));
        attrs.set("minimum", builder.getArrayAttr(minimum)); attrs.set("children", builder.getArrayAttr(childEvidence));
        attrs.set("parameter_arguments", builder.getDenseI64ArrayAttr(
            {children[0]->boundArgument, children[1]->boundArgument}));
        attrs.set("activation_binding", activationBinding ? activationBinding :
            builder.getDictionaryAttr({builder.getNamedAttr("origin", builder.getStringAttr("none"))}));
        attrs.set("participation_source", builder.getI64IntegerAttr(participationSource));
        attrs.set("publication_cut_source", builder.getI64IntegerAttr(publicationCutSource));
        attrs.set("crossing_minimum", crossingMinimum); attrs.set("port_closure", portClosure);
        attrs.set("closure_updates", builder.getI64IntegerAttr(portClosureUpdates));
        attrs.set("closure_maximum_pieces", builder.getI64IntegerAttr(portClosureMaximumPieces));
        attrs.set("closure_size_measure", builder.getStringAttr(
            "maximum stored reach-cell disjuncts after coalesce; not transient or solver complexity"));
        attrs.set("chain_witnesses", chainWitnesses); attrs.set("shared_slot_witnesses", selectorWitnesses);
        attrs.set("static_pair_alternatives", builder.getI64IntegerAttr(4));
        attrs.set("static_endpoint_templates", builder.getI64IntegerAttr(8));
        attrs.set("executed_boundary_pairs", builder.getStringAttr(choice ?
            "(active ? n1 : n2)>0 ? 2 : 0" : "active && (n1>0 || n2>0) ? 2 : 0"));
        attrs.set("closure_certificate", builder.getStringAttr(
            "complete SharedUnion forward requirements; witnessed total chain"));
        return builder.getDictionaryAttr(attrs);
    }
    auto piece = [&](int64_t source, int64_t consumer, ArrayRef<int64_t> equalities,
                     ArrayRef<int64_t> inequalities, unsigned sourceDepth, unsigned targetDepth) {
        // Row columns: [source ordinal?, target ordinal?, n, active?, constant].
        const unsigned oldWidth = sourceDepth + targetDepth + 2;
        auto guardedRows = [&](ArrayRef<int64_t> rows) {
            SmallVector<int64_t> result;
            for (unsigned begin = 0; begin < rows.size(); begin += oldWidth) {
                llvm::append_range(result, rows.slice(begin, oldWidth - 1));
                if (activation) { result.push_back(0); }
                result.push_back(rows[begin + oldWidth - 1]);
            }
            return result;
        };
        auto selectedEqualities = guardedRows(equalities);
        auto selectedInequalities = guardedRows(inequalities);
        if (activation) {
            selectedEqualities.append(oldWidth - 1, 0);
            selectedEqualities.push_back(1); selectedEqualities.push_back(-int64_t(activationOutcome));
        }
        return builder.getDictionaryAttr({
            builder.getNamedAttr("source", builder.getI64IntegerAttr(source)),
            builder.getNamedAttr("consumer", builder.getI64IntegerAttr(consumer)),
            builder.getNamedAttr("source_depth", builder.getI64IntegerAttr(sourceDepth)),
            builder.getNamedAttr("consumer_depth", builder.getI64IntegerAttr(targetDepth)),
            builder.getNamedAttr("equalities", builder.getDenseI64ArrayAttr(selectedEqualities)),
            builder.getNamedAttr("inequalities", builder.getDenseI64ArrayAttr(selectedInequalities))});
    };
    SmallVector<Attribute> minimum;
    if (!regionOnly && (!repeatedPair() || hasPrefix)) {
        minimum.push_back(piece(entry, first(), {1, 0, 0}, {0, 1, -1}, 0, 1));
    }
    for (std::size_t position = 0; position + 1 < bodies.size(); ++position) {
        minimum.push_back(piece(bodies[position], bodies[position + 1], {-1, 1, 0, 0},
                                {1, 0, 0, 0, 0, -1, 1, -1}, 1, 1));
    }
    minimum.push_back(piece(last(), first(), {-1, 1, 0, -1}, {1, 0, 0, 0, 0, -1, 1, -1}, 1, 1));
    if (!regionOnly) { minimum.push_back(piece(last(), exit, {1, -1, 1}, {0, 1, -1}, 1, 0)); }
    NamedAttrList attributes;
    if (repeatedPair()) {
        attributes.set("original_barrier_source", builder.getI64IntegerAttr(originalBarrierSource));
        attributes.set("source_barrier_equivalence", builder.getStringAttr(
            "original drain previous P -> current P is implied by original READY and RELEASE storage chain"));
        attributes.set("protocol", builder.getStringAttr("repeated-ready-release-and-final-exit"));
    }
    auto port = [&](const SingleStreamBoundPort& endpoint) {
        return builder.getDictionaryAttr({
            builder.getNamedAttr("site", builder.getI64IntegerAttr(endpoint.site)),
            builder.getNamedAttr("ordinal_coefficient", builder.getI64IntegerAttr(endpoint.coefficient)),
            builder.getNamedAttr("ordinal_offset", builder.getI64IntegerAttr(endpoint.offset)),
            builder.getNamedAttr("activation_outcome", builder.getBoolAttr(activationOutcome)),
            builder.getNamedAttr("activation_argument", builder.getI64IntegerAttr(activationArgument)),
            builder.getNamedAttr("activation_binding", activationBinding ? activationBinding :
                builder.getDictionaryAttr({builder.getNamedAttr("origin", builder.getStringAttr("none"))})),
            builder.getNamedAttr("participation_source", builder.getI64IntegerAttr(participationSource)),
            builder.getNamedAttr("presence_parameter_count", builder.getI64IntegerAttr(activation ? 2 : 1)),
            builder.getNamedAttr("presence_equalities", builder.getDenseI64ArrayAttr(
                activation ? ArrayRef<int64_t>{0, 1, -int64_t(activationOutcome)} : ArrayRef<int64_t>{})),
            builder.getNamedAttr("presence_inequalities", builder.getDenseI64ArrayAttr(
                activation ? ArrayRef<int64_t>{1, 0, -1} : ArrayRef<int64_t>{1, -1})),
            builder.getNamedAttr("presence", builder.getStringAttr(activation ?
                (activationOutcome ? "active && n > 0" : "!active && n > 0") : "n > 0"))});
    };
    attributes.set("bound_ports", builder.getArrayAttr({port(firstPort()), port(lastPort())}));
    attributes.set("region_only", builder.getBoolAttr(regionOnly));
    attributes.set("prefix_realization", builder.getStringAttr(hasPrefix ?
        "first original MTE2 barrier; no inserted command" : "none"));
    attributes.set("representation", builder.getStringAttr("single-stream-affine-boundaries"));
    attributes.set("minimum", builder.getArrayAttr(minimum));
    attributes.set("parameter_argument", builder.getI64IntegerAttr(boundArgument));
    attributes.set("activation_outcome", builder.getBoolAttr(activationOutcome));
    attributes.set("activation_argument", builder.getI64IntegerAttr(activationArgument));
    if (activationBinding) { attributes.set("activation_binding", activationBinding); }
    attributes.set("participation_source", builder.getI64IntegerAttr(participationSource));
    if (boundConstant) { attributes.set("parameter_constant", boundConstant); }
    if (frames.size() > 1) { return frameEvidence(context); }
    attributes.set("publication_cut_source", builder.getI64IntegerAttr(publicationCutSource));
    attributes.set("loop_source", builder.getI64IntegerAttr(loopSource));
    attributes.set("guard_source", builder.getI64IntegerAttr(guardSource));
    attributes.set("slot_selection_sources", builder.getDenseI64ArrayAttr(selectorSources));
    SmallVector<int64_t> bodySites(bodies.begin(), bodies.end());
    attributes.set("body_sites", builder.getDenseI64ArrayAttr(bodySites));
    attributes.set("exit_banks", builder.getI64IntegerAttr(banks));
    attributes.set("shared_slot_witnesses", selectorWitnesses);
    attributes.set("closure_certificate", builder.getStringAttr(
        "complete shared union requirements dominated by witnessed lexical and wrap hazards"));
    attributes.set("chain_witnesses", chainWitnesses);
    attributes.set("alias_precision", builder.getStringAttr("union modeled covers; physical closure may be weaker"));
    attributes.set("effect_domain", builder.getStringAttr("shared-modeled"));
    attributes.set("fragment_model", builder.getStringAttr(
        repeatedPair() ? "original shared READY/RELEASE generating chain; complete union closure equivalence" :
        "auxiliary value atoms and shared bank coordinates; closure-equivalent to complete union requirements"));
    attributes.set("overwrite", builder.getStringAttr("not-qualified"));
    return attributes.getDictionary(context);
}
} // namespace mlir::pto::frontiersynch
