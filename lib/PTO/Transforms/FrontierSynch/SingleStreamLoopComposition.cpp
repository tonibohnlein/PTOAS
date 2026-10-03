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
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include <algorithm>
#include <optional>
namespace mlir::pto::frontiersynch {
namespace {
bool constant(Value value, int64_t expected)
{
    APInt integer;
    return matchPattern(value, m_ConstantInt(&integer)) && integer == expected;
}
} // namespace
namespace {
using PortGuard = presburger::PresburgerSet;
PortGuard portGuard(ArrayRef<int64_t> rows, bool choice = false)
{
    presburger::IntegerPolyhedron poly(presburger::PresburgerSpace::getSetSpace(choice ? 3 : 2));
    const unsigned width = choice ? 4 : 3;
    for (unsigned i = 0; i < rows.size(); i += width) { poly.addInequality(rows.slice(i, width)); }
    return PortGuard(poly);
}
ArrayAttr guardRows(MLIRContext* context, const PortGuard& guard, bool activation = false)
{
    Builder builder(context);
    SmallVector<Attribute> disjuncts;
    for (const auto& poly : guard.getAllDisjuncts()) {
        SmallVector<int64_t> equalities, inequalities;
        for (unsigned i = 0; i < poly.getNumEqualities(); ++i) {
            for (const auto& coefficient : poly.getEquality(i)) {
                equalities.push_back(static_cast<int64_t>(coefficient));
            }
        }
        for (unsigned i = 0; i < poly.getNumInequalities(); ++i) {
            for (const auto& coefficient : poly.getInequality(i)) {
                inequalities.push_back(static_cast<int64_t>(coefficient));
            }
        }
        if (activation) {
            auto addAxis = [](SmallVector<int64_t>& rows) {
                SmallVector<int64_t> result;
                for (unsigned i = 0; i < rows.size(); i += 3) {
                    result.append({rows[i], rows[i + 1], 0, rows[i + 2]});
                }
                rows = std::move(result);
            };
            addAxis(equalities); addAxis(inequalities);
            equalities.append({0, 0, 1, -1});
        }
        disjuncts.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("equalities", builder.getDenseI64ArrayAttr(equalities)),
            builder.getNamedAttr("inequalities", builder.getDenseI64ArrayAttr(inequalities))}));
    }
    return builder.getArrayAttr(disjuncts);
}
// Twelve I/C ports, not twelve payload occurrences. Child first/last aliases
// are joined exactly on n=1; absent ports have no identity or bypass event.
struct SharedPortClosure {
    static constexpr unsigned size = 12;
    SmallVector<PortGuard, 0> reach, identity;
    std::size_t updates = 0, maximumPieces = 0;
    explicit SharedPortClosure(bool choice = false)
    {
        auto empty = PortGuard::getEmpty(presburger::PresburgerSpace::getSetSpace(choice ? 3 : 2));
        reach.assign(size * size, empty); identity.assign(size * size, empty);
    }
    void edge(unsigned source, unsigned target, const PortGuard& guard)
    {
        if (!guard.getNumDisjuncts()) { return; }
        auto& value = reach[source * size + target];
        value = value.unionSet(guard).coalesce();
        ++updates; maximumPieces = std::max<std::size_t>(maximumPieces, value.getNumDisjuncts());
    }
    void same(unsigned source, unsigned target, const PortGuard& guard)
    { identity[source * size + target].unionInPlace(guard); edge(source, target, guard); }
    void close()
    {
        for (unsigned middle = 0; middle < size; ++middle) {
            for (unsigned source = 0; source < size; ++source) {
                if (reach[source * size + middle].getNumDisjuncts() == 0) { continue; }
                for (unsigned target = 0; target < size; ++target) {
                    if (reach[middle * size + target].getNumDisjuncts() == 0) { continue; }
                    edge(source, target, reach[source * size + middle].intersect(reach[middle * size + target]));
                }
            }
        }
    }
    PortGuard cover(unsigned source, unsigned target, const PortGuard& requirement) const
    {
        auto result = requirement;
        for (unsigned middle = 0; middle < size; ++middle) {
            auto path = reach[source * size + middle].intersect(reach[middle * size + target]);
            // An alias of either endpoint is not a distinct intermediate.
            path = path.subtract(identity[source * size + middle].unionSet(identity[target * size + middle]));
            result = result.subtract(path);
        }
        return result.coalesce();
    }
};
} // namespace
LogicalResult SingleStreamLoop::qualifySequentialSource(func::FuncOp function, const SyncInput& input,
    const TraceDemandAnalysis& trace, std::string& reason)
{
    const auto& a = children[0]; const auto& b = children[1];
    if (!a || !b || !a->regionOnly || !b->regionOnly || a->repeatedPair() || b->repeatedPair() ||
        a->bodies.size() != 1 || b->bodies.size() != 1 || a->first() != 1 || b->first() != 2 ||
        a->loop == b->loop || a->bound == b->bound || a->banks != b->banks ||
        a->participation != b->participation || a->activation != b->activation ||
        !a->activationOutcome || !b->activationOutcome) { return failure(); }
    auto* scope = a->participation ? &cast<scf::IfOp>(a->participation).getThenRegion().front() : &function.front();
    if (a->loop->getBlock() != scope || b->loop->getBlock() != scope || !a->loop->isBeforeInBlock(b->loop)) {
        return failure();
    }
    const auto* entry = trace.sites()[0].phase; const auto* exit = trace.sites()[3].phase;
    auto guard = exit->elementOp->getParentOfType<scf::IfOp>();
    auto any = guard ? guard.getCondition().getDefiningOp<arith::OrIOp>() : arith::OrIOp{};
    auto firstPositive = any ? any.getLhs().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
    auto secondPositive = any ? any.getRhs().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
    auto positive = [&](arith::CmpIOp cmp, Value bound) {
        return cmp && cmp.getPredicate() == arith::CmpIPredicate::sgt &&
               cmp.getLhs() == bound && constant(cmp.getRhs(), 0);
    };
    if (entry->kPipeValue != PipelineType::PIPE_MTE2 || exit->kPipeValue != PipelineType::PIPE_MTE3 ||
        entry->elementOp->getBlock() != &function.front() || !guard || guard->getBlock() != scope ||
        !b->loop->isBeforeInBlock(guard) || !positive(firstPositive, a->bound) || !positive(secondPositive, b->bound) ||
        exit->elementOp->getBlock() != &guard.getThenRegion().front() ||
        (!guard.getElseRegion().empty() && !guard.getElseRegion().front().without_terminator().empty())) {
        return failure();
    }
    const auto& participation = input.target().participation();
    if (a->participation) {
        if (!participation.thenOnly() || a->participation != participation.owner ||
            failed(entryCut(input, function, entry->elementOp, a->loop, participation, reason))) {
            return failure();
        }
    } else if (!entry->elementOp->isBeforeInBlock(a->loop) ||
               !prefixPure(entry->elementOp->getNextNode(), a->loop)) { return failure(); }
    if (!input.target().valueAvailable(b->bound, a->loop, false) ||
        !prefixPure(a->loop->getNextNode(), b->loop)) { return failure(); }
    bool extra = false;
    function.walk([&](Operation* operation) {
        if ((operation != function && operation != a->loop && operation != b->loop && operation != guard &&
             operation != a->participation && operation->getNumRegions()) || isa<BarrierOp>(operation)) {
            extra = true;
        }
    });
    if (extra) { return failure(); }
    Value family;
    for (const auto& child : children) {
        const SyncSlotSelection* output = nullptr;
        for (const auto* memory : trace.sites()[child->last()].phase->defVec) {
            if (auto* selected = input.slotSelection(memory->baseBuffer)) { output = selected; }
        }
        if (!output) { return failure(); }
        auto source = output->selected.getDefiningOp<MultiTileGetOp>().getSource();
        if (family && family != source) { return failure(); }
        family = source;
    }
    bool finalRead = false;
    for (const auto* memory : exit->useVec) {
        auto selected = memory->baseBuffer.getDefiningOp<MultiTileGetOp>();
        if (!selected || selected.getSource() != family) { continue; }
        auto modulo = selected.getSlot().getDefiningOp<arith::RemUIOp>();
        auto last = modulo ? modulo.getLhs().getDefiningOp<arith::SubIOp>() : arith::SubIOp{};
        auto bound = last ? last.getLhs().getDefiningOp<arith::SelectOp>() : arith::SelectOp{};
        // Select the positive original bound INSIDE anyNonempty, THEN subtract.
        // Even an unselected INT_MIN bound is never decremented in emitted C++.
        if (!modulo || !constant(modulo.getRhs(), a->banks) || !last || !constant(last.getRhs(), 1) || !bound ||
            bound->getBlock() != &guard.getThenRegion().front() || bound.getCondition() != secondPositive ||
            bound.getTrueValue() != b->bound || bound.getFalseValue() != a->bound) { return failure(); }
        finalRead = true;
    }
    if (!finalRead) { return failure(); }
    protocol = SingleStreamProtocol::SequentialBoundaries;
    loop = a->loop; exitGuard = guard;
    this->entry = 0; this->exit = 3; bodies = {1, 2}; banks = a->banks;
    activation = a->activation; this->participation = a->participation;
    activationArgument = a->activationArgument; activationBinding = a->activationBinding;
    participationSource = a->participationSource;
    publicationCut = a->loop; publicationCutSource = input.target().sourceIdentity(a->loop).value_or(-1);
    guardSource = input.target().sourceIdentity(guard).value_or(-1);
    selectorWitnesses = input.slotAttribute(function.getContext());
    if (publicationCutSource < 0 || guardSource < 0) { return failure(); }
    return success();
}
LogicalResult SingleStreamLoop::qualifyExclusiveSource(func::FuncOp function, const SyncInput& input,
    const TraceDemandAnalysis& trace, std::string& reason)
{
    const auto& a = children[0]; const auto& b = children[1];
    const auto& captured = input.target().participation();
    auto choice = dyn_cast_or_null<scf::IfOp>(captured.owner);
    if (!a || !b || !choice || !a->regionOnly || !b->regionOnly || a->repeatedPair() || b->repeatedPair() ||
        a->bodies.empty() || b->bodies.empty() || a->first() != 1 || b->first() != a->last() + 1 ||
        a->bound == b->bound || a->banks != b->banks || a->participation != choice || b->participation != choice ||
        a->activation != captured.condition || b->activation != captured.condition ||
        !a->activationOutcome || b->activationOutcome || a->armRegion != &choice.getThenRegion() ||
        b->armRegion != &choice.getElseRegion() || a->loop->getParentRegion() != a->armRegion ||
        b->loop->getParentRegion() != b->armRegion || choice->getBlock() != &function.front()) { return failure(); }
    if (trace.sites().size() != a->bodies.size() + b->bodies.size() + 2) { return failure(); }
    for (const auto& child : children) {
        for (auto [position, site] : llvm::enumerate(child->bodies)) {
            if (site != child->first() + position || site >= trace.sites().size() - 1 ||
                trace.sites()[site].anchor->getParentOp() != child->loop) { return failure(); }
        }
    }
    const auto exitSite = trace.sites().size() - 1;
    const auto* entryPhase = trace.sites()[0].phase; const auto* exitPhase = trace.sites()[exitSite].phase;
    auto guard = exitPhase->elementOp->getParentOfType<scf::IfOp>();
    auto positive = guard ? guard.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
    auto selected = positive ? positive.getLhs().getDefiningOp<arith::SelectOp>() : arith::SelectOp{};
    if (entryPhase->kPipeValue != PipelineType::PIPE_MTE2 || exitPhase->kPipeValue != PipelineType::PIPE_MTE3 ||
        entryPhase->elementOp->getBlock() != &function.front() || !guard || guard->getBlock() != &function.front() ||
        !choice->isBeforeInBlock(guard) || !entryPhase->elementOp->isBeforeInBlock(choice) ||
        !positive || positive.getPredicate() != arith::CmpIPredicate::sgt || !constant(positive.getRhs(), 0) ||
        !selected || selected.getCondition() != captured.condition || selected.getTrueValue() != a->bound ||
        selected.getFalseValue() != b->bound || exitPhase->elementOp->getBlock() != &guard.getThenRegion().front() ||
        (!guard.getElseRegion().empty() && !guard.getElseRegion().front().without_terminator().empty()) ||
        !prefixPure(entryPhase->elementOp->getNextNode(), choice) ||
        !prefixPure(choice->getNextNode(), guard)) { return failure(); }
    for (const auto& child : children) {
        if (!input.target().valueAvailable(captured.condition, child->loop, false) ||
            !input.target().valueAvailable(child->bound, child->loop, false) ||
            !prefixPure(&child->armRegion->front().front(), child->loop) ||
            !prefixPure(child->loop->getNextNode(), child->armRegion->front().getTerminator())) {
            reason = "exclusive publication cut has intervening prerequisites or unavailable values";
            return failure();
        }
    }
    bool extra = false;
    function.walk([&](Operation* operation) {
        if ((operation != function && operation != a->loop && operation != b->loop && operation != choice &&
             operation != guard && operation->getNumRegions()) || isa<BarrierOp>(operation)) { extra = true; }
    });
    if (extra) { return failure(); }
    Value family;
    for (const auto& child : children) {
        const SyncSlotSelection* output = nullptr;
        for (const auto* memory : trace.sites()[child->last()].phase->defVec) {
            if (auto* candidate = input.slotSelection(memory->baseBuffer)) { output = candidate; }
        }
        if (!output) { return failure(); }
        auto source = output->selected.getDefiningOp<MultiTileGetOp>().getSource();
        if (family && family != source) { return failure(); }
        family = source;
    }
    bool finalRead = false;
    for (const auto* memory : exitPhase->useVec) {
        auto tile = memory->baseBuffer.getDefiningOp<MultiTileGetOp>();
        if (!tile || tile.getSource() != family) { continue; }
        auto modulo = tile.getSlot().getDefiningOp<arith::RemUIOp>();
        auto last = modulo ? modulo.getLhs().getDefiningOp<arith::SubIOp>() : arith::SubIOp{};
        if (!modulo || !constant(modulo.getRhs(), a->banks) || !last || !constant(last.getRhs(), 1) ||
            last.getLhs() != selected.getResult() || last->getBlock() != &guard.getThenRegion().front()) {
            return failure();
        }
        finalRead = true;
    }
    if (!finalRead) { return failure(); }
    protocol = SingleStreamProtocol::ExclusiveArmBoundaries;
    loop = a->loop; exitGuard = guard; entry = 0; exit = exitSite; banks = a->banks;
    bodies.assign(a->bodies.begin(), a->bodies.end());
    llvm::append_range(bodies, b->bodies);
    activation = captured.condition; participation = captured.owner;
    activationBinding = input.target().activationAttribute();
    activationArgument = a->activationArgument; participationSource = a->participationSource;
    publicationCut = choice; publicationCutSource = input.target().sourceIdentity(choice).value_or(-1);
    guardSource = input.target().sourceIdentity(guard).value_or(-1);
    selectorWitnesses = input.slotAttribute(function.getContext());
    return success(publicationCutSource >= 0 && guardSource >= 0);
}
LogicalResult SingleStreamLoop::buildSequentialClosure(MLIRContext* context, std::string& reason)
{
    const bool choice = protocol == SingleStreamProtocol::ExclusiveArmBoundaries;
    const SmallVector<std::pair<unsigned, unsigned>> pairs = choice ?
        SmallVector<std::pair<unsigned, unsigned>>{{0, 1}, {0, 2}, {1, 3}, {2, 3}} :
        SmallVector<std::pair<unsigned, unsigned>>{{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}};
    SharedPortClosure closure(choice);
    const auto universe = portGuard({}, choice);
    const auto positiveA = choice ? portGuard({1, 0, 0, -1, 0, 0, 1, -1, 0, 0, -1, 1}, true) :
                                   portGuard({1, 0, -1});
    const auto positiveB = choice ? portGuard({0, 1, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0}, true) :
                                   portGuard({0, 1, -1});
    const auto both = positiveA.intersect(positiveB), anyPositive = positiveA.unionSet(positiveB);
    // E I/C, A first I/C, A last I/C, B first I/C, B last I/C, X I/C.
    closure.same(0, 0, universe); closure.same(1, 1, universe); closure.edge(0, 1, universe);
    closure.same(10, 10, anyPositive); closure.same(11, 11, anyPositive); closure.edge(10, 11, anyPositive);
    for (unsigned child = 0; child < 2; ++child) {
        const auto present = child ? positiveB : positiveA;
        for (unsigned from = 0; from < 4; ++from) {
            for (unsigned to = 0; to < 4; ++to) {
                auto query = children[child]->portQuery(from >= 2,
                    from % 2 ? PeriodicEventKind::Completion : PeriodicEventKind::Start,
                    to >= 2, to % 2 ? PeriodicEventKind::Completion : PeriodicEventKind::Start);
                if (failed(query) || query->bound != children[child]->bound || query->activation != activation ||
                    query->armRegion != children[child]->armRegion ||
                    query->activationOutcome != children[child]->activationOutcome) {
                        return failure();
                    }
                const auto source = 2 + child * 4 + from, target = 2 + child * 4 + to;
                if (query->reachable) {
                    SmallVector<int64_t> row{child ? 0 : query->coefficient,
                                             child ? query->coefficient : 0};
                    if (choice) { row.push_back(0); }
                    row.push_back(query->constant);
                    closure.edge(source, target, present.intersect(portGuard(row, choice)));
                }
                if (from % 2 == to % 2) {
                    auto same = from == to ? present :
                        children[child]->first() != children[child]->last() ? PortGuard::getEmpty(present.getSpace()) :
                        present.intersect(
                        choice ? (child ? portGuard({0, -1, 0, 1}, true) : portGuard({-1, 0, 0, 1}, true)) :
                        (child ? portGuard({0, -1, 1}) : portGuard({-1, 0, 1})));
                    closure.same(source, target, same);
                }
            }
        }
    }
    // Native crossings are I/I and C/C, never a fabricated C/I bypass.
    for (unsigned from = 2; !choice && from < 6; ++from) {
        for (unsigned to = 6; to < 10; ++to) {
            if (from % 2 == to % 2) { closure.edge(from, to, both); }
        }
    }
    const SmallVector<unsigned> sources = choice ? SmallVector<unsigned>{1, 1, 5, 9} :
                                                  SmallVector<unsigned>{1, 1, 5, 5, 9};
    const SmallVector<unsigned> targets = choice ? SmallVector<unsigned>{2, 6, 10, 10} :
                                                  SmallVector<unsigned>{2, 6, 6, 10, 10};
    const SmallVector<PortGuard, 0> original = choice ?
        SmallVector<PortGuard, 0>{positiveA, positiveB, positiveA, positiveB} :
        SmallVector<PortGuard, 0>{positiveA, positiveB, both, positiveA, positiveB};
    for (unsigned i = 0; i < pairs.size(); ++i) { closure.edge(sources[i], targets[i], original[i]); }
    closure.close();
    const SmallVector<PortGuard, 0> expected = choice ? original :
        SmallVector<PortGuard, 0>{positiveA, positiveB.subtract(positiveA), both,
                                positiveA.subtract(positiveB), positiveB};
    Builder builder(context); SmallVector<Attribute> minimum, queries;
    for (unsigned i = 0; i < pairs.size(); ++i) {
        auto cover = closure.cover(sources[i], targets[i], original[i]);
        if (!cover.isEqual(expected[i])) {
            reason = "shared-port reduction differs from adjacent crossings"; return failure();
        }
        minimum.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("source", builder.getI64IntegerAttr(
                pairs[i].first == 0 ? entry : children[pairs[i].first - 1]->last())),
            builder.getNamedAttr("consumer", builder.getI64IntegerAttr(
                pairs[i].second == 3 ? exit : children[pairs[i].second - 1]->first())),
            builder.getNamedAttr("guard", guardRows(context, cover))}));
    }
    for (unsigned from = 0; from < SharedPortClosure::size; ++from) {
        for (unsigned to = 0; to < SharedPortClosure::size; ++to) {
            queries.push_back(builder.getDictionaryAttr({
                builder.getNamedAttr("source_port", builder.getI64IntegerAttr(from)),
                builder.getNamedAttr("consumer_port", builder.getI64IntegerAttr(to)),
                builder.getNamedAttr("guard", guardRows(
                    context, closure.reach[from * 12 + to].coalesce(), bool(activation) && !choice))}));
        }
    }
    crossingMinimum = builder.getArrayAttr(minimum); portClosure = builder.getArrayAttr(queries);
    portClosureUpdates = closure.updates; portClosureMaximumPieces = closure.maximumPieces;
    return success();
}
FailureOr<std::shared_ptr<const SingleStreamLoop>> SingleStreamLoop::composeSequentialRegions(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
    ArrayRef<std::shared_ptr<const SingleStreamLoop>> children, CostLedger& costs, std::string& reason)
{
    std::optional<CostScope> qualification;
    qualification.emplace(costs, CostStage::Effects);
    reason = "two-child shared-port composition premises missing";
    if (children.size() != 2 || trace.sites().size() != 4) {
        return failure();
    }
    auto result = std::shared_ptr<SingleStreamLoop>(new SingleStreamLoop());
    result->children.assign(children.begin(), children.end());
    if (failed(result->qualifySequentialSource(function, input, trace, reason))) { return failure(); }
    // Every original forward SharedUnion requirement is implied by these real
    // adjacent hazards; conversely each link is an original shared requirement.
    // This closes the full modeled graph, not only selected rotating operands.
    const std::pair<unsigned, unsigned> pairs[] = {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 3}};
    SmallVector<Attribute> witnesses;
    for (auto [source, target] : pairs) {
        auto witness = crossingWitness(input, trace.sites()[source].phase, trace.sites()[target].phase,
                                        source, target, 0, function.getContext());
        if (failed(witness)) { reason = "actual first/last or inter-child modeled hazard missing"; return failure(); }
        witnesses.push_back(*witness);
    }
    result->chainWitnesses = Builder(function.getContext()).getArrayAttr(witnesses);
    qualification.reset();
    {
        CostScope backend(costs, CostStage::Backend);
        if (failed(result->buildSequentialClosure(function.getContext(), reason))) { return failure(); }
    }
    reason.clear();
    return std::shared_ptr<const SingleStreamLoop>(result);
}
FailureOr<std::shared_ptr<const SingleStreamLoop>> SingleStreamLoop::composeExclusiveRegions(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
    ArrayRef<std::shared_ptr<const SingleStreamLoop>> children, CostLedger& costs, std::string& reason)
{
    std::optional<CostScope> qualification;
    qualification.emplace(costs, CostStage::Effects);
    reason = "exclusive-arm boundary composition premises missing";
    if (children.size() != 2 || trace.sites().size() < 4) {
        return failure();
    }
    auto result = std::shared_ptr<SingleStreamLoop>(new SingleStreamLoop());
    result->children.assign(children.begin(), children.end());
    if (failed(result->qualifyExclusiveSource(function, input, trace, reason))) { return failure(); }
    const std::pair<std::size_t, std::size_t> pairs[] = {
        {result->entry, children[0]->first()}, {result->entry, children[1]->first()},
        {children[0]->last(), result->exit}, {children[1]->last(), result->exit}};
    SmallVector<Attribute> witnesses;
    for (auto [source, target] : pairs) {
        auto witness = crossingWitness(input, trace.sites()[source].phase, trace.sites()[target].phase,
                                       source, target, 0, function.getContext());
        if (failed(witness)) { reason = "actual selected-arm first/last modeled hazard missing"; return failure(); }
        witnesses.push_back(*witness);
    }
    result->chainWitnesses = Builder(function.getContext()).getArrayAttr(witnesses);
    qualification.reset();
    {
        CostScope backend(costs, CostStage::Backend);
        if (failed(result->buildSequentialClosure(function.getContext(), reason))) { return failure(); }
    }
    reason.clear();
    return std::shared_ptr<const SingleStreamLoop>(result);
}
} // namespace mlir::pto::frontiersynch
