// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Counted-reader event queries from original shared phases and admitted presence.
#include "RegionalRequests.h"
#include "PTO/Transforms/InsertSync/SyncStorageBounds.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
namespace mlir::pto::frontiersynch {
LogicalResult qualifyCountedReaderPattern(SelectedAnalysis& selected, SymbolicSchemaHandle schema,
    const SyncInput& input, bool extrasEmpty, bool affineRows, CostLedger& costs, std::string& reason)
{
    auto outer = selected.loop;
    if (!outer || selected.sites.size() != 2) {
        reason = "counted reader requires two sites in a unit-period outer loop"; return failure();
    }
    auto writer = selected.sites[0], reader = selected.sites[1];
    auto sites = schema->sites();
    if (writer >= sites.size() || reader >= sites.size() || sites[writer].coordinates.size() != 1 ||
        sites[reader].coordinates.size() != 2 || sites[writer].coordinates[0] != outer.getInductionVar() ||
        sites[reader].coordinates[0] != outer.getInductionVar()) {
        reason = "counted reader original coordinate identity mismatch"; return failure();
    }
    auto coordinate = dyn_cast<BlockArgument>(sites[reader].coordinates[1]);
    auto inner = coordinate ? dyn_cast<scf::ForOp>(coordinate.getOwner()->getParentOp()) : scf::ForOp{};
    if (!inner || inner->getParentOp() != outer ||
        sites[writer].phase->elementOp->getParentOp() != outer ||
        sites[reader].phase->elementOp->getParentOp() != inner ||
        !sites[writer].phase->elementOp->isBeforeInBlock(inner)) {
        reason = "counted reader requires writer then its sole nested reader loop"; return failure();
    }
    {
        CostScope qualification(costs, CostStage::Effects);
        APInt lower, step;
        for (auto loop : {outer, inner}) {
            if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) || !lower.isZero() ||
                !matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isOne()) {
                reason = "counted reader requires qualified zero-origin unit progression"; return failure();
            }
        }
        auto count = inner.getUpperBound();
        APInt constant;
        bool nonnegative = count == outer.getInductionVar() ||
            (matchPattern(count, m_ConstantInt(&constant)) && !constant.isNegative());
        if (auto maximum = count.getDefiningOp<arith::MaxSIOp>()) {
            bool zero = llvm::any_of(maximum->getOperands(), [&](Value operand) {
                return matchPattern(operand, m_ConstantInt(&constant)) && constant.isZero();
            });
            nonnegative |= zero && !outer->isProperAncestor(maximum);
        }
        if (!nonnegative && !affineRows) {
            reason = "counted reader nonnegative affine row count not established by original source";
            return failure();
        }
        auto* p = sites[writer].phase;
        auto* q = sites[reader].phase;
        if (p->kPipeValue == q->kPipeValue ||
            !extrasEmpty) {
            reason = "counted reader requires distinct pipes and complete shared prerequisite association";
            return failure();
        }
        for (auto* phase : {p, q}) {
            for (auto memories : {ArrayRef<const BaseMemInfo*>(phase->useVec),
                                  ArrayRef<const BaseMemInfo*>(phase->defVec)}) {
                for (auto* memory : memories) {
                    auto value = memory->baseBuffer;
                    auto* definition = value ? value.getDefiningOp() : nullptr;
                    if (!value || (definition && outer->isProperAncestor(definition)) ||
                        (isa<BlockArgument>(value) && (value.getParentBlock()->getParentOp() == outer ||
                         outer->isProperAncestor(value.getParentBlock()->getParentOp())))) {
                        reason = "counted reader storage is not persistent across rows"; return failure();
                    }
                }
            }
        }
        // Extra reads are harmless internally only if every static hazard agrees
        // with this template. Their parent crossings remain in shared primitives.
        const CompoundInstanceElement* phases[] = {p, q};
        for (unsigned a = 0; a < 2; ++a) {
            for (unsigned b = 0; b < 2; ++b) {
                DepBaseMemInfoPairVec witnesses;
                bool raw = input.memory().DepBetween(phases[a]->defVec, phases[b]->useVec, witnesses);
                bool war = input.memory().DepBetween(phases[a]->useVec, phases[b]->defVec, witnesses);
                bool waw = input.memory().DepBetween(phases[a]->defVec, phases[b]->defVec, witnesses);
                if (raw != (a == 0 && b == 1) || war != (a == 1 && b == 0) ||
                    waw != (a == 0 && b == 0)) {
                    reason = "counted reader full shared hazard signature does not match"; return failure();
                }
            }
        }
    }
    return success();
}
LogicalResult buildCountedReaderQueries(SelectedAnalysis& selected, const SignedInputs& inputs,
                                       const SyncInput& input, CostLedger& costs, std::string& reason)
{
    using Kind = PeriodicEventKind;
    using Int = llvm::DynamicAPInt;
    auto space = inputs.context->space();
    auto schema = space->schema();
    if (space->period() != Int(1) ||
        failed(qualifyCountedReaderPattern(selected, schema, input, inputs.extras->empty(), false, costs, reason))) {
        return failure();
    }
    auto sites = schema->sites();
    auto writer = selected.sites[0], reader = selected.sites[1];
    CostScope backend(costs, CostStage::Backend);
    SmallVector<SignedPiece> reach, native, identity, present, generators;
    auto make = [&](std::size_t a, std::size_t b, Kind source, Kind target) {
        SignedPiece result;
        result.domain = {a, source}; result.range = {b, target};
        result.residues.assign(sites[a].coordinates.size() + sites[b].coordinates.size() +
                               schema->parameters().size(), Int(0));
        return result;
    };
    auto difference = [](SignedPiece& piece, unsigned axis, int bound) {
        SignedAtom atom;
        atom.terms = {{{SignedAxisRole::Domain, axis}, 1}, {{SignedAxisRole::Range, axis}, -1}};
        atom.bound = Int(bound); piece.atoms.push_back(atom);
    };
    auto equal = [&](SignedPiece& piece, unsigned axis) {
        difference(piece, axis, 0);
        SignedAtom reverse;
        reverse.terms = {{{SignedAxisRole::Range, axis}, 1}, {{SignedAxisRole::Domain, axis}, -1}};
        piece.atoms.push_back(reverse);
    };
    for (auto a : {writer, reader}) {
        for (auto b : {writer, reader}) {
            for (auto source : {Kind::Start, Kind::Completion}) {
                for (auto target : {Kind::Start, Kind::Completion}) {
                    auto piece = make(a, b, source, target);
                    bool acquire = source == Kind::Completion && target == Kind::Start;
                    difference(piece, 0, a == reader && b == writer ? -1 : (a == b && acquire ? -1 : 0));
                    if (a == reader && b == reader && !acquire) {
                        // Lexicographic reader order, retaining original nested axes.
                        piece.atoms.clear(); difference(piece, 0, -1); reach.push_back(piece); native.push_back(piece);
                        piece.atoms.clear(); equal(piece, 0); difference(piece, 1, 0);
                    }
                    reach.push_back(piece);
                    if (a == b && !acquire) { native.push_back(piece); }
                    if (a == b && source == target) {
                        auto id = make(a, b, source, target);
                        for (unsigned axis = 0; axis < sites[a].coordinates.size(); ++axis) { equal(id, axis); }
                        identity.push_back(id);
                    }
                }
            }
        }
    }
    for (const auto& occurrence : inputs.present->pieces()) {
        for (auto kind : {Kind::Start, Kind::Completion}) {
            auto event = occurrence; event.range.kind = kind; present.push_back(event);
        }
    }
    auto events = SignedRelation::import(space, SymbolicTuple::Unit, SymbolicTuple::Event, present);
    auto restrict = [&](ArrayRef<SignedPiece> pieces) -> SignedRelationHandle {
        auto value = SignedRelation::import(space, SymbolicTuple::Event, SymbolicTuple::Event, pieces);
        if (!value.succeeded() || !events.succeeded()) { return {}; }
        auto context = value.value->restrictContext(inputs.context);
        if (!context.succeeded()) { return {}; }
        auto domain = context.value->restrictDomain(events.value);
        if (!domain.succeeded()) { return {}; }
        auto range = domain.value->restrictRange(events.value);
        return range.succeeded() ? range.value : SignedRelationHandle{};
    };
    auto r = restrict(reach), n = restrict(native), id = restrict(identity);
    auto candidates = inputs.generators->unite(inputs.extras);
    if (!r || !n || !id || !candidates.succeeded()) {
        reason = "counted reader event/query representation obligation"; return failure();
    }
    auto ordered = candidates.value->intersect(inputs.reference);
    if (!ordered.succeeded()) { reason = "counted reader reference restriction failed"; return failure(); }
    for (auto piece : ordered.value->pieces()) {
        piece.domain.kind = Kind::Completion; piece.range.kind = Kind::Start; generators.push_back(std::move(piece));
    }
    auto g = restrict(generators);
    auto strict = r->subtract(id);
    auto alternatives = strict.succeeded() ? strict.value->compose(strict.value) : strict;
    if (!g || !alternatives.succeeded()) {
        reason = "counted reader strict-path representation obligation"; return failure();
    }
    auto local = g->subtract(n);
    auto minimum = local.succeeded() ? local.value->subtract(alternatives.value) : local;
    if (!minimum.succeeded()) { reason = "counted reader cover representation obligation"; return failure(); }
    selected.kind = SelectedAnalysis::Kind::Signed;
    selected.route = "regional-counted-readers";
    selected.context = inputs.context; selected.minimum = minimum.value;
    selected.native = n; selected.reachability = r;
    selected.contract.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
        interfaceBit(DemandInterface::RegionQueries) | interfaceBit(DemandInterface::UniformMembership);
    return success();
}
} // namespace mlir::pto::frontiersynch
