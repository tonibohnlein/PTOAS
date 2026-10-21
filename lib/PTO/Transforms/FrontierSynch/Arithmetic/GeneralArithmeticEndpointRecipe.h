// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owned exact selector recipes; no producer callback or prepared SSA escapes.
#ifndef PTO_FRONTIER_GENERALARITHMETICENDPOINTRECIPE_H
#define PTO_FRONTIER_GENERALARITHMETICENDPOINTRECIPE_H
#include "ArithmeticEndpointExpressions.h"
#include "../Shared/OccurrenceBinding.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::pto::frontiersynch::detail {
template<class Selectors>
class ArithmeticEndpointRecipeBuilder {
public:
    ArithmeticEndpointRecipeBuilder(const ArithmeticProgram& program,
        const Selectors& selectors, PreparedLogicalPlan& plan, std::string& error)
        : program(program), selectors(selectors), plan(plan), error(error) {}
    LogicalResult run(ArithmeticRecordMap& exported)
    {
        for (const auto& selector : selectors.forward) {
            for (const auto& piece : selector.pieces) {
                if (selector.inputSite >= program.sites.size() || piece.outputSite >= program.sites.size()) {
                    return failure();
                }
                const Pair pair{selector.inputSite, piece.outputSite};
                if (records.find(pair) != records.end()) { continue; }
                if (records.size() >= UINT32_MAX) { return failure(); }
                records.emplace(pair, records.size());
            }
        }
        if (failed(families())) { return failure(); }
        for (const auto& selector : selectors.forward) {
            if (failed(prepare(selector, true))) { return failure(); }
        }
        for (const auto& selector : selectors.inverse) {
            if (failed(prepare(selector, false))) { return failure(); }
        }
        if (!arena->constructionError().empty()) { error = arena->constructionError(); return failure(); }
        if (!endpoints.empty()) {
            auto recipe = buildContextualFamilyEndpointRecipe(arena, endpoints);
            if (failed(recipe)) {
                error = "arithmetic family binding unavailable at its original cut"; return failure();
            }
            plan.familyRecipe = std::move(*recipe);
        }
        plan.groupedFamilies = true;
        plan.independentPieces = true;
        exported = records;
        return success();
    }
private:
    using Id = RegionExpressions::Id;
    using Pair = std::pair<std::size_t, std::size_t>;
    struct Partner {
        Id guard = RegionExpressions::invalid;
        SmallVector<Id> coordinates;
    };
    const ArithmeticProgram& program;
    const Selectors& selectors;
    PreparedLogicalPlan& plan;
    std::string& error;
    std::shared_ptr<RegionExpressions> arena = std::make_shared<RegionExpressions>();
    ArithmeticRecordMap records;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> namespaces;
    SmallVector<ContextualEndpointFamily> endpoints;
    LogicalResult families()
    {
        for (const auto& [pair, record] : records) {
            const auto& source = program.sites[pair.first];
            const auto& target = program.sites[pair.second];
            if (!source.phase || !target.phase || !source.phase->elementOp || !target.phase->elementOp ||
                pair.first > UINT32_MAX || pair.second > UINT32_MAX) { return failure(); }
            auto* from = source.phase->elementOp->getNextNode();
            auto* to = target.phase->elementOp;
            if (!from) { error = "arithmetic family has no original publication cut"; return failure(); }
            const auto p = static_cast<uint32_t>(source.phase->kPipeValue);
            const auto q = static_cast<uint32_t>(target.phase->kPipeValue);
            EndpointFamily family;
            family.id = static_cast<uint32_t>(record);
            family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
            family.sourceCut = {from->getBlock(), from}; family.targetCut = {to->getBlock(), to};
            family.members.push_back({family.id, static_cast<uint32_t>(pair.first),
                static_cast<uint32_t>(pair.second), {}, {}});
            for (auto point : source.fixedCoordinates) {
                family.members.front().sourceCoordinates.push_back({point.loop, point.induction});
            }
            for (auto point : target.fixedCoordinates) {
                family.members.front().targetCoordinates.push_back({point.loop, point.induction});
            }
            plan.families.push_back(std::move(family));
            auto entry = namespaces.try_emplace({p, q}, static_cast<uint32_t>(record)).first;
            entry->second = std::min(entry->second, static_cast<uint32_t>(record));
        }
        return success();
    }
    template<class Piece>
    LogicalResult merge(const Piece& piece, ArrayRef<Id> inputs,
        bool needCoordinates, Partner& partner)
    {
        SmallVector<uint64_t> residues(piece.inputResidues.begin(), piece.inputResidues.end());
        llvm::append_range(residues, piece.parameterResidues);
        auto present = endpointPredicate(*arena, piece.domain, inputs, selectors.period, residues, error);
        if (present == RegionExpressions::invalid) { return failure(); }
        if (needCoordinates) {
            if (piece.outputs.size() != program.sites[piece.outputSite].loops.size()) { return failure(); }
            SmallVector<Id> tuple;
            for (const auto& coordinate : piece.outputs) {
                auto value = endpointCoordinate(*arena, coordinate, inputs, selectors.period, residues, error);
                if (value == RegionExpressions::invalid) { return failure(); }
                tuple.push_back(value);
            }
            if (partner.guard == RegionExpressions::invalid) { partner.coordinates = std::move(tuple); }
            else {
                if (partner.coordinates.size() != tuple.size()) { return failure(); }
                for (unsigned i = 0; i < tuple.size(); ++i) {
                    partner.coordinates[i] = arena->select(present, tuple[i], partner.coordinates[i]);
                }
            }
        }
        // Exact minimum selectors agree on every overlapping tagged tuple.
        partner.guard = partner.guard == RegionExpressions::invalid ? present : arena->lor(partner.guard, present);
        return success();
    }
    template<class Selector>
    LogicalResult prepare(const Selector& selector, bool forward)
    {
        if (selector.inputSite >= program.sites.size()) { return failure(); }
        const auto& site = program.sites[selector.inputSite];
        if (!site.phase || !site.phase->elementOp || selector.inputDimensions != site.loops.size() ||
            selector.parameterCount != program.parameters.size()) { return failure(); }
        const auto pipe = static_cast<uint32_t>(site.phase->kPipeValue);
        const bool local = selector.outputPipe == pipe;
        if (forward && local) { return success(); }
        auto* before = forward ? site.phase->elementOp->getNextNode() : site.phase->elementOp;
        if (!before) { return failure(); }
        SmallVector<TemplateCoordinate> coordinates;
        for (auto point : site.fixedCoordinates) { coordinates.push_back({point.loop, point.induction}); }
        auto atCut = originalFixedPointGuard(*arena, coordinates, before, error);
        if (failed(atCut)) { return failure(); }
        SmallVector<Id> inputs;
        for (auto loop : site.loops) { inputs.push_back(arena->input(loop.getInductionVar())); }
        for (auto parameter : program.parameters) {
            if (!parameter || (!parameter.getType().isIndex() && !parameter.getType().isInteger(1))) {
                return failure();
            }
            inputs.push_back(arena->input(parameter));
        }
        std::map<std::size_t, Partner> partners;
        for (const auto& piece : selector.pieces) {
            if (piece.outputSite >= program.sites.size() ||
                failed(merge(piece, inputs, !forward && !local, partners[piece.outputSite]))) { return failure(); }
        }
        // F* functionality makes distinct tagged output sites mutually exclusive.
        // Retain that exact producer evidence in the existing domain binding so
        // common insertion can share one command without inspecting predicates.
        Id selectedPartner = RegionExpressions::invalid;
        if (!local && partners.size() > 1) {
            selectedPartner = arena->constant(0);
            uint64_t position = 0;
            for (const auto& [other, partner] : partners) {
                (void)other;
                selectedPartner = arena->select(partner.guard, arena->constant(position++), selectedPartner);
            }
        }
        uint64_t position = 0;
        for (auto& [other, partner] : partners) {
            const Pair pair = forward ? Pair{selector.inputSite, other} : Pair{other, selector.inputSite};
            auto record = records.find(pair);
            if (record == records.end()) { return failure(); }
            ContextualEndpointFamily endpoint;
            auto& piece = endpoint.piece;
            piece.kind = local ? EndpointKind::Barrier : forward ? EndpointKind::Set : EndpointKind::Wait;
            piece.sourcePipe = forward ? pipe : selector.outputPipe;
            piece.targetPipe = forward ? selector.outputPipe : pipe;
            piece.namespaceId = namespaces.at({piece.sourcePipe, piece.targetPipe});
            piece.cut = {before->getBlock(), before}; piece.order = endpoints.size();
            piece.members.push_back({static_cast<uint32_t>(record->second), coordinates});
            endpoint.activity = arena->land(partner.guard, *atCut); endpoint.ordinal = arena->constant(0);
            if (selectedPartner != RegionExpressions::invalid) {
                endpoint.residueCoordinate = selectedPartner;
                endpoint.modulus = partners.size(); endpoint.residue = position;
            }
            ++position;
            if (!local) {
                if (forward) {
                    for (auto loop : site.loops) {
                        endpoint.matchingCoordinates.push_back(arena->input(loop.getInductionVar()));
                    }
                } else { endpoint.matchingCoordinates = std::move(partner.coordinates); }
                plan.nestedIdentities |= !endpoint.matchingCoordinates.empty();
            }
            endpoints.push_back(std::move(endpoint));
        }
        return success();
    }
};
using GeneralArithmeticEndpointRecipeBuilder = ArithmeticEndpointRecipeBuilder<GeneralArithmeticSelectors>;
using DifferenceArithmeticEndpointRecipeBuilder = ArithmeticEndpointRecipeBuilder<ArithmeticSelectors>;
} // namespace mlir::pto::frontiersynch::detail
#endif
