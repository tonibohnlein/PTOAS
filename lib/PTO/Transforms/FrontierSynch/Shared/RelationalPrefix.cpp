// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite constructive adapter over retained relational crossings. No coordinate
// synthesis: each admitted original site executes at most once per invocation.
#include "InvocationPrefixCollection.h"
#include "OccurrenceBinding.h"
#include "RegionalCapabilitiesInternal.h"
#include "PrefixExpression.h"
#include "PrefixDemand.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalRelations.h"
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
bool nonrepeatingFrame(
    const RegionalAnalysis& region, const RegionalRelationData& data, const std::shared_ptr<const SyncInput>& input)
{
    // completeRequiredOrder certifies H0. newCrossings owns only the crossing
    // covers; original child contributors own internal F*, so exactMinimum on
    // the combined relational inventory is deliberately not required.
    if (!input || input.get() != data.input || !data.context.function || !data.completeRequiredOrder ||
        !data.analysis.error.empty() || !data.enclosing.empty() || !data.analysis.period ||
        data.parameters.size() != data.analysis.parameterCount || data.sites.size() > UINT32_MAX ||
        region.anchors.size() != data.sites.size() || !validRegionalOccurrenceSchema(region) ||
        region.occurrenceLoops.size() != data.sites.size() || region.accessModel != &input->accesses() ||
        region.gmAliasPolicy != input->memory().gmPolicy()) {
        return false;
    }
    std::set<const CompoundInstanceElement*> phases;
    for (std::size_t i = 0; i < data.sites.size(); ++i) {
        const auto& site = data.sites[i];
        const auto& anchor = region.anchors[i];
        auto* op = site.phase ? site.phase->elementOp : nullptr;
        if (!op || site.phase != anchor.phase || !phases.insert(site.phase).second || !site.loops.empty() ||
            !site.fixedCoordinates.empty() || !anchor.coordinates.empty() || region.occurrenceLoops[i] ||
            (!region.outerLoops.empty() && (i >= region.outerLoops.size() || !region.outerLoops[i].empty())) ||
            (!region.outerDivisors.empty() && (i >= region.outerDivisors.size() || !region.outerDivisors[i].empty())) ||
            anchor.before.block != op->getBlock() || anchor.before.before != op ||
            anchor.after.block != op->getBlock() || anchor.after.before != op->getNextNode()) {
            return false;
        }
        auto* parent = op->getParentOp();
        while (parent && parent != data.context.function) {
            if (!isa<scf::IfOp>(parent)) {
                return false;
            }
            parent = parent->getParentOp();
        }
        if (parent != data.context.function) {
            return false;
        }
    }
    return llvm::all_of(data.parameters, [&](Expr id) { return id < region.expressions->size(); });
}
struct CrossingRecipe {
    uint32_t source, target;
    Expr guard;
};
FailureOr<std::vector<CrossingRecipe>> crossings(const RegionalAnalysis& region)
{
    const auto& relation = *region.relations;
    const auto& data = relation.data;
    auto& e = *region.expressions;
    std::map<std::pair<uint32_t, uint32_t>, Expr> guards;
    for (const auto& [key, pieces] : relation.newCrossings) {
        if (key.source.site >= data.sites.size() || key.target.site >= data.sites.size() ||
            key.source.site >= key.target.site || key.source.event != ArithmeticEvent::Completion ||
            key.target.event != ArithmeticEvent::Start || !key.source.residues.empty() ||
            !key.target.residues.empty() || key.parameterResidues.size() != data.parameters.size() ||
            llvm::any_of(key.parameterResidues, [&](uint64_t r) { return r >= data.analysis.period; })) {
            return failure();
        }
        const auto source = static_cast<uint32_t>(key.source.site), target = static_cast<uint32_t>(key.target.site);
        Expr guard = e.boolean(false);
        for (const auto& piece : pieces) {
            if (piece.dimensions() != data.parameters.size()) {
                return failure();
            }
            guard =
                e.lor(guard, e.integerPredicate(piece, data.parameters, data.analysis.period, key.parameterResidues));
        }
        // Preserve exact original endpoint activity even when a caller cannot
        // expose the relation producer's internal endpoint-domain certificate.
        auto a = regionalPresence(region, {source, e.constant(0), PeriodicEventKind::Completion});
        auto b = regionalPresence(region, {target, e.constant(0), PeriodicEventKind::Start});
        if (!a || !b || guard >= e.size()) {
            return failure();
        }
        guard = e.land(guard, e.land(*a, *b));
        auto [entry, added] = guards.emplace(std::make_pair(source, target), guard);
        if (!added) {
            entry->second = e.lor(entry->second, guard);
        }
    }
    std::vector<CrossingRecipe> result;
    for (auto [pair, guard] : guards) {
        if (e.constantValue(guard) != 0) {
            result.push_back({pair.first, pair.second, guard});
        }
    }
    return result;
}
LogicalResult collectCrossings(
    const std::shared_ptr<const RegionalAnalysis>& region, InvocationPrefixCollection& collection, std::string& error,
    const std::function<LogicalResult(InvocationPrefixCollection&, std::string&)>& children)
{
    auto records = crossings(*region);
    if (failed(records)) {
        error = "relational prefix crossing has an unsupported finite schema";
        return failure();
    }
    if (children) {
        if (failed(children(collection, error))) {
            return failure();
        }
    } else {
        for (const auto& child : region->relations->children) {
            if (child.anchors.empty()) {
                continue;
            }
            if (failed(collectRegionalInvocationPrefix(child, collection, error))) {
                return failure();
            }
        }
    }
    auto& recipe = collection.recipe();
    auto type = IndexType::get(collection.function().getContext());
    std::map<uint32_t, std::size_t> captures;
    for (auto record : *records) {
        if (region->anchors[record.source].phase->kPipeValue == region->anchors[record.target].phase->kPipeValue) {
            continue;
        }
        if (captures.count(record.source)) {
            continue;
        }
        const auto slot = recipe.slots.size();
        captures.emplace(record.source, slot);
        recipe.slots.push_back({type, IntegerAttr::get(type, 0)});
        if (failed(collection.source(
                region->anchors[record.source].phase,
                [slot](OpBuilder&, const IRMapping&, MutableArrayRef<Value> state, Value rank, std::string& error) {
                    if (slot >= state.size()) {
                        error = "relational source rank lost its invocation slot";
                        return failure();
                    }
                    state[slot] = rank;
                    return success();
                },
                error))) {
            return failure();
        }
    }
    std::map<Operation*, SmallVector<Expr>> roots;
    for (auto record : *records) {
        roots[region->anchors[record.target].phase->elementOp].push_back(record.guard);
    }
    std::map<Operation*, std::vector<PrefixIncomingSelector>> prepared;
    for (const auto& [cut, values] : roots) {
        prepared.emplace(
            cut, preparePrefixExpressions(region->expressions, values, cut, collection, "relational-incoming"));
    }
    std::map<Operation*, std::size_t> next;
    for (auto record : *records) {
        auto* cut = region->anchors[record.target].phase->elementOp;
        auto guard = prepared.at(cut)[next[cut]++];
        const auto p = static_cast<uint32_t>(region->anchors[record.source].phase->kPipeValue);
        const auto q = static_cast<uint32_t>(region->anchors[record.target].phase->kPipeValue);
        PrefixIncomingSelector rank;
        if (p != q) {
            const auto slot = captures.at(record.source);
            rank = [slot](OpBuilder&, const IRMapping&, ArrayRef<Value> state, std::string& error) -> FailureOr<Value> {
                if (slot >= state.size()) {
                    error = "relational acquisition lost its source rank slot";
                    return failure();
                }
                return state[slot];
            };
        }
        if (failed(appendPrefixIncoming(collection, cut, {p, q}, std::move(guard), std::move(rank), error))) {
            return failure();
        }
    }
    return success();
}
} // namespace
std::shared_ptr<const RegionalPrefixContribution> relationalPrefixContribution(
    const RegionalAnalysis& region, std::shared_ptr<const SyncInput> input,
    std::function<LogicalResult(InvocationPrefixCollection&, std::string&)> children)
{
    if (!region.relations || !region.expressions || !nonrepeatingFrame(region, region.relations->data, input)) {
        return {};
    }
    for (const auto& owner : region.relations->auxiliaryOwners) {
        region.expressions->retainInputOwner(owner);
    }
    auto frame = std::make_shared<RegionalAnalysis>(region);
    frame->invocationPrefix.reset();
    auto contribution = std::make_shared<RegionalPrefixContribution>();
    contribution->input = std::move(input);
    contribution->expressions = region.expressions;
    contribution->function = region.relations->data.context.function;
    contribution->accessModel = region.accessModel;
    contribution->aliasPolicy = region.gmAliasPolicy;
    contribution->matchesOccurrenceFrame = [frame](const RegionalAnalysis& current) {
        return samePrefixOccurrenceFrame(*frame, current);
    };
    contribution->occurrence = [frame](RegionalEvent event, std::string& error) -> PrefixOccurrencePtr {
        auto& e = *frame->expressions;
        if (!validRegionalEvent(*frame, event) || !event.visits.empty() || e.constantValue(event.ordinal) != 0) {
            error = "nonrepeating relational occurrence requires ordinal zero without visits";
            return {};
        }
        auto conditions = regionalOccurrenceConstraints(*frame, event, error);
        if (failed(conditions)) {
            return {};
        }
        auto result = std::make_shared<PrefixOccurrence>();
        result->anchor = frame->anchors[event.type];
        result->match = e.boolean(true);
        for (auto condition : *conditions) {
            result->match = e.land(result->match, condition);
        }
        if (frame->endpointSiteGuard) {
            result->match = e.land(result->match, *frame->endpointSiteGuard);
        }
        if (frame->endpointInvocationGuard) {
            result->match = e.land(result->match, *frame->endpointInvocationGuard);
        }
        return result;
    };
    contribution->collect = [frame, children = std::move(children)](
                                InvocationPrefixCollection& collection, std::string& error) {
        return collectCrossings(frame, collection, error, children);
    };
    return contribution;
}
} // namespace mlir::pto::frontiersynch
