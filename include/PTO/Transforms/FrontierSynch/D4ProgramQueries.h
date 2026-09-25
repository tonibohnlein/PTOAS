// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_D4PROGRAMQUERIES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_D4PROGRAMQUERIES_H

#include "PTO/Transforms/FrontierSynch/D4Relations.h"
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include <algorithm>
#include <map>
#include <set>

namespace mlir::pto::frontiersynch {

// A view of ProgramAnalysis, not another original/selected analysis state.
// Qualification producers supply child profiles in the enclosing interpretation.
// This facet validates their snapshot, physical witnesses and actual cuts before
// the D4 algebra. It keeps the immutable obligation universe even on Unknown.
struct D4ProgramResult {
    D4RelationComposition relation;
    const OriginalObligations* obligations = nullptr;
};
struct D4ProgramReadResult {
    D4ReadComposition relation;
    const OriginalObligations* obligations = nullptr;
};
class D4ProgramQueries {
public:
    explicit D4ProgramQueries(const ProgramAnalysis& analysis) : analysis(analysis) {}

    // Native D1 supplier for a child containing exactly two mandatory related
    // accesses to this cell. More general D1 alternatives remain step 7's job.
    // No exact child profile is obtained from an All intersection alone.
    D4ChildUse fixedChild(const OriginalInterval& context, std::size_t child, std::size_t first, std::size_t last,
                         const std::shared_ptr<FactoredUseArena>& arena) const
    {
        D4ChildUse result;
        result.interpretation = context;
        result.child = child;
        result.kind = D4ChildUse::Kind::CompleteUses;
        result.rule = D4ChildUse::Rule::D1;
        result.first = {first, 0, context.owner, context.query.selector.cell, {first, OriginalCut::Before}};
        result.last = {last, 0, context.owner, context.query.selector.cell, {last, OriginalCut::After}};
        result.missingPremise = "native D1 child first/last correspondence is unqualified";
        const auto& original = analysis.structure();
        if (!current(context, arena.get()) || first >= original.operations.size() ||
            last >= original.operations.size() || !belongs(first, child) || !belongs(last, child)) {
            return result;
        }
        const auto all = analysis.all({child, context.query.selector.cell, true, true, {}});
        if (!all.complete) {
            return result;
        }
        std::set<std::size_t> operations;
        for (const auto& access : all.readers) { operations.insert(access.operation); }
        for (const auto& access : all.writers) { operations.insert(access.operation); }
        if (operations != std::set<std::size_t>{first, last} ||
            !analysis.occurrences().fixedVisit(first, last, context.query.selector.cell, FactoredUseNode::Hazard::RAW).exact) {
            return result;
        }
        const auto& uses = analysis.originalUsesAt(first, context.query.selector.cell);
        const auto* firstUse = uses.site(first);
        const auto* lastUse = uses.site(last);
        if (!firstUse || !lastUse || uses.arena != arena || !firstUse->visited || !lastUse->visited ||
            firstUse->applicability != lastUse->applicability ||
            (*arena)[firstUse->applicability].containsUnresolved ||
            !firstUse->access->write || !firstUse->access->definiteWrite ||
            !lastUse->access->read || lastUse->access->write ||
            !resolveOriginalCut(original, result.first.cut) || !resolveOriginalCut(original, result.last.cut)) {
            return result;
        }
        result.nonempty = firstUse->applicability;
        result.occurrenceQualified = true;
        auto correspondence = std::make_shared<D4ExistingCorrespondence>();
        correspondence->rule = D4CorrespondenceRule::D1;
        correspondence->interpretation = context;
        correspondence->firstUse = result.first;
        correspondence->lastUse = result.last;
        correspondence->nonempty = result.nonempty;
        correspondence->source = first;
        correspondence->target = last;
        correspondence->qualified = true;
        result.correspondence = std::move(correspondence);
        result.missingPremise.clear();
        return result;
    }

    D4ChildUse noUseChild(const OriginalInterval& context, std::size_t child) const
    {
        D4ChildUse result;
        result.interpretation = context;
        result.child = child;
        const auto all = analysis.all({child, context.query.selector.cell, true, true, {}});
        result.noUseProved = analysis.originalValues().current() && context.query.version == analysis.structure().version &&
                             all.complete && all.readers.empty() && all.writers.empty();
        if (!result.noUseProved) {
            result.missingPremise = "native child has no proved complete All exclusion";
        }
        return result;
    }

    // D2 clients supply their ALREADY QUALIFIED child first/last/domain profiles.
    // bankRelation().exactPermutation alone is not this premise: in particular
    // it does not qualify a child-local reset in an enclosing continuation.
    // This API does not derive or replace a supplied occurrence descriptor.
    D4ProgramResult completeUses(const OriginalInterval& context, std::shared_ptr<const FactoredUseArena> arena,
                                std::vector<D4ChildUse> children) const
    {
        D4ProgramResult result;
        result.obligations = &analysis.obligations();
        const auto missing = validate(context, arena.get(), children, false);
        if (!missing.empty()) {
            D4ChildUse unknown;
            unknown.interpretation = context;
            unknown.kind = D4ChildUse::Kind::CompleteUses;
            unknown.missingPremise = missing;
            result.relation = composeD4Relations(context, std::move(arena), {unknown});
            return result;
        }
        result.relation = composeD4Relations(context, std::move(arena), children);
        if (!current(context, result.relation.conditions.get())) {
            result.relation.complete = false;
            result.relation.unresolved.push_back({NoControlId, "D4 original interval, arena or cuts are stale/invalid"});
        }
        return result;
    }

    D4ProgramReadResult readFragments(const OriginalInterval& context, std::shared_ptr<FactoredUseArena> arena,
                                   const D4UseRole& enclosingUse, std::vector<D4ChildUse> children) const
    {
        const auto missing = validate(context, arena.get(), children, true);
        const bool incoming = enclosingUse.site == NoControlId &&
                              context.query.occurrence.incomingInterface != NoControlId &&
                              enclosingUse.coordinate == context.query.occurrence.incomingInterface &&
                              enclosingUse.owner == context.owner && enclosingUse.cell == context.query.selector.cell;
        const bool valid = missing.empty() && current(context, arena.get()) &&
                           (incoming || writerWitness(context, enclosingUse));
        if (!valid) {
            D4ChildUse unknown;
            unknown.interpretation = context;
            unknown.kind = D4ChildUse::Kind::ReadFragment;
            unknown.missingPremise = missing.empty() ? "D4 enclosing producer is unqualified" : missing;
            children = {unknown};
        }
        auto result = composeD4ReadFragments(context, std::move(arena), enclosingUse, children);
        if (!valid) {
            result.complete = false;
            result.boundaries.complete = false;
            result.boundaries.unresolved.push_back({NoControlId, "D4 unfinished use lacks its original owning interface"});
        }
        return {std::move(result), &analysis.obligations()};
    }

private:
    bool current(const OriginalInterval& context, const FactoredUseArena* arena) const
    {
        const auto& original = analysis.structure();
        if (!analysis.originalValues().current() || !arena || context.query.version != original.version ||
            arena->frame().original != reinterpret_cast<std::uintptr_t>(&original) ||
            arena->frame().snapshot != original.version || arena->frame().owner != context.owner ||
            arena->frame().cell != context.query.selector.cell) {
            return false;
        }
        const auto prepared = analysis.prepareInterval(context.query);
        const auto start = resolveOriginalCut(original, context.query.start);
        const auto stop = resolveOriginalCut(original, context.query.stop);
        if (!prepared.valid || prepared.interval != context || !start || !stop) { return false; }
        // A body-visit predicate arena is not a whole-loop invocation. Nor may a
        // while-after visit supply the before region's final-false interpretation.
        if (arena->frame().kind != FactoredUseFrame::Kind::Invocation) {
            if (context.owner >= original.originalSites.size()) { return false; }
            auto* owner = original.originalSites[context.owner];
            const bool counted = arena->frame().kind == FactoredUseFrame::Kind::ForBody;
            if (!owner || (counted ? !isa<scf::ForOp>(owner) : !isa<scf::WhileOp>(owner)) ||
                context.query.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::AfterBackedge) {
                return false; // A fixed body arena is not two successive visits.
            }
            const auto child = arena->frame().kind == FactoredUseFrame::Kind::WhileAfter ? 1u : 0u;
            if (child >= owner->getNumRegions()) { return false; }
            const auto* region = &owner->getRegion(child);
            auto within = [&](mlir::Operation* at) {
                for (; at; at = at->getParentOp()) {
                    if (at->getParentRegion() == region) { return true; }
                }
                return false;
            };
            if (!within(start->before) || !within(stop->before)) { return false; }
        } else if (arena->frame().owner != NoControlId) {
            return false;
        }
        return true;
    }
    bool belongs(std::size_t operation, std::size_t child) const
    {
        const auto& original = analysis.structure();
        if (operation >= original.operations.size()) { return false; }
        if (child == NoControlId) { return true; }
        if (child >= original.originalSites.size()) { return false; }
        for (auto* at = original.operations[operation].instruction->elementOp; at; at = at->getParentOp()) {
            if (at == original.originalSites[child]) { return true; }
        }
        return false;
    }
    bool witness(const OriginalInterval& context, const D4UseRole& use, bool filterEngine = true) const
    {
        const auto& original = analysis.structure();
        if (use.site >= original.operations.size() || use.owner != context.owner ||
            use.cell != context.query.selector.cell || !resolveOriginalCut(original, use.cut)) {
            return false;
        }
        const auto& operation = original.operations[use.site];
        if (filterEngine && context.query.selector.engine &&
            unsigned(operation.instruction->kPipeValue) != *context.query.selector.engine) {
            return false;
        }
        for (const auto& access : operation.accesses) {
            if (access.cell == use.cell && (access.read || access.write) &&
                (context.query.selector.physicalRelation == NoControlId ||
                 access.physicalRelation == context.query.selector.physicalRelation)) {
                return true;
            }
        }
        return false;
    }
    bool writerWitness(const OriginalInterval& context, const D4UseRole& use) const
    {
        if (!witness(context, use, false)) { return false; }
        for (const auto& access : analysis.structure().operations[use.site].accesses) {
            if (access.cell == use.cell && access.write && access.definiteWrite &&
                (context.query.selector.physicalRelation == NoControlId ||
                 access.physicalRelation == context.query.selector.physicalRelation)) {
                return true;
            }
        }
        return false;
    }
    bool atAccessBoundary(const D4UseRole& use, OriginalCut::Side side) const
    {
        const auto& original = analysis.structure();
        const auto actual = resolveOriginalCut(original, use.cut);
        const auto required = resolveOriginalCut(original, {use.site, side});
        return actual && required && actual->block == required->block && actual->before == required->before;
    }
    std::string validate(const OriginalInterval& context, const FactoredUseArena* arena,
                         std::vector<D4ChildUse>& children, bool readers) const
    {
        if (!current(context, arena)) { return "D4 original interval, arena or cuts are stale/invalid"; }
        const auto& original = analysis.structure();
        // This is a complete may-footprint query over the SAME cut interval,
        // not a first-hit query. Include interfering writes from other selectors
        // and engines too. No supplied profile may omit a genuine reload.
        auto allQuery = context.query;
        allQuery.selector.read = allQuery.selector.write = true;
        allQuery.selector.engine.reset();
        allQuery.selector.physicalRelation = NoControlId;
        const auto allInterval = analysis.prepareInterval(allQuery);
        if (!allInterval.valid) { return allInterval.reason; }
        const auto all = analysis.mayAfter(allInterval.interval);
        if (all.status == OriginalMayAfter::Status::Unknown) { return all.reason; }
        std::set<std::size_t> required;
        for (const auto& use : all.witnesses) {
            bool matching = !readers;
            const auto& operation = original.operations[use.operation];
            for (const auto& access : operation.accesses) {
                if (access.cell != context.query.selector.cell) { continue; }
                if (readers && access.write) { return "D4 unfinished read interval contains an intervening write"; }
                matching |= access.read && (!context.query.selector.engine ||
                            unsigned(operation.instruction->kPipeValue) == *context.query.selector.engine);
            }
            if (matching) { required.insert(use.operation); }
        }
        if (readers && !context.query.selector.engine) {
            return "D4 reader composition needs a separate query for each reader engine";
        }
        // Original tree order, independent of incidental original-site numbering.
        std::map<std::size_t, std::size_t> rank;
        std::vector<const Region*> pending{&original.body};
        while (!pending.empty()) {
            const auto* region = pending.back();
            pending.pop_back();
            if (region->kind == Region::Operation) { rank.emplace(region->operation, rank.size()); }
            for (auto it = region->children.rbegin(); it != region->children.rend(); ++it) { pending.push_back(&*it); }
        }
        std::set<std::size_t> covered;
        std::optional<std::size_t> previousEnd;
        for (auto& child : children) {
            if (child.kind == D4ChildUse::Kind::NoUse) {
                child.noUseProved = child.noUseProved && noUseChild(context, child.child).noUseProved;
            } else if (!witness(context, child.first) || !witness(context, child.last) ||
                       !atAccessBoundary(child.first, OriginalCut::Before) ||
                       !atAccessBoundary(child.last, OriginalCut::After) ||
                       !belongs(child.first.site, child.child) || !belongs(child.last.site, child.child)) {
                child.occurrenceQualified = false;
                child.missingPremise = "D4 child endpoints lack the same physical witnesses or legal cuts";
            }
            const auto contents = analysis.all({child.child, context.query.selector.cell, true, true, {}});
            if (!contents.complete) { return "D4 child has an unresolved physical projection"; }
            std::set<std::size_t> sites;
            for (const auto& access : contents.readers) { if (required.count(access.operation)) { sites.insert(access.operation); } }
            for (const auto& access : contents.writers) { if (required.count(access.operation)) { sites.insert(access.operation); } }
            // Every declared endpoint must be in this interval, not merely in
            // some visit of the same enclosing function.
            if (child.kind != D4ChildUse::Kind::NoUse &&
                (!required.count(child.first.site) || !required.count(child.last.site))) {
                return "D4 child endpoint lies outside the stated physical-use interval";
            }
            if (sites.empty()) { continue; }
            std::size_t begin = NoControlId, end = 0;
            for (auto site : sites) {
                if (!covered.insert(site).second || !rank.count(site)) {
                    return "D4 child projections overlap or duplicate an original use";
                }
                begin = std::min(begin, rank.at(site));
                end = std::max(end, rank.at(site));
            }
            if (previousEnd && begin <= *previousEnd) { return "D4 child order differs from the original sequence"; }
            previousEnd = end;
            // Qualification itself is never set true here from an All hit or a
            // periodic address candidate. The D1/D2 producer owns that premise.
        }
        return covered == required ? std::string{} : "D4 child summaries omit a physical use in the original interval";
    }
    const ProgramAnalysis& analysis;
};
} // namespace mlir::pto::frontiersynch
#endif
