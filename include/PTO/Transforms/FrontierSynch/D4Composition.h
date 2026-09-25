// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_D4COMPOSITION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_D4COMPOSITION_H

#include "PTO/Transforms/FrontierSynch/FactoredUse.h"
#include <algorithm>
#include <set>

namespace mlir::pto::frontiersynch {

// D4 composes ORIGINAL transfers, not their outputs on a fresh child entry.
// Every child is interpreted in the same physical owner. Its four formal
// parameters stand for incoming/following writer and reader histories. They are
// private DAG handles, not runtime state or four new invocation interfaces.
class D4ChildSummary {
public:
    D4ChildSummary(FactoredUseProjection projection, std::shared_ptr<FactoredUseArena> arena)
    {
        if (!arena || !(arena->frame() == projection.frame)) {
            summary.frame = projection.frame;
            summary.cell = projection.frame.cell;
            summary.reason = "D4 child requires its enclosing physical-use arena";
            return;
        }
        using N = FactoredUseNode;
        parameters.arena = arena;
        parameters.incoming = {
            arena->incoming(projection.frame.owner, N::Boundary::Entry, N::Role::Writer),
            arena->incoming(projection.frame.owner, N::Boundary::Entry, N::Role::Reader)};
        parameters.following = {
            arena->incoming(projection.frame.owner, N::Boundary::Exit, N::Role::Writer),
            arena->incoming(projection.frame.owner, N::Boundary::Exit, N::Role::Reader)};
        summary = FactoredUseBuilder(std::move(projection), parameters).take();
    }
    const FactoredUseResult& transfer() const { return summary; }
    const FactoredUseInterface& inputs() const { return parameters; }

private:
    FactoredUseResult summary;
    FactoredUseInterface parameters;
};

struct D4CompositionWork {
    std::size_t summaries = 0, applications = 0, substitutedNodes = 0;
    std::size_t summaryNodes = 0, applicationNodes = 0;
};
struct D4ChildApplication {
    std::shared_ptr<const D4ChildSummary> summary;
    FactoredUseState incoming, outgoing, following, preceding;
    std::size_t demands = 0;
};
struct D4CompositionResult {
    FactoredUseResult uses;
    // Child order and interfaces are retained even for an unresolved repeat.
    // No child exit invents an initial writer or clears a reader history.
    std::vector<D4ChildApplication> children;
    D4CompositionWork work;
};

namespace d4_detail {
// Substitute formal history roots, sharing all original guards, accesses,
// physical witnesses and actual input DAGs. In particular an inserted input is
// not traversed: it may contain the *same static site* from a preceding use.
// Iterative postorder avoids recursion proportional to a long reader suffix.
class Apply {
public:
    Apply(const D4ChildSummary& child, D4CompositionWork& work)
        : arena(child.inputs().arena), work(work)
    {}
    void bind(FactoredUseState formal, FactoredUseState actual)
    {
        replacements.emplace(formal.writers, actual.writers);
        replacements.emplace(formal.readers, actual.readers);
    }
    std::size_t operator()(std::size_t root)
    {
        std::vector<std::pair<std::size_t, bool>> pending{{root, false}};
        while (!pending.empty()) {
            const auto [id, ready] = pending.back();
            pending.pop_back();
            if (mapped.count(id)) {
                continue;
            }
            if (const auto found = replacements.find(id); found != replacements.end()) {
                mapped.emplace(id, found->second);
                continue;
            }
            // Copy before calling any arena constructor (vector reallocation).
            const auto node = (*arena)[id];
            using N = FactoredUseNode;
            const bool binary = node.kind == N::Kind::Both || node.kind == N::Kind::Choose ||
                                node.kind == N::Kind::Unresolved;
            const bool unary = node.kind == N::Kind::Demand;
            if (!binary && !unary) {
                mapped.emplace(id, id);
                continue;
            }
            if (!ready) {
                pending.push_back({id, true});
                pending.push_back({node.left, false});
                if (binary) {
                    pending.push_back({node.right, false});
                }
                if (node.kind == N::Kind::Choose) {
                    pending.push_back({node.condition, false});
                }
                continue;
            }
            ++work.substitutedNodes;
            const auto left = mapped.at(node.left);
            const auto right = binary ? mapped.at(node.right) : node.right;
            const auto condition = node.kind == N::Kind::Choose ? mapped.at(node.condition) : node.condition;
            if (left == node.left && right == node.right && condition == node.condition) {
                mapped.emplace(id, id);
                continue;
            }
            std::size_t replacement = id;
            switch (node.kind) {
                case N::Kind::Both:
                    replacement = arena->both(left, right, node.sort);
                    break;
                case N::Kind::Choose:
                    replacement = arena->choose(condition, left, right, node.sort);
                    break;
                case N::Kind::Demand:
                    replacement = arena->demand(left, node.access, node.hazard);
                    break;
                case N::Kind::Unresolved:
                    replacement = arena->unresolved(node.owner, node.sort, node.role, left, right);
                    break;
                default:
                    break;
            }
            mapped.emplace(id, replacement);
        }
        return mapped.at(root);
    }
    FactoredUseState state(FactoredUseState value) { return {(*this)(value.writers), (*this)(value.readers)}; }

private:
    std::shared_ptr<FactoredUseArena> arena;
    D4CompositionWork& work;
    std::unordered_map<std::size_t, std::size_t> replacements, mapped;
};
inline bool validState(const FactoredUseArena& arena, FactoredUseState state)
{
    return arena.hasSort(state.writers, FactoredUseNode::Sort::Origins) &&
           arena.hasSort(state.readers, FactoredUseNode::Sort::Origins);
}
} // namespace d4_detail

// Apply each child once forward and once backward. Each application has one
// memo table; no distribution of Choose over independent Both expressions.
// Reusing a summary for a *different dynamic visit* requires an occurrence
// interface and is NOT a second application of its static Access leaves here.
inline D4CompositionResult composeD4Children(
    const std::vector<std::shared_ptr<const D4ChildSummary>>& children, FactoredUseInterface boundary)
{
    D4CompositionResult out;
    auto& result = out.uses;
    result.arena = boundary.arena;
    if (!boundary.arena) {
        result.reason = "D4 composition needs an explicit owning interface";
        return out;
    }
    auto& arena = *boundary.arena;
    result.frame = arena.frame();
    result.cell = arena.frame().cell;
    result.finalWriters = boundary.incoming.writers;
    result.finalReaders = boundary.incoming.readers;
    result.entryNextWriters = boundary.following.writers;
    result.entryNextReaders = boundary.following.readers;
    if (!d4_detail::validState(arena, boundary.incoming) || !d4_detail::validState(arena, boundary.following)) {
        result.reason = "D4 interface history has an invalid expression sort";
        return out;
    }
    // Validate the entire fixed visit before exporting any composed answer.
    // An unfinished child cannot silently change the physical owner/frame.
    std::set<std::size_t> operations;
    for (const auto& child : children) {
        if (!child || child->inputs().arena != boundary.arena || !(child->transfer().frame == arena.frame())) {
            result.reason = "D4 child owner/occurrence frame differs; qualified re-entry mapping is required";
            return out;
        }
        for (const auto& site : child->transfer().sites) {
            if (!site.access || !operations.insert(site.access->operation).second) {
                result.reason = "D4 fixed-visit composition cannot identify two visits by one static site";
                return out;
            }
        }
    }
    result.complete = true;
    result.work.inputNodes = arena.nodes().size();
    const auto attempts = arena.attempts();
    std::vector<std::unique_ptr<d4_detail::Apply>> applications;
    std::vector<std::size_t> siteOffsets;
    auto state = boundary.incoming;
    for (const auto& child : children) {
        const auto& value = child->transfer();
        if (!value.complete) {
            result.complete = false;
            if (result.reason.empty()) {
                result.reason = value.reason;
            }
        }
        auto apply = std::make_unique<d4_detail::Apply>(*child, out.work);
        apply->bind(child->inputs().incoming, state);
        D4ChildApplication application;
        application.summary = child;
        application.incoming = state;
        application.outgoing = apply->state({value.finalWriters, value.finalReaders});
        application.demands = (*apply)(value.demands);
        result.demands = arena.both(result.demands, application.demands, FactoredUseNode::Sort::Demands);
        siteOffsets.push_back(result.sites.size());
        for (auto site : value.sites) {
            site.applicability = (*apply)(site.applicability);
            site.priorWriters = (*apply)(site.priorWriters);
            site.priorReaders = (*apply)(site.priorReaders);
            for (auto& demand : site.demands) {
                demand = (*apply)(demand);
            }
            result.siteIndex.emplace(site.access->operation, result.sites.size());
            result.sites.push_back(std::move(site));
        }
        for (auto repeated : value.repeatedInterfaces) {
            if (!repeated.backward) {
                repeated.incoming = apply->state(repeated.incoming);
                repeated.outgoing = apply->state(repeated.outgoing);
                result.repeatedInterfaces.push_back(std::move(repeated));
            }
        }
        state = application.outgoing;
        out.children.push_back(std::move(application));
        applications.push_back(std::move(apply));
        result.work.forwardSteps += value.work.forwardSteps;
        result.work.backwardSteps += value.work.backwardSteps;
        result.work.weakWrites += value.work.weakWrites;
    }
    result.finalWriters = state.writers;
    result.finalReaders = state.readers;
    state = boundary.following;
    for (std::size_t i = children.size(); i > 0; --i) {
        const auto index = i - 1;
        const auto& child = *children[index];
        const auto& value = child.transfer();
        auto& apply = *applications[index];
        // Forward roots contain only entry parameters, backward roots only exit
        // parameters. Binding the latter cannot invalidate the former memo table.
        apply.bind(child.inputs().following, state);
        auto& application = out.children[index];
        application.following = state;
        application.preceding = apply.state({value.entryNextWriters, value.entryNextReaders});
        for (std::size_t site = 0; site < value.sites.size(); ++site) {
            auto& destination = result.sites[siteOffsets[index] + site];
            destination.nextWriters = apply(value.sites[site].nextWriters);
            destination.nextReaders = apply(value.sites[site].nextReaders);
        }
        for (auto repeated : value.repeatedInterfaces) {
            if (repeated.backward) {
                repeated.incoming = apply.state(repeated.incoming);
                repeated.outgoing = apply.state(repeated.outgoing);
                result.repeatedInterfaces.push_back(std::move(repeated));
            }
        }
        state = application.preceding;
    }
    result.entryNextWriters = state.writers;
    result.entryNextReaders = state.readers;
    result.work.constructorCalls = arena.attempts() - attempts;
    result.work.addedNodes = arena.nodes().size() - result.work.inputNodes;
    out.work.applications = children.size();
    out.work.applicationNodes = result.work.addedNodes;
    return out;
}

// Form finite child templates from one qualified projection. Sequence wrappers
// do not create physical owners or new visit identities. Choices remain whole
// shared conditional transfers; an OpaqueRepeat stays opaque (including its
// original incoming/continuation obstruction), not one unrolled child visit.
inline D4CompositionResult buildD4Composition(FactoredUseProjection projection, FactoredUseInterface boundary)
{
    if (!boundary.arena || !(boundary.arena->frame() == projection.frame)) {
        D4CompositionResult invalid;
        invalid.uses.frame = projection.frame;
        invalid.uses.cell = projection.frame.cell;
        invalid.uses.reason = "D4 projection differs from its owning interface";
        return invalid;
    }
    auto& arena = *boundary.arena;
    const auto nodes = arena.nodes().size(), attempts = arena.attempts();
    std::vector<const FactoredUseRegion*> parts, pending{&projection.body};
    while (!pending.empty()) {
        const auto* region = pending.back();
        pending.pop_back();
        if (region->kind == FactoredUseRegion::Kind::Sequence) {
            for (auto child = region->children.rbegin(); child != region->children.rend(); ++child) {
                pending.push_back(&*child);
            }
        } else {
            parts.push_back(region);
        }
    }
    std::set<std::size_t> used;
    std::vector<std::shared_ptr<const D4ChildSummary>> summaries;
    for (const auto* part : parts) {
        FactoredUseProjection child;
        child.frame = projection.frame;
        child.body = *part;
        std::vector<FactoredUseRegion*> work{&child.body};
        while (!work.empty()) {
            auto* region = work.back();
            work.pop_back();
            if (region->kind == FactoredUseRegion::Kind::Access) {
                if (region->access >= projection.accesses.size() || !used.insert(region->access).second) {
                    // Let the existing transfer retain the unresolved original
                    // projection; never publish a composition of a proper subset.
                    D4CompositionResult invalid;
                    invalid.uses = FactoredUseBuilder(std::move(projection), std::move(boundary)).take();
                    invalid.uses.complete = false;
                    invalid.uses.reason = "D4 projection contains an invalid or repeated access index";
                    return invalid;
                }
                child.accesses.push_back(projection.accesses[region->access]);
                region->access = child.accesses.size() - 1;
            }
            for (auto it = region->children.rbegin(); it != region->children.rend(); ++it) {
                work.push_back(&*it);
            }
        }
        summaries.push_back(std::make_shared<D4ChildSummary>(std::move(child), boundary.arena));
    }
    if (used.size() != projection.accesses.size()) {
        D4CompositionResult invalid;
        invalid.uses = FactoredUseBuilder(std::move(projection), std::move(boundary)).take();
        invalid.uses.complete = false;
        invalid.uses.reason = "D4 projection has accesses outside its child summaries";
        return invalid;
    }
    const auto formed = arena.nodes().size() - nodes;
    auto result = composeD4Children(summaries, std::move(boundary));
    result.work.summaries = summaries.size();
    result.work.summaryNodes = formed;
    result.uses.work.inputNodes = nodes;
    result.uses.work.addedNodes = arena.nodes().size() - nodes;
    result.uses.work.constructorCalls = arena.attempts() - attempts;
    return result;
}

} // namespace mlir::pto::frontiersynch
#endif
