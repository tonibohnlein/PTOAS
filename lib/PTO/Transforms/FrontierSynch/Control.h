// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_FRONTIERSYNCH_CONTROL_H
#define PTO_FRONTIERSYNCH_CONTROL_H
#include "PTO/Transforms/FrontierSynch/OriginalProgramPoints.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace mlir::pto::frontiersynch::detail {
struct AnalysisContext {
    enum Kind { Function, ThenArm, ElseArm, ForBody, WhileBefore, WhileAfter } kind = Function;
    std::size_t parent = NoControlId, ownerSite = NoControlId;
};
struct ControlSite {
    std::vector<std::size_t> successors;
    // Parallel edge metadata. Backedge owners are CONTROL entry IDs, as before;
    // loopEntries translates the original loop identity to this representation.
    std::vector<std::size_t> backedgeOwners;
    std::vector<bool> childEntries, bypasses;
};
struct ControlGraph {
    std::vector<ControlSite> sites;
    std::vector<AnalysisContext> contexts;
    std::vector<std::size_t> cutContexts, cutRanks;
    std::size_t entry = 0, exit = 0;
    std::vector<std::size_t> operations;
    std::vector<bool> legalCuts;
    std::map<OriginalCut, std::size_t> cuts;
    std::map<std::size_t, std::size_t> scopeParents, loopEntries;
    std::vector<std::size_t> nodeOwners, operationOwners;
    std::vector<bool> repeated, entryReachable;
    bool valid = true;
    std::string reason;
    OriginalProgramVersion version;
};
inline std::vector<bool> reachableSites(const ControlGraph& g)
{
    std::vector<bool> reached(g.sites.size());
    if (!g.valid || g.entry >= g.sites.size()) {
        return reached;
    }
    std::vector<std::size_t> todo{g.entry};
    reached[g.entry] = true;
    while (!todo.empty()) {
        const auto at = todo.back();
        todo.pop_back();
        for (auto next : g.sites[at].successors) {
            if (!reached[next]) {
                reached[next] = true;
                todo.push_back(next);
            }
        }
    }
    return reached;
}

// One original-control projection, also used by marginal provenance. Payload
// node IDs stay 0..count-1 and the function exit stays count. The additional
// epsilon ports distinguish REAL cut visits: an if join is not its next payload,
// a loop exit is not its header, and a child entry is not an incoming interface.
// This overload is independent of MLIR so its exact production implementation
// can be tested with a separately supplied legal-cut contract.
inline ControlGraph buildControlGraph(
    const Region& body, std::size_t count, const std::function<bool(const OriginalCut&)>& isLegal)
{
    ControlGraph g;
    g.exit = count;
    g.sites.resize(count + 1);
    g.contexts = {AnalysisContext{}};
    g.operations.assign(count + 1, NoControlId);
    g.cutContexts.assign(count + 1, 0);
    g.cutRanks.assign(count + 1, NoControlId);
    g.legalCuts.assign(count + 1, false);
    g.nodeOwners.assign(count + 1, NoControlId);
    g.operationOwners.assign(count, NoControlId);
    g.repeated.assign(count + 1, false);
    g.scopeParents.emplace(NoControlId, NoControlId);
    std::vector<bool> represented(count);
    std::size_t rank = 0;
    auto node = [&](std::size_t owner, std::size_t context, bool repeated) {
        const auto id = g.sites.size();
        g.sites.emplace_back();
        g.operations.push_back(NoControlId);
        g.cutContexts.push_back(context);
        g.cutRanks.push_back(NoControlId);
        g.legalCuts.push_back(false);
        g.nodeOwners.push_back(owner);
        g.repeated.push_back(repeated);
        return id;
    };
    auto edge = [&](std::size_t from, std::size_t to, std::size_t loop = NoControlId, bool child = false,
                    bool bypass = false) {
        g.sites[from].successors.push_back(to);
        g.sites[from].backedgeOwners.push_back(loop);
        g.sites[from].childEntries.push_back(child);
        g.sites[from].bypasses.push_back(bypass);
    };
    auto cut = [&](const OriginalCut& position, std::size_t at) {
        if (!g.cuts.emplace(position, at).second) {
            g.valid = false;
            g.reason = "duplicate original cut identity";
        }
        g.legalCuts[at] = isLegal(position);
    };
    struct Fragment {
        std::size_t entry, exit;
    };
    std::function<Fragment(const Region&, std::size_t, std::size_t, bool)> build;
    build = [&](const Region& r, std::size_t parent, std::size_t context, bool repeated) -> Fragment {
        if (r.kind == Region::Operation) {
            if (r.operation >= count || represented[r.operation]) {
                g.valid = false;
                g.reason = "invalid or duplicate original payload identity";
                const auto empty = node(parent, context, repeated);
                return {empty, empty};
            }
            const auto op = r.operation;
            represented[op] = true;
            g.operations[op] = op;
            g.cutContexts[op] = context;
            g.cutRanks[op] = rank++;
            g.nodeOwners[op] = parent;
            g.operationOwners[op] = parent;
            g.repeated[op] = repeated;
            const auto after = node(parent, context, repeated);
            cut({op, OriginalCut::Before}, op);
            cut({op, OriginalCut::After}, after);
            edge(op, after);
            return {op, after};
        }
        const bool named = r.originalOwner != NoControlId;
        const auto owner = named ? r.originalOwner : parent;
        if (named && !g.scopeParents.emplace(owner, parent).second) {
            g.valid = false;
            g.reason = "duplicate original scope identity";
        }
        if (r.kind != Region::Sequence && !named) {
            g.valid = false;
            g.reason = "structured control has no original owner identity";
        }
        const auto entry = node(owner, context, repeated), exit = node(owner, context, repeated);
        if (named) {
            cut(OriginalCut::scope(owner, OriginalCut::Before), entry);
            cut(OriginalCut::scope(owner, OriginalCut::After), exit);
        }
        if (r.kind == Region::Sequence) {
            auto previous = entry;
            for (const auto& child : r.children) {
                const auto part = build(child, owner, context, repeated);
                edge(previous, part.entry);
                previous = part.exit;
            }
            edge(previous, exit);
            return {entry, exit};
        }
        const auto expected = r.kind == Region::For ? 1u : 2u;
        if (r.children.size() != expected) {
            g.valid = false;
            g.reason = "invalid original structured child population";
            edge(entry, exit);
            return {entry, exit};
        }
        auto child = [&](std::size_t index, AnalysisContext::Kind kind, bool repeats) {
            const auto childContext = g.contexts.size();
            g.contexts.push_back({kind, context, entry});
            const auto in = node(owner, childContext, repeats), out = node(owner, childContext, repeats);
            cut(OriginalCut::childBoundary(owner, index, OriginalCut::Before), in);
            cut(OriginalCut::childBoundary(owner, index, OriginalCut::After), out);
            const auto part = build(r.children[index], owner, childContext, repeats);
            edge(in, part.entry);
            edge(part.exit, out);
            return Fragment{in, out};
        };
        if (r.kind == Region::Choice) {
            const auto yes = child(0, AnalysisContext::ThenArm, repeated);
            const auto no = child(1, AnalysisContext::ElseArm, repeated);
            edge(entry, yes.entry, NoControlId, true);
            edge(entry, no.entry, NoControlId, true);
            edge(yes.exit, exit);
            edge(no.exit, exit);
        } else if (r.kind == Region::For) {
            g.loopEntries.emplace(owner, entry);
            const auto header = node(owner, context, true);
            const auto part = child(0, AnalysisContext::ForBody, true);
            edge(entry, header);
            edge(header, part.entry, NoControlId, true);
            // Preserve the baseline conservative bypass, even for a proved nonempty
            // counted loop; this graph is a may language, not the counted D3 decoder.
            edge(header, exit, NoControlId, false, true);
            edge(part.exit, header, entry);
        } else if (r.kind == Region::While) {
            g.loopEntries.emplace(owner, entry);
            const auto before = child(0, AnalysisContext::WhileBefore, true);
            const auto after = child(1, AnalysisContext::WhileAfter, true);
            const auto decision = node(owner, context, true);
            edge(entry, before.entry, NoControlId, true);
            edge(before.exit, decision);
            edge(decision, after.entry, NoControlId, true);
            edge(decision, exit, NoControlId, false, true);
            // The final false test still executes the before region. The after
            // region's backedge returns there, not through external loop entry.
            edge(after.exit, before.entry, entry, true);
        } else {
            g.valid = false;
            g.reason = "unknown original control kind";
        }
        return {entry, exit};
    };
    g.entry = node(NoControlId, 0, false);
    cut(OriginalCut::scope(NoControlId, OriginalCut::Before), g.entry);
    cut(OriginalCut::scope(NoControlId, OriginalCut::After), g.exit);
    // Retain the legacy flat semantic-fixture spelling.
    Region flat;
    const Region* root = &body;
    if (body.kind == Region::Sequence && body.children.empty() && count != 0) {
        flat = body;
        for (std::size_t op = 0; op < count; ++op) {
            Region leaf;
            leaf.kind = Region::Operation;
            leaf.operation = op;
            flat.children.push_back(std::move(leaf));
        }
        root = &flat;
    }
    const auto program = build(*root, NoControlId, 0, false);
    edge(g.entry, program.entry);
    edge(program.exit, g.exit);
    g.cutRanks[g.exit] = rank;
    if (std::find(represented.begin(), represented.end(), false) != represented.end()) {
        g.valid = false;
        g.reason = "unrepresented original payload";
    }
    g.entryReachable = reachableSites(g);
    return g;
}
// resolveOriginalCut is supplied by the OriginalStructure adapter via ADL.
// There is no second instruction-effect translation in this projection.
template <typename OriginalProgram>
ControlGraph buildControlGraph(const OriginalProgram& p)
{
    auto graph = buildControlGraph(
        p.body, p.operations.size(), [&](const OriginalCut& cut) { return bool(resolveOriginalCut(p, cut)); });
    graph.version = p.version;
    return graph;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
