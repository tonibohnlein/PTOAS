// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FACTOREDPROVENANCE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FACTOREDPROVENANCE_H

#include "PTO/Transforms/FrontierSynch/FactoredUse.h"
#include "PTO/Transforms/FrontierSynch/OriginalStructure.h"
#include "PTO/Transforms/FrontierSynch/OriginalValueQueries.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <map>

namespace mlir::pto::frontiersynch {

// Selects one original invocation or one visit of a repeated child. No body
// visit stands for the complete repeat; before/after while visits are distinct.
struct FactoredProjectionScope {
    FactoredUseFrame::Kind kind = FactoredUseFrame::Kind::Invocation;
    std::size_t owner = NoControlId;
    bool operator<(const FactoredProjectionScope& other) const
    {
        return std::tie(kind, owner) < std::tie(other.kind, other.owner);
    }
};

class FactoredProvenance {
public:
    explicit FactoredProvenance(
        const OriginalStructure& original, std::size_t cell, FactoredProjectionScope scope = {},
        FactoredUseInterface boundary = {}, const OriginalValueQueries* shared = nullptr)
    {
        std::unique_ptr<OriginalValueQueries> ownedValues;
        if (!shared) {
            ownedValues = std::make_unique<OriginalValueQueries>(original);
            shared = ownedValues.get();
        }
        FactoredUseFrame frame;
        frame.original = reinterpret_cast<std::uintptr_t>(&original);
        frame.snapshot = original.version;
        frame.cell = cell;
        frame.owner = scope.owner;
        frame.kind = scope.kind;
        result.frame = frame;
        result.cell = cell;
        result.arena = boundary.arena ? boundary.arena : std::make_shared<FactoredUseArena>(frame);
        if (!(result.arena->frame() == frame)) {
            result.reason = "factored inputs require the same original instance, cell, and occurrence scope";
            return;
        }
        const auto* root = projectionRoot(original.body, scope);
        if (cell >= original.cells.size() || !root) {
            result.reason = "invalid original physical cell or fixed-use projection scope";
            return;
        }
        if (!boundary.arena) {
            boundary.arena = result.arena;
            // No child is silently fresh. These are named original-history inputs,
            // not a fictitious completed writer/reader or runtime history variable.
            boundary.incoming = {
                boundary.arena->incoming(scope.owner, FactoredUseNode::Boundary::Entry, FactoredUseNode::Role::Writer),
                boundary.arena->incoming(scope.owner, FactoredUseNode::Boundary::Entry, FactoredUseNode::Role::Reader)};
            if (scope.kind != FactoredUseFrame::Kind::Invocation) {
                boundary.following = {
                    boundary.arena->incoming(
                        scope.owner, FactoredUseNode::Boundary::Exit, FactoredUseNode::Role::Writer),
                    boundary.arena->incoming(
                        scope.owner, FactoredUseNode::Boundary::Exit, FactoredUseNode::Role::Reader)};
            }
            // Invocation following roots are empty *at its designated original exit*,
            // not a claim about a caller's continuation beyond that horizon.
        }
        ProjectionAdapter adapter(original, cell, *boundary.arena, *shared);
        FactoredUseProjection projection;
        projection.frame = frame;
        projection.body = adapter.project(*root, projection.accesses);
        FactoredUseBuilder builder(std::move(projection), std::move(boundary));
        result = builder.take();
        projectionWork = adapter.work;
    }

    const FactoredUseResult& get() const { return result; }
    FactoredUseResult take() { return std::move(result); }
    struct ProjectionWork {
        std::size_t syntax = 0, effectIncidences = 0, guards = 0;
    };
    const ProjectionWork& preparationWork() const { return projectionWork; }

private:
    static const Region* findOwner(const Region& region, std::size_t owner)
    {
        if (region.kind != Region::Operation && region.originalOwner == owner) {
            return &region;
        }
        for (const auto& child : region.children) {
            if (const auto* found = findOwner(child, owner)) {
                return found;
            }
        }
        return nullptr;
    }
    static const Region* projectionRoot(const Region& body, FactoredProjectionScope scope)
    {
        if (scope.kind == FactoredUseFrame::Kind::Invocation) {
            return scope.owner == NoControlId ? &body : nullptr;
        }
        if (scope.owner == NoControlId) {
            return nullptr;
        }
        const auto* owner = findOwner(body, scope.owner);
        if (!owner) {
            return nullptr;
        }
        if (scope.kind == FactoredUseFrame::Kind::ForBody) {
            return owner->kind == Region::For && owner->children.size() == 1 ? &owner->children[0] : nullptr;
        }
        if ((scope.kind != FactoredUseFrame::Kind::WhileBefore && scope.kind != FactoredUseFrame::Kind::WhileAfter) ||
            owner->kind != Region::While || owner->children.size() != 2) {
            return nullptr;
        }
        return &owner->children[scope.kind == FactoredUseFrame::Kind::WhileBefore ? 0 : 1];
    }
    struct ProjectionAdapter {
        const OriginalStructure& original;
        std::size_t cell;
        FactoredUseArena& arena;
        const OriginalValueQueries& values;
        DenseMap<mlir::Operation*, std::size_t> siteIds;
        ProjectionWork work;
        ProjectionAdapter(
            const OriginalStructure& original, std::size_t cell, FactoredUseArena& arena,
            const OriginalValueQueries& values)
            : original(original), cell(cell), arena(arena), values(values)
        {
            for (std::size_t id = 0; id < original.originalSites.size(); ++id) {
                siteIds.try_emplace(original.originalSites[id], id);
            }
        }
        std::size_t guard(const Region& region)
        {
            ++work.guards;
            if (region.originalOwner >= original.originalSites.size()) {
                return NoFactoredId;
            }
            auto choice = dyn_cast<scf::IfOp>(original.originalSites[region.originalOwner]);
            if (!choice) {
                return NoFactoredId;
            }
            const auto identity = values.identity(choice.getCondition());
            if (!identity.value) {
                return NoFactoredId;
            }
            return arena.test(
                {reinterpret_cast<std::uintptr_t>(identity.value.getAsOpaquePointer()), identity.occurrenceScope});
        }
        bool summarize(const Region& region, bool& reads, bool& writes)
        {
            ++work.syntax;
            if (region.kind == Region::Operation) {
                if (region.operation >= original.operations.size()) {
                    return false;
                }
                for (const auto& effect : original.operations[region.operation].accesses) {
                    ++work.effectIncidences;
                    if (effect.cell == cell) {
                        reads |= effect.read;
                        writes |= effect.write;
                    }
                }
            }
            for (const auto& child : region.children) {
                if (!summarize(child, reads, writes)) {
                    return false;
                }
            }
            return true;
        }
        FactoredUseRegion project(const Region& region, std::vector<std::shared_ptr<const FactoredUseAccess>>& accesses)
        {
            ++work.syntax;
            FactoredUseRegion out;
            out.owner = region.originalOwner;
            if (region.kind == Region::For || region.kind == Region::While) {
                out.kind = FactoredUseRegion::Kind::OpaqueRepeat;
                // Do not unroll or evaluate one body as though it were the repeat.
                for (const auto& child : region.children) {
                    if (!summarize(child, out.mayRead, out.mayWrite)) {
                        out.kind = FactoredUseRegion::Kind::Unresolved;
                    }
                }
                return out;
            }
            if (region.kind == Region::Operation) {
                if (region.operation >= original.operations.size()) {
                    out.kind = FactoredUseRegion::Kind::Unresolved;
                    return out;
                }
                out.kind = FactoredUseRegion::Kind::Access;
                out.access = accesses.size();
                auto access = std::make_shared<FactoredUseAccess>();
                access->operation = region.operation;
                const auto& effects = original.operations[region.operation].accesses;
                for (std::size_t incidence = 0; incidence < effects.size(); ++incidence) {
                    ++work.effectIncidences;
                    const auto& effect = effects[incidence];
                    if (effect.cell != cell) {
                        continue;
                    }
                    access->read |= effect.read;
                    access->write |= effect.write;
                    // This is a consumer of the common coverage qualification, not a
                    // geometry-based substitute for that proof (Phase A step 3).
                    access->definiteWrite |= effect.write && effect.definiteWrite;
                    if (effect.read) {
                        access->readIncidences.push_back(incidence);
                    }
                    if (effect.write) {
                        access->writeIncidences.push_back(incidence);
                    }
                }
                accesses.push_back(std::move(access));
                return out;
            }
            if (region.kind == Region::Choice) {
                out.kind = FactoredUseRegion::Kind::Choice;
                out.condition = guard(region);
            }
            for (const auto& child : region.children) {
                out.children.push_back(project(child, accesses));
            }
            return out;
        }
    };
    FactoredUseResult result;
    ProjectionWork projectionWork;
};

// One original instance owns the default projections. Different cells, loop
// bodies, and while regions have different cache entries. Explicit boundary
// applications are not cached under these default-input keys.
class FactoredUseService {
public:
    explicit FactoredUseService(const OriginalStructure& original, const OriginalValueQueries* values = nullptr)
        : original(original),
          version(original.version),
          values(values),
          scopes(original.operations.size()),
          represented(original.operations.size())
    {
        index(original.body, {});
        invalid.reason = "invalid original operation or physical cell";
        changed.reason = "original program changed; rebuild factored use service";
    }
    const FactoredUseResult& root(std::size_t cell) const { return get(cell, {}); }
    const FactoredUseResult& at(std::size_t operation, std::size_t cell) const
    {
        return operation < scopes.size() && represented[operation] ? get(cell, scopes[operation]) : invalid;
    }
    // The supplied roots may come from a previously formed acyclic transfer.
    // The arena's frame must match this exact scope. Cross-reentry transport is
    // deliberately not inferred; later D4 supplies that separate qualification.
    FactoredUseResult applyAt(std::size_t operation, std::size_t cell, FactoredUseInterface boundary) const
    {
        if (original.version != version) {
            return changed;
        }
        if (operation >= scopes.size() || !represented[operation] || cell >= original.cells.size() || !boundary.arena) {
            return invalid;
        }
        return FactoredProvenance(original, cell, scopes[operation], std::move(boundary), values).take();
    }

private:
    const FactoredUseResult& get(std::size_t cell, FactoredProjectionScope scope) const
    {
        if (original.version != version) {
            return changed;
        }
        if (cell >= original.cells.size()) {
            return invalid;
        }
        const auto key = std::make_pair(cell, scope);
        auto found = cache.find(key);
        if (found == cache.end()) {
            found = cache
                        .emplace(
                            key,
                            std::make_unique<FactoredProvenance>(original, cell, scope, FactoredUseInterface{}, values))
                        .first;
        }
        return found->second->get();
    }
    void index(const Region& region, FactoredProjectionScope scope)
    {
        if (region.kind == Region::Operation && region.operation < scopes.size()) {
            scopes[region.operation] = scope;
            represented[region.operation] = true;
        }
        for (std::size_t child = 0; child < region.children.size(); ++child) {
            auto next = scope;
            if (region.kind == Region::For) {
                next = {FactoredUseFrame::Kind::ForBody, region.originalOwner};
            } else if (region.kind == Region::While) {
                next = {
                    child == 0 ? FactoredUseFrame::Kind::WhileBefore : FactoredUseFrame::Kind::WhileAfter,
                    region.originalOwner};
            }
            index(region.children[child], next);
        }
    }
    const OriginalStructure& original;
    OriginalProgramVersion version;
    const OriginalValueQueries* values;
    std::vector<FactoredProjectionScope> scopes;
    std::vector<bool> represented;
    mutable std::map<std::pair<std::size_t, FactoredProjectionScope>, std::unique_ptr<FactoredProvenance>> cache;
    FactoredUseResult invalid, changed;
};

} // namespace mlir::pto::frontiersynch
#endif
