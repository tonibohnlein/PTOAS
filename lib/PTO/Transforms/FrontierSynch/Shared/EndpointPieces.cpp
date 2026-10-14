// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Group source and destination sites independently from original recipe coordinates.
#include "PTO/Transforms/FrontierSynch/Shared/EndpointPieces.h"
#include "EndpointOrder.h"
#include "MixedEndpointPieces.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Namespace = std::tuple<uint32_t, uint32_t, uint64_t>;
struct Point {
    EndpointPiece piece;
    std::size_t recipe = 0;
    std::size_t owner = 0;
    bool numerical = true;
    std::map<std::size_t, bool> residualOrder;
};
bool sameCut(TemplateEndpointCut a, TemplateEndpointCut b)
{
    return a.block == b.block && a.before == b.before;
}
bool sameCoordinates(ArrayRef<TemplateCoordinate> a, ArrayRef<TemplateCoordinate> b, bool values)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (auto [x, y] : llvm::zip(a, b)) {
        if (x.loop != y.loop || (values && x.induction != y.induction)) {
            return false;
        }
    }
    return true;
}
bool disjoint(const Point& a, const Point& b)
{
    for (auto x : a.piece.members.front().coordinates) {
        for (auto y : b.piece.members.front().coordinates) {
            if (x.loop == y.loop && x.induction != y.induction) {
                return true;
            }
        }
    }
    return false;
}
bool compatible(const EndpointPiece& a, const EndpointPiece& b)
{
    if (a.namespaceId != b.namespaceId || a.sourcePipe != b.sourcePipe || a.targetPipe != b.targetPipe ||
        a.displacement != b.displacement || a.kind != b.kind || !sameCut(a.cut, b.cut) ||
        !sameCoordinates(a.members.front().coordinates, b.members.front().coordinates, false)) {
        return false;
    }
    return llvm::none_of(a.members, [&b](const EndpointPieceMember& member) {
        return sameCoordinates(member.coordinates, b.members.front().coordinates, true);
    });
}
std::string collect(const NumericTemplateEndpoints& plan, std::vector<Point>& points)
{
    std::map<Namespace, uint32_t> namespaces;
    std::set<std::pair<uint32_t, EndpointKind>> seen;
    for (std::size_t i = 0; i < plan.logical.recipes.size(); ++i) {
        const auto& recipe = plan.logical.recipes[i];
        if (recipe.source >= plan.anchors.size() || recipe.target >= plan.anchors.size() ||
            !seen.emplace(recipe.record, recipe.kind).second) {
            return "invalid or duplicate endpoint-piece recipe";
        }
        const auto& source = plan.anchors[recipe.source];
        const auto& target = plan.anchors[recipe.target];
        if (!source.phase || !target.phase) {
            return "endpoint-piece recipe has no original phase";
        }
        const auto p = static_cast<uint32_t>(source.phase->kPipeValue);
        const auto q = static_cast<uint32_t>(target.phase->kPipeValue);
        const bool publish = recipe.kind == EndpointKind::Set;
        if ((p == q ? recipe.kind != EndpointKind::Barrier :
                      recipe.kind != EndpointKind::Set && recipe.kind != EndpointKind::Wait) ||
            recipe.pipe != (publish ? p : q)) {
            return "endpoint-piece recipe has inconsistent pipes or command kind";
        }
        const auto& anchor = publish ? source : target;
        auto cut = publish ? source.after : target.before;
        if (!cut.block || !cut.before) {
            return "endpoint-piece recipe has no insertion cut";
        }
        std::set<Operation*> axes;
        for (auto coordinate : anchor.coordinates) {
            if (!coordinate.loop || !axes.insert(coordinate.loop.getOperation()).second) {
                return "endpoint-piece coordinates require distinct original loops";
            }
        }
        const Namespace key{p, q, recipe.displacement};
        auto [entry, inserted] = namespaces.try_emplace(key, recipe.record);
        entry->second = std::min(entry->second, recipe.record);
        EndpointPiece piece{0, p, q, recipe.displacement, recipe.kind, cut, 0,
                            {{recipe.record, anchor.coordinates}}};
        points.push_back({std::move(piece), i, 0});
    }
    for (auto& point : points) {
        auto& piece = point.piece;
        piece.namespaceId = namespaces.at({piece.sourcePipe, piece.targetPipe, piece.displacement});
    }
    return {};
}
bool earlier(const Point& a, const Point& b)
{
    const auto pipeA = a.piece.kind == EndpointKind::Barrier ? a.piece.sourcePipe : 0;
    const auto pipeB = b.piece.kind == EndpointKind::Barrier ? b.piece.sourcePipe : 0;
    return std::tie(a.piece.kind, pipeA, a.recipe) < std::tie(b.piece.kind, pipeB, b.recipe);
}
EndpointOrder order(ArrayRef<Point> points, std::size_t count)
{
    std::vector<std::set<std::size_t>> edges(count);
    // Points at different original cuts cannot constrain one another. Keep
    // their candidate inventories separate before testing coordinate exclusion.
    std::map<std::pair<Block*, Operation*>, std::vector<std::size_t>> cuts;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto cut = points[i].piece.cut;
        cuts[{cut.block, cut.before}].push_back(i);
    }
    auto connect = [&](std::size_t first, std::size_t second) {
        const auto& a = points[first];
        const auto& b = points[second];
        if (a.owner == b.owner || disjoint(a, b)) { return; }
        const auto from = earlier(a, b) ? a.owner : b.owner;
        const auto to = earlier(a, b) ? b.owner : a.owner;
        edges[from].insert(to);
    };
    for (const auto& [cut, selected] : cuts) {
        SmallVector<std::size_t> numerical, residual;
        for (auto index : selected) {
            (points[index].numerical ? numerical : residual).push_back(index);
        }
        // Residual singletons never merge. Keep their original relative order
        // with a chain instead of constructing a dense residual-pair graph.
        // Exclusion comparisons are needed only for the numerical groups that
        // may move. This preserves order conservatively between residuals.
        llvm::sort(residual, [&](auto a, auto b) { return earlier(points[a], points[b]); });
        for (std::size_t i = 1; i < residual.size(); ++i) {
            edges[points[residual[i - 1]].owner].insert(points[residual[i]].owner);
        }
        for (std::size_t i = 0; i < numerical.size(); ++i) {
            for (std::size_t j = i + 1; j < numerical.size(); ++j) { connect(numerical[i], numerical[j]); }
            for (auto other : residual) { connect(numerical[i], other); }
        }
    }
    return orderEndpointGroups(edges);
}
std::vector<EndpointPiece> group(std::vector<Point>& points, const std::set<std::size_t>& split = {})
{
    std::vector<EndpointPiece> pieces;
    // On refinement, keep independent pieces outside a cyclic group intact.
    std::map<std::size_t, std::size_t> unchanged;
    std::set<std::size_t> opaque;
    std::vector<std::map<std::size_t, bool>> constraints;
    for (auto& point : points) {
        auto found = pieces.end();
        if (split.empty()) {
            if (point.numerical) {
                found = std::find_if(pieces.begin(), pieces.end(), [&](const EndpointPiece& candidate) {
                    const auto index = static_cast<std::size_t>(&candidate - pieces.data());
                    if (opaque.count(index) || !compatible(candidate, point.piece)) { return false; }
                    return llvm::all_of(point.residualOrder, [&](const auto& constraint) {
                        auto found = constraints[index].find(constraint.first);
                        return found == constraints[index].end() || found->second == constraint.second;
                    });
                });
            }
        } else if (!split.count(point.owner)) {
            auto entry = unchanged.find(point.owner);
            if (entry != unchanged.end()) {
                found = pieces.begin() + entry->second;
            } else {
                unchanged.emplace(point.owner, pieces.size());
            }
        }
        if (found == pieces.end()) {
            point.owner = pieces.size();
            if (!point.numerical) { opaque.insert(point.owner); }
            pieces.push_back(point.piece);
            constraints.push_back(point.residualOrder);
        } else {
            point.owner = static_cast<std::size_t>(found - pieces.begin());
            found->members.push_back(point.piece.members.front());
            constraints[point.owner].insert(point.residualOrder.begin(), point.residualOrder.end());
        }
    }
    return pieces;
}
EndpointPieces reject(std::string message)
{
    EndpointPieces result;
    result.error = std::move(message);
    return result;
}
void retainResidualOrder(std::vector<Point>& points)
{
    using CutKind = std::tuple<Block*, Operation*, EndpointKind>;
    std::map<CutKind, SmallVector<std::size_t>> residuals;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto& point = points[i];
        if (!point.numerical) {
            residuals[{point.piece.cut.block, point.piece.cut.before, point.piece.kind}].push_back(i);
        }
    }
    // Avoid grouping across a coactive residual and then splitting every
    // numerical member of the resulting cycle. Retain before/after constraints
    // as groups form; the full graph still verifies all numerical interactions.
    for (auto& point : points) {
        if (!point.numerical) { continue; }
        auto found = residuals.find({point.piece.cut.block, point.piece.cut.before, point.piece.kind});
        if (found == residuals.end()) { continue; }
        for (auto index : found->second) {
            if (!disjoint(point, points[index])) { point.residualOrder.emplace(index, earlier(point, points[index])); }
        }
    }
}
EndpointPieces partition(std::vector<Point>& points)
{
    EndpointPieces result;
    result.pieces = group(points);
    auto ordering = order(points, result.pieces.size());
    if (!ordering.cyclic.empty()) {
        result.pieces = group(points, ordering.cyclic);
        ordering = order(points, result.pieces.size());
    }
    if (!ordering.cyclic.empty()) {
        return reject("endpoint pieces could not preserve coincident command order");
    }
    for (std::size_t i = 0; i < ordering.positions.size(); ++i) {
        result.pieces[ordering.positions[i]].order = i;
    }
    return result;
}
} // namespace
EndpointPieces buildEndpointPieces(const NumericTemplateEndpoints& plan)
{
    if (!plan.logical.error.empty()) { return reject("cannot partition a failed logical endpoint plan"); }
    std::vector<Point> points;
    if (auto error = collect(plan, points); !error.empty()) { return reject(std::move(error)); }
    return partition(points);
}
EndpointPieces buildMixedEndpointPieces(ArrayRef<EndpointPiece> supplied, ArrayRef<uint8_t> numerical)
{
    if (supplied.size() != numerical.size()) { return reject("mixed endpoint qualification count differs"); }
    std::vector<Point> points;
    std::map<Namespace, uint32_t> namespaces;
    std::set<std::tuple<uint32_t, EndpointKind, Operation*>> seen;
    for (std::size_t i = 0; i < supplied.size(); ++i) {
        const auto& piece = supplied[i];
        if (!piece.cut.before || piece.cut.block != piece.cut.before->getBlock() || piece.members.size() != 1 ||
            (piece.kind != EndpointKind::Set && piece.kind != EndpointKind::Wait &&
             piece.kind != EndpointKind::Barrier) ||
            ((piece.sourcePipe == piece.targetPipe) != (piece.kind == EndpointKind::Barrier))) {
            return reject("invalid mixed endpoint point");
        }
        const auto record = piece.members.front().record;
        if (!seen.emplace(record, piece.kind, piece.cut.before).second) {
            return reject("duplicate mixed endpoint at original cut");
        }
        std::set<Operation*> axes;
        for (auto coordinate : piece.members.front().coordinates) {
            if (!coordinate.loop || !axes.insert(coordinate.loop.getOperation()).second) {
                return reject("mixed endpoint coordinates require distinct original loops");
            }
        }
        const Namespace key{piece.sourcePipe, piece.targetPipe, piece.displacement};
        auto entry = namespaces.emplace(key, record).first;
        entry->second = std::min(entry->second, record);
        points.push_back({piece, i, 0, bool(numerical[i])});
    }
    for (auto& point : points) {
        auto& piece = point.piece;
        piece.namespaceId = namespaces.at({piece.sourcePipe, piece.targetPipe, piece.displacement});
    }
    retainResidualOrder(points);
    return partition(points);
}
} // namespace mlir::pto::frontiersynch
