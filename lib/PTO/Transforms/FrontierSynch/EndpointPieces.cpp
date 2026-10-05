// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Group source and destination sites independently from original recipe coordinates.
#include "PTO/Transforms/FrontierSynch/EndpointPieces.h"
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
    if (a.namespaceId != b.namespaceId || a.kind != b.kind || !sameCut(a.cut, b.cut) ||
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
struct Order {
    std::vector<std::size_t> positions;
    std::set<std::size_t> split;
};
Order order(ArrayRef<Point> points, std::size_t count)
{
    std::vector<std::set<std::size_t>> edges(count);
    std::vector<std::size_t> incoming(count, 0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = i + 1; j < points.size(); ++j) {
            const auto& a = points[i];
            const auto& b = points[j];
            if (a.owner == b.owner || !sameCut(a.piece.cut, b.piece.cut) || disjoint(a, b)) {
                continue;
            }
            const auto from = earlier(a, b) ? a.owner : b.owner;
            const auto to = earlier(a, b) ? b.owner : a.owner;
            if (edges[from].insert(to).second) {
                ++incoming[to];
            }
        }
    }
    Order result;
    for (std::size_t i = 0; i < count; ++i) {
        if (!incoming[i]) {
            result.positions.push_back(i);
        }
    }
    for (std::size_t next = 0; next < result.positions.size(); ++next) {
        for (const auto target : edges[result.positions[next]]) {
            if (!--incoming[target]) {
                result.positions.push_back(target);
            }
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (incoming[i]) {
            result.split.insert(i);
        }
    }
    return result;
}
std::vector<EndpointPiece> group(std::vector<Point>& points, const std::set<std::size_t>& split = {})
{
    std::vector<EndpointPiece> pieces;
    // On refinement, keep independent pieces outside a cyclic group intact.
    std::map<std::size_t, std::size_t> unchanged;
    for (auto& point : points) {
        auto found = pieces.end();
        if (split.empty()) {
            found = std::find_if(pieces.begin(), pieces.end(), [&point](const EndpointPiece& candidate) {
                return compatible(candidate, point.piece);
            });
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
            pieces.push_back(point.piece);
        } else {
            point.owner = static_cast<std::size_t>(found - pieces.begin());
            found->members.push_back(point.piece.members.front());
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
} // namespace
EndpointPieces buildEndpointPieces(const NumericTemplateEndpoints& plan)
{
    if (!plan.logical.error.empty()) {
        return reject("cannot partition a failed logical endpoint plan");
    }
    std::vector<Point> points;
    if (auto error = collect(plan, points); !error.empty()) {
        return reject(std::move(error));
    }
    EndpointPieces result;
    result.pieces = group(points);
    auto ordering = order(points, result.pieces.size());
    if (!ordering.split.empty()) {
        result.pieces = group(points, ordering.split);
        ordering = order(points, result.pieces.size());
    }
    if (!ordering.split.empty()) {
        return reject("endpoint pieces could not preserve coincident command order");
    }
    for (std::size_t i = 0; i < ordering.positions.size(); ++i) {
        result.pieces[ordering.positions[i]].order = i;
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
