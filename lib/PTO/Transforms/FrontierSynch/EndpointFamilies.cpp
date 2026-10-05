// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Form exact finite endpoint families from analysis coordinates, without reading emitted guards.
#include "PTO/Transforms/FrontierSynch/EndpointFamilies.h"
#include <algorithm>
#include <map>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
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
bool validCoordinates(ArrayRef<TemplateCoordinate> coordinates)
{
    std::set<Operation*> seen;
    for (auto coordinate : coordinates) {
        if (!coordinate.loop || !seen.insert(coordinate.loop.getOperation()).second) {
            return false;
        }
    }
    return true;
}
bool disjoint(ArrayRef<TemplateCoordinate> a, ArrayRef<TemplateCoordinate> b)
{
    for (auto x : a) {
        for (auto y : b) {
            if (x.loop == y.loop && x.induction != y.induction) {
                return true;
            }
        }
    }
    return false;
}
bool compatible(const EndpointFamily& a, const EndpointFamily& b)
{
    const auto& x = a.members.front();
    const auto& y = b.members.front();
    return std::tie(a.sourcePipe, a.targetPipe, a.displacement, a.local) ==
               std::tie(b.sourcePipe, b.targetPipe, b.displacement, b.local) &&
        sameCut(a.sourceCut, b.sourceCut) && sameCut(a.targetCut, b.targetCut) &&
        sameCoordinates(x.sourceCoordinates, y.sourceCoordinates, false) &&
        sameCoordinates(x.targetCoordinates, y.targetCoordinates, false);
}
bool uniqueMember(const EndpointFamily& family, const EndpointFamilyMember& candidate)
{
    for (const auto& member : family.members) {
        if (sameCoordinates(member.sourceCoordinates, candidate.sourceCoordinates, true) ||
            sameCoordinates(member.targetCoordinates, candidate.targetCoordinates, true)) {
            return false;
        }
    }
    return true;
}
struct Record {
    EndpointFamily family;
    unsigned kinds = 0;
};
std::string collect(const NumericTemplateEndpoints& plan, std::map<uint32_t, Record>& records)
{
    for (const auto& recipe : plan.logical.recipes) {
        if (recipe.source >= plan.anchors.size() || recipe.target >= plan.anchors.size()) {
            return "family recipe has an invalid endpoint index";
        }
        const auto& source = plan.anchors[recipe.source];
        const auto& target = plan.anchors[recipe.target];
        if (!source.phase || !target.phase || !source.after.block || !target.before.block ||
            !validCoordinates(source.coordinates) || !validCoordinates(target.coordinates)) {
            return "family recipe has unavailable cuts or coordinates";
        }
        const auto p = static_cast<uint32_t>(source.phase->kPipeValue);
        const auto q = static_cast<uint32_t>(target.phase->kPipeValue);
        const bool local = p == q;
        const bool validKind = local ? recipe.kind == EndpointKind::Barrier :
            recipe.kind == EndpointKind::Set || recipe.kind == EndpointKind::Wait;
        if (!validKind || recipe.pipe != (recipe.kind == EndpointKind::Set ? p : q)) {
            return "family recipe has inconsistent command kind or pipe";
        }
        EndpointFamily family{recipe.record, p, q, recipe.displacement, local, source.after, target.before,
            {{recipe.record, recipe.source, recipe.target, source.coordinates, target.coordinates}}};
        auto [position, inserted] = records.try_emplace(recipe.record, Record{family, 0});
        auto& record = position->second;
        const auto bit = 1U << static_cast<unsigned>(recipe.kind);
        const auto& previous = record.family.members.front();
        if ((!inserted && (!compatible(record.family, family) || previous.source != recipe.source ||
                           previous.target != recipe.target)) || (record.kinds & bit)) {
            return "family record has duplicate or inconsistent endpoints";
        }
        record.kinds |= bit;
    }
    for (const auto& entry : records) {
        const auto& record = entry.second;
        const unsigned expected = record.family.local ? 2U : 5U; // Barrier or SET plus WAIT.
        if (record.kinds != expected) {
            return "family record is missing its matching endpoint";
        }
    }
    return {};
}
struct Piece {
    std::size_t family = 0;
    EndpointKind kind = EndpointKind::Set;
    uint32_t pipe = 0;
    uint32_t record = 0;
    ArrayRef<TemplateCoordinate> coordinates;
    std::size_t recipeOrder = 0;
};
struct CutPieces {
    TemplateEndpointCut cut;
    std::vector<Piece> pieces;
};
void addPiece(std::vector<CutPieces>& cuts, TemplateEndpointCut cut, Piece piece)
{
    auto found = std::find_if(cuts.begin(), cuts.end(), [cut](const CutPieces& entry) {
        return sameCut(cut, entry.cut);
    });
    if (found == cuts.end()) {
        cuts.push_back({cut, {piece}});
    } else {
        found->pieces.push_back(piece);
    }
}
std::vector<CutPieces> pieces(const NumericTemplateEndpoints& plan,
                             const std::vector<EndpointFamily>& families)
{
    std::map<std::pair<uint32_t, EndpointKind>, std::size_t> recipeOrder;
    for (std::size_t i = 0; i < plan.logical.recipes.size(); ++i) {
        const auto& recipe = plan.logical.recipes[i];
        recipeOrder[{recipe.record, recipe.kind}] = i;
    }
    std::vector<CutPieces> cuts;
    for (std::size_t index = 0; index < families.size(); ++index) {
        const auto& family = families[index];
        for (const auto& member : family.members) {
            if (!family.local) {
                addPiece(cuts, family.sourceCut, {index, EndpointKind::Set, family.sourcePipe,
                    member.record, member.sourceCoordinates, recipeOrder.at({member.record, EndpointKind::Set})});
            }
            addPiece(cuts, family.targetCut,
                {index, family.local ? EndpointKind::Barrier : EndpointKind::Wait,
                 family.targetPipe, member.record, member.targetCoordinates,
                 recipeOrder.at({member.record, family.local ? EndpointKind::Barrier : EndpointKind::Wait})});
        }
    }
    for (auto& cut : cuts) {
        std::sort(cut.pieces.begin(), cut.pieces.end(), [](const Piece& a, const Piece& b) {
            const auto pipeA = a.kind == EndpointKind::Barrier ? a.pipe : 0;
            const auto pipeB = b.kind == EndpointKind::Barrier ? b.pipe : 0;
            return std::tie(a.kind, pipeA, a.recipeOrder) < std::tie(b.kind, pipeB, b.recipeOrder);
        });
    }
    return cuts;
}
using Node = std::pair<std::size_t, EndpointKind>;
struct CutOrder {
    std::vector<Node> ordered;
    std::set<std::size_t> split;
};
CutOrder order(const CutPieces& cut)
{
    std::map<Node, std::set<Node>> edges;
    std::map<Node, std::size_t> incoming;
    for (const auto& piece : cut.pieces) {
        incoming.try_emplace({piece.family, piece.kind}, 0);
    }
    for (std::size_t i = 0; i < cut.pieces.size(); ++i) {
        for (std::size_t j = i + 1; j < cut.pieces.size(); ++j) {
            const auto& a = cut.pieces[i];
            const auto& b = cut.pieces[j];
            const Node source{a.family, a.kind}, target{b.family, b.kind};
            if (source != target && !disjoint(a.coordinates, b.coordinates) && edges[source].insert(target).second) {
                ++incoming[target];
            }
        }
    }
    CutOrder result;
    for (const auto& entry : incoming) {
        if (!entry.second) {
            result.ordered.push_back(entry.first);
        }
    }
    for (std::size_t next = 0; next < result.ordered.size(); ++next) {
        for (const auto& target : edges[result.ordered[next]]) {
            if (!--incoming[target]) {
                result.ordered.push_back(target);
            }
        }
    }
    for (const auto& entry : incoming) {
        if (entry.second) {
            result.split.insert(entry.first.first);
        }
    }
    return result;
}
void splitFamilies(std::vector<EndpointFamily>& families, const std::set<std::size_t>& split)
{
    std::vector<EndpointFamily> result;
    for (std::size_t index = 0; index < families.size(); ++index) {
        const auto& family = families[index];
        if (!split.count(index)) {
            result.push_back(family);
            continue;
        }
        for (const auto& member : family.members) {
            EndpointFamily singleton{member.record, family.sourcePipe, family.targetPipe,
                family.displacement, family.local, family.sourceCut, family.targetCut, {member}};
            result.push_back(std::move(singleton));
        }
    }
    families = std::move(result);
}
EndpointFamilies reject(std::string error)
{
    EndpointFamilies result;
    result.error = std::move(error);
    return result;
}
} // namespace
EndpointFamilies buildEndpointFamilies(const NumericTemplateEndpoints& plan)
{
    if (!plan.logical.error.empty()) {
        return reject("cannot form families from a failed logical plan");
    }
    std::map<uint32_t, Record> records;
    if (auto error = collect(plan, records); !error.empty()) {
        return reject(std::move(error));
    }
    EndpointFamilies result;
    for (const auto& entry : records) {
        const auto& candidate = entry.second.family;
        auto found = std::find_if(result.families.begin(), result.families.end(),
            [&candidate](const EndpointFamily& family) {
                return compatible(family, candidate) && uniqueMember(family, candidate.members.front());
            });
        if (found == result.families.end()) {
            result.families.push_back(candidate);
        } else {
            found->members.push_back(candidate.members.front());
        }
    }
    std::set<std::size_t> split;
    for (const auto& cut : pieces(plan, result.families)) {
        const auto cutOrder = order(cut);
        split.insert(cutOrder.split.begin(), cutOrder.split.end());
    }
    splitFamilies(result.families, split);
    for (const auto& cut : pieces(plan, result.families)) {
        const auto cutOrder = order(cut);
        if (!cutOrder.split.empty()) {
            return reject("endpoint families could not preserve coincident command order");
        }
        for (std::size_t position = 0; position < cutOrder.ordered.size(); ++position) {
            auto [index, kind] = cutOrder.ordered[position];
            auto& family = result.families[index];
            (kind == EndpointKind::Set ? family.sourceOrder : family.targetOrder) = position;
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
