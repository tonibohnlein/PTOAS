// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_FAMILYENDPOINTNORMALIZATION_H
#define PTO_FRONTIER_FAMILYENDPOINTNORMALIZATION_H
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyExpressions.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <set>
namespace mlir::pto::frontiersynch::detail {
template<class Endpoint, class Combine>
auto combineBarrierGuards(ArrayRef<const Endpoint*> endpoints, Combine combine)
{
    using Scalar = decltype(Endpoint{}.guard);
    std::map<uint32_t, Scalar> guards;
    for (const auto* endpoint : endpoints) {
        if (endpoint->kind != LogicalCommandKind::Barrier) { continue; }
        auto [entry, fresh] = guards.emplace(endpoint->sourcePipe, endpoint->guard);
        if (!fresh) { entry->second = combine(entry->second, endpoint->guard); }
    }
    return guards;
}
template<class Scalar> struct ScalarGuardPattern {
    enum Kind { Opaque, Constant, And, Select, Equal } kind = Opaque;
    SmallVector<Scalar> operands;
    std::optional<APInt> constant;
};
template<class Scalar>
ScalarGuardPattern<Scalar> scalarGuardPattern(StringRef name, Attribute properties, ArrayRef<Scalar> operands)
{
    ScalarGuardPattern<Scalar> result;
    auto attributes = dyn_cast_or_null<DictionaryAttr>(properties);
    if (name == "arith.constant" && attributes) {
        if (auto value = attributes.getAs<IntegerAttr>("value")) {
            result.kind = ScalarGuardPattern<Scalar>::Constant; result.constant = value.getValue();
        }
    } else if (name == "arith.andi" && operands.size() == 2) { result.kind = ScalarGuardPattern<Scalar>::And; }
    else if (name == "arith.select" && operands.size() == 3) { result.kind = ScalarGuardPattern<Scalar>::Select; }
    else if (name == "arith.cmpi" && operands.size() == 2 && attributes) {
        auto predicate = attributes.getAs<arith::CmpIPredicateAttr>("predicate");
        if (predicate && predicate.getValue() == arith::CmpIPredicate::eq) {
            result.kind = ScalarGuardPattern<Scalar>::Equal;
        }
    }
    result.operands.assign(operands.begin(), operands.end());
    return result;
}
template<class Consumer>
std::optional<SmallVector<typename Consumer::Scalar>> residualGuard(typename Consumer::Scalar guard,
    ArrayRef<TemplateCoordinate> coordinates, Consumer& consumer)
{
    using Scalar = typename Consumer::Scalar;
    using Pattern = ScalarGuardPattern<Scalar>;
    SmallVector<Scalar> pending{guard}, residual;
    llvm::DenseSet<Scalar> seen, bound;
    while (!pending.empty()) {
        auto value = pending.pop_back_val();
        if (!seen.insert(value).second) { continue; }
        auto pattern = consumer.pattern(value);
        if (pattern.constant && pattern.constant->isOne()) { continue; }
        if (pattern.kind == Pattern::And) {
            pending.append(pattern.operands.begin(), pattern.operands.end()); continue;
        }
        if (pattern.kind == Pattern::Select) {
            auto fallback = consumer.pattern(pattern.operands[2]);
            if (fallback.constant && fallback.constant->isZero()) {
                pending.push_back(pattern.operands[0]); pending.push_back(pattern.operands[1]); continue;
            }
        }
        bool coordinateTest = false;
        if (pattern.kind == Pattern::Equal) {
            for (auto coordinate : coordinates) {
                auto induction = consumer.original(coordinate.loop.getInductionVar());
                auto other = pattern.operands[0] == induction ? pattern.operands[1] :
                    pattern.operands[1] == induction ? pattern.operands[0] : Scalar{};
                auto literal = other ? consumer.pattern(other).constant : std::optional<APInt>{};
                if (literal && literal->isSignedIntN(64) && literal->getSExtValue() == coordinate.induction) {
                    bound.insert(induction); coordinateTest = true; break;
                }
            }
        }
        if (!coordinateTest) { residual.push_back(value); }
    }
    if (bound.size() != coordinates.size()) { return std::nullopt; }
    llvm::sort(residual, [&](Scalar a, Scalar b) { return consumer.less(a, b); });
    return residual;
}
template<class Endpoint>
bool disjointDomains(const Endpoint& a, const Endpoint& b, bool residueOnly = false)
{
    if (a.domain.empty() || b.domain.empty()) { return false; }
    for (const auto& left : a.domain) {
        for (const auto& right : b.domain) {
            bool excludes = false;
            for (const auto& x : left) {
                for (const auto& y : right) {
                    excludes |= (!residueOnly || x.modulus) && x.coordinate == y.coordinate &&
                        x.modulus == y.modulus && x.value != y.value;
                }
            }
            if (!excludes) { return false; }
        }
    }
    return true;
}
// These producer-owned necessary domains need not describe the complete guard.
// Distinct residues of one shared scalar prove exclusion even for opaque exact
// arithmetic predicates. Preserve all predicate work and select only provenance.
template<class Endpoint>
bool sameAlternativeMapping(const Endpoint& a, const Endpoint& b)
{
    return a.record == b.record && a.identity == b.identity &&
        a.sourcePipe == b.sourcePipe && a.targetPipe == b.targetPipe &&
        !a.memberCoordinates.empty() && !b.memberCoordinates.empty() &&
        bool(a.memberCoordinates.front()) == bool(b.memberCoordinates.front()) &&
        llvm::ArrayRef(a.memberCoordinates).drop_front() == llvm::ArrayRef(b.memberCoordinates).drop_front() &&
        ((!a.physicalMapping && !b.physicalMapping) || (a.physicalMapping && b.physicalMapping &&
            equivalentPhysicalMaps(*a.physicalMapping, *b.physicalMapping)));
}
template<class Plan, class Consumer>
LogicalResult coalesceResidueAlternatives(Plan& plan, Consumer& consumer)
{
    if (!plan.independentPieces) { return success(); }
    using Key = std::pair<Operation*, LogicalCommandKind>;
    std::map<Key, std::size_t> previous;
    decltype(plan.endpoints) retained;
    for (auto& endpoint : plan.endpoints) {
        if (endpoint.kind == LogicalCommandKind::Barrier) { retained.push_back(std::move(endpoint)); continue; }
        auto [position, fresh] = previous.emplace(Key{endpoint.before, endpoint.kind}, retained.size());
        if (fresh || !sameAlternativeMapping(retained[position->second], endpoint) ||
            !disjointDomains(retained[position->second], endpoint, true)) {
            position->second = retained.size(); retained.push_back(std::move(endpoint)); continue;
        }
        auto& preceding = retained[position->second];
        if (failed(consumer.at(endpoint.before, [&]() {
            auto& member = preceding.memberCoordinates.front();
            if (member) { member = consumer.select(endpoint.guard, endpoint.memberCoordinates.front(), member); }
            preceding.guard = consumer.lor(preceding.guard, endpoint.guard);
            return success();
        }))) { return failure(); }
        preceding.records.append(endpoint.records.begin(), endpoint.records.end());
        preceding.domain.append(endpoint.domain.begin(), endpoint.domain.end());
    }
    plan.endpoints = std::move(retained);
    return success();
}
// Move an alternative earlier only across endpoints that cannot execute with
// it. All potentially simultaneous commands retain their original relative
// order, including commands in different directions. The bounded search is a
// compaction heuristic, never an admission limit or a semantic approximation.
template <typename Plan, typename DomainFor>
void scheduleExclusiveAlternatives(Plan& plan, DomainFor domain)
{
    using Domain = decltype(domain(plan.endpoints.front()));
    SmallVector<Domain> domains;
    for (const auto& endpoint : plan.endpoints) {
        domains.push_back(endpoint.kind == LogicalCommandKind::Barrier || endpoint.memberCoordinates.empty() ?
            Domain{} : domain(endpoint));
    }
    using Key = std::pair<Operation*, LogicalCommandKind>;
    std::map<Key, SmallVector<std::size_t>> cuts;
    for (std::size_t i = 0; i < plan.endpoints.size(); ++i) {
        const auto& endpoint = plan.endpoints[i];
        if (endpoint.kind == LogicalCommandKind::Barrier) { continue; }
        cuts[{endpoint.before, endpoint.kind}].push_back(i);
    }
    constexpr std::size_t lookback = 128;
    for (const auto& [cut, indices] : cuts) {
        (void)cut;
        for (std::size_t i = 1; i < indices.size(); ++i) {
            const auto& current = plan.endpoints[indices[i]];
            if (current.records.size() != 1 || !domains[indices[i]]) { continue; }
            std::optional<std::size_t> destination;
            for (std::size_t j = i; j && i - j < lookback; --j) {
                const auto& previous = plan.endpoints[indices[j - 1]];
                if (!disjointDomains(current, previous)) { break; }
                if (previous.records.size() == 1 && domains[indices[j - 1]] &&
                    domains[indices[i]]->axes == domains[indices[j - 1]]->axes &&
                    domains[indices[i]]->residual == domains[indices[j - 1]]->residual &&
                    current.record == previous.record &&
                    current.sourcePipe == previous.sourcePipe &&
                    current.targetPipe == previous.targetPipe && current.identity == previous.identity &&
                    llvm::ArrayRef(current.memberCoordinates).drop_front() ==
                        llvm::ArrayRef(previous.memberCoordinates).drop_front()) { destination = j - 1; }
            }
            if (!destination || *destination + 1 == i) { continue; }
            auto moved = std::move(plan.endpoints[indices[i]]);
            auto movedDomain = std::move(domains[indices[i]]);
            for (std::size_t j = i; j > *destination + 1; --j) {
                plan.endpoints[indices[j]] = std::move(plan.endpoints[indices[j - 1]]);
                domains[indices[j]] = std::move(domains[indices[j - 1]]);
            }
            plan.endpoints[indices[*destination + 1]] = std::move(moved);
            domains[indices[*destination + 1]] = std::move(movedDomain);
        }
    }
}
// Combine consecutive command alternatives only when exact finite coordinates
// prove mutual exclusion. Certificates retain every record; labels remain
// provenance, and simultaneously active commands are never collapsed.
template<class Plan, class Consumer>
LogicalResult coalesceEndpointAlternatives(Plan& plan, Consumer& consumer)
{
    using Scalar = typename Consumer::Scalar;
    using Endpoint = typename decltype(plan.endpoints)::value_type;
    if (!plan.independentPieces || plan.endpoints.empty()) { return success(); }
    auto* cut = plan.endpoints.front().before;
    const auto width = DataLayout::closest(cut).getTypeSizeInBits(IndexType::get(cut->getContext()));
    if (width.isScalable() || width.getFixedValue() != 64) { return success(); }
    struct Domain {
        SmallVector<Operation*> axes;
        std::set<std::vector<int64_t>> points;
        SmallVector<Scalar> residual;
    };
    std::map<uint32_t, std::pair<const EndpointFamily*, const EndpointFamilyMember*>> members;
    for (const auto& family : plan.families) {
        for (const auto& member : family.members) { members.emplace(member.record, std::make_pair(&family, &member)); }
    }
    auto domain = [&](const Endpoint& endpoint) -> std::optional<Domain> {
        Domain result;
        // Already grouped domains keep their existing compact representation.
        if (endpoint.records.size() != 1) { return std::nullopt; }
        for (auto record : endpoint.records) {
            const auto [family, member] = members.at(record);
            if (family->sourceChoices || family->targetChoices) { return std::nullopt; }
            const auto& coordinates = endpoint.kind == LogicalCommandKind::Set ?
                member->sourceCoordinates : member->targetCoordinates;
            if (coordinates.empty()) { return std::nullopt; }
            auto remaining = residualGuard(endpoint.guard, coordinates, consumer);
            if (!remaining) { return std::nullopt; }
            result.residual = std::move(*remaining);
            SmallVector<Operation*> axes;
            std::vector<int64_t> point;
            for (auto coordinate : coordinates) {
                axes.push_back(coordinate.loop.getOperation()); point.push_back(coordinate.induction);
            }
            if (result.points.empty()) { result.axes = axes; }
            if (result.axes != axes || !result.points.insert(std::move(point)).second) { return std::nullopt; }
        }
        return result.points.empty() ? std::nullopt : std::optional<Domain>(std::move(result));
    };
    scheduleExclusiveAlternatives(plan, domain);
    using CutKind = std::pair<Operation*, LogicalCommandKind>;
    std::map<CutKind, std::size_t> previous;
    std::map<std::size_t, Domain> domains;
    std::vector<bool> removed(plan.endpoints.size(), false), combined(plan.endpoints.size(), false);
    for (std::size_t i = 0; i < plan.endpoints.size(); ++i) {
        auto& endpoint = plan.endpoints[i];
        if (endpoint.kind == LogicalCommandKind::Barrier) { continue; }
        if (endpoint.memberCoordinates.empty()) {
            previous.erase({endpoint.before, endpoint.kind});
            continue;
        }
        auto current = domain(endpoint);
        auto [entry, fresh] = previous.emplace(CutKind{endpoint.before, endpoint.kind}, i);
        const auto old = entry->second;
        auto known = domains.find(old);
        auto& preceding = plan.endpoints[old];
        const bool compatible =
            !fresh && current && known != domains.end() && preceding.record == endpoint.record &&
            preceding.identity == endpoint.identity &&
            ((!preceding.physicalMapping && !endpoint.physicalMapping) ||
             (preceding.physicalMapping && endpoint.physicalMapping &&
              equivalentPhysicalMaps(*preceding.physicalMapping, *endpoint.physicalMapping))) &&
            preceding.sourcePipe == endpoint.sourcePipe && preceding.targetPipe == endpoint.targetPipe &&
            llvm::ArrayRef(preceding.memberCoordinates).drop_front() ==
                llvm::ArrayRef(endpoint.memberCoordinates).drop_front() &&
            known->second.axes == current->axes && known->second.residual == current->residual &&
            llvm::none_of(current->points, [&](const auto& point) { return known->second.points.count(point); });
        if (!compatible) {
            entry->second = i;
            if (current) { domains.emplace(i, std::move(*current)); }
            continue;
        }
        preceding.records.append(endpoint.records.begin(), endpoint.records.end());
        if (preceding.domain.empty() || endpoint.domain.empty()) { preceding.domain.clear(); }
        else { preceding.domain.append(endpoint.domain.begin(), endpoint.domain.end()); }
        known->second.points.insert(current->points.begin(), current->points.end());
        removed[i] = true; combined[old] = true;
    }
    decltype(plan.endpoints) retained;
    for (std::size_t i = 0; i < plan.endpoints.size(); ++i) {
        if (removed[i]) { continue; }
        auto& endpoint = plan.endpoints[i];
        if (combined[i]) {
            SmallVector<SmallVector<TemplateCoordinate>> points;
            SmallVector<int64_t> labels;
            for (auto record : endpoint.records) {
                const auto* member = members.at(record).second;
                points.push_back(endpoint.kind == LogicalCommandKind::Set ?
                    member->sourceCoordinates : member->targetCoordinates);
                labels.push_back(record);
            }
            const bool needMember = !endpoint.physicalMapping || physicalMapRequiresMember(*endpoint.physicalMapping);
            auto selected = buildFamilyExpressions(points, labels);
            if (!selected.error.empty()) { return failure(); }
            if (failed(consumer.at(endpoint.before, [&]() {
                    auto selection = consumer.selection(*selected.recipe, needMember);
                    endpoint.memberCoordinates.front() = selection.second;
                    endpoint.guard = selection.first;
                    for (auto factor : domains.at(i).residual) {
                        endpoint.guard = consumer.land(endpoint.guard, factor);
                    }
                    return success();
                }))) { return failure(); }
        }
        retained.push_back(std::move(endpoint));
    }
    plan.endpoints = std::move(retained);
    return success();
}
} // namespace mlir::pto::frontiersynch::detail
#endif
