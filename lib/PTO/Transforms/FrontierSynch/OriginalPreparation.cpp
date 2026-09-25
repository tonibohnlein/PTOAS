// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include <algorithm>

namespace mlir::pto::frontiersynch {
OriginalPreparation::OriginalPreparation(const ProgramAnalysis& source)
    : analysis(source), version(source.structure().version)
{
    prepare();
}
bool OriginalPreparation::current() const
{
    return version == analysis.structure().version && analysis.obligations().complete() && analysis.requests();
}
const std::vector<OriginalPreparedHook>& OriginalPreparation::hooks() const
{
    static const std::vector<OriginalPreparedHook> empty;
    return formed() ? preparedHooks : empty;
}
const std::vector<OriginalSourceBucket>& OriginalPreparation::sourceBuckets() const
{
    static const std::vector<OriginalSourceBucket> empty;
    return formed() ? buckets : empty;
}
const std::vector<std::size_t>& OriginalPreparation::bucketsAt(std::size_t phase) const
{
    static const std::vector<std::size_t> empty;
    const auto found = bySource.find(phase);
    return formed() && found != bySource.end() ? found->second : empty;
}
const std::vector<OriginalObligationFamilyId>& OriginalPreparation::atDeadline(std::size_t site) const
{
    static const std::vector<OriginalObligationFamilyId> empty;
    const auto found = byDeadline.find(site);
    return formed() && found != byDeadline.end() ? found->second : empty;
}
void OriginalPreparation::subscribe(OriginalCut cut, bool conservative, DescriptorFactRef ref)
{
    auto found = hookIndex.find(cut);
    if (found == hookIndex.end()) {
        OriginalPreparedHook hook;
        hook.sufficient = cut;
        hook.conservative = conservative;
        const auto& original = analysis.structure();
        if (resolveOriginalCut(original, cut)) {
            hook.executable = cut;
        } else {
            hook.obstruction = "named original source cut has no executable insertion gap";
            // Retain the analytical phase and the explicitly later conservative
            // hook separately. The later gap is never advertised as exact.
            if (cut.kind == OriginalCut::Kind::Payload && cut.operation < original.operations.size()) {
                OriginalCut outer{original.operations[cut.operation].enclosingAfter, OriginalCut::After};
                if (resolveOriginalCut(original, outer)) { hook.executable = outer; }
            }
            failures.push_back(hook.obstruction);
        }
        found = hookIndex.emplace(cut, preparedHooks.size()).first;
        preparedHooks.push_back(std::move(hook));
    }
    auto& hook = preparedHooks[found->second];
    hook.conservative |= conservative;
    if (ref.index != NoControlId && std::find(hook.references.begin(), hook.references.end(), ref) == hook.references.end()) {
        hook.references.push_back(ref);
    }
}
void OriginalPreparation::prepare()
{
    if (!current()) { failures.push_back("original request repertoire is not current"); return; }
    const auto& original = analysis.structure();
    const auto& requests = *analysis.requests();
    const auto& slots = requests.descriptors();
    using Role = OriginalObligationKey::Role;
    std::map<std::pair<std::size_t, Role>, std::size_t> bucketIndex;
    for (std::size_t site = 0; site < original.originalSites.size(); ++site) {
        for (auto id : analysis.obligationsAt(site)) {
            const auto* family = analysis.obligations().get(id);
            if (!family) { failures.push_back("invalid family in original deadline index"); continue; }
            byDeadline[site].push_back(id);
            if (family->representation == OriginalObligationFamily::Representation::Typed) {
                const auto bucket = buckets.size();
                buckets.push_back({NoControlId, Role::ValueProducer, {id}});
                for (auto source : family->typedSources) {
                    if (!source.incoming) {
                        bySource[source.operation].push_back(bucket);
                        subscribe({source.operation, OriginalCut::After}, true);
                    }
                }
                continue;
            }
            const auto key = std::make_pair(family->key.cell, family->key.sourceRole());
            auto inserted = bucketIndex.emplace(key, buckets.size());
            if (inserted.second) { buckets.push_back({key.first, key.second, {}}); }
            buckets[inserted.first->second].families.push_back(id);
        }
    }
    // One incidence walk and shared bucket references, not family-origin output.
    for (std::size_t phase = 0; phase < original.operations.size(); ++phase) {
        std::set<std::size_t> ids;
        for (const auto& access : original.operations[phase].accesses) {
            ++work.accessIncidences;
            for (auto role : {Role::Reader, Role::Writer}) {
                if (!(role == Role::Reader ? access.read : access.write)) { continue; }
                auto found = bucketIndex.find({access.cell, role});
                if (found != bucketIndex.end()) { ids.insert(found->second); }
            }
        }
        if (!ids.empty()) {
            auto& sourceBuckets = bySource[phase];
            sourceBuckets.insert(sourceBuckets.end(), ids.begin(), ids.end());
            subscribe({phase, OriginalCut::After}, true);
        }
    }
    std::vector<DescriptorSupportLink> pending;
    std::set<DescriptorFactRef> visitedReferences;
    auto observe = [&](DescriptorFactRef ref) {
        if (!visitedReferences.insert(ref).second) { return; }
        ++work.references;
        if (const auto* qualification = requests.observation(ref)) {
            for (const auto& prerequisite : qualification->prerequisites) {
                subscribe({prerequisite.sourcePhase, OriginalCut::After}, true, ref);
            }
        } else {
            failures.push_back("completion prerequisite has no current typed qualification");
        }
    };
    std::set<std::size_t> visitedNodes;
    std::vector<std::size_t> todo;
    for (const auto& group : requests.groups()) {
        for (const auto& slot : group.declaredSlots) { todo.push_back(slot.root); }
    }
    while (!todo.empty()) {
        auto id = todo.back(); todo.pop_back();
        if (!visitedNodes.insert(id).second) { continue; }
        ++work.descriptorNodes;
        const auto& node = slots.node(id);
        if (node.kind == OriginalDescriptorSlots::Kind::Both || node.kind == OriginalDescriptorSlots::Kind::Choose) {
            todo.push_back(node.left); todo.push_back(node.right);
        } else if (node.kind == OriginalDescriptorSlots::Kind::Leaf) {
            const auto& facts = slots.facts(node.facts);
            pending.insert(pending.end(), facts.support.begin(), facts.support.end());
            for (auto ref : facts.obligations) {
                if (!visitedReferences.insert(ref).second) { continue; }
                ++work.references;
                const auto* reference = requests.reference(ref);
                if (!reference) { failures.push_back("stale obligation reference in descriptor"); continue; }
                if (reference->origin && !reference->origin->incoming) {
                    subscribe({reference->origin->operation, OriginalCut::After}, true, ref);
                }
            }
            for (auto ref : facts.observations) { observe(ref); }
            const auto& boundary = slots.getBoundary(node.source);
            for (const auto& endpoint : boundary.endpoints) {
                subscribe(endpoint.cut, !boundary.qualified(), {});
                for (auto witness : boundary.witnesses) { subscribe(endpoint.cut, !boundary.qualified(), witness); }
                for (auto ref : endpoint.observation.prerequisites) { observe(ref); }
            }
            for (const auto& endpoint : slots.getBoundary(node.target).endpoints) {
                for (auto ref : endpoint.observation.prerequisites) { observe(ref); }
            }
        }
    }
    // Register declarations before following any edge; cycles are finite role
    // cycles, and shifts stay attached to the original links.
    for (const auto& answer : requests.boundaryAnswers()) {
        for (const auto& role : answer.declaredRoles) {
            if (role.id.kind != DescriptorFactRef::Kind::SupportRole || role.id.arena != 0 || role.id.universe != version ||
                role.id.index == NoControlId || role.version != version || role.interval.query.version != version) {
                failures.push_back("invalid or stale support role declaration");
                continue;
            }
            auto retained = role;
            const auto interval = analysis.prepareInterval(role.interval.query);
            if (!role.qualified || !interval.valid || interval.interval != role.interval ||
                role.source != role.interval.query.start) {
                retained.qualified = false;
                failures.push_back("support role source or occurrence interval is unresolved");
            }
            auto inserted = roles.emplace(role.id, std::move(retained));
            if (!inserted.second) {
                inserted.first->second.qualified = false;
                failures.push_back("duplicate support role declaration");
            }
        }
    }
    std::set<DescriptorFactRef> visitedRoles;
    while (!pending.empty()) {
        const auto link = pending.back(); pending.pop_back(); ++work.links;
        auto found = roles.find(link.role);
        if (found == roles.end()) { failures.push_back("support link has no declared role"); continue; }
        const auto& role = found->second;
        const auto& relation = role.interval.query;
        auto matches = [&](const FixedVisitCorrespondence& proof) {
            if (!proof.exact || !proof.alternatives ||
                proof.alternatives->sources.frame.kind != FactoredUseFrame::Kind::Invocation) { return false; }
            OriginalIntervalRequest expected;
            expected.version = version;
            expected.selector.cell = proof.alternatives->sources.frame.cell;
            expected.selector.read = proof.alternatives->sources.hazard == FactoredUseNode::Hazard::RAW;
            expected.selector.write = !expected.selector.read;
            expected.start = {proof.source, OriginalCut::After};
            expected.stop = {proof.target, OriginalCut::Before};
            expected.occurrence.source = proof.source;
            expected.occurrence.target = proof.target;
            expected.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::FirstReach;
            return relation == expected;
        };
        const auto* justification = requests.answer(link.justification);
        bool justified = justification && matches(justification->fixedVisit);
        if (const auto* reference = requests.reference(link.justification)) {
            const auto* family = analysis.obligations().get(reference->family);
            if (family && reference->origin && !reference->origin->incoming &&
                family->key.kind != OriginalObligationKey::Kind::Typed &&
                family->key.cell == relation.selector.cell &&
                reference->origin->operation == relation.occurrence.source &&
                family->key.consumerOperation == relation.occurrence.target) {
                const auto hazard = family->key.kind == OriginalObligationKey::Kind::RAW ? FactoredUseNode::Hazard::RAW :
                                    family->key.kind == OriginalObligationKey::Kind::WAR ? FactoredUseNode::Hazard::WAR :
                                                                                       FactoredUseNode::Hazard::WAW;
                justified = matches(analysis.occurrences().fixedVisit(
                    reference->origin->operation, family->key.consumerOperation, family->key.cell, hazard));
            }
        }
        const bool linkQualified = justified && link.occurrenceShift == 0 &&
                                   relation.occurrence.stopVisit == OriginalOccurrenceContext::StopVisit::FirstReach &&
                                   link.applicability.domain == DescriptorPredicate::Domain::True;
        if (!linkQualified) {
            failures.push_back("support link occurrence shift or applicability lacks a supported qualification");
        }
        subscribe(role.source, !role.qualified || !linkQualified, role.id);
        if (!visitedRoles.insert(link.role).second) { continue; }
        ++work.roles;
        pending.insert(pending.end(), role.links.begin(), role.links.end());
    }
    frozen = true;
}
namespace {
// Extract necessary literals only. No valuation enumeration or claim of a
// complete SAT solver: remaining conditions stay attached to the opportunity.
bool compatibleConditions(const ObligationConditions& predicates, std::vector<std::size_t> roots)
{
    using K = ObligationConditions::Node::Kind;
    std::vector<std::pair<std::size_t, bool>> todo;
    for (auto root : roots) { todo.push_back({root, true}); }
    std::set<std::pair<std::size_t, bool>> visited;
    std::map<std::size_t, bool> literals;
    while (!todo.empty()) {
        const auto [id, truth] = todo.back(); todo.pop_back();
        if (!visited.insert({id, truth}).second) { continue; }
        const auto* n = predicates.get(id);
        if (!n) { return false; }
        if (n->kind == K::True || n->kind == K::False) {
            if ((n->kind == K::True) != truth) { return false; }
        } else if (n->kind == K::Atom) {
            auto inserted = literals.emplace(n->left, truth);
            if (!inserted.second && inserted.first->second != truth) { return false; }
        } else if (n->kind == K::Not) {
            todo.push_back({n->left, !truth});
        } else if ((n->kind == K::And && truth) || (n->kind == K::Or && !truth)) {
            todo.push_back({n->left, truth}); todo.push_back({n->right, truth});
        } else if (n->kind == K::Choose) {
            const auto excluded = truth ? ObligationConditions::no : ObligationConditions::yes;
            if (n->left == excluded) { todo.push_back({n->guard, false}); todo.push_back({n->right, truth}); }
            if (n->right == excluded) { todo.push_back({n->guard, true}); todo.push_back({n->left, truth}); }
        }
    }
    return true;
}
FactoredUseNode::Hazard consequenceHazard(OriginalObligationKey::Kind kind)
{
    return kind == OriginalObligationKey::Kind::RAW ? FactoredUseNode::Hazard::RAW :
           kind == OriginalObligationKey::Kind::WAR ? FactoredUseNode::Hazard::WAR : FactoredUseNode::Hazard::WAW;
}
}
const OriginalConsequenceResult& OriginalPreparation::consequences(
    OriginalObligationId due, const OriginalInterval& interval) const
{
    ++work.consequenceQueries;
    // Invalid identities must never hit an entry cached for this analysis.
    static const OriginalConsequenceResult invalid = [] {
        OriginalConsequenceResult r; r.obstructions.push_back("stale or invalid original consequence query"); return r;
    }();
    const auto& model = analysis.obligations();
    const auto* demand = model.get(due.family);
    if (!formed() || !demand || interval.query.version != version) { return invalid; }
    const ConsequenceKey key{due.family.index, due.origin, interval};
    const auto found = consequenceCache.find(key);
    if (found != consequenceCache.end()) { return found->second; }
    ++work.consequenceComputations;
    OriginalConsequenceResult result;
    auto finish = [&]() -> const OriginalConsequenceResult& {
        return consequenceCache.emplace(key, std::move(result)).first->second;
    };
    const auto a = due.origin.operation, b = demand->key.consumerOperation;
    auto qualifies = [&](const OriginalObligationFamily& family, std::size_t source) {
        if (family.key.kind == OriginalObligationKey::Kind::Typed || !family.expression ||
            family.expression->frame.kind != FactoredUseFrame::Kind::Invocation ||
            family.representation != OriginalObligationFamily::Representation::Factored) { return false; }
        return analysis.occurrences().fixedVisit(source, family.key.consumerOperation, family.key.cell,
                                                 consequenceHazard(family.key.kind)).exact;
    };
    const auto membership = model.membership(due.family, due.origin);
    OriginalIntervalRequest expected;
    expected.version = version;
    expected.selector.cell = demand->key.cell;
    expected.selector.read = demand->key.kind == OriginalObligationKey::Kind::RAW;
    expected.selector.write = !expected.selector.read;
    expected.start = {a, OriginalCut::After};
    expected.stop = {b, OriginalCut::Before};
    expected.occurrence.source = a;
    expected.occurrence.target = b;
    expected.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::FirstReach;
    auto prepared = analysis.prepareInterval(interval.query);
    if (!(interval.query == expected) || !prepared.valid || prepared.interval != interval || due.origin.incoming ||
        interval.owner != NoControlId || interval.query.occurrence.stopVisit != OriginalOccurrenceContext::StopVisit::FirstReach ||
        interval.query.occurrence.source != a || interval.query.occurrence.target != b ||
        interval.query.start != OriginalCut{a, OriginalCut::After} ||
        interval.query.stop != OriginalCut{b, OriginalCut::Before} ||
        interval.query.selector.cell != demand->key.cell ||
        membership.status != ObligationMembership::Status::Guarded || !qualifies(*demand, a)) {
        result.obstructions.push_back("two-link query requires a qualified invocation D1 due relation and its exact deadline interval; recurring middle matching is unresolved");
        return finish();
    }
    struct Candidate { OriginalObligationId first, second; std::size_t middle, p, q; };
    std::vector<Candidate> candidates;
    const auto& original = analysis.structure();
    for (auto secondId : atDeadline(demand->key.consumerOriginal)) {
        const auto* second = model.get(secondId);
        if (!second || second->key.consumerOperation != b || second->key.kind == OriginalObligationKey::Kind::Typed) { continue; }
        const auto origins = model.origins(secondId); // Explicit output-sensitive query, never formation.
        for (const auto& incoming : origins.members) {
            const auto z = incoming.source.operation;
            if (incoming.source.incoming || z == a || z == b || z >= original.operations.size() ||
                incoming.status != ObligationMembership::Status::Guarded || !qualifies(*second, z)) { continue; }
            for (auto firstId : atDeadline(original.operations[z].original)) {
                const auto* first = model.get(firstId);
                // The same translated phase supplies I_z -> C_z. Do not join
                // two arbitrary phases of a multi-phase original instruction.
                if (!first || first->key.consumerOperation != z || !qualifies(*first, a)) { continue; }
                const auto outgoing = model.membership(firstId, due.origin);
                if (outgoing.status != ObligationMembership::Status::Guarded) { continue; }
                candidates.push_back({outgoing.id(), incoming.id(), z, outgoing.condition, incoming.condition});
            }
        }
    }
    // Copy only when queried, after membership has formed its condition nodes.
    // A conjunction is a conditional opportunity, never unconditional credit.
    result.predicates = model.predicates();
    for (const auto& candidate : candidates) {
        if (!compatibleConditions(result.predicates, {membership.condition, candidate.p, candidate.q})) { continue; }
        const auto condition = result.predicates.both(membership.condition,
            result.predicates.both(candidate.p, candidate.q));
        if (condition == ObligationConditions::no) { continue; }
        result.opportunities.push_back({due, candidate.first, candidate.second, candidate.middle,
                                       {candidate.middle, OriginalCut::After}, condition});
    }
    return finish();
}
} // namespace mlir::pto::frontiersynch
