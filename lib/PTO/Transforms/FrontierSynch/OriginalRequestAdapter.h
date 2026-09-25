// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_FRONTIERSYNCH_ORIGINALREQUESTADAPTER_H
#define PTO_FRONTIERSYNCH_ORIGINALREQUESTADAPTER_H

#include "PTO/Transforms/FrontierSynch/ProgramAnalysis.h"
#include <set>

namespace mlir::pto::frontiersynch {
namespace detail {
inline DescriptorPredicate obligationPredicate(const OriginalObligations& model, std::size_t root)
{
    using Domain = DescriptorPredicate::Domain;
    if (root == ObligationConditions::yes) { return {}; }
    if (root == ObligationConditions::no) { return {Domain::False, 0, 0}; }
    return {Domain::Obligation, reinterpret_cast<std::uintptr_t>(&model), root};
}
inline OriginalCut requestCut(ObligationCut cut, std::size_t owner)
{
    using Kind = ObligationCut::Kind;
    if (cut.kind == Kind::PayloadBefore || cut.kind == Kind::PayloadAfter) {
        return {cut.site, cut.kind == Kind::PayloadBefore ? OriginalCut::Before : OriginalCut::After};
    }
    if (cut.kind == Kind::OriginalBefore || cut.kind == Kind::OriginalAfter) {
        return OriginalCut::scope(cut.site, cut.kind == Kind::OriginalBefore ? OriginalCut::Before : OriginalCut::After);
    }
    return OriginalCut::scope(owner, cut.kind == Kind::OwnerEntry ? OriginalCut::Before : OriginalCut::After);
}
// This adapter invokes the existing scalar qualifier, not another scalar
// solver. Conjunction/disjunction need every operand. A nonconstant Choose
// whose arms cannot BOTH be evaluated remains unresolved here; the full shared
// condition is retained for the directional query provider. This conservative
// bridge must not turn unavailable guards into unconditional endpoints.
inline OriginalValueQualification requestObservation(
    const ProgramAnalysis& analysis, DescriptorPredicate predicate, OriginalValueCut cut)
{
    using Status = OriginalValueQualification::Status;
    OriginalValueQualification invalid;
    invalid.obstructions.push_back("request endpoint predicate is not qualified");
    if (!analysis.originalValues().legal(cut)) {
        invalid.status = Status::NotObservableHere;
        invalid.obstructions = {"request endpoint is not a legal original insertion cut"};
        return invalid;
    }
    OriginalValueQualification available;
    available.status = Status::Available;
    if (predicate.domain == DescriptorPredicate::Domain::True ||
        predicate.domain == DescriptorPredicate::Domain::False) { return available; }
    if (predicate.domain != DescriptorPredicate::Domain::Obligation ||
        predicate.arena != reinterpret_cast<std::uintptr_t>(&analysis.obligations())) { return invalid; }
    const auto& predicates = analysis.obligations().predicates();
    std::map<std::size_t, OriginalValueQualification> memo;
    std::vector<std::pair<std::size_t, bool>> todo{{predicate.root, false}};
    while (!todo.empty()) {
        auto [id, finish] = todo.back(); todo.pop_back();
        if (memo.count(id)) { continue; }
        const auto* n = predicates.get(id);
        if (!n) { memo[id] = invalid; continue; }
        using Kind = ObligationConditions::Node::Kind;
        if (n->kind == Kind::False || n->kind == Kind::True) {
            memo[id] = available;
        } else if (n->kind == Kind::Atom) {
            memo[id] = analysis.originalValues().qualify(analysis.obligationGuardValue(n->left), cut);
        } else if (!finish) {
            todo.push_back({id, true}); todo.push_back({n->left, false});
            if (n->kind != Kind::Not) { todo.push_back({n->right, false}); }
            if (n->kind == Kind::Choose) { todo.push_back({n->guard, false}); }
        } else {
            auto q = memo.at(n->left);
            if (n->kind != Kind::Not) { q = OriginalValueQueries::combine(std::move(q), memo.at(n->right)); }
            if (n->kind == Kind::Choose) { q = OriginalValueQueries::combine(std::move(q), memo.at(n->guard)); }
            memo[id] = std::move(q);
        }
    }
    return memo.at(predicate.root);
}
inline DescriptorObservation requestObservationRecord(
    const OriginalValueQualification& q, DescriptorPredicate predicate, std::size_t localIndex)
{
    DescriptorObservation out;
    out.predicate = predicate;
    using Status = OriginalValueQualification::Status;
    out.status = q.status == Status::Available ? DescriptorObservation::Status::Available :
                 q.status == Status::NeedsCompletion ? DescriptorObservation::Status::NeedsCompletion :
                 q.status == Status::NotObservableHere ? DescriptorObservation::Status::NotObservableHere :
                                                         DescriptorObservation::Status::Unresolved;
    out.unresolved = q.obstructions;
    if (!q.prerequisites.empty()) {
        // A local reference is rebound to the owning request repertoire below.
        out.prerequisites.push_back({DescriptorFactRef::Kind::CompletionPrerequisite, 0, localIndex});
    }
    out.independentlyDischargeable = q.status == Status::NeedsCompletion && !q.prerequisites.empty();
    return out;
}
inline OriginalRequestAnswers defaultRequestAnswers(const ProgramAnalysis& analysis, const OriginalRequestReference& ref)
{
    OriginalRequestAnswers answer;
    const auto* family = analysis.obligations().get(ref.family);
    if (!family) { answer.unresolved.push_back("stale original obligation family"); return answer; }
    const auto& key = family->key;
    auto& source = answer.exactSource;
    auto& target = answer.exactTarget;
    source.direction = DescriptorBoundary::Direction::Source;
    target.direction = DescriptorBoundary::Direction::Target;
    OriginalIntervalRequest interval;
    interval.version = key.originalVersion;
    interval.selector.cell = key.cell;
    interval.occurrence.source = ref.origin && !ref.origin->incoming ? ref.origin->operation : NoControlId;
    interval.occurrence.target = key.consumerOperation;
    interval.occurrence.qualification = key.occurrenceInterpretation;
    interval.start = requestCut(key.start, key.owner);
    interval.stop = requestCut(key.stop, key.owner);
    interval.continuationOwner = key.owner;
    interval.includeStoppingAccess = key.includeStop;
    if (key.occurrences == OriginalObligationKey::Occurrences::FixedUseProjection) {
        interval.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::FirstReach;
    }
    const bool physical = key.kind != OriginalObligationKey::Kind::Typed;
    OriginalIntervalResult prepared;
    if (physical) { prepared = analysis.prepareInterval(interval); }
    source.interval = target.interval = prepared.valid ? prepared.interval : OriginalInterval{interval, key.owner};
    source.intervalQualified = target.intervalQualified = prepared.valid;
    // The default only adapts legal immediate fixed-use endpoints already
    // supported in this baseline. D1/D2/D4 and directional-cover gaps are not
    // hidden by the new request API or by inspecting a bank's may-footprint.
    if (physical && ref.origin && !ref.origin->incoming && prepared.valid) {
        const auto hazard = key.kind == OriginalObligationKey::Kind::RAW ? FactoredUseNode::Hazard::RAW :
                            key.kind == OriginalObligationKey::Kind::WAR ? FactoredUseNode::Hazard::WAR :
                                                                          FactoredUseNode::Hazard::WAW;
        answer.fixedVisit = analysis.occurrences().fixedVisit(ref.origin->operation, key.consumerOperation, key.cell, hazard);
        if (!answer.fixedVisit.exact) {
            answer.periodic = analysis.periodicUseFor(*ref.obligation());
        }
    }
    answer.occurrenceQualified = answer.fixedVisit.exact || (answer.periodic && answer.periodic->exact);
    if (answer.periodic && answer.periodic->exact) {
        answer.unresolved.push_back("periodic relation retained; per-bank descriptor endpoint domains are not adapted");
    }
    if (!answer.fixedVisit.exact) {
        answer.unresolved.push_back(answer.fixedVisit.reason.empty() ?
            "original occurrence pairing needs its D1/D2/D4 or typed-interface premise" : answer.fixedVisit.reason);
    }
    if (!prepared.valid) {
        answer.unresolved.push_back(physical ? prepared.reason : "typed endpoint interval qualification retained");
    }
    const auto& original = analysis.structure();
    auto direct = [&](DescriptorBoundary& b, std::size_t operation, bool after, std::size_t observationIndex) {
        if (operation >= original.operations.size()) {
            b.unresolved.push_back("source/target role has no local payload cut");
            return;
        }
        const auto& op = original.operations[operation];
        DescriptorEndpoint endpoint;
        endpoint.cut = {operation, after ? OriginalCut::After : OriginalCut::Before};
        endpoint.condition = ref.applicability;
        endpoint.legal = after ? op.afterExecutable : op.beforeExecutable;
        auto qualification = requestObservation(analysis, ref.applicability, analysis.originalValues().phaseCut(operation, after));
        endpoint.observation = requestObservationRecord(qualification, ref.applicability, observationIndex);
        answer.observations.push_back(std::move(qualification));
        b.endpoints.push_back(std::move(endpoint));
        b.form = DescriptorBoundary::Form::Exact;
        b.referenceCoverage = answer.fixedVisit.exact;
        b.multiplicity = answer.fixedVisit.exact ? DescriptorBoundary::Multiplicity::ExactlyParticipating :
                                                DescriptorBoundary::Multiplicity::Unresolved;
        if (!answer.fixedVisit.exact) { b.unresolved.push_back("direct boundary needs qualified occurrence pairing"); }
        if (!b.intervalQualified) { b.unresolved.push_back("original interval qualification is missing"); }
    };
    if (ref.origin && !ref.origin->incoming) { direct(source, ref.origin->operation, true, answer.observations.size()); }
    else { source.unresolved.push_back("incoming or unexpanded source relation has no qualified publication cut"); }
    direct(target, key.consumerOperation, false, answer.observations.size());
    // Each leaf names one original role. Prepare that role's own singleton
    // interval and selector independently on each side; never substitute the
    // source-to-deadline interval for both families. Broader supplied families
    // use their independently qualified provider answers.
    auto covering = [&](bool sourceSide) {
        auto& boundary = sourceSide ? answer.coveringSource : answer.coveringTarget;
        boundary.direction = sourceSide ? DescriptorBoundary::Direction::Source : DescriptorBoundary::Direction::Target;
        const auto phase = sourceSide && ref.origin && !ref.origin->incoming ? ref.origin->operation :
                           !sourceSide ? key.consumerOperation : NoControlId;
        if (!physical || phase >= original.operations.size() || !original.operations[phase].instruction) {
            boundary.unresolved.push_back("incoming, typed or unexpanded role lacks a qualified directional family interval");
            return;
        }
        OriginalIntervalRequest role;
        role.version = key.originalVersion;
        role.start = {phase, OriginalCut::Before};
        role.stop = {phase, OriginalCut::After};
        role.includeStoppingAccess = true;
        role.selector.cell = key.cell;
        const auto kind = sourceSide ? key.sourceRole() : key.consumerRole();
        role.selector.read = kind == OriginalObligationKey::Role::Reader;
        role.selector.write = !role.selector.read;
        role.selector.engine = unsigned(original.operations[phase].instruction->kPipeValue);
        role.occurrence.stopVisit = OriginalOccurrenceContext::StopVisit::FirstReach;
        const auto preparedRole = analysis.prepareInterval(role);
        if (!preparedRole.valid) { boundary.unresolved.push_back(preparedRole.reason); return; }
        const auto result = sourceSide ? analysis.sourceCovering(preparedRole.interval) :
                                         analysis.targetCovering(preparedRole.interval);
        (sourceSide ? answer.sourceCoveringQuery : answer.targetCoveringQuery) = result;
        boundary.interval = preparedRole.interval;
        boundary.intervalQualified = true;
        boundary.referenceCoverage = result.referenceCoverage;
        boundary.continuationCases = result.cases;
        boundary.mayExecuteWithoutAccess = result.mayExecuteWithoutAccess;
        if (result.status == CoveringBoundary::Status::NoHit) {
            boundary.form = DescriptorBoundary::Form::NoHit;
            return;
        }
        if (result.status != CoveringBoundary::Status::Covering || !result.cut) {
            boundary.unresolved.push_back(result.reason);
            return;
        }
        boundary.form = DescriptorBoundary::Form::Covering;
        boundary.multiplicity = result.exactlyOneVisitPerInterval ? DescriptorBoundary::Multiplicity::OncePerInterval :
                                                                  DescriptorBoundary::Multiplicity::Unresolved;
        auto qualification = result.endpointQualification;
        for (const auto& control : result.controlQualifications) {
            qualification = OriginalValueQueries::combine(std::move(qualification), control.qualification);
        }
        DescriptorEndpoint endpoint;
        endpoint.cut = *result.cut;
        endpoint.legal = bool(resolveOriginalCut(original, endpoint.cut));
        endpoint.observation = requestObservationRecord(qualification, {}, answer.observations.size());
        answer.observations.push_back(std::move(qualification));
        boundary.endpoints.push_back(std::move(endpoint));
        auto work = boundary.interval;
        work.query.start = result.enclosedWork.begin;
        work.query.stop = result.enclosedWork.end;
        boundary.extraWork.push_back(std::move(work));
    };
    covering(true);
    covering(false);
    return answer;
}
} // namespace detail

std::unique_ptr<OriginalRequests> OriginalRequests::build(
    const ProgramAnalysis& analysis, const OriginalRequestBoundaryProvider& provider)
{
    auto out = std::unique_ptr<OriginalRequests>(new OriginalRequests(analysis.obligations()));
    if (!out->model.complete()) { out->error = "original obligations are not frozen"; return out; }
    auto& slots = out->slots;
    using Id = OriginalDescriptorSlots::Id;
    const auto& original = analysis.structure();
    const auto unknownEngine = std::optional<unsigned>{};
    auto engineOf = [&](std::size_t operation) -> std::optional<unsigned> {
        return operation < original.operations.size() && original.operations[operation].instruction ?
                   std::optional<unsigned>(unsigned(original.operations[operation].instruction->kPipeValue)) :
                                                       unknownEngine;
    };
    // Native answer providers are invoked only here. No callback is retained in
    // the returned object, and no selected-state query enters slot formation.
    auto makeLeaf = [&](const OriginalObligationFamily& family, std::optional<ObligationOrigin> origin,
                        std::size_t expression, std::optional<unsigned> engine) -> Id {
        OriginalRequestReference reference;
        reference.family = family.id; reference.origin = origin;
        reference.sourceExpression = expression; reference.sourceEngine = engine;
        auto condition = family.applicability;
        if (origin) {
            const auto member = out->model.membership(family.id, *origin);
            if (member.status == ObligationMembership::Status::Excluded) { return 0; }
            if (member.status == ObligationMembership::Status::Invalid) {
                out->error = "invalid original membership during descriptor preparation";
                return NoControlId;
            }
            condition = member.condition;
        }
        reference.applicability = detail::obligationPredicate(out->model, condition);
        const Id index = out->references.size();
        out->references.push_back(reference);
        auto answer = provider ? provider(analysis, reference) : detail::defaultRequestAnswers(analysis, reference);
        DescriptorLeafFacts facts;
        facts.obligations.push_back(out->ref(DescriptorFactRef::Kind::Obligation, index));
        facts.applicability = reference.applicability;
        facts.sourceEngine = engine.value_or(unsigned(PipelineType::PIPE_UNASSIGNED));
        facts.targetEngine = engineOf(family.key.consumerOperation).value_or(unsigned(PipelineType::PIPE_UNASSIGNED));
        facts.occurrenceMatching = answer.occurrence;
        facts.occurrenceMatching.push_back(out->ref(DescriptorFactRef::Kind::Occurrence, index));
        facts.matchingQualified = answer.occurrenceQualified || answer.fixedVisit.exact;
        facts.support = answer.support;
        facts.unresolved = answer.unresolved;
        const auto base = out->qualifications.size();
        for (const auto& q : answer.observations) {
            facts.observations.push_back(out->ref(DescriptorFactRef::Kind::CompletionPrerequisite, out->qualifications.size()));
            out->qualifications.push_back(q);
        }
        auto adopt = [&](DescriptorBoundary& b) {
            if (b.interval.query.version && b.interval.query.version != family.key.originalVersion) {
                out->error = "boundary answer belongs to a different original-program snapshot";
            }
            b.witnesses.push_back(out->ref(DescriptorFactRef::Kind::BoundaryQuery, index));
            for (auto& endpoint : b.endpoints) {
                for (auto& prerequisite : endpoint.observation.prerequisites) {
                    if (prerequisite.kind == DescriptorFactRef::Kind::CompletionPrerequisite && prerequisite.arena == 0) {
                        if (prerequisite.index >= answer.observations.size()) {
                            out->error = "invalid local endpoint-observation reference";
                        } else {
                            prerequisite = out->ref(DescriptorFactRef::Kind::CompletionPrerequisite, base + prerequisite.index);
                        }
                    }
                }
            }
            return slots.boundary(b);
        };
        const auto es = adopt(answer.exactSource), cs = adopt(answer.coveringSource);
        const auto et = adopt(answer.exactTarget), ct = adopt(answer.coveringTarget);
        out->answers.push_back(std::move(answer));
        return slots.leaf(std::move(facts), {es, cs, et, ct});
    };
    // Families are scanned in original-site order. Marginal populations remain
    // opaque; this does NOT call origins() or materialize compatibility pairs.
    for (std::size_t site = 0; site < original.originalSites.size(); ++site) {
        for (auto familyId : out->model.atOriginalSite(site)) {
            const auto* family = out->model.get(familyId);
            if (!family) { out->error = "stale family during request preparation"; return out; }
            std::map<std::optional<unsigned>, Id> roots;
            if (family->representation == OriginalObligationFamily::Representation::Factored && family->expression) {
                const auto& e = *family->expression;
                // Each original source expression gets its own memo. Sharing is
                // retained through choices/Both; independent engines are split
                // by role, not multiplied into joint history valuations.
                std::map<Id, std::map<std::optional<unsigned>, Id>> mapped;
                std::vector<std::pair<Id, bool>> todo{{family->sources, false}};
                if (family->incomingReaderSurvival) { todo.push_back({family->incomingReaderSurvival, false}); }
                while (!todo.empty()) {
                    auto [id, finish] = todo.back(); todo.pop_back();
                    if (mapped.count(id)) { continue; }
                    if (id >= e.nodes().size()) { out->error = "invalid original source-expression reference"; return out; }
                    const auto n = e.nodes()[id];
                    using Kind = FactoredUseNode::Kind;
                    if (n.kind == Kind::Both || n.kind == Kind::Choose) {
                        if (!finish) {
                            todo.push_back({id, true}); todo.push_back({n.left, false}); todo.push_back({n.right, false});
                            continue;
                        }
                        std::set<std::optional<unsigned>> engines;
                        for (const auto& entry : mapped[n.left]) { engines.insert(entry.first); }
                        for (const auto& entry : mapped[n.right]) { engines.insert(entry.first); }
                        for (auto engine : engines) {
                            auto left = mapped[n.left].find(engine), right = mapped[n.right].find(engine);
                            const Id a = left == mapped[n.left].end() ? 0 : left->second;
                            const Id b = right == mapped[n.right].end() ? 0 : right->second;
                            DescriptorPredicate p{DescriptorPredicate::Domain::FactoredUse,
                                                  reinterpret_cast<std::uintptr_t>(e.arena.get()), n.condition};
                            mapped[id][engine] = n.kind == Kind::Both ? slots.both(a, b) : slots.choose(p, a, b);
                        }
                        // Record even an empty transformed node in the memo.
                        mapped.try_emplace(id);
                    } else if (n.kind == Kind::Empty) {
                        mapped.try_emplace(id);
                    } else {
                        std::optional<ObligationOrigin> origin;
                        auto engine = unknownEngine;
                        if (n.kind == Kind::Access) { origin = ObligationOrigin{n.operation, false}; engine = engineOf(n.operation); }
                        else if (n.kind == Kind::Incoming) { origin = ObligationOrigin::entry(); }
                        mapped[id][engine] = makeLeaf(*family, origin, id, engine);
                    }
                }
                roots = mapped[family->sources];
                if (family->incomingReaderSurvival) {
                    for (const auto& [engine, root] : mapped[family->incomingReaderSurvival]) {
                        roots[engine] = slots.both(roots[engine], root);
                    }
                }
            } else if (family->representation == OriginalObligationFamily::Representation::Typed) {
                for (auto origin : family->typedSources) {
                    auto engine = origin.incoming ? unknownEngine : engineOf(origin.operation);
                    roots[engine] = slots.both(roots[engine], makeLeaf(*family, origin, NoControlId, engine));
                }
            } else {
                roots[unknownEngine] = makeLeaf(*family, {}, family->sources, unknownEngine);
            }
            for (const auto& [engine, root] : roots) {
                if (!root) { continue; } // Only a proved empty original expression.
                OriginalRequestGroup group;
                group.families.push_back(familyId); group.owner = family->key.owner; group.deadline = site;
                group.sourceRole = family->key.sourceRole(); group.targetRole = family->key.consumerRole();
                group.sourceEngine = engine; group.targetEngine = engineOf(family->key.consumerOperation);
                group.sourceExpression = family->sources;
                const auto applicable = slots.choose(detail::obligationPredicate(out->model, family->applicability), root, 0);
                group.declaredSlots = slots.form(applicable);
                for (const auto& slot : group.declaredSlots) {
                    if (slot.fullyDescribed) { group.descriptors.push_back(slot); }
                }
                out->requestGroups.push_back(std::move(group));
            }
        }
    }
    // Also form the empty input when the program has no obligations.
    if (out->requestGroups.empty()) { slots.form(0); }
    std::stable_sort(out->requestGroups.begin(), out->requestGroups.end(), [](const auto& a, const auto& b) {
        return std::tie(a.owner, a.deadline, a.sourceEngine, a.targetEngine, a.sourceRole, a.targetRole, a.families[0].index) <
               std::tie(b.owner, b.deadline, b.sourceEngine, b.targetEngine, b.sourceRole, b.targetRole, b.families[0].index);
    });
    for (std::size_t id = 0; id < out->requestGroups.size(); ++id) {
        auto& group = out->requestGroups[id]; group.id = id;
        out->deadlines[group.deadline].push_back(id);
    }
    if (!slots.freeze()) { out->error = slots.reason(); }
    return out;
}
} // namespace mlir::pto::frontiersynch
#endif
