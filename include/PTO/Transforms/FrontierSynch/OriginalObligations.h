// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALOBLIGATIONS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALOBLIGATIONS_H

#include "PTO/Transforms/FrontierSynch/FactoredUse.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::pto::frontiersynch {

inline constexpr std::size_t NoObligationIndex = std::numeric_limits<std::size_t>::max();

// Family roots are stable for an immutable original-program analysis, not across rebuilds.
// The universe stamp rejects a handle borrowed from another ProgramAnalysis.
// Neither a request, a descriptor, nor an emitted endpoint allocates these IDs.
struct OriginalObligationFamilyId {
    std::shared_ptr<const unsigned char> universe;
    std::size_t index = NoObligationIndex;
    bool operator==(const OriginalObligationFamilyId& other) const
    {
        return universe == other.universe && index == other.index;
    }
    bool operator!=(const OriginalObligationFamilyId& other) const { return !(*this == other); }
};

struct ObligationOrigin {
    // Incoming denotes the explicit, possibly unresolved entry-interface family,
    // not a fictitious operation that has already completed. Its role is in Key.
    std::size_t operation = NoObligationIndex;
    bool incoming = false;
    static ObligationOrigin entry() { return {NoObligationIndex, true}; }
    bool operator==(const ObligationOrigin& other) const
    {
        return operation == other.operation && incoming == other.incoming;
    }
    bool operator<(const ObligationOrigin& other) const
    {
        return std::tie(incoming, operation) < std::tie(other.incoming, other.operation);
    }
};

// A functional ID for one original source-use/consumer relation. The frozen
// family supplies cell, hazard, effects, guards, occurrence scope and deadline;
// origin identifies its source role (or its explicit incoming interface).
// Constructing this value allocates no record. In particular, independent
// reader origins remain separate conjuncts/progress units, even if they share
// the same demand-expression root. Groups/descriptors cannot manufacture IDs.
struct OriginalObligationId {
    OriginalObligationFamilyId family;
    ObligationOrigin origin;
    bool operator==(const OriginalObligationId& other) const
    {
        return family == other.family && origin == other.origin;
    }
    bool operator!=(const OriginalObligationId& other) const { return !(*this == other); }
};

struct ObligationCut {
    enum class Kind { OwnerEntry, OwnerExit, PayloadBefore, PayloadAfter, OriginalBefore, OriginalAfter };
    Kind kind = Kind::OwnerEntry;
    std::size_t site = NoObligationIndex;
    bool operator<(const ObligationCut& other) const { return std::tie(kind, site) < std::tie(other.kind, other.site); }
    bool operator==(const ObligationCut& other) const { return kind == other.kind && site == other.site; }
};

struct OriginalObligationKey {
    enum class Kind { RAW, WAR, WAW, Typed } kind = Kind::RAW;
    enum class Role { Writer, Reader, ValueProducer, ValueConsumer };
    enum class Scope { WholeOperation } scope = Scope::WholeOperation;
    // This is a relation over original visits, NOT a certified endpoint pairing.
    enum class Occurrences { FixedUseProjection, OriginalControlPaths } occurrences = Occurrences::OriginalControlPaths;
    OriginalProgramVersion originalVersion;
    std::size_t cell = NoObligationIndex;
    std::size_t consumerOriginal = NoObligationIndex, consumerOperation = NoObligationIndex;
    std::size_t owner = NoObligationIndex, occurrenceInterpretation = 0;
    ObligationCut start, stop;
    bool includeStop = false;
    std::size_t typedValue = NoObligationIndex, typedCause = NoObligationIndex;
    Role sourceRole() const
    {
        return kind == Kind::Typed ? Role::ValueProducer : kind == Kind::WAR ? Role::Reader : Role::Writer;
    }
    Role consumerRole() const
    {
        return kind == Kind::Typed ? Role::ValueConsumer : kind == Kind::RAW ? Role::Reader : Role::Writer;
    }
    auto fields() const
    {
        return std::tie(
            originalVersion, cell, kind, consumerOriginal, consumerOperation, owner, occurrenceInterpretation,
            occurrences, start, stop, includeStop, scope, typedValue, typedCause);
    }
    bool operator<(const OriginalObligationKey& other) const { return fields() < other.fields(); }
};

// A compact Boolean DAG over identities of original condition values. Conditions
// here describe applicability. They DO NOT certify observability at an endpoint.
class ObligationConditions {
public:
    using Id = std::size_t;
    struct Node {
        enum class Kind { False, True, Atom, Not, And, Or, Choose } kind;
        Id left = 0, right = 0, guard = 0;
    };
    using Valuation = std::function<std::optional<bool>(std::size_t)>;
    static constexpr Id no = 0, yes = 1;
    ObligationConditions() : nodes{{Node::Kind::False}, {Node::Kind::True}} {}
    Id atom(std::size_t value) { return intern(Node::Kind::Atom, value, 0); }
    Id negate(Id a)
    {
        if (a == no || a == yes) {
            return a == no ? yes : no;
        }
        if (nodes[a].kind == Node::Kind::Not) {
            return nodes[a].left;
        }
        return intern(Node::Kind::Not, a, 0);
    }
    Id both(Id a, Id b)
    {
        if (a == no || b == no) {
            return no;
        }
        if (a == yes) {
            return b;
        }
        if (b == yes || a == b) {
            return a;
        }
        if (opposites(a, b)) {
            return no;
        }
        if (b < a) {
            std::swap(a, b);
        }
        return intern(Node::Kind::And, a, b);
    }
    Id either(Id a, Id b)
    {
        if (a == yes || b == yes) {
            return yes;
        }
        if (a == no) {
            return b;
        }
        if (b == no || a == b) {
            return a;
        }
        if (opposites(a, b)) {
            return yes;
        }
        if (b < a) {
            std::swap(a, b);
        }
        return intern(Node::Kind::Or, a, b);
    }
    Id choose(Id guard, Id a, Id b)
    {
        if (a == b) {
            return a;
        }
        if (guard == yes || guard == no) {
            return guard == yes ? a : b;
        }
        return intern(Node::Kind::Choose, a, b, guard);
    }
    const Node* get(Id id) const { return id < nodes.size() ? &nodes[id] : nullptr; }
    std::size_t size() const { return nodes.size(); }
    std::optional<bool> evaluate(Id root, const Valuation& values) const
    {
        if (root >= nodes.size()) {
            return std::nullopt;
        }
        // Evaluate only the reachable DAG, iteratively, including deep reader chains.
        std::map<Id, std::optional<bool>> memo;
        std::vector<std::pair<Id, bool>> todo{{root, false}};
        while (!todo.empty()) {
            auto [id, finish] = todo.back();
            todo.pop_back();
            if (memo.count(id)) {
                continue;
            }
            const auto n = nodes[id];
            if (n.kind == Node::Kind::False || n.kind == Node::Kind::True) {
                memo[id] = n.kind == Node::Kind::True;
            } else if (n.kind == Node::Kind::Atom) {
                memo[id] = values ? values(n.left) : std::nullopt;
            } else if (n.kind == Node::Kind::Choose) {
                const auto guard = memo.find(n.guard);
                if (guard == memo.end()) {
                    todo.push_back({id, true});
                    todo.push_back({n.guard, false});
                } else if (!guard->second) {
                    memo[id] = std::nullopt;
                } else {
                    const auto arm = *guard->second ? n.left : n.right;
                    const auto answer = memo.find(arm);
                    if (answer == memo.end()) {
                        todo.push_back({id, true});
                        todo.push_back({arm, false});
                    } else {
                        memo[id] = answer->second;
                    }
                }
            } else if (!finish) {
                todo.push_back({id, true});
                todo.push_back({n.left, false});
                if (n.kind != Node::Kind::Not) {
                    todo.push_back({n.right, false});
                }
            } else if (n.kind == Node::Kind::Not) {
                auto a = memo.at(n.left);
                memo[id] = a ? std::optional<bool>(!*a) : std::nullopt;
            } else {
                auto a = memo.at(n.left), b = memo.at(n.right);
                const bool conjunction = n.kind == Node::Kind::And;
                if ((a && *a != conjunction) || (b && *b != conjunction)) {
                    memo[id] = !conjunction;
                } else if (a && b) {
                    memo[id] = conjunction;
                } else {
                    memo[id] = std::nullopt;
                }
            }
        }
        return memo.at(root);
    }

private:
    bool opposites(Id a, Id b) const
    {
        return (nodes[a].kind == Node::Kind::Not && nodes[a].left == b) ||
               (nodes[b].kind == Node::Kind::Not && nodes[b].left == a);
    }
    Id intern(Node::Kind kind, Id a, Id b, Id guard = 0)
    {
        const auto key = std::make_tuple(kind, a, b, guard);
        auto found = unique.find(key);
        if (found != unique.end()) {
            return found->second;
        }
        const auto id = nodes.size();
        nodes.push_back({kind, a, b, guard});
        unique.emplace(key, id);
        return id;
    }
    std::vector<Node> nodes;
    std::map<std::tuple<Node::Kind, Id, Id, Id>, Id> unique;
};

struct OriginalObligationFamily {
    OriginalObligationFamilyId id;
    OriginalObligationKey key;
    enum class Representation { Factored, Marginal, Typed } representation = Representation::Marginal;
    std::size_t applicability = ObligationConditions::yes;
    // Borrowed immutable nodes owned by OriginalLifetimes. Demand nodes reference
    // the old incoming source expression, so a subsequent strong update cannot
    // erase a previously registered obligation. For Marginal families this may
    // retain a local projection for separate queries; it does not define global
    // membership or replace cross-visit obligations.
    const FactoredUseResult* expression = nullptr;
    std::size_t demandNode = NoObligationIndex, sources = 0;
    // Optional additional reader interface. The integrated adapter uses the
    // actual priorReaders source expression, including its incoming root.
    std::size_t incomingReaderSurvival = 0;
    std::vector<ObligationOrigin> typedSources;
    const std::vector<std::size_t>* consumerEffects = nullptr;
    bool guardsIdentified = false;
    std::string unresolved;
};

struct ObligationMembership {
    enum class Status { Invalid, Excluded, Guarded, Conservative } status = Status::Invalid;
    OriginalObligationFamilyId obligation;
    ObligationOrigin source;
    std::size_t condition = ObligationConditions::no;
    std::string reason;
    OriginalObligationId id() const { return {obligation, source}; }
};
struct ObligationOriginEnumeration {
    // Complete population of represented (possibly conservative) sources, not
    // proof that every returned source occurs on one execution.
    bool complete = false;
    std::vector<ObligationMembership> members;
    std::string reason;
};
struct OriginalObligationStats {
    std::size_t families = 0, factoredFamilies = 0, marginalFamilies = 0, typedFamilies = 0;
    std::size_t membershipQueries = 0, originNodesInspected = 0;
    std::size_t enumeratedOrigins = 0, witnessesRequested = 0;
};

// The callback boundary is the existing marginal analysis, not another solver.
// It receives the complete immutable family key. An unanswerable membership is
// nullopt, never false; positive marginals never acquire guarded exactness.
class OriginalObligations {
public:
    using GuardIdentity = std::function<std::size_t(FactoredGuardIdentity)>;
    using MarginalMember = std::function<std::optional<bool>(const OriginalObligationKey&, ObligationOrigin)>;
    using MarginalOrigins = std::function<std::optional<std::vector<ObligationOrigin>>(const OriginalObligationKey&)>;
    OriginalObligations(
        GuardIdentity guards = {}, MarginalMember member = {}, MarginalOrigins origins = {},
        std::function<bool()> current = {})
        : guardIdentity(std::move(guards)),
          marginalMember(std::move(member)),
          marginalOrigins(std::move(origins)),
          current(std::move(current))
    {}
    OriginalObligations(const OriginalObligations&) = delete;
    OriginalObligations& operator=(const OriginalObligations&) = delete;

    // Formation-only API. freeze() closes the universe before construction.
    ObligationConditions& formingConditions() { return conditions; }
    OriginalObligationFamilyId add(OriginalObligationFamily family)
    {
        if (frozen) {
            return {};
        }
        if (!conditions.get(family.applicability) || !wellFormed(family)) {
            valid = false;
            return {};
        }
        auto found = byKey.find(family.key);
        if (found != byKey.end()) {
            const auto& old = families[found->second];
            // A duplicate registration must be the same original relation, not a
            // second realization or a conflicting answer hidden behind the same ID.
            if (old.representation != family.representation || old.expression != family.expression ||
                old.sources != family.sources || old.applicability != family.applicability ||
                old.incomingReaderSurvival != family.incomingReaderSurvival ||
                old.typedSources != family.typedSources || old.consumerEffects != family.consumerEffects ||
                old.demandNode != family.demandNode || old.guardsIdentified != family.guardsIdentified ||
                old.unresolved != family.unresolved) {
                valid = false;
                return {};
            }
            return old.id;
        }
        const auto index = families.size();
        family.id = {universe, index};
        byKey.emplace(family.key, index);
        byDeadline[family.key.consumerOriginal].push_back(family.id);
        byOperation[family.key.consumerOperation].push_back(family.id);
        ++work.families;
        if (family.representation == OriginalObligationFamily::Representation::Factored) {
            ++work.factoredFamilies;
        }
        if (family.representation == OriginalObligationFamily::Representation::Marginal) {
            ++work.marginalFamilies;
        }
        if (family.representation == OriginalObligationFamily::Representation::Typed) {
            ++work.typedFamilies;
        }
        families.push_back(std::move(family));
        return families.back().id;
    }
    std::size_t importCondition(const FactoredUseResult& expression, std::size_t root)
    {
        return conditionExpression(expression, root);
    }
    void freeze() { frozen = true; }
    bool complete() const { return frozen && valid && (!current || current()); }
    const OriginalObligationFamily* get(OriginalObligationFamilyId id) const
    {
        return (!current || current()) && id.universe == universe && id.index < families.size() ? &families[id.index] :
                                                                                                  nullptr;
    }
    const std::vector<OriginalObligationFamilyId>& atOriginalSite(std::size_t site) const
    {
        static const std::vector<OriginalObligationFamilyId> empty;
        const auto it = byDeadline.find(site);
        return (current && !current()) || it == byDeadline.end() ? empty : it->second;
    }
    const std::vector<OriginalObligationFamilyId>& atOperation(std::size_t operation) const
    {
        static const std::vector<OriginalObligationFamilyId> empty;
        const auto it = byOperation.find(operation);
        return (current && !current()) || it == byOperation.end() ? empty : it->second;
    }
    const OriginalObligationStats& stats() const { return work; }
    const ObligationConditions& predicates() const { return conditions; }

    ObligationMembership membership(OriginalObligationId id) const { return membership(id.family, id.origin); }
    ObligationMembership membership(OriginalObligationFamilyId id, ObligationOrigin source) const
    {
        ++work.membershipQueries;
        ObligationMembership answer;
        answer.obligation = id;
        answer.source = source;
        const auto* f = get(id);
        if (!f || !complete() || (source.incoming && source.operation != NoObligationIndex)) {
            answer.reason = "invalid or unfrozen original obligation";
            return answer;
        }
        auto predicate = ObligationConditions::no;
        bool conservative = !f->guardsIdentified || source.incoming;
        if (f->representation == OriginalObligationFamily::Representation::Factored) {
            if (!f->expression || !f->expression->complete) {
                answer.status = ObligationMembership::Status::Conservative;
                answer.condition = f->applicability;
                answer.reason = "missing factored original-use expression";
                return answer;
            }
            predicate = memberExpression(*f->expression, f->sources, source);
            if (source.incoming && f->incomingReaderSurvival) {
                predicate =
                    conditions.either(predicate, memberExpression(*f->expression, f->incomingReaderSurvival, source));
            }
        } else if (f->representation == OriginalObligationFamily::Representation::Typed) {
            if (std::find(f->typedSources.begin(), f->typedSources.end(), source) != f->typedSources.end()) {
                predicate = ObligationConditions::yes;
            }
            conservative = true; // Original-value and dynamic occurrence qualification is separate.
        } else {
            const auto may = marginalMember ? marginalMember(f->key, source) : std::nullopt;
            predicate = !may || *may ? ObligationConditions::yes : ObligationConditions::no;
            conservative = true;
        }
        answer.condition = conditions.choose(f->applicability, predicate, ObligationConditions::no);
        answer.status = answer.condition == ObligationConditions::no ? ObligationMembership::Status::Excluded :
                        conservative                                 ? ObligationMembership::Status::Conservative :
                                                                       ObligationMembership::Status::Guarded;
        answer.reason = f->unresolved;
        return answer;
    }

    // Output-sensitive enumeration is explicit. Neither add() nor membership()
    // enumerates this population or distributes independent choices into cases.
    ObligationOriginEnumeration origins(OriginalObligationFamilyId id) const
    {
        ObligationOriginEnumeration enumeration;
        auto& out = enumeration.members;
        const auto* f = get(id);
        if (!f || !complete()) {
            enumeration.reason = "invalid or unfrozen original obligation";
            return enumeration;
        }
        std::vector<ObligationOrigin> candidates;
        if (f->representation == OriginalObligationFamily::Representation::Factored && f->expression) {
            std::vector<std::size_t> pending{f->sources};
            if (f->incomingReaderSurvival) {
                candidates.push_back(ObligationOrigin::entry());
            }
            std::vector<bool> seen(f->expression->nodes().size());
            while (!pending.empty()) {
                const auto n = pending.back();
                pending.pop_back();
                if (n >= seen.size() || seen[n]) {
                    continue;
                }
                seen[n] = true;
                ++work.originNodesInspected;
                const auto& node = f->expression->nodes()[n];
                if (node.kind == FactoredUseNode::Kind::Access) {
                    candidates.push_back({node.operation, false});
                }
                if (node.kind == FactoredUseNode::Kind::Incoming) {
                    candidates.push_back(ObligationOrigin::entry());
                }
                if (node.kind == FactoredUseNode::Kind::Both || node.kind == FactoredUseNode::Kind::Choose) {
                    pending.push_back(node.left);
                    pending.push_back(node.right);
                }
            }
        } else if (f->representation == OriginalObligationFamily::Representation::Typed) {
            candidates = f->typedSources;
        } else if (marginalOrigins) {
            const auto population = marginalOrigins(f->key);
            if (!population) {
                enumeration.reason = "marginal origin population is unresolved";
                return enumeration;
            }
            candidates = *population;
        } else {
            enumeration.reason = "marginal origin enumeration is unavailable";
            return enumeration;
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        for (auto source : candidates) {
            auto answer = membership(id, source);
            if (answer.status != ObligationMembership::Status::Excluded) {
                out.push_back(std::move(answer));
            }
        }
        work.enumeratedOrigins += out.size();
        enumeration.complete = true;
        return enumeration;
    }

    // This yields a guarded semantic/effect witness. It is not a concrete feasible
    // trace, an executable guard certificate, or selected completion evidence.
    ObligationMembership witness(OriginalObligationId id) const { return witness(id.family, id.origin); }
    ObligationMembership witness(OriginalObligationFamilyId id, ObligationOrigin source) const
    {
        ++work.witnessesRequested;
        return membership(id, source);
    }

private:
    bool wellFormed(const OriginalObligationFamily& f)
    {
        if (f.representation != OriginalObligationFamily::Representation::Factored) {
            return true;
        }
        if (!f.expression || !f.expression->complete || f.expression->cell != f.key.cell) {
            return false;
        }
        const auto& e = *f.expression;
        const auto isOrigin = [&](std::size_t n) {
            return n < e.nodes().size() && e.nodes()[n].sort == FactoredUseNode::Sort::Origins;
        };
        if (!isOrigin(f.sources) || !isOrigin(f.incomingReaderSurvival)) {
            return false;
        }
        if (f.demandNode != NoObligationIndex) {
            if (f.demandNode >= e.nodes().size()) {
                return false;
            }
            const auto& d = e.nodes()[f.demandNode];
            const auto expected = f.key.kind == OriginalObligationKey::Kind::RAW ? FactoredUseNode::Hazard::RAW :
                                  f.key.kind == OriginalObligationKey::Kind::WAR ? FactoredUseNode::Hazard::WAR :
                                                                                   FactoredUseNode::Hazard::WAW;
            if (d.kind != FactoredUseNode::Kind::Demand || d.operation != f.key.consumerOperation ||
                d.left != f.sources || d.hazard != expected) {
                return false;
            }
        }
        const auto checked = checkedExpressions.find(&e);
        if (checked != checkedExpressions.end()) {
            return checked->second;
        }
        bool ok = !e.nodes().empty() && e.nodes()[0].kind == FactoredUseNode::Kind::Empty;
        for (std::size_t i = 1; i < e.nodes().size() && ok; ++i) {
            const auto& n = e.nodes()[i];
            if (n.kind == FactoredUseNode::Kind::Both || n.kind == FactoredUseNode::Kind::Choose) {
                ok = n.left < i && n.right < i;
                if (ok) {
                    ok = (!n.left || e.nodes()[n.left].sort == n.sort) &&
                         (!n.right || e.nodes()[n.right].sort == n.sort);
                }
            } else if (n.kind == FactoredUseNode::Kind::Demand) {
                ok = n.left < i && isOrigin(n.left) && n.sort == FactoredUseNode::Sort::Demands;
            } else if (n.kind == FactoredUseNode::Kind::True || n.kind == FactoredUseNode::Kind::Test) {
                ok = n.sort == FactoredUseNode::Sort::Condition;
            } else {
                ok = n.sort == FactoredUseNode::Sort::Origins;
            }
            if (ok && n.kind == FactoredUseNode::Kind::Choose) {
                ok = n.condition < i && e.nodes()[n.condition].sort == FactoredUseNode::Sort::Condition;
            }
        }
        checkedExpressions.emplace(&e, ok);
        return ok;
    }
    std::size_t conditionExpression(const FactoredUseResult& expression, std::size_t root) const
    {
        auto& mapped = conditionCache[expression.arena.get()];
        const auto& nodes = expression.nodes();
        // Original condition expressions are topological. Share the translation of
        // every reachable condition handle across all obligation memberships.
        while (mapped.size() <= root && mapped.size() < nodes.size()) {
            const auto id = mapped.size();
            const auto& node = nodes[id];
            auto predicate = ObligationConditions::no;
            if (node.kind == FactoredUseNode::Kind::True) {
                predicate = ObligationConditions::yes;
            } else if (node.kind == FactoredUseNode::Kind::Test) {
                auto inserted = guardIds.emplace(node.guard, guardIds.size());
                predicate = conditions.atom(guardIdentity ? guardIdentity(node.guard) : inserted.first->second);
            } else if (node.kind == FactoredUseNode::Kind::Choose && node.sort == FactoredUseNode::Sort::Condition) {
                predicate = conditions.choose(mapped.at(node.condition), mapped.at(node.left), mapped.at(node.right));
            }
            mapped.push_back(predicate);
        }
        return mapped.at(root);
    }
    std::size_t memberExpression(const FactoredUseResult& expression, std::size_t root, ObligationOrigin origin) const
    {
        using K = FactoredUseNode::Kind;
        const auto keyFor = [&](std::size_t node) { return std::make_tuple(&expression, node, origin); };
        std::vector<std::pair<std::size_t, bool>> pending{{root, false}};
        while (!pending.empty()) {
            const auto [id, finish] = pending.back();
            pending.pop_back();
            const auto key = keyFor(id);
            if (memberCache.count(key)) {
                continue;
            }
            const auto& n = expression.nodes()[id];
            ++work.originNodesInspected;
            const bool combine = n.kind == K::Both || n.kind == K::Choose;
            if (combine && !finish) {
                // wellFormed() checked the immutable DAG once during formation.
                pending.push_back({id, true});
                pending.push_back({n.left, false});
                pending.push_back({n.right, false});
                continue;
            }
            auto result = ObligationConditions::no;
            if (n.kind == K::Incoming) {
                result = origin.incoming ? ObligationConditions::yes : ObligationConditions::no;
            }
            if (n.kind == K::Access) {
                result = !origin.incoming && n.operation == origin.operation ? ObligationConditions::yes :
                                                                               ObligationConditions::no;
            }
            if (combine) {
                const auto a = memberCache.at(keyFor(n.left)), b = memberCache.at(keyFor(n.right));
                result = n.kind == K::Both ? conditions.either(a, b) :
                                             conditions.choose(conditionExpression(expression, n.condition), a, b);
            }
            memberCache.emplace(key, result);
        }
        return memberCache.at(keyFor(root));
    }
    std::shared_ptr<const unsigned char> universe = std::make_shared<const unsigned char>(0);
    GuardIdentity guardIdentity;
    MarginalMember marginalMember;
    MarginalOrigins marginalOrigins;
    std::function<bool()> current;
    bool frozen = false, valid = true;
    std::map<const FactoredUseResult*, bool> checkedExpressions;
    std::vector<OriginalObligationFamily> families;
    std::map<OriginalObligationKey, std::size_t> byKey;
    std::map<std::size_t, std::vector<OriginalObligationFamilyId>> byDeadline, byOperation;
    mutable ObligationConditions conditions;
    mutable std::map<FactoredGuardIdentity, std::size_t> guardIds;
    mutable std::map<const FactoredUseArena*, std::vector<std::size_t>> conditionCache;
    mutable std::map<std::tuple<const FactoredUseResult*, std::size_t, ObligationOrigin>, std::size_t> memberCache;
    mutable OriginalObligationStats work;
};

} // namespace mlir::pto::frontiersynch
#endif
