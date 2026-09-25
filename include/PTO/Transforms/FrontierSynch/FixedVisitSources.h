// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FIXEDVISITSOURCES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FIXEDVISITSOURCES_H

#include "PTO/Transforms/FrontierSynch/FactoredUse.h"
#include <algorithm>
#include <map>
#include <set>

namespace mlir::pto::frontiersynch {

// D1 is a query of the OLD source expression of a demand, not a nearest-access
// scan. These conditions belong to this result's arena, not the provenance or
// reader-frontier arenas. No obligation IDs, selected endpoints, or credit are
// created. Materializing origins is explicit and charged separately from formation.
struct FixedVisitSource {
    std::size_t operation = NoFactoredId;
    bool incoming = false;
    std::size_t condition = 0, sourceCondition = 0, targetCondition = 0;
    std::shared_ptr<const FactoredUseAccess> witness;
};
struct FixedVisitSources {
    bool complete = false;
    bool exclusiveWriters = false;
    bool independentReaders = false;
    FactoredUseFrame frame;
    FactoredUseNode::Hazard hazard = FactoredUseNode::Hazard::RAW;
    std::size_t target = NoFactoredId, applicability = 0, noSource = 0;
    std::shared_ptr<const FactoredUseArena> predicates;
    std::vector<FixedVisitSource> alternatives;
    std::string reason;
    std::size_t inspectedNodes = 0;
};

// Constant-folding and sharing of conditional expressions. This deliberately is
// not Boolean minimization, a BDD construction, or a valuation enumerator. The
// cofactor at one endpoint uses only its entailed original lexical conditions.
class FixedVisitPredicates {
public:
    using Id = std::size_t;
    using Node = FactoredUseNode;
    using Facts = std::map<FactoredGuardIdentity, bool>;
    explicit FixedVisitPredicates(FactoredUseFrame frame) : arena(std::make_shared<FactoredUseArena>(frame)) {}
    std::shared_ptr<FactoredUseArena> arena;
    Id choose(Id guard, Id yes, Id no)
    {
        if (guard == 0 || guard == 1 || yes == no) {
            return yes == no ? yes : guard ? yes : no;
        }
        // The repeated test is the same defining value AND occurrence scope.
        // (g ? (g ? a : b) : c) = (g ? a : c), and its dual.
        auto y = (*arena)[yes], n = (*arena)[no];
        if (y.kind == Node::Kind::Choose && y.condition == guard) {
            yes = y.left;
        }
        if (n.kind == Node::Kind::Choose && n.condition == guard) {
            no = n.right;
        }
        if (yes == no) {
            return yes;
        }
        if (yes == 1 && no == 0) {
            return guard;
        }
        const auto key = std::make_tuple(guard, yes, no);
        const auto old = choices.find(key);
        if (old != choices.end()) {
            return old->second;
        }
        const auto id = arena->choose(guard, yes, no, Node::Sort::Condition);
        choices.emplace(key, id);
        return id;
    }
    Id both(Id a, Id b) { return choose(a, b, 0); }
    Id either(Id a, Id b) { return choose(a, 1, b); }
    Id negate(Id a) { return choose(a, 0, 1); }

    // Every referenced node precedes its parent in the arena. Use iterative
    // walks so thousands of independent optional readers do not exhaust the stack.
    std::vector<Id> reachable(Id root) const
    {
        std::set<Id> seen;
        std::vector<Id> todo{root};
        while (!todo.empty()) {
            const auto id = todo.back();
            todo.pop_back();
            if (!seen.insert(id).second) {
                continue;
            }
            const auto& n = (*arena)[id];
            if (n.kind == Node::Kind::Choose) {
                todo.push_back(n.condition);
                todo.push_back(n.left);
                todo.push_back(n.right);
            }
        }
        return {seen.begin(), seen.end()};
    }
    Id restrictTo(Id root, const Facts& facts)
    {
        std::map<Id, Id> mapped;
        for (auto id : reachable(root)) {
            const auto n = (*arena)[id];
            auto result = id;
            if (n.kind == Node::Kind::Test) {
                const auto fact = facts.find(n.guard);
                if (fact != facts.end()) {
                    result = fact->second ? 1 : 0;
                }
            } else if (n.kind == Node::Kind::Choose) {
                result = choose(mapped.at(n.condition), mapped.at(n.left), mapped.at(n.right));
            }
            mapped.emplace(id, result);
        }
        return mapped.at(root);
    }
    // Extract only consequences of a condition's truth, never assume a branch
    // of a genuine disjunction. Fixed-use site activation is a lexical conjunction.
    bool entailed(Id root, Facts& facts) const
    {
        std::vector<std::pair<Id, bool>> todo{{root, true}};
        std::set<std::pair<Id, bool>> seen;
        while (!todo.empty()) {
            const auto item = todo.back();
            todo.pop_back();
            if (!seen.insert(item).second) {
                continue;
            }
            const auto [id, value] = item;
            if (id <= 1) {
                if (bool(id) != value) {
                    return false;
                }
                continue;
            }
            const auto& n = (*arena)[id];
            if (n.kind == Node::Kind::Test) {
                auto entry = facts.emplace(n.guard, value);
                if (!entry.second && entry.first->second != value) {
                    return false;
                }
            } else if (n.kind == Node::Kind::Choose) {
                if (n.right == (value ? 0u : 1u)) {
                    todo.push_back({n.condition, true});
                    todo.push_back({n.left, value});
                } else if (n.left == (value ? 0u : 1u)) {
                    todo.push_back({n.condition, false});
                    todo.push_back({n.right, value});
                }
            }
        }
        return true;
    }

private:
    std::map<std::tuple<Id, Id, Id>, Id> choices;
};

inline FixedVisitSources queryFixedVisitSources(
    const FactoredUseResult& uses, std::size_t target, FactoredUseNode::Hazard hazard)
{
    using Node = FactoredUseNode;
    using Id = std::size_t;
    FixedVisitSources result;
    result.frame = uses.frame;
    result.target = target;
    result.hazard = hazard;
    result.independentReaders = hazard == Node::Hazard::WAR;
    const auto demand = uses.requirement(target, hazard);
    if (!demand.valid || !uses.arena || demand.hasUnresolved()) {
        result.reason = "D1 requires a qualified fixed-use demand expression";
        return result;
    }
    FixedVisitPredicates predicates(uses.frame);
    result.predicates = predicates.arena;
    const auto& input = uses.nodes();
    std::vector<Id> conditions(input.size(), NoFactoredId);
    conditions[0] = 0;
    conditions[1] = 1;
    // Import condition references only. The original provenance DAG is immutable.
    for (Id id = 2; id < input.size(); ++id) {
        const auto& n = input[id];
        if (n.sort != Node::Sort::Condition) {
            continue;
        }
        if (n.kind == Node::Kind::Test) {
            conditions[id] = predicates.arena->test(n.guard);
        } else if (n.kind == Node::Kind::Choose && conditions[n.condition] != NoFactoredId &&
                   conditions[n.left] != NoFactoredId && conditions[n.right] != NoFactoredId) {
            conditions[id] = predicates.choose(conditions[n.condition], conditions[n.left], conditions[n.right]);
        }
    }
    result.applicability = conditions.at(demand.applicability);
    if (result.applicability == NoFactoredId) {
        result.reason = "D1 target applicability has an unresolved original condition";
        return result;
    }
    FixedVisitPredicates::Facts targetFacts;
    if (!predicates.entailed(result.applicability, targetFacts)) {
        result.complete = true; // An incompatible ORIGINAL lexical path, not a missing producer.
        result.applicability = 0;
        result.exclusiveWriters = !result.independentReaders;
        return result;
    }
    std::set<Id> reachable;
    std::vector<Id> todo{demand.sources};
    std::map<Id, std::shared_ptr<const FactoredUseAccess>> origins;
    bool incoming = false;
    while (!todo.empty()) {
        const auto id = todo.back();
        todo.pop_back();
        if (!reachable.insert(id).second) {
            continue;
        }
        ++result.inspectedNodes;
        const auto& n = input.at(id);
        if (n.kind == Node::Kind::Access) {
            origins.emplace(n.operation, n.access);
        } else if (n.kind == Node::Kind::Incoming) {
            if (n.boundary != Node::Boundary::Entry || n.owner != uses.frame.owner ||
                n.role != (result.independentReaders ? Node::Role::Reader : Node::Role::Writer)) {
                result.reason = "D1 incoming source needs a qualified matching entry interface";
                return result;
            }
            incoming = true;
        } else if (n.kind == Node::Kind::Both || n.kind == Node::Kind::Choose) {
            todo.push_back(n.left);
            todo.push_back(n.right);
        } else if (n.kind != Node::Kind::Empty) {
            result.reason = "D1 source relation has an unresolved occurrence";
            return result;
        }
    }
    // Count represented sources symbolically, saturated at two. Both is a
    // conjunction of obligations, NOT an exclusive choice. This withholds the
    // single-full-cell-origin certificate after a partial/may overwrite.
    std::map<Id, std::pair<Id, Id>> population;
    for (auto id : reachable) {
        const auto& n = input[id];
        Id any = 0, several = 0;
        if (n.kind == Node::Kind::Access || n.kind == Node::Kind::Incoming) {
            any = 1;
        } else if (n.kind == Node::Kind::Both || n.kind == Node::Kind::Choose) {
            const auto a = population.at(n.left), b = population.at(n.right);
            if (n.kind == Node::Kind::Choose) {
                const auto g = predicates.restrictTo(conditions.at(n.condition), targetFacts);
                any = predicates.choose(g, a.first, b.first);
                several = predicates.choose(g, a.second, b.second);
            } else {
                any = predicates.either(a.first, b.first);
                several = predicates.either(
                    predicates.either(a.second, b.second), predicates.both(a.first, b.first));
            }
        }
        population.emplace(id, std::make_pair(any, several));
    }
    const auto counts = population.at(demand.sources);
    result.noSource = predicates.both(result.applicability, predicates.negate(counts.first));
    result.exclusiveWriters = !result.independentReaders && counts.second == 0;
    if (incoming || (!result.independentReaders && result.noSource != 0)) {
        origins.emplace(NoFactoredId, nullptr);
    }
    // Propagate applicability down the shared source DAG once. Multiple paths
    // to one node are disjoined before visiting it (reverse topological order).
    // Both sends the SAME condition to its two simultaneous requirements;
    // Choose conjoins opposite arms of the SAME original guard. This is not a
    // traversal per origin and never expands combinations of independent readers.
    std::map<Id, Id> active{{demand.sources, result.applicability}}, membership;
    auto add = [&](Id child, Id condition) {
        active[child] = predicates.either(active[child], condition);
    };
    for (auto it = reachable.rbegin(); it != reachable.rend(); ++it) {
        ++result.inspectedNodes;
        const auto id = *it;
        const auto& n = input[id];
        const auto condition = active[id];
        if (n.kind == Node::Kind::Access || n.kind == Node::Kind::Incoming) {
            const auto operation = n.kind == Node::Kind::Access ? n.operation : NoFactoredId;
            membership[operation] = predicates.either(membership[operation], condition);
        } else if (n.kind == Node::Kind::Both) {
            add(n.left, condition);
            add(n.right, condition);
        } else if (n.kind == Node::Kind::Choose) {
            const auto guard = predicates.restrictTo(conditions.at(n.condition), targetFacts);
            add(n.left, predicates.both(condition, guard));
            add(n.right, predicates.both(condition, predicates.negate(guard)));
        }
    }
    for (const auto& [operation, witness] : origins) {
        auto condition = membership[operation];
        if (operation == NoFactoredId && !result.independentReaders) {
            // No producer is a named unbound incoming case, never infeasible.
            condition = predicates.either(condition, result.noSource);
        }
        const auto targetCondition = predicates.restrictTo(condition, targetFacts);
        if (targetCondition == 0) {
            continue;
        }
        FixedVisitPredicates::Facts sourceFacts;
        if (operation != NoFactoredId) {
            const auto* site = uses.site(operation);
            if (!site || !site->visited || site->applicability >= conditions.size() ||
                conditions[site->applicability] == NoFactoredId ||
                !predicates.entailed(conditions[site->applicability], sourceFacts)) {
                result.reason = "D1 source has no compatible visit in the target's original frame";
                return result;
            }
        }
        result.alternatives.push_back(
            {operation, operation == NoFactoredId, condition, predicates.restrictTo(condition, sourceFacts),
             targetCondition, witness});
    }
    result.complete = true;
    if (!result.independentReaders && !result.exclusiveWriters) {
        result.reason = "D1 retains simultaneous origins after a partial or possible overwrite";
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
#endif
