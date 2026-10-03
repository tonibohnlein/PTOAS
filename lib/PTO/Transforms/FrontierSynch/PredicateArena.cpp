// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/GuardedAnalysis.h"
#include "mlir/IR/Matchers.h"
#include <algorithm>
#include <cassert>
namespace mlir::pto::frontiersynch {
PredicateArena::PredicateArena()
{
    expressions.push_back({PredicateKind::False, 0, 0, {}});
    expressions.push_back({PredicateKind::True, 0, 0, {}});
}
FailureOr<Predicate> PredicateArena::atom(Value condition)
{
    if (!condition || !condition.getType().isInteger(1)) {
        return failure();
    }
    APInt constant;
    if (matchPattern(condition, m_ConstantInt(&constant))) {
        return constant.isZero() ? Predicate(0) : Predicate(1);
    }
    auto [entry, added] = atoms.try_emplace(condition, expressions.size());
    if (added) {
        expressions.push_back({PredicateKind::Atom, 0, 0, condition});
    }
    return entry->second;
}
FailureOr<Predicate> PredicateArena::atom(Value condition, GuardEvaluationId evaluation)
{
    if (!condition || !condition.getType().isInteger(1)) {
        return failure();
    }
    APInt constant;
    if (matchPattern(condition, m_ConstantInt(&constant))) {
        return constant.isZero() ? Predicate(0) : Predicate(1);
    }
    auto key = std::make_pair(condition, std::make_pair(evaluation.nameSpace, evaluation.evaluation));
    auto [entry, added] = evaluatedAtoms.try_emplace(key, expressions.size());
    if (added) {
        expressions.push_back({PredicateKind::Atom, 0, 0, condition, evaluation});
    }
    return entry->second;
}
Predicate PredicateArena::intern(PredicateKind kind, Predicate first, Predicate second)
{
    auto key = std::make_pair(static_cast<unsigned>(kind), std::make_pair(first, second));
    auto [entry, added] = interned.try_emplace(key, expressions.size());
    if (added) {
        expressions.push_back({kind, first, second, {}});
    }
    return entry->second;
}
Predicate PredicateArena::negate(Predicate operand)
{
    assert(valid(operand) && "predicate belongs to arena");
    if (operand < 2) {
        return 1 - operand;
    }
    if (expressions[operand].kind == PredicateKind::Not) {
        return expressions[operand].first;
    }
    return intern(PredicateKind::Not, operand, 0);
}
Predicate PredicateArena::combine(PredicateKind kind, Predicate first, Predicate second)
{
    assert(valid(first) && valid(second) && "predicates belong to arena");
    const Predicate identity = kind == PredicateKind::And ? 1 : 0;
    const Predicate absorbing = 1 - identity;
    if (first == absorbing || second == absorbing) {
        return absorbing;
    }
    if (first == identity || first == second) {
        return second;
    }
    if (second == identity) {
        return first;
    }
    const bool complements = (expressions[first].kind == PredicateKind::Not && expressions[first].first == second) ||
                             (expressions[second].kind == PredicateKind::Not && expressions[second].first == first);
    if (complements) {
        return absorbing;
    }
    return intern(kind, std::min(first, second), std::max(first, second));
}
Predicate PredicateArena::conjunction(Predicate first, Predicate second)
{
    return combine(PredicateKind::And, first, second);
}
Predicate PredicateArena::disjunction(Predicate first, Predicate second)
{
    return combine(PredicateKind::Or, first, second);
}
} // namespace mlir::pto::frontiersynch
