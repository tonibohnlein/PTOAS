// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_LOGICALSYNCRELATIONS_H
#define PTO_TRANSFORMS_INSERTSYNC_LOGICALSYNCRELATIONS_H

#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::pto::logical_sync {
using Relation = presburger::PresburgerRelation;
// NotEstablished is a negative result of this sufficient query, not proof of
// impossibility. Algebra results are exact; Proved does not certify a program.
enum class QueryStatus { Proved, NotEstablished, Unsupported, BudgetExhausted };
struct RelationResult {
    QueryStatus status = QueryStatus::Unsupported;
    std::optional<Relation> relation;
    std::string reason;
    explicit operator bool() const { return status == QueryStatus::Proved && relation.has_value(); }
};

// All operations retain integer existential variables. Never use projectOut,
// rational shadows, or the pinned union PWMA lexopt implementation here.
// Work accounting bounds query requests/term growth, not time inside MLIR.
class RelationQueries {
    uint64_t remaining, used = 0;
    bool charge(const Relation& relation);
public:
    explicit RelationQueries(uint64_t budget = 8000000) : remaining(budget) {}
    uint64_t work() const { return used; }
    RelationResult compose(const Relation& first, const Relation& second);
    RelationResult subtract(const Relation& from, const Relation& remove);
    QueryStatus contains(const Relation& supply, const Relation& requirement);

    // R: source -> sink. Both Before arguments must be actual strict schedule
    // order over qualified occurrence domains in the SAME
    // execution (including guards and invocation coordinates). Keep byte in
    // the sink coordinates until per-byte latest producers have been chosen.
    // Coverage of every original sink is checked: an unbounded predecessor
    // family must not silently become an empty set of requirements.
    RelationResult latestSources(const Relation& requirements, const Relation& sourceBefore);
    RelationResult firstTargets(const Relation& requirements, const Relation& targetBefore);
    // latest: source -> target; sourceThrough is inclusive source order.
    // Only acquisitions in that execution can dominate a later demand.
    RelationResult staircase(const Relation& latest, const Relation& sourceThrough,
                             const Relation& targetBefore);
};

// An exact underapproximation of completion for ONE immutable logical plan.
// Early success does not mark a fixed point. A subsequent reuse query can
// continue composition. Replacing a plan requires a new instance.
class CompletionQueries {
    Relation known;
    bool fixed = false;
public:
    explicit CompletionQueries(Relation primitiveSupply) : known(std::move(primitiveSupply)) {}
    QueryStatus prove(const Relation& requirement, RelationQueries& queries, unsigned rounds = 8);
    const Relation& supply() const { return known; }
    bool fixedPoint() const { return fixed; }
};
} // namespace mlir::pto::logical_sync
#endif
