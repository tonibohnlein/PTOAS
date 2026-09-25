// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COVERINGBOUNDARY_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COVERINGBOUNDARY_H
#include "PTO/Transforms/FrontierSynch/OriginalProgramPoints.h"

namespace mlir::pto::frontiersynch {
enum class CoveringDirection { Source, Target };

// A reference to ORIGINAL work, not a flattened execution or a selected prefix.
// A structured item denotes the entire original choice/repeat with all its
// alternatives, effects and multiplicities. Sequence wrappers are transparent.
struct CoveringWorkItem {
    Region::Kind kind = Region::Operation;
    std::size_t operation = NoControlId, owner = NoControlId;
    OriginalCut before, after;
};
// The cut-delimited interval is authoritative, including original scalar/control
// work not represented as translated payload leaves. Items are structural
// references into that interval, not an exhaustive flattened instruction list.
struct CoveringWorkScope {
    OriginalCut begin, end;
    std::vector<CoveringWorkItem> items;
};
struct CoveringBoundary {
    enum class Status { Unknown, NoHit, Covering } status = Status::Unknown;
    enum class Obstruction {
        None, InvalidInterval, EngineRequired, OccurrenceUnqualified,
        NotSingleEntrySingleExit, RepetitionUnqualified, CutUnavailable
    } obstruction = Obstruction::None;
    CoveringDirection direction = CoveringDirection::Source;
    OriginalInterval interval;
    // Only a Covering answer has an endpoint. NoHit never invents one.
    std::optional<OriginalCut> cut;
    std::string reason;

    // All matching issues in interval precede (Source) or follow (Target) this
    // boundary, per interval visit. This is reference coverage, NOT completion,
    // matching, event reuse, an exactly-participating frontier or an order proof.
    bool referenceCoverage = false;
    bool conservativeEffects = false;
    // Opaque qualifier/predicate references restrict the requested family; the
    // cut may cover a larger may-family without claiming their participation.
    bool conservativeSelection = false;
    // Static may-incidences, never a claim that every listed site executes.
    std::vector<std::size_t> mayAccesses;
    OriginalContinuationCases cases;

    // The rule emits no new observation: true at the cut, within its existing
    // original control. Exactly one cut visit per admitted interval visit,
    // including its empty-access paths. The enclosing loop may invoke it again.
    bool guardIsTrue = false;
    bool exactlyOneVisitPerInterval = false;
    // A conservative possibility, NOT an executable no-hit guard. Construction
    // must include these paths in its matching/ordering checks; the original
    // factored relation can refine them without changing this boundary.
    bool mayExecuteWithoutAccess = false;
    bool mayRepeatInInvocation = false;
    OriginalCut visitEntry, visitExit;

    // Source: interval entry .. cut. Target: cut .. interval end. This preserves
    // unrelated work inside a retained choice/loop. trimmedWork is the proved
    // access-free suffix/prefix excluded by the rule. These are not an exact
    // delta against another frontier, nor a proof of additional payload order.
    CoveringWorkScope enclosedWork, trimmedWork;
    // Unchanged original context before/after the interval is deliberately NOT
    // asserted empty. The whole engine prefix at cut also includes that context;
    // Phase B must inspect its actual snapshot and selected command word.
    bool requiresWholePrefixCheck = true;
    // loopOwners includes failed qualifications on Unknown results as well.
    std::vector<std::size_t> loopOwners, finiteLoopOwners;
};
// Counts query/inspection work and references in returned arrays, not allocated
// bytes, wall time, or a bound hiding ordered-map/key-comparison costs.
struct CoveringBoundaryStats {
    std::size_t queries = 0, cacheHits = 0, summaryEvaluations = 0;
    std::size_t intervalNodes = 0, materializedReferences = 0;
    std::size_t effectQueries = 0;
};
} // namespace mlir::pto::frontiersynch
#endif
