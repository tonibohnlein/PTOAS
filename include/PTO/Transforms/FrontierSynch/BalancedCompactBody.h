// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Structural slots and distributed endpoints on unchanged original branch arms.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_BALANCEDCOMPACTBODY_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_BALANCEDCOMPACTBODY_H
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace mlir::pto::frontiersynch {
struct BalancedCompactAlternative {
    const CompoundInstanceElement* phase = nullptr;
    TemplateEndpointCut before, after;
    // Original outer-to-inner scf.if arm regions inside this body. Exactly one
    // alternative per slot executes; no predicate is evaluated at another cut.
    SmallVector<Region*> path;
};
struct BalancedCompactSlot {
    uint32_t pipe = 0;
    std::vector<BalancedCompactAlternative> alternatives;
};
enum class BalancedCompactIssue { None, InvalidBinding, UnsupportedStructure, DifferentSignatures, LeafContract };
struct BalancedCompactBody {
    std::string error;
    BalancedCompactIssue issue = BalancedCompactIssue::None;
    scf::ForOp loop;
    std::vector<BalancedCompactSlot> slots;
    DenseMap<const CompoundInstanceElement*, uint32_t> phaseSlots;
    // Original prerequisite identities, including incoming/outgoing and
    // phase-less control/interface consumers. No slot mapping is inferred here.
    std::vector<ValuePrerequisite> prerequisites;
    std::vector<Operation*> unresolvedPrerequisites;
    bool carriedPrerequisites = false;
    uint64_t syntaxNodes = 0, signatureComparisons = 0, pathElements = 0;
};
// All original IR/input/index objects are borrowed unchanged. Recursion compares
// pipe words at each scf.if join, including empty arms; no inner loops, dummy
// payloads, path enumeration or branch satisfiability. Ordinary fixed bodies
// use singleton alternatives. Shared leaf contracts and prerequisites survive.
// Work/storage is polynomial in syntax/explicit signatures and path metadata;
// counters exclude the original PhaseIndex construction and prerequisite scan.
BalancedCompactBody recognizeBalancedCompactBody(scf::ForOp loop, const SyncInput& input, const PhaseIndex& index);

// analysis is a certified slot relation covering all required shared effects AND
// prerequisite obligations, with actual local-barrier strengthening accounted
// for in its selected order. No proof of that semantic premise is inferred here.
// body must be the unmodified successful recognizer result. The function must
// contain the same original body; no mutation is performed by
// preparation. Supported executable arithmetic: 64-bit index, nonnegative
// constant lower bound, positive constant step, arbitrary original upper bound.
// Every recipe is copied independently to all alternatives on each side, using
// the SAME original record ID and source iteration ordinal. Original control
// chooses one source and one target; future branch guards are never replayed.
// Grouped metadata qualifies exhaustive original cut choices structurally. An
// explicit enclosing loop chain (outermost first) permits singleton fixed-body
// slots; its invocation coordinates are added by the enclosing exporter.
// Multi-arm bodies currently require no enclosing loops. Original enclosing
// scf.if arms choose the whole invocation and require no predicate replay.
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareBalancedCompactInsertion(
    func::FuncOp function, const BalancedCompactBody& body, const PeriodicAnalysis& analysis,
    int64_t planId, std::string& error, ArrayRef<scf::ForOp> enclosing = {});
} // namespace mlir::pto::frontiersynch
#endif
