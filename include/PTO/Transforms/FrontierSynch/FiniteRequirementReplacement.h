// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact finite replacement certificates from original shared-model scans.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_FINITEREQUIREMENTREPLACEMENT_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_FINITEREQUIREMENTREPLACEMENT_H
#include "PTO/Transforms/FrontierSynch/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RequirementProvenance.h"
namespace mlir::pto::frontiersynch {
class FiniteRequirementFrame;
using FiniteRequirements = std::shared_ptr<const FiniteRequirementFrame>;
struct FiniteReplacementRecord {
    RequirementRecordKey key;
    StorageGenerator edge;
    RegionExpressions::Id guard = RegionExpressions::invalid;
};
// A privately captured finite invocation. Owns all numerical records and original
// ownership, borrows unchanged original IR/SyncInput. No periodic unrolling or
// reduced-survivor reconstruction. The namespace maps original scan indices.
class FiniteRequirementFrame {
public:
    const RequirementSnapshot& original() const { return provenance; }
    const ExplicitAnalysis& analysis() const { return analyzed; }
    llvm::ArrayRef<StorageGenerator> native() const { return nativeEdges; }
    Block* invocationBlock() const { return invocation; }
private:
    FiniteRequirementFrame() = default;
    RequirementSnapshot provenance;
    Block* invocation = nullptr;
    ExplicitAnalysis analyzed;
    std::vector<StorageGenerator> nativeEdges;
    friend FiniteRequirements captureFiniteRequirementsInBlock(func::FuncOp, Block&, const SyncInput&,
        llvm::ArrayRef<const CompoundInstanceElement*>, std::shared_ptr<RegionExpressions>,
        uint64_t, llvm::ArrayRef<RequirementGroupId>, RequirementGroupId, std::string&);
    friend FiniteRequirements captureFiniteRequirements(func::FuncOp, const SyncInput&,
        llvm::ArrayRef<const CompoundInstanceElement*>, std::shared_ptr<RegionExpressions>,
        uint64_t, llvm::ArrayRef<RequirementGroupId>, RequirementGroupId, std::string&);
};
// Rebuilds the shared PhaseIndex and original explicit scan, then captures ALL
// witnesses before reduction. atomGroups has one entry per shared physical cell;
// equal IDs merge cells into one original group. Residual/supplied requirements
// get the fixed group. Unknown geometry stays in the shared model; no recovery.
// Phases form a complete contiguous explicit span, with no enclosing loop. This
// is one invocation's frame; enclosing conditional presence belongs to its owner.
FiniteRequirements captureFiniteRequirements(func::FuncOp function, const SyncInput& input,
    llvm::ArrayRef<const CompoundInstanceElement*> phases, std::shared_ptr<RegionExpressions> expressions,
    uint64_t producer, llvm::ArrayRef<RequirementGroupId> atomGroups,
    RequirementGroupId prerequisiteGroup, std::string& error);
// Explicit relative body invocation, for a separately qualified repetition.
// Phases belong directly to invocation; no dynamic occurrences are unfolded.
// Capturing this frame alone establishes no equality across invocations.
FiniteRequirements captureFiniteRequirementsInBlock(func::FuncOp function, Block& invocation, const SyncInput& input,
    llvm::ArrayRef<const CompoundInstanceElement*> phases, std::shared_ptr<RegionExpressions> expressions,
    uint64_t producer, llvm::ArrayRef<RequirementGroupId> atomGroups,
    RequirementGroupId prerequisiteGroup, std::string& error);
struct FiniteReplacementProof {
    std::string error;
    RequirementReplacement evidence;
    FiniteRequirements frame;
    // Keep definitions alive alongside the opaque evidence; these immutable
    // keys/edges are needed when rebuilding graphs after group-mask application.
    std::shared_ptr<const std::vector<FiniteReplacementRecord>> records;
    uint64_t closureUpdates = 0, equalityChecks = 0;
};
// B is intrinsic/native order plus the original fixed prerequisites. Proves
// TC(B union candidate) == TC(B union ALL ORIGINAL group requirements) on beta.
// Candidate records must be fresh, forward C->I edges in this finite frame.
// Existing sound Boolean implication checks every closure entry, never sampled
// valuations. Unproved guarded equality rejects evidence only. The immutable
// original snapshot or its revisions may be used; foreign ownership is rejected.
// Other groups/crossings never participate in the proof or get removed. Applying
// the returned evidence uses applyRequirementReplacement BEFORE identity merging.
// Work: O(V^3) Boolean gates, O(V^2) implication requests on their reachable DAGs;
// retained closure is O(V^2), proof DAG O(V^3), plus candidate/original records.
FiniteReplacementProof certifyFiniteRequirementReplacement(FiniteRequirements frame,
    RequirementSnapshot snapshot, RequirementGroupId group, RegionExpressions::Id beta,
    std::vector<FiniteReplacementRecord> candidate);
class FiniteRequirementSelection;
using FiniteSelection = std::shared_ptr<const FiniteRequirementSelection>;
// Only the concrete finite checker can create this selected-record snapshot.
// Its guards may change ownership/records, but its closure equals the original
// frame uniformly. Consumers may reuse original exact queries, not cover tags.
class FiniteRequirementSelection {
public:
    const FiniteRequirements& frame() const { return origin; }
    const RequirementSnapshot& snapshot() const { return selected; }
    llvm::ArrayRef<FiniteReplacementRecord> definitions() const { return records; }
private:
    FiniteRequirementSelection() = default;
    FiniteRequirements origin;
    RequirementSnapshot selected;
    std::vector<FiniteReplacementRecord> records;
    friend FiniteSelection applyCertifiedFiniteReplacement(FiniteRequirements, FiniteSelection,
        RequirementGroupId, RegionExpressions::Id, std::vector<FiniteReplacementRecord>, std::string&);
};
// previous may be null for the original scan. Failure leaves it untouched.
// Certification and group masking happen before any downstream identity merge;
// all original and introduced typed definitions remain owned by the result.
FiniteSelection applyCertifiedFiniteReplacement(FiniteRequirements frame, FiniteSelection previous,
    RequirementGroupId group, RegionExpressions::Id beta,
    std::vector<FiniteReplacementRecord> candidate, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
