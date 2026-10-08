// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Upper storage-class schemas preserved across mixed finite/compact composition.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTSTORAGEBOUNDARY_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTSTORAGEBOUNDARY_H
#include "PTO/Transforms/FrontierSynch/BoundingSequence.h"
#include "PTO/Transforms/FrontierSynch/ConditionalCompactInput.h"
#include "PTO/Transforms/FrontierSynch/FiniteRequirementReplacement.h"
namespace mlir::pto::frontiersynch {
struct CompactClassAccessBoundary {
    uint32_t storageClass = 0, pipe = 0;
    std::vector<std::size_t> contributors; // Actual site alternatives, for universal hardware protection.
    bool read = false, write = false;
    RegionalSelector first, last;
};
class CompactClassBoundary;
using CompactClasses = std::shared_ptr<const CompactClassBoundary>;
struct CompactClassComposition;
struct CompactClassRepetition;
// This upper specification retains EVERY site, without treating uncertain
// writers as kills. Class IDs are local to this immutable owner; identity across
// children is established by comparing their ORIGINAL shared effect unions.
// Lower storage crossings are not inferred from these may-access descriptions.
class CompactClassBoundary {
public:
    const BoundingRegionalResult& bounds() const { return order; }
    const RegionalAnalysis& nativeExports() const { return selectors; }
    llvm::ArrayRef<std::vector<std::size_t>> classes() const { return effects; }
    llvm::ArrayRef<CompactClassAccessBoundary> accesses() const { return sites; }
    const std::string& exportError() const { return unavailable; }
    Block* invocationBlock() const { return invocation; }
    const std::shared_ptr<const CompactOrderBounds>& compact() const { return compactOrders; }
    const FiniteRequirements& finiteFrame() const { return finite; }
    const FiniteSelection& finiteSelection() const { return selectedFinite; }
private:
    CompactClassBoundary() = default;
    BoundingRegionalResult order;
    RegionalAnalysis selectors;
    std::vector<std::vector<std::size_t>> effects;
    std::vector<CompactClassAccessBoundary> sites;
    std::string unavailable;
    Block* invocation = nullptr;
    Operation* firstAnchor = nullptr;
    Operation* lastAnchor = nullptr;
    std::vector<const CompoundInstanceElement*> originalPhases;
    std::shared_ptr<const CompactOrderBounds> compactOrders;
    std::shared_ptr<const CompactWriterReaderInput> compactInput;
    FiniteRequirements finite;
    FiniteSelection selectedFinite;
    std::shared_ptr<const BoundingSequenceResult> composition;
    std::shared_ptr<const void> repetition;
    friend CompactClasses withCompactClassPreparation(CompactClasses,
        std::function<FailureOr<std::unique_ptr<PreparedLogicalPlan>>(ArrayRef<scf::ForOp>)>);
    friend CompactClasses captureCompactClassBoundary(scf::ForOp, const PhaseIndex&,
        std::shared_ptr<const CompactFixedBodyContext>, const CompactWriterReaderBindings&, std::string&);
    friend CompactClasses captureFiniteClassBoundary(FiniteRequirements, FiniteSelection, std::string&);
    friend CompactClassRepetition repeatCompactClassBoundary(func::FuncOp, scf::ForOp, CompactClasses);
    friend CompactClassComposition composeCompactClassBoundariesInBlock(
        func::FuncOp, Block&, const SyncInput&, std::vector<CompactClasses>);
    friend CompactClassComposition composeCompactClassBoundaries(
        func::FuncOp, const SyncInput&, std::vector<CompactClasses>);
};
// Reuses shared fixed/balanced extraction and derives its M3 upper graph itself;
// it cannot bind an unrelated caller-provided class list to an existing graph.
// Native lower facts only. Existing producer bound interfaces remain available
// for richer supplied lower facts/crossings. Missing sequence exports preserve
// all mathematical snapshots; balanced abstract cuts currently remain unavailable.
CompactClasses captureCompactClassBoundary(scf::ForOp loop, const PhaseIndex& index,
    std::shared_ptr<const CompactFixedBodyContext> domain,
    const CompactWriterReaderBindings& bindings, std::string& error);
CompactClasses captureFiniteClassBoundary(FiniteRequirements frame, FiniteSelection selection, std::string& error);
// Attach recipes for exactly the immutable upper selected graph. This trusted
// producer hook changes no graph/domain/storage claim; failures remain detached.
CompactClasses withCompactClassPreparation(CompactClasses original,
    std::function<FailureOr<std::unique_ptr<PreparedLogicalPlan>>(ArrayRef<scf::ForOp>)> preparation);
struct CompactClassCrossings {
    std::string error;
    std::vector<BoundingSequenceCrossing> upper;
    uint64_t accessPairs = 0, effectPairs = 0;
};
// For every ordered child pair, a possible conflict contributes last(source
// site)->first(target site). Native same-pipe chains cover every intervening
// visit, including arbitrary compact trip counts. No uncertain kill, byte copy,
// source expansion or lower conflict inference. Only shared uniform separation
// discharges pairs; all other class unions interact conservatively. Generic
// scalar protection or invocation-wide protection of every actual contributor
// pair can discharge storage crossings; prerequisite edges are untouched.
// O(t^2 + A^2 + sum of tested class-union effect products), independent of trips.
CompactClassCrossings collectCompactClassCrossings(llvm::ArrayRef<CompactClasses> children);
struct CompactClassComposition {
    std::string error;
    std::vector<CompactClasses> original; // Retained even when no composition export can be built.
    std::shared_ptr<const BoundingSequenceResult> mathematical;
    CompactClasses boundary; // Reusable schema with original sites remapped, never killed.
    CompactClassCrossings crossings;
};
// A qualified facade for BoundingSequence's supplied-storage-crossing mode.
// Lower composition adds native/value prerequisites only; upper adds the full
// conservative class crossing inventory. Schemas survive via mathematicalOwner
// and are concatenated for subsequent merges. No physical-byte storage capability
// is asserted, and no endpoint/allocation/realized-excess claim is manufactured.
CompactClassComposition composeCompactClassBoundaries(
    func::FuncOp function, const SyncInput& input, std::vector<CompactClasses> children);
// Explicit relative invocation variant. The children must cover an ordered
// contiguous span of invocation, and every child carries the identical block.
// No invariance across repeated invocations is implied by this composition.
CompactClassComposition composeCompactClassBoundariesInBlock(
    func::FuncOp function, Block& invocation, const SyncInput& input, std::vector<CompactClasses> children);
} // namespace mlir::pto::frontiersynch
#endif
