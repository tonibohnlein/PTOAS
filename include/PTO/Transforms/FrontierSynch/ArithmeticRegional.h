// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICREGIONAL_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ARITHMETICREGIONAL_H
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticStorageSelectors.h"
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
// Owned symbolic export. Original IR/shared input are borrowed as in the
// regional contract; all imported relations and parameter bindings are owned.
// Composition consumes these relations without reconstructing child accesses.
struct ArithmeticRegionalRelations {
    const SyncInput* input = nullptr;
    std::shared_ptr<const SyncInput> inputOwner;
    std::shared_ptr<const PhaseIndex> indexOwner;
    SmallVector<std::pair<Value, int64_t>> entryConstants;
    ArithmeticProgram program;
    GeneralArithmeticDemandAnalysis analysis;
    ArithmeticStorageSelectors selectors;
    std::vector<ArithmeticOccurrenceDomain> occurrences;
    std::vector<RegionExpressions::Id> parameters;
    std::vector<scf::ForOp> enclosing;
};
// Session mathematics contains no arena-local IDs or optional storage exports.
std::shared_ptr<const ArithmeticRegionalRelations> analyzeArithmeticRegionDemands(
    ArithmeticRegionContext context, const PhaseIndex& index, const SyncInput& input,
    const ArithmeticProgram* certified, std::string& error);
// Pure phase mathematics: constants specialize the producer; remaining entry
// values are bound only by a later transactional export. No arena IDs escape.
std::shared_ptr<const ArithmeticRegionalRelations> analyzeSpecializedArithmeticDemands(
    ArithmeticRegionContext context, const PhaseIndex& index, const SyncInput& input,
    ArrayRef<ArithmeticLimits> profiles, ArithmeticEntryConstant entryConstant, std::string& error);
// The binding must reestablish every consumed constant and describe the same
// certified phase/interval. Remaining symbolic parameters may vary pointwise.
FailureOr<RegionalAnalysis> exportSpecializedArithmeticRegion(
    const ArithmeticRegionalRelations& demands, std::shared_ptr<RegionExpressions> expressions,
    bool selectors, std::string& error,
    std::function<std::optional<RegionExpressions::Id>(Value)> parameterBinding);
// Probe finite storage support from a cached form before reducing demands.
// No expressions or query callbacks escape this transaction.
bool hasFiniteArithmeticStorage(const ArithmeticProgram& program, const SyncInput& input,
    std::shared_ptr<RegionExpressions> expressions, std::string& error);
// Each requested snapshot has independent callback state in the common arena.
// Failed construction rolls back its unpublished nodes and arena diagnostics.
FailureOr<RegionalAnalysis> exportArithmeticRegion(
    const ArithmeticRegionalRelations& demands, std::shared_ptr<RegionExpressions> expressions,
    bool selectors, std::string& error, std::shared_ptr<const SyncInput> inputOwner = {});
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareArithmeticRegion(
    const ArithmeticRegionalRelations& demands, std::shared_ptr<RegionExpressions> expressions,
    ArrayRef<scf::ForOp> enclosing, std::string& error,
    std::shared_ptr<const SyncInput> inputOwner = {});
// Analyze one original region with shared entry bindings. The finite-boundary
// adapter enumerates bounded physical bytes, never dynamic payload occurrences;
// adjacent bytes with identical selector tuples share one exported cell.
// A certified phase view can bind shared entry parameters before queries and
// selectors are constructed. Caller-supplied event coordinates remain untouched.
// Preparation retains original IR and requires the parent to enforce that phase.
FailureOr<RegionalAnalysis> analyzeArithmeticRegion(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error,
    std::function<std::optional<RegionExpressions::Id>(Value)> parameterBinding = {});
FailureOr<RegionalAnalysis> analyzeArithmeticRegionWithProfiles(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error, ArrayRef<ArithmeticLimits> profiles,
    std::function<std::optional<RegionExpressions::Id>(Value)> parameterBinding = {});
// Preserve exact reduction independently of selector/query export. The owned
// result retains its original context and expression arena on export failure.
FailureOr<RegionalAnalysis> analyzeArithmeticRegionRetained(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::shared_ptr<const ArithmeticRegionalRelations>& demands, std::string& error);
FailureOr<RegionalAnalysis> analyzeArithmeticRegionRetained(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::shared_ptr<const ArithmeticRegionalRelations>& demands, std::string& error,
    const ArithmeticProgram* certified);
// Request only the finite boundary representation. Symbolic support defers this
// optional route before demand/selector construction; callers retain the ordinary
// symbolic-capable analysis above as a fallback. No occurrence expansion occurs.
FailureOr<RegionalAnalysis> analyzeFiniteArithmeticRegion(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error);
FailureOr<RegionalAnalysis> analyzeFiniteArithmeticRegionWithProfiles(ArithmeticRegionContext context,
    const PhaseIndex& index, const SyncInput& input, std::shared_ptr<RegionExpressions> expressions,
    std::string& error, ArrayRef<ArithmeticLimits> profiles);
}
#endif
