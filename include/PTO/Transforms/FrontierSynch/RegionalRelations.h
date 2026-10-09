// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Producer-independent exact relations. Coordinate tuples identify original payload occurrences.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALRELATIONS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALRELATIONS_H
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
namespace mlir::pto::frontiersynch {
// The algebra uses original signed induction coordinates and period/residue
// quotients. Metadata below names existing sites, not a request to recover or
// analyze their source program. Required order is strict; native order includes
// identities. Internal demand recipes remain owned by the original producers.
struct RegionalRelationData {
    const SyncInput* input = nullptr;
    ArithmeticRegionContext context;
    SmallVector<ArithmeticSite> sites;
    SmallVector<Value> parameterValues;
    std::vector<RegionExpressions::Id> parameters;
    std::vector<scf::ForOp> enclosing;
    SmallVector<ValuePrerequisite> incomingPrerequisites;
    SmallVector<std::size_t> dischargedEffects;
    // Exact required closure is independent of whether every internal cover
    // has this relational representation. Owned producer recipes supply the
    // remaining internal covers when analysis.exactMinimum is false.
    bool completeRequiredOrder = false;
    GeneralArithmeticDemandAnalysis analysis;
    RegionExpressions::RelationCost translationCost;
    ArithmeticStorageSelectors selectors;
    std::vector<ArithmeticOccurrenceDomain> occurrences;
};
struct RegionalRelations {
    RegionalRelationData data;
    GeneralArithmeticRelation newCrossings;
    std::string exportObligation;
    // These owners retain the exact internal demands, including producers whose
    // compact demand representation is not an integer relation.
    std::vector<RegionalAnalysis> children;
    std::vector<std::shared_ptr<Block>> auxiliaryOwners;
};
RegionalRelationData arithmeticRelationData(const ArithmeticRegionalRelations& input);
FailureOr<RegionalRelationData> composeRegionalRelationData(
    const RegionalRelationData& left, const RegionalRelationData& right,
    ArithmeticRegionContext context, std::string& error);
// Lazy relation export and composition. Unsupported callback vocabulary is an
// export obligation; the supplied child analyses are never changed or rerun.
FailureOr<RegionalAnalysis> composeSymbolicRegionalSequence(
    ArrayRef<RegionalAnalysis> children, func::FuncOp function, const SyncInput& input,
    const PhaseIndex& index, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
