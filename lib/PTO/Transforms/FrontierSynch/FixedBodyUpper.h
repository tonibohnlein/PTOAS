// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Selected upper requirements over shared may-access origins, not new effects.
#ifndef PTO_FRONTIERSYNCH_FIXED_BODY_UPPER_H
#define PTO_FRONTIERSYNCH_FIXED_BODY_UPPER_H
#include "PTO/Transforms/FrontierSynch/PeriodicDemandAnalysis.h"
#include "PTO/Transforms/FrontierSynch/CostLedger.h"
#include "PTO/Transforms/FrontierSynch/AnalysisContract.h"
#include <memory>
#include <string>
namespace mlir::pto {
class SyncInput;
}
namespace mlir::pto::frontiersynch {
enum class UpperHazard { RAW, WAR, WAW };
struct UpperOriginDistance {
    std::size_t source, consumer, sourceMemory, targetMemory;
    // No numerical sentinel represents infinity.
    UpperHazard hazard;
    llvm::DynamicAPInt minimum;
    PeriodicDistance maximum;
};
struct UpperFrontierTerm {
    std::size_t consumer, pipe;
    llvm::DynamicAPInt count, offset, startup;
};
class FixedBodyUpper {
public:
    static AnalysisContract contract()
    {
        AnalysisContract result;
        result.closure = SelectedClosure::SoundUpper;
        result.interfaces =
            interfaceBit(DemandInterface::MinimumRepresentation) | interfaceBit(DemandInterface::CompletionFrontiers) |
            interfaceBit(DemandInterface::PeriodicThresholds) | interfaceBit(DemandInterface::RegionQueries) |
            interfaceBit(DemandInterface::UniformMembership);
        return result;
    }
    static FailureOr<std::shared_ptr<const FixedBodyUpper>> build(
        const SyncInput& input, ArrayRef<const CompoundInstanceElement*> phases, CostLedger& costs,
        std::string& reason);
    const PeriodicDemandReduction& selected() const { return upper; }
    ArrayRef<UpperOriginDistance> distances() const { return origins; }
    ArrayRef<UpperFrontierTerm> excessTerms() const { return terms; }
    ArrayRef<PeriodicPrerequisite> consolidated() const { return candidates; }
    llvm::DynamicAPInt excessBound(const llvm::DynamicAPInt& trips) const;
    const llvm::DynamicAPInt& quadraticNumerator() const { return gamma; }
    std::size_t comparisons() const { return queryComparisons; }

private:
    FixedBodyUpper() = default;
    LogicalResult recover(
        const SyncInput& input, ArrayRef<const CompoundInstanceElement*> phases, CostLedger& costs,
        std::string& reason);
    LogicalResult reduce(ArrayRef<const CompoundInstanceElement*> phases, std::string& reason);
    LogicalResult summarize(ArrayRef<const CompoundInstanceElement*> phases, std::string& reason);
    PeriodicDemandReduction upper;
    SmallVector<UpperOriginDistance> origins;
    SmallVector<PeriodicPrerequisite> candidates;
    SmallVector<UpperFrontierTerm> terms;
    llvm::DynamicAPInt gamma = llvm::DynamicAPInt(0);
    std::size_t queryComparisons = 0;
};
} // namespace mlir::pto::frontiersynch
#endif
