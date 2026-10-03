// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_PERIODIC_EVENT_ASSIGNMENT_H
#define PTO_FRONTIERSYNCH_PERIODIC_EVENT_ASSIGNMENT_H
#include "PTO/Transforms/FrontierSynch/PeriodicDemandAnalysis.h"
#include <string>
namespace mlir::pto::frontiersynch {
enum class PeriodicAssignmentStatus { Certified, Counterexample, Unsupported };
struct PeriodicEventAssignment {
    PeriodicAssignmentStatus status = PeriodicAssignmentStatus::Unsupported;
    // nullopt means infinite, never an integer sentinel. Required concerns the
    // requested prefix (when supplied), uniformRequired all finite prefixes.
    PeriodicDistance required, uniformRequired;
    SmallVector<PeriodicDistance> firstReusableSeparations;
    SmallVector<PeriodicPrerequisite> phases;
    SmallVector<unsigned> eligibleIds;
    std::optional<llvm::DynamicAPInt> handoffCount;
    std::string reason;
    // phase indexes the ordered pool phases, not a static operation or loop IV.
    // The ID indexes the supplied eligible vector, preserving reserved holes.
    FailureOr<unsigned> sourceId(std::size_t phase, const llvm::DynamicAPInt& sourcePeriod) const;
    FailureOr<unsigned> consumerId(std::size_t phase, const llvm::DynamicAPInt& consumerPeriod) const;
};
// Implements prop:periodic-physical-budget for a complete directed pool of
// retained covers in a certified numerical repeating-prefix closure. Records
// may be unordered; the returned phases are sorted by source position.
// The caller MUST qualify order-exact direct insertion, adjacent local covers,
// target command/visibility/invocation contracts and legal ID selection. These
// are not inferred from a mathematical quotient. Exceptional entry/interface
// paths, independent occurrence guards and resets require another certifier.
// prefixPayloads selects a finite prefix from period zero; absence asks for a
// uniform assignment over all prefixes. Numeric trips/delays are not expanded.
// Pool validation costs O(g+c log(c+1)); the budget uses c^2 threshold queries
// and O(c) storage, excluding quotient query and integer bit costs.
PeriodicEventAssignment assignPeriodicEvents(const PeriodicDemandReduction& reduction,
    ArrayRef<std::size_t> poolRecords, ArrayRef<unsigned> eligibleIds,
    std::optional<llvm::DynamicAPInt> prefixPayloads = std::nullopt);
} // namespace mlir::pto::frontiersynch
#endif
