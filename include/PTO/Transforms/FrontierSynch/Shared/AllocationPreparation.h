// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owned allocation capabilities for one unchanged logical command graph.
#ifndef PTO_FRONTIERSYNCH_ALLOCATIONPREPARATION_H
#define PTO_FRONTIERSYNCH_ALLOCATIONPREPARATION_H
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <memory>
namespace mlir::pto::frontiersynch {
struct PreparedLogicalPlan;
class AllocationProfile;
struct RegionalAllocationSummary;
class AllocationProofArchive;
enum class AllocationStrength { Independent, Shared };
class AllocationPreparation {
public:
    using Summary = std::shared_ptr<RegionalAllocationSummary>;
    using Legacy = std::function<void(PreparedLogicalPlan&)>;
    using Builder = std::function<Summary(PreparedLogicalPlan&, AllocationStrength)>;
    AllocationPreparation() = default;
    AllocationPreparation& operator=(Legacy prepare);
    // Cheap and stronger requests share retained mathematics but cache distinct
    // answers, including failure. Shared is terminal: no later request rebuilds
    // the weaker construction. Copies refer to the same owned capability.
    static AllocationPreparation candidates(Builder build);
    Summary request(PreparedLogicalPlan& plan, AllocationStrength strength);
    // Read an already completed answer without invoking a producer. A null
    // answer can mean unavailable or not requested; request() retains failure
    // memoization. Shared-first has the same terminal semantics as request().
    Summary retained(AllocationStrength strength) const;
    struct CompletedCertificate {
        Summary owner;
        DictionaryAttr evidence;
    };
    // Only certificates completed by a required parent export are retained.
    // The exact frozen protocol and completed summary owner bind their reuse;
    // each eventual eligible pool still requires ordinary certificate decoding.
    void retainCertificate(const PreparedLogicalPlan& plan, const Summary& owner, DictionaryAttr evidence,
        ArrayRef<int64_t> eligibleIds = {}) const;
    SmallVector<CompletedCertificate> retainedCertificates(
        const PreparedLogicalPlan& plan, ArrayRef<int64_t> eligibleIds = {}) const;
    // Preserve completed proof data and owners when a callback is consumed.
    // This archive contains no builder and never constructs missing evidence.
    void preserveCertificates(PreparedLogicalPlan& plan) const;
    // Compatibility for unmigrated one-shot producers and wrappers.
    void operator()(PreparedLogicalPlan& plan);
    explicit operator bool() const; // A builder may still hold non-SSA values.
private:
    struct State;
    std::shared_ptr<State> state;
};
// Requesting an alternative does not select it or change the logical graph.
std::shared_ptr<RegionalAllocationSummary> requestAllocationSummary(
    PreparedLogicalPlan& plan, AllocationStrength strength, AllocationProfile* profile = nullptr);
} // namespace mlir::pto::frontiersynch
#endif
