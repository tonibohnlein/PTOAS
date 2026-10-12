// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Retain actual Boolean evaluations, separately from proving or emitting guard recipes.
#ifndef PTO_FRONTIERSYNCH_BOUNDEDPREDICATEHISTORY_H
#define PTO_FRONTIERSYNCH_BOUNDEDPREDICATEHISTORY_H
#include "StructuredPrivateState.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <cstdint>
namespace mlir::pto::frontiersynch {
struct BoundedPredicateHistoryRequirement {
    Value predicate;
    scf::ForOp loop;
    // Lag counts executed visits of this loop, not raw induction-variable units.
    // The caller separately certifies each guard's original-coordinate mapping.
    uint64_t maximumLag = 0;
};
struct BoundedPredicateHistoryRead {
    Value value;
    Value valid;
};
class BoundedPredicateHistory {
public:
    // Appends initialization, explicit invocation resets, per-visit shifts and
    // snapshots. Install these actions before reads at the same cut. On failure
    // the supplied vectors are unchanged. Requirements must be distinct.
    // Original IR and anchors must remain alive and unchanged during use.
    static FailureOr<BoundedPredicateHistory> append(func::FuncOp original,
        llvm::ArrayRef<BoundedPredicateHistoryRequirement> requirements,
        std::vector<PrivateStateSlot>& slots, std::vector<PrivateStateAction>& actions, std::string& error);
    // Called only inside the state threader's mapped clone callback. No guard is
    // emitted and no validity assumption is made: an invalid value is defined
    // false but must remain masked by valid. Nested ifs are supported; another
    // loop between the cut/definition and tracked loop is explicitly unsupported.
    FailureOr<BoundedPredicateHistoryRead> read(std::size_t requirement, uint64_t lag, Operation* originalCut,
        const IRMapping& mapping, llvm::ArrayRef<Value> state, std::string& error) const;
    std::size_t slotCount() const;
private:
    struct Layout;
    explicit BoundedPredicateHistory(std::shared_ptr<const Layout> layout) : layout(std::move(layout)) {}
    std::shared_ptr<const Layout> layout;
};
} // namespace mlir::pto::frontiersynch
#endif
