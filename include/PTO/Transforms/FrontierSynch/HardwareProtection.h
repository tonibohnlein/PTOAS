// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared target access protection, independent of recognition and insertion.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_HARDWAREPROTECTION_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_HARDWAREPROTECTION_H
#include "PTO/Transforms/FrontierSynch/LifetimeScan.h"
#include "mlir/IR/Types.h"
#include <unordered_set>
namespace mlir {
class Operation;
namespace pto {
class SyncInput;
class CompoundInstanceElement;
namespace frontiersynch {
// A group certifies protection between every ordered pair of its WRITERS to
// one resource, on one pipe. It is not a completion-before-start edge. Group
// identities include dynamic scope; zero means ordinary software ordering.
inline bool hardwareProtectsConflict(uint32_t sourcePipe, uint64_t sourceGroup,
                                     uint32_t targetPipe, uint64_t targetGroup)
{
    return sourceGroup != 0 && sourceGroup == targetGroup && sourcePipe == targetPipe;
}
// Consume a reference-ordered sequence of exact occurrences. The caller supplies
// the accumulator's normalized physical atoms, NOT SSA allocation identities.
// Reuse this builder across adjacent explicit pieces; endScope at unresolved
// control/region boundaries. Separate repeated visits must be observed again.
class HardwareProtectionBuilder {
public:
    void observe(Operation* operation, ExplicitEffects& occurrence,
                 llvm::ArrayRef<uint32_t> accumulatorAtoms);
    void endScope();
private:
    uint64_t nextGroup = 0;
    uint64_t activeGroup = 0;
    Type accumulatorType;
    std::unordered_set<uint32_t> activeAtoms;
};
// Conservative applicability check for routes that have no conditional
// hardware-protection export. Uses the shared target rule, including pairs
// crossing repeated visits; it does not change effects or add native edges.
bool mayHaveHardwareProtectedPair(const SyncInput& input,
                                 llvm::ArrayRef<const CompoundInstanceElement*> phases);
} // namespace frontiersynch
} // namespace pto
} // namespace mlir
#endif
