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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include <unordered_set>
#include <optional>
namespace mlir {
class Operation;
namespace pto {
class SyncInput;
class SyncStorageEffects;
class CompoundInstanceElement;
namespace frontiersynch {
inline constexpr uint64_t invocationProtectionBit = uint64_t{1} << 63;
// An initializer publishes protected output, but never consumes protection.
inline constexpr uint64_t protectionResetBit = uint64_t{1} << 62;
// A group certifies ordered writer pairs to one resource on one pipe, except
// pairs targeting a reset. It is not a completion-before-start edge. Identities
// include dynamic scope; zero means ordinary software ordering.
inline bool hardwareProtectsConflict(uint32_t sourcePipe, uint64_t sourceGroup,
                                     uint32_t targetPipe, uint64_t targetGroup)
{
    return sourceGroup != 0 && targetGroup != 0 && !(targetGroup & protectionResetBit) &&
        (sourceGroup & ~protectionResetBit) == targetGroup && sourcePipe == targetPipe;
}
// Consume a reference-ordered sequence of exact occurrences. The caller supplies
// the accumulator's normalized physical atoms, NOT SSA allocation identities.
// This is the leaf transfer used by StructuredProtection, not a backend-local
// analysis. Structured consumers project its scoped facts into occurrences.
class HardwareProtectionBuilder {
public:
    void observe(Operation* operation, ExplicitEffects& occurrence,
                 llvm::ArrayRef<uint32_t> accumulatorAtoms);
    void endScope();
    uint64_t currentGroup() const { return activeGroup; }
private:
    uint64_t nextGroup = 0;
    uint64_t activeGroup = 0;
    Type accumulatorType;
    std::unordered_set<uint32_t> activeAtoms;
};
// Static facts are scoped to one visit of a nonuniform enclosing loop. Uniform
// structured children share the incoming chain; reset writers retain ordinary
// incoming demands. Consumers must project facts into their occurrence context.
struct ProtectionFact {
    uint64_t group = 0;
    Operation* scope = nullptr;
};
class StructuredProtection {
public:
    StructuredProtection() : facts() {}
    llvm::DenseMap<const CompoundInstanceElement*, ProtectionFact> facts;
    uint64_t lookup(const CompoundInstanceElement* phase) const;
    uint64_t at(const CompoundInstanceElement* phase) const;
    uint64_t inLoop(const CompoundInstanceElement* phase, scf::ForOp loop) const;
    uint64_t within(const CompoundInstanceElement* phase, llvm::ArrayRef<scf::ForOp> enclosing) const;
};
StructuredProtection structuredProtection(const SyncStorageEffects& storage);
struct MatrixProtectionInfo { Type accumulator; bool initializes = false; };
std::optional<MatrixProtectionInfo> matrixProtectionInfo(Operation* operation);
// Protection for residual effect pairs with identical buffer operands. This
// preserves the same target rule when physical addresses are symbolic. Results
// are indexed by shared effect ID; zero requires ordinary software ordering.
std::vector<uint64_t> modeledProtectionGroups(const SyncInput& input,
    llvm::ArrayRef<const CompoundInstanceElement*> phases, const StructuredProtection& protection);
} // namespace frontiersynch
} // namespace pto
} // namespace mlir
#endif
