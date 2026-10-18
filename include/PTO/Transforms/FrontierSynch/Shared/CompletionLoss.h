// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Execution-parametric CI-pair ceilings; no runtime synchronization analysis.
#ifndef PTO_COMPLETION_LOSS_H
#define PTO_COMPLETION_LOSS_H
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/APInt.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>
namespace mlir::pto::frontiersynch {
struct PeriodicExcessCertificate;
struct CompletionLossAtom {
    // Original CFG vertex; opaque child visits supply their actual payload count.
    uint64_t vertex = 0;
    std::optional<uint64_t> payloads;
    bool exactRealizedOrder = false;
    // Stable postorder operation ordinals in the retained original function.
    // Both absent denotes a synthetic CFG node. Child endpoints bound its
    // contiguous original root interval; inner payloads belong to that visit.
    std::optional<uint64_t> firstOperation, lastOperation;
};
struct CompletionLossVisit { uint64_t atom = 0, payloads = 0; };
struct CompletionLossValue {
    llvm::APInt payloads{256, 0}, bound{256, 0};
    std::string error;
    // Trusted internal certificate: Ui bounds this child's REALIZED CI excess.
    // Checked unsigned arithmetic; a rejected append leaves both numbers intact.
    bool append(const llvm::APInt& count, const llvm::APInt& internal);
    // Use the existing fixed-body engine. Equality to its upper graph is a
    // separate producer obligation; selected-order bounds alone do not suffice.
    bool appendPeriodic(const PeriodicExcessCertificate& certificate, bool realizedUpperEquality);
};
class CompletionLossCertificate {
public:
    // Trusted production partition. Actual visits must be a finite execution
    // of this immutable owner's CFG; the evaluator is not a control interpreter.
    CompletionLossCertificate(std::vector<CompletionLossAtom> atoms,
        std::vector<std::vector<std::size_t>> successors, std::size_t entry, std::size_t exit);
    bool valid() const { return shapeValid; }
    const std::vector<CompletionLossAtom>& partition() const { return atoms; }
    CompletionLossValue evaluate(llvm::ArrayRef<CompletionLossVisit> visits) const;
    DictionaryAttr encode(MLIRContext* context) const;
private:
    std::vector<CompletionLossAtom> atoms;
    std::vector<std::vector<std::size_t>> successors;
    std::size_t entry, exit;
    bool shapeValid = false;
};
} // namespace mlir::pto::frontiersynch
#endif
