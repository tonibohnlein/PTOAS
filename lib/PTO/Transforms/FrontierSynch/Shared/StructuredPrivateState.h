// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Transactional scalar-state threading; callbacks supply policy, never payload replay.
#ifndef PTO_FRONTIERSYNCH_STRUCTUREDPRIVATESTATE_H
#define PTO_FRONTIERSYNCH_STRUCTUREDPRIVATESTATE_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ArrayRef.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace mlir::pto::frontiersynch {
struct PrivateStateSlot {
    Type type;
    TypedAttr initial;
};
enum class PrivateStateCut { Before, After };
// Callbacks run synchronously in list order at an original cut. They may create
// bookkeeping in the supplied builder and replace slot Values, but must not
// modify original IR or mapped payloads. Types/extent and final SSA are checked.
// Reset before a loop operation resets that invocation; before its first body
// operation (or yield for an empty body) updates state on each executed visit.
using PrivateStateActionBody = std::function<LogicalResult(
    OpBuilder&, const IRMapping&, llvm::MutableArrayRef<Value>, std::string&)>;
struct PrivateStateAction {
    Operation* anchor = nullptr;
    PrivateStateCut cut = PrivateStateCut::Before;
    PrivateStateActionBody apply;
};
enum class PrivateStateFailure {
    None, InvalidInput, InvalidSlot, InvalidCut, UnsupportedControl, Action, Verification
};
struct PrivateStateError {
    PrivateStateFailure kind = PrivateStateFailure::None;
    Operation* original = nullptr;
    std::string message;
};
struct StructuredPrivateState {
    // Own the symbol context as well as the selected clone. Whole containing
    // module staging is linear in that module's IR; only function is rewritten.
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    IRMapping mapping;
    std::vector<Value> finalState;
};
// Supports one-block func.func with scf.for/scf.if structured control and
// regionless operations. Existing arguments/results/payloads remain unchanged;
// scalar state is appended only to structured control interfaces. No hidden
// reset is introduced. Failure destroys the detached candidate. Original IR
// and anchors must stay alive while using the returned original-keyed mapping.
FailureOr<std::unique_ptr<StructuredPrivateState>> materializeStructuredPrivateState(
    func::FuncOp original, llvm::ArrayRef<PrivateStateSlot> slots,
    llvm::ArrayRef<PrivateStateAction> actions, PrivateStateError& error);
} // namespace mlir::pto::frontiersynch
#endif
