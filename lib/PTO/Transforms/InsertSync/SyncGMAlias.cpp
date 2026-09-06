// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Preserve all possible argument roots through views and structured forwarding.
// Opaque producers, integer arithmetic and exhausted traces cannot prove noalias.
#include "PTO/Transforms/InsertSync/SyncGMAlias.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr unsigned kMaximumRootValues = 256;

bool appendLoopSources(scf::ForOp loop, unsigned index, SmallVectorImpl<Value>& pending)
{
    if (index >= loop.getInitArgs().size() || index >= loop.getYieldedValues().size()) {
        return false;
    }
    pending.push_back(loop.getInitArgs()[index]);
    pending.push_back(loop.getYieldedValues()[index]);
    return true;
}

bool appendSources(Value value, func::FuncOp function, SmallVectorImpl<Value>& pending, SmallVectorImpl<Value>& roots)
{
    if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (argument.getOwner() == &function.getBody().front()) {
            if (!isa<PtrType, TensorViewType, PartitionTensorViewType>(value.getType())) {
                return false;
            }
            roots.push_back(value);
            return true;
        }
        auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
        if (!loop || argument.getArgNumber() == 0 || argument.getOwner() != loop.getBody()) {
            return false;
        }
        return appendLoopSources(loop, argument.getArgNumber() - 1, pending);
    }
    Operation* op = value.getDefiningOp();
    if (!op) {
        return false;
    }
    if (auto cast = dyn_cast<CastPtrOp>(op)) {
        if (!isa<PtrType>(cast.getInput().getType())) {
            return false;
        }
        pending.push_back(cast.getInput());
        return true;
    }
    if (isa<AddPtrOp, MakeTensorViewOp, PartitionViewOp, SubViewOp, TReshapeOp, BitcastOp>(op)) {
        pending.push_back(op->getOperand(0));
        return true;
    }
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
        pending.push_back(select.getTrueValue());
        pending.push_back(select.getFalseValue());
        return true;
    }
    unsigned index = cast<OpResult>(value).getResultNumber();
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
        return appendLoopSources(loop, index, pending);
    }
    if (auto choice = dyn_cast<scf::IfOp>(op)) {
        if (choice.getThenRegion().empty() || choice.getElseRegion().empty()) {
            return false;
        }
        for (Region* region : {&choice.getThenRegion(), &choice.getElseRegion()}) {
            auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
            if (!yield || index >= yield.getResults().size()) {
                return false;
            }
            pending.push_back(yield.getResults()[index]);
        }
        return true;
    }
    return false;
}
} // namespace

FailureOr<InsertSyncGMAliasMode> mlir::pto::resolveInsertSyncGMAlias(func::FuncOp function, StringRef overrideMode)
{
    // Distinct SSA arguments are not a noalias promise. Callers may supply one
    // explicitly; it is usable only after a complete argument-root proof.
    StringRef mode = "may-alias";
    if (Attribute attribute = function->getAttr("pto.gm_alias")) {
        auto text = dyn_cast<StringAttr>(attribute);
        if (!text || (text.getValue() != "may-alias" && text.getValue() != "assume-disjoint-arguments")) {
            function.emitError("invalid pto.gm_alias contract");
            return failure();
        }
        mode = text.getValue();
    }
    if (!overrideMode.empty()) {
        mode = overrideMode;
    }
    if (mode != "may-alias" && mode != "assume-disjoint-arguments") {
        function.emitError("InsertSync GM mode must be may-alias or assume-disjoint-arguments");
        return failure();
    }
    return mode == "may-alias" ? InsertSyncGMAliasMode::MayAlias : InsertSyncGMAliasMode::DisjointArguments;
}

InsertSyncGMRoots mlir::pto::traceInsertSyncGMRoots(func::FuncOp function, Value value)
{
    InsertSyncGMRoots result;
    if (!function || function.getBody().empty() || !value) {
        return result;
    }
    SmallVector<Value, 8> pending{value};
    llvm::DenseSet<Value> visited;
    while (!pending.empty()) {
        Value current = pending.pop_back_val();
        if (!current) {
            return result;
        }
        if (!visited.insert(current).second) {
            continue;
        }
        if (visited.size() > kMaximumRootValues || !appendSources(current, function, pending, result.arguments)) {
            return result;
        }
    }
    result.complete = !result.arguments.empty();
    return result;
}

bool mlir::pto::disjointInsertSyncGMRoots(func::FuncOp function, Value first, Value second)
{
    auto left = traceInsertSyncGMRoots(function, first);
    auto right = traceInsertSyncGMRoots(function, second);
    return left.complete && right.complete &&
           llvm::none_of(left.arguments, [&](Value root) { return llvm::is_contained(right.arguments, root); });
}
