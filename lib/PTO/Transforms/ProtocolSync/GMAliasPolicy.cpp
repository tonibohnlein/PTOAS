// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- GMAliasPolicy.cpp - Root-preserving GM caller contracts -----------===//
//
// Root recovery is deliberately independent of the legacy alias-family map.
// Integer round trips, opaque calls and unsupported forwarding stay unknown.
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/ProtocolSync/GMAliasPolicy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::protocol_sync;

namespace {

constexpr unsigned kMaximumRootTraceValues = 256;

bool appendLoopSources(scf::ForOp loop, unsigned index, SmallVectorImpl<Value>& pending)
{
    const bool missingLoopSource =
        loop.getBody()->empty() || index >= loop.getInitArgs().size() || index >= loop.getYieldedValues().size();
    if (missingLoopSource) {
        return false;
    }
    pending.push_back(loop.getInitArgs()[index]);
    pending.push_back(loop.getYieldedValues()[index]);
    return true;
}

bool appendBlockArgumentSources(
    func::FuncOp function, BlockArgument argument, SmallVectorImpl<Value>& pending, SmallVectorImpl<Value>& roots)
{
    const bool isEntryArgument = !function.getBody().empty() && argument.getOwner() == &function.getBody().front();
    if (isEntryArgument) {
        if (!isa<PtrType, TensorViewType, PartitionTensorViewType>(argument.getType())) {
            return false;
        }
        roots.push_back(argument);
        return true;
    }
    auto loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
    const bool isUnsupportedArgument = !loop || argument.getArgNumber() == 0 || argument.getOwner() != loop.getBody();
    if (isUnsupportedArgument) {
        return false;
    }
    return appendLoopSources(loop, argument.getArgNumber() - 1, pending);
}

bool appendResultSources(OpResult result, SmallVectorImpl<Value>& pending)
{
    Operation* operation = result.getOwner();
    const bool leavesPointerDomain = isa<CastPtrOp>(operation) && (operation->getNumOperands() != 1 ||
                                                                   !isa<PtrType>(operation->getOperand(0).getType()) ||
                                                                   !isa<PtrType>(result.getType()));
    if (leavesPointerDomain) {
        return false;
    }
    if (isa<AddPtrOp, CastPtrOp, MakeTensorViewOp, PartitionViewOp, SubViewOp, TReshapeOp, BitcastOp>(operation)) {
        const bool missingSource = operation->getNumOperands() == 0;
        if (missingSource) {
            return false;
        }
        pending.push_back(operation->getOperand(0));
        return true;
    }
    if (auto select = dyn_cast<arith::SelectOp>(operation)) {
        pending.push_back(select.getTrueValue());
        pending.push_back(select.getFalseValue());
        return true;
    }
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
        return appendLoopSources(loop, result.getResultNumber(), pending);
    }
    if (auto choice = dyn_cast<scf::IfOp>(operation)) {
        const bool missingAlternative = choice.getThenRegion().empty() || choice.getElseRegion().empty();
        if (missingAlternative) {
            return false;
        }
        for (Region* region : {&choice.getThenRegion(), &choice.getElseRegion()}) {
            auto yield = dyn_cast<scf::YieldOp>(region->front().getTerminator());
            const bool missingYield = !yield || result.getResultNumber() >= yield.getResults().size();
            if (missingYield) {
                return false;
            }
            pending.push_back(yield.getResults()[result.getResultNumber()]);
        }
        return true;
    }
    return false;
}

} // namespace

std::optional<SyncGMAliasMode> mlir::pto::protocol_sync::parseSyncGMAliasMode(StringRef value)
{
    if (value == "may-alias") {
        return SyncGMAliasMode::MayAlias;
    }
    if (value == "assume-disjoint-arguments") {
        return SyncGMAliasMode::AssumeDisjointArguments;
    }
    return std::nullopt;
}

StringRef mlir::pto::protocol_sync::stringifySyncGMAliasMode(SyncGMAliasMode mode)
{
    return mode == SyncGMAliasMode::AssumeDisjointArguments ? "assume-disjoint-arguments" : "may-alias";
}

FailureOr<SyncGMAliasMode> mlir::pto::protocol_sync::resolveSyncGMAliasMode(
    func::FuncOp function, std::optional<SyncGMAliasMode> overrideMode)
{
    if (!function) {
        return failure();
    }
    SyncGMAliasMode recorded = SyncGMAliasMode::MayAlias;
    if (Attribute attribute = function->getAttr(kSyncGMAliasContract)) {
        auto text = dyn_cast<StringAttr>(attribute);
        std::optional<SyncGMAliasMode> parsed = text ? parseSyncGMAliasMode(text.getValue()) : std::nullopt;
        if (!parsed) {
            function.emitError("pto.gm_alias must be 'may-alias' or 'assume-disjoint-arguments'");
            return failure();
        }
        recorded = *parsed;
    }
    return overrideMode.value_or(recorded);
}

SyncGMArgumentRoots mlir::pto::protocol_sync::traceSyncGMArgumentRoots(func::FuncOp function, Value value)
{
    SyncGMArgumentRoots result;
    if (!function) {
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
        const bool exceededTraceLimit = visited.size() > kMaximumRootTraceValues;
        if (exceededTraceLimit) {
            return result;
        }
        if (auto argument = dyn_cast<BlockArgument>(current)) {
            if (!appendBlockArgumentSources(function, argument, pending, result.roots)) {
                return result;
            }
        } else if (auto definition = dyn_cast<OpResult>(current)) {
            if (!appendResultSources(definition, pending)) {
                return result;
            }
        } else {
            return result;
        }
    }
    llvm::sort(result.roots, [](Value first, Value second) {
        return cast<BlockArgument>(first).getArgNumber() < cast<BlockArgument>(second).getArgNumber();
    });
    result.complete = !result.roots.empty();
    return result;
}

bool mlir::pto::protocol_sync::haveDisjointGMArgumentRoots(
    const StructuredSyncIR& schedule, const SyncAccess& first, const SyncAccess& second)
{
    const bool outsideContract = schedule.getGMAliasMode() != SyncGMAliasMode::AssumeDisjointArguments ||
                                 first.storage.space != AddressSpace::GM || second.storage.space != AddressSpace::GM;
    if (outsideContract) {
        return false;
    }
    const SyncGMArgumentRoots left = traceSyncGMArgumentRoots(schedule.getFunction(), first.value);
    const SyncGMArgumentRoots right = traceSyncGMArgumentRoots(schedule.getFunction(), second.value);
    return left.complete && right.complete &&
           llvm::none_of(left.roots, [&](Value root) { return llvm::is_contained(right.roots, root); });
}
