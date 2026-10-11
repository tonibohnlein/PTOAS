// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent expected program-point facts and query-work checks.
#include "PTO/Transforms/InsertSync/SyncDescriptorState.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/InsertSync/SyncScalarFacts.h"
#include "../../lib/PTO/Transforms/InsertSync/SyncScalarEvolution.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/raw_ostream.h"
#include <climits>
using namespace mlir;
LogicalResult checkDescriptorState(func::FuncOp function)
{
    pto::SyncDescriptorState facts(function);
    pto::SyncInput input;
    if (failed(input.build(function, pto::SyncInstructionView::PipeEnvelopes))) {
        return failure();
    }
    auto retained = input.descriptorState();
    if (!retained) {
        return failure();
    }
    unsigned checked = 0;
    bool good = true;
    function.walk([&](Operation* op) {
        auto expected = op->getAttrOfType<DenseI64ArrayAttr>("test.shape");
        if (!expected) {
            return;
        }
        if (op->getNumOperands() == 0 || expected.size() != 2) {
            good = false;
            return;
        }
        auto shape = facts.constantShape(op->getOperand(0), op);
        if (retained->constantShape(op->getOperand(0), op) != shape) {
            op->emitError("shared input and independent reaching-state query disagree");
            good = false;
        }
        const bool unknown = expected[0] < 0;
        if (unknown ? shape.has_value() : (!shape || (*shape)[0] != expected[0] || (*shape)[1] != expected[1])) {
            op->emitError("reaching descriptor shape differs from independent expectation");
            good = false;
        }
        auto work = facts.work();
        for (unsigned repeat = 0; repeat < 64; ++repeat) {
            if (facts.constantShape(op->getOperand(0), op) != shape) {
                good = false;
            }
        }
        auto after = facts.work();
        if (after.sourceVisits != work.sourceVisits || after.lookups != work.lookups + 64) {
            good = false;
        }
        ++checked;
    });
    if (auto limit = function->getAttrOfType<IntegerAttr>("test.max_descriptor_visits")) {
        if (limit.getInt() < 0 || facts.work().sourceVisits > static_cast<uint64_t>(limit.getInt())) {
            function.emitError("descriptor traversal exceeded independently bounded nested-work family");
            good = false;
        }
    }
    pto::SyncScalarFacts sharedScalars;
    DenseMap<Value, unsigned> identities;
    function.walk([&](Operation* op) {
        for (auto value : op->getOperands()) {
            identities.try_emplace(value, identities.size());
        }
        for (auto value : op->getResults()) {
            identities.try_emplace(value, identities.size());
        }
    });
    function.walk([&](Operation* op) {
        for (auto value : op->getOperands()) {
            if (!value.getType().isIntOrIndex()) {
                continue;
            }
            for (unsigned numbering = 0; numbering < 2; ++numbering) {
                auto bind = [&](Value original) {
                    auto id = identities.find(original);
                    if (id == identities.end() || id->second > (UINT_MAX - 1) / 2) {
                        return AffineExpr{};
                    }
                    return getAffineSymbolExpr(numbering ? id->second * 2 + 1 : id->second, function.getContext());
                };
                pto::detail::ScalarEvolution independent(function.getContext(), op);
                if (sharedScalars.value(value, op, bind) != independent.value(value, bind)) {
                    op->emitError("shared scalar cache differs from fresh context or caller namespace");
                    good = false;
                }
            }
        }
    });
    if (!checked || !good) {
        return failure();
    }
    if (failed(input.build(function, pto::SyncInstructionView::PipeEnvelopes)) || input.descriptorState() == retained) {
        return failure();
    }
    function.walk([&](Operation* op) {
        if (op->hasAttr("test.shape") && retained->constantShape(op->getOperand(0), op) !=
                                             input.descriptorState()->constantShape(op->getOperand(0), op)) {
            good = false;
        }
    });
    // A rebuilt owner cannot answer another function's program points.
    pto::SyncDescriptorState empty(func::FuncOp{});
    function.walk([&](pto::TGetValOp op) {
        if (empty.before(op->getOperand(0), op)) {
            good = false;
        }
    });
    if (!good) {
        return failure();
    }
    llvm::outs() << "descriptor-state: " << function.getSymName() << " checked=" << checked
                 << " visits=" << facts.work().sourceVisits << " stable-lookups\n";
    return success();
}
