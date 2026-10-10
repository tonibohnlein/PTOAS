// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SyncScalarConstraints.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
using namespace mlir;
namespace mlir::pto::detail {
namespace {
class Collector {
    unsigned indexBits;
    ScalarConstraints facts;
    bool contradictory = false;
    llvm::DenseSet<std::pair<Value, unsigned>> visited;

    void comparison(arith::CmpIOp cmp, bool truth)
    {
        Value value = cmp.getLhs();
        APInt constant;
        auto predicate = cmp.getPredicate();
        if (!matchPattern(cmp.getRhs(), m_ConstantInt(&constant))) {
            if (!matchPattern(cmp.getLhs(), m_ConstantInt(&constant))) {
                return;
            }
            value = cmp.getRhs();
            switch (predicate) {
                case arith::CmpIPredicate::slt:
                    predicate = arith::CmpIPredicate::sgt;
                    break;
                case arith::CmpIPredicate::sle:
                    predicate = arith::CmpIPredicate::sge;
                    break;
                case arith::CmpIPredicate::sgt:
                    predicate = arith::CmpIPredicate::slt;
                    break;
                case arith::CmpIPredicate::sge:
                    predicate = arith::CmpIPredicate::sle;
                    break;
                case arith::CmpIPredicate::ult:
                    predicate = arith::CmpIPredicate::ugt;
                    break;
                case arith::CmpIPredicate::ule:
                    predicate = arith::CmpIPredicate::uge;
                    break;
                case arith::CmpIPredicate::ugt:
                    predicate = arith::CmpIPredicate::ult;
                    break;
                case arith::CmpIPredicate::uge:
                    predicate = arith::CmpIPredicate::ule;
                    break;
                default:
                    break;
            }
        }
        if (!truth) {
            predicate = arith::invertPredicate(predicate);
        }
        unsigned bits = value.getType().isIndex() ?
                            indexBits :
                            (isa<IntegerType>(value.getType()) ? cast<IntegerType>(value.getType()).getWidth() : 0);
        if (!bits || bits > 64 || !constant.isSignedIntN(bits)) {
            return;
        }
        constant = constant.sextOrTrunc(bits);
        APInt low = APInt::getSignedMinValue(bits), high = APInt::getSignedMaxValue(bits);
        bool isUnsigned = false, empty = false;
        // Strict adjustments occur only after checking the corresponding
        // machine endpoint. No host signed increment can overflow.
        switch (predicate) {
            case arith::CmpIPredicate::eq:
                low = high = constant;
                break;
            case arith::CmpIPredicate::ne:
                return;
            case arith::CmpIPredicate::slt:
                empty = constant == low;
                high = constant - 1;
                break;
            case arith::CmpIPredicate::sle:
                high = constant;
                break;
            case arith::CmpIPredicate::sgt:
                empty = constant == high;
                low = constant + 1;
                break;
            case arith::CmpIPredicate::sge:
                low = constant;
                break;
            case arith::CmpIPredicate::ult:
            case arith::CmpIPredicate::ule:
                isUnsigned = true;
                low = APInt::getZero(bits);
                empty = predicate == arith::CmpIPredicate::ult && constant.isZero();
                high = predicate == arith::CmpIPredicate::ult ? constant - 1 : constant;
                break;
            case arith::CmpIPredicate::ugt:
            case arith::CmpIPredicate::uge:
                isUnsigned = true;
                high = APInt::getAllOnes(bits);
                empty = predicate == arith::CmpIPredicate::ugt && constant.isAllOnes();
                low = predicate == arith::CmpIPredicate::ugt ? constant + 1 : constant;
                break;
        }
        if (empty) {
            contradictory = true;
            return;
        }
        // An unsigned interval spanning the sign bit has the full signed
        // hull. In particular x >=u 1 supplies no nonnegative signed bound.
        if (isUnsigned && low.isNegative() != high.isNegative()) {
            return;
        }
        auto interval = std::make_pair(low.getSExtValue(), high.getSExtValue());
        auto [entry, inserted] = facts.try_emplace(value, interval);
        if (!inserted) {
            entry->second.first = std::max(entry->second.first, interval.first);
            entry->second.second = std::min(entry->second.second, interval.second);
            contradictory |= entry->second.first > entry->second.second;
        }
    }
    void condition(Value value, bool truth)
    {
        SmallVector<std::pair<Value, bool>> work{{value, truth}};
        while (!work.empty()) {
            auto [current, required] = work.pop_back_val();
            if (!visited.insert({current, required}).second) {
                continue;
            }
            if (auto cmp = current.getDefiningOp<arith::CmpIOp>()) {
                comparison(cmp, required);
            } else if (
                (required && current.getDefiningOp<arith::AndIOp>()) ||
                (!required && current.getDefiningOp<arith::OrIOp>())) {
                // Conditions are i1. A true conjunction and a false
                // disjunction require both operands; the other cases do not.
                for (Value operand : current.getDefiningOp()->getOperands()) {
                    work.emplace_back(operand, required);
                }
            }
        }
    }

public:
    explicit Collector(unsigned indexBits) : indexBits(indexBits) {}
    ScalarConstraints collect(Operation* anchor)
    {
        // Each child identifies the exact containing arm. Starting at the
        // parent excludes the if's own condition, sibling arms and later users.
        for (Operation* child = anchor; child && child->getParentOp(); child = child->getParentOp()) {
            if (auto branch = dyn_cast<scf::IfOp>(child->getParentOp())) {
                auto* region = child->getParentRegion();
                if (region == &branch.getThenRegion()) {
                    condition(branch.getCondition(), true);
                } else if (region == &branch.getElseRegion()) {
                    condition(branch.getCondition(), false);
                }
            }
        }
        // Do not exploit unreachable paths to invent expressions. Discard all
        // refinements if these supported facts contradict one another.
        return contradictory ? ScalarConstraints{} : std::move(facts);
    }
};
} // namespace
ScalarConstraints enclosingScalarConstraints(Operation* anchor, unsigned indexBits)
{
    return Collector(indexBits).collect(anchor);
}
} // namespace mlir::pto::detail
