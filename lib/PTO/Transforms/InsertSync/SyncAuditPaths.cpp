// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Independent concrete control expansion. The planner's loop copies and guard
// classes are deliberately not consulted. A path keeps immutable predicate
// choices; loop-local predicates are fresh at each dynamic iteration.
#include "PTO/Transforms/InsertSync/SyncAuditPaths.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <optional>
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
constexpr unsigned kMaximumPaths = 64;
constexpr unsigned kMaximumActions = 4096;
constexpr unsigned kMaximumTrips = 16;
constexpr unsigned kMaximumDepth = 16;
struct Path {
    SmallVector<Operation*> operations;
    llvm::DenseMap<Value, bool> predicates;
    llvm::DenseMap<Value, APInt> scalarValues;
    bool feasible = true;
};

class Expander {
public:
    InsertSyncAuditPaths run(func::FuncOp function)
    {
        SmallVector<Path> paths(1);
        if (expand(function.getBody(), paths, 0)) {
            for (auto& path : paths) {
                result.pathFeasible.push_back(path.feasible);
                result.paths.push_back(std::move(path.operations));
            }
        }
        return std::move(result);
    }

private:
    bool fail(Operation* op, StringRef reason)
    {
        result.witness = op;
        result.unsupported = reason.str();
        return false;
    }


    // Literal-loop audit only. PTOAS's A2/A3 index lowering is 64-bit.
    // Unhandled or overflowing index expressions remain unknown. Integer
    // arithmetic retains its explicit bit width; no range is fabricated.
    std::optional<APInt> scalar(Value value, const Path& path, unsigned depth = 0)
    {
        if (!value || depth > 32) {
            return std::nullopt;
        }
        auto known = path.scalarValues.find(value);
        if (known != path.scalarValues.end()) {
            return known->second;
        }
        auto selected = path.predicates.find(value);
        if (selected != path.predicates.end()) {
            return APInt(1, selected->second);
        }
        IntegerAttr literal;
        if (matchPattern(value, m_Constant(&literal))) {
            return literal.getValue();
        }
        Operation* op = value.getDefiningOp();
        if (!op || op->getNumRegions()) {
            return std::nullopt;
        }
        if (auto cast = dyn_cast<arith::IndexCastOp>(op)) {
            auto input = scalar(cast.getIn(), path, depth + 1);
            if (!input) {
                return std::nullopt;
            }
            unsigned width = 64;
            if (auto type = dyn_cast<IntegerType>(value.getType())) {
                width = type.getWidth();
            } else if (!isa<IndexType>(value.getType())) {
                return std::nullopt;
            }
            return input->sextOrTrunc(width);
        }
        if (auto select = dyn_cast<arith::SelectOp>(op)) {
            auto condition = scalar(select.getCondition(), path, depth + 1);
            if (!condition) {
                return std::nullopt;
            }
            return scalar(condition->isZero() ? select.getFalseValue() : select.getTrueValue(), path, depth + 1);
        }
        if (op->getNumOperands() != 2) {
            return std::nullopt;
        }
        auto left = scalar(op->getOperand(0), path, depth + 1);
        auto right = scalar(op->getOperand(1), path, depth + 1);
        if (!left || !right || left->getBitWidth() != right->getBitWidth()) {
            return std::nullopt;
        }
        if (auto compare = dyn_cast<arith::CmpIOp>(op)) {
            bool answer = false;
            switch (compare.getPredicate()) {
                case arith::CmpIPredicate::eq: answer = *left == *right; break;
                case arith::CmpIPredicate::ne: answer = *left != *right; break;
                case arith::CmpIPredicate::slt: answer = left->slt(*right); break;
                case arith::CmpIPredicate::sle: answer = left->sle(*right); break;
                case arith::CmpIPredicate::sgt: answer = left->sgt(*right); break;
                case arith::CmpIPredicate::sge: answer = left->sge(*right); break;
                case arith::CmpIPredicate::ult: answer = left->ult(*right); break;
                case arith::CmpIPredicate::ule: answer = left->ule(*right); break;
                case arith::CmpIPredicate::ugt: answer = left->ugt(*right); break;
                case arith::CmpIPredicate::uge: answer = left->uge(*right); break;
            }
            return APInt(1, answer);
        }
        bool overflow = false;
        std::optional<APInt> answer;
        if (isa<arith::AddIOp>(op)) {
            answer = left->sadd_ov(*right, overflow);
        } else if (isa<arith::SubIOp>(op)) {
            answer = left->ssub_ov(*right, overflow);
        } else if (isa<arith::MulIOp>(op)) {
            answer = left->smul_ov(*right, overflow);
        } else if (isa<arith::AndIOp>(op)) {
            answer = *left & *right;
        } else if (isa<arith::OrIOp>(op)) {
            answer = *left | *right;
        } else if (isa<arith::XOrIOp>(op)) {
            answer = *left ^ *right;
        } else if (isa<arith::RemUIOp>(op) && !right->isZero()) {
            answer = left->urem(*right);
        }
        // Also leave signed overflow unknown for integer ops: this avoids
        // assuming wrapping when the op carries a poison-producing flag.
        if (overflow) {
            return std::nullopt;
        }
        return answer;
    }

    bool freeBooleanInput(Value value)
    {
        auto argument = dyn_cast<BlockArgument>(value);
        return argument && isa<func::FuncOp>(argument.getOwner()->getParentOp()) && value.getType().isInteger(1);
    }

    bool choice(scf::IfOp op, SmallVectorImpl<Path>& paths, unsigned depth)
    {
        if (op.getNumResults()) {
            return fail(op, "result-bearing choice requires value interpretation");
        }
        SmallVector<Path> alternatives;
        for (const Path& path : paths) {
            auto fixed = scalar(op.getCondition(), path);
            auto previous = path.predicates.find(op.getCondition());
            for (bool takeThen : {true, false}) {
                if ((fixed && takeThen != !fixed->isZero()) ||
                    (previous != path.predicates.end() && takeThen != previous->second)) {
                    continue;
                }
                SmallVector<Path> arm{path};
                if (!fixed && previous == path.predicates.end() && !freeBooleanInput(op.getCondition())) {
                    arm.front().feasible = false;
                }
                arm.front().predicates[op.getCondition()] = takeThen;
                Region& region = takeThen ? op.getThenRegion() : op.getElseRegion();
                if (!region.empty() && !expand(region, arm, depth + 1)) {
                    return false;
                }
                if (alternatives.size() + arm.size() > kMaximumPaths) {
                    return fail(op, "audit path budget exceeded");
                }
                for (auto& entry : arm) {
                    alternatives.push_back(std::move(entry));
                }
            }
        }
        paths.assign(std::make_move_iterator(alternatives.begin()), std::make_move_iterator(alternatives.end()));
        return true;
    }

    bool loop(scf::ForOp op, SmallVectorImpl<Path>& paths, unsigned depth)
    {
        IntegerAttr lower, upper, step;
        if (!op.getInitArgs().empty() || !matchPattern(op.getLowerBound(), m_Constant(&lower)) ||
            !matchPattern(op.getUpperBound(), m_Constant(&upper)) || !matchPattern(op.getStep(), m_Constant(&step)) ||
            lower.getValue().getBitWidth() > 64 || upper.getValue().getBitWidth() > 64 ||
            step.getValue().getBitWidth() > 64 || !step.getValue().isStrictlyPositive()) {
            return fail(op, "symbolic loop requires an inductive certificate; finite sampling is not verification");
        }
        auto distance = upper.getValue().sextOrTrunc(128) - lower.getValue().sextOrTrunc(128);
        auto increment = step.getValue().sextOrTrunc(128);
        uint64_t trips = 0;
        if (distance.isStrictlyPositive()) {
            auto count = (distance + increment - 1).udiv(increment);
            if (count.ugt(kMaximumTrips)) {
                return fail(op, "audit finite-trip budget exceeded");
            }
            trips = count.getZExtValue();
        }
        for (uint64_t iteration = 0; iteration < trips; ++iteration) {
            for (auto& path : paths) {
                llvm::SmallVector<Value> localPredicates;
                for (auto [value, selected] : path.predicates) {
                    (void)selected;
                    if (op.getRegion().isAncestor(value.getParentRegion())) {
                        localPredicates.push_back(value);
                    }
                }
                for (Value value : localPredicates) {
                    path.predicates.erase(value);
                }
                SmallVector<Value> localValues;
                for (const auto& entry : path.scalarValues) {
                    Value value = entry.first;
                    if (op.getRegion().isAncestor(value.getParentRegion())) {
                        localValues.push_back(value);
                    }
                }
                for (Value value : localValues) {
                    path.scalarValues.erase(value);
                }
                APInt current = lower.getValue().sextOrTrunc(128) + APInt(128, iteration) * increment;
                if (!current.isSignedIntN(64)) {
                    return fail(op, "audit induction value exceeds supported index range");
                }
                path.scalarValues[op.getInductionVar()] = current.trunc(64);
            }
            if (!expand(op.getRegion(), paths, depth + 1)) {
                return false;
            }
        }
        return true;
    }

    bool expand(Region& region, SmallVectorImpl<Path>& paths, unsigned depth)
    {
        if (depth > kMaximumDepth || !llvm::hasSingleElement(region)) {
            return fail(region.getParentOp(), "audit region/depth budget");
        }
        for (Operation& operation : region.front()) {
            if (auto op = dyn_cast<scf::IfOp>(operation)) {
                if (!choice(op, paths, depth)) {
                    return false;
                }
            } else if (auto op = dyn_cast<scf::ForOp>(operation)) {
                if (!loop(op, paths, depth)) {
                    return false;
                }
            } else if (operation.getNumRegions()) {
                return fail(&operation, "unsupported structured operation");
            } else if (!isa<scf::YieldOp>(operation)) {
                for (auto& path : paths) {
                    if (path.operations.size() >= kMaximumActions) {
                        return fail(&operation, "audit action budget exceeded");
                    }
                    path.operations.push_back(&operation);
                }
            }
        }
        return true;
    }

    InsertSyncAuditPaths result;
};
} // namespace

InsertSyncAuditPaths mlir::pto::buildInsertSyncAuditPaths(func::FuncOp function) { return Expander().run(function); }
