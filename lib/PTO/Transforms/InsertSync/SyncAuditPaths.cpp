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
};

class Expander {
public:
    InsertSyncAuditPaths run(func::FuncOp function)
    {
        SmallVector<Path> paths(1);
        if (expand(function.getBody(), paths, 0)) {
            for (auto& path : paths) {
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

    bool choice(scf::IfOp op, SmallVectorImpl<Path>& paths, unsigned depth)
    {
        if (op.getNumResults()) {
            return fail(op, "result-bearing choice requires value interpretation");
        }
        SmallVector<Path> alternatives;
        IntegerAttr constant;
        bool fixed = matchPattern(op.getCondition(), m_Constant(&constant));
        for (const Path& path : paths) {
            auto previous = path.predicates.find(op.getCondition());
            for (bool takeThen : {true, false}) {
                if ((fixed && takeThen != !constant.getValue().isZero()) ||
                    (previous != path.predicates.end() && takeThen != previous->second)) {
                    continue;
                }
                SmallVector<Path> arm{path};
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
