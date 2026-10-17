// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Diagnostics over actual SSA bindings, including enclosing command conditions.
// Region results follow both yielded alternatives structurally; dynamic accounting
// follows only the executed arm in the test interpreter.
#ifndef PTO_FRONTIER_CONTROLSSAPROFILE_H
#define PTO_FRONTIER_CONTROLSSAPROFILE_H
#include "ControlRootProfile.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
namespace mlir::pto::frontiersynch::detail {
class ControlSSAProfile {
public:
    static SmallVector<Value> dependencies(Value value)
    {
        SmallVector<Value> operands;
        auto* operation = value.getDefiningOp();
        if (!operation) {
            return operands;
        }
        if (operation->getName().getDialectNamespace() == "arith") {
            llvm::append_range(operands, operation->getOperands());
        } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
            operands.push_back(branch.getCondition());
            const auto index = cast<OpResult>(value).getResultNumber();
            for (auto& region : branch->getRegions()) {
                if (!region.empty()) {
                    operands.push_back(region.front().getTerminator()->getOperand(index));
                }
            }
        }
        return operands;
    }
    static bool arithmetic(Value value)
    {
        auto* operation = value.getDefiningOp();
        return operation && operation->getName().getDialectNamespace() == "arith";
    }
    // Potential control roles seed executed eager arithmetic even if the
    // eventual command is inactive. Cycle-safe propagation through loop results
    // and carried arguments is separate from the acyclic static DAG report.
    llvm::DenseMap<Operation*, unsigned> operationRoles;
    void potentialRoles(SmallVector<std::pair<Value, unsigned>> pending)
    {
        llvm::DenseMap<Value, unsigned> seen;
        while (!pending.empty()) {
            const auto [value, role] = pending.pop_back_val();
            if (seen[value] & role) {
                continue;
            }
            seen[value] |= role;
            if (arithmetic(value)) {
                operationRoles[value.getDefiningOp()] |= role;
            }
            auto operands = dependencies(value);
            scf::ForOp loop;
            std::optional<unsigned> carried;
            if (auto argument = dyn_cast<BlockArgument>(value)) {
                loop = dyn_cast_or_null<scf::ForOp>(argument.getOwner()->getParentOp());
                if (loop && argument.getArgNumber()) {
                    carried = argument.getArgNumber() - 1;
                }
            } else if ((loop = value.getDefiningOp<scf::ForOp>())) {
                carried = cast<OpResult>(value).getResultNumber();
            }
            if (loop) {
                operands.append({loop.getLowerBound(), loop.getUpperBound(), loop.getStep()});
                if (carried) {
                    operands.push_back(loop.getInitArgs()[*carried]);
                    operands.push_back(loop.getBody()->getTerminator()->getOperand(*carried));
                }
            }
            for (auto operand : operands) {
                pending.emplace_back(operand, role);
            }
        }
    }
    llvm::DenseMap<Operation*, unsigned> commandBranches;
    llvm::json::Object result;
    explicit ControlSSAProfile(func::FuncOp function)
    {
        ControlRootProfile<Value> profile;
        SmallVector<std::pair<Value, unsigned>> roots;
        auto add = [&](Value value, unsigned role) {
            profile.add(value, role);
            roots.emplace_back(value, 1U << role);
        };
        function.walk([&](Operation* command) {
            const auto name = command->getName().getStringRef();
            const bool logical = name == "pto.logical_set" || name == "pto.logical_wait";
            const bool physical = name == "pto.set_flag" || name == "pto.wait_flag" || name == "pto.set_flag_dyn" ||
                                  name == "pto.wait_flag_dyn";
            if (!logical && !physical && name != "pto.barrier") {
                return;
            }
            for (auto value : command->getOperands()) {
                add(value, physical ? 2 : 1);
            }
            auto* child = command;
            for (auto* parent = child->getParentOp(); parent && parent != function;
                 child = parent, parent = parent->getParentOp()) {
                if (auto branch = dyn_cast<scf::IfOp>(parent)) {
                    add(branch.getCondition(), 0);
                    commandBranches[parent] |= child->getParentRegion() == &branch.getThenRegion() ? 1 : 2;
                } else if (isa<scf::ForOp>(parent)) {
                    commandBranches[parent] = 1;
                }
            }
        });
        potentialRoles(std::move(roots));
        result = profile.report(dependencies, arithmetic);
        result["scope"] = "arith-ssa-including-original-dependencies-loop-arguments-are-boundaries";
    }
};
} // namespace mlir::pto::frontiersynch::detail
#endif
