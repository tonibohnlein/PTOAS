// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Optional execution-instance accounting for finite control diagnostics. SSA
// rebindings in loops point to fresh instances; shared inactive-test dependencies
// are charged once, and skipped branch computations create no instance.
#ifndef PTO_TEST_CONTROL_EVALUATION_CHECKS_H
#define PTO_TEST_CONTROL_EVALUATION_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/ControlSSAProfile.h"
#include "llvm/ADT/DenseSet.h"
namespace finite_control_checks {
class Evaluation {
    struct Node {
        bool arithmetic;
        unsigned roles;
        mlir::SmallVector<uint64_t, 3> operands;
    };
    mlir::pto::frontiersynch::detail::ControlSSAProfile profile;
    llvm::DenseMap<mlir::Value, uint64_t> bindings;
    std::vector<Node> nodes{{false, 0, {}}};
    llvm::DenseSet<uint64_t> inactive;
    uint64_t tests = 0, falseTests = 0, inactiveTests = 0;

public:
    mlir::SmallVector<uint64_t> yielded;
    explicit Evaluation(mlir::func::FuncOp function) : profile(function) {}
    mlir::SmallVector<uint64_t> inputs(mlir::ValueRange values) const
    {
        mlir::SmallVector<uint64_t> result;
        for (auto value : values) {
            result.push_back(bindings.lookup(value));
        }
        return result;
    }
    void bind(mlir::ValueRange values, llvm::ArrayRef<uint64_t> ids)
    {
        for (auto [value, id] : llvm::zip(values, ids)) {
            bindings[value] = id;
        }
    }
    uint64_t join(llvm::ArrayRef<uint64_t> operands)
    {
        Node node{false, 0, {}};
        llvm::append_range(node.operands, operands);
        const auto id = nodes.size();
        nodes.push_back(std::move(node));
        return id;
    }
    void mark(uint64_t id, unsigned role)
    {
        mlir::SmallVector<uint64_t> pending{id};
        while (!pending.empty()) {
            const auto current = pending.pop_back_val();
            if (!current || (nodes[current].roles & role)) {
                continue;
            }
            nodes[current].roles |= role;
            llvm::append_range(pending, nodes[current].operands);
        }
    }
    void operation(mlir::Operation& operation)
    {
        if (operation.getName().getDialectNamespace() != "arith") {
            return;
        }
        Node node{true, 0, {}};
        llvm::append_range(node.operands, inputs(operation.getOperands()));
        const auto id = nodes.size();
        nodes.push_back(std::move(node));
        for (auto value : operation.getResults()) {
            bindings[value] = id;
        }
        const auto roles = profile.operationRoles.lookup(&operation);
        for (unsigned role = 0; role < 3; ++role) {
            if (roles & (1U << role)) {
                mark(id, 1U << role);
            }
        }
    }
    void command(mlir::Operation& operation, unsigned role)
    {
        for (auto id : inputs(operation.getOperands())) {
            mark(id, 1U << role);
        }
    }
    void branchResults(mlir::scf::IfOp branch)
    {
        const auto condition = bindings.lookup(branch.getCondition());
        for (auto [value, selected] : llvm::zip(branch.getResults(), yielded)) {
            const uint64_t dependencies[]{condition, selected};
            bindings[value] = join(dependencies);
        }
    }
    void loopEnter(mlir::scf::ForOp loop)
    {
        bindings.erase(loop.getInductionVar());
        const mlir::Value bounds[]{loop.getLowerBound(), loop.getUpperBound(), loop.getStep()};
        if (profile.commandBranches.count(loop.getOperation())) {
            for (auto id : inputs(bounds)) {
                mark(id, 1);
            }
        }
    }
    void loopResults(mlir::scf::ForOp loop, llvm::ArrayRef<uint64_t> state)
    {
        const mlir::Value bounds[]{loop.getLowerBound(), loop.getUpperBound(), loop.getStep()};
        for (auto [value, selected] : llvm::zip(loop.getResults(), state)) {
            auto dependencies = inputs(bounds);
            dependencies.push_back(selected);
            bindings[value] = join(dependencies);
        }
    }
    void induction(mlir::scf::ForOp loop)
    {
        const mlir::Value values[]{loop.getLowerBound(), loop.getUpperBound(), loop.getStep(), loop.getInductionVar()};
        bindings[loop.getInductionVar()] = join(inputs(values));
    }
    void branch(mlir::scf::IfOp branch, bool condition)
    {
        const auto arms = profile.commandBranches.lookup(branch.getOperation());
        if (!arms) {
            return;
        }
        mark(bindings.lookup(branch.getCondition()), 1);
        ++tests;
        if (!condition) {
            ++falseTests;
        }
        const auto absent = condition ? 2U : 1U;
        if (!(arms & absent)) {
            return;
        }
        ++inactiveTests;
        mlir::SmallVector<uint64_t> pending{bindings.lookup(branch.getCondition())};
        while (!pending.empty()) {
            const auto id = pending.pop_back_val();
            if (!id || !inactive.insert(id).second) {
                continue;
            }
            llvm::append_range(pending, nodes[id].operands);
        }
    }
    llvm::json::Object report()
    {
        std::array<uint64_t, 8> visits{};
        uint64_t inactiveArithmetic = 0;
        for (const auto& node : nodes) {
            if (node.arithmetic) {
                ++visits[node.roles];
            }
        }
        for (auto id : inactive) {
            inactiveArithmetic += nodes[id].arithmetic;
        }
        llvm::json::Array categories;
        uint64_t total = 0;
        for (unsigned mask = 1; mask < visits.size(); ++mask) {
            total += visits[mask];
            categories.push_back(llvm::json::Object{{"role_mask", mask}, {"evaluations", visits[mask]}});
        }
        return llvm::json::Object{
            {"static", std::move(profile.result)},
            {"evaluated_union", total},
            {"unattributed_arithmetic_evaluations", visits[0]},
            {"evaluated_roles", std::move(categories)},
            {"command_guard_tests", tests},
            {"false_command_guard_tests", falseTests},
            {"inactive_command_arm_tests", inactiveTests},
            {"inactive_test_dependency_evaluations", inactiveArithmetic},
            {"scope", "executed-arithmetic-including-original-dependencies"}};
    }
};
} // namespace finite_control_checks
#endif
