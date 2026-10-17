// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Independent value and accounting checks for the finite threshold projection.
#ifndef PTO_COMPACT_FINITE_CONTROL_CHECKS_H
#define PTO_COMPACT_FINITE_CONTROL_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/FixedRankThresholds.h"
#include "ControlEvaluationChecks.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RegionConstruction.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
namespace compactFiniteControlChecks {
namespace fs = mlir::pto::frontiersynch;
inline bool roots()
{
    fs::detail::ControlRootProfile<unsigned> profile;
    // Seventy distinct guard roots all reach node 1; duplicate uses must not
    // inflate distinct roots, and a selector reuses the same shared node.
    for (unsigned i = 2; i < 72; ++i) {
        profile.add(i, 0);
        profile.add(i, 0);
    }
    profile.add(1, 1);
    auto report = profile.report(
        [](unsigned i) { return i == 1 ? llvm::SmallVector<unsigned>{} : llvm::SmallVector<unsigned>{1}; },
        [](unsigned) { return true; });
    auto* categories = report.getArray("categories");
    if (!categories || report.getInteger("root_uses") != 141 || report.getInteger("union_nodes") != 71 ||
        report.getInteger("shared_nodes") != 1 || report.getInteger("shared_guard_nodes") != 1 ||
        report.getInteger("root_batches") != 2 || report.getInteger("scratch_node_words") != 71 ||
        report.getInteger("root_memberships") != 141 ||
        (*categories)[0].getAsObject()->getInteger("largest_root") != 2 ||
        (*categories)[0].getAsObject()->getInteger("roots") != 70) {
        return false;
    }
    fs::detail::ControlRootProfile<unsigned> cyclic;
    cyclic.add(1, 0);
    auto invalid =
        cyclic.report([](unsigned) { return llvm::SmallVector<unsigned>{1}; }, [](unsigned) { return true; });
    return bool(invalid.getString("error"));
}
inline bool thresholds(mlir::MLIRContext& context)
{
    mlir::Block block;
    fs::RegionExpressions e;
    fs::GuardedRankProjectionWork work;
    fs::detail::FixedRankThresholds projection(e, work);
    auto input = [&](mlir::Type type) { return e.input(block.addArgument(type, mlir::UnknownLoc::get(&context))); };
    const auto p = input(mlir::IntegerType::get(&context, 1));
    const auto q = input(mlir::IntegerType::get(&context, 1));
    const auto opaque = input(mlir::IndexType::get(&context));
    const auto zero = e.constant(0), high = e.constant(UINT64_MAX);
    const auto a = e.select(p, high, zero);
    projection.record(a, p, high, zero);
    const auto b = e.select(e.land(q, e.lt(a, opaque)), opaque, a);
    projection.record(b, q, opaque, a, true);
    std::vector<std::pair<uint64_t, fs::RegionExpressions::Id>> queries;
    for (uint64_t k : {uint64_t(0), uint64_t(1), uint64_t(3), UINT64_MAX}) {
        const auto root = projection.below(b, k);
        const auto visits = work.visits;
        if (projection.below(b, k) != root || work.visits != visits) {
            return false;
        }
        queries.emplace_back(k, root);
    }
    for (unsigned mask = 0; mask < 4; ++mask) {
        for (uint64_t x : {uint64_t(0), uint64_t(2), UINT64_MAX}) {
            fs::RegionExpressions::Transaction scratch(e);
            fs::RegionExpressions::Substitution bindings(
                {{p, e.boolean(mask & 1)}, {q, e.boolean(mask & 2)}, {opaque, e.constant(x)}});
            const uint64_t av = mask & 1 ? UINT64_MAX : 0;
            const uint64_t bv = mask & 2 ? std::max(av, x) : av;
            for (auto [k, root] : queries) {
                if (e.constantValue(e.substitute(root, bindings)) != uint64_t(bv < k)) {
                    return false;
                }
            }
        }
    }
    // The unselected numeric input is deliberately left unbound. A false q
    // must mask it; no strict Boolean fold may expose the skipped operand.
    fs::RegionExpressions::Substitution masked({{p, e.boolean(false)}, {q, e.boolean(false)}});
    return e.constantValue(e.substitute(queries[1].second, masked)) == 1 && work.hits == queries.size();
}
inline bool thresholdBudget(mlir::MLIRContext& context)
{
    mlir::Block block;
    fs::RegionExpressions e;
    fs::GuardedRankProjectionWork work;
    fs::detail::FixedRankThresholds projection(e, work);
    std::vector<fs::RegionExpressions::Id> guards, answers;
    auto rank = e.constant(0);
    for (unsigned i = 0; i < 64; ++i) {
        auto input = block.addArgument(mlir::IntegerType::get(&context, 1), mlir::UnknownLoc::get(&context));
        const auto guard = e.input(input), previous = rank, value = e.constant(i + 1);
        guards.push_back(guard);
        rank = e.select(guard, value, previous);
        projection.record(rank, guard, value, previous);
    }
    bool fallback = false;
    const auto before = e.size();
    for (unsigned k = 1; k <= 64; ++k) {
        auto answer = projection.below(rank, k);
        fallback |= answer == e.lt(rank, e.constant(k));
        answers.push_back(answer);
    }
    // Include test-created incumbent comparisons in the construction envelope.
    if (!fallback || work.visits > 8 * (work.descriptions + work.requests + 1) ||
        work.entries > work.visits + work.requests || e.size() - before > 3 * work.visits + 2 * work.requests + 2) {
        return false;
    }
    for (unsigned sample = 0; sample < 66; ++sample) {
        fs::RegionExpressions::Transaction scratch(e);
        std::vector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> bindings;
        uint64_t expected = 0, random = sample + 1;
        for (unsigned i = 0; i < guards.size(); ++i) {
            random = random * 6364136223846793005ULL + 1;
            const bool present = sample == 65 || (sample != 0 && (random >> 63));
            if (present) {
                expected = i + 1;
            }
            bindings.emplace_back(guards[i], e.boolean(present));
        }
        fs::RegionExpressions::Substitution substitution(bindings);
        for (unsigned k = 1; k <= answers.size(); ++k) {
            if (e.constantValue(e.substitute(answers[k - 1], substitution)) != uint64_t(expected < k)) {
                return false;
            }
        }
    }
    return true;
}
inline bool independentReadScaling(mlir::MLIRContext& context)
{
    for (unsigned count : {16U, 32U, 64U, 128U}) {
        mlir::Block block;
        fs::RegionExpressions e;
        std::vector<fs::GuardedRankPayload> payloads{{0, e.boolean(true)}};
        std::vector<fs::GuardedRankEdge> edges;
        for (unsigned i = 1; i <= count; ++i) {
            auto input = block.addArgument(mlir::IntegerType::get(&context, 1), mlir::UnknownLoc::get(&context));
            payloads.push_back({1, e.input(input)});
            edges.push_back({0, i, e.boolean(true)});
        }
        const auto result =
            fs::reduceGuardedRanks(e, payloads, edges, {}, fs::GuardedRankLabels::Reference, false, true);
        // Fixed two pipes, one writer, independent entry guards: input, rank
        // registers, each requested fixed threshold and retention are linear.
        if (!result.error.empty() || e.size() > 64 * count + 16 || result.projectionWork.entries > 16 * count ||
            result.projectionWork.visits > 48 * count) {
            return false;
        }
    }
    return true;
}
inline bool boundedThresholdScaling(mlir::MLIRContext& context)
{
    for (unsigned count : {8U, 16U, 32U, 64U}) {
        for (unsigned pipes : {1U, 3U}) {
            mlir::Block block;
            std::vector<mlir::Value> inputs;
            for (unsigned i = 0; i < count; ++i) {
                inputs.push_back(
                    block.addArgument(mlir::IntegerType::get(&context, 1), mlir::UnknownLoc::get(&context)));
            }
            std::array<uint64_t, 2> constructed{}, live{}, visits{}, entries{}, requests{}, descriptions{};
            for (unsigned projected = 0; projected < 2; ++projected) {
                fs::RegionExpressions e;
                std::vector<fs::GuardedRankPayload> payloads;
                std::vector<fs::GuardedRankEdge> edges;
                for (unsigned i = 0; i < count; ++i) {
                    payloads.push_back({i % pipes, e.input(inputs[i])});
                    for (unsigned j = 0; j < i; ++j) {
                        edges.push_back({j, i, e.boolean(true)});
                    }
                }
                const auto result =
                    fs::reduceGuardedRanks(e, payloads, edges, {}, fs::GuardedRankLabels::Reference, false, projected);
                const auto& work = result.projectionWork;
                if (!result.error.empty() || work.visits > 8 * (work.descriptions + work.requests + 1) ||
                    work.entries > work.visits + work.requests) {
                    return false;
                }
                constructed[projected] = e.size();
                visits[projected] = work.visits;
                entries[projected] = work.entries;
                requests[projected] = work.requests;
                descriptions[projected] = work.descriptions;
                llvm::SmallVector<fs::RegionExpressions::Id> roots;
                for (auto edge : result.retained) {
                    roots.push_back(edge.guard);
                }
                auto report = fs::detail::RegionConstructionAccess::profileRoots(e, roots, {});
                live[projected] = *report.getInteger("union_nodes");
            }
            // Independent occurrence predicates do not enter the optional
            // shared-presence construction, even when explicitly requested.
            if (live[1] != live[0] || constructed[1] != constructed[0] || requests[1] || visits[1]) {
                return false;
            }
            llvm::outs() << "finite-threshold-scaling "
                         << llvm::json::Value(
                                llvm::json::Object{
                                    {"sites", count},
                                    {"pipes", pipes},
                                    {"baseline_nodes", constructed[0]},
                                    {"candidate_nodes", constructed[1]},
                                    {"baseline_guard_nodes", live[0]},
                                    {"candidate_guard_nodes", live[1]},
                                    {"projection_visits", visits[1]},
                                    {"projection_entries", entries[1]},
                                    {"projection_requests", requests[1]},
                                    {"numeric_descriptions", descriptions[1]}})
                         << "\n";
        }
    }
    return true;
}
inline bool inactiveEvaluation(mlir::MLIRContext& context)
{
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  func.func @inactive() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c = arith.cmpi ult, %c0, %c1 : index
    %f = scf.if %c -> (i1) {
      %false = arith.constant false
      scf.yield %false : i1
    } else {
      %true = arith.constant true
      scf.yield %true : i1
    }
    %id = arith.remui %c1, %c6 : index
    scf.if %f {
      %skipped = arith.remui %id, %c6 : index
      "pto.logical_set"(%id) : (index) -> ()
      "pto.logical_wait"(%skipped) : (index) -> ()
    }
    return
  }
})mlir",
        &context);
    if (!module) {
        return false;
    }
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    finite_control_checks::Evaluation counter(function);
    for (auto& op : function.front()) {
        counter.operation(op);
        if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(&op)) {
            const bool selected = branch.getNumResults() != 0;
            counter.branch(branch, selected);
            if (selected) {
                for (auto& inner : branch.getThenRegion().front()) {
                    counter.operation(inner);
                }
                counter.yielded = counter.inputs(branch.getThenRegion().front().back().getOperands());
                counter.branchResults(branch);
            }
        }
    }
    auto report = counter.report();
    return report.getInteger("evaluated_union") == 6 && report.getInteger("command_guard_tests") == 1 &&
           report.getInteger("false_command_guard_tests") == 1 &&
           report.getInteger("inactive_command_arm_tests") == 1 &&
           report.getInteger("inactive_test_dependency_evaluations") == 4;
}
inline bool evaluation(mlir::MLIRContext& context)
{
    context.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect>();
    context.allowUnregisteredDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        R"mlir(
module {
  func.func @control(%p: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c = arith.cmpi ult, %c0, %c1 : index
    %r = scf.if %c -> (index) {
      %a = arith.addi %c1, %c2 : index
      scf.yield %a : index
    } else {
      scf.yield %c0 : index
    }
    %s = scf.for %i = %c0 to %c2 step %c1 iter_args(%v = %r) -> (index) {
      %n = arith.addi %v, %c1 : index
      scf.yield %n : index
    }
    "pto.logical_set"(%s) : (index) -> ()
    return
  }
})mlir",
        &context);
    if (!module) {
        return false;
    }
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    finite_control_checks::Evaluation counter(function);
    for (auto& op : function.front()) {
        counter.operation(op);
        if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(&op)) {
            for (auto& inner : branch.getThenRegion().front()) {
                counter.operation(inner);
            }
            counter.yielded = counter.inputs(branch.getThenRegion().front().back().getOperands());
            counter.branchResults(branch);
        } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(&op)) {
            counter.loopEnter(loop);
            auto state = counter.inputs(loop.getInitArgs());
            for (unsigned i = 0; i < 2; ++i) {
                counter.induction(loop);
                counter.bind(loop.getRegionIterArgs(), state);
                for (auto& inner : *loop.getBody()) {
                    counter.operation(inner);
                }
                state = counter.inputs(loop.getBody()->back().getOperands());
            }
            counter.loopResults(loop, state);
        } else if (op.getName().getStringRef() == "pto.logical_set") {
            counter.command(op, 1);
        }
    }
    auto report = counter.report();
    // Three constants, the branch condition, its chosen add, and two loop
    // adds. The result-bearing branch has no command inside, but its condition
    // still contributes to the final identity. Each loop visit is distinct.
    return report.getInteger("evaluated_union") == 7 && report.getInteger("command_guard_tests") == 0;
}
inline bool run(mlir::MLIRContext& context)
{
    return roots() && thresholds(context) && thresholdBudget(context) && independentReadScaling(context) &&
           boundedThresholdScaling(context) && evaluation(context) && inactiveEvaluation(context);
}
} // namespace compactFiniteControlChecks
#endif
