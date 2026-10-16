// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Measure fresh guarded preparation without changing recipe ownership or caching.
#include "PTO/Transforms/FrontierSynch/Finite/FiniteAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Finite/FiniteGuardedInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/Shared/DeferredFamily.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/JSON.h"
#include <chrono>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Clock = std::chrono::steady_clock;
llvm::json::Object difference(const fs::FiniteGuardedState::PreparationWork& before,
                              const fs::FiniteGuardedState::PreparationWork& after)
{
    return llvm::json::Object{{"requests", after.requests - before.requests},
            {"inventories", after.inventories - before.inventories},
            {"demand_visits", after.demandVisits - before.demandVisits},
            {"filter_queries", after.filterQueries - before.filterQueries},
            {"publication_requests", after.publicationRequests - before.publicationRequests},
            {"recipe_builds", after.recipeBuilds - before.recipeBuilds}};
}
}
LogicalResult checkFiniteGuardedPreparation(const std::shared_ptr<fs::FiniteAnalysis>& owner)
{
    if (!owner || !owner->guarded() || !owner->guarded()->state) { return failure(); }
    auto& state = *owner->guarded()->state;
    std::string original;
    llvm::raw_string_ostream printed(original);
    state.function.print(printed);
    printed.flush();
    llvm::json::Array samples;
    std::vector<std::unique_ptr<fs::PreparedLogicalPlan>> plans;
    const auto demandCount = state.retained.size();
    const auto firstNodes = state.arena->size();
    for (unsigned request = 0; request < 7; ++request) {
        const auto before = state.preparationWork;
        const auto nodes = state.arena->size();
        fs::RegionalDemandFilter filter;
        if (request == 2 || request == 4) {
            const auto accept = state.arena->boolean(request == 2);
            filter = [accept](const auto&, const auto&) { return std::optional<fs::RegionExpressions::Id>(accept); };
        } else if (request == 3) {
            const auto yes = state.arena->boolean(true), no = state.arena->boolean(false);
            filter = [yes, no](const auto& source, const auto&) {
                return std::optional<fs::RegionExpressions::Id>(source.type % 2 ? yes : no);
            };
        } else if (request == 5) {
            filter = [](const auto&, const auto&) { return std::optional<fs::RegionExpressions::Id>{}; };
        }
        const auto start = Clock::now();
        auto result = filter ? state.prepare(filter) : owner->prepare();
        const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
        const bool refused = request == 5 && demandCount != 0;
        if (failed(result) != refused) { return failure(); }
        auto sample = difference(before, state.preparationWork);
        sample["request"] = request;
        sample["seconds"] = seconds;
        sample["new_expression_nodes"] = state.arena->size() - nodes;
        sample["refused"] = refused;
        if (succeeded(result)) {
            auto& plan = **result;
            if (request == 1 || request == 2 || request == 6) {
                if (plan.families.size() != plans.front()->families.size()) { return failure(); }
            }
            if (request == 4 && !plan.families.empty()) { return failure(); }
            std::vector<std::pair<uint32_t, uint32_t>> expectedEdges, actualEdges;
            for (const auto& edge : state.retained) {
                if (state.arena->constantValue(edge.guard) == 0 || request == 4 ||
                    (request == 3 && edge.source % 2 == 0)) { continue; }
                expectedEdges.emplace_back(edge.source, edge.target);
            }
            for (const auto& family : plan.families) {
                for (const auto& member : family.members) { actualEdges.emplace_back(member.source, member.target); }
            }
            if (actualEdges != expectedEdges) { return failure(); }
            sample["families"] = plan.families.size();
            sample["recipe_present"] = bool(plan.familyRecipe);
            if (!plan.preparation.empty() || !plan.endpoints.empty()) { return failure(); }
            if (request == 4 && plan.familyRecipe) { return failure(); }
            sample["reuses_first_recipe"] = !plans.empty() && plan.familyRecipe &&
                plan.familyRecipe == plans.front()->familyRecipe;
            plans.push_back(std::move(*result));
        }
        samples.push_back(std::move(sample));
    }
    std::string after;
    llvm::raw_string_ostream finalPrint(after);
    state.function.print(finalPrint);
    finalPrint.flush();
    if (original != after || state.retained.size() != demandCount || !state.insertionError.empty()) {
        return failure();
    }
    llvm::outs() << llvm::json::Value(llvm::json::Object{
        {"retained_demands", demandCount}, {"initial_expression_nodes", firstNodes},
        {"final_expression_nodes", state.arena->size()}, {"samples", std::move(samples)}}) << "\n";
    return success();
}

// Exercise the real regional interface with contiguous original-operation children,
// then use the ordinary family allocation and insertion services.
LogicalResult emitFiniteComposed(func::FuncOp function, pto::GMAliasPolicy policy)
{
    auto storage = std::make_shared<pto::SyncInput>(policy);
    auto index = std::make_shared<fs::PhaseIndex>();
    if (failed(storage->build(function, pto::SyncInstructionView::PipeEnvelopes)) ||
        failed(index->build(function, *storage))) { return failure(); }
    auto input = fs::finiteFunctionInput(function, storage, index);
    if (!input) { return failure(); }
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::vector<fs::RegionalAnalysis> children;
    for (auto* root : input->roots()) {
        auto child = fs::analyzeFinite(fs::finiteRegionInput(input, ArrayRef<Operation*>{root}),
                                      fs::FiniteStrategy::Guarded, arena);
        if (!child || !child->error().empty()) { return failure(); }
        children.push_back(fs::finiteRegionalResult(child, arena));
    }
    auto composed = fs::composeRegionalSequence(function, arena, std::move(children), true, true,
                                                index->sourceFacts());
    if (!composed.error.empty() || !composed.state || composed.state->children.size() < 2 ||
        composed.state->crossings.empty() || composed.state->relationalResult) {
        return function.emitError("finite composition fixture needs multiple children and crossing demands");
    }
    llvm::errs() << "finite-composition-profile " << llvm::json::Value(llvm::json::Object{
        {"children", composed.state->children.size()}, {"crossings", composed.state->crossings.size()},
        {"expressions", arena->size()}}) << "\n";
    auto plan = fs::prepareSequenceInsertion(composed);
    if (failed(plan)) { return function.emitError("finite composition preparation failed"); }
    (*plan)->profileControl = true;
    fs::prepareAllocationSupport(**plan);
    fs::profileLogicalControl(function, **plan, "prepared");
    auto route = function->getAttrOfType<StringAttr>("test.finite_output");
    if (!route) { return failure(); }
    if (route.getValue() == "combined") {
        if (failed(fs::planPhysicalFamilyOutput(function, **plan, {0, 1, 2, 3, 4, 5}))) { return failure(); }
    } else if (route.getValue() == "saved") {
        (*plan)->deferredOutput = true;
    } else if (route.getValue() != "logical") { return failure(); }
    function->removeAttr("test.finite_output");
    return success(succeeded(fs::insertLogicalSynchronization(function, **plan)) && succeeded(verify(function)));
}
