// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Inspect logical recipes without turning logical identities into hardware IDs.
#include "PTO/Transforms/FrontierSynch/NumericTemplateEndpoints.h"
#include "mlir/IR/AsmState.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
StringRef kindName(fs::EndpointKind kind)
{
    switch (kind) {
    case fs::EndpointKind::Set: return "set";
    case fs::EndpointKind::Barrier: return "barrier";
    case fs::EndpointKind::Wait: return "wait";
    default: return "invalid";
    }
}
std::string valueName(Value value, AsmState& state)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    value.printAsOperand(stream, state);
    return text;
}
}
llvm::json::Object dumpLogicalEndpoints(const fs::LogicalEndpointPlan& plan)
{
    llvm::json::Array recipes;
    for (const auto& recipe : plan.recipes) {
        recipes.push_back(llvm::json::Object{{"record", recipe.record}, {"source", recipe.source},
            {"target", recipe.target}, {"pipe", recipe.pipe}, {"displacement", recipe.displacement},
            {"kind", kindName(recipe.kind)}});
    }
    return llvm::json::Object{{"error", plan.error}, {"recipes", std::move(recipes)},
        {"selection_recipes_ready", plan.error.empty()}, {"physical_ids_ready", false}, {"ir_emitted", false}};
}
llvm::json::Object dumpNumericTemplateEndpoints(const fs::NumericTemplateEndpoints& plan, AsmState& state)
{
    auto result = dumpLogicalEndpoints(plan.logical);
    llvm::json::Array anchors;
    llvm::DenseMap<std::pair<Block*, Operation*>, uint64_t> cuts;
    auto cutID = [&cuts](fs::TemplateEndpointCut cut) {
        auto insertion = cuts.try_emplace({cut.block, cut.before}, cuts.size());
        return insertion.first->second;
    };
    for (const auto& anchor : plan.anchors) {
        llvm::json::Array coordinates;
        for (auto coordinate : anchor.coordinates) {
            coordinates.push_back(llvm::json::Object{
                {"induction", valueName(coordinate.loop.getInductionVar(), state)}, {"value", coordinate.induction}});
        }
        anchors.push_back(llvm::json::Object{{"phase", anchor.phase->GetIndex()},
            {"coordinates", std::move(coordinates)}, {"before_cut", cutID(anchor.before)},
            {"after_cut", cutID(anchor.after)}});
    }
    llvm::json::Array groups;
    for (const auto& group : plan.groups) {
        llvm::json::Array recipes;
        for (auto recipe : group.recipes) {
            recipes.push_back(recipe);
        }
        groups.push_back(llvm::json::Object{{"cut", cutID(group.cut)}, {"recipes", std::move(recipes)}});
    }
    result["groups"] = std::move(groups);
    result["anchors"] = std::move(anchors);
    result["canonical_cuts"] = cuts.size();
    return result;
}
