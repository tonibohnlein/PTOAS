// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/PrefixInput.h"
#include "CompletionStorage.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include <algorithm>
#include <chrono>
namespace mlir::pto::frontiersynch {
namespace {
using Clock = std::chrono::steady_clock;
class GraphBuilder {
public:
    PrefixInput result;
    DenseMap<Operation*, std::size_t> sites;
    std::size_t node(std::optional<std::size_t> site, const std::vector<std::size_t>& before)
    {
        auto id = result.graph.size();
        result.graph.push_back({site, {}});
        for (auto previous : before) {
            result.graph[previous].successors.push_back(id);
        }
        return id;
    }
    std::vector<std::size_t> block(Block& block, std::vector<std::size_t> before)
    {
        for (auto& operation : block) {
            if (auto loop = dyn_cast<scf::ForOp>(operation)) {
                auto header = node(std::nullopt, before);
                for (auto backedge : this->block(*loop.getBody(), {header})) {
                    result.graph[backedge].successors.push_back(header);
                }
                before = {header};
            } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
                auto gate = node(std::nullopt, before);
                auto left = this->block(branch.getThenRegion().front(), {gate});
                auto right = branch.getElseRegion().empty() ? std::vector<std::size_t>{gate} :
                                                              this->block(branch.getElseRegion().front(), {gate});
                left.insert(left.end(), right.begin(), right.end());
                std::sort(left.begin(), left.end());
                left.erase(std::unique(left.begin(), left.end()), left.end());
                before = {node(std::nullopt, left)};
            } else {
                auto found = sites.find(&operation);
                if (found != sites.end()) {
                    result.order.push_back(found->second);
                    before = {node(found->second, before)};
                }
                if (operation.getNumRegions()) {
                    result.error = "unsupported structured region";
                }
            }
        }
        return before;
    }
};
} // namespace
PrefixInput buildPrefixInput(func::FuncOp function, ArrayRef<const CompoundInstanceElement*> phases)
{
    GraphBuilder builder;
    auto start = Clock::now();
    for (auto [id, phase] : llvm::enumerate(phases)) {
        if (!phase || !phase->elementOp || !builder.sites.try_emplace(phase->elementOp, id).second) {
            builder.result.error = "original cut needs exactly one modeled phase";
            return std::move(builder.result);
        }
    }
    CompletionStorage storage(phases);
    builder.result.classes = storage.size();
    for (const auto* phase : phases) {
        auto modes = storage.effects({phase});
        builder.result.sites.push_back({static_cast<uint32_t>(phase->kPipeValue), {modes.begin(), modes.end()}, {}});
    }
    auto classified = Clock::now();
    builder.result.classSeconds = std::chrono::duration<double>(classified - start).count();
    if (!llvm::hasSingleElement(function.getBody())) {
        builder.result.error = "one structured function block required";
    } else {
        builder.block(function.front(), {0});
    }
    if (builder.result.error.empty() && builder.result.order.size() != phases.size()) {
        builder.result.error = "modeled phase is not at an original structured cut";
    }
    builder.result.graphSeconds = std::chrono::duration<double>(Clock::now() - classified).count();
    return std::move(builder.result);
}
} // namespace mlir::pto::frontiersynch
