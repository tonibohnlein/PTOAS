// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/EndpointCutChoices.h"
#include "llvm/ADT/DenseSet.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
namespace {
using Path = SmallVector<Region*>;
bool partitionsBranches(ArrayRef<Path> paths, std::size_t depth)
{
    if (paths.empty()) { return false; }
    if (paths.front().size() == depth) { return paths.size() == 1; }
    auto* branch = paths.front()[depth]->getParentOp();
    SmallVector<Path> thenPaths, elsePaths;
    for (const auto& path : paths) {
        if (path.size() <= depth || path[depth]->getParentOp() != branch) { return false; }
        (path[depth] == &branch->getRegion(0) ? thenPaths : elsePaths).push_back(path);
    }
    return partitionsBranches(thenPaths, depth + 1) && partitionsBranches(elsePaths, depth + 1);
}
bool validate(scf::ForOp loop, ArrayRef<Operation*> cuts, bool commands)
{
    if (!loop || cuts.empty()) { return false; }
    SmallVector<Path> paths;
    llvm::DenseSet<Operation*> seen;
    for (auto* cut : cuts) {
        if (!cut || !seen.insert(cut).second) { return false; }
        Path path;
        auto* region = cut->getParentRegion();
        if (commands) {
            auto wrapper = region ? dyn_cast_or_null<scf::IfOp>(region->getParentOp()) : scf::IfOp{};
            if (!wrapper || !wrapper->hasAttr("pto.endpoint_cut") || region != &wrapper.getThenRegion()) {
                return false;
            }
            region = wrapper->getParentRegion();
        }
        while (region && region != &loop.getRegion()) {
            auto branch = dyn_cast_or_null<scf::IfOp>(region->getParentOp());
            if (!branch) { return false; }
            path.push_back(region);
            region = branch->getParentRegion();
        }
        if (!region) { return false; }
        std::reverse(path.begin(), path.end());
        paths.push_back(std::move(path));
    }
    return partitionsBranches(paths, 0);
}
} // namespace
std::shared_ptr<const EndpointCutChoices> qualifyEndpointCutChoices(
    scf::ForOp loop, ArrayRef<TemplateEndpointCut> cuts)
{
    SmallVector<Operation*> operations;
    for (const auto& cut : cuts) {
        if (!cut.before || cut.block != cut.before->getBlock()) { return nullptr; }
        operations.push_back(cut.before);
    }
    if (!validate(loop, operations, false)) { return nullptr; }
    return std::shared_ptr<const EndpointCutChoices>(new EndpointCutChoices(loop, cuts));
}
bool validateEndpointCommandChoices(scf::ForOp loop, ArrayRef<Operation*> commands)
{
    return validate(loop, commands, true);
}
} // namespace mlir::pto::frontiersynch
