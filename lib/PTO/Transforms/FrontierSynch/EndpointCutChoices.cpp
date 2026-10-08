// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/EndpointCutChoices.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
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
std::shared_ptr<const EndpointCutChoices> qualifyTerminalCleanupCuts(ArrayRef<TemplateEndpointCut> cuts)
{
    if (cuts.size() != 2 || !cuts[0].before || !cuts[1].before || cuts[0].before == cuts[1].before) {
        return nullptr;
    }
    auto function = cuts[1].before->getParentOfType<func::FuncOp>();
    if (!function || !function.getBody().hasOneBlock() ||
        cuts[1].before != function.front().getTerminator() || !isa<func::ReturnOp>(cuts[1].before)) {
        return nullptr;
    }
    for (const auto& cut : cuts) {
        if (cut.block != cut.before->getBlock() || cut.before->getParentOfType<func::FuncOp>() != function) {
            return nullptr;
        }
        for (auto* parent = cut.before->getParentOp(); parent != function; parent = parent->getParentOp()) {
            if (!isa<scf::IfOp>(parent)) { return nullptr; }
        }
    }
    return std::shared_ptr<const EndpointCutChoices>(new EndpointCutChoices({}, cuts));
}
bool validateTerminalCleanupCommands(func::FuncOp function, ArrayRef<Operation*> commands)
{
    if (commands.size() != 2 || !function.getBody().hasOneBlock()) { return false; }
    unsigned terminal = 0;
    for (auto* command : commands) {
        auto wrapper = dyn_cast_or_null<scf::IfOp>(command->getParentOp());
        if (!wrapper || !wrapper->hasAttr("pto.endpoint_cut") ||
            command->getParentRegion() != &wrapper.getThenRegion()) { return false; }
        for (auto* parent = wrapper->getParentOp(); parent != function; parent = parent->getParentOp()) {
            if (!isa<scf::IfOp>(parent)) { return false; }
        }
        if (wrapper->getBlock() != &function.front()) { continue; }
        bool suffix = true;
        for (auto* next = wrapper->getNextNode(); next; next = next->getNextNode()) {
            if (isa<func::ReturnOp>(next) || isMemoryEffectFree(next)) { continue; }
            // Other epilogue WAITs and the invocation drain are permitted;
            // no publication or original payload may follow cleanup.
            bool harmless = true;
            next->walk([&](Operation* op) {
                if (isa<scf::IfOp, scf::YieldOp>(op) || isMemoryEffectFree(op)) { return; }
                auto name = op->getName().getStringRef();
                harmless &= name == "pto.logical_wait" || name == "pto.barrier";
            });
            suffix &= harmless;
        }
        terminal += suffix;
    }
    return terminal >= 1;
}
bool validateEndpointCommandChoices(scf::ForOp loop, ArrayRef<Operation*> commands)
{
    return validate(loop, commands, true);
}
} // namespace mlir::pto::frontiersynch
