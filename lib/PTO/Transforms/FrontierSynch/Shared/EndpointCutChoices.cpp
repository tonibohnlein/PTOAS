// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/EndpointCutChoices.h"
#include "CountedLoop.h"
#include "../../InsertSync/SyncScalarReplay.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
namespace {
using Path = SmallVector<Region*>;
bool transparentSection(Operation* operation)
{
    // Physical-context projection materializes implicit section predicates
    // before transferring a proof. Raw implicitly guarded sections must first
    // go through that normalization; they are not unconditional containers.
    return isa<SectionCubeOp>(operation) ||
        (isa<SectionVectorOp>(operation) &&
         (operation->hasAttr("pto.sync_context_only") || !needsA5NoSplitVectorGuard(operation)));
}
bool constant(Value value, int64_t expected)
{
    APInt number;
    return value && matchPattern(value, m_ConstantInt(&number)) && number.isSignedIntN(64) &&
        number.getSExtValue() == expected;
}
bool sameBound(Value a, Value b)
{
    APInt x, y;
    return a == b || (matchPattern(a, m_ConstantInt(&x)) && matchPattern(b, m_ConstantInt(&y)) &&
        x.sextOrTrunc(64) == y.sextOrTrunc(64));
}
bool ordinal(Value value, scf::ForOp loop)
{
    // Direct IV and unsigned distance from the lower bound are injective.
    // Dividing that distance by the original positive step is the exact visit
    // ordinal on the counted-loop domain, even when the bounds straddle zero.
    if (value == loop.getInductionVar()) { return true; }
    if (auto division = value.getDefiningOp<arith::DivUIOp>()) {
        auto domain = CountedLoop::get(loop);
        if (!domain || !constant(division.getRhs(), domain->step)) { return false; }
        value = division.getLhs();
        if (value == loop.getInductionVar() && constant(loop.getLowerBound(), 0)) { return true; }
    }
    auto difference = value.getDefiningOp<arith::SubIOp>();
    return difference && difference.getLhs() == loop.getInductionVar() &&
        sameBound(difference.getRhs(), loop.getLowerBound());
}
bool invariant(Value root, Operation* outer, DominanceInfo& dominance)
{
    SmallVector<Value> work{root};
    llvm::DenseSet<Value> seen;
    while (!work.empty()) {
        auto value = work.pop_back_val();
        if (!seen.insert(value).second) { continue; }
        auto* region = value.getParentRegion();
        if (region && region->getParentOp() && dominance.properlyDominates(value, outer)) { continue; }
        auto* op = value.getDefiningOp();
        // Detached or endpoint-local arithmetic may depend only on invariant
        // values. In particular, a memory read within a visit is not invariant.
        if (!mlir::pto::detail::canReplayScalar(op)) { return false; }
        llvm::append_range(work, op->getOperands());
    }
    return true;
}
void conjuncts(Value root, SmallVectorImpl<Value>& atoms)
{
    SmallVector<Value> work{root};
    llvm::DenseSet<Value> seen;
    while (!work.empty()) {
        auto value = work.pop_back_val();
        if (!value || !seen.insert(value).second) { continue; }
        if (auto both = value.getDefiningOp<arith::AndIOp>()) {
            if (value.getType().isInteger(1)) { llvm::append_range(work, both->getOperands()); continue; }
        }
        if (auto select = value.getDefiningOp<arith::SelectOp>()) {
            if (constant(select.getFalseValue(), 0)) {
                work.push_back(select.getCondition()); work.push_back(select.getTrueValue()); continue;
            }
        }
        atoms.push_back(value);
    }
}
bool singletonConsumer(Operation* cut, Value predicate, func::FuncOp function)
{
    if (predicate && !predicate.getType().isInteger(1)) { return false; }
    SmallVector<scf::ForOp> loops;
    SmallVector<Value> atoms;
    conjuncts(predicate, atoms);
    auto* descendant = cut;
    for (auto* parent = cut->getParentOp(); parent != function; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            if (!CountedLoop::get(loop)) { return false; }
            loops.push_back(loop);
        } else if (auto branch = dyn_cast<scf::IfOp>(parent)) {
            if (descendant->getParentRegion() == &branch.getThenRegion()) {
                conjuncts(branch.getCondition(), atoms);
            }
        } else if (!transparentSection(parent)) { return false; }
        descendant = parent;
    }
    if (loops.empty()) { return true; }
    if (!predicate) { return false; }
    if (constant(predicate, 0)) { return true; }
    DominanceInfo dominance(function);
    for (auto loop : loops) {
        bool selected = llvm::any_of(atoms, [&](Value atom) {
            auto equality = atom.getDefiningOp<arith::CmpIOp>();
            if (!equality || equality.getPredicate() != arith::CmpIPredicate::eq) { return false; }
            return (ordinal(equality.getLhs(), loop) && invariant(equality.getRhs(), loops.back(), dominance)) ||
                (ordinal(equality.getRhs(), loop) && invariant(equality.getLhs(), loops.back(), dominance));
        });
        if (!selected) { return false; }
    }
    return true;
}
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
            if (transparentSection(region->getParentOp())) {
                region = region->getParentOp()->getParentRegion();
                continue;
            }
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
bool terminalSuffix(Operation* command, func::FuncOp function)
{
    for (auto* current = command; current; current = current->getParentOp()) {
        for (auto* next = current->getNextNode(); next; next = next->getNextNode()) {
            if (isa<func::ReturnOp>(next) || isMemoryEffectFree(next)) { continue; }
            // Other epilogue WAITs and the invocation drain are permitted;
            // no publication or original payload may follow cleanup.
            bool harmless = true;
            next->walk([&](Operation* op) {
                if (isa<scf::IfOp, scf::YieldOp>(op) || transparentSection(op) || isMemoryEffectFree(op)) { return; }
                auto name = op->getName().getStringRef();
                harmless &= name == "pto.logical_wait" || name == "pto.barrier";
            });
            if (!harmless) { return false; }
        }
        if (current->getParentOp() == function) { return current->getBlock() == &function.front(); }
        if (!transparentSection(current->getParentOp())) { return false; }
    }
    return false;
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
std::shared_ptr<const EndpointCutChoices> qualifyTerminalCleanupCuts(
    ArrayRef<TemplateEndpointCut> cuts, Value consumerPredicate)
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
    }
    if (!singletonConsumer(cuts[0].before, consumerPredicate, function)) { return nullptr; }
    return std::shared_ptr<const EndpointCutChoices>(new EndpointCutChoices({}, cuts, consumerPredicate));
}
bool validateTerminalCleanupCommands(func::FuncOp function, ArrayRef<Operation*> commands)
{
    if (commands.size() != 2 || !function.getBody().hasOneBlock()) { return false; }
    unsigned terminal = 0;
    for (auto* command : commands) {
        auto wrapper = dyn_cast_or_null<scf::IfOp>(command->getParentOp());
        if (!wrapper || !wrapper->hasAttr("pto.endpoint_cut") ||
            command->getParentRegion() != &wrapper.getThenRegion()) { return false; }
        if (!singletonConsumer(wrapper, wrapper.getCondition(), function)) { return false; }
        terminal += terminalSuffix(wrapper, function);
    }
    return terminal >= 1;
}
bool validateEndpointCommandChoices(scf::ForOp loop, ArrayRef<Operation*> commands)
{
    return validate(loop, commands, true);
}
} // namespace mlir::pto::frontiersynch
