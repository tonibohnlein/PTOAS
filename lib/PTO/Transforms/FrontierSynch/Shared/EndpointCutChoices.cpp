// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/EndpointCutChoices.h"
#include "CleanupSelectionProof.h"
#include "../../InsertSync/SyncScalarReplay.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
// One structural interpreter for exact predicates, before and after scalar
// lowering. Arena inputs are original SSA leaves, not a second control model.
class CutPredicate {
public:
    using Id = RegionExpressions::Id;
    using Ref = std::pair<Id, Value>;
    using Binary = detail::CleanupPredicateBinary;
    explicit CutPredicate(const RegionExpressions* arena = nullptr) : arena(arena) {}
    Ref get(Value value) const { return {RegionExpressions::invalid, value}; }
    Ref get(Id id) const
    {
        if (!arena || !arena->valid(id)) { return get(Value()); }
        const auto& node = arena->nodes[id];
        return node.kind == RegionExpressions::Kind::Input ? get(node.value) : Ref{id, {}};
    }
    bool valid(Ref ref) const { return ref.second || (arena && arena->valid(ref.first)); }
    bool boolean(Ref ref) const
    {
        return ref.second ? ref.second.getType().isInteger(1) : arena && arena->isBoolean(ref.first);
    }
    std::optional<uint64_t> number(Ref ref) const
    {
        if (!ref.second) { return arena ? arena->constantValue(ref.first) : std::nullopt; }
        APInt number;
        if (!matchPattern(ref.second, m_ConstantInt(&number)) || !number.isSignedIntN(64)) {
            return std::nullopt;
        }
        return number.sextOrTrunc(64).getZExtValue();
    }
    std::optional<std::pair<Ref, Ref>> binary(Ref ref, Binary kind) const
    {
        if (!ref.second) {
            if (!arena || !arena->valid(ref.first)) { return std::nullopt; }
            const auto& node = arena->nodes[ref.first];
            using Kind = RegionExpressions::Kind;
            const auto expected = kind == Binary::And ? Kind::And : kind == Binary::Equal ? Kind::Eq :
                kind == Binary::Subtract ? Kind::Sub : Kind::Div;
            if (node.kind != expected) { return std::nullopt; }
            return std::make_pair(get(node.a), get(node.b));
        }
        auto* op = ref.second.getDefiningOp();
        bool matches = op && ((kind == Binary::And && isa<arith::AndIOp>(op) && boolean(ref)) ||
            (kind == Binary::Subtract && isa<arith::SubIOp>(op)) ||
            (kind == Binary::Divide && isa<arith::DivUIOp>(op)));
        if (kind == Binary::Equal) {
            auto compare = dyn_cast_or_null<arith::CmpIOp>(op);
            matches = compare && compare.getPredicate() == arith::CmpIPredicate::eq;
        }
        if (!matches) { return std::nullopt; }
        return std::make_pair(get(op->getOperand(0)), get(op->getOperand(1)));
    }
    std::optional<std::pair<Ref, Ref>> masked(Ref ref) const
    {
        if (!ref.second) {
            if (!arena || !arena->valid(ref.first)) { return std::nullopt; }
            const auto& node = arena->nodes[ref.first];
            if (node.kind != RegionExpressions::Kind::Select || number(get(node.c)) != 0) {
                return std::nullopt;
            }
            return std::make_pair(get(node.a), get(node.b));
        }
        auto select = ref.second.getDefiningOp<arith::SelectOp>();
        if (!select || number(get(select.getFalseValue())) != 0) { return std::nullopt; }
        return std::make_pair(get(select.getCondition()), get(select.getTrueValue()));
    }
    Value original(Ref ref) const { return ref.second; }
    bool appendOperands(Ref ref, SmallVectorImpl<Ref>& operands) const
    {
        if (ref.second) {
            auto* op = ref.second.getDefiningOp();
            if (!mlir::pto::detail::canReplayScalar(op)) { return false; }
            for (auto operand : op->getOperands()) { operands.push_back(get(operand)); }
        } else {
            if (!arena || !arena->valid(ref.first)) { return false; }
            SmallVector<Id> inputs;
            arena->appendOperands(arena->nodes[ref.first], inputs);
            for (Id input : inputs) { operands.push_back(get(input)); }
        }
        return true;
    }
private:
    const RegionExpressions* arena;
};
bool isEndpointSelectionContainer(Operation* operation)
{
    // Physical-context projection materializes implicit section predicates
    // before transferring a proof. Raw implicitly guarded sections must first
    // go through that normalization; they are not unconditional containers.
    return isa<SectionCubeOp>(operation) ||
        (isa<SectionVectorOp>(operation) &&
         (operation->hasAttr("pto.sync_context_only") || !needsA5NoSplitVectorGuard(operation)));
}
namespace {
using Path = SmallVector<Region*>;
bool singletonConsumer(Operation* cut, const CutPredicate& view, CutPredicate::Ref predicate, func::FuncOp function)
{
    return detail::singletonCleanupConsumer(cut, view, predicate, function);
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
            if (isEndpointSelectionContainer(region->getParentOp())) {
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
                if (isa<scf::IfOp, scf::YieldOp>(op) || isEndpointSelectionContainer(op) || isMemoryEffectFree(op)) {
                    return;
                }
                auto name = op->getName().getStringRef();
                harmless &= name == "pto.logical_wait" || name == "pto.barrier";
            });
            if (!harmless) { return false; }
        }
        if (current->getParentOp() == function) { return current->getBlock() == &function.front(); }
        if (!isEndpointSelectionContainer(current->getParentOp())) { return false; }
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
namespace {
func::FuncOp terminalFunction(ArrayRef<TemplateEndpointCut> cuts)
{
    if (cuts.size() != 2 || !cuts[0].before || !cuts[1].before || cuts[0].before == cuts[1].before) {
        return {};
    }
    auto function = cuts[1].before->getParentOfType<func::FuncOp>();
    if (!function || !function.getBody().hasOneBlock() ||
        cuts[1].before != function.front().getTerminator() || !isa<func::ReturnOp>(cuts[1].before)) {
        return {};
    }
    for (const auto& cut : cuts) {
        if (cut.block != cut.before->getBlock() || cut.before->getParentOfType<func::FuncOp>() != function) {
            return {};
        }
    }
    return function;
}
} // namespace
std::shared_ptr<const EndpointCutChoices> qualifyTerminalCleanupCuts(
    ArrayRef<TemplateEndpointCut> cuts, Value consumerPredicate)
{
    auto function = terminalFunction(cuts);
    CutPredicate view;
    if (!function || !singletonConsumer(cuts[0].before, view, view.get(consumerPredicate), function)) {
        return nullptr;
    }
    return std::shared_ptr<const EndpointCutChoices>(new EndpointCutChoices({}, cuts, consumerPredicate));
}
bool qualifyTerminalCleanupSelection(ArrayRef<TemplateEndpointCut> cuts,
    const RegionExpressions& arena, uint32_t consumerPredicate)
{
    auto function = terminalFunction(cuts);
    CutPredicate view(&arena);
    auto predicate = view.get(consumerPredicate);
    return function && arena.constructionError().empty() && view.valid(predicate) && view.boolean(predicate) &&
        singletonConsumer(cuts[0].before, view, predicate, function);
}
std::shared_ptr<const EndpointCutChoices> qualifyTerminalCleanupCuts(
    ArrayRef<TemplateEndpointCut> cuts, std::shared_ptr<const RegionExpressions> arena, uint32_t predicate)
{
    if (!arena || !qualifyTerminalCleanupSelection(cuts, *arena, predicate)) { return {}; }
    auto result = std::shared_ptr<EndpointCutChoices>(new EndpointCutChoices({}, cuts));
    result->terminalArena = std::move(arena);
    result->terminalExpression = predicate;
    return result;
}
bool EndpointCutChoices::validPendingSelection() const
{
    return terminalArena && qualifyTerminalCleanupSelection(alternatives, *terminalArena, terminalExpression);
}
bool validateTerminalCleanupCommands(func::FuncOp function, ArrayRef<Operation*> commands)
{
    if (commands.size() != 2 || !function.getBody().hasOneBlock()) { return false; }
    unsigned terminal = 0;
    for (auto* command : commands) {
        auto wrapper = dyn_cast_or_null<scf::IfOp>(command->getParentOp());
        if (!wrapper || !wrapper->hasAttr("pto.endpoint_cut") ||
            command->getParentRegion() != &wrapper.getThenRegion()) { return false; }
        CutPredicate view;
        if (!singletonConsumer(wrapper, view, view.get(wrapper.getCondition()), function)) { return false; }
        terminal += terminalSuffix(wrapper, function);
    }
    return terminal >= 1;
}
bool validateEndpointCommandChoices(scf::ForOp loop, ArrayRef<Operation*> commands)
{
    return validate(loop, commands, true);
}
} // namespace mlir::pto::frontiersynch
