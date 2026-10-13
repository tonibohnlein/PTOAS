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
// One structural interpreter for exact predicates, before and after scalar
// lowering. Arena inputs are original SSA leaves, not a second control model.
class CutPredicate {
public:
    using Id = RegionExpressions::Id;
    using Ref = std::pair<Id, Value>;
    enum class Binary { And, Equal, Subtract, Divide };
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
    void conjuncts(Ref root, SmallVectorImpl<Ref>& atoms) const
    {
        SmallVector<Ref> work{root};
        llvm::DenseSet<Ref> seen;
        while (!work.empty()) {
            auto ref = work.pop_back_val();
            if (!valid(ref) || !seen.insert(ref).second) { continue; }
            auto both = binary(ref, Binary::And);
            if (!both) { both = masked(ref); }
            if (both) { work.push_back(both->first); work.push_back(both->second); }
            else { atoms.push_back(ref); }
        }
    }
    bool ordinal(Ref ref, scf::ForOp loop) const
    {
        // Unsigned distance/positive step is the exact visit coordinate,
        // including a signed counted-loop interval crossing zero.
        auto iv = get(loop.getInductionVar());
        if (ref == iv) { return true; }
        if (auto division = binary(ref, Binary::Divide)) {
            auto domain = CountedLoop::get(loop);
            if (!domain || number(division->second) != static_cast<uint64_t>(domain->step)) { return false; }
            ref = division->first;
            if (ref == iv && number(get(loop.getLowerBound())) == 0) { return true; }
        }
        auto difference = binary(ref, Binary::Subtract);
        if (!difference || difference->first != iv) { return false; }
        auto lower = get(loop.getLowerBound());
        return difference->second == lower ||
            (number(difference->second) && number(difference->second) == number(lower));
    }
    bool invariant(Ref root, Operation* outer, DominanceInfo& dominance) const
    {
        SmallVector<Ref> work{root};
        llvm::DenseSet<Ref> seen;
        while (!work.empty()) {
            auto ref = work.pop_back_val();
            if (!valid(ref)) { return false; }
            if (!seen.insert(ref).second) { continue; }
            if (ref.second) {
                auto* region = ref.second.getParentRegion();
                if (region && region->getParentOp() && dominance.properlyDominates(ref.second, outer)) {
                    continue;
                }
                auto* op = ref.second.getDefiningOp();
                if (!mlir::pto::detail::canReplayScalar(op)) { return false; }
                for (auto operand : op->getOperands()) { work.push_back(get(operand)); }
            } else {
                SmallVector<Id> operands;
                arena->appendOperands(arena->nodes[ref.first], operands);
                for (Id operand : operands) { work.push_back(get(operand)); }
            }
        }
        return true;
    }
private:
    const RegionExpressions* arena;
};
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
bool singletonConsumer(Operation* cut, const CutPredicate& view, CutPredicate::Ref predicate, func::FuncOp function)
{
    if (view.valid(predicate) && !view.boolean(predicate)) { return false; }
    SmallVector<scf::ForOp> loops;
    SmallVector<CutPredicate::Ref> atoms;
    view.conjuncts(predicate, atoms);
    auto* descendant = cut;
    for (auto* parent = cut->getParentOp(); parent != function; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            if (!CountedLoop::get(loop)) { return false; }
            loops.push_back(loop);
        } else if (auto branch = dyn_cast<scf::IfOp>(parent)) {
            if (descendant->getParentRegion() == &branch.getThenRegion()) {
                view.conjuncts(view.get(branch.getCondition()), atoms);
            }
        } else if (!transparentSection(parent)) { return false; }
        descendant = parent;
    }
    if (loops.empty()) { return true; }
    if (!view.valid(predicate)) { return false; }
    if (view.number(predicate) == 0) { return true; }
    DominanceInfo dominance(function);
    for (auto loop : loops) {
        bool selected = llvm::any_of(atoms, [&](CutPredicate::Ref atom) {
            auto equality = view.binary(atom, CutPredicate::Binary::Equal);
            if (!equality) { return false; }
            return (view.ordinal(equality->first, loop) && view.invariant(equality->second, loops.back(), dominance)) ||
                (view.ordinal(equality->second, loop) && view.invariant(equality->first, loops.back(), dominance));
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
