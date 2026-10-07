// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Analyze whole physical executions, then insert at their original lexical cuts.
// All edits are transactional on a working copy; projection size is at most twice
// the static function size and never depends on loop trip counts.
#include "PTO/Transforms/FrontierSynch/ExecutionContexts.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
namespace {
StringRef sectionContext(Operation* op)
{
    if (isa<SectionCubeOp>(op)) { return "cube"; }
    if (isa<SectionVectorOp>(op)) { return "vector"; }
    return {};
}
StringRef contextOf(Operation* op)
{
    for (; op && !isa<func::FuncOp>(op); op = op->getParentOp()) {
        if (auto attr = op->getAttrOfType<StringAttr>(ContextAttr)) { return attr.getValue(); }
        auto core = sectionContext(op);
        if (!core.empty()) { return core; }
    }
    return {};
}
struct FunctionCopy {
    OwningOpRef<ModuleOp> module;
    IRMapping mapping;
    func::FuncOp function;
    explicit FunctionCopy(func::FuncOp original)
        : module(ModuleOp::create(original.getLoc()))
    {
        if (auto parent = original->getParentOfType<ModuleOp>()) {
            (*module)->setAttrs(parent->getAttrs());
            // Symbol metadata is needed by shared helper-call interpretation.
            for (auto sibling : parent.getOps<func::FuncOp>()) {
                if (sibling == original) { continue; }
                module->push_back(sibling.cloneWithoutRegions());
            }
        }
        function = cast<func::FuncOp>(original->clone(mapping));
        module->push_back(function);
    }
    void commit(func::FuncOp original)
    {
        original->setAttrs(function->getAttrs());
        original.getBody().takeBody(function.getBody());
    }
};
// EmitC normally supplies this predicate implicitly. Materialize it before
// projection so dependencies and both synchronization endpoints see it too.
void materializeSectionPredicates(func::FuncOp function)
{
    SmallVector<SectionVectorOp> guarded;
    const bool functionGuard = needsA5NoSplitVectorGuard(function);
    function.walk([&](SectionVectorOp section) {
        if (!section->hasAttr("pto.sync_context_only") && needsA5NoSplitVectorGuard(section)) {
            guarded.push_back(section);
        }
    });
    if (guarded.empty()) { return; }
    function.getContext()->getOrLoadDialect<scf::SCFDialect>();
    // Capture the function-wide lowering decision before the new explicit query
    // changes hasExplicitSubblockControl. This predicate applies uniformly to
    // the entire invocation, including inserted completion commands.
    OpBuilder b(&function.front(), function.front().begin());
    function->setAttr("pto.sync_subblock_guard", b.getBoolAttr(functionGuard));
    auto subblock = b.create<GetSubBlockIdxOp>(function.getLoc(), b.getI64Type());
    auto zero = b.create<arith::ConstantIntOp>(function.getLoc(), 0, 64);
    auto predicate = b.create<arith::CmpIOp>(function.getLoc(), arith::CmpIPredicate::eq, subblock, zero);
    for (auto section : guarded) {
        auto& body = section.getBody().front();
        SmallVector<Operation*> original;
        for (auto& op : body) { original.push_back(&op); }
        b.setInsertionPointToStart(&body);
        auto branch = b.create<scf::IfOp>(section.getLoc(), predicate, false);
        for (auto* op : original) { op->moveBefore(branch.thenBlock()->getTerminator()); }
        section->setAttr("pto.sync_subblock_guard", b.getBoolAttr(false));
    }
}

struct Cut { Block* block = nullptr; Operation* before = nullptr; };
class Projection {
public:
    Projection(func::FuncOp original, StringRef context) : copy(original), context(context)
    {
        for (auto [source, target] : copy.mapping.getValueMap()) { inverse.map(target, source); }
        for (auto [source, target] : copy.mapping.getOperationMap()) {
            inverse.map(target, source);
            cuts[target] = {source->getBlock(), source};
        }
        copy.function->setAttr(FunctionKernelKindAttr::name, FunctionKernelKindAttr::get(
            original.getContext(), context == "cube" ? FunctionKernelKind::Cube : FunctionKernelKind::Vector));
    }
    LogicalResult build()
    {
        SmallVector<Operation*> sections;
        copy.function.walk<WalkOrder::PreOrder>([&](Operation* op) {
            if (!sectionContext(op).empty()) { sections.push_back(op); }
        });
        // Inner sections first: their lexical cuts stay distinct after inlining.
        for (auto* op : llvm::reverse(sections)) {
            if (sectionContext(op) != context) { op->erase(); continue; }
            auto* source = inverse.lookupOrNull(op);
            if (!source || !op->getRegion(0).hasOneBlock()) { return failure(); }
            auto& body = op->getRegion(0).front();
            auto& originalBody = source->getRegion(0).front();
            OpBuilder b(op);
            auto entry = b.create<arith::ConstantIndexOp>(op->getLoc(), 0);
            auto exit = b.create<arith::ConstantIndexOp>(op->getLoc(), 0);
            cuts[entry] = {&originalBody, originalBody.empty() ? nullptr : &originalBody.front()};
            cuts[exit] = {&originalBody, nullptr};
            op->getBlock()->getOperations().splice(exit->getIterator(), body.getOperations());
            op->erase();
        }
        return success();
    }
    LogicalResult remap(PreparedLogicalPlan& plan, DenseMap<Block*, Operation*>& ends)
    {
        auto anchor = [&](Operation* op) -> Operation* {
            auto found = cuts.find(op);
            if (found == cuts.end() || !found->second.block) { return nullptr; }
            if (found->second.before) { return found->second.before; }
            auto& end = ends[found->second.block];
            if (!end) {
                OpBuilder b(copy.function.getContext());
                b.setInsertionPointToEnd(found->second.block);
                end = b.create<arith::ConstantIndexOp>(copy.function.getLoc(), 0);
            }
            return end;
        };
        for (auto& stage : plan.preparation) {
            stage.before = anchor(stage.before);
            if (!stage.before) { return failure(); }
            stage.code->walk([&](Operation* op) {
                for (auto& operand : op->getOpOperands()) {
                    operand.set(inverse.lookupOrDefault(operand.get()));
                }
            });
        }
        for (auto& endpoint : plan.endpoints) {
            endpoint.before = anchor(endpoint.before);
            if (!endpoint.before) { return failure(); }
            endpoint.guard = inverse.lookupOrDefault(endpoint.guard);
            if (endpoint.identity) { endpoint.identity = inverse.lookupOrDefault(endpoint.identity); }
            for (auto& coordinate : endpoint.memberCoordinates) { coordinate = inverse.lookupOrDefault(coordinate); }
        }
        for (auto& family : plan.families) {
            family.sourceCut.before = anchor(family.sourceCut.before);
            if (!family.sourceCut.before) { return failure(); }
            family.sourceCut.block = family.sourceCut.before->getBlock();
            family.targetCut.before = anchor(family.targetCut.before);
            if (!family.targetCut.before) { return failure(); }
            family.targetCut.block = family.targetCut.before->getBlock();
            for (auto& member : family.members) {
                for (auto* tuple : {&member.sourceCoordinates, &member.targetCoordinates}) {
                    for (auto& coordinate : *tuple) {
                        coordinate.loop = dyn_cast_or_null<scf::ForOp>(inverse.lookupOrNull(coordinate.loop));
                        if (!coordinate.loop) { return failure(); }
                    }
                }
            }
        }
        return success();
    }
    FunctionCopy copy;
    StringRef context;
private:
    IRMapping inverse;
    DenseMap<Operation*, Cut> cuts;
};
SmallVector<StringRef> contexts(func::FuncOp function)
{
    bool cube = false, vector = false;
    function.walk([&](Operation* op) {
        cube |= isa<SectionCubeOp>(op);
        vector |= isa<SectionVectorOp>(op);
    });
    auto core = recoverSyncPhysicalCore(function);
    if (core == SyncPhysicalCore::AIC || core == SyncPhysicalCore::AIV) {
        return {core == SyncPhysicalCore::AIC ? "cube" : "vector"};
    }
    SmallVector<StringRef> result;
    if (cube) { result.push_back("cube"); }
    if (vector) { result.push_back("vector"); }
    return result;
}
// Projection can share total arithmetic across section boundaries. Hoist only
// preparation whose operands are already available outside the original section;
// source payload/control definitions are never moved or duplicated.
void liftPreparation(func::FuncOp function, PreparedLogicalPlan& plan)
{
    DominanceInfo dominance(function);
    DenseMap<Value, Operation*> placement;
    std::vector<LogicalPreparation> result;
    for (auto& stage : plan.preparation) {
        Operation* outside = nullptr;
        for (auto* parent = stage.before->getParentOp(); parent && parent != function;
             parent = parent->getParentOp()) {
            if (!sectionContext(parent).empty()) { outside = parent; }
        }
        std::unique_ptr<Block> lifted = std::make_unique<Block>();
        for (auto& op : llvm::make_early_inc_range(*stage.code)) {
            bool canLift = outside && !op.getNumRegions() && isMemoryEffectFree(&op) && isSpeculatable(&op);
            for (auto operand : op.getOperands()) {
                auto found = placement.find(operand);
                if (found != placement.end()) {
                    canLift &= outside &&
                        (found->second == outside || dominance.properlyDominates(found->second, outside));
                } else {
                    auto* definition = operand.getDefiningOp();
                    canLift &= operand.getParentRegion() && (!definition || definition->getParentOp()) &&
                        outside && dominance.properlyDominates(operand, outside);
                }
            }
            for (auto value : op.getResults()) { placement[value] = canLift ? outside : stage.before; }
            if (canLift) { op.moveBefore(lifted.get(), lifted->end()); }
        }
        if (!lifted->empty()) { result.push_back({outside, std::move(lifted)}); }
        result.push_back(std::move(stage));
    }
    plan.preparation = std::move(result);
}
void storeLoopIds(func::FuncOp function, StringRef context)
{
    Builder b(function.getContext());
    function.walk([&](scf::ForOp loop) {
        if (auto id = loop->removeAttr("pto.family_loop")) {
            NamedAttrList entries(loop->getAttrOfType<DictionaryAttr>(ContextLoopsAttr));
            entries.set(context, id);
            loop->setAttr(ContextLoopsAttr, entries.getDictionary(b.getContext()));
        }
    });
}
void guardNewCommands(func::FuncOp function, StringRef context, const DenseSet<Operation*>& old)
{
    SmallVector<Operation*> roots;
    function.walk<WalkOrder::PreOrder>([&](Operation* op) {
        if (old.contains(op)) { return WalkResult::advance(); }
        roots.push_back(op);
        return WalkResult::skip();
    });
    for (auto* op : roots) {
        op->setAttr(ContextAttr, StringAttr::get(function.getContext(), context));
        if (contextOf(op->getParentOp()) == context || isMemoryEffectFree(op)) { continue; }
        OpBuilder b(op);
        Operation* wrapper = context == "cube" ?
            b.create<SectionCubeOp>(op->getLoc()).getOperation() :
            b.create<SectionVectorOp>(op->getLoc()).getOperation();
        // This wrapper selects a compiled core only. It must not introduce the
        // extra subblock-zero predicate used by ordinary no-split A5 sections.
        wrapper->setAttr("pto.sync_context_only", b.getUnitAttr());
        auto& body = wrapper->getRegion(0).emplaceBlock();
        op->moveBefore(&body, body.end());
    }
}
DictionaryAttr takePlan(func::FuncOp function)
{
    NamedAttrList attrs;
    for (StringRef name : {StringRef("pto.endpoint_families"), StringRef(FiniteAllocationAttr),
                           StringRef(CyclicAllocationAttr)}) {
        if (auto value = function->removeAttr(name)) { attrs.set(name, value); }
    }
    return attrs.getDictionary(function.getContext());
}
} // namespace
bool hasPhysicalSections(func::FuncOp function)
{
    return function.walk([](Operation* op) {
        return sectionContext(op).empty() ? WalkResult::advance() : WalkResult::interrupt();
    }).wasInterrupted();
}
bool belongsToActiveContext(func::FuncOp function, Operation* operation)
{
    auto active = function->getAttrOfType<StringAttr>(ActiveContextAttr);
    if (!active) { return true; }
    auto context = contextOf(operation);
    return context.empty() || context == active.getValue();
}
LogicalResult insertContextSynchronization(func::FuncOp function, PrepareContext prepare)
{
    if (function->hasAttr(ContextPlansAttr)) { return function.emitError("context plans already exist"); }
    FunctionCopy transaction(function);
    auto working = transaction.function;
    materializeSectionPredicates(working);
    std::vector<std::unique_ptr<Projection>> projections;
    std::vector<std::unique_ptr<PreparedLogicalPlan>> plans;
    DenseMap<Block*, Operation*> ends;
    for (auto context : contexts(working)) {
        auto projection = std::make_unique<Projection>(working, context);
        if (failed(projection->build())) { return function.emitError("invalid physical section projection"); }
        auto plan = prepare(projection->copy.function);
        if (failed(plan)) { return function.emitError() << "analysis failed in " << context << " execution"; }
        if (failed(projection->remap(**plan, ends))) {
            return function.emitError("section endpoint cannot be mapped to an original cut");
        }
        liftPreparation(working, **plan);
        (*plan)->planId = plans.size();
        if ((*plan)->allocationCertificate) {
            NamedAttrList attrs((*plan)->allocationCertificate);
            attrs.set("plan", IntegerAttr::get(IntegerType::get(function.getContext(), 64), (*plan)->planId));
            (*plan)->allocationCertificate = attrs.getDictionary(function.getContext());
        }
        projections.push_back(std::move(projection));
        plans.push_back(std::move(*plan));
    }
    NamedAttrList saved;
    for (std::size_t i = 0; i < plans.size(); ++i) {
        DenseSet<Operation*> old;
        working.walk([&](Operation* op) { old.insert(op); });
        if (failed(insertLogicalSynchronization(working, *plans[i]))) { return failure(); }
        auto context = projections[i]->context;
        guardNewCommands(working, context, old);
        storeLoopIds(working, context);
        saved.set(context, takePlan(working));
    }
    for (auto [block, end] : ends) { end->erase(); }
    working->setAttr(ContextPlansAttr, saved.getDictionary(function.getContext()));
    if (failed(verify(working))) { return function.emitError("context insertion violates original SSA scope"); }
    transaction.commit(function);
    return success();
}
LogicalResult allocateContextSynchronization(func::FuncOp function, ArrayRef<int64_t> eligibleIds)
{
    auto bundle = function->getAttrOfType<DictionaryAttr>(ContextPlansAttr);
    if (!bundle || bundle.empty()) { return function.emitError("missing execution-context certificates"); }
    DenseSet<int64_t> planIds;
    for (auto entry : bundle) {
        auto saved = dyn_cast<DictionaryAttr>(entry.getValue());
        if (!saved || (entry.getName() != "cube" && entry.getName() != "vector")) {
            return function.emitError("malformed execution-context certificate");
        }
        for (auto attr : saved) {
            if (attr.getName() != "pto.endpoint_families" && attr.getName() != FiniteAllocationAttr &&
                attr.getName() != CyclicAllocationAttr) {
                return function.emitError("unexpected execution-context certificate field");
            }
        }
        auto finite = saved.getAs<DictionaryAttr>(FiniteAllocationAttr);
        auto cyclic = saved.getAs<DictionaryAttr>(CyclicAllocationAttr);
        if (finite && cyclic) { return function.emitError("ambiguous execution-context allocation"); }
        if (auto certificate = finite ? finite : cyclic) {
            auto id = certificate.getAs<IntegerAttr>("plan");
            if (!id || !id.getType().isInteger(64) || id.getInt() < 0 || !planIds.insert(id.getInt()).second) {
                return function.emitError("missing or duplicate execution-context plan identity");
            }
        }
    }
    auto covered = function.walk([&](Operation* op) -> WalkResult {
        if (!isa<LogicalSetOp, LogicalWaitOp>(op)) { return WalkResult::advance(); }
        auto context = contextOf(op);
        auto saved = context.empty() ? DictionaryAttr() : bundle.getAs<DictionaryAttr>(context);
        auto finite = saved ? saved.getAs<DictionaryAttr>(FiniteAllocationAttr) : DictionaryAttr();
        auto cyclic = saved ? saved.getAs<DictionaryAttr>(CyclicAllocationAttr) : DictionaryAttr();
        auto certificate = finite ? finite : cyclic;
        auto expected = certificate ? certificate.getAs<IntegerAttr>("plan") : IntegerAttr();
        auto actual = op->getAttrOfType<IntegerAttr>("plan_id");
        if (!expected || expected != actual) {
            op->emitError("logical endpoint lacks its execution-context allocation certificate");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    if (covered.wasInterrupted()) { return failure(); }
    FunctionCopy transaction(function);
    auto working = transaction.function;
    for (auto entry : bundle) {
        auto saved = dyn_cast<DictionaryAttr>(entry.getValue());
        if (!saved || (entry.getName() != "cube" && entry.getName() != "vector")) {
            return function.emitError("malformed execution-context certificate");
        }
        working->setAttr(ActiveContextAttr, entry.getName());
        for (auto attr : saved) { working->setAttr(attr.getName(), attr.getValue()); }
        working.walk([&](scf::ForOp loop) {
            if (auto ids = loop->getAttrOfType<DictionaryAttr>(ContextLoopsAttr)) {
                if (auto id = ids.get(entry.getName())) { loop->setAttr("pto.family_loop", id); }
            }
        });
        if (failed(allocatePhysicalEventIds(working, eligibleIds))) { return failure(); }
    }
    if (working.walk([](Operation* op) {
            return isa<LogicalSetOp, LogicalWaitOp>(op) ? WalkResult::interrupt() : WalkResult::advance();
        }).wasInterrupted()) {
        return function.emitError("execution-context allocation left unresolved logical endpoints");
    }
    working->removeAttr(ActiveContextAttr);
    working->removeAttr(ContextPlansAttr);
    working.walk([](Operation* op) { op->removeAttr(ContextLoopsAttr); op->removeAttr(ContextAttr); });
    if (failed(verify(working))) { return failure(); }
    transaction.commit(function);
    return success();
}
} // namespace mlir::pto::frontiersynch
