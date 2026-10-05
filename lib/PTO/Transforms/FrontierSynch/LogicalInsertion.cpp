// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Transactional preflight, then direct logical endpoint insertion at actual cuts.
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <algorithm>
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
std::optional<int64_t> constant(Value value)
{
    APInt number;
    if (!matchPattern(value, m_ConstantInt(&number)) || !number.isSignedIntN(64)) {
        return std::nullopt;
    }
    return number.getSExtValue();
}
bool sameRecipe(const EndpointRecipe& a, const EndpointRecipe& b)
{
    return std::tie(a.record, a.source, a.target, a.pipe, a.displacement, a.kind) ==
           std::tie(b.record, b.source, b.target, b.pipe, b.displacement, b.kind);
}
bool sameCut(TemplateEndpointCut a, TemplateEndpointCut b)
{
    return a.block == b.block && a.before == b.before;
}
bool samePlan(const NumericTemplateEndpoints& a, const NumericTemplateEndpoints& b)
{
    if (!a.logical.error.empty() || a.outer != b.outer || a.anchors.size() != b.anchors.size() ||
        a.groups.size() != b.groups.size() || a.logical.recipes.size() != b.logical.recipes.size()) {
        return false;
    }
    for (auto [left, right] : llvm::zip(a.logical.recipes, b.logical.recipes)) {
        if (!sameRecipe(left, right)) {
            return false;
        }
    }
    for (auto [left, right] : llvm::zip(a.anchors, b.anchors)) {
        if (left.phase != right.phase || left.coordinates.size() != right.coordinates.size() ||
            !sameCut(left.before, right.before) || !sameCut(left.after, right.after)) {
            return false;
        }
        for (auto [x, y] : llvm::zip(left.coordinates, right.coordinates)) {
            if (x.loop != y.loop || x.induction != y.induction) {
                return false;
            }
        }
    }
    for (auto [left, right] : llvm::zip(a.groups, b.groups)) {
        if (!sameCut(left.cut, right.cut) || left.recipes != right.recipes) {
            return false;
        }
    }
    return true;
}
bool concretePipe(uint32_t pipe)
{
    // This is the logical operation's enum contract, not device qualification.
    return pipe <= static_cast<uint32_t>(PIPE::PIPE_FIX) && pipe != static_cast<uint32_t>(PIPE::PIPE_ALL);
}
bool coordinateFits(TemplateCoordinate coordinate)
{
    if (!coordinate.loop) {
        return false;
    }
    auto type = coordinate.loop.getInductionVar().getType();
    if (type.isIndex()) {
        return true;
    }
    auto integer = dyn_cast<IntegerType>(type);
    return integer && APInt(64, static_cast<uint64_t>(coordinate.induction), true).isSignedIntN(integer.getWidth());
}
bool validCuts(const NumericTemplateEndpoints& plan, DominanceInfo& dominance)
{
    for (const auto& group : plan.groups) {
        const auto cut = group.cut;
        if (!cut.block || !cut.before || cut.before->getBlock() != cut.block ||
            !plan.outer->isProperAncestor(cut.before)) {
            return false;
        }
        for (auto index : group.recipes) {
            if (index >= plan.logical.recipes.size()) {
                return false;
            }
            const auto& recipe = plan.logical.recipes[index];
            const auto type = recipe.kind == EndpointKind::Set ? recipe.source : recipe.target;
            if (type >= plan.anchors.size() || recipe.source >= plan.anchors.size() ||
                recipe.target >= plan.anchors.size() || !concretePipe(recipe.pipe) ||
                !concretePipe(static_cast<uint32_t>(plan.anchors[recipe.source].phase->kPipeValue)) ||
                !concretePipe(static_cast<uint32_t>(plan.anchors[recipe.target].phase->kPipeValue))) {
                return false;
            }
            for (auto coordinate : plan.anchors[type].coordinates) {
                if (!coordinateFits(coordinate) || !coordinate.loop->isProperAncestor(cut.before) ||
                    !dominance.properlyDominates(coordinate.loop.getInductionVar(), cut.before)) {
                    return false;
                }
            }
        }
    }
    return true;
}
const StructureNode* selectRoot(func::FuncOp function, const ProgramRecognition& program)
{
    const StructureNode* selected = nullptr;
    for (const auto& node : program.nodes) {
        if (!node.numericTemplate || node.numericTemplate->result.state != RecognitionState::Applicable) {
            continue;
        }
        if (selected || !node.numericTemplate->outer || !node.periodicAnalysis || !node.logicalEndpoints ||
            node.numericTemplate->outer->getParentOp() != function.getOperation()) {
            return nullptr;
        }
        selected = &node;
    }
    return selected;
}
bool freshScope(func::FuncOp function, const ProgramRecognition& program, const StructureNode& node)
{
    auto outer = node.numericTemplate->outer;
    for (const auto& payload : program.payloads) {
        if (!payload.phase || !payload.phase->elementOp || !outer->isProperAncestor(payload.phase->elementOp)) {
            return false;
        }
    }
    bool clean = true;
    function.walk([&](Operation* operation) {
        if (isa<LogicalSetOp, LogicalWaitOp, SetFlagOp, WaitFlagOp, SetFlagDynOp, WaitFlagDynOp,
                RecordEventOp, WaitEventOp, BarrierOp>(operation)) {
            clean = false;
        }
    });
    return clean;
}
LogicalResult preflight(func::FuncOp function, const ProgramRecognition& program, const StructureNode*& node)
{
    node = selectRoot(function, program);
    if (!node || !function.getBody().hasOneBlock()) {
        return function.emitError("logical insertion requires one complete whole-function numeric template");
    }
    // Numeric-template arithmetic and logical occurrence identities use 64-bit
    // index values. Reject an explicit incompatible IR layout before mutation.
    const auto indexBits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (indexBits.isScalable() || indexBits.getFixedValue() != 64) {
        return function.emitError("logical insertion requires the numeric template's 64-bit index representation");
    }
    const auto& input = *node->numericTemplate;
    auto outer = input.outer;
    auto lower = constant(outer.getLowerBound()), step = constant(outer.getStep());
    const bool bounds = lower && step && *lower == input.lower && *step == input.step && *step > 0 &&
        outer.getInductionVar().getType().isIndex();
    auto expected = buildNumericTemplateEndpoints(input, *node->periodicAnalysis);
    DominanceInfo dominance(function);
    const bool registered = RegisteredOperationName::lookup("pto.logical_set", function.getContext()) &&
        RegisteredOperationName::lookup("pto.logical_wait", function.getContext());
    if (!bounds || !registered || !expected.logical.error.empty() || !samePlan(*node->logicalEndpoints, expected) ||
        !freshScope(function, program, *node) || !validCuts(expected, dominance)) {
        return function.emitError("logical insertion has an unsupported or unavailable endpoint obligation");
    }
    return success();
}
class Emitter {
public:
    explicit Emitter(const NumericTemplateEndpoints& plan) : plan(plan), builder(plan.outer->getContext()) {}
    void run()
    {
        if (plan.logical.recipes.empty()) {
            return;
        }
        initializeOrdinals();
        for (const auto& group : plan.groups) {
            emitCut(group);
        }
    }
private:
    const NumericTemplateEndpoints& plan;
    OpBuilder builder;
    Value trips;
    Value ordinal;
    Value zero;
    Value one;
    Value number(uint64_t value, Location location)
    {
        auto type = builder.getIndexType();
        return builder.create<arith::ConstantOp>(location, type, IntegerAttr::get(type, APInt(64, value)));
    }
    Value compare(arith::CmpIPredicate predicate, Value a, Value b, Location location)
    {
        return builder.create<arith::CmpIOp>(location, predicate, a, b);
    }
    void initializeOrdinals()
    {
        auto loop = plan.outer;
        auto location = loop.getLoc();
        builder.setInsertionPoint(loop);
        zero = number(0, location);
        one = number(1, location);
        auto positive = compare(arith::CmpIPredicate::sgt, loop.getUpperBound(), loop.getLowerBound(), location);
        auto difference = builder.create<arith::SubIOp>(location, loop.getUpperBound(), loop.getLowerBound());
        auto span = builder.create<arith::SelectOp>(location, positive, difference, zero);
        auto quotient = builder.create<arith::DivUIOp>(location, span, loop.getStep());
        auto remainder = builder.create<arith::RemUIOp>(location, span, loop.getStep());
        auto partial = compare(arith::CmpIPredicate::ne, remainder, zero, location);
        auto extra = builder.create<arith::SelectOp>(location, partial, one, zero);
        trips = builder.create<arith::AddIOp>(location, quotient, extra);
        builder.setInsertionPointToStart(loop.getBody());
        auto offset = builder.create<arith::SubIOp>(location, loop.getInductionVar(), loop.getLowerBound());
        ordinal = builder.create<arith::DivUIOp>(location, offset, loop.getStep());
    }
    Value predicate(const EndpointRecipe& recipe, Location location)
    {
        auto delay = number(recipe.displacement, location);
        auto selected = compare(arith::CmpIPredicate::ult, ordinal, trips, location);
        Value boundary;
        if (recipe.kind == EndpointKind::Set) {
            auto remaining = builder.create<arith::SubIOp>(location, trips, delay);
            auto hasTarget = compare(arith::CmpIPredicate::ult, delay, trips, location);
            auto beforeEnd = compare(arith::CmpIPredicate::ult, ordinal, remaining, location);
            boundary = builder.create<arith::AndIOp>(location, hasTarget, beforeEnd);
        } else {
            boundary = compare(arith::CmpIPredicate::uge, ordinal, delay, location);
        }
        selected = builder.create<arith::AndIOp>(location, selected, boundary);
        const auto type = recipe.kind == EndpointKind::Set ? recipe.source : recipe.target;
        for (auto coordinate : plan.anchors[type].coordinates) {
            auto induction = coordinate.loop.getInductionVar();
            auto expected = builder.create<arith::ConstantOp>(location, induction.getType(),
                builder.getIntegerAttr(induction.getType(), coordinate.induction));
            auto equal = compare(arith::CmpIPredicate::eq, induction, expected, location);
            selected = builder.create<arith::AndIOp>(location, selected, equal);
        }
        return selected;
    }
    void command(const EndpointRecipe& recipe, Value guard, Location location)
    {
        auto branch = builder.create<scf::IfOp>(location, guard, false);
        OpBuilder::InsertionGuard restore(builder);
        builder.setInsertionPointToStart(branch.thenBlock());
        const auto producer = plan.anchors[recipe.source].phase->kPipeValue;
        const auto consumer = plan.anchors[recipe.target].phase->kPipeValue;
        auto identity = ordinal;
        if (recipe.kind == EndpointKind::Wait) {
            identity = builder.create<arith::SubIOp>(location, ordinal, number(recipe.displacement, location));
        }
        OperationState state(location, recipe.kind == EndpointKind::Set ? "pto.logical_set" : "pto.logical_wait");
        state.addOperands(identity);
        state.addAttribute("src_pipe", PipeAttr::get(builder.getContext(), static_cast<PIPE>(producer)));
        state.addAttribute("dst_pipe", PipeAttr::get(builder.getContext(), static_cast<PIPE>(consumer)));
        state.addAttribute("plan_id", builder.getI64IntegerAttr(0));
        state.addAttribute("record_id", builder.getI64IntegerAttr(recipe.record));
        builder.create(state);
    }
    void emitCut(const TemplateEndpointGroup& group)
    {
        builder.setInsertionPoint(group.cut.before);
        auto location = group.cut.before->getLoc();
        SmallVector<std::pair<const EndpointRecipe*, Value>> selected;
        std::map<uint32_t, Value> barriers;
        for (auto index : group.recipes) {
            const auto& recipe = plan.logical.recipes[index];
            auto guard = predicate(recipe, location);
            if (recipe.kind != EndpointKind::Barrier) {
                selected.push_back({&recipe, guard});
            } else {
                auto [entry, inserted] = barriers.emplace(recipe.pipe, guard);
                if (!inserted) {
                    entry->second = builder.create<arith::OrIOp>(location, entry->second, guard);
                }
            }
        }
        for (const auto& entry : selected) {
            if (entry.first->kind == EndpointKind::Set) {
                command(*entry.first, entry.second, location);
            }
        }
        for (const auto& [pipe, guard] : barriers) {
            auto branch = builder.create<scf::IfOp>(location, guard, false);
            OpBuilder::InsertionGuard restore(builder);
            builder.setInsertionPointToStart(branch.thenBlock());
            builder.create<BarrierOp>(location, PipeAttr::get(builder.getContext(), static_cast<PIPE>(pipe)));
        }
        for (const auto& entry : selected) {
            if (entry.first->kind == EndpointKind::Wait) {
                command(*entry.first, entry.second, location);
            }
        }
    }
};
} // namespace
LogicalResult insertLogicalSynchronization(func::FuncOp function, const ProgramRecognition& program)
{
    if (!function) {
        return failure();
    }
    const StructureNode* node = nullptr;
    if (failed(preflight(function, program, node))) {
        return failure();
    }
    Emitter emitter(*node->logicalEndpoints);
    emitter.run();
    return success();
}
} // namespace mlir::pto::frontiersynch
