// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// One count per source pipe, one acquisition watermark per selected direction.
#include "InvocationPrefix.h"
#include "PrefixExpressionProfile.h"
#include "PrefixExpressionSession.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <algorithm>
#include <limits>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
bool PrefixDirection::operator<(const PrefixDirection& other) const
{
    return std::tie(sourcePipe, targetPipe) < std::tie(other.sourcePipe, other.targetPipe);
}
namespace {
struct Layout {
    std::map<uint32_t, std::size_t> counts;
    std::map<PrefixDirection, std::size_t> watermarks;
    std::map<PrefixDirection, uint32_t> records;
    int64_t plan = 0;
    std::size_t userSlots = 0;
};
void command(OpBuilder& b, Location loc, PrefixDirection direction, uint32_t record,
             int64_t plan, Value ordinal, bool publish)
{
    OperationState op(loc, publish ? "pto.logical_set" : "pto.logical_wait");
    op.addOperands(ordinal);
    op.addAttribute("src_pipe", PipeAttr::get(b.getContext(), static_cast<PIPE>(direction.sourcePipe)));
    op.addAttribute("dst_pipe", PipeAttr::get(b.getContext(), static_cast<PIPE>(direction.targetPipe)));
    op.addAttribute("record_id", b.getI64IntegerAttr(record));
    op.addAttribute("plan_id", b.getI64IntegerAttr(plan));
    b.create(op);
}
void acquire(OpBuilder& b, Location loc, const Layout& layout, PrefixDirection direction,
             Value required, llvm::MutableArrayRef<Value> state)
{
    auto& watermark = state[layout.watermarks.at(direction)];
    auto one = b.create<arith::ConstantIndexOp>(loc, 1);
    auto first = b.create<arith::AddIOp>(loc, watermark, one);
    auto end = b.create<arith::AddIOp>(loc, required, one);
    auto loop = b.create<scf::ForOp>(loc, first, end, one);
    {
        OpBuilder::InsertionGuard restore(b);
        b.setInsertionPointToStart(loop.getBody());
        command(b, loc, direction, layout.records.at(direction), layout.plan, loop.getInductionVar(), false);
    }
    auto later = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, required, watermark);
    watermark = b.create<arith::SelectOp>(loc, later, required, watermark);
}
LogicalResult check(func::FuncOp f, const InvocationPrefixRecipe& recipe, PrivateStateError& error)
{
    const bool basics = f && recipe.privateScalarControl && recipe.planId >= 0 &&
        f.getBody().hasOneBlock() && isa<func::ReturnOp>(f.front().getTerminator());
    if (!basics) {
        error = {PrivateStateFailure::InvalidInput, f, "prefix realization requires scalar control and one invocation"};
        return failure();
    }
    const auto bits = DataLayout::closest(f).getTypeSizeInBits(IndexType::get(f.getContext()));
    const bool width = !bits.isScalable() && bits.getFixedValue() == 64;
    if (!width) {
        error = {PrivateStateFailure::InvalidInput, f, "prefix realization requires 64-bit index control"};
        return failure();
    }
    for (auto [pipe, bound] : recipe.sourceBounds) {
        const bool invalidBound = bound >= uint64_t(INT64_MAX) || !isConcreteSyncPipe(static_cast<PIPE>(pipe));
        if (invalidBound) {
            error = {PrivateStateFailure::InvalidInput, f, "prefix source count and exclusive sentinel must fit index"};
            return failure();
        }
    }
    bool occupied = f->hasAttr("pto.prefix_protocol") || f->hasAttr("pto.endpoint_families");
    f.walk([&](Operation* op) { occupied |= isa<LogicalSetOp, LogicalWaitOp>(op); });
    const bool extent = recipe.directions.size() <= UINT32_MAX &&
        recipe.sourceBounds.size() <= recipe.slots.max_size() - recipe.slots.size() &&
        recipe.directions.size() <= recipe.slots.max_size() - recipe.slots.size() - recipe.sourceBounds.size();
    if (occupied || !extent) {
        error = {PrivateStateFailure::InvalidInput, f, "prefix requires a fresh namespace and representable state"};
        return failure();
    }
    std::set<PrefixDirection> directions;
    for (auto direction : recipe.directions) {
        const bool valid = direction.sourcePipe != direction.targetPipe &&
            isConcreteSyncPipe(static_cast<PIPE>(direction.targetPipe)) &&
            recipe.sourceBounds.count(direction.sourcePipe) && directions.insert(direction).second;
        if (!valid) {
            error = {PrivateStateFailure::InvalidInput, f, "prefix directions require distinct certified source pipes"};
            return failure();
        }
    }
    std::set<std::pair<Operation*, uint32_t>> sources;
    for (const auto& source : recipe.sources) {
        const bool valid = source.original && f->isProperAncestor(source.original) &&
            !source.original->hasTrait<OpTrait::IsTerminator>() && recipe.sourceBounds.count(source.pipe) &&
            sources.emplace(source.original, source.pipe).second;
        if (!valid) {
            error = {PrivateStateFailure::InvalidCut, source.original,
                     "prefix source occurrence has invalid provenance"};
            return failure();
        }
    }
    std::set<std::tuple<Operation*, uint32_t, uint32_t>> consumers;
    for (const auto& consumer : recipe.consumers) {
        const bool valid = consumer.original && f->isProperAncestor(consumer.original) && consumer.incoming &&
            directions.count(consumer.direction) &&
            consumers.emplace(consumer.original, consumer.direction.sourcePipe, consumer.direction.targetPipe).second;
        if (!valid) {
            error = {PrivateStateFailure::InvalidCut, consumer.original,
                     "prefix consumer lacks an exact incoming selector"};
            return failure();
        }
    }
    return success();
}
Layout appendSlots(func::FuncOp f, const InvocationPrefixRecipe& recipe, std::vector<PrivateStateSlot>& slots)
{
    Layout layout; layout.plan = recipe.planId; layout.userSlots = slots.size();
    auto type = IndexType::get(f.getContext());
    auto zero = IntegerAttr::get(type, 0);
    for (auto [pipe, bound] : recipe.sourceBounds) {
        layout.counts.emplace(pipe, slots.size()); slots.push_back({type, zero});
    }
    for (auto direction : recipe.directions) {
        layout.watermarks.emplace(direction, slots.size()); slots.push_back({type, zero});
        layout.records.emplace(direction, layout.records.size());
    }
    return layout;
}
PrivateStateAction sourceAction(const PrefixSource& source, const Layout& layout)
{
    return {source.original, PrivateStateCut::After,
        [source, layout](OpBuilder& b, const IRMapping& mapping,
                         llvm::MutableArrayRef<Value> state, std::string& error) {
            const auto loc = source.original->getLoc();
            auto& n = state[layout.counts.at(source.pipe)];
            auto one = b.create<arith::ConstantIndexOp>(loc, 1);
            n = b.create<arith::AddIOp>(loc, n, one);
            for (auto [direction, slot] : layout.watermarks) {
                if (direction.sourcePipe == source.pipe) {
                    command(b, loc, direction, layout.records.at(direction), layout.plan, n, true);
                }
            }
            return source.snapshot ?
                source.snapshot(b, mapping, state.take_front(layout.userSlots), n, error) : success();
        }};
}
PrivateStateAction consumerAction(const PrefixConsumer& consumer, const Layout& layout)
{
    return {consumer.original, PrivateStateCut::Before,
        [consumer, layout](OpBuilder& b, const IRMapping& mapping,
                           llvm::MutableArrayRef<Value> state, std::string& error) {
            auto required = consumer.incoming(b, mapping, state.take_front(layout.userSlots), error);
            const bool valid = succeeded(required) && *required && (*required).getType().isIndex();
            if (!valid) {
                if (error.empty()) { error = "exact incoming prefix selector did not produce an index"; }
                return failure();
            }
            acquire(b, consumer.original->getLoc(), layout, consumer.direction, *required, state);
            return success();
        }};
}
void metadata(func::FuncOp f, const InvocationPrefixRecipe& recipe, const Layout& layout)
{
    Builder b(f.getContext());
    SmallVector<Attribute> directions;
    for (auto [direction, record] : layout.records) {
        directions.push_back(b.getDictionaryAttr({
            b.getNamedAttr("source", b.getI64IntegerAttr(direction.sourcePipe)),
            b.getNamedAttr("target", b.getI64IntegerAttr(direction.targetPipe)),
            b.getNamedAttr("record", b.getI64IntegerAttr(record)),
            b.getNamedAttr("bound", b.getI64IntegerAttr(recipe.sourceBounds.at(direction.sourcePipe)))}));
    }
    f->setAttr("pto.prefix_protocol", b.getDictionaryAttr({
        b.getNamedAttr("version", b.getI64IntegerAttr(1)),
        b.getNamedAttr("plan", b.getI64IntegerAttr(recipe.planId)),
        b.getNamedAttr("directions", b.getArrayAttr(directions))}));
}
} // namespace
FailureOr<std::unique_ptr<StructuredPrivateState>> materializeInvocationPrefix(
    func::FuncOp original, const InvocationPrefixRecipe& recipe, PrivateStateError& error)
{
    if (failed(check(original, recipe, error))) { return failure(); }
    auto expressionScope = recipe.expressionSession ? recipe.expressionSession->begin() : nullptr;
    if (recipe.expressionSession && !expressionScope) {
        error = {PrivateStateFailure::InvalidInput, original, "prefix expression recipe is already materializing"};
        return failure();
    }
    if (recipe.expressionProfile) { recipe.expressionProfile->resetEmission(); }
    original.getContext()->getOrLoadDialect<PTODialect>();
    auto slots = recipe.slots;
    const auto layout = appendSlots(original, recipe, slots);
    std::vector<PrivateStateAction> actions;
    for (auto action : recipe.actions) {
        auto apply = action.apply;
        action.apply = [apply, layout](OpBuilder& b, const IRMapping& mapping,
            llvm::MutableArrayRef<Value> state, std::string& error) {
            if (!apply) { error = "prefix contribution has no state action"; return failure(); }
            return apply(b, mapping, state.take_front(layout.userSlots), error);
        };
        actions.push_back(std::move(action));
    }
    auto sources = recipe.sources;
    std::stable_sort(sources.begin(), sources.end(), [](const auto& a, const auto& b) { return a.pipe < b.pipe; });
    for (const auto& source : sources) { actions.push_back(sourceAction(source, layout)); }
    auto consumers = recipe.consumers;
    std::stable_sort(consumers.begin(), consumers.end(),
        [](const auto& a, const auto& b) { return a.direction < b.direction; });
    for (const auto& consumer : consumers) { actions.push_back(consumerAction(consumer, layout)); }
    actions.push_back({original.front().getTerminator(), PrivateStateCut::Before,
        [layout](OpBuilder& b, const IRMapping&, llvm::MutableArrayRef<Value> state, std::string&) {
            for (auto [direction, slot] : layout.watermarks) {
                acquire(b, b.getUnknownLoc(), layout, direction, state[layout.counts.at(direction.sourcePipe)], state);
            }
            return success();
        }});
    auto staged = materializeStructuredPrivateState(original, slots, actions, error);
    if (failed(staged)) { return failure(); }
    metadata((*staged)->function, recipe, layout);
    return staged;
}
} // namespace mlir::pto::frontiersynch
