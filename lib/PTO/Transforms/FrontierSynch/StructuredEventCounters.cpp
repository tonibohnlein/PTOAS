// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Rebuild only SCF signatures; takeBody preserves payload identity and order.
// Every state is a bounded index into its eligible-ID vector, initialized once
// at invocation entry and advanced only alongside its physical command.
#include "StructuredEventCounters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
LogicalResult reject(std::string& reason, const char* message)
{
    reason = message;
    return failure();
}
LogicalResult checkControl(Operation* operation, std::string& reason)
{
    if (operation->getNumRegions() == 0) { return success(); }
    if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp>(operation)) {
        return reject(reason, "structured event counters require scf.for, scf.if, or scf.while region controls");
    }
    unsigned expectedRegions = isa<scf::ForOp>(operation) ? 1 : 2;
    if (operation->getNumRegions() != expectedRegions) {
        return reject(reason, "structured event counters require standard SCF region counts");
    }
    for (auto [index, region] : llvm::enumerate(operation->getRegions())) {
        if (region.empty() && isa<scf::IfOp>(operation) && index == 1) { continue; }
        if (!llvm::hasSingleElement(region) || region.front().empty()) {
            return reject(reason, "structured event counters do not support multi-block or empty control regions");
        }
        auto* terminator = region.front().getTerminator();
        bool conditionRegion = isa<scf::WhileOp>(operation) && index == 0;
        if ((conditionRegion && !isa<scf::ConditionOp>(terminator)) ||
            (!conditionRegion && !isa<scf::YieldOp>(terminator))) {
            return reject(reason, "structured event counters require standard SCF region terminators");
        }
    }
    return success();
}
LogicalResult checkPools(ArrayRef<StructuredEventPool> pools, std::string& reason)
{
    if (pools.size() > std::numeric_limits<std::size_t>::max() / 2) {
        return reject(reason, "structured event counter state size exceeds supported bounds");
    }
    for (const auto& pool : pools) {
        if (pool.eligibleIds.empty()) {
            return reject(reason, "structured event counters require a nonempty certified eligible-ID pool");
        }
        llvm::SmallSet<unsigned, 8> seen;
        for (unsigned id : pool.eligibleIds) {
            if (id > static_cast<unsigned>(EVENT::EVENT_ID7) || !seen.insert(id).second) {
                return reject(reason, "structured event pool IDs must be distinct supported event IDs");
            }
        }
    }
    return success();
}
LogicalResult checkEndpoints(func::FuncOp pending, ArrayRef<StructuredEventPool> pools,
    const DenseMap<Operation*, std::size_t>& endpoints, std::string& reason)
{
    std::size_t count = 0;
    bool invalid = false;
    pending.walk([&](Operation* operation) {
        if (!isa<LogicalSetOp, LogicalWaitOp>(operation)) { return; }
        ++count;
        auto found = endpoints.find(operation);
        if (found == endpoints.end() || found->second >= pools.size()) { invalid = true; return; }
        const auto& pool = pools[found->second];
        auto source = operation->getAttrOfType<PipeAttr>("src_pipe");
        auto target = operation->getAttrOfType<PipeAttr>("dst_pipe");
        if (!source || !target || source.getPipe() != pool.source || target.getPipe() != pool.target) {
            invalid = true;
        }
    });
    if (invalid || count != endpoints.size()) {
        return reject(reason, "structured event counter endpoints must exactly match their certified pools");
    }
    return success();
}
using Counters = SmallVector<Value>;
class CounterLowering {
public:
    CounterLowering(ArrayRef<StructuredEventPool> pools,
        const DenseMap<Operation*, std::size_t>& endpoints)
        : pools(pools), endpoints(endpoints)
    {
        for (const auto& pool : pools) {
            offsets.push_back(slots);
            if (pool.eligibleIds.size() > 1) { slots += 2; }
        }
    }
    void run(func::FuncOp function)
    {
        Counters counters;
        if (slots != 0) {
            OpBuilder builder(&function.getBody().front(), function.getBody().front().begin());
            Value zero = builder.create<arith::ConstantIndexOp>(function.getLoc(), 0);
            counters.assign(slots, zero);
        }
        lowerBlock(function.getBody().front(), counters);
    }
private:
    ArrayRef<StructuredEventPool> pools;
    const DenseMap<Operation*, std::size_t>& endpoints;
    SmallVector<std::size_t> offsets;
    std::size_t slots = 0;

    Value lookup(OpBuilder& builder, Location location, Value ordinal, ArrayRef<unsigned> ids)
    {
        Value physical = builder.create<arith::ConstantIndexOp>(location, ids.front());
        for (std::size_t index = 1; index < ids.size(); ++index) {
            Value position = builder.create<arith::ConstantIndexOp>(location, index);
            Value equal = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, ordinal, position);
            Value id = builder.create<arith::ConstantIndexOp>(location, ids[index]);
            physical = builder.create<arith::SelectOp>(location, equal, id, physical);
        }
        return physical;
    }
    Value advance(OpBuilder& builder, Location location, Value ordinal, std::size_t capacity)
    {
        Value one = builder.create<arith::ConstantIndexOp>(location, 1);
        Value next = builder.create<arith::AddIOp>(location, ordinal, one);
        Value bound = builder.create<arith::ConstantIndexOp>(location, capacity);
        Value wrap = builder.create<arith::CmpIOp>(location, arith::CmpIPredicate::eq, next, bound);
        Value zero = builder.create<arith::ConstantIndexOp>(location, 0);
        return builder.create<arith::SelectOp>(location, wrap, zero, next);
    }
    void lowerEndpoint(Operation* operation, std::size_t poolIndex, Counters& counters)
    {
        OpBuilder builder(operation);
        Location location = operation->getLoc();
        const auto& pool = pools[poolIndex];
        auto source = PipeAttr::get(builder.getContext(), pool.source);
        auto target = PipeAttr::get(builder.getContext(), pool.target);
        bool set = isa<LogicalSetOp>(operation);
        if (pool.eligibleIds.size() == 1) {
            auto id = EventAttr::get(builder.getContext(), static_cast<EVENT>(pool.eligibleIds.front()));
            if (set) { builder.create<SetFlagOp>(location, source, target, id); }
            else { builder.create<WaitFlagOp>(location, source, target, id); }
        } else {
            std::size_t slot = offsets[poolIndex] + (set ? 0 : 1);
            Value ordinal = counters[slot];
            Value id = lookup(builder, location, ordinal, pool.eligibleIds);
            if (set) { builder.create<SetFlagDynOp>(location, source, target, id); }
            else { builder.create<WaitFlagDynOp>(location, source, target, id); }
            counters[slot] = advance(builder, location, ordinal, pool.eligibleIds.size());
        }
        operation->erase();
    }
    Operation* expand(Operation* operation, ValueRange extraOperands, const Counters& counters)
    {
        OperationState state(operation->getLoc(), operation->getName());
        state.addOperands(operation->getOperands());
        state.addOperands(extraOperands);
        state.addTypes(operation->getResultTypes());
        for (Value counter : counters) { state.addTypes(counter.getType()); }
        state.addAttributes(operation->getAttrs());
        state.propertiesAttr = operation->getPropertiesAsAttribute();
        for (unsigned index = 0; index < operation->getNumRegions(); ++index) { state.addRegion(); }
        OpBuilder builder(operation);
        Operation* replacement = builder.create(state);
        for (unsigned index = 0; index < operation->getNumRegions(); ++index) {
            replacement->getRegion(index).takeBody(operation->getRegion(index));
        }
        return replacement;
    }
    Counters appendArguments(Block& block, const Counters& counters, Location location)
    {
        Counters arguments;
        for (Value counter : counters) { arguments.push_back(block.addArgument(counter.getType(), location)); }
        return arguments;
    }
    void lowerRegion(Region& region, Counters counters)
    {
        lowerBlock(region.front(), counters);
        region.front().getTerminator()->insertOperands(region.front().getTerminator()->getNumOperands(), counters);
    }
    void finish(Operation* original, Operation* replacement, Counters& counters)
    {
        unsigned originalCount = original->getNumResults();
        for (unsigned index = 0; index < originalCount; ++index) {
            original->getResult(index).replaceAllUsesWith(replacement->getResult(index));
        }
        counters.assign(replacement->getResults().begin() + originalCount, replacement->getResults().end());
        original->erase();
    }
    void lowerFor(scf::ForOp loop, Counters& counters)
    {
        Operation* replacement = expand(loop, counters, counters);
        Region& body = replacement->getRegion(0);
        Counters arguments = appendArguments(body.front(), counters, loop.getLoc());
        lowerRegion(body, arguments);
        finish(loop, replacement, counters);
    }
    void lowerIf(scf::IfOp conditional, Counters& counters)
    {
        Operation* replacement = expand(conditional, {}, counters);
        for (Region& region : replacement->getRegions()) {
            if (region.empty()) {
                region.push_back(new Block());
                OpBuilder builder(&region.front(), region.front().end());
                builder.create<scf::YieldOp>(conditional.getLoc());
            }
            lowerRegion(region, counters);
        }
        finish(conditional, replacement, counters);
    }
    void lowerWhile(scf::WhileOp loop, Counters& counters)
    {
        Operation* replacement = expand(loop, counters, counters);
        for (Region& region : replacement->getRegions()) {
            Counters arguments = appendArguments(region.front(), counters, loop.getLoc());
            lowerRegion(region, arguments);
        }
        finish(loop, replacement, counters);
    }
    void lowerBlock(Block& block, Counters& counters)
    {
        for (Operation& operation : llvm::make_early_inc_range(block)) {
            auto found = endpoints.find(&operation);
            if (found != endpoints.end()) { lowerEndpoint(&operation, found->second, counters); continue; }
            if (counters.empty()) {
                for (Region& region : operation.getRegions()) {
                    if (!region.empty()) { lowerBlock(region.front(), counters); }
                }
            } else if (auto loop = dyn_cast<scf::ForOp>(operation)) { lowerFor(loop, counters); }
            else if (auto conditional = dyn_cast<scf::IfOp>(operation)) { lowerIf(conditional, counters); }
            else if (auto loop = dyn_cast<scf::WhileOp>(operation)) { lowerWhile(loop, counters); }
        }
    }
};
} // namespace
LogicalResult preflightStructuredCounters(func::FuncOp pending, std::string& reason)
{
    if (!pending || !llvm::hasSingleElement(pending.getBody())) {
        return reject(reason, "structured event counters require a single-block function body");
    }
    WalkResult result = pending.walk([&](Operation* operation) {
        if (operation == pending.getOperation()) { return WalkResult::advance(); }
        if (failed(checkControl(operation, reason))) { return WalkResult::interrupt(); }
        return WalkResult::advance();
    });
    return result.wasInterrupted() ? failure() : success();
}
LogicalResult lowerStructuredCounters(func::FuncOp pending, ArrayRef<StructuredEventPool> pools,
    const DenseMap<Operation*, std::size_t>& endpoints, std::string& reason)
{
    if (failed(preflightStructuredCounters(pending, reason)) || failed(checkPools(pools, reason)) ||
        failed(checkEndpoints(pending, pools, endpoints, reason))) { return failure(); }
    CounterLowering lowering(pools, endpoints);
    lowering.run(pending);
    return success();
}
} // namespace mlir::pto::frontiersynch
