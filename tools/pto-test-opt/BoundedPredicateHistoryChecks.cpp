// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Execute the threaded scalar IR against independently unfolded visit histories.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/BoundedPredicateHistory.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
struct Observation {
    uint64_t outer, iv, lag, value, valid, stage;
};
struct Fixture {
    Fixture(MLIRContext& context, unsigned trips, unsigned invocations)
        : module(ModuleOp::create(UnknownLoc::get(&context)))
    {
        OpBuilder b(&context); auto loc = b.getUnknownLoc();
        b.setInsertionPointToEnd(module->getBody());
        auto inputType = b.getFunctionType({b.getIndexType(), b.getIndexType(), b.getI64Type()}, {b.getI1Type()});
        b.create<func::FuncOp>(loc, "history_defined", inputType).setPrivate();
        b.create<func::FuncOp>(loc, "history_value", inputType).setPrivate();
        auto observerType = b.getFunctionType(
            {b.getIndexType(), b.getIndexType(), b.getIndexType(), b.getI1Type(), b.getI1Type(), b.getIndexType()}, {});
        b.create<func::FuncOp>(loc, "history_observe", observerType).setPrivate();
        function = b.create<func::FuncOp>(loc, "history",
            b.getFunctionType({b.getI64Type(), b.getI64Type()}, {}));
        b.setInsertionPointToStart(function.addEntryBlock());
        auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
        auto one = b.create<arith::ConstantIndexOp>(loc, 1);
        auto count = b.create<arith::ConstantIndexOp>(loc, invocations);
        auto lower = b.create<arith::ConstantIndexOp>(loc, -5);
        auto upper = b.create<arith::ConstantIndexOp>(loc, -5 + 3 * static_cast<int64_t>(trips));
        auto step = b.create<arith::ConstantIndexOp>(loc, 3);
        outer = b.create<scf::ForOp>(loc, zero, count, one);
        b.setInsertionPointToStart(outer.getBody());
        loop = b.create<scf::ForOp>(loc, lower, upper, step);
        b.setInsertionPointToStart(loop.getBody());
        auto defined = b.create<func::CallOp>(loc, "history_defined", TypeRange{b.getI1Type()},
            ValueRange{outer.getInductionVar(), loop.getInductionVar(), function.getArgument(1)});
        branch = b.create<scf::IfOp>(loc, defined.getResult(0), true);
        b.setInsertionPointToStart(branch.thenBlock());
        predicate = b.create<func::CallOp>(loc, "history_value", TypeRange{b.getI1Type()},
            ValueRange{outer.getInductionVar(), loop.getInductionVar(), function.getArgument(0)}).getResult(0);
        b.setInsertionPointAfter(outer); b.create<func::ReturnOp>(loc);
    }
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    scf::ForOp outer, loop;
    scf::IfOp branch;
    Value predicate;
};
class Interpreter {
public:
    explicit Interpreter(unsigned trips) : trips(trips) {}
    bool run(func::FuncOp function, uint64_t valuesMask, uint64_t definitionsMask)
    {
        values.clear(); observations.clear(); valueCalls = definitionCalls = 0;
        values[function.getArgument(0)] = valuesMask;
        values[function.getArgument(1)] = definitionsMask;
        SmallVector<uint64_t> ignored;
        return block(function.getBody().front(), ignored);
    }
    std::vector<Observation> observations;
    unsigned valueCalls = 0, definitionCalls = 0;
private:
    bool call(func::CallOp op, llvm::ArrayRef<uint64_t> args)
    {
        const bool invalidCall = op.getCallee() == "history_observe" && args.size() == 6;
        if (invalidCall) {
            observations.push_back({args[0], args[1], args[2], args[3], args[4], args[5]});
            return true;
        }
        const bool wideVisit = args.size() != 3 || op.getNumResults() != 1;
        if (wideVisit) { return false; }
        const auto iv = APInt(64, args[1]).getSExtValue();
        const auto ordinal = static_cast<uint64_t>((iv + 5) / 3);
        const auto visit = args[0] * trips + ordinal;
        if (visit >= 64) { return false; }
        const bool valueProducer = op.getCallee() == "history_value";
        if (valueProducer) { ++valueCalls; }
        else if (op.getCallee() == "history_defined") { ++definitionCalls; }
        else { return false; }
        values[op.getResult(0)] = (args[2] >> visit) & 1;
        return true;
    }
    bool loop(scf::ForOp op, llvm::ArrayRef<uint64_t> args)
    {
        const bool invalidArguments = args.size() < 3;
        if (invalidArguments) { return false; }
        const auto lower = APInt(64, args[0]).getSExtValue();
        const auto upper = APInt(64, args[1]).getSExtValue();
        const auto step = APInt(64, args[2]).getSExtValue();
        if (step <= 0) { return false; }
        auto initial = args.drop_front(3);
        SmallVector<uint64_t> carried(initial.begin(), initial.end());
        for (int64_t iv = lower; iv < upper; iv += step) {
            values[op.getInductionVar()] = static_cast<uint64_t>(iv);
            const bool invalidCarried = carried.size() != op.getRegionIterArgs().size();
            if (invalidCarried) { return false; }
            for (auto pair : llvm::zip(op.getRegionIterArgs(), carried)) {
                values[std::get<0>(pair)] = std::get<1>(pair);
            }
            SmallVector<uint64_t> next;
            const bool bodyValid = block(*op.getBody(), next);
            if (!bodyValid) { return false; }
            carried = std::move(next);
        }
        const bool resultsMatch = carried.size() == op.getNumResults();
        if (!resultsMatch) { return false; }
        for (auto pair : llvm::zip(op.getResults(), carried)) { values[std::get<0>(pair)] = std::get<1>(pair); }
        return true;
    }
    bool branch(scf::IfOp op, llvm::ArrayRef<uint64_t> args)
    {
        const bool invalidCondition = args.size() != 1;
        if (invalidCondition) { return false; }
        auto& region = args[0] ? op.getThenRegion() : op.getElseRegion();
        SmallVector<uint64_t> results;
        const bool regionValid = region.empty() || block(region.front(), results);
        if (!regionValid) { return false; }
        const bool invalidResults = results.size() != op.getNumResults();
        if (invalidResults) { return false; }
        for (auto pair : llvm::zip(op.getResults(), results)) { values[std::get<0>(pair)] = std::get<1>(pair); }
        return true;
    }
    bool block(Block& body, SmallVector<uint64_t>& result)
    {
        for (auto& operation : body) {
            SmallVector<uint64_t> args;
            for (auto operand : operation.getOperands()) {
                auto found = values.find(operand);
                if (found == values.end()) { return false; }
                args.push_back(found->second);
            }
            if (isa<scf::YieldOp, func::ReturnOp>(operation)) { result = args; return true; }
            if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
                auto number = dyn_cast<IntegerAttr>(constant.getValue());
                const bool invalidConstant = !number || number.getValue().getBitWidth() > 64;
                if (invalidConstant) { return false; }
                values[constant.getResult()] = number.getValue().getZExtValue();
            } else if (auto invocation = dyn_cast<func::CallOp>(operation)) {
                if (!call(invocation, args)) { return false; }
            } else if (auto repeat = dyn_cast<scf::ForOp>(operation)) {
                if (!loop(repeat, args)) { return false; }
            } else if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
                if (!branch(conditional, args)) { return false; }
            } else { return false; }
        }
        return false;
    }
    unsigned trips;
    llvm::DenseMap<Value, uint64_t> values;
};
bool rejectsForeignReads(const fs::BoundedPredicateHistory& history, Operation* anchor,
                         const IRMapping& mapping, llvm::ArrayRef<Value> state, unsigned bound)
{
    std::string error;
    IRMapping foreign;
    if (succeeded(history.read(0, 0, anchor, foreign, state, error))) { return false; }
    if (succeeded(history.read(1, 0, anchor, mapping, state, error))) { return false; }
    if (succeeded(history.read(0, bound + 1, anchor, mapping, state, error))) { return false; }
    if (succeeded(history.read(0, 0, nullptr, mapping, state, error))) { return false; }
    if (succeeded(history.read(0, 0, anchor, mapping, {}, error))) { return false; }
    return succeeded(history.read(0, 0, anchor, mapping, state, error));
}
std::string printed(Operation* operation)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
}
fs::PrivateStateAction observe(const fs::BoundedPredicateHistory& history, const Fixture& fixture,
                               Operation* anchor, fs::PrivateStateCut cut, unsigned stage, unsigned bound)
{
    auto outerLoop = fixture.outer, innerLoop = fixture.loop;
    auto outer = outerLoop.getInductionVar(), iv = innerLoop.getInductionVar();
    return {anchor, cut, [history, anchor, outer, iv, stage, bound](OpBuilder& b, const IRMapping& mapping,
        llvm::MutableArrayRef<Value> state, std::string& error) {
        const bool contextValid = rejectsForeignReads(history, anchor, mapping, state, bound);
        if (!contextValid) { error = "history read isolation failed"; return failure(); }
        for (unsigned lag = 0; lag <= bound; ++lag) {
            auto read = history.read(0, lag, anchor, mapping, state, error);
            if (failed(read)) { return failure(); }
            auto at = b.create<arith::ConstantIndexOp>(anchor->getLoc(), lag);
            auto kind = b.create<arith::ConstantIndexOp>(anchor->getLoc(), stage);
            b.create<func::CallOp>(anchor->getLoc(), "history_observe", TypeRange{},
                ValueRange{mapping.lookup(outer), mapping.lookup(iv), at, read->value, read->valid, kind});
        }
        return success();
    }};
}
bool expected(const Interpreter& interpreter, unsigned trips, unsigned invocations, unsigned bound,
              uint64_t valueMask, uint64_t definitionMask)
{
    unsigned active = 0;
    for (unsigned visit = 0; visit < trips * invocations; ++visit) {
        active += static_cast<unsigned>((definitionMask >> visit) & 1);
    }
    if (interpreter.valueCalls != active || interpreter.definitionCalls != trips * invocations ||
        interpreter.observations.size() != (2 * trips * invocations + active) * (bound + 1)) { return false; }
    for (const auto& read : interpreter.observations) {
        const auto ordinal = static_cast<unsigned>((APInt(64, read.iv).getSExtValue() + 5) / 3);
        bool valid = read.lag <= ordinal && !(read.stage == 0 && read.lag == 0);
        const auto visit = valid ? read.outer * trips + ordinal - read.lag : 0;
        valid &= ((definitionMask >> visit) & 1) != 0;
        const bool value = valid && ((valueMask >> visit) & 1);
        const bool invalidRead = read.valid != static_cast<uint64_t>(valid) ||
            read.value != static_cast<uint64_t>(value);
        if (invalidRead) { return false; }
    }
    return true;
}
bool unfolding(MLIRContext& context, unsigned trips, unsigned invocations, unsigned bound, uint64_t& cases)
{
    Fixture fixture(context, trips, invocations);
    if (failed(verify(*fixture.module))) { return false; }
    std::vector<fs::PrivateStateSlot> slots;
    std::vector<fs::PrivateStateAction> actions;
    std::string error;
    auto history = fs::BoundedPredicateHistory::append(fixture.function,
        {{fixture.predicate, fixture.loop, bound}}, slots, actions, error);
    const bool historyValid = succeeded(history) && history->slotCount() == 2 * (bound + 1);
    if (!historyValid) { return false; }
    actions.push_back(observe(*history, fixture, &fixture.loop.getBody()->front(),
        fs::PrivateStateCut::Before, 0, bound));
    actions.push_back(observe(*history, fixture, fixture.branch, fs::PrivateStateCut::After, 1, bound));
    actions.push_back(observe(*history, fixture, fixture.predicate.getDefiningOp(),
        fs::PrivateStateCut::After, 2, bound));
    const auto originalText = printed(fixture.module->getOperation());
    fs::PrivateStateError diagnostic;
    auto threaded = fs::materializeStructuredPrivateState(fixture.function, slots, actions, diagnostic);
    const bool originalPreserved = originalText == printed(fixture.module->getOperation());
    const bool invalidClone = failed(threaded) || failed(verify(*fixture.module)) || !originalPreserved;
    if (invalidClone) { return false; }
    Interpreter interpreter(trips);
    const auto patterns = uint64_t(1) << (trips * invocations);
    for (uint64_t defined = 0; defined < patterns; ++defined) {
        for (uint64_t values = 0; values < patterns; ++values) {
            const bool invalidExecution = !interpreter.run((*threaded)->function, values, defined) ||
                !expected(interpreter, trips, invocations, bound, values, defined);
            if (invalidExecution) { return false; }
            ++cases;
        }
    }
    return true;
}
bool invalidRequirements(MLIRContext& context)
{
    Fixture fixture(context, 2, 2);
    std::vector<fs::PrivateStateSlot> slots;
    std::vector<fs::PrivateStateAction> actions;
    std::string error;
    for (const auto& requirement : std::vector<fs::BoundedPredicateHistoryRequirement>{
        {fixture.predicate, fixture.loop, UINT64_MAX}, {fixture.predicate, fixture.outer, 1},
        {fixture.function.getArgument(0), fixture.loop, 1}, {fixture.predicate, {}, 1}}) {
        auto history = fs::BoundedPredicateHistory::append(fixture.function, {requirement}, slots, actions, error);
        const bool nonTransactional = succeeded(history) || !slots.empty() || !actions.empty() || error.empty();
        if (nonTransactional) { return false; }
    }
    fs::BoundedPredicateHistoryRequirement requirement{fixture.predicate, fixture.loop, 1};
    auto duplicate = fs::BoundedPredicateHistory::append(fixture.function,
        {requirement, requirement}, slots, actions, error);
    return failed(duplicate) && slots.empty() && actions.empty();
}
} // namespace
bool runBoundedPredicateHistoryChecks()
{
    MLIRContext context; context.disableMultithreading();
    context.loadDialect<func::FuncDialect, scf::SCFDialect, arith::ArithDialect>();
    uint64_t cases = 0;
    for (unsigned invocations : {1U, 2U}) {
        for (unsigned trips = 0; trips <= 3; ++trips) {
            for (unsigned bound : {0U, 2U}) {
                if (!unfolding(context, trips, invocations, bound, cases)) {
                    llvm::errs() << "bounded predicate history unfolding failed\n";
                    return false;
                }
            }
        }
    }
    if (!invalidRequirements(context)) { return false; }
    llvm::outs() << "bounded predicate history checks passed: " << cases << " unfolded patterns\n";
    return true;
}
