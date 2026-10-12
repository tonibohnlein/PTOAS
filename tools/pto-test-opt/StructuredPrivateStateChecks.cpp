// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent small-integer execution checks for transactional private state.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/StructuredPrivateState.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
struct Fixture {
    explicit Fixture(MLIRContext& context) : module(ModuleOp::create(UnknownLoc::get(&context)))
    {
        OpBuilder b(&context); auto loc = b.getUnknownLoc();
        b.setInsertionPointToEnd(module->getBody());
        b.create<func::FuncOp>(loc, "payload", b.getFunctionType({b.getIndexType()}, {})).setPrivate();
        function = b.create<func::FuncOp>(loc, "state", b.getFunctionType(
            {b.getIndexType(), b.getI1Type()}, {b.getIndexType()}));
        b.setInsertionPointToStart(function.addEntryBlock());
        auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
        auto one = b.create<arith::ConstantIndexOp>(loc, 1);
        auto two = b.create<arith::ConstantIndexOp>(loc, 2);
        outer = b.create<scf::ForOp>(loc, zero, function.getArgument(0), one, ValueRange{zero});
        b.setInsertionPointToStart(outer.getBody());
        auto choice = b.create<scf::IfOp>(loc, TypeRange{b.getIndexType()}, function.getArgument(1), true, true);
        b.setInsertionPointToStart(choice.thenBlock());
        first = b.create<func::CallOp>(loc, "payload", TypeRange{}, ValueRange{outer.getInductionVar()});
        auto sum = b.create<arith::AddIOp>(loc, outer.getRegionIterArgs()[0], one);
        b.create<scf::YieldOp>(loc, ValueRange{sum});
        b.setInsertionPointToStart(choice.elseBlock());
        b.create<scf::YieldOp>(loc, outer.getRegionIterArgs());
        b.setInsertionPointAfter(choice);
        inner = b.create<scf::ForOp>(loc, zero, two, one);
        b.setInsertionPointToStart(inner.getBody());
        auto optional = b.create<scf::IfOp>(loc, function.getArgument(1), false);
        b.setInsertionPointToStart(optional.thenBlock());
        second = b.create<func::CallOp>(loc, "payload", TypeRange{}, ValueRange{inner.getInductionVar()});
        b.setInsertionPointAfter(inner);
        b.create<scf::YieldOp>(loc, choice.getResults());
        b.setInsertionPointAfter(outer);
        b.create<func::ReturnOp>(loc, outer.getResults());
    }
    OwningOpRef<ModuleOp> module;
    func::FuncOp function;
    scf::ForOp outer, inner;
    Operation *first = nullptr, *second = nullptr;
};
struct Execution {
    llvm::DenseMap<Value, int64_t> values;
    SmallVector<int64_t> payloads;
    bool valid = true;
    int64_t get(Value value)
    {
        auto found = values.find(value);
        if (found == values.end()) { valid = false; return 0; }
        return found->second;
    }
    SmallVector<int64_t> read(ValueRange operands)
    {
        SmallVector<int64_t> output;
        for (auto value : operands) { output.push_back(get(value)); }
        return output;
    }
    void assign(ValueRange results, ArrayRef<int64_t> data)
    {
        const bool sameSize = results.size() == data.size();
        if (!sameSize) { valid = false; return; }
        auto pairs = llvm::zip(results, data);
        for (auto [value, number] : pairs) { values[value] = number; }
    }
    SmallVector<int64_t> block(Block& body)
    {
        for (auto& operation : body) {
            if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
                auto attribute = dyn_cast<IntegerAttr>(constant.getValue());
                if (!attribute) { valid = false; return {}; }
                values[constant] = attribute.getValue().getSExtValue();
            } else if (auto add = dyn_cast<arith::AddIOp>(operation)) {
                values[add] = get(add.getLhs()) + get(add.getRhs());
            } else if (auto select = dyn_cast<arith::SelectOp>(operation)) {
                values[select] = get(get(select.getCondition()) ? select.getTrueValue() : select.getFalseValue());
            } else if (auto call = dyn_cast<func::CallOp>(operation)) {
                payloads.push_back(get(call.getOperand(0)));
            } else if (auto loop = dyn_cast<scf::ForOp>(operation)) {
                auto carried = read(loop.getInitArgs());
                const auto upper = get(loop.getUpperBound()), step = get(loop.getStep());
                if (step <= 0) { valid = false; return {}; }
                for (auto visit = get(loop.getLowerBound()); visit < upper; visit += step) {
                    values[loop.getInductionVar()] = visit;
                    assign(loop.getRegionIterArgs(), carried);
                    carried = block(*loop.getBody());
                }
                assign(loop.getResults(), carried);
            } else if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
                auto& arm = get(conditional.getCondition()) ? conditional.getThenRegion() : conditional.getElseRegion();
                auto returned = arm.empty() ? SmallVector<int64_t>{} : block(arm.front());
                assign(conditional.getResults(), returned);
            } else if (isa<func::ReturnOp, scf::YieldOp>(operation)) {
                return read(operation.getOperands());
            } else { valid = false; return {}; }
        }
        valid = false;
        return {};
    }
    SmallVector<int64_t> run(func::FuncOp function, int64_t trips, bool take)
    {
        values[function.getArgument(0)] = trips;
        values[function.getArgument(1)] = take;
        return block(function.front());
    }
};
std::string render(Operation* operation)
{
    std::string text; llvm::raw_string_ostream out(text); operation->print(out); return text;
}
std::vector<fs::PrivateStateAction> actions(Fixture& fixture)
{
    using Cut = fs::PrivateStateCut;
    auto resetCount = [](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
        state[0] = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 7); return success();
    };
    auto increment = [](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
        auto one = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 1);
        state[0] = b.create<arith::AddIOp>(b.getUnknownLoc(), state[0], one); return success();
    };
    auto resetValid = [](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
        state[1] = b.create<arith::ConstantIntOp>(b.getUnknownLoc(), 0, 1); return success();
    };
    auto nestedCount = [](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
        auto two = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 2);
        auto three = b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 3);
        auto amount = b.create<arith::SelectOp>(b.getUnknownLoc(), state[1], three, two);
        state[0] = b.create<arith::AddIOp>(b.getUnknownLoc(), state[0], amount); return success();
    };
    auto setValid = [](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
        state[1] = b.create<arith::ConstantIntOp>(b.getUnknownLoc(), 1, 1); return success();
    };
    return {{fixture.outer, Cut::Before, resetCount}, {fixture.first, Cut::After, increment},
        {fixture.inner, Cut::Before, resetValid}, {fixture.second, Cut::After, nestedCount},
        {fixture.second, Cut::After, setValid}};
}
bool successfulCases(MLIRContext& context)
{
    Fixture fixture(context);
    if (failed(verify(*fixture.module))) { return false; }
    auto before = render(fixture.module->getOperation());
    Builder b(&context);
    std::vector<fs::PrivateStateSlot> slots{
        {b.getIndexType(), b.getIndexAttr(0)}, {b.getI1Type(), b.getBoolAttr(false)}};
    fs::PrivateStateError error;
    auto recipe = actions(fixture);
    auto staged = fs::materializeStructuredPrivateState(fixture.function, slots, recipe, error);
    const bool ready = succeeded(staged);
    if (!ready) {
        llvm::errs() << error.message << "\n";
        return false;
    }
    auto& mapping = (*staged)->mapping;
    auto mappedLoop = dyn_cast_or_null<scf::ForOp>(mapping.lookupOrNull(fixture.outer.getOperation()));
    const bool maps = mappedLoop && mappedLoop.getNumResults() == 3 &&
        mapping.lookupOrNull(fixture.outer.getInductionVar()) == mappedLoop.getInductionVar() &&
        mapping.lookupOrNull(fixture.outer.getResult(0)) == mappedLoop.getResult(0) &&
        mapping.lookupOrNull(fixture.first) != fixture.first;
    if (!maps) { return false; }
    for (int64_t trips : {0, 1, 3}) {
        for (bool take : {false, true}) {
            Execution old, current;
            auto expected = old.run(fixture.function, trips, take);
            auto actual = current.run((*staged)->function, trips, take);
            const auto count = current.get((*staged)->finalState[0]);
            const bool valid = current.get((*staged)->finalState[1]) != 0;
            const bool exact = old.valid && current.valid && expected == actual && old.payloads == current.payloads &&
                count == 7 + (take ? 6 * trips : 0) && valid == (take && trips > 0);
            if (!exact) { return false; }
        }
    }
    return before == render(fixture.module->getOperation());
}
bool siblingStateFailure(MLIRContext& context)
{
    Fixture fixture(context);
    Builder b(&context);
    fs::PrivateStateError error;
    std::vector<fs::PrivateStateSlot> slots{{b.getIndexType(), b.getIndexAttr(0)}};
    const auto before = render(fixture.module->getOperation());
    auto choice = cast<scf::IfOp>(fixture.first->getParentOp());
    auto thenValue = choice.thenBlock()->getTerminator()->getOperand(0);
    auto sibling = [thenValue](OpBuilder&, const IRMapping& mapping,
                              MutableArrayRef<Value> state, std::string&) {
        state[0] = mapping.lookup(thenValue);
        return success();
    };
    std::vector<fs::PrivateStateAction> recipe{
        {choice.elseBlock()->getTerminator(), fs::PrivateStateCut::Before, sibling}};
    auto bad = fs::materializeStructuredPrivateState(fixture.function, slots, recipe, error);
    const bool rejected = failed(bad) && error.kind == fs::PrivateStateFailure::Verification;
    if (!rejected || before != render(fixture.module->getOperation())) { return false; }
    auto retry = fs::materializeStructuredPrivateState(fixture.function, slots, {}, error);
    return succeeded(retry) && error.kind == fs::PrivateStateFailure::None &&
        before == render(fixture.module->getOperation());
}
bool failures(MLIRContext& context)
{
    Fixture fixture(context), foreign(context);
    Builder b(&context); fs::PrivateStateError error;
    std::vector<fs::PrivateStateSlot> slots{{b.getIndexType(), b.getIndexAttr(0)}};
    const auto before = render(fixture.module->getOperation());
    auto wrongType = [](OpBuilder& b, const IRMapping&, MutableArrayRef<Value> state, std::string&) {
        state[0] = b.create<arith::ConstantIntOp>(b.getUnknownLoc(), 1, 1); return success();
    };
    std::vector<fs::PrivateStateAction> recipe{{fixture.first, fs::PrivateStateCut::After, wrongType}};
    auto bad = fs::materializeStructuredPrivateState(fixture.function, slots, recipe, error);
    const bool wrongTypeRejected = failed(bad) && error.kind == fs::PrivateStateFailure::Action;
    if (!wrongTypeRejected) { return false; }
    recipe[0].anchor = foreign.first;
    bad = fs::materializeStructuredPrivateState(fixture.function, slots, recipe, error);
    const bool foreignRejected = failed(bad) && error.kind == fs::PrivateStateFailure::InvalidCut;
    if (!foreignRejected) { return false; }
    recipe[0].anchor = fixture.first;
    recipe[0].apply = [](OpBuilder& b, const IRMapping&, MutableArrayRef<Value>, std::string& reason) {
        b.create<arith::ConstantIndexOp>(b.getUnknownLoc(), 99);
        reason = "deliberate failure after staged work"; return failure();
    };
    bad = fs::materializeStructuredPrivateState(fixture.function, slots, recipe, error);
    const bool rolledBack = failed(bad) && before == render(fixture.module->getOperation());
    if (!rolledBack) { return false; }
    auto good = fs::materializeStructuredPrivateState(fixture.function, slots, {}, error);
    if (failed(good)) { return false; }
    auto noState = fs::materializeStructuredPrivateState(fixture.function, {}, {}, error);
    if (failed(noState)) { return false; }
    OpBuilder build(fixture.function.front().getTerminator());
    auto unsupported = build.create<scf::ExecuteRegionOp>(build.getUnknownLoc(), TypeRange{});
    unsupported.getRegion().push_back(new Block);
    build.setInsertionPointToEnd(&unsupported.getRegion().front());
    build.create<scf::YieldOp>(build.getUnknownLoc());
    const auto unsupportedBefore = render(fixture.module->getOperation());
    bad = fs::materializeStructuredPrivateState(fixture.function, slots, {}, error);
    return failed(bad) && error.kind == fs::PrivateStateFailure::UnsupportedControl &&
        unsupportedBefore == render(fixture.module->getOperation());
}
} // namespace
bool runStructuredPrivateStateChecks()
{
    MLIRContext context; context.disableMultithreading();
    context.loadDialect<func::FuncDialect, scf::SCFDialect, arith::ArithDialect>();
    const bool passed = successfulCases(context) && siblingStateFailure(context) && failures(context);
    if (!passed) { return false; }
    llvm::outs() << "structured private state checks passed\n";
    return true;
}
