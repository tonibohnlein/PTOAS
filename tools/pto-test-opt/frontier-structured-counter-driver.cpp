// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Test-only scalar interpreter: compare actual command executions before and
// after lowering, independently of the counter transformation's SCF rewrite.
#include "../../lib/PTO/Transforms/FrontierSynch/StructuredEventCounters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
namespace {
using namespace mlir;
namespace fs = pto::frontiersynch;
using Numbers = SmallVector<int64_t>;
struct Command {
    bool set;
    int64_t id;
};
class ScalarExecution {
public:
    SmallVector<Command> commands;
    FailureOr<Numbers> run(func::FuncOp function, int64_t trips, bool guard)
    {
        values[function.getArgument(0)] = trips;
        values[function.getArgument(1)] = guard;
        return block(function.getBody().front());
    }
private:
    DenseMap<Value, int64_t> values;
    unsigned budget = 4096;
    FailureOr<Numbers> operands(Operation& operation)
    {
        Numbers result;
        for (Value operand : operation.getOperands()) {
            auto found = values.find(operand);
            if (found == values.end()) { return failure(); }
            result.push_back(found->second);
        }
        return result;
    }
    LogicalResult bind(ValueRange results, ArrayRef<int64_t> numbers)
    {
        if (results.size() != numbers.size()) { return failure(); }
        for (auto [value, number] : llvm::zip(results, numbers)) {
            // The owned fixture uses small scalars; bounding every produced
            // value keeps arithmetic in this test interpreter overflow-free.
            if (number < -1024 || number > 1024) { return failure(); }
            values[value] = number;
        }
        return success();
    }
    FailureOr<Numbers> loop(scf::ForOp operation, ArrayRef<int64_t> inputs)
    {
        if (inputs.size() < 3 || inputs[2] <= 0) { return failure(); }
        Numbers state(inputs.drop_front(3));
        for (int64_t induction = inputs[0]; induction < inputs[1]; induction += inputs[2]) {
            Numbers arguments{induction};
            llvm::append_range(arguments, state);
            if (failed(bind(operation.getBody()->getArguments(), arguments))) { return failure(); }
            auto next = block(*operation.getBody());
            if (failed(next)) { return failure(); }
            state = *next;
        }
        return state;
    }
    FailureOr<Numbers> loop(scf::WhileOp operation, ArrayRef<int64_t> inputs)
    {
        Numbers state(inputs);
        while (true) {
            if (failed(bind(operation.getBefore().front().getArguments(), state))) { return failure(); }
            auto condition = block(operation.getBefore().front());
            if (failed(condition) || condition->empty()) { return failure(); }
            Numbers forwarded(ArrayRef<int64_t>(*condition).drop_front());
            if (!condition->front()) { return forwarded; }
            if (failed(bind(operation.getAfter().front().getArguments(), forwarded))) { return failure(); }
            auto next = block(operation.getAfter().front());
            if (failed(next)) { return failure(); }
            state = *next;
        }
    }
    FailureOr<Numbers> scalar(Operation& operation, ArrayRef<int64_t> inputs)
    {
        if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
            auto integer = dyn_cast<IntegerAttr>(constant.getValue());
            if (!integer || integer.getValue().getBitWidth() > 64) { return failure(); }
            return Numbers{integer.getValue().getSExtValue()};
        }
        if (isa<arith::IndexCastOp>(operation)) {
            if (inputs.size() != 1) { return failure(); }
            return Numbers{inputs[0]};
        }
        if (isa<arith::AddIOp, arith::CmpIOp>(operation) && inputs.size() != 2) { return failure(); }
        if (isa<arith::SelectOp>(operation) && inputs.size() != 3) { return failure(); }
        if (isa<arith::AddIOp>(operation)) { return Numbers{inputs[0] + inputs[1]}; }
        if (isa<arith::SelectOp>(operation)) { return Numbers{inputs[0] ? inputs[1] : inputs[2]}; }
        if (auto comparison = dyn_cast<arith::CmpIOp>(operation)) {
            switch (comparison.getPredicate()) {
            case arith::CmpIPredicate::eq: return Numbers{inputs[0] == inputs[1]};
            case arith::CmpIPredicate::slt: return Numbers{inputs[0] < inputs[1]};
            default: return failure();
            }
        }
        return failure();
    }
    bool command(Operation& operation, ArrayRef<int64_t> inputs)
    {
        bool set = isa<pto::LogicalSetOp, pto::SetFlagOp, pto::SetFlagDynOp>(operation);
        if (isa<pto::LogicalSetOp, pto::LogicalWaitOp>(operation)) {
            commands.push_back({set, -1});
        } else if (isa<pto::SetFlagDynOp, pto::WaitFlagDynOp>(operation)) {
            if (inputs.size() != 1) { return false; }
            commands.push_back({set, inputs.front()});
        } else if (isa<pto::SetFlagOp, pto::WaitFlagOp>(operation)) {
            auto id = operation.getAttrOfType<pto::EventAttr>("event_id");
            if (!id) { return false; }
            commands.push_back({set, static_cast<int64_t>(id.getEvent())});
        } else { return false; }
        return true;
    }
    FailureOr<Numbers> execute(Operation& operation, ArrayRef<int64_t> inputs)
    {
        if (command(operation, inputs)) { return Numbers{}; }
        if (auto counted = dyn_cast<scf::ForOp>(operation)) { return loop(counted, inputs); }
        if (auto repeated = dyn_cast<scf::WhileOp>(operation)) { return loop(repeated, inputs); }
        if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
            if (inputs.size() != 1) { return failure(); }
            Region& region = inputs.front() ? conditional.getThenRegion() : conditional.getElseRegion();
            return region.empty() ? FailureOr<Numbers>(Numbers{}) : block(region.front());
        }
        return scalar(operation, inputs);
    }
    FailureOr<Numbers> block(Block& body)
    {
        for (Operation& operation : body) {
            if (budget == 0) { return failure(); }
            --budget;
            auto inputs = operands(operation);
            if (failed(inputs)) { return failure(); }
            if (isa<scf::YieldOp, scf::ConditionOp, func::ReturnOp>(operation)) { return *inputs; }
            auto outputs = execute(operation, *inputs);
            if (failed(outputs) || failed(bind(operation.getResults(), *outputs))) { return failure(); }
        }
        return failure();
    }
};
constexpr llvm::StringLiteral fixture = R"mlir(
module {
  func.func @counters(%trips: index, %guard: i1) -> index {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    pto.sync.logical_set [<PIPE_MTE2>, <PIPE_V>, 0]()
    %outer = scf.for %i = %zero to %trips step %one iter_args(%sum = %zero) -> (index) {
      %inner = scf.for %j = %zero to %two step %one iter_args(%value = %sum) -> (index) {
        pto.sync.logical_wait [<PIPE_MTE2>, <PIPE_V>, 0]()
        %chosen = scf.if %guard -> (index) {
          pto.sync.logical_set [<PIPE_MTE2>, <PIPE_V>, 0]()
          %incremented = arith.addi %value, %one : index
          scf.yield %incremented : index
        } else {
          pto.sync.logical_wait [<PIPE_MTE2>, <PIPE_V>, 0]()
          %incremented = arith.addi %value, %two : index
          scf.yield %incremented : index
        }
        scf.if %guard {
          pto.sync.logical_wait [<PIPE_MTE2>, <PIPE_V>, 0]()
        }
        scf.yield %chosen : index
      }
      pto.sync.logical_set [<PIPE_MTE2>, <PIPE_V>, 0]()
      scf.yield %inner : index
    }
    %while = scf.while (%position = %zero) : (index) -> i64 {
      pto.sync.logical_set [<PIPE_MTE2>, <PIPE_V>, 0]()
      %continue = arith.cmpi slt, %position, %two : index
      %forward = arith.index_cast %position : index to i64
      scf.condition(%continue) %forward : i64
    } do {
    ^bb0(%position: i64):
      pto.sync.logical_wait [<PIPE_MTE2>, <PIPE_V>, 0]()
      %index = arith.index_cast %position : i64 to index
      %next = arith.addi %index, %one : index
      scf.yield %next : index
    }
    %last = arith.index_cast %while : i64 to index
    %result = arith.addi %outer, %last : index
    pto.sync.logical_wait [<PIPE_MTE2>, <PIPE_V>, 0]()
    return %result : index
  }
})mlir";
struct PayloadAnchor {
    Operation* operation;
    Block* block;
    DictionaryAttr attributes;
};
bool checkExecution(func::FuncOp original, func::FuncOp lowered, ArrayRef<unsigned> ids,
                    int64_t trips, bool guard)
{
    ScalarExecution source;
    ScalarExecution physical;
    auto expected = source.run(original, trips, guard);
    auto actual = physical.run(lowered, trips, guard);
    int64_t expectedPayload = trips * (guard ? 2 : 4) + 2;
    if (failed(expected) || failed(actual) || *expected != *actual ||
        *actual != Numbers{expectedPayload} || source.commands.size() != physical.commands.size()) { return false; }
    std::size_t sets = 0;
    std::size_t waits = 0;
    for (auto [before, after] : llvm::zip(source.commands, physical.commands)) {
        std::size_t& ordinal = before.set ? sets : waits;
        if (before.set != after.set || after.id != ids[ordinal % ids.size()]) { return false; }
        ++ordinal;
    }
    return true;
}
bool checkFixture(MLIRContext& context, ArrayRef<unsigned> ids)
{
    auto module = parseSourceString<ModuleOp>(fixture, &context);
    if (!module || failed(verify(*module))) { return false; }
    auto original = *module->getOps<func::FuncOp>().begin();
    original.walk([&](Operation* operation) {
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(operation)) {
            operation->setAttr("test.preserve", StringAttr::get(&context, "control"));
            operation->setAttr("test.original_results", IntegerAttr::get(IntegerType::get(&context, 32),
                operation->getNumResults()));
        }
    });
    OwningOpRef<func::FuncOp> pending(cast<func::FuncOp>(original->clone()));
    SmallVector<PayloadAnchor> anchors;
    DenseMap<Operation*, std::size_t> endpoints;
    pending->walk([&](Operation* operation) {
        if (isa<pto::LogicalSetOp, pto::LogicalWaitOp>(operation)) { endpoints[operation] = 0; }
        if (operation->getDialect() && operation->getDialect()->getNamespace() == "arith") {
            anchors.push_back({operation, operation->getBlock(), operation->getAttrDictionary()});
        }
    });
    fs::StructuredEventPool pool{pto::PIPE::PIPE_MTE2, pto::PIPE::PIPE_V, SmallVector<unsigned>(ids)};
    std::string reason;
    if (failed(fs::lowerStructuredCounters(*pending, {pool}, endpoints, reason)) || failed(verify(*pending))) {
        llvm::errs() << "structured counter lowering failed: " << reason << "\n";
        return false;
    }
    for (const auto& anchor : anchors) {
        if (anchor.operation->getBlock() != anchor.block ||
            anchor.operation->getAttrDictionary() != anchor.attributes) { return false; }
    }
    bool logicalRemains = false;
    bool invalidControl = false;
    unsigned extra = ids.size() == 1 ? 0 : 2;
    pending->walk([&](Operation* operation) {
        if (isa<pto::LogicalSetOp, pto::LogicalWaitOp>(operation)) { logicalRemains = true; }
        if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(operation)) {
            auto attribute = operation->getAttrOfType<StringAttr>("test.preserve");
            auto originalResults = operation->getAttrOfType<IntegerAttr>("test.original_results");
            if (!attribute || attribute.getValue() != "control" || !originalResults ||
                operation->getNumResults() != static_cast<unsigned>(originalResults.getInt()) + extra) {
                invalidControl = true;
            }
        }
    });
    if (logicalRemains || invalidControl) { return false; }
    for (int64_t trips : {0, 1, 4}) {
        for (bool guard : {false, true}) {
            if (!checkExecution(original, *pending, ids, trips, guard)) { return false; }
        }
    }
    return true;
}
} // namespace
int runStructuredCounterChecks(mlir::MLIRContext& context)
{
    if (!checkFixture(context, {1, 3, 5}) || !checkFixture(context, {3})) {
        llvm::errs() << "structured counter scalar execution oracle failed\n";
        return 1;
    }
    llvm::outs() << "structured counter scalar execution and payload preservation oracles passed\n";
    return 0;
}
