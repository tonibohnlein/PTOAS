// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Execute inserted arith/scf control, never regenerate commands from recipes.
#include "SyncLogicalInsertionChecks.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateInsertion.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
constexpr uint64_t MaxVisits = 2000000;
constexpr uint64_t MaxEvents = 100000;
constexpr unsigned MaxDepth = 64;
class Interpreter {
public:
    explicit Interpreter(const fs::NumericTemplate& input) : input(input) {
        for (uint32_t i = 0; i < input.payloads.size(); ++i) {
            types[input.payloads[i].phase->elementOp].push_back(i);
        }
    }
    llvm::json::Object run(func::FuncOp function);
private:
    bool fail(StringRef message) { error = message.str(); return false; }
    Attribute lookup(Value value) const { return values.lookup(value); }
    std::optional<int64_t> integer(Value value) const;
    bool block(Block& body, SmallVectorImpl<Attribute>& yielded, unsigned depth);
    bool operation(Operation& op, unsigned depth);
    bool loop(scf::ForOp op, unsigned depth);
    bool branch(scf::IfOp op, unsigned depth);
    bool payload(Operation& op, ArrayRef<uint32_t> candidates);
    bool command(Operation& op);
    void folded(Operation& op);
    void bind(ValueRange targets, ArrayRef<Attribute> sources);
    const fs::NumericTemplate& input;
    DenseMap<Value, Attribute> values;
    DenseMap<Operation*, SmallVector<uint32_t>> types;
    llvm::json::Array events;
    std::string error;
    uint64_t visits = 0;
    uint64_t payloads = 0;
    uint64_t outerTrips = 0;
};
std::optional<int64_t> Interpreter::integer(Value value) const
{
    const auto attr = dyn_cast_or_null<IntegerAttr>(lookup(value));
    if (!attr || !attr.getValue().isSignedIntN(64)) {
        return std::nullopt;
    }
    return attr.getValue().getSExtValue();
}
void Interpreter::bind(ValueRange targets, ArrayRef<Attribute> sources)
{
    for (auto [target, source] : llvm::zip(targets, sources)) {
        if (source) {
            values[target] = source;
        } else {
            values.erase(target);
        }
    }
}
bool Interpreter::block(Block& body, SmallVectorImpl<Attribute>& yielded, unsigned depth)
{
    if (depth > MaxDepth) {
        return fail("trace control depth exceeded");
    }
    for (auto& op : body) {
        if (++visits > MaxVisits || events.size() >= MaxEvents) {
            return fail("trace visit or event limit exceeded");
        }
        if (isa<scf::YieldOp, func::ReturnOp>(op)) {
            for (auto operand : op.getOperands()) {
                yielded.push_back(lookup(operand));
            }
            return true;
        }
        if (!operation(op, depth)) {
            return false;
        }
    }
    return true;
}
bool Interpreter::loop(scf::ForOp op, unsigned depth)
{
    const auto lower = integer(op.getLowerBound()), upper = integer(op.getUpperBound());
    const auto step = integer(op.getStep());
    if (!lower || !upper || !step || *step <= 0) {
        return fail("trace requires known signed counted-loop bounds");
    }
    SmallVector<Attribute> state;
    for (auto initial : op.getInitArgs()) {
        state.push_back(lookup(initial));
    }
    int64_t induction = *lower;
    while (induction < *upper) {
        if (op == input.outer) {
            ++outerTrips;
        }
        values[op.getInductionVar()] = IntegerAttr::get(op.getInductionVar().getType(), induction);
        bind(op.getRegionIterArgs(), state);
        SmallVector<Attribute> next;
        if (!block(*op.getBody(), next, depth + 1) || next.size() != state.size()) {
            return fail(error.empty() ? "trace loop yield arity mismatch" : StringRef(error));
        }
        state = std::move(next);
        int64_t following = 0;
        if (llvm::AddOverflow(induction, *step, following)) {
            return fail("trace loop induction overflow");
        }
        induction = following;
    }
    bind(op.getResults(), state);
    return true;
}
bool Interpreter::branch(scf::IfOp op, unsigned depth)
{
    const auto condition = dyn_cast_or_null<IntegerAttr>(lookup(op.getCondition()));
    if (!condition) {
        return fail("trace branch condition is unknown");
    }
    auto& region = condition.getValue().isZero() ? op.getElseRegion() : op.getThenRegion();
    SmallVector<Attribute> yielded;
    if (!region.empty() && !block(region.front(), yielded, depth + 1)) {
        return false;
    }
    if (yielded.size() != op.getNumResults()) {
        return fail("trace branch yield arity mismatch");
    }
    bind(op.getResults(), yielded);
    return true;
}
bool Interpreter::payload(Operation& op, ArrayRef<uint32_t> candidates)
{
    std::optional<uint32_t> selected;
    for (auto type : candidates) {
        bool matches = true;
        for (const auto& coordinate : input.payloads[type].coordinates) {
            auto loopOp = coordinate.loop;
            const auto actual = integer(loopOp.getInductionVar());
            matches &= actual && *actual == coordinate.induction;
        }
        if (matches) {
            if (selected) {
                return fail("trace payload has ambiguous template coordinates");
            }
            selected = type;
        }
    }
    auto outer = input.outer;
    const auto induction = integer(outer.getInductionVar());
    if (!selected || !induction || *induction < input.lower || input.step <= 0) {
        return fail("trace payload is outside its template domain");
    }
    const uint64_t difference = static_cast<uint64_t>(*induction) - static_cast<uint64_t>(input.lower);
    if (difference % static_cast<uint64_t>(input.step)) {
        return fail("trace payload outer coordinate is off grid");
    }
    const auto& item = input.payloads[*selected];
    events.push_back(llvm::json::Object{{"kind", "payload"}, {"type", *selected},
        {"ordinal", difference / static_cast<uint64_t>(input.step)},
        {"pipe", static_cast<unsigned>(item.phase->kPipeValue)}});
    ++payloads;
    for (auto result : op.getResults()) {
        values.erase(result); // Payload data are deliberately not simulated.
    }
    return true;
}
bool Interpreter::command(Operation& op)
{
    const auto name = op.getName().getStringRef();
    if (name == "pto.barrier") {
        const auto pipe = op.getAttrOfType<pto::PipeAttr>("pipe");
        if (!pipe) {
            return fail("trace barrier has no pipe");
        }
        events.push_back(llvm::json::Object{{"kind", "barrier"}, {"gap", payloads},
                                          {"pipe", static_cast<unsigned>(pipe.getPipe())}});
        return true;
    }
    const auto source = op.getAttrOfType<pto::PipeAttr>("src_pipe");
    const auto target = op.getAttrOfType<pto::PipeAttr>("dst_pipe");
    if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp, pto::WaitFlagDynOp>(op)) {
        auto fixed = op.getAttrOfType<pto::EventAttr>("event_id");
        auto id = fixed ? std::optional<int64_t>(static_cast<int64_t>(fixed.getEvent())) :
            (op.getNumOperands() == 1 ? integer(op.getOperand(0)) : std::nullopt);
        if (!source || !target || !id) {
            return fail("trace physical command has an unknown ID");
        }
        const bool publish = isa<pto::SetFlagOp, pto::SetFlagDynOp>(op);
        events.push_back(llvm::json::Object{{"kind", publish ? "set" : "wait"}, {"gap", payloads},
            {"pipe", static_cast<unsigned>((publish ? source : target).getPipe())},
            {"source_pipe", static_cast<unsigned>(source.getPipe())},
            {"target_pipe", static_cast<unsigned>(target.getPipe())}, {"physical_id", *id}});
        return true;
    }
    const auto plan = op.getAttrOfType<IntegerAttr>("plan_id"), record = op.getAttrOfType<IntegerAttr>("record_id");
    const auto ordinal = op.getNumOperands() == 1 ?
        dyn_cast_or_null<IntegerAttr>(lookup(op.getOperand(0))) : IntegerAttr();
    if (!source || !target || !plan || !record || !ordinal || ordinal.getValue().getActiveBits() > 64) {
        return fail("trace logical command has an unknown identity");
    }
    const bool publish = name == "pto.logical_set";
    events.push_back(llvm::json::Object{{"kind", publish ? "set" : "wait"}, {"gap", payloads},
        {"pipe", static_cast<unsigned>((publish ? source : target).getPipe())},
        {"source_pipe", static_cast<unsigned>(source.getPipe())},
        {"target_pipe", static_cast<unsigned>(target.getPipe())},
        {"plan", plan.getInt()}, {"record", record.getInt()}, {"source_ordinal", ordinal.getValue().getZExtValue()}});
    return true;
}
void Interpreter::folded(Operation& op)
{
    SmallVector<Attribute> operands;
    for (auto operand : op.getOperands()) {
        operands.push_back(lookup(operand));
    }
    SmallVector<OpFoldResult> results;
    if (failed(op.fold(operands, results)) || results.size() != op.getNumResults()) {
        for (auto result : op.getResults()) {
            values.erase(result);
        }
        return;
    }
    for (auto [value, result] : llvm::zip(op.getResults(), results)) {
        Attribute attr = result.dyn_cast<Attribute>();
        if (!attr) {
            attr = lookup(result.get<Value>());
        }
        if (attr) {
            values[value] = attr;
        } else {
            values.erase(value);
        }
    }
}
bool Interpreter::operation(Operation& op, unsigned depth)
{
    if (auto counted = dyn_cast<scf::ForOp>(op)) {
        return loop(counted, depth);
    }
    if (auto conditional = dyn_cast<scf::IfOp>(op)) {
        return branch(conditional, depth);
    }
    const auto found = types.find(&op);
    if (found != types.end()) {
        return payload(op, found->second);
    }
    const auto name = op.getName().getStringRef();
    if (name == "pto.logical_set" || name == "pto.logical_wait" || name == "pto.barrier" ||
        isa<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp, pto::WaitFlagDynOp>(op)) {
        return command(op);
    }
    if (op.getNumRegions()) {
        return fail("trace encountered unsupported nested control");
    }
    folded(op);
    return true;
}
llvm::json::Object Interpreter::run(func::FuncOp function)
{
    auto supplied = function->getAttrOfType<DenseI64ArrayAttr>("test.trace_arguments");
    std::size_t index = 0;
    for (auto argument : function.getArguments()) {
        if (argument.getType().isIntOrIndex() && supplied && index < static_cast<std::size_t>(supplied.size())) {
            values[argument] = IntegerAttr::get(argument.getType(), supplied[index++]);
        }
    }
    SmallVector<Attribute> ignored;
    const bool valid = !function.isDeclaration() && function.getBody().hasOneBlock() &&
        block(function.getBody().front(), ignored, 0);
    if (!valid && error.empty()) {
        error = "trace requires a single-block defined function";
    }
    return llvm::json::Object{{"function", function.getSymName()}, {"error", error},
        {"payloads", payloads}, {"outer_trips", outerTrips}, {"visits", visits}, {"events", std::move(events)}};
}
} // namespace
llvm::json::Object traceLogicalInsertion(func::FuncOp function, const fs::NumericTemplate& input)
{
    return Interpreter(input).run(function);
}
LogicalResult runLogicalInsertionChecks(func::FuncOp function, pto::GMAliasPolicy policy, bool physical)
{
    fs::FrontierAnalysis analysis(function);
    if (failed(analysis.initialize(policy))) {
        return failure();
    }
    const auto* program = analysis.result();
    const fs::NumericTemplate* input = nullptr;
    for (const auto& node : program->nodes) {
        if (node.numericTemplate && node.numericTemplate->result.state == fs::RecognitionState::Applicable) {
            if (input) {
                return function.emitError("trace requires one accepted numeric template");
            }
            input = &*node.numericTemplate;
        }
    }
    auto prepared = fs::prepareNumericTemplateInsertion(function, *program);
    if (!input || failed(prepared) || failed(fs::insertLogicalSynchronization(function, **prepared)) ||
        failed(verify(function))) {
        return failure();
    }
    auto trace = traceLogicalInsertion(function, *input);
    const bool valid = trace.getString("error").value_or("missing trace status").empty();
    if (physical && valid) {
        auto ids = function->getAttrOfType<DenseI64ArrayAttr>("test.eligible_ids");
        auto render = [&]() {
            std::string text;
            llvm::raw_string_ostream stream(text);
            function.print(stream);
            return text;
        };
        // Test-only reference emission: retain every enumerated endpoint.
        if (function->hasAttr("test.uncompact_endpoints")) {
            function.walk([](Operation* op) { op->removeAttr("pto.endpoint_cut"); });
        }
        const auto before = render();
        const auto eligible = ids ? ids.asArrayRef() : ArrayRef<int64_t>();
        const bool allocated = succeeded(fs::allocatePhysicalEventIds(function, eligible));
        auto after = allocated ? traceLogicalInsertion(function, *input) : llvm::json::Object{};
        llvm::outs() << llvm::json::Value(llvm::json::Object{{"allocated", allocated},
            {"unchanged_on_failure", allocated || before == render()}, {"logical", std::move(trace)},
            {"physical", std::move(after)}}) << "\n";
        return verify(function);
    }
    llvm::outs() << llvm::json::Value(std::move(trace)) << "\n";
    return success(valid);
}
