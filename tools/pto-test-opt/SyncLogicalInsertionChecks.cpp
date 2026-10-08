// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Execute inserted arith/scf control, never regenerate commands from recipes.
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/MixedStrideAnalysis.h"
#include "PTO/Transforms/FrontierSynch/VaryingRotatingRegional.h"
#include "SyncLogicalInsertionChecks.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateInsertion.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingRegional.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticInsertion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "PTO/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
constexpr uint64_t MaxVisits = 2000000;
constexpr uint64_t MaxRequestedVisits = 20000000;
constexpr uint64_t MaxEvents = 100000;
constexpr unsigned MaxDepth = 64;
class Interpreter {
public:
    explicit Interpreter(const fs::NumericTemplate& input) : input(&input) {
        for (uint32_t i = 0; i < input.payloads.size(); ++i) {
            types[input.payloads[i].phase->elementOp].push_back(i);
        }
    }
    explicit Interpreter(ArrayRef<const pto::CompoundInstanceElement*> phases) : genericPhases(phases) {
        for (uint32_t i = 0; i < phases.size(); ++i) {
            types[phases[i]->elementOp].push_back(i);
        }
    }
    llvm::json::Object run(func::FuncOp function);
    std::optional<bool> truth(Value value) const {
        auto result = integer(value);
        return result ? std::optional<bool>(*result != 0) : std::nullopt;
    }
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
    const fs::NumericTemplate* input = nullptr;
    SmallVector<const pto::CompoundInstanceElement*> genericPhases;
    SmallVector<int64_t> coordinates;
    DenseMap<Value, Attribute> values;
    DenseMap<Operation*, SmallVector<uint32_t>> types;
    llvm::json::Array events;
    std::string error;
    uint64_t visits = 0;
    uint64_t visitLimit = MaxVisits;
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
        if (++visits > visitLimit || events.size() >= MaxEvents) {
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
        if (input && op == input->outer) {
            ++outerTrips;
        }
        values[op.getInductionVar()] = IntegerAttr::get(op.getInductionVar().getType(), induction);
        bind(op.getRegionIterArgs(), state);
        SmallVector<Attribute> next;
        coordinates.push_back(induction);
        const bool executed = block(*op.getBody(), next, depth + 1);
        coordinates.pop_back();
        if (!executed || next.size() != state.size()) {
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
    if (!input) {
        if (candidates.size() != 1) {
            return fail("structured trace requires one phase per payload anchor");
        }
        const auto type = candidates.front();
        llvm::json::Array tuple;
        for (auto coordinate : coordinates) {
            tuple.push_back(coordinate);
        }
        llvm::json::Object event{{"kind", "payload"}, {"type", type},
            {"coordinates", std::move(tuple)},
            {"pipe", static_cast<unsigned>(genericPhases[type]->kPipeValue)}};
        if (auto label = op.getAttrOfType<StringAttr>("test.label")) {
            event["label"] = label.getValue();
        }
        events.push_back(std::move(event));
        ++payloads;
        for (auto result : op.getResults()) {
            values.erase(result);
        }
        return true;
    }
    std::optional<uint32_t> selected;
    for (auto type : candidates) {
        bool matches = true;
        for (const auto& coordinate : input->payloads[type].coordinates) {
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
    auto outer = input->outer;
    const auto induction = integer(outer.getInductionVar());
    if (!selected || !induction || *induction < input->lower || input->step <= 0) {
        return fail("trace payload is outside its template domain");
    }
    const uint64_t difference = static_cast<uint64_t>(*induction) - static_cast<uint64_t>(input->lower);
    if (difference % static_cast<uint64_t>(input->step)) {
        return fail("trace payload outer coordinate is off grid");
    }
    const auto& item = input->payloads[*selected];
    events.push_back(llvm::json::Object{{"kind", "payload"}, {"type", *selected},
        {"ordinal", difference / static_cast<uint64_t>(input->step)},
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
    const auto ordinal = op.getNumOperands() >= 1 ?
        dyn_cast_or_null<IntegerAttr>(lookup(op.getOperand(0))) : IntegerAttr();
    if (!source || !target || !plan || !record || !ordinal || ordinal.getValue().getActiveBits() > 64) {
        return fail("trace logical command has an unknown identity");
    }
    llvm::json::Array members;
    for (auto operand : op.getOperands().drop_front()) {
        auto member = dyn_cast_or_null<IntegerAttr>(lookup(operand));
        if (!member || member.getValue().getActiveBits() > 64) {
            return fail("trace logical command has an unknown family member");
        }
        members.push_back(member.getValue().getZExtValue());
    }
    const bool publish = name == "pto.logical_set";
    llvm::json::Object event(llvm::json::Object{{"kind", publish ? "set" : "wait"}, {"gap", payloads},
        {"pipe", static_cast<unsigned>((publish ? source : target).getPipe())},
        {"source_pipe", static_cast<unsigned>(source.getPipe())},
        {"target_pipe", static_cast<unsigned>(target.getPipe())},
        {"plan", plan.getInt()}, {"record", record.getInt()}, {"source_ordinal", ordinal.getValue().getZExtValue()}});
    if (op.hasAttr("pto.endpoint_piece")) {
        event["record_label"] = true;
    }
    if (!members.empty()) {
        event["members"] = std::move(members);
    }
    events.push_back(std::move(event));
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
    // This is an explicit test-only ceiling for scalar interpretation. Event
    // and depth limits remain fixed; production analysis does not consult it.
    if (auto requested = function->getAttr("test.trace_visit_limit")) {
        auto integer = dyn_cast<IntegerAttr>(requested);
        if (!integer || !integer.getValue().isSignedIntN(64) || integer.getInt() <= 0 ||
            static_cast<uint64_t>(integer.getInt()) > MaxRequestedVisits) {
            return llvm::json::Object{{"function", function.getSymName()},
                {"error", "test.trace_visit_limit must be in [1, 20000000]"}};
        }
        visitLimit = static_cast<uint64_t>(integer.getInt());
    }
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
        {"payloads", payloads}, {"outer_trips", outerTrips}, {"visits", visits},
        {"visit_limit", visitLimit}, {"events", std::move(events)}};
}
} // namespace
llvm::json::Object traceStructuredLogicalInsertion(func::FuncOp function,
    ArrayRef<const mlir::pto::CompoundInstanceElement*> phases)
{
    return Interpreter(phases).run(function);
}
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
    if (failed(analysis.analyzeNumericCandidates())) {
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

LogicalResult runStructuredInsertionChecks(func::FuncOp function, pto::GMAliasPolicy policy)
{
    pto::SyncInput input(policy);
    if (failed(input.build(function))) {
        return failure();
    }
    auto render = [&]() {
        std::string text;
        llvm::raw_string_ostream stream(text);
        function.print(stream);
        return text;
    };
    const auto before = render();
    if (auto requested = function->getAttrOfType<DenseI64ArrayAttr>("test.varying_queries")) {
        auto program = fs::recognizeProgram(function,input);
        fs::PhaseIndex index;
        if (failed(program) || failed(index.build(function,input))) { return failure(); }
        auto arena = std::make_shared<fs::RegionExpressions>();
        std::string error;
        FailureOr<fs::RegionalAnalysis> result = failure();
        for (const auto& node : program->nodes) {
            if (node.varyingRotating && node.varyingRotating->result.state == fs::RecognitionState::Applicable) {
                result = fs::varyingRotatingRegionalResult(function,*node.varyingRotating,index,input,arena,error);
                break;
            }
        }
        if (failed(result) || requested.size()%3) { llvm::errs() << error; return failure(); }
        std::vector<fs::RegionalEvent> events;
        for (int64_t i=0;i<requested.size();i+=3) {
            for (auto kind : {fs::PeriodicEventKind::Start,fs::PeriodicEventKind::Completion}) {
                events.push_back({static_cast<uint32_t>(requested[i]),arena->constant(requested[i+2]),kind,
                    {arena->constant(requested[i+1])}});
            }
        }
        Block code;
        OpBuilder builder(function.getContext()); builder.setInsertionPointToEnd(&code);
        auto* cut = function.front().getTerminator();
        fs::RegionExpressions::CutEmission context;
        SmallVector<Value> answers;
        for (auto a : events) {
            for (auto b : events) {
                auto query = fs::regionalReachability(*result,a,b);
                if (!query) { return failure(); }
                auto value = arena->emitContextual(*query,builder,cut,context);
                if (failed(value)) { llvm::errs() << arena->error(); return failure(); }
                answers.push_back(*value);
            }
        }
        function.front().getOperations().splice(cut->getIterator(),code.getOperations());
        Interpreter interpreter(input.instructions());
        auto trace = interpreter.run(function);
        llvm::json::Array queries;
        for (auto value : answers) {
            auto answer = interpreter.truth(value);
            if (!answer) { return failure(); }
            queries.push_back(*answer);
        }
        llvm::outs() << llvm::json::Value(llvm::json::Object{{"queries",std::move(queries)},
            {"error",trace.getString("error").value_or("missing trace status")}}) << "\n";
        return verify(function);
    }
    bool accepted = false;
    std::shared_ptr<fs::BoundedLifetimeDemandResult> boundedDemands;
    if (function->hasAttr("test.mixed_stride_insertion")) {
        auto program = fs::recognizeProgram(function, input);
        std::string error;
        if (succeeded(program)) {
            auto prepared = fs::prepareMixedStrideInsertion(function, input, *program, error);
            accepted = succeeded(prepared) && succeeded(fs::insertLogicalSynchronization(function, **prepared));
        }
        if (!accepted) { llvm::errs() << error << "\n"; }
    } else if (function->hasAttr("test.bounded_lifetime_insertion")) {
        auto program = fs::recognizeProgram(function,input);
        std::string error;
        if (succeeded(program)) {
            auto prepared = fs::prepareBoundedLifetimeInsertion(function,input,*program,error,&boundedDemands);
            accepted = succeeded(prepared) && succeeded(fs::insertLogicalSynchronization(function,**prepared));
        }
        if (!accepted) { llvm::errs() << error << "\n"; }
    } else if (function->hasAttr("test.arithmetic_insertion")) {
        fs::FrontierAnalysis analysis(function);
        std::string error;
        if (succeeded(analysis.initialize(policy)) && succeeded(analysis.recognizeArithmetic())) {
            const auto& program = *analysis.result()->arithmetic;
            auto demands = fs::analyzeGeneralArithmeticDemands(program);
            auto prepared = fs::prepareGeneralArithmeticInsertion(function, program, demands, error);
            accepted = succeeded(prepared) && succeeded(fs::insertLogicalSynchronization(function, **prepared));
            if (accepted && failed(verify(function))) { return failure(); }
        }
    } else if (function->hasAttr("test.recompose")) {
        auto program = fs::recognizeProgram(function, input);
        if (failed(program)) { return failure(); }
        fs::RegionalAnalysis child;
        {
            auto original = fs::analyzeSequence(function, input, *program);
            child = fs::sequenceRegionalResult(original);
        } // Exported callbacks must retain the original analysis state.
        auto arena = child.expressions;
        fs::RegionalAnalysis empty;
        empty.expressions = arena;
        empty.capabilities = {true, true, true, true};
        empty.presence = [](fs::RegionalEvent) -> std::optional<fs::RegionExpressions::Id> {
            return std::nullopt;
        };
        empty.reachability = [](fs::RegionalEvent, fs::RegionalEvent) -> std::optional<fs::RegionExpressions::Id> {
            return std::nullopt;
        };
        empty.prepare = []() -> FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> {
            return std::make_unique<fs::PreparedLogicalPlan>(0);
        };
        std::vector<fs::RegionalAnalysis> regions(8, empty);
        regions.push_back(std::move(child));
        regions.insert(regions.end(), 8, empty);
        auto composed = fs::composeRegionalSequence(function, arena, std::move(regions));
        auto prepared = fs::prepareSequenceInsertion(composed);
        accepted = succeeded(prepared) && succeeded(fs::insertLogicalSynchronization(function, **prepared));
    } else {
        auto prepared = fs::prepareFunctionSynchronization(function, policy);
        accepted = succeeded(prepared) && succeeded(fs::insertLogicalSynchronization(function, **prepared));
        if (accepted && failed(verify(function))) { return failure(); }
    }
    // The preparation stack (including recognition and detached endpoint code)
    // is gone. The cached DAG must still support demand analysis, particularly
    // after a replay failure has discarded the executable plan.
    if (boundedDemands) {
        auto replay = fs::analyzeLifetimeWindow(boundedDemands->expressions, boundedDemands->window);
        const auto& original = boundedDemands->analysis.sourceDemands;
        if (!replay.error.empty() || replay.sourceDemands.size() != original.size()) { return failure(); }
        for (std::size_t i = 0; i < original.size(); ++i) {
            const auto& a = original[i];
            const auto& b = replay.sourceDemands[i];
            if (a.source != b.source || a.target != b.target || a.guard != b.guard) { return failure(); }
            for (bool truth : {false, true}) {
                std::vector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> bindings;
                for (auto [id, value] : boundedDemands->expressions.referencedInputs(a.guard)) {
                    if (!value || !value.getType()) { return failure(); }
                    auto argument = dyn_cast<BlockArgument>(value);
                    auto replacement = value.getType().isInteger(1) ? boundedDemands->expressions.boolean(truth) :
                        boundedDemands->expressions.constant(
                            argument && argument.getOwner() == &boundedDemands->symbols ? 0 : 7);
                    bindings.push_back({id, replacement});
                }
                fs::RegionExpressions::Substitution substitution(bindings);
                auto concrete = boundedDemands->expressions.substitute(a.guard, substitution);
                if (!boundedDemands->expressions.constantValue(concrete)) { return failure(); }
            }
        }
    }
    auto trace = accepted ? Interpreter(input.instructions()).run(function) : llvm::json::Object{};
    const bool valid = !accepted || trace.getString("error").value_or("missing trace status").empty();
    const auto boundedCount = boundedDemands ? static_cast<int64_t>(boundedDemands->analysis.sourceDemands.size()) : 0;
    llvm::json::Object report{{"function", function.getSymName()},
        {"accepted", accepted}, {"unchanged_on_failure", accepted || before == render()},
        {"bounded_demands_retained", bool(boundedDemands)},
        {"bounded_demand_count", boundedCount}};
    auto ids = function->getAttrOfType<DenseI64ArrayAttr>("test.eligible_ids");
    if (accepted && ids) {
        auto logicalIR = render();
        bool allocated = succeeded(fs::allocatePhysicalEventIds(function,ids.asArrayRef()));
        report["allocated"] = allocated;
        report["allocation_unchanged_on_failure"] = allocated || logicalIR == render();
        if (allocated) {
            // Counter allocation may replace the function body transactionally.
            // Recover anchors from the resulting IR rather than retaining
            // preallocation pointers to cloned or erased payload operations.
            pto::SyncInput physicalInput(policy);
            if (failed(physicalInput.build(function))) { return failure(); }
            report["physical"] = Interpreter(physicalInput.instructions()).run(function);
        }
    }
    report["trace"] = std::move(trace);
    llvm::outs() << llvm::json::Value(std::move(report)) << "\n";
    return success(valid && succeeded(verify(function)));
}

namespace {
// Independent unfolded event graph for three original arithmetic siblings.
// The IR fixture uses overlapping 32-byte UB windows at byte4*i. The middle
// store also overwrites one fixed GM destination. Native edges order starts
// and completions separately; they never serialize complete payload intervals.
bool checkSymbolicArithmeticSiblings(func::FuncOp function, const pto::SyncInput& input)
{
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
    SmallVector<scf::ForOp> loops;
    for (auto& operation : function.front()) {
        if (auto loop = dyn_cast<scf::ForOp>(operation)) { loops.push_back(loop); }
    }
    if (loops.size() != 3) { return false; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::vector<fs::RegionalAnalysis> children;
    std::string error;
    for (auto loop : loops) {
        std::string deferred;
        auto finite = fs::analyzeFiniteArithmeticRegion({function, loop}, index, input, arena, deferred);
        if (succeeded(finite) || deferred.find("finite regional export deferred") == std::string::npos) {
            llvm::errs() << "symbolic sibling unexpectedly required a finite export\n"; return false;
        }
        auto child = fs::analyzeArithmeticRegion({function, loop}, index, input, arena, error);
        if (failed(child) || child->anchors.size() != 1 || !child->arithmeticRelations) {
            llvm::errs() << "symbolic child: " << error << "\n"; return false;
        }
        // Retain every nondischarged fixed GM family alongside the symbolic
        // UB family. Globally read-only GM effects may already be discharged
        // by the shared analysis; no finite selector is needed for those.
        bool finiteGM = false;
        for (const auto& access : child->accessBoundary) {
            const bool gm = input.accesses().effects()[access.effect].memory->scope == pto::AddressSpace::GM;
            finiteGM |= gm;
            if (access.representedByCells != gm) { return false; }
        }
        if ((finiteGM && child->storageBoundary.empty()) || child->symbolicStorageEffects.empty()) {
            llvm::errs() << "partial arithmetic storage families were not retained\n"; return false;
        }
        children.push_back(std::move(*child));
    }
    auto first = fs::composeRegionalSequence(function, arena, {children[0], children[1]}, true, false);
    if (!first.error.empty()) { llvm::errs() << first.error << "\n"; return false; }
    auto pair = fs::sequenceRegionalResult(first);
    if (!pair.arithmeticRelations || !pair.storageSelectors) { return false; }
    auto whole = fs::composeRegionalSequence(function, arena, {pair, children[2]}, true, false);
    if (!whole.error.empty()) { llvm::errs() << whole.error << "\n"; return false; }
    auto regional = fs::sequenceRegionalResult(whole);
    auto verify = [&](const fs::RegionalAnalysis& region, unsigned stages) {
        if (!region.arithmeticRelations || region.anchors.size() != stages) { return false; }
        const auto& relations = *region.arithmeticRelations;
        if (relations.program.parameters.size() != 1 || relations.program.parameters[0] != function.getArgument(2)) {
            return false;
        }
        auto& e = *arena;
        for (unsigned trips : {0U, 1U, 2U, 4U, 9U}) {
            const unsigned payloads = stages * trips, events = 2 * payloads;
            std::vector<std::vector<bool>> native(events, std::vector<bool>(events));
            std::vector<std::vector<bool>> required = native;
            auto pipe = [&](unsigned occurrence) {
                return region.anchors[occurrence / trips].phase->kPipeValue;
            };
            for (unsigned a = 0; a < payloads; ++a) {
                native[2*a][2*a+1] = required[2*a][2*a+1] = true;
                for (unsigned b = a + 1; b < payloads; ++b) {
                    if (pipe(a) == pipe(b)) {
                        native[2*a][2*b] = required[2*a][2*b] = true;
                        native[2*a+1][2*b+1] = required[2*a+1][2*b+1] = true;
                    }
                    const unsigned sa = a / trips, sb = b / trips, ia = a % trips, ib = b % trips;
                    const bool overlap = 4*ia < 4*ib + 32 && 4*ib < 4*ia + 32;
                    const bool conflict = (overlap && (sa != 1 || sb != 1)) || (sa == 1 && sb == 1);
                    if (conflict) { required[2*a+1][2*b] = true; }
                }
            }
            auto close = [&](auto& graph) {
                for (unsigned k = 0; k < events; ++k) {
                    for (unsigned a = 0; a < events; ++a) {
                        for (unsigned b = 0; b < events; ++b) {
                            graph[a][b] = graph[a][b] || (graph[a][k] && graph[k][b]);
                        }
                    }
                }
            };
            close(native); close(required);
            fs::RegionExpressions::Substitution bindings({{e.input(function.getArgument(2)), e.constant(trips)}});
            auto event = [&](unsigned type, unsigned ordinal, unsigned kind) {
                return fs::RegionalEvent{type, e.constant(ordinal), kind ? fs::PeriodicEventKind::Completion :
                                                                         fs::PeriodicEventKind::Start};
            };
            // Include absent coordinates and the zero-trip invocation.
            for (unsigned sa = 0; sa < stages; ++sa) {
                for (unsigned sb = 0; sb < stages; ++sb) {
                    for (unsigned ia = 0; ia <= trips; ++ia) {
                        for (unsigned ib = 0; ib <= trips; ++ib) {
                            for (unsigned ka = 0; ka < 2; ++ka) {
                                for (unsigned kb = 0; kb < 2; ++kb) {
                                    auto query = fs::regionalReachability(region, event(sa, ia, ka), event(sb, ib, kb));
                                    const unsigned a = 2*(sa*trips + ia)+ka, b = 2*(sb*trips + ib)+kb;
                                    const bool expected = ia < trips && ib < trips && (a == b || required[a][b]);
                                    if (!query || e.constantValue(e.substitute(*query, bindings)) !=
                                                  uint64_t(expected)) {
                                        llvm::errs() << "symbolic sibling query mismatch\n"; return false;
                                    }
                                }
                            }
                            bool minimum = false;
                            for (const auto& [key, pieces] : relations.analysis.minimumDemands) {
                                if (key.source.site != sa || key.target.site != sb) { continue; }
                                SmallVector<uint64_t> residues(key.source.residues.begin(), key.source.residues.end());
                                llvm::append_range(residues, key.target.residues);
                                llvm::append_range(residues, key.parameterResidues);
                                for (const auto& piece : pieces) {
                                    auto member = e.integerPredicate(piece,
                                        {e.constant(ia), e.constant(ib), e.constant(trips)},
                                        relations.analysis.period, residues);
                                    auto value = e.constantValue(member);
                                    if (!value) { return false; }
                                    minimum |= *value != 0;
                                }
                            }
                            bool expected = false;
                            if (ia < trips && ib < trips) {
                                const unsigned a = 2*(sa*trips + ia)+1, b = 2*(sb*trips + ib);
                                expected = required[a][b] && !native[a][b];
                                for (unsigned k = 0; k < events; ++k) {
                                    expected &= !(required[a][k] && required[k][b]);
                                }
                            }
                            if (minimum != expected) {
                                llvm::errs() << "symbolic sibling cover mismatch\n"; return false;
                            }
                        }
                    }
                }
            }
        }
        return true;
    };
    if (!verify(pair, 2) || !verify(regional, 3)) { return false; }
    auto prepared = fs::prepareSequenceInsertion(whole);
    if (failed(prepared)) {
        llvm::errs() << "symbolic endpoint export: " << whole.insertionError << "\n"; return false;
    }
    if (failed(fs::insertLogicalSynchronization(function, **prepared)) || failed(mlir::verify(function))) {
        return false;
    }
    auto trace = Interpreter(input.instructions()).run(function);
    if (!trace.getString("error").value_or("missing trace").empty()) {
        llvm::errs() << llvm::json::Value(std::move(trace)); return false;
    }
    llvm::outs() << "symbolic arithmetic siblings: exact queries, covers, recursive exports, and endpoints passed\n";
    return true;
}
// Reduced from lossless_block_cast: a finite scalar fill feeds one vector
// consumer. Different elements retain different last writers, so this tests
// actual selector cardinality rather than a large uniform enclosing interval.
bool checkFiniteArithmeticFill(func::FuncOp function, const pto::SyncInput& input)
{
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
    scf::ForOp loop;
    SmallVector<Operation*> suffix;
    for (auto& operation : function.front()) {
        if (auto candidate = dyn_cast<scf::ForOp>(operation)) { loop = candidate; }
        else if (loop && !operation.hasTrait<OpTrait::IsTerminator>()) { suffix.push_back(&operation); }
    }
    if (!loop) { return false; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::string error;
    auto child = fs::analyzeFiniteArithmeticRegion({function, loop}, index, input, arena, error);
    constexpr uint64_t elements = 17 * 16;
    if (failed(child) || !child->symbolicStorageEffects.empty() ||
        child->storageBoundary.size() != elements || child->cost.cells != elements ||
        child->cost.boundaryBytes != elements * 4) {
        llvm::errs() << "finite arithmetic fill lost its charged selectors: " << error << "\n";
        return false;
    }
    for (uint64_t element = 0; element < elements; ++element) {
        const auto& boundary = child->storageBoundary[element];
        if (boundary.cell.begin != 4 * element || boundary.cell.end != 4 * (element + 1) ||
            boundary.lastWriters.size() != 1) { return false; }
        const auto& writer = boundary.lastWriters.front();
        if (writer.event.visits.size() != 1 || arena->constantValue(writer.present) != 1 ||
            arena->constantValue(writer.event.ordinal) != element % 16 ||
            arena->constantValue(writer.event.visits.front()) != element / 16) { return false; }
    }
    auto finite = fs::analyzeFiniteGuarded(function, suffix, index, input, arena);
    if (!finite.error.empty()) { return false; }
    auto composed = fs::composeRegionalSequence(function, arena,
        {*child, fs::finiteGuardedRegionalResult(finite)}, true, false);
    if (!composed.error.empty() || failed(fs::prepareSequenceInsertion(composed))) {
        llvm::errs() << "finite arithmetic fill composition: " << composed.error << "\n";
        return false;
    }
    auto program = fs::recognizeProgram(function, input);
    if (failed(program)) { return false; }
    auto dispatched = fs::analyzeSequence(function, input, *program);
    if (!dispatched.error.empty() || dispatched.cost.arithmeticRegions == 0 ||
        failed(fs::prepareSequenceInsertion(dispatched))) { return false; }
    llvm::outs() << "finite arithmetic fill: 272 exact selectors and mixed sequence endpoints passed\n";
    return true;
}
// Incoming scalar ordering must target the first executed site occurrence,
// including a guarded nonzero prefix inside a nested arithmetic child.
bool checkArithmeticEntryPrerequisite(func::FuncOp function, const pto::SyncInput& input)
{
    auto expected = function->getAttrOfType<BoolAttr>("test.arithmetic_entry_prerequisite");
    if (!expected) { return true; }
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
    SmallVector<Operation*> prefix;
    scf::ForOp loop;
    for (auto& op : function.front()) {
        if (auto candidate = dyn_cast<scf::ForOp>(op)) { loop = candidate; break; }
        prefix.push_back(&op);
    }
    if (!loop) { return false; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto finite = fs::analyzeFiniteGuarded(function, prefix, index, input, arena);
    auto first = fs::finiteGuardedRegionalResult(finite);
    std::string error;
    auto second = fs::analyzeArithmeticRegion({function, loop}, index, input, arena, error);
    if (!finite.error.empty() || failed(second) || first.anchors.size() != 1 || second->anchors.size() != 1) {
        return false;
    }
    if (!second->firstSitePayloads.count(0)) { return false; }
    const auto zero = arena->constant(0);
    const fs::SequenceEvent source{0, 0, zero, fs::PeriodicEventKind::Completion};
    auto check = [&](fs::RegionalAnalysis child) {
        auto composed = fs::composeRegionalSequence(function, arena, {first, std::move(child)}, true, false);
        if (!composed.error.empty()) { return false; }
        for (uint64_t row = 0; row < 2; ++row) {
            for (uint64_t column = 0; column < 4; ++column) {
                auto reaches = fs::sequenceEventReachability(composed, source,
                    {1, 0, arena->constant(column), fs::PeriodicEventKind::Start, {arena->constant(row)}});
                const bool required = expected.getValue() && column >= 2;
                if (!reaches || arena->constantValue(*reaches) != static_cast<uint64_t>(required)) { return false; }
            }
        }
        return true;
    };
    if (!check(*second)) { return false; }
    auto wrapper = fs::composeRegionalSequence(function, arena, {*second}, false, false);
    if (!wrapper.error.empty()) { return false; }
    return check(fs::sequenceRegionalResult(wrapper));
}
// Exercise the public composition API, rather than boundaryLoop's private path.
bool checkSlicePrerequisite(func::FuncOp function, const pto::SyncInput& input)
{
    auto bounds = function->getAttrOfType<DenseI64ArrayAttr>("test.slice_bounds");
    if (!bounds) { return true; }
    if (bounds.size() != 2 || bounds[0] < 0 || bounds[1] < bounds[0]) { return false; }
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
    SmallVector<Operation*> prefix;
    scf::ForOp loop;
    for (auto& op : function.front()) {
        if (auto candidate = dyn_cast<scf::ForOp>(op)) { loop = candidate; break; }
        prefix.push_back(&op);
    }
    if (!loop) { return false; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto finite = fs::analyzeFiniteGuarded(function, prefix, index, input, arena);
    auto first = fs::finiteGuardedRegionalResult(finite);
    auto recognized = fs::recognizeGuardedRotating(loop, index, input, input.accesses());
    auto periodic = fs::analyzeGuardedRotating(loop, input, recognized, arena);
    if (!finite.error.empty() || !periodic.error.empty()) { return false; }
    const auto zero = arena->constant(0), begin = arena->constant(bounds[0]), end = arena->constant(bounds[1]);
    std::string error;
    auto second = fs::guardedRotatingRegionalResult(function, input, periodic, error, fs::PeriodicSlice{begin, end});
    if (failed(second) || first.anchors.size() != 1 || second->anchors.size() != 1) { return false; }
    auto compose = [&](fs::RegionalAnalysis slice) {
        return fs::composeRegionalSequence(function, arena, {first, std::move(slice)});
    };
    auto composed = compose(*second);
    if (!composed.error.empty()) { return false; }
    auto required = fs::sequenceEventReachability(composed,
        {0, 0, zero, fs::PeriodicEventKind::Completion}, {1, 0, begin, fs::PeriodicEventKind::Start});
    if (!required || arena->constantValue(*required) != 1) { return false; }
    auto absent = fs::sequenceEventReachability(composed,
        {0, 0, zero, fs::PeriodicEventKind::Completion}, {1, 0, zero, fs::PeriodicEventKind::Start});
    if (!absent || arena->constantValue(*absent) != 0) { return false; }
    second->firstOrdinal = arena->boolean(true);
    if (compose(*second).error.empty()) { return false; }
    second->firstOrdinal = fs::RegionExpressions::invalid;
    return !compose(*second).error.empty();
}
} // namespace
LogicalResult runSequenceAnalysisChecks(func::FuncOp function, pto::GMAliasPolicy policy)
{
    pto::SyncInput input(policy);
    if (failed(input.build(function))) { return failure(); }
    if (function->hasAttr("test.finite_arithmetic_fill")) {
        return success(checkFiniteArithmeticFill(function, input));
    }
    if (function->hasAttr("test.symbolic_arithmetic_siblings")) {
        return success(checkSymbolicArithmeticSiblings(function, input));
    }
    auto program = fs::recognizeProgram(function, input);
    if (failed(program)) { return failure(); }
    std::string before;
    llvm::raw_string_ostream original(before);
    function.print(original);
    bool regionScope = true;
    // Independent scope-contract probes deliberately analyze each child on
    // its own. Keep them on the dedicated fixture: an unsupported child may
    // require expensive arithmetic even when its parent has a cheap route.
    const bool checkRegionScope = function->hasAttr("test.check_region_scope");
    for (std::size_t node = 1; checkRegionScope && node < program->nodes.size(); ++node) {
        const auto& entry = program->nodes[node];
        if (entry.kind != fs::StructureKind::ExplicitRun && entry.kind != fs::StructureKind::Loop) { continue; }
        auto child = fs::analyzeSequenceRegion(function, input, *program, node,
            std::make_shared<fs::RegionExpressions>());
        if (!child.error.empty()) { continue; }
        auto childPlan = fs::prepareSequenceInsertion(child);
        if (!entry.loops.empty()) { regionScope &= failed(childPlan); }
        else if (succeeded(childPlan)) { regionScope &= !(**childPlan).completeInvocation; }
    }
    auto analysis = fs::analyzeSequence(function, input, *program);
    uint64_t boundedRegions = 0, boundedRecords = 0;
    bool boundedCacheStable = true;
    for (const auto& node : program->nodes) {
        if (!node.boundedDemands) { continue; }
        ++boundedRegions; boundedRecords += node.boundedDemands->analysis.sourceDemands.size();
        fs::PhaseIndex index;
        if (failed(index.build(function, input))) { return failure(); }
        std::string diagnostic;
        auto again = fs::cachedBoundedLifetimeRegion(function, node, index, input, diagnostic);
        boundedCacheStable &= succeeded(again) && *again == node.boundedDemands;
        // A fresh modeled input cannot borrow another input's result, even
        // when it describes the same unchanged original IR.
        pto::SyncInput other(policy);
        if (failed(other.build(function))) { return failure(); }
        auto foreign = fs::cachedBoundedLifetimeRegion(function, node, index, other, diagnostic);
        boundedCacheStable &= failed(foreign) && !diagnostic.empty();
    }
    // Reuse this actual analysis: copied recognition objects must not inherit
    // evidence bound to the original program, even with identical node values.
    auto foreignProgram = *program;
    fs::recordSequenceContractAttempt(foreignProgram, input, analysis);
    bool contractAudit = foreignProgram.sequenceContract &&
        foreignProgram.sequenceContract->membership == fs::ContractStatus::Unproved;
    fs::recordSequenceContractAttempt(*program, input, analysis);
    const auto expectedMembership = analysis.error.empty() && analysis.state ?
        fs::ContractStatus::Established : fs::ContractStatus::Unproved;
    contractAudit &= program->sequenceContract && program->sequenceContract->membership == expectedMembership;
    fs::recordSequenceEndpointAttempt(*program, "synthetic endpoint capability gap");
    contractAudit &= program->sequenceContract->membership == expectedMembership &&
        program->sequenceContract->endpointRecipes == fs::ContractImplementation::Unavailable;
    fs::refreshProgramContractAudit(*program);
    contractAudit &= llvm::any_of(program->contractAudit, [&](const auto& candidate) {
        return candidate.kind == fs::ContractClass::Sequence && candidate.node == 0 &&
            candidate.membership == expectedMembership &&
            candidate.endpointRecipes == fs::ContractImplementation::Unavailable;
    });
    auto prepared = fs::prepareSequenceInsertion(analysis);
    fs::recordSequenceEndpointAttempt(*program, failed(prepared) ? analysis.insertionError : "");
    contractAudit &= program->sequenceContract->membership == expectedMembership;
    auto regional = fs::sequenceRegionalResult(analysis);
    auto* expressions = fs::sequenceExpressions(analysis);
    bool validQueries = analysis.error.empty() && bool(analysis.state);
    for (uint32_t port = 0; port < analysis.occurrences.size(); ++port) {
        auto result = fs::sequenceEventReachability(analysis, port, fs::PeriodicEventKind::Start,
                                                   port, fs::PeriodicEventKind::Completion);
        validQueries &= result.has_value();
    }
    uint64_t emitted = 0;
    if (succeeded(prepared)) {
        // Detached preparation owns every expression operation at original cuts.
        for (const auto& stage : (**prepared).preparation) {
            emitted += stage.code->getOperations().size();
        }
    }
    std::string after;
    llvm::raw_string_ostream current(after);
    function.print(current);
    llvm::outs() << llvm::json::Value(llvm::json::Object{
        {"function", function.getSymName()}, {"error", analysis.error},
        {"bounded_cached_regions", boundedRegions}, {"bounded_cached_demands", boundedRecords},
        {"bounded_cache_stable", boundedCacheStable},
        {"insertion_error", analysis.insertionError},
        {"prepared", succeeded(prepared)},
        {"nested_matching", succeeded(prepared) && (**prepared).nestedIdentities},
        {"symbolic_storage_effects", regional.symbolicStorageEffects.size()},
        {"storage_selector_interface", bool(regional.storageSelectors)},
        {"allocation_interface", failed(prepared) ? "no-logical-plan" :
            ((**prepared).regionalAllocation || (**prepared).allocationCertificate ? "constructed" : "unavailable")},
        {"queries_available", validQueries}, {"unchanged", before == after}, {"contract_audit", contractAudit},
        {"slice_prerequisite", checkSlicePrerequisite(function, input)},
        {"region_scope", checkRegionScope ? llvm::json::Value(regionScope) : llvm::json::Value(nullptr)},
        {"arithmetic_entry_prerequisite", checkArithmeticEntryPrerequisite(function, input)},
        {"children", analysis.cost.children}, {"cells", analysis.cost.cells},
        {"ports", analysis.cost.ports}, {"crossings", analysis.cost.crossings},
        {"physical_fragments", analysis.cost.physicalFragments},
        {"arithmetic_regions", analysis.cost.arithmeticRegions}, {"boundary_bytes", analysis.cost.boundaryBytes},
        {"child_preparation", fs::sequencePreparationCounts(analysis).first},
        {"crossing_preparation", fs::sequencePreparationCounts(analysis).second},
        {"rotating_residues", analysis.cost.rotatingResidues}, {"numeric_visits", analysis.cost.numericVisits},
        {"repeated_regions", analysis.cost.repeatedRegions}, {"phase_descriptions", analysis.cost.phaseDescriptions},
        {"selector_comparisons", analysis.cost.selectorComparisons},
        {"crossing_candidates", analysis.cost.crossingCandidates},
        {"implication_checks", analysis.cost.implicationChecks},
        {"expressions", expressions ? expressions->size() : 0}, {"emitted", emitted}}) << "\n";
    return success(before == after && regionScope && contractAudit && boundedCacheStable);
}

LogicalResult runFiniteGuardedAnalysisChecks(func::FuncOp function, pto::GMAliasPolicy policy)
{
    pto::SyncInput input(policy);
    fs::PhaseIndex index;
    if (failed(input.build(function)) || failed(index.build(function, input))) { return failure(); }
    SmallVector<Operation*> roots;
    for (auto& op : function.front()) { roots.push_back(&op); }
    auto render = [&]() {
        std::string text;
        llvm::raw_string_ostream stream(text);
        function.print(stream);
        return text;
    };
    const auto before = render();
    auto analysis = fs::analyzeFiniteGuarded(function, roots, index, input);
    auto regional = fs::finiteGuardedRegionalResult(analysis);
    auto prepared = fs::prepareFiniteGuardedInsertion(analysis);
    bool queriesAvailable = true;
    uint64_t emitted = 0;
    if (succeeded(prepared)) {
        for (const auto& stage : (**prepared).preparation) { emitted += stage.code->getOperations().size(); }
    }
    llvm::json::Object report{{"error", analysis.error}, {"insertion_error", analysis.insertionError},
        {"prepared", succeeded(prepared)}, {"unchanged", before == render()},
        {"cells", analysis.cost.cells}, {"ports", analysis.cost.ports},
        {"analysis_expressions", analysis.cost.expressionNodes},
        {"expressions", regional.expressions ? regional.expressions->size() : 0}, {"emitted", emitted}};
    if (regional.reachability && regional.expressions) {
        const auto zero = regional.expressions->constant(0);
        for (uint32_t type = 0; type < regional.anchors.size(); ++type) {
            queriesAvailable &= regional.reachability({type, zero, fs::PeriodicEventKind::Start},
                {type, zero, fs::PeriodicEventKind::Completion}).has_value();
        }
        // Query execution is test-only and requested only for entry-available
        // predicates. Analysis and detached preparation were checked unchanged above.
        if (function->hasAttr("test.queries")) {
            llvm::json::Array labels;
            for (const auto& anchor : regional.anchors) {
                auto label = anchor.phase->elementOp->getAttrOfType<StringAttr>("test.label");
                labels.push_back(label ? label.getValue() : StringRef());
            }
            OpBuilder builder(function.getContext());
            auto* cut = function.front().getTerminator();
            Block queryCode;
            builder.setInsertionPointToEnd(&queryCode);
            llvm::DenseMap<fs::RegionExpressions::Id, Value> memo;
            SmallVector<Value> values;
            for (uint32_t a = 0; a < 2 * regional.anchors.size(); ++a) {
                for (uint32_t b = 0; b < 2 * regional.anchors.size(); ++b) {
                    auto predicate = regional.reachability({a/2, zero, a%2 ? fs::PeriodicEventKind::Completion :
                        fs::PeriodicEventKind::Start}, {b/2, zero, b%2 ? fs::PeriodicEventKind::Completion :
                        fs::PeriodicEventKind::Start});
                    if (!predicate) { return failure(); }
                    auto value = regional.expressions->emit(*predicate, builder, cut, memo);
                    if (failed(value)) { return failure(); }
                    values.push_back(*value);
                }
            }
            function.front().getOperations().splice(cut->getIterator(), queryCode.getOperations());
            Interpreter interpreter(input.instructions());
            auto trace = interpreter.run(function);
            if (!trace.getString("error").value_or("missing trace").empty()) { return failure(); }
            llvm::json::Array queries;
            for (auto value : values) {
                auto evaluated = interpreter.truth(value);
                if (!evaluated) { return failure(); }
                queries.push_back(*evaluated);
            }
            report["labels"] = std::move(labels);
            report["queries"] = std::move(queries);
            report["trace"] = std::move(trace);
        }
    }
    report["queries_available"] = queriesAvailable;
    llvm::outs() << llvm::json::Value(std::move(report)) << "\n";
    return verify(function);
}
