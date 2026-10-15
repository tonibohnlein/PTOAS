// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent scalar execution of transformed structured counter IR.
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalExecutedCounters.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticInsertion.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticHandoffAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
constexpr const char* program = R"mlir(
module attributes {pto.target_arch = "a3"} {
  func.func @counter(%n: index) -> index {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %three = arith.constant 3 : index
    %sum = scf.for %outer = %zero to %two step %one iter_args(%carry = %zero) -> index {
      %inner = scf.for %i = %zero to %n step %one iter_args(%acc = %carry) -> index {
        %residue = arith.remui %i, %three : index
        %active = arith.cmpi ne, %residue, %one : index
        %value = scf.if %active -> index {
          pto.logical_set [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 0 ordinal %zero members(%outer, %i)
          scf.yield %i : index
        } else {
          scf.yield %zero : index
        }
        scf.if %active {
          pto.logical_wait [<PIPE_MTE2>, <PIPE_MTE1>] plan 0 record 0 ordinal %zero members(%outer, %i)
        }
        %next = arith.addi %acc, %value : index
        scf.yield %next : index
      }
      scf.yield %inner : index
    }
    return %sum : index
  }
}
)mlir";
using Trace = std::vector<std::pair<bool, uint64_t>>;
using Values = llvm::DenseMap<Value, uint64_t>;
FailureOr<SmallVector<uint64_t>> execute(Block& body, Values& values, Trace& trace)
{
    auto read = [&](Value value) { return values.lookup(value); };
    for (auto& operation : body) {
        auto* op = &operation;
        if (isa<scf::YieldOp, func::ReturnOp>(op)) {
            SmallVector<uint64_t> result;
            for (auto operand : op->getOperands()) { result.push_back(read(operand)); }
            return result;
        }
        if (auto loop = dyn_cast<scf::ForOp>(op)) {
            SmallVector<uint64_t> state;
            for (auto initial : loop.getInitArgs()) { state.push_back(read(initial)); }
            for (auto i = read(loop.getLowerBound()); i < read(loop.getUpperBound()); i += read(loop.getStep())) {
                values[loop.getInductionVar()] = i;
                for (auto [argument, value] : llvm::zip(loop.getRegionIterArgs(), state)) { values[argument] = value; }
                auto next = execute(*loop.getBody(), values, trace);
                if (failed(next)) { return failure(); }
                state = std::move(*next);
            }
            for (auto [result, value] : llvm::zip(loop.getResults(), state)) { values[result] = value; }
            continue;
        }
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
            Block* selected = read(branch.getCondition()) ? branch.thenBlock() : branch.elseBlock();
            if (!selected) { continue; }
            auto result = execute(*selected, values, trace);
            if (failed(result)) { return failure(); }
            for (auto [output, value] : llvm::zip(branch.getResults(), *result)) { values[output] = value; }
            continue;
        }
        if (isa<pto::LogicalSetOp, pto::LogicalWaitOp>(op)) {
            trace.emplace_back(isa<pto::LogicalSetOp>(op), read(op->getOperands().back())); continue;
        }
        if (isa<pto::SetFlagDynOp, pto::WaitFlagDynOp>(op)) {
            trace.emplace_back(isa<pto::SetFlagDynOp>(op), read(op->getOperand(0))); continue;
        }
        APInt constant;
        uint64_t result = 0;
        if (matchPattern(op, m_ConstantInt(&constant))) { result = constant.getSExtValue(); }
        else if (isa<arith::AddIOp>(op)) { result = read(op->getOperand(0)) + read(op->getOperand(1)); }
        else if (isa<arith::RemUIOp>(op)) { result = read(op->getOperand(0)) % read(op->getOperand(1)); }
        else if (isa<arith::SubIOp>(op)) { result = read(op->getOperand(0)) - read(op->getOperand(1)); }
        else if (isa<arith::MulIOp>(op)) { result = read(op->getOperand(0)) * read(op->getOperand(1)); }
        else if (isa<arith::FloorDivSIOp>(op)) {
            auto a = static_cast<int64_t>(read(op->getOperand(0)));
            auto b = static_cast<int64_t>(read(op->getOperand(1)));
            result = a / b - (a % b < 0 ? 1 : 0);
        } else if (isa<arith::IndexCastOp, arith::ExtUIOp>(op)) { result = read(op->getOperand(0)); }
        else if (isa<arith::AndIOp>(op)) { result = read(op->getOperand(0)) & read(op->getOperand(1)); }
        else if (isa<arith::OrIOp>(op)) { result = read(op->getOperand(0)) | read(op->getOperand(1)); }
        else if (auto select = dyn_cast<arith::SelectOp>(op)) {
            result = read(select.getCondition()) ? read(select.getTrueValue()) : read(select.getFalseValue());
        } else if (auto comparison = dyn_cast<arith::CmpIOp>(op)) {
            auto a = read(comparison.getLhs()), b = read(comparison.getRhs());
            if (comparison.getPredicate() == arith::CmpIPredicate::eq) { result = a == b; }
            else if (comparison.getPredicate() == arith::CmpIPredicate::ne) { result = a != b; }
            else if (comparison.getPredicate() == arith::CmpIPredicate::sle) {
                result = static_cast<int64_t>(a) <= static_cast<int64_t>(b);
            } else { return failure(); }
        } else { return failure(); }
        if (op->getNumResults() != 1) { return failure(); }
        values[op->getResult(0)] = result;
    }
    return failure();
}
DictionaryAttr certificate(MLIRContext* context, unsigned width)
{
    Builder b(context);
    auto family = b.getDictionaryAttr({b.getNamedAttr("record", b.getI64IntegerAttr(0)),
        b.getNamedAttr("source", b.getI64IntegerAttr(static_cast<int64_t>(pto::PIPE::PIPE_MTE2))),
        b.getNamedAttr("target", b.getI64IntegerAttr(static_cast<int64_t>(pto::PIPE::PIPE_MTE1))),
        b.getNamedAttr("width", b.getI64IntegerAttr(width))});
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
        b.getNamedAttr("plan", b.getI64IntegerAttr(0)), b.getNamedAttr("scope", b.getStringAttr("function")),
        b.getNamedAttr("strategy", b.getStringAttr("executed-family-counters")),
        b.getNamedAttr("families", b.getArrayAttr({family})),
        b.getNamedAttr("total_budget", b.getI64IntegerAttr(width))});
}
std::string printed(func::FuncOp function)
{
    std::string result; llvm::raw_string_ostream stream(result); function.print(stream); return result;
}
bool periodThree(MLIRContext& context)
{
    constexpr const char* input = R"mlir(
      module {
        func.func @period3() -> index {
          %zero = arith.constant 0 : index
          %one = arith.constant 1 : index
          %eleven = arith.constant 11 : index
          %sum = scf.for %i = %zero to %eleven step %one iter_args(%acc = %zero) -> index {
            %source = arith.addi %i, %zero : index
            %target = arith.addi %acc, %source : index
            scf.yield %target : index
          }
          return %sum : index
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(input, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("period3");
    scf::ForOp loop; function.walk([&](scf::ForOp found) { loop = found; });
    auto* source = &loop.getBody()->front(); auto* target = source->getNextNode();
    // Supplied certified sites intentionally bypass the bounded recognizer;
    // this tests the generic arithmetic emitter's residue interface directly.
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_MTE2, source->getName());
    pto::CompoundInstanceElement b(1, {}, {}, pto::PipelineType::PIPE_MTE1, target->getName());
    a.elementOp = source; b.elementOp = target;
    fs::ArithmeticProgram program; program.context = {function, function.getOperation()};
    program.sites.push_back({&a, {loop}, {}}); program.sites.push_back({&b, {loop}, {}});
    fs::GeneralArithmeticDemandAnalysis analysis; analysis.exactMinimum = true; analysis.period = 3;
    for (uint64_t residue : {0U, 2U}) {
        fs::ArithmeticRelationKey key{{0, fs::ArithmeticEvent::Completion, {residue}},
                                     {1, fs::ArithmeticEvent::Start, {residue}}, {}};
        auto relation = fs::IntegerSystem::create(2,
            {{{fs::BoundInteger(1), fs::BoundInteger(-1)}, fs::BoundInteger(0)},
             {{fs::BoundInteger(-1), fs::BoundInteger(1)}, fs::BoundInteger(0)},
             {{fs::BoundInteger(-1), fs::BoundInteger(0)}, fs::BoundInteger(0)},
             {{fs::BoundInteger(1), fs::BoundInteger(0)}, fs::BoundInteger((10 - residue) / 3)}});
        if (failed(relation)) { return false; }
        analysis.minimumDemands[key] = {*relation};
    }
    std::string error;
    auto prepared = fs::prepareGeneralArithmeticRegionalInsertion(function, program, analysis, error);
    if (failed(prepared)) { return false; }
    if (failed(fs::insertLogicalSynchronization(function, **prepared))) { return false; }
    Values values; Trace actual, expected;
    for (uint64_t i = 0; i < 11; ++i) {
        if (i % 3 != 1) { expected.emplace_back(true, i); expected.emplace_back(false, i); }
    }
    auto result = execute(function.front(), values, actual);
    return succeeded(result) && result->size() == 1 && result->front() == 55 && actual == expected;
}
bool check(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(program, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("counter");
    const auto original = printed(function);
    {
        ScopedDiagnosticHandler diagnostics(&context, [](Diagnostic&) { return success(); });
        if (succeeded(fs::allocateExecutedFamilyCounters(function, certificate(&context, 2), {3})) ||
            printed(function) != original) { return false; }
    }
    if (failed(fs::allocateExecutedFamilyCounters(function, certificate(&context, 2), {1, 3}))) { return false; }
    for (uint64_t trips = 0; trips <= 12; ++trips) {
        Values values; values[function.getArgument(0)] = trips; Trace actual, expected;
        uint64_t sum = 0, count = 0;
        for (unsigned outer = 0; outer < 2; ++outer) {
            for (uint64_t i = 0; i < trips; ++i) {
                if (i % 3 == 1) { continue; }
                auto id = count++ % 2 ? 3 : 1;
                expected.emplace_back(true, id); expected.emplace_back(false, id); sum += i;
            }
        }
        auto result = execute(function.front(), values, actual);
        if (failed(result) || result->size() != 1 || result->front() != sum || actual != expected) { return false; }
    }
    return true;
}
bool coordinateModulo(MLIRContext& context)
{
    constexpr const char* input = R"mlir(
      module attributes {pto.target_arch = "a3"} {
        func.func @delayed() -> index {
          %zero = arith.constant 0 : index
          %three = arith.constant 3 : index
          %two = arith.constant 2 : index
          %sixteen = arith.constant 16 : index
          %sum = scf.for %i = %three to %sixteen step %two iter_args(%acc = %zero) -> index {
            %source = arith.addi %i, %zero : index
            %target = arith.addi %acc, %source : index
            scf.yield %target : index
          }
          return %sum : index
        }
      }
    )mlir";
    auto module = parseSourceString<ModuleOp>(input, &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("delayed");
    scf::ForOp loop; function.walk([&](scf::ForOp found) { loop = found; });
    auto* source = &loop.getBody()->front(); auto* target = source->getNextNode();
    pto::CompoundInstanceElement a(0, {}, {}, pto::PipelineType::PIPE_MTE2, source->getName());
    pto::CompoundInstanceElement b(1, {}, {}, pto::PipelineType::PIPE_MTE1, target->getName());
    a.elementOp = source; b.elementOp = target;
    fs::ArithmeticProgram program; program.context = {function, function.getOperation()};
    program.sites.push_back({&a, {loop}, {}}); program.sites.push_back({&b, {loop}, {}});
    fs::GeneralArithmeticDemandAnalysis analysis; analysis.exactMinimum = true; analysis.period = 2;
    auto row = [](int64_t x, int64_t y, int64_t bound) {
        return fs::IntegerConstraint{{fs::BoundInteger(x), fs::BoundInteger(y)}, fs::BoundInteger(bound)};
    };
    // Sources 3,5,...,13 match consumers 5,7,...,15. A later source can reuse
    // only after that consumer, so same-iteration textual SET/WAIT is insufficient.
    auto demands = fs::IntegerSystem::create(2, {row(1, -1, -1), row(-1, 1, 1),
        row(-1, 0, -1), row(1, 0, 6)});
    auto required = fs::IntegerSystem::create(2, {row(1, -1, -1)});
    if (failed(demands) || failed(required)) { return false; }
    analysis.minimumDemands[{{0, fs::ArithmeticEvent::Completion, {1}},
        {1, fs::ArithmeticEvent::Start, {1}}, {}}] = {*demands};
    analysis.requiredOrder[{{1, fs::ArithmeticEvent::Completion, {1}},
        {0, fs::ArithmeticEvent::Start, {1}}, {}}] = {*required};
    const auto p = static_cast<uint32_t>(pto::PIPE::PIPE_MTE2), q = static_cast<uint32_t>(pto::PIPE::PIPE_MTE1);
    auto proof = fs::buildArithmeticCoordinateHandoffAllocation(analysis, {p, q});
    if (!proof.error.empty() || proof.families.size() != 1 || proof.families[0].lifetime.width != 3 ||
        proof.families[0].sourceCoordinate != std::optional<unsigned>(0)) { return false; }
    std::string error;
    auto prepared = fs::prepareGeneralArithmeticRegionalInsertion(function, program, analysis, error);
    if (failed(prepared)) { return false; }
    // Serialize the proved assignment into the ordinary regional consumer.
    Builder builder(&context);
    auto tuple = builder.getDictionaryAttr({builder.getNamedAttr("coordinate_count", builder.getI64IntegerAttr(2)),
        builder.getNamedAttr("base", builder.getI64IntegerAttr(0)),
        builder.getNamedAttr("terms", builder.getArrayAttr({builder.getDenseI64ArrayAttr({1, 1, 0, 3, 1})}))});
    auto group = builder.getDictionaryAttr({builder.getNamedAttr("source", builder.getI64IntegerAttr(p)),
        builder.getNamedAttr("target", builder.getI64IntegerAttr(q)),
        builder.getNamedAttr("budget", builder.getI64IntegerAttr(3)),
        builder.getNamedAttr("records", builder.getDenseI64ArrayAttr({0})),
        builder.getNamedAttr("strides", builder.getDenseI64ArrayAttr({0})),
        builder.getNamedAttr("phases", builder.getDenseI64ArrayAttr({0})),
        builder.getNamedAttr("conflicts", builder.getDenseI64ArrayAttr({})),
        builder.getNamedAttr("tuple_rules", builder.getArrayAttr({tuple}))});
    (*prepared)->allocationCertificate = builder.getDictionaryAttr({
        builder.getNamedAttr("version", builder.getI64IntegerAttr(2)),
        builder.getNamedAttr("kind", builder.getStringAttr("finite")),
        builder.getNamedAttr("plan", builder.getI64IntegerAttr(0)),
        builder.getNamedAttr("strategy", builder.getStringAttr("regional-palettes")),
        builder.getNamedAttr("macro_reservations", builder.getUnitAttr()),
        builder.getNamedAttr("groups", builder.getArrayAttr({group}))});
    if (failed(fs::insertLogicalSynchronization(function, **prepared)) ||
        failed(fs::allocatePhysicalEventIds(function, {0, 2, 5}))) { return false; }
    Values values; Trace actual, expected;
    const uint64_t palette[] = {0, 2, 5};
    for (uint64_t i = 3; i < 16; i += 2) {
        if (i <= 13) { expected.emplace_back(true, palette[i % 3]); }
        if (i >= 5) { expected.emplace_back(false, palette[(i - 2) % 3]); }
    }
    auto result = execute(function.front(), values, actual);
    // Independently check that every same-ID reuse waits for the previous
    // consumer iteration; a SET followed textually by WAIT does not suffice.
    for (uint64_t first = 3; first <= 13; first += 2) {
        for (uint64_t next = first + 2; next <= 13; next += 2) {
            if (first % 3 == next % 3 && first + 2 >= next) { return false; }
        }
    }
    return succeeded(result) && actual == expected;
}
} // namespace
int runPhysicalExecutedCounterChecks()
{
    MLIRContext context;
    context.loadDialect<pto::PTODialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect>();
    if (!check(context) || !periodThree(context) || !coordinateModulo(context)) {
        llvm::errs() << "executed arithmetic counter checks failed\n"; return 1;
    }
    llvm::outs() << "executed arithmetic counters: guards, nested loops, results, zero trips and rollback passed\n";
    return 0;
}
