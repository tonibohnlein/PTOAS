// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Compare emitted signed DBM witnesses against the independent arbitrary-precision selector oracle.
#ifndef PTO_DIFFERENCE_ARITHMETIC_RECIPE_CHECKS_H
#define PTO_DIFFERENCE_ARITHMETIC_RECIPE_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticEndpointExpressions.h"
namespace {
using WideValues = llvm::DenseMap<Value, APInt>;
bool executeDifferenceRecipe(Block& block, WideValues& values)
{
    for (auto& op : block) {
        SmallVector<APInt> operands;
        for (Value input : op.getOperands()) {
            auto found = values.find(input);
            if (found == values.end()) { return false; }
            operands.push_back(found->second);
        }
        APInt result;
        if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
            result = cast<IntegerAttr>(constant.getValue()).getValue();
        } else if (isa<arith::IndexCastOp>(op)) {
            auto type = op.getResult(0).getType();
            result = operands[0].sextOrTrunc(type.isIndex() ? 64 : cast<IntegerType>(type).getWidth());
        } else if (auto extend = dyn_cast<arith::ExtUIOp>(op)) {
            result = operands[0].zext(cast<IntegerType>(extend.getType()).getWidth());
        } else if (isa<arith::FloorDivSIOp>(op)) {
            if (operands[1].isZero() || operands[1].isNegative()) { return false; }
            result = operands[0].sdiv(operands[1]);
            if (operands[0].isNegative() && !operands[0].srem(operands[1]).isZero()) { --result; }
        } else if (isa<arith::AddIOp>(op)) { result = operands[0] + operands[1]; }
        else if (isa<arith::SubIOp>(op)) { result = operands[0] - operands[1]; }
        else if (isa<arith::MulIOp>(op)) { result = operands[0] * operands[1]; }
        else if (isa<arith::AndIOp>(op)) { result = operands[0] & operands[1]; }
        else if (isa<arith::OrIOp>(op)) { result = operands[0] | operands[1]; }
        else if (isa<arith::SelectOp>(op)) { result = operands[0].isZero() ? operands[2] : operands[1]; }
        else if (auto comparison = dyn_cast<arith::CmpIOp>(op)) {
            bool value;
            if (comparison.getPredicate() == arith::CmpIPredicate::eq) { value = operands[0] == operands[1]; }
            else if (comparison.getPredicate() == arith::CmpIPredicate::sle) { value = operands[0].sle(operands[1]); }
            else { return false; }
            result = APInt(1, value);
        } else { return false; }
        if (op.getNumResults() != 1) { return false; }
        values[op.getResult(0)] = std::move(result);
    }
    return true;
}
bool differenceArithmeticRecipeChecks(MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(
        module { func.func @recipe(%coordinate: index, %parameter: i1) { return } }
    )mlir", &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("recipe");
    auto* cut = function.front().getTerminator();
    using Integer = fs::BoundInteger;
    Integer large(1);
    for (unsigned i = 0; i < 118; ++i) { large *= Integer(2); }
    const int64_t samples[] = {INT64_MIN, INT64_MIN + 1, -9, -2, -1, 0, 1, 7, INT64_MAX - 1, INT64_MAX};
    for (uint64_t period : {uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(INT64_MAX)}) {
        for (int64_t sample : samples) {
            for (int64_t parameter : {0, 1}) {
                Integer quotient = Integer(sample) / Integer(static_cast<int64_t>(period));
                if (sample < 0 && Integer(sample) % Integer(static_cast<int64_t>(period)) != 0) { --quotient; }
                const auto residue = static_cast<uint64_t>(static_cast<int64_t>(Integer(sample) -
                    quotient * Integer(static_cast<int64_t>(period))));
                for (bool empty : {false, true}) {
                    fs::ArithmeticEndpointSelector selector;
                    selector.inputDimensions = 1; selector.parameterCount = 1;
                    fs::ArithmeticSelectorPiece piece;
                    SmallVector<fs::DifferenceBoundConstraint> atoms;
                    if (empty) { atoms.push_back({0, 0, Integer(-1)}); }
                    auto domain = fs::DifferenceBoundSystem::create(2, atoms);
                    if (failed(domain)) { return false; }
                    piece.domain = *domain; piece.inputResidues = {residue};
                    piece.parameterResidues = {static_cast<uint64_t>(parameter) % period};
                    // The first and fourth terms tie at the true maximum. The
                    // huge negative variable/constant terms would truncate to
                    // incorrect large values without pre-maximum saturation.
                    piece.outputs = {{{{1, Integer(0)}, {1, -large}, {0, -large},
                        {1, Integer(0)}, {2, -large}}, residue}};
                    selector.pieces = {piece};
                    auto oracle = fs::evaluateArithmeticSelector(selector, period,
                        {Integer(sample)}, {Integer(parameter)});
                    if (failed(oracle) || bool(*oracle) == empty) { return false; }
                    fs::RegionExpressions arena;
                    SmallVector<fs::RegionExpressions::Id> inputs{
                        arena.input(function.getArgument(0)), arena.input(function.getArgument(1))};
                    SmallVector<uint64_t> residues{residue, static_cast<uint64_t>(parameter) % period};
                    std::string error;
                    auto activity = fs::detail::endpointPredicate(arena, piece.domain, inputs, period, residues, error);
                    auto coordinate = fs::detail::endpointCoordinate(arena, piece.outputs[0],
                        inputs, period, residues, error);
                    if (!error.empty() || activity == fs::RegionExpressions::invalid ||
                        coordinate == fs::RegionExpressions::invalid) { return false; }
                    Block preparation;
                    OpBuilder builder(&context); builder.setInsertionPointToEnd(&preparation);
                    llvm::DenseMap<fs::RegionExpressions::Id, Value> memo;
                    auto guard = arena.emit(activity, builder, cut, memo);
                    auto witness = arena.emit(coordinate, builder, cut, memo);
                    if (failed(guard) || failed(witness)) { return false; }
                    WideValues values;
                    values[function.getArgument(0)] = APInt(64, static_cast<uint64_t>(sample));
                    values[function.getArgument(1)] = APInt(1, parameter);
                    if (!executeDifferenceRecipe(preparation, values) ||
                        !values.count(*guard) || !values.count(*witness) ||
                        (!values.lookup(*guard).isZero()) != bool(*oracle)) { return false; }
                    if (*oracle && Integer(values.lookup(*witness).getSExtValue()) != (**oracle).coordinates[0]) {
                        return false;
                    }
                }
            }
        }
    }
    fs::RegionExpressions arena;
    const auto input = arena.input(function.getArgument(0));
    std::string error;
    fs::ArithmeticSelectorOutput absent;
    if (fs::detail::endpointCoordinate(arena, absent, {input}, 1, {0}, error) !=
        fs::RegionExpressions::invalid) { return false; }
    fs::ArithmeticSelectorOutput tooWide{{{1, large * Integer(2)}}, 0};
    if (fs::detail::endpointCoordinate(arena, tooWide, {input}, 1, {0}, error) !=
        fs::RegionExpressions::invalid || error.empty()) { return false; }
    auto tooWideDomain = fs::DifferenceBoundSystem::create(1, {{1, 0, large * Integer(2)}});
    error.clear();
    return succeeded(tooWideDomain) && fs::detail::endpointPredicate(arena, *tooWideDomain,
        {input}, 1, {0}, error) == fs::RegionExpressions::invalid && !error.empty();
}
bool differenceCertifiedAllocation(MLIRContext& context, unsigned route)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(
        module {
          func.func @dedicated(%x: index) -> index {
            %source0 = arith.addi %x, %x : index
            %target0 = arith.addi %source0, %x : index
            %source1 = arith.addi %target0, %x : index
            %target1 = arith.addi %source1, %x : index
            return %target1 : index
          }
        }
    )mlir", &context);
    if (!module) { return false; }
    auto function = module->lookupSymbol<func::FuncOp>("dedicated");
    fs::ArithmeticProgram program; program.context = {function, function.getOperation()};
    std::vector<std::unique_ptr<pto::CompoundInstanceElement>> phases;
    unsigned site = 0;
    for (auto& op : function.front()) {
        if (isa<func::ReturnOp>(op)) { continue; }
        phases.push_back(std::make_unique<pto::CompoundInstanceElement>(site,
            SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
            site % 2 ? pto::PipelineType::PIPE_MTE1 : pto::PipelineType::PIPE_MTE2, op.getName()));
        phases.back()->elementOp = &op;
        program.sites.push_back({phases.back().get(), {}, {}});
        ++site;
    }
    fs::ArithmeticDemandAnalysis analysis; analysis.exactMinimum = true; analysis.period = 1;
    for (unsigned source : {0U, 2U}) {
        analysis.minimumDemands.edit()[{{source, fs::ArithmeticEvent::Completion, {}},
            {source + 1, fs::ArithmeticEvent::Start, {}}, {}}] = {fs::DifferenceBoundSystem{}};
    }
    std::string error;
    auto plan = fs::prepareArithmeticInsertion(function, program, analysis, error);
    if (failed(plan) || !(*plan)->familyRecipe || !(*plan)->preparation.empty() ||
        !(*plan)->allocationCertificate) { return false; }
    auto strategy = (*plan)->allocationCertificate.getAs<StringAttr>("strategy");
    if (!strategy || strategy.getValue() != "dedicated-families") { return false; }
    if (route == 1 && failed(fs::planPhysicalFamilyOutput(function, **plan, {2, 5}))) { return false; }
    (*plan)->deferredOutput = route == 2;
    if (failed(fs::insertLogicalSynchronization(function, **plan))) { return false; }
    OwningOpRef<ModuleOp> replay;
    if (route == 2) {
        std::string serialized;
        llvm::raw_string_ostream stream(serialized); module->print(stream); stream.flush();
        (*plan).reset();
        replay = parseSourceString<ModuleOp>(serialized, &context);
        if (!replay) { return false; }
        function = replay->lookupSymbol<func::FuncOp>("dedicated");
    }
    if (route != 1 && failed(fs::allocatePhysicalEventIds(function, {2, 5}))) { return false; }
    Values values; values[function.getArgument(0)] = 3;
    Trace actual;
    auto returned = execute(function.front(), values, actual);
    return succeeded(verify(function)) && succeeded(returned) && *returned == SmallVector<uint64_t>{15} &&
        actual == Trace{{true, 2}, {false, 2}, {true, 5}, {false, 5}};
}
} // namespace
#endif
