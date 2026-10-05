// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticInsertion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Pair = std::pair<std::size_t, std::size_t>;
class Preparer {
public:
    Preparer(func::FuncOp function, const ArithmeticProgram& program,
             const ArithmeticSelectors& selectors, PreparedLogicalPlan& plan, std::string& error)
        : function(function), program(program), selectors(selectors), plan(plan), error(error),
          builder(function.getContext()), wide(builder.getIntegerType(128)) {}
    LogicalResult run()
    {
        for (const auto& selector : selectors.forward) {
            for (const auto& piece : selector.pieces) {
                records.try_emplace({selector.inputSite, piece.outputSite}, records.size());
            }
        }
        for (const auto& selector : selectors.forward) {
            if (failed(prepare(selector, true))) { return failure(); }
        }
        for (const auto& selector : selectors.inverse) {
            if (failed(prepare(selector, false))) { return failure(); }
        }
        return success();
    }
private:
    struct Cut {
        Block* code = nullptr;
        std::map<std::string, Value> constants;
        llvm::DenseMap<Value, Value> quotients, residues;
    };
    struct Partner { Value guard; SmallVector<Value> sourceCoordinates; };
    func::FuncOp function;
    const ArithmeticProgram& program;
    const ArithmeticSelectors& selectors;
    PreparedLogicalPlan& plan;
    std::string& error;
    OpBuilder builder;
    Type wide;
    std::map<Operation*, Cut> cuts;
    std::map<Pair, int64_t> records;
    Value number(const BoundInteger& value, Cut& cut, Location loc)
    {
        std::string text;
        llvm::raw_string_ostream stream(text);
        stream << value;
        if (auto found = cut.constants.find(text); found != cut.constants.end()) { return found->second; }
        APInt integer;
        StringRef digits(text);
        const bool negative = digits.consume_front("-");
        if (digits.getAsInteger(10, integer) || integer.getActiveBits() > 119) {
            error = "arithmetic selector constant exceeds the checked i128 emission range";
            return {};
        }
        auto result = builder.create<arith::ConstantOp>(loc, wide,
            IntegerAttr::get(wide, negative ? -integer.zextOrTrunc(128) : integer.zextOrTrunc(128))).getResult();
        cut.constants.emplace(text, result);
        return result;
    }
    Value number(int64_t value, Cut& cut, Location loc) { return number(BoundInteger(value), cut, loc); }
    Value truth(bool value, Location loc) { return builder.create<arith::ConstantIntOp>(loc, value, 1); }
    Value both(Value a, Value b, Location loc) { return builder.create<arith::AndIOp>(loc, a, b); }
    Value either(Value a, Value b, Location loc) { return builder.create<arith::OrIOp>(loc, a, b); }
    Value negate(Value a, Location loc) { return builder.create<arith::XOrIOp>(loc, a, truth(true, loc)); }
    Value compare(arith::CmpIPredicate predicate, Value a, Value b, Location loc)
    {
        return builder.create<arith::CmpIOp>(loc, predicate, a, b);
    }
    LogicalResult coordinate(Value input, Cut& cut, Location loc)
    {
        if (!input || (!input.getType().isIndex() && !input.getType().isInteger(1))) { return failure(); }
        if (cut.quotients.count(input)) { return success(); }
        Value original = input.getType().isIndex() ?
            builder.create<arith::IndexCastOp>(loc, wide, input).getResult() :
            builder.create<arith::ExtUIOp>(loc, wide, input).getResult();
        auto period = number(static_cast<int64_t>(selectors.period), cut, loc);
        if (!period) { return failure(); }
        auto quotient = builder.create<arith::FloorDivSIOp>(loc, original, period);
        auto product = builder.create<arith::MulIOp>(loc, quotient, period);
        cut.quotients[input] = quotient;
        cut.residues[input] = builder.create<arith::SubIOp>(loc, original, product);
        return success();
    }
    FailureOr<Value> domain(const ArithmeticSelectorPiece& piece, ArrayRef<Value> original,
                            ArrayRef<Value> quotients, Cut& cut, Location loc)
    {
        Value guard = truth(true, loc);
        SmallVector<uint64_t> residues(piece.inputResidues.begin(), piece.inputResidues.end());
        llvm::append_range(residues, piece.parameterResidues);
        if (residues.size() != original.size() || quotients.size() != original.size() + 1) { return failure(); }
        for (auto [i, residue] : llvm::enumerate(residues)) {
            auto expected = number(static_cast<int64_t>(residue), cut, loc);
            if (!expected) { return failure(); }
            auto equal = compare(arith::CmpIPredicate::eq, cut.residues.lookup(original[i]), expected, loc);
            guard = both(guard, equal, loc);
        }
        for (const auto& atom : piece.domain.constraints()) {
            if (atom.lhs >= quotients.size() || atom.rhs >= quotients.size()) { return failure(); }
            auto bound = number(atom.bound, cut, loc);
            if (!bound) { return failure(); }
            auto difference = builder.create<arith::SubIOp>(loc, quotients[atom.lhs], quotients[atom.rhs]);
            guard = both(guard, compare(arith::CmpIPredicate::sle, difference, bound, loc), loc);
        }
        return guard;
    }
    FailureOr<SmallVector<Value>> output(const ArithmeticSelectorPiece& piece, ArrayRef<Value> quotients,
                                         Cut& cut, Location loc)
    {
        SmallVector<Value> values;
        for (const auto& coordinate : piece.outputs) {
            Value selected;
            for (const auto& term : coordinate.lowerBounds) {
                if (term.input >= quotients.size()) { return failure(); }
                auto offset = number(term.offset, cut, loc);
                if (!offset) { return failure(); }
                Value value = builder.create<arith::AddIOp>(loc, quotients[term.input], offset);
                selected = selected ? builder.create<arith::MaxSIOp>(loc, selected, value).getResult() : value;
            }
            if (!selected) { return failure(); }
            auto period = number(static_cast<int64_t>(selectors.period), cut, loc);
            auto residue = number(static_cast<int64_t>(coordinate.residue), cut, loc);
            if (!period || !residue) { return failure(); }
            auto product = builder.create<arith::MulIOp>(loc, selected, period);
            auto original = builder.create<arith::AddIOp>(loc, product, residue);
            // On the piece domain this is an actual source occurrence in the
            // admitted IR execution, hence a representable original index.
            // Off-domain truncation is total and its value is never published.
            values.push_back(builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), original));
        }
        return values;
    }
    LogicalResult prepare(const ArithmeticEndpointSelector& selector, bool forward)
    {
        if (selector.inputSite >= program.sites.size()) { return failure(); }
        const auto& site = program.sites[selector.inputSite];
        if (!site.phase || selector.inputDimensions != site.loops.size()) { return failure(); }
        const auto pipe = static_cast<uint32_t>(site.phase->kPipeValue);
        const bool local = selector.outputPipe == pipe;
        if (forward && local) { return success(); }
        auto* before = forward ? site.phase->elementOp->getNextNode() : site.phase->elementOp;
        if (!before) { return failure(); }
        auto& cut = cuts[before];
        if (!cut.code) { cut.code = &plan.addPreparation(before); }
        builder.setInsertionPointToEnd(cut.code);
        auto loc = before->getLoc();
        SmallVector<Value> original, quotients{number(0, cut, loc)};
        for (auto loop : site.loops) { original.push_back(loop.getInductionVar()); }
        llvm::append_range(original, program.parameters);
        for (auto input : original) {
            if (failed(coordinate(input, cut, loc))) { return failure(); }
            quotients.push_back(cut.quotients.lookup(input));
        }
        std::map<std::size_t, Partner> partners;
        Value seen = truth(false, loc);
        for (const auto& piece : selector.pieces) {
            if (piece.outputSite >= program.sites.size()) { return failure(); }
            auto present = domain(piece, original, quotients, cut, loc);
            if (failed(present)) { return failure(); }
            auto selected = both(*present, negate(seen, loc), loc);
            seen = either(seen, *present, loc);
            auto& partner = partners[piece.outputSite];
            if (!partner.guard) { partner.guard = truth(false, loc); }
            partner.guard = either(partner.guard, selected, loc);
            if (!forward && !local) {
                auto tuple = output(piece, quotients, cut, loc);
                if (failed(tuple)) { return failure(); }
                if (partner.sourceCoordinates.empty()) { partner.sourceCoordinates = *tuple; }
                else {
                    if (partner.sourceCoordinates.size() != tuple->size()) { return failure(); }
                    for (unsigned i = 0; i < tuple->size(); ++i) {
                        partner.sourceCoordinates[i] = builder.create<arith::SelectOp>(
                            loc, selected, (*tuple)[i], partner.sourceCoordinates[i]);
                    }
                }
            }
        }
        for (auto& [other, partner] : partners) {
            const Pair pair = forward ? Pair{selector.inputSite, other} : Pair{other, selector.inputSite};
            auto record = records.find(pair);
            if (record == records.end()) { return failure(); }
            auto kind = local ? LogicalCommandKind::Barrier :
                        forward ? LogicalCommandKind::Set : LogicalCommandKind::Wait;
            Value identity;
            if (!local) { identity = builder.create<arith::ConstantIndexOp>(loc, 0); }
            PreparedLogicalEndpoint endpoint{before, kind, forward ? pipe : selector.outputPipe,
                forward ? selector.outputPipe : pipe, record->second, partner.guard, identity};
            if (!local) {
                if (forward) {
                    for (auto loop : site.loops) { endpoint.memberCoordinates.push_back(loop.getInductionVar()); }
                } else { endpoint.memberCoordinates = std::move(partner.sourceCoordinates); }
            }
            plan.endpoints.push_back(std::move(endpoint));
        }
        return success();
    }
};
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareArithmeticInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const ArithmeticDemandAnalysis& analysis, std::string& error)
{
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (!analysis.error.empty() || !analysis.exactMinimum || !analysis.adjacentLocalDemands ||
        bits.isScalable() || bits.getFixedValue() != 64 || analysis.period > 2 ||
        analysis.parameterCount != program.parameters.size()) {
        error = "arithmetic insertion requires exact adjacent demands and a supported index representation";
        return failure();
    }
    SmallVector<uint32_t> pipes;
    for (const auto& site : program.sites) {
        if (!site.phase) { return failure(); }
        pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue));
    }
    auto selectors = buildArithmeticSelectors(analysis, pipes);
    if (!selectors.error.empty()) { error = selectors.error; return failure(); }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->completeInvocation = !program.sites.empty();
    if (failed(Preparer(function, program, selectors, *plan, error).run())) {
        if (error.empty()) { error = "arithmetic selector cannot be emitted at its original cut"; }
        return failure();
    }
    return plan;
}
} // namespace mlir::pto::frontiersynch
