// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticInsertion.h"
#include "PTO/Transforms/FrontierSynch/CompactAllocation.h"
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Pair = std::pair<std::size_t, std::size_t>;
template<class SelectorSet>
class Preparer {
public:
    Preparer(func::FuncOp function, const ArithmeticProgram& program,
             const SelectorSet& selectors, PreparedLogicalPlan& plan, std::string& error)
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
    const std::map<Pair, int64_t>& recordMap() const { return records; }
    LogicalResult exportRegion()
    {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> namespaces;
        if (records.size() > UINT32_MAX) {
            error = "arithmetic regional record count exceeds representation";
            return failure();
        }
        for (const auto& [pair, record] : records) {
            const auto& source = program.sites[pair.first];
            const auto& target = program.sites[pair.second];
            Operation* from = source.phase->elementOp->getNextNode();
            Operation* to = target.phase->elementOp;
            if (!from || !to) { error = "arithmetic regional endpoint has no original legal cut"; return failure(); }
            const auto p = static_cast<uint32_t>(source.phase->kPipeValue);
            const auto q = static_cast<uint32_t>(target.phase->kPipeValue);
            EndpointFamily family;
            family.id = static_cast<uint32_t>(record);
            family.sourcePipe = p; family.targetPipe = q; family.local = p == q;
            family.sourceCut = {from->getBlock(), from}; family.targetCut = {to->getBlock(), to};
            family.members.push_back({family.id, static_cast<uint32_t>(pair.first),
                                      static_cast<uint32_t>(pair.second), {}, {}});
            plan.families.push_back(std::move(family));
            auto [entry, added] = namespaces.emplace(std::make_pair(p,q), static_cast<uint32_t>(record));
            entry->second = std::min(entry->second, static_cast<uint32_t>(record));
        }
        for (auto& endpoint : plan.endpoints) {
            const auto record = static_cast<uint32_t>(endpoint.record);
            endpoint.records = {record};
            endpoint.piece = &endpoint - plan.endpoints.data();
            if (endpoint.kind != LogicalCommandKind::Barrier) {
                auto& cut = cuts.at(endpoint.before);
                builder.setInsertionPointToEnd(cut.code); activeCut = &cut;
                Value member = emit<arith::ConstantIndexOp>(endpoint.before->getLoc(), record);
                endpoint.memberCoordinates.insert(endpoint.memberCoordinates.begin(), member);
                plan.nestedIdentities |= endpoint.memberCoordinates.size() > 1;
            }
            endpoint.record = namespaces.at({endpoint.sourcePipe, endpoint.targetPipe});
        }
        plan.groupedFamilies = true; plan.independentPieces = true;
        plan.completeInvocation = false;
        return success();
    }
private:
    struct Cut {
        Block* code = nullptr;
        Operation* before = nullptr;
        RegionExpressions::CutEmission replay;
        std::map<std::string, Value> constants;
        llvm::DenseMap<Value, Value> quotients, residues;
        llvm::DenseMap<std::size_t, SmallVector<Operation*>> arithmetic;
    };
    struct Partner { Value guard; SmallVector<Value> sourceCoordinates; };
    func::FuncOp function;
    const ArithmeticProgram& program;
    const SelectorSet& selectors;
    PreparedLogicalPlan& plan;
    std::string& error;
    OpBuilder builder;
    Type wide;
    std::map<Operation*, Cut> cuts;
    std::map<Pair, int64_t> records;
    Cut* activeCut = nullptr;
    RegionExpressions expressions;
    // Intern before returning a Value: endpoint recipes retain raw Value handles,
    // so erasing duplicates after preparing endpoints would invalidate them.
    // All calls below construct pure, region-free arithmetic in this one cut.
    template<class Op, class... Args>
    Value emit(Location loc, Args&&... args)
    {
        Value value = builder.createOrFold<Op>(loc, std::forward<Args>(args)...);
        auto* operation = value.getDefiningOp();
        if (!operation || operation->getBlock() != activeCut->code ||
            operation != &activeCut->code->back()) { return value; }
        const auto hash = OperationEquivalence::computeHash(operation, OperationEquivalence::directHashValue,
            OperationEquivalence::ignoreHashValue, OperationEquivalence::IgnoreLocations);
        auto& bucket = activeCut->arithmetic[static_cast<std::size_t>(hash)];
        for (auto* prior : bucket) {
            if (prior == operation) { return value; }
            if (OperationEquivalence::isEquivalentTo(operation, prior, OperationEquivalence::exactValueMatch,
                    nullptr, OperationEquivalence::IgnoreLocations)) {
                Value shared = prior->getResult(0);
                operation->erase();
                return shared;
            }
        }
        bucket.push_back(operation);
        return value;
    }
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
        auto result = emit<arith::ConstantOp>(loc, wide,
            IntegerAttr::get(wide, negative ? -integer.zextOrTrunc(128) : integer.zextOrTrunc(128)));
        cut.constants.emplace(text, result);
        return result;
    }
    Value number(int64_t value, Cut& cut, Location loc) { return number(BoundInteger(value), cut, loc); }
    Value truth(bool value, Location loc) { return emit<arith::ConstantIntOp>(loc, value, 1); }
    Value both(Value a, Value b, Location loc)
    {
        if (a == b) { return a; }
        APInt constant;
        if (matchPattern(a, m_ConstantInt(&constant))) { return constant.isZero() ? a : b; }
        if (matchPattern(b, m_ConstantInt(&constant))) { return constant.isZero() ? b : a; }
        return emit<arith::AndIOp>(loc, a, b);
    }
    Value either(Value a, Value b, Location loc)
    {
        if (a == b) { return a; }
        APInt constant;
        if (matchPattern(a, m_ConstantInt(&constant))) { return constant.isZero() ? b : a; }
        if (matchPattern(b, m_ConstantInt(&constant))) { return constant.isZero() ? a : b; }
        return emit<arith::OrIOp>(loc, a, b);
    }
    Value compare(arith::CmpIPredicate predicate, Value a, Value b, Location loc)
    {
        return emit<arith::CmpIOp>(loc, predicate, a, b);
    }
    LogicalResult coordinate(Value input, Cut& cut, Location loc)
    {
        if (!input || (!input.getType().isIndex() && !input.getType().isInteger(1))) { return failure(); }
        if (cut.quotients.count(input)) { return success(); }
        auto available = expressions.emitContextual(expressions.input(input), builder, cut.before, cut.replay);
        if (failed(available)) {
            error = expressions.lastEmissionError();
            return failure();
        }
        Value original = input.getType().isIndex() ?
            emit<arith::IndexCastOp>(loc, wide, *available) :
            emit<arith::ExtUIOp>(loc, wide, *available);
        auto period = number(static_cast<int64_t>(selectors.period), cut, loc);
        if (!period) { return failure(); }
        auto quotient = emit<arith::FloorDivSIOp>(loc, original, period);
        auto product = emit<arith::MulIOp>(loc, quotient, period);
        cut.quotients[input] = quotient;
        cut.residues[input] = emit<arith::SubIOp>(loc, original, product);
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
            auto difference = emit<arith::SubIOp>(loc, quotients[atom.lhs], quotients[atom.rhs]);
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
                Value value = emit<arith::AddIOp>(loc, quotients[term.input], offset);
                selected = selected ? emit<arith::MaxSIOp>(loc, selected, value) : value;
            }
            if (!selected) { return failure(); }
            auto period = number(static_cast<int64_t>(selectors.period), cut, loc);
            auto residue = number(static_cast<int64_t>(coordinate.residue), cut, loc);
            if (!period || !residue) { return failure(); }
            auto product = emit<arith::MulIOp>(loc, selected, period);
            auto original = emit<arith::AddIOp>(loc, product, residue);
            // On the piece domain this is an actual source occurrence in the
            // admitted IR execution, hence a representable original index.
            // Off-domain truncation is total and its value is never published.
            values.push_back(emit<arith::IndexCastOp>(loc, builder.getIndexType(), original));
        }
        return values;
    }
    // Each input quotient is a signed 64-bit coordinate. Reserve substantial
    // i128 headroom before emitting any multiplied or summed affine expression.
    FailureOr<Value> affine(const IntegerAffine& expression, ArrayRef<Value> quotients,
                            Cut& cut, Location loc)
    {
        if (expression.coefficients.size() + 1 != quotients.size()) { return failure(); }
        BoundInteger inputMagnitude(1), limit(1);
        for (unsigned i = 0; i < 63; ++i) { inputMagnitude *= 2; }
        for (unsigned i = 0; i < 119; ++i) { limit *= 2; }
        auto magnitude = llvm::abs(expression.constant);
        for (const auto& coefficient : expression.coefficients) {
            magnitude += llvm::abs(coefficient) * inputMagnitude;
        }
        if (magnitude >= limit) {
            error = "arithmetic selector affine expression exceeds the checked i128 emission range";
            return failure();
        }
        Value sum = number(expression.constant, cut, loc);
        if (!sum) { return failure(); }
        for (auto [i, coefficient] : llvm::enumerate(expression.coefficients)) {
            if (coefficient == 0) { continue; }
            auto scalar = number(coefficient, cut, loc);
            if (!scalar) { return failure(); }
            Value term = emit<arith::MulIOp>(loc, quotients[i + 1], scalar);
            sum = emit<arith::AddIOp>(loc, sum, term);
        }
        return sum;
    }
    FailureOr<Value> domain(const GeneralArithmeticSelectorPiece& piece, ArrayRef<Value> original,
                            ArrayRef<Value> quotients, Cut& cut, Location loc)
    {
        if (piece.domain.isEmpty()) { return truth(false, loc); }
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
            auto lhs = affine({atom.coefficients, BoundInteger(0)}, quotients, cut, loc);
            auto bound = number(atom.bound, cut, loc);
            if (failed(lhs) || !bound) { return failure(); }
            guard = both(guard, compare(arith::CmpIPredicate::sle, *lhs, bound, loc), loc);
        }
        for (const auto& atom : piece.domain.congruences()) {
            auto lhs = affine({atom.coefficients, BoundInteger(0)}, quotients, cut, loc);
            auto modulus = number(atom.modulus, cut, loc), residue = number(atom.residue, cut, loc);
            if (failed(lhs) || !modulus || !residue || atom.modulus <= 0) { return failure(); }
            auto quotient = emit<arith::FloorDivSIOp>(loc, *lhs, modulus);
            auto multiple = emit<arith::MulIOp>(loc, quotient, modulus);
            auto remainder = emit<arith::SubIOp>(loc, *lhs, multiple);
            guard = both(guard, compare(arith::CmpIPredicate::eq, remainder, residue, loc), loc);
        }
        return guard;
    }
    FailureOr<SmallVector<Value>> output(const GeneralArithmeticSelectorPiece& piece,
                                         ArrayRef<Value> quotients, Cut& cut, Location loc)
    {
        SmallVector<Value> values;
        for (const auto& coordinate : piece.outputs) {
            auto numerator = affine(coordinate.numerator, quotients, cut, loc);
            auto denominator = number(coordinate.denominator, cut, loc);
            auto period = number(static_cast<int64_t>(selectors.period), cut, loc);
            auto residue = number(static_cast<int64_t>(coordinate.residue), cut, loc);
            if (failed(numerator) || !denominator || !period || !residue || coordinate.denominator <= 0) {
                return failure();
            }
            auto quotient = emit<arith::FloorDivSIOp>(loc, *numerator, denominator);
            auto product = emit<arith::MulIOp>(loc, quotient, period);
            auto original = emit<arith::AddIOp>(loc, product, residue);
            values.push_back(emit<arith::IndexCastOp>(loc, builder.getIndexType(), original));
        }
        return values;
    }
    template<class Selector>
    LogicalResult prepare(const Selector& selector, bool forward)
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
        cut.before = before;
        if (!cut.code) { cut.code = &plan.addPreparation(before); }
        builder.setInsertionPointToEnd(cut.code);
        activeCut = &cut;
        auto loc = before->getLoc();
        SmallVector<Value> original, quotients{number(0, cut, loc)};
        for (auto loop : site.loops) { original.push_back(loop.getInductionVar()); }
        llvm::append_range(original, program.parameters);
        for (auto input : original) {
            if (failed(coordinate(input, cut, loc))) { return failure(); }
            quotients.push_back(cut.quotients.lookup(input));
        }
        std::map<std::size_t, Partner> partners;
        for (const auto& piece : selector.pieces) {
            if (piece.outputSite >= program.sites.size()) { return failure(); }
            auto present = domain(piece, original, quotients, cut, loc);
            if (failed(present)) { return failure(); }
            // Exact minimum demands have one partner per pipe. Overlapping
            // pieces therefore select the same tagged tuple, and need no
            // first-match exclusion circuit.
            auto selected = *present;
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
                        partner.sourceCoordinates[i] = emit<arith::SelectOp>(
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
            if (!local) { identity = emit<arith::ConstantIndexOp>(loc, 0); }
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
    if (!analysis.error.empty() || !analysis.exactMinimum ||
        bits.isScalable() || bits.getFixedValue() != 64 || analysis.period > 2 ||
        analysis.parameterCount != program.parameters.size()) {
        error = "arithmetic insertion requires reduced demands and a supported index representation";
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
    Preparer<ArithmeticSelectors> preparer(function, program, selectors, *plan, error);
    if (failed(preparer.run())) {
        if (error.empty()) { error = "arithmetic selector cannot be emitted at its original cut"; }
        return failure();
    }
    plan->allocationCertificate = arithmeticAllocationCertificate(analysis, pipes, preparer.recordMap(),
                                                                  plan->planId, function.getContext());
    return plan;
}
namespace {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGeneral(
    func::FuncOp function, const ArithmeticProgram& program,
    const GeneralArithmeticDemandAnalysis& analysis, std::string& error, bool regional)
{
    const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
    if (!analysis.error.empty() || !analysis.exactMinimum ||
        bits.isScalable() || bits.getFixedValue() != 64 || analysis.period > 2 ||
        analysis.parameterCount != program.parameters.size()) {
        error = "arithmetic insertion requires reduced demands and a supported index representation";
        return failure();
    }
    SmallVector<uint32_t> pipes;
    for (const auto& site : program.sites) {
        if (!site.phase) { return failure(); }
        pipes.push_back(static_cast<uint32_t>(site.phase->kPipeValue));
    }
    auto selectors = buildGeneralArithmeticSelectors(analysis, pipes);
    if (!selectors.error.empty()) { error = selectors.error; return failure(); }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->completeInvocation = !program.sites.empty();
    Preparer<GeneralArithmeticSelectors> preparer(function, program, selectors, *plan, error);
    if (failed(preparer.run()) || (regional && failed(preparer.exportRegion()))) {
        if (error.empty()) { error = "arithmetic selector cannot be emitted at its original cut"; }
        return failure();
    }
    return plan;
}
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGeneralArithmeticInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const GeneralArithmeticDemandAnalysis& analysis, std::string& error)
{
    return prepareGeneral(function, program, analysis, error, false);
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareGeneralArithmeticRegionalInsertion(
    func::FuncOp function, const ArithmeticProgram& program,
    const GeneralArithmeticDemandAnalysis& analysis, std::string& error)
{
    return prepareGeneral(function, program, analysis, error, true);
}
} // namespace mlir::pto::frontiersynch
