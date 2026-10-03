// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Compile functional Presburger endpoint pieces into arithmetic and guards only.
#include "GeneralQueries.h"
#include "PTO/IR/PTO.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
using Int = llvm::DynamicAPInt;
using R = presburger::PresburgerRelation;
using I = presburger::IntegerRelation;
using Tuple = SymbolicTuple;
std::string decimal(const Int& value)
{
    std::string text;
    llvm::raw_string_ostream stream(text); stream << value; return text;
}
FailureOr<std::size_t> constantTag(ArrayRef<Int> row, std::size_t count)
{
    if (row.empty() || llvm::any_of(row.drop_back(), [](const Int& value) { return value != Int(0); }) ||
        row.back() < Int(0) || row.back() >= Int(static_cast<int64_t>(count))) { return failure(); }
    std::size_t result = 0;
    if (StringRef(decimal(row.back())).getAsInteger(10, result)) { return failure(); }
    return result;
}
FailureOr<R> pipeRelation(const R& input, SymbolicSchemaHandle schema,
                                 PipelineType source, PipelineType target, unsigned s, unsigned t)
{
    R result = R::getEmpty(input.getSpace());
    unsigned width = input.getNumDomainVars(), kind = schema->coordinateDepth() + 1;
    // Shared primitives and composed queries preserve literal original tags.
    // Filter tagged pieces before relation operations; never expand every pair.
    for (auto poly : input.getAllDisjuncts()) {
        auto a = poly.getConstantBound64(presburger::BoundType::EQ, 0);
        auto b = poly.getConstantBound64(presburger::BoundType::EQ, width);
        if (!a || !b || *a < 0 || *b < 0 || static_cast<std::size_t>(*a) >= schema->sites().size() ||
            static_cast<std::size_t>(*b) >= schema->sites().size()) { return failure(); }
        if (schema->sites()[*a].phase->kPipeValue != source ||
            schema->sites()[*b].phase->kPipeValue != target) { continue; }
        poly.addBound(presburger::BoundType::EQ, kind, Int(s));
        poly.addBound(presburger::BoundType::EQ, width + kind, Int(t));
        result.unionInPlace(poly);
    }
    return result;
}
presburger::SymbolicLexOpt lexopt(const R& relation, bool maximum)
{
    presburger::SymbolicLexOpt result(relation.getSpace());
    for (auto poly : relation.getAllDisjuncts()) {
        if (maximum) {
            // LLVM19's IntegerRelation lexmax reconstructs a MultiAffineFunction
            // without its DivisionRepr. Negate the range, use the supported
            // lexmin backend, then reflect its full owned division metadata.
            for (unsigned j = poly.getNumDomainVars();
                 j < poly.getNumDomainVars() + poly.getNumRangeVars(); ++j) {
                for (unsigned i = 0; i < poly.getNumEqualities(); ++i) { poly.atEq(i, j) = -poly.atEq(i, j); }
                for (unsigned i = 0; i < poly.getNumInequalities(); ++i) { poly.atIneq(i, j) = -poly.atIneq(i, j); }
            }
        }
        auto piece = poly.findSymbolicIntegerLexMin();
        result.lexopt = result.lexopt.unionLexMin(piece.lexopt);
        // Any unbounded disjunct defeats this endpoint's functionality proof,
        // even if another disjunct has a lexically earlier bounded optimum.
        // LLVM19's wrapper incorrectly intersects with an initially empty set.
        result.unboundedDomain = result.unboundedDomain.unionSet(piece.unboundedDomain);
    }
    if (maximum) {
        presburger::PWMAFunction reflected(result.lexopt.getSpace());
        for (const auto& piece : result.lexopt.getAllPieces()) {
            auto matrix = piece.output.getOutputMatrix();
            for (unsigned row = 0; row < matrix.getNumRows(); ++row) { matrix.negateRow(row); }
            presburger::MultiAffineFunction output(piece.output.getSpace(), matrix, piece.output.getDivs());
            reflected.addPiece({piece.domain, std::move(output)});
        }
        result.lexopt = std::move(reflected);
    }
    return result;
}
bool rowWidth(ArrayRef<Int> row, ArrayRef<unsigned> variables, unsigned& width)
{
    if (row.size() != variables.size() + 1) { return false; }
    std::uint64_t maximum = 0;
    for (auto [i, coefficient] : llvm::enumerate(row)) {
        if (coefficient == Int(0)) { continue; }
        std::uint64_t bits = 4 * decimal(coefficient).size() + 2;
        if (i < variables.size()) { bits += variables[i]; }
        maximum = std::max(maximum, bits);
    }
    // Sum magnitudes conservatively, without relying on decimal-to-bit rounding.
    std::uint64_t count = row.size();
    while (count > 1) { ++maximum; count = (count + 1) / 2; }
    maximum += 2;
    if (maximum > IntegerType::kMaxWidth) { return false; }
    width = static_cast<unsigned>(maximum); return true;
}
bool divisionWidths(const presburger::DivisionRepr& divisions, SmallVector<unsigned>& variables, unsigned& maximum)
{
    auto base = variables.size();
    if (divisions.getDivOffset() != base) { return false; }
    variables.resize(divisions.getNumVars(), 0);
    for (unsigned i = 0; i < divisions.getNumDivs(); ++i) {
        if (!divisions.hasRepr(i) || divisions.getDenom(i) <= Int(0)) { return false; }
        auto dividend = divisions.getDividend(i);
        // Emit only acyclic, represented floor divisions. Existential locals
        // and forward/cyclic dependence need a different executable adapter.
        for (unsigned j = base + i; j < divisions.getNumVars(); ++j) {
            if (dividend[j] != Int(0)) { return false; }
        }
        unsigned width = 0;
        if (!rowWidth(dividend, variables, width)) { return false; }
        std::uint64_t denominatorBits = 4 * decimal(divisions.getDenom(i)).size() + 2;
        if (denominatorBits > IntegerType::kMaxWidth) { return false; }
        maximum = std::max({maximum, width, static_cast<unsigned>(denominatorBits)});
        variables[base + i] = width;
    }
    return true;
}
LogicalResult qualifyFunction(presburger::PWMAFunction& function, SymbolicSchemaHandle schema,
                              unsigned baseWidth, unsigned& width, std::string& reason)
{
    unsigned eventWidth = schema->coordinateDepth() + 2;
    for (const auto& piece : function.getAllPieces()) {
        const auto& output = piece.output;
        if (output.getNumDomainVars() != eventWidth || output.getNumOutputs() != eventWidth ||
            output.getNumSymbolVars() != schema->parameters().size() ||
            failed(constantTag(output.getOutputExpr(0), schema->sites().size())) ||
            failed(constantTag(output.getOutputExpr(eventWidth - 1), 2))) {
            reason = "general endpoint tag/kind or shared tuple identity not executable"; return failure();
        }
        SmallVector<unsigned> widths(eventWidth + output.getNumSymbolVars(), baseWidth);
        if (!divisionWidths(output.getDivs(), widths, width)) {
            reason = "general endpoint contains unrepresented or cyclic division locals"; return failure();
        }
        for (unsigned i = 0; i < eventWidth; ++i) {
            unsigned bits = 0;
            if (!rowWidth(output.getOutputExpr(i), widths, bits)) {
                reason = "general endpoint arithmetic exceeds MLIR integer width"; return failure();
            }
            width = std::max(width, bits);
        }
        for (const auto& poly : piece.domain.getAllDisjuncts()) {
            SmallVector<unsigned> bounds(eventWidth + poly.getNumSymbolVars(), baseWidth);
            if (!divisionWidths(poly.getLocalReprs(), bounds, width)) {
                reason = "general endpoint guard has unrepresented division locals"; return failure();
            }
            for (unsigned i = 0; i < poly.getNumEqualities() + poly.getNumInequalities(); ++i) {
                auto row = i < poly.getNumEqualities() ? poly.getEquality(i) :
                    poly.getInequality(i - poly.getNumEqualities());
                unsigned bits = 0;
                if (!rowWidth(row, bounds, bits)) {
                    reason = "general endpoint guard arithmetic exceeds MLIR integer width"; return failure();
                }
                width = std::max(width, bits);
            }
        }
    }
    return success();
}
class Codegen {
public:
    Codegen(OpBuilder& builder, Location loc, unsigned width) : builder(builder), loc(loc),
        type(builder.getIntegerType(width)) {}
    Value constant(const Int& number)
    {
        return builder.create<arith::ConstantOp>(loc,
            builder.getIntegerAttr(type, APInt(type.getWidth(), decimal(number), 10)));
    }
    Value integer(Value value)
    {
        if (isa<IndexType>(value.getType())) { return builder.create<arith::IndexCastOp>(loc, type, value); }
        if (value.getType().isInteger(1) || cast<IntegerType>(value.getType()).isUnsigned()) {
            return builder.create<arith::ExtUIOp>(loc, type, value);
        }
        return builder.create<arith::ExtSIOp>(loc, type, value);
    }
    Value boolean(bool value) { return builder.create<arith::ConstantIntOp>(loc, value, 1); }
    Value row(ArrayRef<Int> coefficients, ArrayRef<Value> variables)
    {
        Value sum = constant(coefficients.back());
        for (auto [i, value] : llvm::enumerate(variables)) {
            if (coefficients[i] != Int(0)) {
                auto term = builder.create<arith::MulIOp>(loc, value, constant(coefficients[i]));
                sum = builder.create<arith::AddIOp>(loc, sum, term);
            }
        }
        return sum;
    }
    SmallVector<Value> divisions(const presburger::DivisionRepr& repr, ArrayRef<Value> original)
    {
        SmallVector<Value> values(original);
        values.resize(repr.getNumVars());
        for (unsigned i = 0; i < repr.getNumDivs(); ++i) {
            auto numerator = repr.getDividend(i);
            // Future locals have zero coefficients by qualification; substitute
            // zero explicitly so row compilation never reads a null SSA value.
            for (unsigned j = original.size() + i; j < values.size(); ++j) { values[j] = constant(Int(0)); }
            auto sum = row(numerator, values);
            values[original.size() + i] = builder.create<arith::FloorDivSIOp>(loc, sum, constant(repr.getDenom(i)));
        }
        return values;
    }
    Value guard(const presburger::PresburgerSet& domain, ArrayRef<Value> original)
    {
        Value enabled = boolean(false);
        for (const auto& poly : domain.getAllDisjuncts()) {
            auto values = divisions(poly.getLocalReprs(), original);
            Value present = boolean(true);
            for (unsigned i = 0; i < poly.getNumEqualities() + poly.getNumInequalities(); ++i) {
                bool equal = i < poly.getNumEqualities();
                auto terms = equal ? poly.getEquality(i) : poly.getInequality(i - poly.getNumEqualities());
                auto test = builder.create<arith::CmpIOp>(loc,
                    equal ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge, row(terms, values), constant(Int(0)));
                present = builder.create<arith::AndIOp>(loc, present, test);
            }
            enabled = builder.create<arith::OrIOp>(loc, enabled, present);
        }
        return enabled;
    }
private:
    OpBuilder& builder;
    Location loc;
    IntegerType type;
};
} // namespace
LogicalResult prepareGeneralEndpoints(SelectedAnalysis& selected, std::string& reason)
{
    if (!selected.general || !selected.structured || selected.structured->sites().empty()) {
        reason = "general endpoint source is missing"; return failure();
    }
    auto query = selected.general;
    auto schema = query->schema;
    auto plan = std::make_shared<GeneralEndpointPlan>();
    unsigned baseWidth = 66;
    DataLayout layout = DataLayout::closest(selected.structured->sites().front().anchor);
    SmallVector<PipelineType> pipes;
    DominanceInfo dominance;
    for (const auto& site : selected.structured->sites()) {
        if (!llvm::is_contained(pipes, site.phase->kPipeValue)) { pipes.push_back(site.phase->kPipeValue); }
        for (Value iv : site.inductionVariables) {
            baseWidth = std::max(baseWidth, unsigned(layout.getTypeSizeInBits(iv.getType()).getFixedValue()) + 2);
        }
        for (Value parameter : schema->parameters()) {
            if (!dominance.dominates(parameter, site.anchor)) {
                reason = "general endpoint parameter unavailable at original cut"; return failure();
            }
            baseWidth = std::max(
                baseWidth, unsigned(layout.getTypeSizeInBits(parameter.getType()).getFixedValue()) + 2);
        }
    }
    auto space = *schema->space(Tuple::Event, Tuple::Event);
    for (auto source : pipes) {
        for (auto target : pipes) {
            auto filtered = pipeRelation(query->minimum, schema, source, target, 1, 0);
            if (failed(filtered)) { reason = "general minimum literal source identities missing"; return failure(); }
            R relation = std::move(*filtered);
            relation.setSpace(space);
            if (relation.isIntegerEmpty()) { continue; }
            if (source == target) {
                // Native C/C and I/I strict chains, joined through one executed
                // occurrence's C/I identity, characterize an intervening payload.
                auto completions = pipeRelation(query->native, schema, source, target, 1, 1);
                auto starts = pipeRelation(query->native, schema, source, target, 0, 0);
                if (failed(completions) || failed(starts)) {
                    reason = "general native literal source identities missing"; return failure();
                }
                auto cc = std::move(*completions), ii = std::move(*starts);
                I identity(space);
                for (unsigned i = 0; i < space.getNumDomainVars(); ++i) {
                    SmallVector<Int> row(identity.getNumCols(), Int(0));
                    row[i] = Int(1); row[space.getNumDomainVars() + i] = Int(-1); identity.addEquality(row);
                }
                cc = cc.subtract(R(identity)); cc.setSpace(space);
                ii = ii.subtract(R(identity)); ii.setSpace(space);
                I flip(space);
                for (unsigned i = 0; i + 1 < space.getNumDomainVars(); ++i) {
                    SmallVector<Int> row(flip.getNumCols(), Int(0));
                    row[i] = Int(1); row[space.getNumDomainVars() + i] = Int(-1); flip.addEquality(row);
                }
                flip.addBound(presburger::BoundType::EQ, space.getNumDomainVars() - 1, Int(1));
                flip.addBound(presburger::BoundType::EQ, 2 * space.getNumDomainVars() - 1, Int(0));
                cc.compose(R(flip)); cc.setSpace(space); cc.compose(ii); cc.setSpace(space);
                if (!relation.intersect(cc).isIntegerEmpty()) {
                    reason = "unmet local-adjacency premise for exact direct construction"; return failure();
                }
            }
            for (bool outgoing : {true, false}) {
                if (outgoing && source == target) { continue; }
                auto oriented = relation;
                if (!outgoing) { oriented.inverse(); oriented.setSpace(space); }
                auto minimum = lexopt(oriented, false);
                auto maximum = lexopt(oriented, true);
                if (!minimum.unboundedDomain.isIntegerEmpty() || !maximum.unboundedDomain.isIntegerEmpty() ||
                    !minimum.lexopt.isEqual(maximum.lexopt)) {
                    reason = "general endpoint relation is not a uniquely matched function"; return failure();
                }
                // Normalize subtraction-generated existential locals before
                // compiling guards; preserve each function's original domain.
                presburger::PWMAFunction normalized(minimum.lexopt.getSpace());
                for (const auto& piece : minimum.lexopt.getAllPieces()) {
                    auto domain = piece.domain.hasOnlyDivLocals() ? piece.domain :
                        presburger::PresburgerSet(piece.domain.computeReprWithOnlyDivLocals());
                    normalized.addPiece({std::move(domain), piece.output});
                }
                if (failed(qualifyFunction(normalized, schema, baseWidth, plan->arithmeticWidth, reason))) {
                    return failure();
                }
                plan->endpoints.emplace_back(source, target, outgoing, std::move(normalized));
            }
        }
    }
    // PTO C++ lowering supports these concrete integer representations.
    // Arbitrary MLIR widths otherwise fall through to a narrower C++ type.
    for (unsigned width : {8U, 16U, 32U, 64U, 128U}) {
        if (plan->arithmeticWidth <= width) {
            plan->arithmeticWidth = width;
            selected.generalEndpoints = plan;
            return success();
        }
    }
    reason = "general endpoint arithmetic exceeds the supported target integer representation";
    return failure();
}
LogicalResult emitGeneralEndpoints(IRMapping& mapping, const SelectedAnalysis& selected, DirectEmissionResult& result)
{
    if (!selected.generalEndpoints || !selected.structured) {
        result.reason = "general executable selector qualification missing"; return failure();
    }
    auto input = selected.structured;
    auto schema = input->schema();
    auto count = schema->sites().size();
    if (count && count > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) / count) {
        result.reason = "general logical key not representable"; return failure();
    }
    DenseMap<const CompoundInstanceElement*, std::size_t> tags;
    for (auto [id, site] : llvm::enumerate(schema->sites())) { tags.try_emplace(site.phase, id); }
    // At each original cut: outgoing publications, local barrier, acquisitions.
    // Local/incoming guards insert before the payload; outgoing guards insert
    // after their own source. Install local barriers before incoming waits.
    for (unsigned pass : {0U, 1U, 2U}) {
        for (const auto& endpoint : selected.generalEndpoints->endpoints) {
            unsigned category = endpoint.outgoing ? 2 : (endpoint.source == endpoint.target ? 0 : 1);
            if (category != pass) { continue; }
            for (const auto& record : input->sites()) {
                auto pipe = endpoint.outgoing ? endpoint.source : endpoint.target;
                if (record.phase->kPipeValue != pipe) { continue; }
                auto found = tags.find(record.phase);
                if (found == tags.end()) { result.reason = "general scoped phase identity missing"; return failure(); }
                auto site = found->second;
                auto* anchor = mapping.lookup(record.anchor);
                OpBuilder builder(anchor->getContext());
                if (endpoint.outgoing) { builder.setInsertionPointAfter(anchor); }
                else { builder.setInsertionPoint(anchor); }
                auto loc = anchor->getLoc();
                Codegen code(builder, loc, selected.generalEndpoints->arithmeticWidth);
                SmallVector<Value> variables{code.constant(Int(static_cast<int64_t>(site)))};
                for (Value iv : record.inductionVariables) { variables.push_back(code.integer(mapping.lookup(iv))); }
                while (variables.size() <= schema->coordinateDepth()) { variables.push_back(code.constant(Int(0))); }
                variables.push_back(code.constant(Int(endpoint.outgoing ? 1 : 0)));
                for (Value parameter : schema->parameters()) {
                    variables.push_back(code.integer(mapping.lookup(parameter)));
                }
                auto previous = code.boolean(false);
                for (const auto& piece : endpoint.function.getAllPieces()) {
                    bool excluded = llvm::all_of(piece.domain.getAllDisjuncts(), [&](const auto& poly) {
                        auto tag = poly.getConstantBound64(presburger::BoundType::EQ, 0);
                        return tag && (*tag < 0 || static_cast<std::size_t>(*tag) != site);
                    });
                    if (excluded) { continue; }
                    auto targetTag = constantTag(piece.output.getOutputExpr(0), count);
                    if (failed(targetTag)) {
                        result.reason = "general endpoint output identity missing"; return failure();
                    }
                    auto present = code.guard(piece.domain, variables);
                    auto fresh = builder.create<arith::XOrIOp>(loc, previous, code.boolean(true));
                    auto enabled = builder.create<arith::AndIOp>(loc, present, fresh);
                    previous = builder.create<arith::OrIOp>(loc, previous, present);
                    auto conditional = builder.create<scf::IfOp>(loc, enabled, false);
                    auto body = conditional.getThenBodyBuilder();
                    if (endpoint.source == endpoint.target) {
                        body.create<pto::BarrierOp>(loc,
                            pto::PipeAttr::get(body.getContext(), static_cast<pto::PIPE>(endpoint.target)));
                        ++result.barriers; continue;
                    }
                    auto a = endpoint.outgoing ? site : *targetTag, b = endpoint.outgoing ? *targetTag : site;
                    SmallVector<Value> generation;
                    if (endpoint.outgoing) {
                        for (Value iv : record.inductionVariables) {
                            auto value = mapping.lookup(iv);
                            if (!isa<IndexType>(value.getType())) { value = body.create<arith::IndexCastOp>(loc,
                                body.getIndexType(), value); }
                            generation.push_back(value);
                        }
                    } else {
                        Codegen output(body, loc, selected.generalEndpoints->arithmeticWidth);
                        auto values = output.divisions(piece.output.getDivs(), variables);
                        for (unsigned i = 0; i < schema->sites()[a].coordinates.size(); ++i) {
                            auto value = output.row(piece.output.getOutputExpr(i + 1), values);
                            generation.push_back(body.create<arith::IndexCastOp>(loc, body.getIndexType(), value));
                        }
                    }
                    auto src = pto::PipeAttr::get(body.getContext(), static_cast<pto::PIPE>(endpoint.source));
                    auto dst = pto::PipeAttr::get(body.getContext(), static_cast<pto::PIPE>(endpoint.target));
                    auto key = body.getI64IntegerAttr(a * count + b);
                    if (endpoint.outgoing) {
                        body.create<pto::LogicalSetOp>(loc, src, dst, key, generation); ++result.sets;
                    }
                    else { body.create<pto::LogicalWaitOp>(loc, src, dst, key, generation); ++result.waits; }
                }
            }
        }
    }
    result.privateSelectors = !selected.generalEndpoints->endpoints.empty();
    return success();
}
} // namespace mlir::pto::frontiersynch
