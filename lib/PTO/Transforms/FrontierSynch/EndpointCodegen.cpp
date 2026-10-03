// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Compile already-proved endpoint maps into pure guards and direct logical mechanisms.
// No generated code queries effects, dependencies, reachability or solver state.
#include "DirectEmissionInternal.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <limits>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Int = llvm::DynamicAPInt;
std::string decimal(const Int& value)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    stream << value;
    return text;
}
class EndpointCodegen {
public:
    EndpointCodegen(StructuredInputHandle input, IRMapping& mapping, DirectEmissionResult& result)
        : input(std::move(input)), mapping(mapping), result(result)
    {}
    LogicalResult emit(
        const SignedSelector& selector, std::size_t site, const StructuredSite& original,
        PipelineType source, PipelineType target, bool outgoing);

private:
    Value integer(OpBuilder& builder, Location loc, Value value);
    Value constant(OpBuilder& builder, Location loc, const Int& value);
    Value boolean(OpBuilder& builder, Location loc, bool value);
    Value guard(
        OpBuilder& builder, Location loc, const SignedRelation& relation, std::size_t site, ArrayRef<Value> coordinates,
        ArrayRef<Value> parameters);
    Value endpoint(
        OpBuilder& builder, Location loc, const SignedOutputCoordinate& expression, ArrayRef<Value> coordinates,
        ArrayRef<Value> parameters);
    StructuredInputHandle input;
    IRMapping& mapping;
    DirectEmissionResult& result;
    IntegerType arithmeticType;
};
unsigned endpointWidth(StructuredInputHandle input, const SignedSelector& selector)
{
    unsigned bits = 66;
    DataLayout layout = DataLayout::closest(input->sites().front().anchor);
    for (Value parameter : input->schema()->parameters()) {
        bits = std::max(bits, unsigned(layout.getTypeSizeInBits(parameter.getType()).getFixedValue()) + 2);
    }
    for (const auto& site : input->sites()) {
        for (Value iv : site.inductionVariables) {
            bits = std::max(bits, unsigned(layout.getTypeSizeInBits(iv.getType()).getFixedValue()) + 2);
        }
    }
    auto bound = [&](const Int& value) {
        auto digits = decimal(value).size();
        if (digits > (IntegerType::kMaxWidth - 4) / 4) {
            bits = IntegerType::kMaxWidth + 1;
        } else {
            bits = std::max(bits, unsigned(digits * 4 + 4));
        }
    };
    for (const auto& piece : selector.pieces()) {
        for (const auto& guardPiece : piece.guard->pieces()) {
            for (const auto& atom : guardPiece.atoms) {
                bound(atom.bound);
            }
        }
        for (const auto& coordinate : piece.coordinates) {
            for (const auto& term : coordinate.lower) {
                bound(term.constant);
            }
        }
    }
    for (unsigned width : {8U, 16U, 32U, 64U, 128U}) {
        if (bits <= width) { return width; }
    }
    return 129;
}
Value EndpointCodegen::integer(OpBuilder& builder, Location loc, Value value)
{
    if (isa<IndexType>(value.getType())) {
        return builder.create<arith::IndexCastOp>(loc, arithmeticType, value);
    }
    if (value.getType().isInteger(1) || cast<IntegerType>(value.getType()).isUnsigned()) {
        return builder.create<arith::ExtUIOp>(loc, arithmeticType, value);
    }
    return builder.create<arith::ExtSIOp>(loc, arithmeticType, value);
}
Value EndpointCodegen::constant(OpBuilder& builder, Location loc, const Int& value)
{
    auto number = APInt(arithmeticType.getWidth(), decimal(value), 10);
    return builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(arithmeticType, number));
}
Value EndpointCodegen::boolean(OpBuilder& builder, Location loc, bool value)
{
    return builder.create<arith::ConstantIntOp>(loc, value, 1);
}
Value EndpointCodegen::guard(
    OpBuilder& builder, Location loc, const SignedRelation& relation, std::size_t site, ArrayRef<Value> coordinates,
    ArrayRef<Value> parameters)
{
    Value admitted = boolean(builder, loc, false);
    for (const auto& piece : relation.pieces()) {
        if (piece.range.site != site) {
            continue;
        }
        Value present = boolean(builder, loc, true);
        for (const auto& atom : piece.atoms) {
            Value sum = constant(builder, loc, Int(0));
            for (const auto& term : atom.terms) {
                Value value = term.axis.role == SignedAxisRole::Parameter ? parameters[term.axis.index] :
                                                                            coordinates[term.axis.index];
                sum = term.sign > 0 ? Value(builder.create<arith::AddIOp>(loc, sum, value)) :
                                      Value(builder.create<arith::SubIOp>(loc, sum, value));
            }
            Value test =
                builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, sum, constant(builder, loc, atom.bound));
            present = builder.create<arith::AndIOp>(loc, present, test);
        }
        admitted = builder.create<arith::OrIOp>(loc, admitted, present);
    }
    return admitted;
}
Value EndpointCodegen::endpoint(
    OpBuilder& builder, Location loc, const SignedOutputCoordinate& expression, ArrayRef<Value> coordinates,
    ArrayRef<Value> parameters)
{
    Value selected;
    for (const auto& term : expression.lower) {
        Value value = constant(builder, loc, term.constant);
        if (term.axis) {
            Value axis = term.axis->role == SignedAxisRole::Parameter ? parameters[term.axis->index] :
                                                                        coordinates[term.axis->index];
            value = term.sign > 0 ? Value(builder.create<arith::AddIOp>(loc, value, axis)) :
                                    Value(builder.create<arith::SubIOp>(loc, value, axis));
        }
        selected = selected ? Value(builder.create<arith::MaxSIOp>(loc, selected, value)) : value;
    }
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), selected);
}
LogicalResult EndpointCodegen::emit(
    const SignedSelector& selector, std::size_t site, const StructuredSite& original,
    PipelineType source, PipelineType target, bool outgoing)
{
    if (!llvm::any_of(selector.pieces(), [&](const auto& piece) {
            return llvm::any_of(
                piece.guard->pieces(), [&](const SignedPiece& guard) { return guard.range.site == site; });
        })) {
        return success();
    }
    Operation* anchor = mapping.lookup(original.anchor);
    OpBuilder builder(anchor->getContext());
    if (outgoing) {
        builder.setInsertionPointAfter(anchor);
    } else {
        builder.setInsertionPoint(anchor);
    }
    auto loc = anchor->getLoc();
    unsigned bits = endpointWidth(input, selector);
    if (bits > 128) {
        result.reason = "endpoint arithmetic exceeds the supported target integer representation";
        return failure();
    }
    arithmeticType = builder.getIntegerType(bits);
    result.privateSelectors = true;
    SmallVector<Value> coordinates, parameters;
    for (Value iv : original.inductionVariables) {
        coordinates.push_back(integer(builder, loc, mapping.lookup(iv)));
    }
    for (Value parameter : input->schema()->parameters()) {
        parameters.push_back(integer(builder, loc, mapping.lookup(parameter)));
    }
    Value previous = boolean(builder, loc, false);
    for (const auto& piece : selector.pieces()) {
        if (piece.guard->empty() ||
            !llvm::any_of(piece.guard->pieces(), [&](const SignedPiece& guard) { return guard.range.site == site; })) {
            continue;
        }
        Value present = guard(builder, loc, *piece.guard, site, coordinates, parameters);
        Value first = builder.create<arith::XOrIOp>(loc, previous, boolean(builder, loc, true));
        Value enabled = builder.create<arith::AndIOp>(loc, present, first);
        previous = builder.create<arith::OrIOp>(loc, previous, present);
        auto selected = builder.create<scf::IfOp>(loc, enabled, false);
        OpBuilder body = selected.getThenBodyBuilder();
        if (source == target) {
            body.create<pto::BarrierOp>(loc, pto::PipeAttr::get(builder.getContext(), static_cast<pto::PIPE>(target)));
            ++result.barriers;
            continue;
        }
        std::size_t sourceSite = outgoing ? site : *piece.output.site;
        std::size_t consumerSite = outgoing ? *piece.output.site : site;
        int64_t key = sourceSite * input->schema()->sites().size() + consumerSite;
        SmallVector<Value> generation;
        if (outgoing) {
            for (Value iv : original.inductionVariables) {
                Value coordinate = mapping.lookup(iv);
                if (!isa<IndexType>(coordinate.getType())) {
                    coordinate = body.create<arith::IndexCastOp>(loc, body.getIndexType(), coordinate);
                }
                generation.push_back(coordinate);
            }
        } else {
            for (const auto& coordinate : piece.coordinates) {
                generation.push_back(endpoint(body, loc, coordinate, coordinates, parameters));
            }
        }
        auto src = pto::PipeAttr::get(builder.getContext(), static_cast<pto::PIPE>(source));
        auto dst = pto::PipeAttr::get(builder.getContext(), static_cast<pto::PIPE>(target));
        if (outgoing) {
            body.create<pto::LogicalSetOp>(loc, src, dst, builder.getI64IntegerAttr(key), generation);
            ++result.sets;
        } else {
            body.create<pto::LogicalWaitOp>(loc, src, dst, builder.getI64IntegerAttr(key), generation);
            ++result.waits;
        }
    }
    return success();
}
} // namespace
namespace {
using SelectorProvider = std::function<SignedResult<SignedSelectorHandle>(PipelineType, PipelineType, bool)>;
LogicalResult emitSelectors(
    IRMapping& mapping, StructuredInputHandle input, const SelectorProvider& provider, DirectEmissionResult& result)
{
    auto schema = input->schema();
    auto count = schema->sites().size();
    if (input->sites().empty()) { return success(); }
    if (!count || count > static_cast<std::size_t>(std::numeric_limits<int64_t>::max()) / count) {
        return failure();
    }
    DenseMap<const CompoundInstanceElement*, std::size_t> tags;
    for (auto [tag, site] : llvm::enumerate(schema->sites())) {
        if (!tags.try_emplace(site.phase, tag).second) { return failure(); }
    }
    SmallVector<std::size_t> localTags;
    for (const auto& record : input->sites()) {
        auto found = tags.find(record.phase);
        if (found == tags.end() || schema->sites()[found->second].coordinates != record.inductionVariables) {
            return failure();
        }
        localTags.push_back(found->second);
    }
    EndpointCodegen emitter(input, mapping, result);
    SmallVector<PipelineType> pipes;
    for (const auto& site : input->sites()) {
        if (!llvm::is_contained(pipes, site.phase->kPipeValue)) {
            pipes.push_back(site.phase->kPipeValue);
        }
    }
    for (bool local : {true, false}) {
        for (auto [site, record] : llvm::enumerate(input->sites())) {
            auto target = record.phase->kPipeValue;
            for (PipelineType source : pipes) {
                if ((source == target) != local) {
                    continue;
                }
                auto selector = provider(source, target, false);
                if (!selector.succeeded() ||
                    failed(emitter.emit(*selector.value, localTags[site], record, source, target, false))) {
                    return failure();
                }
            }
        }
    }
    for (auto [site, record] : llvm::enumerate(input->sites())) {
        for (PipelineType target : pipes) {
            if (target == record.phase->kPipeValue) {
                continue;
            }
            auto selector = provider(record.phase->kPipeValue, target, true);
            if (!selector.succeeded() ||
                failed(emitter.emit(
                    *selector.value, localTags[site], record, record.phase->kPipeValue, target, true))) {
                return failure();
            }
        }
    }
    return success();
}
} // namespace
namespace {
SignedResult<SignedSelectorHandle> selectEndpoints(
    const SelectedAnalysis& selected, PipelineType source, PipelineType target, bool outgoing)
{
    if (selected.signedAnalysis) {
        return outgoing ? selected.signedAnalysis->outgoing(source, target) :
                          selected.signedAnalysis->incoming(source, target);
    }
    SmallVector<SignedPiece> pieces;
    for (const auto& piece : selected.minimum->pieces()) {
        if (selected.minimum->space()->schema()->sites()[*piece.domain.site].phase->kPipeValue == source &&
            selected.minimum->space()->schema()->sites()[*piece.range.site].phase->kPipeValue == target) {
            pieces.push_back(piece);
        }
    }
    auto relation =
        SignedRelation::import(selected.minimum->space(), SymbolicTuple::Event, SymbolicTuple::Event, pieces);
    if (!outgoing && relation.succeeded()) {
        relation = relation.value->inverse();
    }
    return relation.succeeded() ? SignedSelector::build(relation.value) :
                                  SignedResult<SignedSelectorHandle>{relation.status};
}
} // namespace
LogicalResult prepareEndpoints(SelectedAnalysis& selected, std::string& reason)
{
    SmallVector<PipelineType> pipes;
    for (const auto& site : selected.structured->sites()) {
        if (!llvm::is_contained(pipes, site.phase->kPipeValue)) {
            pipes.push_back(site.phase->kPipeValue);
        }
    }
    selected.endpoints.clear();
    for (PipelineType source : pipes) {
        for (PipelineType target : pipes) {
            for (bool outgoing : {false, true}) {
                if (source == target && outgoing) {
                    continue;
                }
                auto selector = selectEndpoints(selected, source, target, outgoing);
                if (!selector.succeeded()) {
                    reason = "matching endpoint obligation: " + signedDiagnostic(selector.status).str();
                    selected.endpoints.clear();
                    return failure();
                }
                if (endpointWidth(selected.structured, *selector.value) > 128) {
                    reason = "endpoint arithmetic exceeds the supported target integer representation";
                    selected.endpoints.clear();
                    return failure();
                }
                selected.endpoints.emplace(std::make_tuple(source, target, outgoing), selector.value);
            }
        }
    }
    return success();
}
LogicalResult emitPreparedEndpoints(IRMapping& mapping, const SelectedAnalysis& selected, DirectEmissionResult& result)
{
    return emitSelectors(
        mapping, selected.structured,
        [&](PipelineType source, PipelineType target, bool outgoing) -> SignedResult<SignedSelectorHandle> {
            auto found = selected.endpoints.find(std::make_tuple(source, target, outgoing));
            if (found == selected.endpoints.end()) {
                return {SignedStatus::InvalidInput};
            }
            return {SignedStatus::Success, found->second};
        },
        result);
}
} // namespace mlir::pto::frontiersynch
