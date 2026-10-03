// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Recognize literal octagonal primitives by substitution, never by approximate projection.
#include "DirectEmissionInternal.h"
namespace mlir::pto::frontiersynch {
namespace {
using Int = llvm::DynamicAPInt;
using Tuple = SymbolicTuple;
struct Column {
    std::optional<SignedAxis> axis;
    Int fixed = Int(0);
};
SmallVector<std::optional<std::size_t>> tags(SymbolicSchemaHandle schema, Tuple tuple)
{
    SmallVector<std::optional<std::size_t>> result;
    if (tuple == Tuple::Unit) {
        result.push_back(std::nullopt);
    } else {
        for (std::size_t site = 0; site < schema->sites().size(); ++site) {
            result.push_back(site);
        }
    }
    return result;
}
void columns(
    SmallVector<Column>& result, SymbolicSchemaHandle schema, std::optional<std::size_t> tag, SignedAxisRole role)
{
    if (!tag) {
        return;
    }
    result.push_back({{}, Int(static_cast<int64_t>(*tag))});
    auto count = schema->sites()[*tag].coordinates.size();
    for (unsigned i = 0; i < schema->coordinateDepth(); ++i) {
        result.push_back({i < count ? std::optional<SignedAxis>({role, i}) : std::nullopt, Int(0)});
    }
}
FailureOr<SignedRelationHandle> import(
    SignedSpaceHandle space, const SymbolicPrimitive& primitive, ArithmeticClass arithmetic)
{
    auto schema = space->schema();
    SmallVector<SignedPiece> pieces;
    for (auto a : tags(schema, primitive.domain())) {
        for (auto b : tags(schema, primitive.range())) {
            SmallVector<Column> axes;
            columns(axes, schema, a, SignedAxisRole::Domain);
            columns(axes, schema, b, SignedAxisRole::Range);
            for (unsigned i = 0; i < schema->parameters().size(); ++i) {
                axes.push_back({SignedAxis{SignedAxisRole::Parameter, i}, Int(0)});
            }
            for (const auto& poly : primitive.relation().getAllDisjuncts()) {
                if (poly.getNumLocalVars()) {
                    return failure();
                }
                SignedPiece piece;
                piece.domain.site = a;
                piece.range.site = b;
                unsigned count = schema->parameters().size();
                count += a ? schema->sites()[*a].coordinates.size() : 0;
                count += b ? schema->sites()[*b].coordinates.size() : 0;
                piece.residues.assign(count, Int(0));
                bool empty = false, supported = true;
                auto row = [&](ArrayRef<Int> coefficients, int sign) {
                    SignedAtom atom;
                    atom.bound = sign * coefficients.back();
                    for (unsigned i = 0; i < axes.size(); ++i) {
                        if (!axes[i].axis) {
                            atom.bound += sign * coefficients[i] * axes[i].fixed;
                        } else if (coefficients[i] != 0) {
                            if (coefficients[i] != 1 && coefficients[i] != -1) {
                                supported = false;
                                return;
                            }
                            atom.terms.push_back({*axes[i].axis, -sign * int(static_cast<int64_t>(coefficients[i]))});
                        }
                    }
                    if (atom.terms.empty()) {
                        empty |= atom.bound < 0;
                    } else if (atom.terms.size() > 2) {
                        supported = false;
                    } else if (
                        arithmetic == ArithmeticClass::Differences && atom.terms.size() == 2 &&
                        atom.terms[0].sign == atom.terms[1].sign) {
                        supported = false;
                    } else {
                        piece.atoms.push_back(std::move(atom));
                    }
                };
                for (unsigned i = 0; i < poly.getNumEqualities(); ++i) {
                    row(poly.getEquality(i), 1);
                    row(poly.getEquality(i), -1);
                }
                for (unsigned i = 0; i < poly.getNumInequalities(); ++i) {
                    row(poly.getInequality(i), 1);
                }
                if (!empty && !supported) {
                    return failure();
                }
                if (!empty) {
                    pieces.push_back(std::move(piece));
                }
            }
        }
    }
    auto result = SignedRelation::import(space, primitive.domain(), primitive.range(), pieces);
    if (!result.succeeded()) {
        return failure();
    }
    return result.value;
}
} // namespace
FailureOr<SignedInputs> specializePrimitives(
    StructuredInputHandle input, ArithmeticClass arithmetic, std::string& reason, SignedSpaceHandle shared)
{
    if (!input || (shared && (shared->schema() != input->schema() || shared->period() != Int(1)))) {
        reason = "signed primitive space must retain the input schema and unit period";
        return failure();
    }
    auto space = shared ? SignedResult<SignedSpaceHandle>{SignedStatus::Success, shared} :
        SignedSpace::create(input->schema(), Int(1));
    if (!space.succeeded()) {
        reason = "endpoint schema cannot be represented by signed relations";
        return failure();
    }
    const auto& original = input->relations();
    auto context = import(space.value, original.context, arithmetic);
    auto present = import(space.value, original.present, arithmetic);
    auto reference = import(space.value, original.reference, arithmetic);
    auto extras = import(space.value, original.extras, arithmetic);
    auto generators = import(space.value, *original.generators, arithmetic);
    if (failed(context) || failed(present) || failed(reference) || failed(extras) || failed(generators)) {
        reason = arithmetic == ArithmeticClass::Differences ? "primitives are outside difference bounds" :
                                                              "primitives are outside integer octagons";
        return failure();
    }
    return SignedInputs{*context, *present, *reference, {}, {}, *extras, *generators};
}
} // namespace mlir::pto::frontiersynch
