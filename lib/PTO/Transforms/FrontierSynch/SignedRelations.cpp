// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SignedInternal.h"
#include <limits>
namespace mlir::pto::frontiersynch {
using namespace signed_detail;
namespace {
using Tuple = SymbolicTuple;
SmallVector<unsigned> sequence(unsigned begin, unsigned count)
{
    SmallVector<unsigned> result;
    for (unsigned i = 0; i < count; ++i) {
        result.push_back(begin + i);
    }
    return result;
}
bool compatible(const SignedRelation& a, const SignedRelationHandle& b)
{
    return b && a.space() == b->space() && a.domain() == b->domain() && a.range() == b->range();
}
enum class Join { Intersection, Compose, Domain, Range, Context };
SignedResult<SignedRelationHandle> join(const SignedRelation& left, SignedRelationHandle right, Join operation)
{
    if (!right || left.space() != right->space()) {
        return {SignedStatus::InvalidInput};
    }
    bool valid = false;
    switch (operation) {
        case Join::Intersection:
            valid = compatible(left, right);
            break;
        case Join::Compose:
            valid = left.range() == right->domain();
            break;
        case Join::Domain:
            valid = right->domain() == Tuple::Unit && right->range() == left.domain();
            break;
        case Join::Range:
            valid = right->domain() == Tuple::Unit && right->range() == left.range();
            break;
        case Join::Context:
            valid = right->domain() == Tuple::Unit && right->range() == Tuple::Unit;
            break;
    }
    if (!valid) {
        return {SignedStatus::InvalidInput};
    }
    const auto outputRange = operation == Join::Compose ? right->range() : left.range();
    SmallVector<SignedPiece, 0> output;
    for (const auto& a : left.pieces()) {
        const auto la = layout(left.space(), left.domain(), left.range(), a).value;
        const auto pa = decode(left.space(), left.domain(), left.range(), a).value;
        for (const auto& b : right->pieces()) {
            const auto lb = layout(right->space(), right->domain(), right->range(), b).value;
            const auto ra = ArrayRef<BigInt>(a.residues), rb = ArrayRef<BigInt>(b.residues);
            if (ra.take_back(la.parameters) != rb.take_back(lb.parameters)) {
                continue;
            }
            if ((operation == Join::Intersection && !signature(a, b)) ||
                (operation == Join::Compose &&
                 (!(a.range == b.domain) || ra.slice(la.domain, la.range) != rb.take_front(lb.domain))) ||
                (operation == Join::Domain &&
                 (!(a.domain == b.range) || ra.take_front(la.domain) != rb.take_front(lb.range))) ||
                (operation == Join::Range &&
                 (!(a.range == b.range) || ra.slice(la.domain, la.range) != rb.take_front(lb.range)))) {
                continue;
            }
            Layout shape = la;
            SignedTag target = a.range;
            SmallVector<BigInt> residues = a.residues;
            SmallVector<unsigned> ma = sequence(0, la.free()), mb;
            unsigned dimensions = la.free();
            if (operation == Join::Compose) {
                const auto count = std::uint64_t(la.domain) + lb.range + la.parameters + la.range;
                if (count >= std::numeric_limits<unsigned>::max()) {
                    return {SignedStatus::RepresentationLimit};
                }
                shape.range = lb.range;
                dimensions = count;
                target = b.range;
                residues.assign(ra.begin(), ra.begin() + la.domain);
                residues.append(rb.begin() + lb.domain, rb.end());
                ma = sequence(0, la.domain);
                ma.append(sequence(shape.free(), la.range));
                ma.append(sequence(shape.domain + shape.range, shape.parameters));
                mb = sequence(shape.free(), lb.domain);
                mb.append(sequence(shape.domain, shape.range));
                mb.append(sequence(shape.domain + shape.range, shape.parameters));
            } else if (operation == Join::Intersection) {
                mb = sequence(0, lb.free());
            } else {
                const unsigned offset = operation == Join::Range ? la.domain : 0;
                if (operation != Join::Context) {
                    mb = sequence(offset, lb.range);
                }
                mb.append(sequence(la.domain + la.range, la.parameters));
            }
            auto combined = remap(pa, ma, dimensions);
            conjoin(combined, remap(decode(right->space(), right->domain(), right->range(), b).value, mb, dimensions));
            combined.project(sequence(0, shape.free()));
            if (combined.feasible()) {
                append(output, encode(combined, shape, a.domain, target, residues), shape);
            }
        }
    }
    return {SignedStatus::Success, Access::make(left.space(), left.domain(), outputRange, std::move(output))};
}
} // namespace
SignedResult<SignedSpaceHandle> SignedSpace::create(SymbolicSchemaHandle schema, const BigInt& period)
{
    if (!schema || period <= 0) {
        return {SignedStatus::InvalidInput};
    }
    auto result = std::shared_ptr<SignedSpace>(new SignedSpace());
    result->owner = std::move(schema);
    result->modulus = period;
    return {SignedStatus::Success, SignedSpaceHandle(result)};
}
SignedResult<SignedRelationHandle> SignedRelation::import(
    SignedSpaceHandle space, SymbolicTuple domain, SymbolicTuple range, ArrayRef<SignedPiece> pieces)
{
    if (!space || failed(space->schema()->arity(domain)) || failed(space->schema()->arity(range))) {
        return {SignedStatus::InvalidInput};
    }
    SmallVector<SignedPiece, 0> normalized;
    for (const auto& piece : pieces) {
        auto poly = decode(space, domain, range, piece);
        if (!poly.succeeded()) {
            return {poly.status};
        }
        const auto shape = layout(space, domain, range, piece).value;
        poly.value.project(sequence(0, shape.free()));
        if (poly.value.feasible()) {
            append(normalized, encode(poly.value, shape, piece.domain, piece.range, piece.residues), shape);
        }
    }
    return {SignedStatus::Success, Access::make(std::move(space), domain, range, std::move(normalized))};
}
SignedResult<SignedRelationHandle> SignedRelation::unite(const SignedRelationHandle& other) const
{
    if (!compatible(*this, other)) {
        return {SignedStatus::InvalidInput};
    }
    SmallVector<SignedPiece, 0> result(data.begin(), data.end());
    for (const auto& piece : other->pieces()) {
        append(result, piece, layout(owner, source, target, piece).value);
    }
    return {SignedStatus::Success, Access::make(owner, source, target, std::move(result))};
}
SignedResult<SignedRelationHandle> SignedRelation::intersect(const SignedRelationHandle& other) const
{
    return join(*this, other, Join::Intersection);
}
SignedResult<SignedRelationHandle> SignedRelation::compose(const SignedRelationHandle& other) const
{
    return join(*this, other, Join::Compose);
}
SignedResult<SignedRelationHandle> SignedRelation::restrictDomain(const SignedRelationHandle& set) const
{
    return join(*this, set, Join::Domain);
}
SignedResult<SignedRelationHandle> SignedRelation::restrictRange(const SignedRelationHandle& set) const
{
    return join(*this, set, Join::Range);
}
SignedResult<SignedRelationHandle> SignedRelation::restrictContext(const SignedRelationHandle& set) const
{
    return join(*this, set, Join::Context);
}
SignedResult<SignedRelationHandle> SignedRelation::inverse() const
{
    SmallVector<SignedPiece, 0> result;
    for (const auto& piece : data) {
        const auto shape = layout(owner, source, target, piece).value;
        SmallVector<unsigned> mapping = sequence(shape.range, shape.domain);
        mapping.append(sequence(0, shape.range));
        mapping.append(sequence(shape.domain + shape.range, shape.parameters));
        auto poly = remap(decode(owner, source, target, piece).value, mapping, shape.free());
        SmallVector<BigInt> residues(
            piece.residues.begin() + shape.domain, piece.residues.begin() + shape.domain + shape.range);
        residues.append(piece.residues.begin(), piece.residues.begin() + shape.domain);
        residues.append(piece.residues.end() - shape.parameters, piece.residues.end());
        result.push_back(
            encode(poly, {shape.range, shape.domain, shape.parameters}, piece.range, piece.domain, residues));
    }
    return {SignedStatus::Success, Access::make(owner, target, source, std::move(result))};
}
SignedResult<SignedRelationHandle> SignedRelation::domainSet() const
{
    SmallVector<SignedPiece, 0> result;
    for (const auto& piece : data) {
        const auto shape = layout(owner, source, target, piece).value;
        auto retained = sequence(0, shape.domain);
        retained.append(sequence(shape.domain + shape.range, shape.parameters));
        auto poly = decode(owner, source, target, piece).value;
        poly.project(retained);
        SmallVector<BigInt> residues;
        for (auto axis : retained) {
            residues.push_back(piece.residues[axis]);
        }
        const Layout output{0, shape.domain, shape.parameters};
        append(result, encode(poly, output, {}, piece.domain, residues), output);
    }
    return {SignedStatus::Success, Access::make(owner, Tuple::Unit, source, std::move(result))};
}
SignedResult<SignedRelationHandle> SignedRelation::rangeSet() const
{
    auto reverse = inverse();
    return reverse.value->domainSet();
}
SignedResult<SignedRelationHandle> SignedRelation::subtract(const SignedRelationHandle& other) const
{
    if (!compatible(*this, other)) {
        return {SignedStatus::InvalidInput};
    }
    SmallVector<SignedPiece, 0> result;
    for (const auto& left : data) {
        const auto shape = layout(owner, source, target, left).value;
        SmallVector<Poly, 0> remaining{decode(owner, source, target, left).value};
        for (const auto& right : other->pieces()) {
            if (!signature(left, right)) {
                continue;
            }
            const auto excluded = decode(owner, source, target, right).value;
            SmallVector<Poly, 0> next;
            for (auto prefix : remaining) {
                // Pairwise-disjoint first-failed-atom branches partition prefix\right.
                // Feasibility runs on copies: no new stored splitting thresholds.
                for (const auto& [direction, bound] : excluded.bounds) {
                    auto outside = prefix;
                    outside.add(Direction{-direction.first, -direction.second}, -bound - 1);
                    if (outside.feasible()) {
                        next.push_back(std::move(outside));
                    }
                    prefix.add(direction, bound);
                    if (!prefix.feasible()) {
                        break;
                    }
                }
            }
            remaining = std::move(next);
            if (remaining.empty()) {
                break;
            }
        }
        for (const auto& poly : remaining) {
            append(result, encode(poly, shape, left.domain, left.range, left.residues), shape);
        }
    }
    return {SignedStatus::Success, Access::make(owner, source, target, std::move(result))};
}
SignedResult<bool> SignedRelation::contains(
    const SignedPoint& a, const SignedPoint& b, ArrayRef<BigInt> parameters) const
{
    auto da = active(owner, source, a.tag), db = active(owner, target, b.tag);
    if (!da.succeeded() || !db.succeeded() || a.coordinates.size() != da.value || b.coordinates.size() != db.value ||
        parameters.size() != owner->schema()->parameters().size()) {
        return {SignedStatus::InvalidInput};
    }
    SmallVector<BigInt> values(a.coordinates.begin(), a.coordinates.end());
    values.append(b.coordinates.begin(), b.coordinates.end());
    values.append(parameters.begin(), parameters.end());
    SmallVector<BigInt> residues, quotients;
    for (const auto& value : values) {
        residues.push_back(llvm::mod(value, owner->period()));
        quotients.push_back(llvm::floorDiv(value, owner->period()));
    }
    for (const auto& piece : data) {
        if (piece.domain == a.tag && piece.range == b.tag && piece.residues == residues &&
            decode(owner, source, target, piece).value.contains(quotients)) {
            return {SignedStatus::Success, true};
        }
    }
    return {SignedStatus::Success, false};
}
} // namespace mlir::pto::frontiersynch
