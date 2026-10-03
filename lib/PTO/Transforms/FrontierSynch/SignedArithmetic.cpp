// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SignedInternal.h"
#include <limits>
namespace mlir::pto::frontiersynch::signed_detail {
bool Poly::add(ArrayRef<std::int64_t> terms, BigInt constant)
{
    std::map<unsigned, int> coefficients;
    for (auto term : terms) {
        if (!term) {
            continue;
        }
        const auto axis = static_cast<std::uint64_t>(term > 0 ? term : -term) - 1;
        if (axis >= dimensions) {
            return false;
        }
        coefficients[axis] += term > 0 ? 1 : -1;
    }
    SmallVector<std::int64_t, 2> normalized;
    for (auto [axis, coefficient] : coefficients) {
        if (coefficient) {
            normalized.push_back((coefficient > 0 ? 1 : -1) * (std::int64_t(axis) + 1));
        }
    }
    if (normalized.empty()) {
        empty |= constant < 0;
        return true;
    }
    if (normalized.size() > 2) {
        return false;
    }
    for (auto [axis, coefficient] : coefficients) {
        if (coefficient > 1 || coefficient < -1) {
            if (normalized.size() != 1 || (coefficient != 2 && coefficient != -2)) {
                return false;
            }
            constant = llvm::floorDiv(constant, BigInt(2));
        }
    }
    std::sort(normalized.begin(), normalized.end());
    const Direction key{normalized[0], normalized.size() == 2 ? normalized[1] : 0};
    auto [entry, inserted] = bounds.emplace(key, constant);
    if (!inserted && constant < entry->second) {
        entry->second = std::move(constant);
    }
    return true;
}
bool Poly::add(Direction direction, const BigInt& constant)
{
    return add(SmallVector<std::int64_t, 2>{direction.first, direction.second}, constant);
}
void Poly::eliminate(unsigned axis)
{
    const auto positive = std::int64_t(axis) + 1;
    SmallVector<std::pair<std::int64_t, BigInt>> lower, upper;
    Poly result(dimensions);
    result.empty = empty;
    for (const auto& [direction, constant] : bounds) {
        auto [a, b] = direction;
        if (a == positive || b == positive) {
            upper.push_back({a == positive ? b : a, constant});
        } else if (a == -positive || b == -positive) {
            lower.push_back({a == -positive ? b : a, constant});
        } else {
            result.add(direction, constant);
        }
    }
    // Integral endpoints: every lower <= every upper is necessary and sufficient.
    // Missing lower/upper lists impose no restriction on remaining coordinates.
    for (const auto& [a, c] : lower) {
        for (const auto& [b, d] : upper) {
            result.add(Direction{a, b}, c + d);
        }
    }
    bounds = std::move(result.bounds);
    empty = result.empty;
}
void Poly::project(ArrayRef<unsigned> retained)
{
    SmallVector<bool> keep(dimensions, false);
    for (auto axis : retained) {
        keep[axis] = true;
    }
    for (unsigned axis = 0; axis < dimensions && !empty; ++axis) {
        if (!keep[axis]) {
            eliminate(axis);
        }
    }
    SmallVector<unsigned> mapping(dimensions, 0);
    for (auto [index, axis] : llvm::enumerate(retained)) {
        mapping[axis] = index;
    }
    *this = remap(*this, mapping, retained.size());
}
bool Poly::feasible() const
{
    Poly copy = *this;
    copy.project({});
    return !copy.empty;
}
bool Poly::implies(const Poly& other) const
{
    if (empty) {
        return true;
    }
    if (other.empty) {
        return !feasible();
    }
    for (const auto& [direction, bound] : other.bounds) {
        auto found = bounds.find(direction);
        if (found != bounds.end() && found->second <= bound) {
            continue;
        }
        Poly outside = *this;
        outside.add(Direction{-direction.first, -direction.second}, -bound - 1);
        if (outside.feasible()) {
            return false;
        }
    }
    return !other.empty;
}
bool Poly::contains(ArrayRef<BigInt> point) const
{
    if (empty || point.size() != dimensions) {
        return false;
    }
    for (const auto& [direction, bound] : bounds) {
        BigInt sum(0);
        for (auto term : {direction.first, direction.second}) {
            if (term) {
                sum += (term > 0 ? 1 : -1) * point[(term > 0 ? term : -term) - 1];
            }
        }
        if (sum > bound) {
            return false;
        }
    }
    return true;
}
Poly remap(const Poly& source, ArrayRef<unsigned> mapping, unsigned dimensions)
{
    Poly result(dimensions);
    result.empty = source.empty;
    for (const auto& [direction, constant] : source.bounds) {
        SmallVector<std::int64_t, 2> terms;
        for (auto term : {direction.first, direction.second}) {
            if (term) {
                terms.push_back((term > 0 ? 1 : -1) * (std::int64_t(mapping[(term > 0 ? term : -term) - 1]) + 1));
            }
        }
        result.add(terms, constant);
    }
    return result;
}
void conjoin(Poly& target, const Poly& source)
{
    target.empty |= source.empty;
    for (const auto& [direction, constant] : source.bounds) {
        target.add(direction, constant);
    }
}
SignedResult<unsigned> active(const SignedSpaceHandle& space, SymbolicTuple role, const SignedTag& tag)
{
    if (!space) {
        return {SignedStatus::InvalidInput};
    }
    if (role == SymbolicTuple::Unit || role == SymbolicTuple::Cell) {
        if (tag.site || tag.kind) {
            return {SignedStatus::InvalidInput};
        }
        return {
            SignedStatus::Success, role == SymbolicTuple::Unit ? 0U : unsigned(space->schema()->cellTypes().size())};
    }
    if ((role != SymbolicTuple::Occurrence && role != SymbolicTuple::Event) || !tag.site ||
        *tag.site >= space->schema()->sites().size() || (role == SymbolicTuple::Occurrence && tag.kind) ||
        (role == SymbolicTuple::Event &&
         (!tag.kind || (*tag.kind != PeriodicEventKind::Start && *tag.kind != PeriodicEventKind::Completion)))) {
        return {SignedStatus::InvalidInput};
    }
    return {SignedStatus::Success, unsigned(space->schema()->sites()[*tag.site].coordinates.size())};
}
SignedResult<Layout> layout(
    const SignedSpaceHandle& space, SymbolicTuple domain, SymbolicTuple range, const SignedPiece& piece)
{
    auto a = active(space, domain, piece.domain), b = active(space, range, piece.range);
    if (!a.succeeded() || !b.succeeded()) {
        return {SignedStatus::InvalidInput};
    }
    const auto parameters = space->schema()->parameters().size();
    const auto count = std::uint64_t(a.value) + b.value + parameters + piece.locals;
    if (count >= std::numeric_limits<unsigned>::max()) {
        return {SignedStatus::RepresentationLimit};
    }
    return {SignedStatus::Success, {a.value, b.value, unsigned(parameters)}};
}
SignedResult<Poly> decode(
    const SignedSpaceHandle& space, SymbolicTuple domain, SymbolicTuple range, const SignedPiece& piece)
{
    auto shape = layout(space, domain, range, piece);
    if (!shape.succeeded()) {
        return {shape.status};
    }
    const auto [a, b, p] = shape.value;
    if (piece.residues.size() != shape.value.free() ||
        llvm::any_of(piece.residues, [&](const auto& r) { return r < 0 || r >= space->period(); })) {
        return {SignedStatus::InvalidInput};
    }
    Poly result(shape.value.free() + piece.locals);
    for (const auto& atom : piece.atoms) {
        if (atom.terms.size() > 2) {
            return {SignedStatus::InvalidInput};
        }
        SmallVector<std::int64_t, 2> terms;
        for (const auto& term : atom.terms) {
            unsigned offset = 0, count = 0;
            switch (term.axis.role) {
                case SignedAxisRole::Domain:
                    count = a;
                    break;
                case SignedAxisRole::Range:
                    offset = a;
                    count = b;
                    break;
                case SignedAxisRole::Parameter:
                    offset = a + b;
                    count = p;
                    break;
                case SignedAxisRole::Local:
                    offset = a + b + p;
                    count = piece.locals;
                    break;
                default:
                    return {SignedStatus::InvalidInput};
            }
            if (term.axis.index >= count || (term.sign != 1 && term.sign != -1)) {
                return {SignedStatus::InvalidInput};
            }
            terms.push_back(term.sign * (std::int64_t(offset + term.axis.index) + 1));
        }
        if (!result.add(terms, atom.bound)) {
            return {SignedStatus::InvalidInput};
        }
    }
    return {SignedStatus::Success, std::move(result)};
}
SignedPiece encode(const Poly& poly, const Layout& shape, SignedTag domain, SignedTag range, ArrayRef<BigInt> residues)
{
    SignedPiece piece;
    piece.domain = domain;
    piece.range = range;
    piece.residues.assign(residues.begin(), residues.end());
    if (poly.empty) {
        piece.atoms.push_back({{}, BigInt(-1)});
        return piece;
    }
    for (const auto& [direction, constant] : poly.bounds) {
        SignedAtom atom;
        atom.bound = constant;
        for (auto term : {direction.first, direction.second}) {
            if (!term) {
                continue;
            }
            unsigned index = (term > 0 ? term : -term) - 1;
            SignedAxisRole role = SignedAxisRole::Domain;
            if (index >= shape.domain + shape.range) {
                index -= shape.domain + shape.range;
                role = SignedAxisRole::Parameter;
            } else if (index >= shape.domain) {
                index -= shape.domain;
                role = SignedAxisRole::Range;
            }
            atom.terms.push_back({{role, index}, term > 0 ? 1 : -1});
        }
        piece.atoms.push_back(std::move(atom));
    }
    return piece;
}
bool signature(const SignedPiece& a, const SignedPiece& b)
{
    return a.domain == b.domain && a.range == b.range && a.residues == b.residues;
}
void append(SmallVectorImpl<SignedPiece>& output, SignedPiece piece, const Layout& shape)
{
    // Cheap syntactic subsumption only; no join or threshold-generating closure.
    auto included = [](const SignedPiece& a, const SignedPiece& b) {
        return llvm::all_of(b.atoms, [&](const SignedAtom& y) {
            return llvm::any_of(a.atoms, [&](const SignedAtom& x) {
                return x.bound <= y.bound && x.terms.size() == y.terms.size() &&
                       llvm::equal(x.terms, y.terms, [](const auto& u, const auto& v) {
                           return u.axis == v.axis && u.sign == v.sign;
                       });
            });
        });
    };
    for (const auto& old : output) {
        if (signature(old, piece) && included(piece, old)) {
            return;
        }
    }
    llvm::erase_if(output, [&](const auto& old) { return signature(old, piece) && included(old, piece); });
    output.push_back(std::move(piece));
}
SignedRelationHandle Access::make(
    SignedSpaceHandle space, SymbolicTuple domain, SymbolicTuple range, SmallVector<SignedPiece, 0> pieces)
{
    auto result = std::shared_ptr<SignedRelation>(new SignedRelation());
    result->owner = std::move(space);
    result->source = domain;
    result->target = range;
    result->data = std::move(pieces);
    return result;
}
SignedPoint point(const SymbolicEvent& event) { return {{event.site, event.kind}, event.coordinates}; }
} // namespace mlir::pto::frontiersynch::signed_detail
