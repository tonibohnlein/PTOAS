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
namespace signed_detail {
SignedResult<SignedRelationHandle> identity(SignedRelationHandle set)
{
    if (!set || set->domain() != SymbolicTuple::Unit) {
        return {SignedStatus::InvalidInput};
    }
    SmallVector<SignedPiece, 0> output;
    for (const auto& piece : set->pieces()) {
        const auto shape = layout(set->space(), set->domain(), set->range(), piece).value;
        if (std::uint64_t(2) * shape.range + shape.parameters >= std::numeric_limits<unsigned>::max()) {
            return {SignedStatus::RepresentationLimit};
        }
        Layout target{shape.range, shape.range, shape.parameters};
        SmallVector<unsigned> mapping;
        for (unsigned i = 0; i < shape.range; ++i) {
            mapping.push_back(i);
        }
        for (unsigned i = 0; i < shape.parameters; ++i) {
            mapping.push_back(2 * shape.range + i);
        }
        auto poly = remap(decode(set->space(), set->domain(), set->range(), piece).value, mapping, target.free());
        for (unsigned i = 0; i < shape.range; ++i) {
            const auto a = std::int64_t(i) + 1, b = std::int64_t(shape.range + i) + 1;
            poly.add(Direction{a, -b}, BigInt(0));
            poly.add(Direction{-a, b}, BigInt(0));
        }
        SmallVector<BigInt> residues(piece.residues.begin(), piece.residues.begin() + shape.range);
        residues.append(piece.residues.begin(), piece.residues.end());
        append(output, encode(poly, target, piece.range, piece.range, residues), target);
    }
    return {SignedStatus::Success, Access::make(set->space(), set->range(), set->range(), std::move(output))};
}
SignedResult<SignedRelationHandle> lift(
    SignedRelationHandle relation, PeriodicEventKind source, PeriodicEventKind target)
{
    if (!relation || relation->domain() != SymbolicTuple::Occurrence ||
        relation->range() != SymbolicTuple::Occurrence) {
        return {SignedStatus::InvalidInput};
    }
    SmallVector<SignedPiece, 0> output(relation->pieces().begin(), relation->pieces().end());
    for (auto& piece : output) {
        piece.domain.kind = source;
        piece.range.kind = target;
    }
    return {
        SignedStatus::Success,
        Access::make(relation->space(), SymbolicTuple::Event, SymbolicTuple::Event, std::move(output))};
}
SignedResult<SignedRelationHandle> filterPipes(SignedRelationHandle relation, PipelineType source, PipelineType target)
{
    if (!relation || relation->domain() != SymbolicTuple::Event || relation->range() != SymbolicTuple::Event) {
        return {SignedStatus::InvalidInput};
    }
    SmallVector<SignedPiece, 0> output;
    for (const auto& piece : relation->pieces()) {
        const auto sites = relation->space()->schema()->sites();
        if (sites[*piece.domain.site].phase->kPipeValue == source &&
            sites[*piece.range.site].phase->kPipeValue == target) {
            output.push_back(piece);
        }
    }
    return {
        SignedStatus::Success,
        Access::make(relation->space(), relation->domain(), relation->range(), std::move(output))};
}
} // namespace signed_detail
namespace {
using R = SignedRelationHandle;
using Tuple = SymbolicTuple;
using Kind = PeriodicEventKind;
bool matches(R value, SignedSpaceHandle space, Tuple domain, Tuple range)
{
    return value && value->space() == space && value->domain() == domain && value->range() == range;
}
class Builder {
public:
    SignedStatus status = SignedStatus::Success;
    R take(SignedResult<R> result)
    {
        if (!result.succeeded() && status == SignedStatus::Success) {
            status = result.status;
        }
        return result.value;
    }
    using Binary = SignedResult<R> (SignedRelation::*)(const R&) const;
    R operation(R a, R b, Binary function)
    {
        if (!a || !b) {
            return {};
        }
        return take((a.get()->*function)(b));
    }
    R unite(R a, R b) { return operation(a, b, &SignedRelation::unite); }
    R intersect(R a, R b) { return operation(a, b, &SignedRelation::intersect); }
    R compose(R a, R b) { return operation(a, b, &SignedRelation::compose); }
    R subtract(R a, R b) { return operation(a, b, &SignedRelation::subtract); }
    R inverse(R a) { return a ? take(a->inverse()) : R{}; }
    R restrict(R relation, R context, R present, bool range)
    {
        relation = operation(relation, context, &SignedRelation::restrictContext);
        relation = operation(relation, present, &SignedRelation::restrictDomain);
        return range ? operation(relation, present, &SignedRelation::restrictRange) : relation;
    }
};
} // namespace
SignedResult<SignedAnalysisHandle> SignedDemandAnalysis::build(SignedSpaceHandle space, const SignedInputs& inputs)
{
    if (!space || !matches(inputs.context, space, Tuple::Unit, Tuple::Unit) ||
        !matches(inputs.present, space, Tuple::Unit, Tuple::Occurrence) ||
        !matches(inputs.reference, space, Tuple::Occurrence, Tuple::Occurrence) ||
        !matches(inputs.extras, space, Tuple::Occurrence, Tuple::Occurrence) ||
        (inputs.generators && !matches(inputs.generators, space, Tuple::Occurrence, Tuple::Occurrence)) ||
        (!inputs.generators && (!matches(inputs.reads, space, Tuple::Occurrence, Tuple::Cell) ||
                                !matches(inputs.writes, space, Tuple::Occurrence, Tuple::Cell)))) {
        return {SignedStatus::InvalidInput};
    }
    Builder b;
    const auto present = b.operation(inputs.present, inputs.context, &SignedRelation::restrictContext);
    const auto reference = b.restrict(inputs.reference, inputs.context, present, true);
    const auto id = b.take(signed_detail::identity(present));
    auto ordered = b.unite(reference, id);
    if (!ordered) {
        return {b.status};
    }
    SmallVector<SignedPiece, 0> same;
    for (const auto& piece : ordered->pieces()) {
        const auto sites = space->schema()->sites();
        if (sites[*piece.domain.site].phase->kPipeValue == sites[*piece.range.site].phase->kPipeValue) {
            same.push_back(piece);
        }
    }
    ordered = Access::make(space, Tuple::Occurrence, Tuple::Occurrence, std::move(same));
    auto native = b.unite(
        b.unite(
            b.take(lift(ordered, Kind::Start, Kind::Start)), b.take(lift(ordered, Kind::Completion, Kind::Completion))),
        b.take(lift(ordered, Kind::Start, Kind::Completion)));
    R memory;
    if (inputs.generators) {
        memory = b.restrict(inputs.generators, inputs.context, present, true);
    } else {
        const auto reads = b.restrict(inputs.reads, inputs.context, present, false);
        const auto writes = b.restrict(inputs.writes, inputs.context, present, false);
        memory = b.unite(b.compose(writes, b.inverse(b.unite(reads, writes))), b.compose(reads, b.inverse(writes)));
    }
    const auto extras = b.restrict(inputs.extras, inputs.context, present, true);
    const auto generators =
        b.take(lift(b.intersect(b.unite(memory, extras), reference), Kind::Completion, Kind::Start));
    if (!generators || !present) {
        return {b.status};
    }
    SmallVector<SignedPiece, 0> events;
    for (const auto& piece : present->pieces()) {
        for (auto kind : {Kind::Start, Kind::Completion}) {
            auto event = piece;
            event.range.kind = kind;
            events.push_back(std::move(event));
        }
    }
    const auto eventPresent = Access::make(space, Tuple::Unit, Tuple::Event, std::move(events));
    if (b.status != SignedStatus::Success || !native) { return {b.status}; }
    return finish(std::move(space), inputs.context, eventPresent, native, generators);
}
SignedResult<SignedAnalysisHandle> SignedDemandAnalysis::finish(SignedSpaceHandle space, R context,
    R eventPresent, R native, R generators)
{
    Builder b;
    const auto eventId = b.take(signed_detail::identity(eventPresent));
    const auto step = b.unite(eventId, b.compose(generators, native));
    auto reach = native;
    for (std::size_t pipe = 0; pipe < space->schema()->pipeCount(); ++pipe) {
        reach = b.compose(reach, step);
    }
    const auto strict = b.subtract(reach, eventId);
    // A strict prefix followed by a strict generator/native final edge is
    // equivalent to a strict intermediate path; neither endpoint can be z.
    const auto alternatives = b.compose(strict, b.unite(generators, b.subtract(native, eventId)));
    const auto minimum = b.subtract(generators, b.unite(native, alternatives));
    if (b.status != SignedStatus::Success || !minimum) {
        return {b.status};
    }
    auto result = std::shared_ptr<SignedDemandAnalysis>(new SignedDemandAnalysis());
    result->owner = std::move(space);
    result->admitted = context;
    result->present = eventPresent;
    result->id = eventId;
    result->n = native;
    result->g = generators;
    result->r = reach;
    result->h = strict;
    result->f = minimum;
    SmallVector<PipelineType> pipes;
    for (const auto& site : result->owner->schema()->sites()) {
        if (!llvm::is_contained(pipes, site.phase->kPipeValue)) {
            pipes.push_back(site.phase->kPipeValue);
        }
    }
    for (auto source : pipes) {
        for (auto target : pipes) {
            auto relation = b.take(filterPipes(minimum, source, target));
            if (!relation) { return {b.status}; }
            auto outgoing = SignedSelector::build(relation);
            if (!outgoing.succeeded()) {
                return {outgoing.status};
            }
            auto inverse = b.inverse(relation);
            if (!inverse) { return {b.status}; }
            auto incoming = SignedSelector::build(inverse);
            if (!incoming.succeeded()) {
                return {incoming.status};
            }
            result->selectors.push_back({source, target, incoming.value, outgoing.value});
        }
    }
    return {SignedStatus::Success, SignedAnalysisHandle(result)};
}
SignedResult<SignedAnalysisHandle> SignedDemandAnalysis::adjacentLocalUpper(SignedSpaceHandle space,
    R context, R native, R minimum)
{
    if (!space || !matches(context, space, Tuple::Unit, Tuple::Unit) ||
        !matches(native, space, Tuple::Event, Tuple::Event) ||
        !matches(minimum, space, Tuple::Event, Tuple::Event)) { return {SignedStatus::InvalidInput}; }
    Builder b;
    const auto present = b.take(native->domainSet());
    const auto id = present ? b.take(signed_detail::identity(present)) : R{};
    SmallVector<SignedPiece, 0> completionOrder, localCovers;
    const auto sites = space->schema()->sites();
    auto collect = [sites](R relation, Kind target, SmallVectorImpl<SignedPiece>& output) {
        for (const auto& piece : relation->pieces()) {
            if (piece.domain.kind == Kind::Completion && piece.range.kind == target &&
                sites[*piece.domain.site].phase->kPipeValue == sites[*piece.range.site].phase->kPipeValue) {
                output.push_back(piece);
            }
        }
    };
    collect(native, Kind::Completion, completionOrder);
    collect(minimum, Kind::Start, localCovers);
    auto order = b.subtract(Access::make(space, Tuple::Event, Tuple::Event, std::move(completionOrder)), id);
    order = b.operation(order, context, &SignedRelation::restrictContext);
    auto adjacent = b.subtract(order, b.compose(order, order));
    if (!adjacent) { return {b.status}; }
    SmallVector<SignedPiece, 0> starts(adjacent->pieces().begin(), adjacent->pieces().end());
    for (auto& piece : starts) { piece.range.kind = Kind::Start; }
    adjacent = Access::make(space, Tuple::Event, Tuple::Event, std::move(starts));
    auto local = Access::make(space, Tuple::Event, Tuple::Event, std::move(localCovers));
    auto consumers = b.take(local->rangeSet());
    adjacent = b.operation(adjacent, consumers, &SignedRelation::restrictRange);
    const auto generators = b.unite(minimum, adjacent);
    if (b.status != SignedStatus::Success || !generators || !present) { return {b.status}; }
    return finish(std::move(space), context, present, native, generators);
}
SignedResult<SignedSelectorHandle> SignedDemandAnalysis::outgoing(PipelineType source, PipelineType target) const
{
    for (const auto& pair : selectors) {
        if (pair.source == source && pair.target == target) {
            return {SignedStatus::Success, pair.outgoing};
        }
    }
    return SignedSelector::build(Access::make(owner, Tuple::Event, Tuple::Event, {}));
}
SignedResult<SignedSelectorHandle> SignedDemandAnalysis::incoming(PipelineType source, PipelineType target) const
{
    for (const auto& pair : selectors) {
        if (pair.source == source && pair.target == target) {
            return {SignedStatus::Success, pair.incoming};
        }
    }
    return SignedSelector::build(Access::make(owner, Tuple::Event, Tuple::Event, {}));
}
} // namespace mlir::pto::frontiersynch
