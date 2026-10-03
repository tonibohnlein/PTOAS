// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SignedInternal.h"
namespace mlir::pto::frontiersynch {
using namespace signed_detail;
SignedResult<SignedSelectorHandle> SignedSelector::build(SignedRelationHandle oriented)
{
    if (!oriented || oriented->domain() != SymbolicTuple::Event || oriented->range() != SymbolicTuple::Event) {
        return {SignedStatus::InvalidInput};
    }
    // Inverse(F) o F self-joins TWO independent output tuples on shared input
    // and parameter axes. Full identity includes output tags and residues.
    auto inverse = oriented->inverse();
    auto outputs = inverse.value->compose(oriented);
    if (!outputs.succeeded()) {
        return {outputs.status};
    }
    auto domain = oriented->rangeSet();
    auto diagonal = signed_detail::identity(domain.value);
    if (!diagonal.succeeded()) {
        return {diagonal.status};
    }
    auto different = outputs.value->subtract(diagonal.value);
    if (!different.succeeded()) {
        return {different.status};
    }
    if (!different.value->empty()) {
        return {SignedStatus::NonFunctional};
    }
    auto result = std::shared_ptr<SignedSelector>(new SignedSelector());
    result->relation = oriented;
    for (const auto& piece : oriented->pieces()) {
        SignedSelectorPiece selector;
        const auto shape = layout(oriented->space(), oriented->domain(), oriented->range(), piece).value;
        const auto poly = decode(oriented->space(), oriented->domain(), oriented->range(), piece).value;
        auto single = Access::make(oriented->space(), oriented->domain(), oriented->range(), {piece});
        selector.guard = single->domainSet().value;
        selector.output = piece.range;
        selector.residues.assign(
            piece.residues.begin() + shape.domain, piece.residues.begin() + shape.domain + shape.range);
        for (unsigned coordinate = 0; coordinate < shape.range; ++coordinate) {
            SmallVector<unsigned> retained;
            for (unsigned i = 0; i < shape.domain; ++i) {
                retained.push_back(i);
            }
            retained.push_back(shape.domain + coordinate);
            for (unsigned i = 0; i < shape.parameters; ++i) {
                retained.push_back(shape.domain + shape.range + i);
            }
            auto projected = poly;
            projected.project(retained);
            SignedOutputCoordinate expression;
            const auto outputAxis = std::int64_t(shape.domain) + 1;
            for (const auto& [direction, constant] : projected.bounds) {
                auto [a, b] = direction;
                const bool lower = a == -outputAxis || b == -outputAxis;
                const bool upper = a == outputAxis || b == outputAxis;
                if (!lower && !upper) {
                    continue;
                }
                const auto other = (a == outputAxis || a == -outputAxis) ? b : a;
                SignedExpressionTerm term;
                term.constant = lower ? -constant : constant;
                if (other) {
                    const auto index = unsigned((other > 0 ? other : -other) - 1);
                    term.axis = index < shape.domain ? SignedAxis{SignedAxisRole::Domain, index} :
                                                       SignedAxis{SignedAxisRole::Parameter, index - shape.domain - 1};
                    term.sign = (lower ? 1 : -1) * (other > 0 ? 1 : -1);
                }
                (lower ? expression.lower : expression.upper).push_back(std::move(term));
            }
            if (expression.lower.empty() || expression.upper.empty()) {
                return {SignedStatus::UnboundedOutput};
            }
            selector.coordinates.push_back(std::move(expression));
        }
        result->data.push_back(std::move(selector));
    }
    return {SignedStatus::Success, SignedSelectorHandle(result)};
}
SignedResult<std::optional<SymbolicEvent>> SignedSelector::evaluate(
    const SymbolicEvent& input, ArrayRef<BigInt> parameters) const
{
    const auto source = point(input);
    auto count = active(space(), SymbolicTuple::Event, source.tag);
    if (!count.succeeded() || count.value != source.coordinates.size()) {
        return {SignedStatus::InvalidEvent};
    }
    if (parameters.size() != space()->schema()->parameters().size()) {
        return {SignedStatus::InvalidBinding};
    }
    SmallVector<BigInt> coordinates, values;
    for (const auto& value : source.coordinates) {
        coordinates.push_back(llvm::floorDiv(value, space()->period()));
    }
    for (const auto& value : parameters) {
        values.push_back(llvm::floorDiv(value, space()->period()));
    }
    for (const auto& piece : data) {
        auto admitted = piece.guard->contains({}, source, parameters);
        if (!admitted.succeeded()) {
            return {admitted.status};
        }
        if (!admitted.value) {
            continue;
        }
        SymbolicEvent result;
        result.site = *piece.output.site;
        result.kind = *piece.output.kind;
        for (unsigned i = 0; i < piece.coordinates.size(); ++i) {
            const auto& expression = piece.coordinates[i];
            auto endpoint = [&](ArrayRef<SignedExpressionTerm> terms, bool maximum) {
                std::optional<BigInt> value;
                for (const auto& term : terms) {
                    BigInt current = term.constant;
                    if (term.axis) {
                        current +=
                            term.sign * (term.axis->role == SignedAxisRole::Domain ? coordinates[term.axis->index] :
                                                                                     values[term.axis->index]);
                    }
                    if (!value || (maximum ? current > *value : current < *value)) {
                        value = std::move(current);
                    }
                }
                return value;
            };
            auto lower = endpoint(expression.lower, true), upper = endpoint(expression.upper, false);
            if (!lower || !upper) {
                return {SignedStatus::UnboundedOutput};
            }
            if (*lower != *upper) {
                return {SignedStatus::NonFunctional};
            }
            result.coordinates.push_back(space()->period() * *lower + piece.residues[i]);
        }
        // Functionality was established over complete outputs, so overlaps agree.
        return {SignedStatus::Success, std::move(result)};
    }
    return {SignedStatus::Success, std::nullopt};
}
llvm::StringRef signedDiagnostic(SignedStatus status)
{
    switch (status) {
        case SignedStatus::Success:
            return "success";
        case SignedStatus::InvalidInput:
            return "invalid-input";
        case SignedStatus::InvalidBinding:
            return "invalid-binding";
        case SignedStatus::InvalidEvent:
            return "invalid-event";
        case SignedStatus::AbsentEndpoint:
            return "absent-endpoint";
        case SignedStatus::InadmissibleContext:
            return "inadmissible-context";
        case SignedStatus::NonFunctional:
            return "non-functional";
        case SignedStatus::UnboundedOutput:
            return "unbounded-output";
        case SignedStatus::RepresentationLimit:
            return "representation-limit";
    }
    return "unknown-status";
}
SignedResult<BoundSignedHandle> BoundSignedAnalysis::bind(
    SignedAnalysisHandle analysis, ArrayRef<ImmutableParameterBinding> bindings)
{
    if (!analysis || !analysis->space()) {
        return {SignedStatus::InvalidInput};
    }
    auto context = decodeSymbolicBindings(analysis->space()->schema(), bindings);
    if (failed(context)) {
        return {SignedStatus::InvalidBinding};
    }
    auto admitted = analysis->context()->contains({}, {}, context->values);
    if (!admitted.succeeded()) {
        return {admitted.status};
    }
    if (!admitted.value) {
        return {SignedStatus::InadmissibleContext};
    }
    context->admitted = true;
    auto result = std::shared_ptr<BoundSignedAnalysis>(new BoundSignedAnalysis());
    result->owner = std::move(analysis);
    result->context = std::make_shared<SymbolicContext>(std::move(*context));
    return {SignedStatus::Success, BoundSignedHandle(result)};
}
SignedResult<bool> BoundSignedAnalysis::contains(const SymbolicEvent& event) const
{
    auto result = owner->presence()->contains({}, point(event), context->values);
    if (result.status == SignedStatus::InvalidInput) {
        result.status = SignedStatus::InvalidEvent;
    }
    return result;
}
SignedResult<bool> BoundSignedAnalysis::pair(
    SignedRelationHandle relation, const SymbolicEvent& source, const SymbolicEvent& target) const
{
    auto a = contains(source), b = contains(target);
    if (!a.succeeded()) {
        return a;
    }
    if (!b.succeeded()) {
        return b;
    }
    if (!a.value || !b.value) {
        return {SignedStatus::AbsentEndpoint};
    }
    return relation->contains(point(source), point(target), context->values);
}
SignedResult<bool> BoundSignedAnalysis::reaches(const SymbolicEvent& source, const SymbolicEvent& target) const
{
    return pair(owner->reachability(), source, target);
}
SignedResult<bool> BoundSignedAnalysis::minimumDemand(const SymbolicEvent& source, const SymbolicEvent& target) const
{
    return pair(owner->minimum(), source, target);
}
SignedResult<std::optional<SymbolicEvent>> BoundSignedAnalysis::incoming(
    const SymbolicEvent& target, PipelineType source) const
{
    auto present = contains(target);
    if (!present.succeeded() || target.kind != PeriodicEventKind::Start) {
        return {SignedStatus::InvalidEvent};
    }
    if (!present.value) {
        return {SignedStatus::AbsentEndpoint};
    }
    auto selector = owner->incoming(source, owner->space()->schema()->sites()[target.site].phase->kPipeValue);
    return selector.succeeded() ? selector.value->evaluate(target, context->values) :
                                  SignedResult<std::optional<SymbolicEvent>>{selector.status};
}
SignedResult<std::optional<SymbolicEvent>> BoundSignedAnalysis::outgoing(
    const SymbolicEvent& source, PipelineType target) const
{
    auto present = contains(source);
    if (!present.succeeded() || source.kind != PeriodicEventKind::Completion) {
        return {SignedStatus::InvalidEvent};
    }
    if (!present.value) {
        return {SignedStatus::AbsentEndpoint};
    }
    auto selector = owner->outgoing(owner->space()->schema()->sites()[source.site].phase->kPipeValue, target);
    return selector.succeeded() ? selector.value->evaluate(source, context->values) :
                                  SignedResult<std::optional<SymbolicEvent>>{selector.status};
}
} // namespace mlir::pto::frontiersynch
