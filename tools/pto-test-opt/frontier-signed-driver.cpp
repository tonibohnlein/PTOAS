// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "frontier-signed-driver.h"
#include "frontier-periodic-driver.h"
namespace frontier_test {
namespace {
namespace fs = mlir::pto::frontiersynch;
using llvm::json::Array;
using llvm::json::Object;
using Tuple = fs::SymbolicTuple;
using R = fs::SignedRelationHandle;
using BigInt = llvm::DynamicAPInt;
mlir::FailureOr<fs::SignedTag> tag(const Array* value)
{
    if (!value || value->size() > 2) {
        return mlir::failure();
    }
    fs::SignedTag result;
    if (!value->empty()) {
        auto site = index((*value)[0], 64);
        if (mlir::failed(site)) {
            return mlir::failure();
        }
        result.site = *site;
    }
    if (value->size() == 2) {
        auto kind = index((*value)[1], 3);
        if (mlir::failed(kind)) {
            return mlir::failure();
        }
        result.kind = static_cast<fs::PeriodicEventKind>(*kind);
    }
    return result;
}
mlir::FailureOr<llvm::SmallVector<BigInt>> numbers(const Array* value)
{
    if (!value || value->size() > 32) {
        return mlir::failure();
    }
    llvm::SmallVector<BigInt> result;
    for (const auto& number : *value) {
        auto parsed = integer(number);
        if (mlir::failed(parsed)) {
            return mlir::failure();
        }
        result.push_back(*parsed);
    }
    return result;
}
fs::SignedResult<R> read(const Array* input, fs::SignedSpaceHandle space, Tuple domain, Tuple range)
{
    if (!input || input->size() > 512) {
        return {fs::SignedStatus::InvalidInput};
    }
    llvm::SmallVector<fs::SignedPiece> pieces;
    for (const auto& value : *input) {
        const auto* object = value.getAsObject();
        if (!object) {
            return {fs::SignedStatus::InvalidInput};
        }
        auto source = tag(object->getArray("domain")), target = tag(object->getArray("range"));
        auto residues = numbers(object->getArray("residues"));
        const auto* atoms = object->getArray("atoms");
        auto locals = object->getInteger("locals").value_or(0);
        if (mlir::failed(source) || mlir::failed(target) || mlir::failed(residues) || !atoms || atoms->size() > 128 ||
            locals < 0 || locals > 16) {
            return {fs::SignedStatus::InvalidInput};
        }
        fs::SignedPiece piece;
        piece.domain = *source;
        piece.range = *target;
        piece.residues = *residues;
        piece.locals = locals;
        for (const auto& item : *atoms) {
            const auto* atom = item.getAsObject();
            const auto* terms = atom ? atom->getArray("terms") : nullptr;
            const auto* constant = atom ? atom->get("bound") : nullptr;
            if (!terms || !constant || terms->size() > 3) {
                return {fs::SignedStatus::InvalidInput};
            }
            auto bound = integer(*constant);
            if (mlir::failed(bound)) {
                return {fs::SignedStatus::InvalidInput};
            }
            fs::SignedAtom parsed;
            parsed.bound = *bound;
            for (const auto& entry : *terms) {
                const auto* term = entry.getAsArray();
                if (!term || term->size() != 3) {
                    return {fs::SignedStatus::InvalidInput};
                }
                auto role = (*term)[0].getAsString();
                auto axis = index((*term)[1], 64);
                auto sign = (*term)[2].getAsInteger();
                if (!role || mlir::failed(axis) || !sign || *sign < -2 || *sign > 2) {
                    return {fs::SignedStatus::InvalidInput};
                }
                const auto kind = *role == "d" ? fs::SignedAxisRole::Domain :
                                  *role == "r" ? fs::SignedAxisRole::Range :
                                  *role == "p" ? fs::SignedAxisRole::Parameter :
                                  *role == "l" ? fs::SignedAxisRole::Local :
                                                 static_cast<fs::SignedAxisRole>(99);
                parsed.terms.push_back({{kind, unsigned(*axis)}, int(*sign)});
            }
            piece.atoms.push_back(std::move(parsed));
        }
        pieces.push_back(std::move(piece));
    }
    auto result = fs::SignedRelation::import(space, domain, range, pieces);
    pieces.clear(); // All subsequent operations must use the owned snapshot.
    return result;
}
Tuple tuple(llvm::StringRef name)
{
    return name == "unit"       ? Tuple::Unit :
           name == "occurrence" ? Tuple::Occurrence :
           name == "event"      ? Tuple::Event :
           name == "cell"       ? Tuple::Cell :
                                  static_cast<Tuple>(99);
}
mlir::FailureOr<fs::SignedPoint> readPoint(const Object* object)
{
    if (!object) {
        return mlir::failure();
    }
    auto kind = tag(object->getArray("tag"));
    auto coords = numbers(object->getArray("coords"));
    if (mlir::failed(kind) || mlir::failed(coords)) {
        return mlir::failure();
    }
    return fs::SignedPoint{*kind, *coords};
}
mlir::FailureOr<fs::SymbolicEvent> event(const llvm::json::Value& value)
{
    const auto* object = value.getAsObject();
    if (!object) {
        return mlir::failure();
    }
    auto site = object->getInteger("site"), kind = object->getInteger("kind");
    auto coords = numbers(object->getArray("coords"));
    if (!site || *site < 0 || *site > 64 || !kind || *kind < 0 || *kind > 2 || mlir::failed(coords)) {
        return mlir::failure();
    }
    return fs::SymbolicEvent{std::size_t(*site), *coords, static_cast<fs::PeriodicEventKind>(*kind)};
}
Array encode(const fs::SignedResult<bool>& result)
{
    return Array{
        fs::signedDiagnostic(result.status),
        result.succeeded() ? llvm::json::Value(result.value) : llvm::json::Value(nullptr)};
}
Array encode(const fs::SignedResult<std::optional<fs::SymbolicEvent>>& result)
{
    llvm::json::Value value(nullptr);
    if (result.succeeded() && result.value) {
        Array coords;
        for (const auto& number : result.value->coordinates) {
            coords.push_back(decimal(number));
        }
        value = Object{
            {"site", std::int64_t(result.value->site)},
            {"kind", std::int64_t(result.value->kind)},
            {"coords", std::move(coords)}};
    }
    return Array{fs::signedDiagnostic(result.status), std::move(value)};
}
bool interchange(
    R relation, const fs::SignedPoint& source, const fs::SignedPoint& target, llvm::ArrayRef<BigInt> parameters,
    bool expected)
{
    auto converted = relation->toSymbolic();
    if (mlir::failed(converted)) {
        return false;
    }
    llvm::SmallVector<BigInt> values;
    auto append = [&](Tuple role, const fs::SignedPoint& point) {
        if (role == Tuple::Event || role == Tuple::Occurrence) {
            values.push_back(BigInt(std::int64_t(*point.tag.site)));
        }
        values.append(point.coordinates);
        if (role == Tuple::Event || role == Tuple::Occurrence) {
            for (unsigned i = point.coordinates.size(); i < relation->space()->schema()->coordinateDepth(); ++i) {
                values.push_back(BigInt(0));
            }
        }
        if (role == Tuple::Event) {
            values.push_back(BigInt(*point.tag.kind == fs::PeriodicEventKind::Start ? 0 : 1));
        }
    };
    append(relation->domain(), source);
    append(relation->range(), target);
    values.append(parameters.begin(), parameters.end());
    return converted->relation().containsPoint(values) == expected;
}
mlir::FailureOr<Object> arithmetic(const Object& object, fs::SignedSpaceHandle space)
{
    auto domain = tuple(object.getString("domain").value_or("cell"));
    auto range = tuple(object.getString("range").value_or("cell"));
    auto left = read(object.getArray("left"), space, domain, range);
    if (!left.succeeded()) {
        return Object{{"status", fs::signedDiagnostic(left.status)}};
    }
    const auto operation = object.getString("operation").value_or("import");
    auto result = left;
    fs::SignedResult<R> right;
    if (operation == "inverse") {
        result = left.value->inverse();
    } else if (operation == "domain") {
        result = left.value->domainSet();
    } else if (operation == "range") {
        result = left.value->rangeSet();
    } else if (operation == "selector") {
        auto selector = fs::SignedSelector::build(left.value);
        if (!selector.succeeded()) {
            return Object{{"status", fs::signedDiagnostic(selector.status)}};
        }
        left.value.reset();
        Array answers;
        if (const auto* queries = object.getArray("queries")) {
            for (const auto& value : *queries) {
                const auto* query = value.getAsObject();
                if (!query) {
                    return mlir::failure();
                }
                auto input = readPoint(query->getObject("source"));
                auto parameters = numbers(query->getArray("parameters"));
                if (mlir::failed(input) || mlir::failed(parameters) || !input->tag.site || !input->tag.kind) {
                    return mlir::failure();
                }
                answers.push_back(encode(
                    selector.value->evaluate({*input->tag.site, input->coordinates, *input->tag.kind}, *parameters)));
            }
        }
        return Object{{"status", "success"}, {"answers", std::move(answers)}};
    } else if (operation != "import") {
        auto otherSpace = object.getBoolean("foreign_space").value_or(false) ?
                              fs::SignedSpace::create(space->schema(), space->period()).value :
                              space;
        right = read(
            object.getArray("right"), otherSpace,
            tuple(object.getString("right_domain").value_or(object.getString("domain").value_or("cell"))),
            tuple(object.getString("right_range").value_or(object.getString("range").value_or("cell"))));
        if (!right.succeeded()) {
            return Object{{"status", fs::signedDiagnostic(right.status)}};
        }
        if (operation == "union") {
            result = left.value->unite(right.value);
        } else if (operation == "intersection") {
            result = left.value->intersect(right.value);
        } else if (operation == "subtract") {
            result = left.value->subtract(right.value);
        } else if (operation == "compose") {
            result = left.value->compose(right.value);
        } else if (operation == "context") {
            result = left.value->restrictContext(right.value);
        } else {
            return mlir::failure();
        }
    }
    if (!result.succeeded()) {
        return Object{{"status", fs::signedDiagnostic(result.status)}};
    }
    bool probes = true;
    Array answers;
    if (const auto* queries = object.getArray("queries")) {
        if (queries->size() > 4096) {
            return mlir::failure();
        }
        for (const auto& value : *queries) {
            const auto* query = value.getAsObject();
            if (!query) {
                return mlir::failure();
            }
            auto a = readPoint(query->getObject("source")), b = readPoint(query->getObject("target"));
            auto parameters = numbers(query->getArray("parameters"));
            if (mlir::failed(a) || mlir::failed(b) || mlir::failed(parameters)) {
                return mlir::failure();
            }
            auto answer = result.value->contains(*a, *b, *parameters);
            answers.push_back(encode(answer));
            if (object.getBoolean("interchange").value_or(false) && answer.succeeded()) {
                probes &= interchange(result.value, *a, *b, *parameters, answer.value);
            }
        }
    }
    return Object{
        {"status", "success"},
        {"answers", std::move(answers)},
        {"empty", result.value->empty()},
        {"pieces", std::int64_t(result.value->pieces().size())},
        {"probes", probes}};
}
} // namespace
mlir::FailureOr<Object> analyzeSigned(const Object& object, fs::SymbolicSchemaHandle schema, mlir::MLIRContext& context)
{
    const auto* encodedPeriod = object.get("period");
    auto period = encodedPeriod ? integer(*encodedPeriod) : mlir::FailureOr<BigInt>(BigInt(1));
    if (mlir::failed(period)) {
        return mlir::failure();
    }
    auto space = fs::SignedSpace::create(schema, *period);
    if (!space.succeeded()) {
        return Object{{"status", fs::signedDiagnostic(space.status)}};
    }
    if (object.getBoolean("arithmetic").value_or(false)) {
        return arithmetic(object, space.value);
    }
    fs::SignedInputs inputs;
    auto admitted = read(object.getArray("context"), space.value, Tuple::Unit, Tuple::Unit);
    auto present = read(object.getArray("present"), space.value, Tuple::Unit, Tuple::Occurrence);
    auto reference = read(object.getArray("reference"), space.value, Tuple::Occurrence, Tuple::Occurrence);
    auto reads = read(object.getArray("reads"), space.value, Tuple::Occurrence, Tuple::Cell);
    auto writes = read(object.getArray("writes"), space.value, Tuple::Occurrence, Tuple::Cell);
    auto extras = read(object.getArray("extras"), space.value, Tuple::Occurrence, Tuple::Occurrence);
    for (const auto& value : {admitted, present, reference, reads, writes, extras}) {
        if (!value.succeeded()) {
            return Object{{"valid", false}, {"status", fs::signedDiagnostic(value.status)}};
        }
    }
    inputs = {admitted.value, present.value, reference.value, reads.value, writes.value, extras.value, {}};
    if (const auto* generators = object.getArray("generators")) {
        auto parsed = read(generators, space.value, Tuple::Occurrence, Tuple::Occurrence);
        if (!parsed.succeeded()) {
            return Object{{"valid", false}, {"status", fs::signedDiagnostic(parsed.status)}};
        }
        inputs.generators = parsed.value;
    }
    auto analysis = fs::SignedDemandAnalysis::build(space.value, inputs);
    if (!analysis.succeeded()) {
        return Object{{"valid", false}, {"status", fs::signedDiagnostic(analysis.status)}};
    }
    inputs = {};
    Array results;
    const auto* contexts = object.getArray("contexts");
    if (!contexts || contexts->size() > 16) {
        return mlir::failure();
    }
    for (const auto& row : *contexts) {
        const auto* entry = row.getAsObject();
        const auto* bindings = entry ? entry->getArray("bindings") : nullptr;
        const auto* events = entry ? entry->getArray("events") : nullptr;
        if (!bindings || bindings->size() != schema->parameters().size() || !events || events->size() > 64) {
            return mlir::failure();
        }
        llvm::SmallVector<fs::ImmutableParameterBinding> values;
        for (unsigned i = 0; i < bindings->size(); ++i) {
            auto number = integer((*bindings)[i]);
            if (mlir::failed(number)) {
                return mlir::failure();
            }
            const auto type = schema->parameters()[i].getType();
            const auto bits = type.isIndex() ? 64 : mlir::cast<mlir::IntegerType>(type).getWidth();
            values.push_back(
                {schema->parameters()[i], mlir::IntegerAttr::get(type, llvm::APInt(bits, decimal(*number), 10))});
        }
        auto bound = fs::BoundSignedAnalysis::bind(analysis.value, values);
        Object result{{"status", fs::signedDiagnostic(bound.status)}};
        if (!bound.succeeded()) {
            results.push_back(std::move(result));
            continue;
        }
        bool probes = fs::BoundSignedAnalysis::bind({}, values).status == fs::SignedStatus::InvalidInput;
        if (!values.empty()) {
            auto wrong = values;
            wrong.front().result = mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 7), 0);
            probes &= fs::BoundSignedAnalysis::bind(analysis.value, wrong).status == fs::SignedStatus::InvalidBinding;
        }
        values.clear();
        llvm::SmallVector<fs::SymbolicEvent> points;
        Array presence, reaches, strict, minimum, incoming, outgoing;
        for (const auto& encoded : *events) {
            auto parsed = event(encoded);
            if (mlir::failed(parsed)) {
                return mlir::failure();
            }
            points.push_back(*parsed);
            presence.push_back(encode(bound.value->contains(*parsed)));
        }
        for (const auto& a : points) {
            Array reachRow, strictRow, minimumRow, incomingRow, outgoingRow;
            for (const auto& b : points) {
                auto reach = bound.value->reaches(a, b), retained = bound.value->minimumDemand(a, b);
                reachRow.push_back(encode(reach));
                auto h = reach;
                if (reach.succeeded()) {
                    h = analysis.value->strictReachability()->contains(
                        {{a.site, a.kind}, a.coordinates}, {{b.site, b.kind}, b.coordinates},
                        bound.value->boundContext()->values);
                }
                strictRow.push_back(encode(h));
                minimumRow.push_back(encode(retained));
                if (object.getBoolean("interchange").value_or(false) && reach.succeeded()) {
                    const fs::SignedPoint pa{{a.site, a.kind}, a.coordinates}, pb{{b.site, b.kind}, b.coordinates};
                    probes &= interchange(
                        analysis.value->reachability(), pa, pb, bound.value->boundContext()->values, reach.value);
                    probes &= interchange(
                        analysis.value->minimum(), pa, pb, bound.value->boundContext()->values, retained.value);
                }
            }
            if (const auto* pipes = object.getArray("selector_pipes")) {
                for (const auto& encoded : *pipes) {
                    auto pipe = index(encoded, 9);
                    if (mlir::failed(pipe)) {
                        return mlir::failure();
                    }
                    incomingRow.push_back(
                        encode(bound.value->incoming(a, static_cast<mlir::pto::PipelineType>(*pipe))));
                    outgoingRow.push_back(
                        encode(bound.value->outgoing(a, static_cast<mlir::pto::PipelineType>(*pipe))));
                }
            }
            reaches.push_back(std::move(reachRow));
            strict.push_back(std::move(strictRow));
            minimum.push_back(std::move(minimumRow));
            incoming.push_back(std::move(incomingRow));
            outgoing.push_back(std::move(outgoingRow));
        }
        result["probes"] = probes;
        result["presence"] = std::move(presence);
        result["reach"] = std::move(reaches);
        result["strict"] = std::move(strict);
        result["minimum"] = std::move(minimum);
        result["incoming"] = std::move(incoming);
        result["outgoing"] = std::move(outgoing);
        results.push_back(std::move(result));
    }
    return Object{
        {"valid", true},
        {"contexts", std::move(results)},
        {"sizes", Array{
                      std::int64_t(analysis.value->generators()->pieces().size()),
                      std::int64_t(analysis.value->reachability()->pieces().size()),
                      std::int64_t(analysis.value->strictReachability()->pieces().size()),
                      std::int64_t(analysis.value->minimum()->pieces().size())}}};
}
} // namespace frontier_test
