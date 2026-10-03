// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bounded raw integer-relation transport; physical semantics live in the independent Python oracle.
#include "frontier-periodic-driver.h"
#include "frontier-signed-driver.h"
#include "PTO/Transforms/FrontierSynch/SymbolicDemandAnalysis.h"
#include "mlir/IR/Block.h"

namespace frontier_test {
namespace {
namespace fs = mlir::pto::frontiersynch;
namespace pb = mlir::presburger;
using llvm::json::Array;
using llvm::json::Object;
using Tuple = fs::SymbolicTuple;
using BigInt = llvm::DynamicAPInt;

mlir::FailureOr<fs::SymbolicPrimitive> readRelation(
    const Array* rows, fs::SymbolicSchemaHandle schema, Tuple source, Tuple target)
{
    if (!rows || rows->size() > 64) {
        return mlir::failure();
    }
    auto space = schema->space(source, target);
    if (mlir::failed(space)) {
        return mlir::failure();
    }
    auto relation = pb::PresburgerRelation::getEmpty(*space);
    for (const auto& row : *rows) {
        const auto* object = row.getAsObject();
        if (!object) {
            return mlir::failure();
        }
        auto locals = object->getInteger("locals");
        const auto* equalities = object->getArray("eq");
        const auto* inequalities = object->getArray("ge");
        if (!locals || *locals < 0 || *locals > 8 || !equalities || !inequalities ||
            equalities->size() + inequalities->size() > 64) {
            return mlir::failure();
        }
        auto pieceSpace = *space;
        pieceSpace.insertVar(pb::VarKind::Local, 0, *locals);
        pb::IntegerRelation piece(pieceSpace);
        for (bool equality : {true, false}) {
            for (const auto& encoded : equality ? *equalities : *inequalities) {
                const auto* values = encoded.getAsArray();
                if (!values || values->size() != piece.getNumCols()) {
                    return mlir::failure();
                }
                llvm::SmallVector<BigInt> coefficients;
                for (const auto& value : *values) {
                    auto parsed = integer(value);
                    if (mlir::failed(parsed)) {
                        return mlir::failure();
                    }
                    coefficients.push_back(*parsed);
                }
                if (equality) {
                    piece.addEquality(coefficients);
                } else {
                    piece.addInequality(coefficients);
                }
            }
        }
        relation.unionInPlace(piece);
    }
    return fs::SymbolicPrimitive::import(schema, source, target, schema->parameters(), relation);
}

mlir::FailureOr<fs::SymbolicEvent> readEvent(const llvm::json::Value& value)
{
    const auto* object = value.getAsObject();
    if (!object) {
        return mlir::failure();
    }
    auto site = object->getInteger("site"), kind = object->getInteger("kind");
    const auto* coordinates = object->getArray("coords");
    if (!site || *site < 0 || !kind || *kind < 0 || *kind > 2 || !coordinates || coordinates->size() > 4) {
        return mlir::failure();
    }
    fs::SymbolicEvent result;
    result.site = *site;
    result.kind = static_cast<fs::PeriodicEventKind>(*kind);
    for (const auto& coordinate : *coordinates) {
        auto parsed = integer(coordinate);
        if (mlir::failed(parsed)) {
            return mlir::failure();
        }
        result.coordinates.push_back(*parsed);
    }
    return result;
}
llvm::json::Value answer(mlir::FailureOr<bool> value)
{
    return mlir::succeeded(value) ? llvm::json::Value(*value) : llvm::json::Value(nullptr);
}

bool structuralProbes(fs::SymbolicAnalysisHandle analysis, const fs::SymbolicInputs& inputs)
{
    auto schema = analysis->schema();
    auto other = fs::SymbolicDemandAnalysis::build(schema, inputs);
    auto evaluator = fs::SymbolicEvaluator::uniform(analysis);
    if (mlir::failed(other) || mlir::failed(evaluator) || evaluator->cachedNodes() != 0 ||
        mlir::succeeded(evaluator->materialize((*other)->identity())) ||
        mlir::succeeded(fs::SymbolicEvaluator::uniform(std::make_shared<fs::SymbolicDemandAnalysis>()))) {
        return false;
    }
    auto bad = inputs;
    bad.present = inputs.context;
    if (mlir::succeeded(fs::SymbolicDemandAnalysis::build(schema, bad))) {
        return false;
    }
    if (schema->cellTypes().empty()) {
        auto cell = fs::SymbolicPrimitive::import(
            schema, Tuple::Unit, Tuple::Cell, schema->parameters(), inputs.context.relation());
        if (mlir::failed(cell)) {
            return false;
        }
        bad = inputs;
        bad.context = *cell;
        if (mlir::succeeded(fs::SymbolicDemandAnalysis::build(schema, bad))) {
            return false;
        }
    }
    auto raw = inputs.present.relation();
    auto noIds =
        pb::PresburgerSpace::getRelationSpace(0, *schema->arity(Tuple::Occurrence), schema->parameters().size());
    raw.setSpace(noIds);
    if (mlir::failed(
            fs::SymbolicPrimitive::import(schema, Tuple::Unit, Tuple::Occurrence, schema->parameters(), raw))) {
        return false;
    }
    // Same-arity source/range tokens are deliberately different, so wrong-role IDs fail.
    auto wrong = schema->space(Tuple::Occurrence, Tuple::Unit);
    noIds.setId(pb::VarKind::Range, 0, wrong->getId(pb::VarKind::Domain, 0));
    raw.setSpace(noIds);
    if (mlir::succeeded(
            fs::SymbolicPrimitive::import(schema, Tuple::Unit, Tuple::Occurrence, schema->parameters(), raw))) {
        return false;
    }
    if (schema->parameters().size() > 1) {
        llvm::SmallVector<mlir::Value> reversed(schema->parameters().rbegin(), schema->parameters().rend());
        if (mlir::succeeded(
                fs::SymbolicPrimitive::import(schema, Tuple::Unit, Tuple::Unit, reversed, inputs.context.relation()))) {
            return false;
        }
    }
    auto conflict = pb::PresburgerRelation::getEmpty(*schema->space(Tuple::Unit, Tuple::Occurrence));
    conflict.unionInPlace(pb::IntegerRelation(noIds));
    if (mlir::succeeded(
            fs::SymbolicPrimitive::import(schema, Tuple::Unit, Tuple::Occurrence, schema->parameters(), conflict))) {
        return false;
    }
    const auto nodes = analysis->nodes();
    const auto strict = analysis->strictReachability().index();
    return nodes.size() == analysis->preparationNodes() + schema->pipeCount() + 6 &&
           nodes[nodes.size() - 3].operation == fs::SymbolicRelationOp::Compose &&
           nodes[nodes.size() - 3].first == strict && nodes[nodes.size() - 3].second == strict;
}
} // namespace

mlir::FailureOr<Object> analyzeSymbolic(
    const Object& object, llvm::ArrayRef<const mlir::pto::CompoundInstanceElement*> anchors, mlir::MLIRContext& context)
{
    const auto* depths = object.getArray("depths");
    const auto* types = object.getArray("parameter_types");
    const auto* contexts = object.getArray("contexts");
    auto cellAxes = object.getInteger("cell_axes");
    if (!depths || depths->size() != anchors.size() || !types || types->size() > 8 || !contexts ||
        contexts->size() > 16 || !cellAxes || *cellAxes < 0 || *cellAxes > 4) {
        return mlir::failure();
    }
    mlir::Block arguments;
    llvm::SmallVector<mlir::Value> parameters;
    for (const auto& type : *types) {
        auto name = type.getAsString();
        if (!name || (*name != "i4096" && *name != "i1" && *name != "u128" && *name != "index")) {
            return mlir::failure();
        }
        mlir::Type parsed = *name == "index" ?
                                mlir::Type(mlir::IndexType::get(&context)) :
                                mlir::Type(
                                    mlir::IntegerType::get(
                                        &context,
                                        *name == "i1"   ? 1 :
                                        *name == "u128" ? 128 :
                                                          4096,
                                        *name == "u128" ? mlir::IntegerType::Unsigned : mlir::IntegerType::Signless));
        parameters.push_back(arguments.addArgument(parsed, mlir::UnknownLoc::get(&context)));
    }
    llvm::SmallVector<fs::SymbolicSite> sites;
    for (std::size_t i = 0; i < anchors.size(); ++i) {
        auto depth = index((*depths)[i], 4);
        if (mlir::failed(depth)) {
            return mlir::failure();
        }
        fs::SymbolicSite site;
        site.phase = anchors[i];
        for (std::size_t j = 0; j < *depth; ++j) {
            site.coordinates.push_back(
                arguments.addArgument(mlir::IndexType::get(&context), mlir::UnknownLoc::get(&context)));
        }
        sites.push_back(std::move(site));
    }
    llvm::SmallVector<mlir::Type> cellTypes(*cellAxes, mlir::IndexType::get(&context));
    auto schema = fs::SymbolicSchema::create(sites, parameters, cellTypes, 29);
    if (mlir::failed(schema)) {
        return mlir::failure();
    }
    if (object.getBoolean("signed").value_or(false)) {
        return analyzeSigned(object, *schema, context);
    }
    fs::SymbolicInputs inputs;
    auto admitted = readRelation(object.getArray("context"), *schema, Tuple::Unit, Tuple::Unit);
    auto present = readRelation(object.getArray("present"), *schema, Tuple::Unit, Tuple::Occurrence);
    auto reference = readRelation(object.getArray("reference"), *schema, Tuple::Occurrence, Tuple::Occurrence);
    auto reads = readRelation(object.getArray("reads"), *schema, Tuple::Occurrence, Tuple::Cell);
    auto writes = readRelation(object.getArray("writes"), *schema, Tuple::Occurrence, Tuple::Cell);
    auto extras = readRelation(object.getArray("extras"), *schema, Tuple::Occurrence, Tuple::Occurrence);
    if (mlir::failed(admitted) || mlir::failed(present) || mlir::failed(reference) || mlir::failed(reads) ||
        mlir::failed(writes) || mlir::failed(extras)) {
        return mlir::failure();
    }
    inputs.context = *admitted;
    inputs.present = *present;
    inputs.reference = *reference;
    inputs.reads = *reads;
    inputs.writes = *writes;
    inputs.extras = *extras;
    if (const auto* generators = object.getArray("generators")) {
        auto parsed = readRelation(generators, *schema, Tuple::Occurrence, Tuple::Occurrence);
        if (mlir::failed(parsed)) {
            return mlir::failure();
        }
        inputs.generators = *parsed;
    }
    auto analysis = fs::SymbolicDemandAnalysis::build(*schema, inputs);
    if (mlir::failed(analysis)) {
        return Object{{"valid", false}};
    }
    Object result{
        {"valid", true},
        {"probes", structuralProbes(*analysis, inputs)},
        {"nodes", static_cast<std::int64_t>((*analysis)->nodes().size())},
        {"prepared", static_cast<std::int64_t>((*analysis)->preparationNodes())}};
    auto uniformEvaluator = fs::SymbolicEvaluator::uniform(*analysis);
    std::optional<fs::MaterializedSymbolicRelation> uniformReach, uniformMinimum;
    if (object.getBoolean("uniform").value_or(false)) {
        auto reach = uniformEvaluator->materialize((*analysis)->reachability());
        auto minimum = uniformEvaluator->materialize((*analysis)->minimum());
        if (mlir::failed(reach) || mlir::failed(minimum)) {
            return mlir::failure();
        }
        uniformReach = std::move(*reach);
        uniformMinimum = std::move(*minimum);
    }
    // The analysis owns imported primitive snapshots.
    inputs = fs::SymbolicInputs();
    Array evaluated;
    for (const auto& row : *contexts) {
        const auto* entry = row.getAsObject();
        const auto* bindings = entry ? entry->getArray("bindings") : nullptr;
        const auto* events = entry ? entry->getArray("events") : nullptr;
        if (!bindings || !events || bindings->size() != parameters.size() || events->size() > 48) {
            return mlir::failure();
        }
        llvm::SmallVector<fs::ImmutableParameterBinding> values;
        for (std::size_t i = 0; i < parameters.size(); ++i) {
            auto number = integer((*bindings)[i]);
            if (mlir::failed(number)) {
                return mlir::failure();
            }
            auto type = parameters[i].getType();
            unsigned width = type.isIndex() ? 64 : mlir::cast<mlir::IntegerType>(type).getWidth();
            values.push_back({parameters[i], mlir::IntegerAttr::get(type, llvm::APInt(width, decimal(*number), 10))});
        }
        auto evaluator = fs::SymbolicEvaluator::bind(*analysis, values);
        if (mlir::failed(evaluator)) {
            return mlir::failure();
        }
        bool probes = evaluator->cachedNodes() == 0;
        if (!values.empty()) {
            auto duplicate = values;
            duplicate.push_back(values.front());
            probes &= mlir::failed(fs::SymbolicEvaluator::bind(*analysis, duplicate));
            auto wrong = values;
            wrong.front().result = mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 7), 0);
            probes &= mlir::failed(fs::SymbolicEvaluator::bind(*analysis, wrong));
            auto unknown = values;
            unknown.front().value =
                arguments.addArgument(values.front().value.getType(), mlir::UnknownLoc::get(&context));
            probes &= mlir::failed(fs::SymbolicEvaluator::bind(*analysis, unknown));
            if (values.size() > 1) {
                auto sameCount = values;
                sameCount.back() = sameCount.front();
                probes &= mlir::failed(fs::SymbolicEvaluator::bind(*analysis, sameCount));
            }
        }
        Array presence, reachability, minimum, uniformReachability, uniformRetained;
        llvm::SmallVector<fs::SymbolicEvent> points;
        for (const auto& event : *events) {
            auto parsed = readEvent(event);
            if (mlir::failed(parsed)) {
                return mlir::failure();
            }
            points.push_back(*parsed);
            presence.push_back(answer(evaluator->contains(*parsed)));
        }
        for (const auto& source : points) {
            Array reachable, retained, uniformRow, uniformMinimumRow;
            for (const auto& target : points) {
                reachable.push_back(answer(evaluator->reaches(source, target)));
                retained.push_back(answer(evaluator->minimumDemand(source, target)));
                if (uniformReach) {
                    auto a = evaluator->contains(source), b = evaluator->contains(target);
                    if (mlir::failed(a) || mlir::failed(b) || !*a || !*b) {
                        uniformRow.push_back(nullptr);
                        uniformMinimumRow.push_back(nullptr);
                    } else {
                        llvm::SmallVector<BigInt> point;
                        for (const auto* endpoint : {&source, &target}) {
                            point.push_back(BigInt(static_cast<std::int64_t>(endpoint->site)));
                            point.append(endpoint->coordinates.begin(), endpoint->coordinates.end());
                            while (point.size() % ((*schema)->coordinateDepth() + 2) !=
                                   (*schema)->coordinateDepth() + 1) {
                                point.push_back(BigInt(0));
                            }
                            point.push_back(BigInt(static_cast<std::int64_t>(endpoint->kind)));
                        }
                        point.append(
                            evaluator->boundContext()->values.begin(), evaluator->boundContext()->values.end());
                        uniformRow.push_back(uniformReach->relation().containsPoint(point));
                        uniformMinimumRow.push_back(uniformMinimum->relation().containsPoint(point));
                    }
                }
            }
            reachability.push_back(std::move(reachable));
            minimum.push_back(std::move(retained));
            if (uniformReach) {
                uniformReachability.push_back(std::move(uniformRow));
                uniformRetained.push_back(std::move(uniformMinimumRow));
            }
        }
        if (evaluator->admitted() && !points.empty()) {
            auto artifact = evaluator->materialize((*analysis)->reachability());
            auto another = fs::SymbolicEvaluator::bind(*analysis, values);
            auto uniform = fs::SymbolicEvaluator::uniform(*analysis);
            auto identity = uniform->materialize((*analysis)->identity());
            probes &= mlir::succeeded(artifact) && mlir::succeeded(identity) &&
                      mlir::failed(another->containsPair(*artifact, points.front(), points.front())) &&
                      mlir::failed(evaluator->containsPair(*identity, points.front(), points.front()));
            // Binding vectors and source input copies may change; snapshots remain independent.
            auto before = evaluator->containsPair(*artifact, points.front(), points.front());
            values.clear();
            auto after = evaluator->containsPair(*artifact, points.front(), points.front());
            probes &= mlir::succeeded(before) == mlir::succeeded(after) && (mlir::failed(before) || *before == *after);
            probes &= artifact->boundContext() && artifact->relation().getNumSymbolVars() == 0 &&
                      identity->relation().getNumSymbolVars() == parameters.size();
        }
        evaluated.push_back(
            Object{
                {"admitted", evaluator->admitted()},
                {"probes", probes},
                {"presence", std::move(presence)},
                {"reach", std::move(reachability)},
                {"minimum", std::move(minimum)},
                {"uniform_reach", std::move(uniformReachability)},
                {"uniform_minimum", std::move(uniformRetained)}});
    }
    result["contexts"] = std::move(evaluated);
    return result;
}
} // namespace frontier_test
