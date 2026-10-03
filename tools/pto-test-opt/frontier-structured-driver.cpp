// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/StructuredInputAdapter.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
namespace {
namespace fs = mlir::pto::frontiersynch;
using namespace mlir;
using llvm::json::Array;
using llvm::json::Object;
using Int = llvm::DynamicAPInt;
using Tuple = fs::SymbolicTuple;
using Set = presburger::PresburgerRelation;
using Row = SmallVector<Int>;
FailureOr<SmallVector<Int>> numbers(const Array& array)
{
    SmallVector<Int> result;
    for (const auto& item : array) {
        auto number = item.getAsInteger();
        if (!number) {
            return failure();
        }
        result.push_back(Int(*number));
    }
    return result;
}
FailureOr<fs::SymbolicPrimitive> primitive(fs::SymbolicSchemaHandle schema, Tuple a, Tuple b, const Set& set)
{
    return fs::SymbolicPrimitive::import(schema, a, b, schema->parameters(), set);
}
FailureOr<fs::ExactStructuredEffects> effects(fs::SymbolicSchemaHandle schema, const Object& test)
{
    auto context = Set::getUniverse(*schema->space(Tuple::Unit, Tuple::Unit));
    if (const auto* bounds = test.getArray("bounds")) {
        if (bounds->size() != schema->parameters().size()) {
            return failure();
        }
        presburger::IntegerRelation piece(*schema->space(Tuple::Unit, Tuple::Unit));
        for (unsigned i = 0; i < bounds->size(); ++i) {
            auto pair = (*bounds)[i].getAsArray();
            if (!pair || pair->size() != 2) {
                return failure();
            }
            auto values = numbers(*pair);
            if (failed(values)) {
                return failure();
            }
            Row lower(piece.getNumCols(), Int(0)), upper = lower;
            lower[i] = Int(1);
            lower.back() = -(*values)[0];
            upper[i] = Int(-1);
            upper.back() = (*values)[1];
            piece.addInequality(lower);
            piece.addInequality(upper);
        }
        context = Set::getEmpty(*schema->space(Tuple::Unit, Tuple::Unit));
        context.unionInPlace(piece);
    }
    if (test.getBoolean("opaque").value_or(false)) {
        if (schema->parameters().size() != 2) {
            return failure();
        }
        auto qualified = Set::getEmpty(*schema->space(Tuple::Unit, Tuple::Unit));
        for (const auto& part : context.getAllDisjuncts()) {
            for (bool truth : {false, true}) {
                auto piece = part;
                Row predicate(piece.getNumCols(), Int(0)), result = predicate;
                predicate[0] = Int(truth ? -1 : 1);
                predicate.back() = Int(truth ? 1 : -2);
                result[1] = Int(1);
                result.back() = Int(truth ? -1 : 0);
                piece.addInequality(predicate);
                piece.addEquality(result);
                qualified.unionInPlace(piece);
            }
        }
        context = std::move(qualified);
    }
    auto accesses = [&](StringRef name) -> FailureOr<fs::SymbolicPrimitive> {
        auto relation = Set::getEmpty(*schema->space(Tuple::Occurrence, Tuple::Cell));
        const auto* rows = test.getArray(name);
        if (!rows) {
            return failure();
        }
        for (const auto& item : *rows) {
            auto array = item.getAsArray();
            if (!array || array->size() < 2) {
                return failure();
            }
            auto values = numbers(*array);
            if (failed(values)) {
                return failure();
            }
            auto site = (*array)[0].getAsInteger();
            if (!site || *site < 0 || static_cast<std::size_t>(*site) >= schema->sites().size() ||
                array->size() != schema->sites()[*site].coordinates.size() + 2) {
                return failure();
            }
            presburger::IntegerRelation piece(*schema->space(Tuple::Occurrence, Tuple::Cell));
            Row tag(piece.getNumCols(), Int(0)), cell = tag;
            tag[0] = Int(1);
            tag.back() = -(*values)[0];
            piece.addEquality(tag);
            cell[schema->coordinateDepth() + 1] = Int(1);
            cell.back() = -(*values)[1];
            for (unsigned i = 2; i < values->size(); ++i) {
                cell[i - 1] = -(*values)[i];
            }
            piece.addEquality(cell);
            relation.unionInPlace(piece);
        }
        return primitive(schema, Tuple::Occurrence, Tuple::Cell, relation);
    };
    auto reads = accesses("reads"), writes = accesses("writes");
    auto admitted = primitive(schema, Tuple::Unit, Tuple::Unit, context);
    auto extras = primitive(
        schema, Tuple::Occurrence, Tuple::Occurrence,
        Set::getEmpty(*schema->space(Tuple::Occurrence, Tuple::Occurrence)));
    if (failed(reads) || failed(writes) || failed(admitted) || failed(extras)) {
        return failure();
    }
    return fs::ExactStructuredEffects{*admitted, *reads, *writes, *extras, std::nullopt};
}
std::string print(Operation* operation)
{
    std::string result;
    llvm::raw_string_ostream output(result);
    operation->print(output);
    return result;
}
FailureOr<Object> run(const Object& test, MLIRContext& context)
{
    auto source = test.getString("source");
    if (!source) {
        return failure();
    }
    auto module = parseSourceString<ModuleOp>(*source, &context);
    if (!module || failed(verify(*module))) {
        return failure();
    }
    auto function = module->lookupSymbol<func::FuncOp>("test");
    if (!function) {
        return failure();
    }
    if (test.getBoolean("bad_terminator").value_or(false)) {
        context.getOrLoadDialect<cf::ControlFlowDialect>();
        // A one-block self-loop is rejected by MLIR verification. Deliberately
        // mutate the verified fixture to test the import boundary defensively.
        auto* terminator = function.getBody().front().getTerminator();
        OpBuilder builder(terminator);
        builder.create<cf::BranchOp>(terminator->getLoc(), &function.getBody().front());
        terminator->erase();
    }
    auto before = print(*module);
    pto::SyncInput input;
    if (failed(input.build(function))) {
        return failure();
    }
    fs::StructuredImportOptions options;
    for (auto argument : function.getArguments()) {
        if (isa<IntegerType, IndexType>(argument.getType())) {
            options.parameters.push_back(argument);
        }
    }
    if (test.getBoolean("opaque").value_or(false)) {
        function.walk([&](Operation* op) {
            if (op->hasAttr("sync.test.parameter") && op->getNumResults() == 1) {
                options.parameters.push_back(op->getResult(0));
            }
        });
    }
    options.cellAxes.push_back(IndexType::get(&context));
    options.physicalPartition = 71;
    unsigned qualifications = 0;
    if (test.getBoolean("qualify").value_or(true)) {
        options.mathematicalArithmetic = [&](Operation*, fs::SymbolicSchemaHandle, const fs::SymbolicPrimitive&) {
            ++qualifications;
            return true; // Test fixtures certify their bounded mathematical executions.
        };
        options.immutableParameter = [](Value, fs::SymbolicSchemaHandle, const fs::SymbolicPrimitive&) { return true; };
    }
    if (test.getBoolean("phase_order").value_or(true)) {
        options.phaseOrder = [&](Operation*, ArrayRef<const pto::CompoundInstanceElement*> phases)
            -> FailureOr<SmallVector<const pto::CompoundInstanceElement*>> {
            auto result = SmallVector<const pto::CompoundInstanceElement*>(phases.begin(), phases.end());
            if (test.getBoolean("bad_phase_order").value_or(false) && !result.empty()) {
                result.push_back(result.front());
            }
            return result;
        };
    }
    auto provider = [&](fs::SymbolicSchemaHandle schema,
                        ArrayRef<fs::StructuredSite>) -> FailureOr<fs::ExactStructuredEffects> {
        if (test.getBoolean("decline").value_or(false)) {
            return failure();
        }
        auto value = effects(schema, test);
        if (failed(value)) {
            return failure();
        }
        if (test.getBoolean("wrong_role").value_or(false)) {
            value->context = value->reads;
        }
        if (test.getBoolean("foreign_schema").value_or(false)) {
            auto foreign = fs::SymbolicSchema::create(schema->sites(), schema->parameters(), schema->cellTypes(), 72);
            if (failed(foreign)) {
                return failure();
            }
            value = effects(*foreign, test);
        }
        return value;
    };
    auto imported = fs::StructuredInputAdapter::build(function, input, options, provider);
    Object output{
        {"status", static_cast<int>(imported.issue.status)},
        {"reason", imported.issue.reason},
        {"qualifications", qualifications},
        {"unchanged", before == print(*module)}};
    if (!imported.succeeded()) {
        return output;
    }
    auto snapshot = imported.input;
    // Subsequent failed construction and caller option destruction cannot mutate
    // the saved relations, context or phase-path metadata.
    options.parameters.clear();
    options.phaseOrder = {};
    options.mathematicalArithmetic = {};
    options.immutableParameter = {};
    auto bad = fs::StructuredInputAdapter::build(function, input, options, {});
    if (bad.succeeded() || snapshot != imported.input) {
        return failure();
    }
    auto schema = snapshot->schema();
    Array depths, pipes;
    for (const auto& site : snapshot->sites()) {
        depths.push_back(static_cast<std::int64_t>(site.inductionVariables.size()));
        pipes.push_back(static_cast<int>(site.phase->kPipeValue));
    }
    output["depths"] = std::move(depths);
    output["pipes"] = std::move(pipes);
    auto analysis = fs::SymbolicDemandAnalysis::build(schema, snapshot->relations());
    if (failed(analysis)) {
        return failure();
    }
    if (test.getBoolean("construction_only").value_or(false)) {
        return output;
    }
    const auto* contexts = test.getArray("contexts");
    if (!contexts) {
        return failure();
    }
    Array contextOutputs;
    for (const auto& item : *contexts) {
        const auto* entry = item.getAsObject();
        if (!entry) {
            return failure();
        }
        auto* values = entry->getArray("values");
        auto* points = entry->getArray("points");
        if (!values || !points || values->size() != schema->parameters().size() || points->size() > 200) {
            return failure();
        }
        auto parameters = numbers(*values);
        if (failed(parameters)) {
            return failure();
        }
        SmallVector<fs::ImmutableParameterBinding> bindings;
        for (unsigned i = 0; i < values->size(); ++i) {
            bindings.push_back(
                {schema->parameters()[i],
                 IntegerAttr::get(schema->parameters()[i].getType(), *(*values)[i].getAsInteger())});
        }
        auto bound = fs::SymbolicEvaluator::bind(*analysis, bindings);
        if (failed(bound)) { return failure(); }
        bool admitted = bound->boundContext()->admitted;
        Object contextOutput{{"admitted", admitted}};
        if (!admitted) {
            contextOutputs.push_back(std::move(contextOutput));
            continue;
        }
        Array present, reference;
        SmallVector<SmallVector<Int>> encoded;
        SmallVector<fs::SymbolicEvent> events;
        for (const auto& point : *points) {
            const auto* tuple = point.getAsArray();
            if (!tuple || tuple->empty()) {
                return failure();
            }
            auto fields = numbers(*tuple);
            auto site = (*tuple)[0].getAsInteger();
            if (failed(fields) || !site || *site < 0 || static_cast<std::size_t>(*site) >= schema->sites().size() ||
                tuple->size() != schema->sites()[*site].coordinates.size() + 1) {
                return failure();
            }
            auto coordinates = SmallVector<Int>(fields->begin() + 1, fields->end());
            events.push_back({static_cast<std::size_t>(*site), coordinates, fs::PeriodicEventKind::Start});
            fields->resize(schema->coordinateDepth() + 1, Int(0));
            encoded.push_back(*fields);
            llvm::append_range(*fields, *parameters);
            present.push_back(snapshot->relations().present.relation().containsPoint(*fields));
        }
        for (const auto& first : encoded) {
            Array row;
            for (const auto& second : encoded) {
                auto point = first;
                llvm::append_range(point, second);
                llvm::append_range(point, *parameters);
                row.push_back(snapshot->relations().reference.relation().containsPoint(point));
            }
            reference.push_back(std::move(row));
        }
        contextOutput["present"] = std::move(present);
        contextOutput["reference"] = std::move(reference);
        if (test.getBoolean("downstream").value_or(false)) {
            Array reach, minimum;
            for (const auto& first : events) {
                for (unsigned kind = 0; kind < 2; ++kind) {
                    auto a = first;
                    a.kind = static_cast<fs::PeriodicEventKind>(kind);
                    Array reachRow, minimumRow;
                    for (const auto& second : events) {
                        for (unsigned otherKind = 0; otherKind < 2; ++otherKind) {
                            auto b = second;
                            b.kind = static_cast<fs::PeriodicEventKind>(otherKind);
                            auto r = bound->reaches(a, b), f = bound->minimumDemand(a, b);
                            reachRow.push_back(succeeded(r) ? llvm::json::Value(*r) : llvm::json::Value(nullptr));
                            minimumRow.push_back(
                                succeeded(f) ? llvm::json::Value(*f) : llvm::json::Value(nullptr));
                        }
                    }
                    reach.push_back(std::move(reachRow));
                    minimum.push_back(std::move(minimumRow));
                }
            }
            contextOutput["reach"] = std::move(reach);
            contextOutput["minimum"] = std::move(minimum);
            if (test.getBoolean("materialize").value_or(false)) {
                auto evaluator = fs::SymbolicEvaluator::bind(*analysis, bindings);
                if (failed(evaluator)) {
                    return failure();
                }
                auto materialized = evaluator->materialize((*analysis)->minimum());
                if (failed(materialized)) {
                    return failure();
                }
                for (const auto& first : events) {
                    auto a = first;
                    a.kind = fs::PeriodicEventKind::Completion;
                    for (const auto& b : events) {
                        auto expected = bound->minimumDemand(a, b);
                        auto actual = evaluator->containsPair(*materialized, a, b);
                        if (succeeded(expected) != succeeded(actual) ||
                            (succeeded(actual) && *actual != *expected)) {
                            return failure();
                        }
                    }
                }
                contextOutput["materialized"] = true;
            }
        }
        contextOutputs.push_back(std::move(contextOutput));
    }
    output["contexts"] = std::move(contextOutputs);
    return output;
}
} // namespace
int runStructuredJSON(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto file = llvm::MemoryBuffer::getFile(path);
    if (!file || (*file)->getBufferSize() > 8 * 1024 * 1024) {
        return 1;
    }
    auto json = llvm::json::parse((*file)->getBuffer());
    if (!json) {
        llvm::consumeError(json.takeError());
        return 1;
    }
    auto* cases = json->getAsArray();
    if (!cases || cases->size() > 100) {
        return 1;
    }
    Array results;
    for (const auto& value : *cases) {
        auto* test = value.getAsObject();
        if (!test) {
            return 1;
        }
        auto result = run(*test, context);
        if (failed(result)) {
            return 1;
        }
        results.push_back(std::move(*result));
    }
    llvm::outs() << llvm::json::Value(std::move(results)) << "\n";
    return 0;
}
