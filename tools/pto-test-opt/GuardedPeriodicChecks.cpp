// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Evaluate one symbolic quotient for multiple small, concrete test valuations.
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicQuotient.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::optional<uint64_t> number(const llvm::json::Object& object, StringRef key)
{
    const auto* value = object.get(key);
    return value ? value->getAsUINT64() : std::nullopt;
}
bool evaluate(Block& code, ArrayRef<BlockArgument> arguments, const llvm::json::Array& sample,
              ArrayRef<Value> roots, llvm::json::Array& output)
{
    if (sample.size() != arguments.size()) { return false; }
    llvm::DenseMap<Value, Attribute> values;
    for (auto [argument, value] : llvm::zip(arguments, sample)) {
        auto integer = value.getAsUINT64();
        if (!integer || (argument.getType().isInteger(1) && *integer > 1)) { return false; }
        const auto bits = argument.getType().isIndex() ? 64 : 1;
        values[argument] = IntegerAttr::get(argument.getType(), APInt(bits, *integer));
    }
    for (auto& operation : code) {
        SmallVector<Attribute> operands;
        for (auto operand : operation.getOperands()) {
            auto value = values.lookup(operand);
            if (!value) { return false; }
            operands.push_back(value);
        }
        SmallVector<OpFoldResult> results;
        if (failed(operation.fold(operands, results)) || results.size() != operation.getNumResults()) {
            return false;
        }
        for (auto [value, result] : llvm::zip(operation.getResults(), results)) {
            auto attr = result.dyn_cast<Attribute>();
            if (!attr) { attr = values.lookup(result.get<Value>()); }
            if (!attr) { return false; }
            values[value] = attr;
        }
    }
    for (auto root : roots) {
        auto value = dyn_cast_or_null<IntegerAttr>(values.lookup(root));
        if (!value || value.getValue().getActiveBits() > 64) { return false; }
        output.push_back(value.getValue().getZExtValue());
    }
    return true;
}
bool graphCase(const llvm::json::Object& input, llvm::json::Object& output, MLIRContext& context)
{
    const auto* pipes = input.getArray("pipes");
    const auto* records = input.getArray("records");
    const auto* samples = input.getArray("samples");
    if (!pipes || pipes->size() > 24 || !records || records->size() > 576 ||
        !samples || samples->size() > 256) { return false; }
    OpBuilder builder(&context);
    auto module = OwningOpRef<ModuleOp>(ModuleOp::create(builder.getUnknownLoc()));
    SmallVector<Type> types(pipes->size(), builder.getI1Type());
    for (std::size_t i = 0; i < records->size(); ++i) {
        types.push_back(builder.getI1Type());
        types.push_back(builder.getIndexType());
    }
    builder.setInsertionPointToEnd(module->getBody());
    auto function = builder.create<func::FuncOp>(builder.getUnknownLoc(), "quotient_test",
                                               builder.getFunctionType(types, {}));
    auto* entry = function.addEntryBlock();
    builder.setInsertionPointToEnd(entry);
    auto cut = builder.create<func::ReturnOp>(builder.getUnknownLoc());
    auto expressions = std::make_shared<fs::RegionExpressions>();
    std::vector<fs::GuardedPeriodicPayload> payloads;
    for (auto [i, pipe] : llvm::enumerate(*pipes)) {
        auto value = pipe.getAsUINT64();
        if (!value || *value > UINT32_MAX) { return false; }
        payloads.push_back({static_cast<uint32_t>(*value), expressions->input(entry->getArgument(i))});
    }
    std::vector<fs::GuardedPeriodicRecord> generators;
    for (auto [i, record] : llvm::enumerate(*records)) {
        const auto* object = record.getAsObject();
        if (!object) { return false; }
        auto source = number(*object, "source"), target = number(*object, "target");
        auto bound = number(*object, "bound");
        if (!source || *source > UINT32_MAX || !target || *target > UINT32_MAX || !bound) { return false; }
        const auto argument = pipes->size() + 2 * i;
        generators.push_back({static_cast<uint32_t>(*source), static_cast<uint32_t>(*target),
            expressions->input(entry->getArgument(argument + 1)),
            expressions->input(entry->getArgument(argument)), *bound});
    }
    auto quotient = fs::analyzeGuardedPeriodicQuotient(expressions, payloads, generators);
    output["error"] = quotient.error;
    output["expressions"] = expressions->size();
    output["edges"] = quotient.graphEdges;
    if (!quotient.error.empty()) { return true; }
    // Emit each shared root once, then reuse that exact circuit for all samples.
    Block code;
    builder.setInsertionPointToEnd(&code);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> memo;
    SmallVector<Value> roots;
    auto append = [&](fs::RegionExpressions::Id id) {
        auto result = expressions->emit(id, builder, cut, memo);
        if (failed(result)) { return false; }
        roots.push_back(*result);
        return true;
    };
    for (auto id : quotient.retained) {
        if (!append(id)) { return false; }
    }
    for (const auto& threshold : quotient.thresholds) {
        if (!append(threshold.reachable) || !append(threshold.distance)) { return false; }
    }
    output["emitted"] = code.getOperations().size();
    llvm::json::Array evaluations;
    for (const auto& sample : *samples) {
        const auto* arguments = sample.getAsArray();
        llvm::json::Array values;
        if (!arguments || !evaluate(code, entry->getArguments(), *arguments, roots, values)) { return false; }
        evaluations.push_back(std::move(values));
    }
    output["values"] = std::move(evaluations);
    return true;
}
} // namespace
int runGuardedPeriodicChecks(llvm::StringRef path)
{
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer || (*buffer)->getBufferSize() > 16 * 1024 * 1024) { return 1; }
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) { llvm::consumeError(parsed.takeError()); return 1; }
    const auto* cases = parsed->getAsArray();
    if (!cases || cases->size() > 128) { return 1; }
    MLIRContext context;
    context.disableMultithreading();
    context.getOrLoadDialect<arith::ArithDialect>();
    context.getOrLoadDialect<func::FuncDialect>();
    llvm::json::Array output;
    for (const auto& item : *cases) {
        const auto* input = item.getAsObject();
        llvm::json::Object result;
        if (!input || !graphCase(*input, result, context)) {
            llvm::errs() << "malformed guarded quotient request or unevaluable circuit\n";
            return 1;
        }
        output.push_back(std::move(result));
    }
    llvm::outs() << llvm::json::Value(std::move(output)) << "\n";
    return 0;
}
