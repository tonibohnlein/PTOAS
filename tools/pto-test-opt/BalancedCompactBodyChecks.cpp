// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Execute prepared endpoint copies through original, independently chosen arms.
#include "PTO/Transforms/FrontierSynch/BalancedCompactBody.h"
#include "SyncLogicalInsertionChecks.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <tuple>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::string render(func::FuncOp function)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    function.print(stream);
    return text;
}
using Identity = std::tuple<int64_t, int64_t, bool>;
using Commands = std::map<Identity, uint64_t>;
std::optional<Commands> commands(const llvm::json::Object& trace)
{
    auto error = trace.getString("error");
    const auto* events = trace.getArray("events");
    if (!error || !error->empty() || !events) { return std::nullopt; }
    Commands result;
    for (const auto& item : *events) {
        const auto* event = item.getAsObject();
        if (!event) { return std::nullopt; }
        auto kind = event->getString("kind");
        if (!kind) { return std::nullopt; }
        if (*kind == "payload") { continue; }
        if (*kind != "set" && *kind != "wait") { return std::nullopt; }
        auto record = event->getInteger("record"), ordinal = event->getInteger("source_ordinal");
        auto gap = event->getInteger("gap");
        if (!record || !ordinal || !gap || *gap < 0 ||
            !result.emplace(Identity{*record, *ordinal, *kind == "set"}, static_cast<uint64_t>(*gap)).second) {
            return std::nullopt;
        }
    }
    return result;
}
std::vector<int64_t> payloads(const llvm::json::Object& trace)
{
    std::vector<int64_t> result;
    if (const auto* events = trace.getArray("events")) {
        for (const auto& item : *events) {
            const auto* event = item.getAsObject();
            if (event && event->getString("kind") == "payload") {
                if (auto type = event->getInteger("type")) { result.push_back(*type); }
            }
        }
    }
    return result;
}
struct OriginalOperation {
    Operation* operation = nullptr;
    Block* block = nullptr;
    SmallVector<Value> operands;
    DictionaryAttr attributes;
};
bool check(func::FuncOp function, const pto::SyncInput& input)
{
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
    scf::ForOp loop;
    bool multiple = false;
    function.walk([&](scf::ForOp candidate) {
        if (candidate->getParentOfType<scf::ForOp>()) { return; }
        multiple |= bool(loop);
        loop = candidate;
    });
    if (multiple) { return false; }
    auto body = fs::recognizeBalancedCompactBody(loop, input, index);
    if (function->hasAttr("test.balanced_reject")) { return !body.error.empty(); }
    auto slots = function->getAttrOfType<IntegerAttr>("test.balanced_slots");
    auto alternatives = function->getAttrOfType<IntegerAttr>("test.balanced_alternatives");
    if (!body.error.empty() || !slots || slots.getInt() != 2 || body.slots.size() != 2 || !alternatives ||
        alternatives.getInt() <= 0 || body.slots[0].pipe == body.slots[1].pipe) { return false; }
    for (const auto& slot : body.slots) {
        if (slot.alternatives.size() != static_cast<uint64_t>(alternatives.getInt())) { return false; }
        for (const auto& alternative : slot.alternatives) {
            auto* owner = alternative.phase->elementOp->getParentRegion();
            for (auto* region : llvm::reverse(alternative.path)) {
                if (region != owner || !isa<scf::IfOp>(region->getParentOp())) { return false; }
                owner = region->getParentOp()->getParentRegion();
            }
            if (owner != &loop.getRegion()) { return false; }
        }
    }
    const std::vector<fs::PeriodicPayload> word{{body.slots[0].pipe}, {body.slots[1].pipe}};
    const auto analysis = fs::analyzePeriodicDemands(word, {{0, 1, 0}, {1, 0, 1}});
    if (!analysis.error.empty() || analysis.retained.size() != 2) { return false; }
    std::string error;
    const auto before = render(function);
    auto prepared = fs::prepareBalancedCompactInsertion(function, body, analysis, 73, error);
    if (failed(prepared) || !error.empty() || render(function) != before ||
        (*prepared)->endpoints.size() != 4 * static_cast<uint64_t>(alternatives.getInt()) ||
        (*prepared)->families.size() != analysis.retained.size() || !(*prepared)->groupedFamilies ||
        !(*prepared)->independentPieces) { return false; }
    std::vector<OriginalOperation> original;
    function.walk([&](Operation* operation) {
        if (operation != function) {
            original.push_back({operation, operation->getBlock(),
                SmallVector<Value>(operation->getOperands()), operation->getAttrDictionary()});
        }
    });
    // Varying the original trace argument changes the execution, not its IR.
    std::vector<std::vector<int64_t>> originalTraces;
    for (int64_t trips : {0, 1, 2, 5}) {
        function->setAttr("test.trace_arguments", DenseI64ArrayAttr::get(function.getContext(), {trips, 0}));
        auto trace = traceStructuredLogicalInsertion(function, input.instructions());
        if (!commands(trace)) { return false; }
        originalTraces.push_back(payloads(trace));
    }
    if (failed(fs::insertLogicalSynchronization(function, **prepared)) || failed(verify(function))) { return false; }
    for (const auto& saved : original) {
        NamedAttrList actual(saved.operation->getAttrDictionary()), expected(saved.attributes);
        // Serialization tags original loops; payload/control semantics stay unchanged.
        actual.erase("pto.family_loop"); expected.erase("pto.family_loop");
        if (saved.operation->getBlock() != saved.block ||
            actual.getDictionary(function.getContext()) != expected.getDictionary(function.getContext()) ||
            !llvm::equal(saved.operation->getOperands(), saved.operands)) { return false; }
    }
    std::size_t traceId = 0;
    for (uint32_t trips : {0, 1, 2, 5}) {
        function->setAttr("test.trace_arguments", DenseI64ArrayAttr::get(function.getContext(), {trips, 0}));
        auto trace = traceStructuredLogicalInsertion(function, input.instructions());
        auto actual = commands(trace);
        Commands expected;
        for (auto record : analysis.retained) {
            const auto edge = analysis.generators[record];
            for (uint32_t i = 0; i < trips; ++i) {
                if (edge.displacement >= trips - i) { continue; }
                expected[{record, i, true}] = 2 * i + edge.source + 1;
                expected[{record, i, false}] = 2 * (i + edge.displacement) + edge.target;
            }
        }
        if (!actual || *actual != expected || payloads(trace) != originalTraces[traceId++]) { return false; }
    }
    return true;
}
} // namespace
int runBalancedCompactBodyChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (!check(function, input)) {
        llvm::errs() << "balanced compact body checks failed: " << function.getSymName() << "\n";
        return 1;
    }
    llvm::outs() << "balanced compact body checks passed: " << function.getSymName() << "\n";
    return 0;
}
