// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Test-only bounded batches of explicit incidences and graph generators. This
// exercises the production analysis, not MLIR translation or target emission.
#include "PTO/Transforms/FrontierSynch/DemandAnalysis.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include "frontier-guarded-driver.h"

namespace {
namespace fs = mlir::pto::frontiersynch;
using llvm::json::Array;
using llvm::json::Object;
constexpr std::size_t MaxCases = 4096;
constexpr std::size_t MaxSites = 32;
constexpr std::size_t MaxFootprints = 8;
constexpr std::size_t MaxFileBytes = 8 * 1024 * 1024;
struct TestInput {
    llvm::SmallVector<std::unique_ptr<mlir::pto::CompoundInstanceElement>> owned;
    llvm::SmallVector<const mlir::pto::CompoundInstanceElement*> sequence;
    llvm::SmallVector<fs::StorageFootprint> footprints;
    llvm::SmallVector<fs::StorageAlias> aliases;
};

mlir::FailureOr<llvm::SmallVector<std::size_t>> indices(const llvm::json::Value& value, std::size_t bound)
{
    auto* array = value.getAsArray();
    if (!array || array->size() > MaxSites) {
        return mlir::failure();
    }
    llvm::SmallVector<std::size_t> result;
    for (const auto& entry : *array) {
        auto id = entry.getAsInteger();
        if (!id || *id < 0 || static_cast<std::uint64_t>(*id) >= bound) {
            return mlir::failure();
        }
        result.push_back(static_cast<std::size_t>(*id));
    }
    return result;
}

mlir::LogicalResult effects(const Object& object, llvm::StringRef name, bool write, TestInput& input)
{
    auto* rows = object.getArray(name);
    if (!rows || rows->size() != input.sequence.size()) {
        return mlir::failure();
    }
    for (auto [site, row] : llvm::enumerate(*rows)) {
        auto ids = indices(row, input.footprints.size());
        if (mlir::failed(ids)) {
            return mlir::failure();
        }
        for (auto id : *ids) {
            input.footprints[id].accesses.push_back({input.sequence[site], !write, write});
        }
    }
    return mlir::success();
}

mlir::FailureOr<TestInput> readInput(const Object& object, mlir::MLIRContext& context)
{
    auto* pipes = object.getArray("pipes");
    auto* aliases = object.getArray("aliases");
    auto count = object.getInteger("footprints");
    if (!pipes || pipes->size() > MaxSites || !aliases || aliases->size() > MaxFootprints * MaxFootprints || !count ||
        *count < 0 || static_cast<std::uint64_t>(*count) > MaxFootprints) {
        return mlir::failure();
    }
    TestInput input;
    input.footprints.resize(static_cast<std::size_t>(*count));
    for (auto [site, pipe] : llvm::enumerate(*pipes)) {
        auto value = pipe.getAsInteger();
        if (!value || *value < 0 || *value >= static_cast<std::int64_t>(mlir::pto::PipelineType::PIPE_NUM)) {
            return mlir::failure();
        }
        auto phase = std::make_unique<mlir::pto::CompoundInstanceElement>(
            static_cast<unsigned>(site), llvm::SmallVector<const mlir::pto::BaseMemInfo*>{},
            llvm::SmallVector<const mlir::pto::BaseMemInfo*>{}, static_cast<mlir::pto::PipelineType>(*value),
            mlir::OperationName("test.phase", &context));
        input.sequence.push_back(phase.get());
        input.owned.push_back(std::move(phase));
    }
    for (const auto& row : *aliases) {
        // Permit the first invalid ID so the production bounds check is tested.
        auto pair = indices(row, input.footprints.size() + 1);
        if (mlir::failed(pair) || pair->size() != 2) {
            return mlir::failure();
        }
        input.aliases.push_back({(*pair)[0], (*pair)[1]});
    }
    if (mlir::failed(effects(object, "reads", false, input)) || mlir::failed(effects(object, "writes", true, input))) {
        return mlir::failure();
    }
    return input;
}

mlir::LogicalResult appendEdges(const Array& array, std::size_t count, llvm::SmallVectorImpl<fs::Demand>& edges)
{
    if (array.size() > MaxSites * MaxSites) {
        return mlir::failure();
    }
    for (const auto& row : array) {
        auto pair = indices(row, count + 1);
        if (mlir::failed(pair) || pair->size() != 2) {
            return mlir::failure();
        }
        edges.push_back({(*pair)[0], (*pair)[1], {}});
    }
    return mlir::success();
}

Array endpoints(llvm::ArrayRef<fs::Demand> demands)
{
    Array result;
    for (const auto& demand : demands) {
        result.push_back(Array{demand.source, demand.consumer});
    }
    return result;
}

Object dump(const fs::LifetimeAnalysis& lifetimes, const fs::RankReduction& reduction, llvm::ArrayRef<fs::Demand> edges)
{
    Array retained, nonadjacent, summaries, ranks, pipes, bypasses, witnesses;
    for (auto id : reduction.retained()) {
        retained.push_back(Array{edges[id].source, edges[id].consumer});
    }
    for (auto id : reduction.nonadjacentLocal()) {
        nonadjacent.push_back(Array{edges[id].source, edges[id].consumer});
    }
    for (const auto& summary : reduction.summaries()) {
        summaries.push_back(Object{{"S", Array(summary.S)}, {"T", Array(summary.T)}});
        ranks.push_back(summary.rank);
    }
    for (auto pipe : reduction.pipes()) {
        pipes.push_back(static_cast<unsigned>(pipe));
    }
    for (const auto& relay : lifetimes.bypasses()) {
        bypasses.push_back(
            Array{relay.writer, relay.reader, relay.consumer, relay.sourceFootprint, relay.consumerFootprint});
    }
    for (const auto& edge : lifetimes.generators()) {
        Array entries;
        for (const auto& witness : edge.witnesses) {
            Object entry{
                {"kind", static_cast<unsigned>(witness.hazard)},
                {"source", witness.sourceFootprint},
                {"consumer", witness.consumerFootprint}};
            if (witness.previousWriter) {
                entry["writer"] = *witness.previousWriter;
            }
            entries.push_back(std::move(entry));
        }
        witnesses.push_back(std::move(entries));
    }
    return Object{
        {"valid", true},
        {"generators", endpoints(lifetimes.generators())},
        {"used", endpoints(edges)},
        {"retained", std::move(retained)},
        {"nonadjacent", std::move(nonadjacent)},
        {"summaries", std::move(summaries)},
        {"ranks", std::move(ranks)},
        {"pipes", std::move(pipes)},
        {"bypasses", std::move(bypasses)},
        {"witnesses", std::move(witnesses)}};
}

mlir::FailureOr<Object> analyze(const Object& object, mlir::MLIRContext& context)
{
    auto input = readInput(object, context);
    if (mlir::failed(input)) {
        return mlir::failure();
    }
    if (object.getArray("guards")) {
        return frontier_test::analyzeGuarded(object, input->sequence, input->footprints, input->aliases, context);
    }
    fs::LifetimeAnalysis lifetimes;
    fs::RankReduction reduction;
    // Warm each result before deliberate invalid cases test failure atomicity.
    if (mlir::failed(lifetimes.build(input->sequence, input->footprints, {}))) {
        return mlir::failure();
    }
    if (mlir::failed(lifetimes.build(input->sequence, input->footprints, input->aliases))) {
        return Object{{"valid", false}, {"empty", lifetimes.generators().empty() && lifetimes.bypasses().empty()}};
    }
    llvm::SmallVector<fs::Demand> edges(lifetimes.generators());
    if (auto* provided = object.getArray("generators")) {
        edges.clear();
        if (mlir::failed(appendEdges(*provided, input->sequence.size(), edges))) {
            return mlir::failure();
        }
    }
    if (auto* extra = object.getArray("extra")) {
        if (mlir::failed(appendEdges(*extra, input->sequence.size(), edges))) {
            return mlir::failure();
        }
    }
    if (mlir::failed(reduction.build(input->sequence, lifetimes.generators()))) {
        return mlir::failure();
    }
    if (mlir::failed(reduction.build(input->sequence, edges))) {
        const bool empty = reduction.pipes().empty() && reduction.summaries().empty() && reduction.retained().empty() &&
                           reduction.nonadjacentLocal().empty();
        return Object{{"valid", false}, {"empty", empty}};
    }
    return dump(lifetimes, reduction, edges);
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        llvm::errs() << "usage: pto-frontier-demand-test cases.json\n";
        return 1;
    }
    auto buffer = llvm::MemoryBuffer::getFile(argv[1]);
    if (!buffer || (*buffer)->getBufferSize() > MaxFileBytes) {
        llvm::errs() << "cannot read bounded demand test input\n";
        return 1;
    }
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
        llvm::errs() << llvm::toString(parsed.takeError()) << "\n";
        return 1;
    }
    auto* cases = parsed->getAsArray();
    if (!cases || cases->size() > MaxCases) {
        llvm::errs() << "invalid demand test batch\n";
        return 1;
    }
    mlir::MLIRContext context;
    context.disableMultithreading();
    llvm::outs() << "[";
    for (auto [position, row] : llvm::enumerate(*cases)) {
        auto* object = row.getAsObject();
        auto result = object ? analyze(*object, context) : mlir::FailureOr<Object>(mlir::failure());
        if (mlir::failed(result)) {
            llvm::errs() << "invalid demand test case\n";
            return 1;
        }
        if (position != 0) {
            llvm::outs() << ",";
        }
        llvm::outs() << llvm::json::Value(std::move(*result));
    }
    llvm::outs() << "]\n";
    return 0;
}
