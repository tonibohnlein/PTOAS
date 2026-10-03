// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bounded numeric-quotient tests reuse shared phase identities, including
// repeated period positions. Finite unfolding exists only in this test adapter.
#include "frontier-periodic-driver.h"
#include <cstdint>

namespace frontier_test {
namespace {
namespace fs = mlir::pto::frontiersynch;
using llvm::json::Array;
using llvm::json::Object;
constexpr std::size_t MaxRecords = 4096;
constexpr std::size_t MaxPrefix = 128;
constexpr std::size_t MaxQueries = 64;

mlir::FailureOr<llvm::SmallVector<fs::PeriodicPrerequisite>> readRecords(const Object& object)
{
    const auto* rows = object.getArray("generators");
    if (!rows || rows->size() > MaxRecords) {
        return mlir::failure();
    }
    llvm::SmallVector<fs::PeriodicPrerequisite> records;
    for (const auto& row : *rows) {
        const auto* array = row.getAsArray();
        if (!array || array->size() != 3) {
            return mlir::failure();
        }
        auto source = index((*array)[0], MaxPrefix);
        auto consumer = index((*array)[1], MaxPrefix);
        auto distance = integer((*array)[2]);
        if (mlir::failed(source) || mlir::failed(consumer) || mlir::failed(distance)) {
            return mlir::failure();
        }
        records.push_back({*source, *consumer, *distance});
    }
    return records;
}

Array stringify(llvm::ArrayRef<llvm::DynamicAPInt> values)
{
    Array result;
    for (const auto& value : values) {
        result.push_back(decimal(value));
    }
    return result;
}
Array thresholds(const fs::PeriodicDemandReduction& analysis, fs::PeriodicEventKind kind,
                 fs::PeriodicEventKind sourceKind = fs::PeriodicEventKind::Completion)
{
    Array matrix;
    for (std::size_t source = 0; source < analysis.sites().size(); ++source) {
        Array row;
        for (std::size_t target = 0; target < analysis.sites().size(); ++target) {
            auto value = analysis.threshold(source, sourceKind, target, kind);
            row.push_back(value->has_value() ? llvm::json::Value(decimal(**value)) : llvm::json::Value(nullptr));
        }
        matrix.push_back(std::move(row));
    }
    return matrix;
}
Object dump(const fs::PeriodicDemandReduction& analysis)
{
    Array records, pipes;
    for (const auto& record : analysis.generators()) {
        records.push_back(
            Array{record.edge.source, record.edge.consumer, decimal(record.edge.distance), Array(record.origins)});
    }
    for (auto pipe : analysis.pipes()) {
        pipes.push_back(static_cast<unsigned>(pipe));
    }
    return Object{
        {"valid", true},
        {"records", std::move(records)},
        {"retained", Array(analysis.retained())},
        {"pipes", std::move(pipes)},
        {"vertices", analysis.vertexCount()},
        {"edges", analysis.edgeCount()},
        {"runs", analysis.shortestPathRuns()},
        {"index_entries", analysis.indexEntryCount()},
        {"from_start_to_start", thresholds(analysis, fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Start)},
        {"from_start_to_completion", thresholds(analysis, fs::PeriodicEventKind::Completion,
                                               fs::PeriodicEventKind::Start)},
        {"start", thresholds(analysis, fs::PeriodicEventKind::Start)},
        {"completion", thresholds(analysis, fs::PeriodicEventKind::Completion)}};
}

mlir::FailureOr<Object> prefixSummary(const fs::PeriodicDemandReduction& analysis, std::size_t count)
{
    if (count != 0 && analysis.sites().empty()) {
        return mlir::failure();
    }
    llvm::SmallVector<const mlir::pto::CompoundInstanceElement*> occurrences;
    llvm::SmallVector<fs::Demand> edges;
    const auto width = analysis.sites().size();
    for (std::size_t site = 0; site < count; ++site) {
        occurrences.push_back(analysis.sites()[site % width]);
        for (const auto& record : analysis.generators()) {
            const auto& edge = record.edge;
            if (edge.source != site % width || edge.distance > static_cast<std::int64_t>(count)) {
                continue;
            }
            const auto delta = static_cast<std::int64_t>(edge.distance);
            const auto target = (site / width + static_cast<std::size_t>(delta)) * width + edge.consumer;
            if (target < count) {
                edges.push_back({site, target, {}});
            }
        }
    }
    fs::RankReduction explicitResult;
    if (mlir::failed(explicitResult.build(occurrences, edges))) {
        return mlir::failure();
    }
    Array rows;
    for (std::size_t site = 0; site < count; ++site) {
        auto period = llvm::DynamicAPInt(static_cast<std::int64_t>(site / width));
        auto prefix = llvm::DynamicAPInt(static_cast<std::int64_t>(count));
        auto start = analysis.frontier(site % width, fs::PeriodicEventKind::Start, period, prefix);
        auto completion = analysis.frontier(site % width, fs::PeriodicEventKind::Completion, period, prefix);
        if (mlir::failed(start) || mlir::failed(completion)) {
            return mlir::failure();
        }
        const auto& summary = explicitResult.summaries()[site];
        rows.push_back(
            Object{
                {"S", stringify(*start)},
                {"T", stringify(*completion)},
                {"explicit_S", Array(summary.S)},
                {"explicit_T", Array(summary.T)}});
    }
    Array presentPipes;
    for (auto pipe : explicitResult.pipes()) {
        presentPipes.push_back(static_cast<unsigned>(pipe));
    }
    return Object{{"prefix", count}, {"rows", std::move(rows)}, {"pipes", std::move(presentPipes)}};
}

mlir::FailureOr<Array> prefixes(const Object& object, const fs::PeriodicDemandReduction& analysis)
{
    const auto* requested = object.getArray("prefixes");
    if (!requested || requested->size() > MaxQueries) {
        return mlir::failure();
    }
    Array results;
    for (const auto& value : *requested) {
        auto count = index(value, MaxPrefix);
        if (mlir::failed(count)) {
            return mlir::failure();
        }
        auto summary = prefixSummary(analysis, *count);
        if (mlir::failed(summary)) {
            return mlir::failure();
        }
        results.push_back(std::move(*summary));
    }
    return results;
}

mlir::FailureOr<Array> queries(const Object& object, const fs::PeriodicDemandReduction& analysis, bool frontier)
{
    const auto* requested = object.getArray(frontier ? "frontier_queries" : "queries");
    if (!requested || requested->size() > MaxQueries) {
        return mlir::failure();
    }
    Array results;
    for (const auto& value : *requested) {
        const auto* row = value.getAsArray();
        if (!row || row->size() != 4) {
            return mlir::failure();
        }
        auto first = index((*row)[0], MaxPrefix);
        auto second = index((*row)[1], MaxPrefix);
        auto third = integer((*row)[2]);
        auto fourth = integer((*row)[3]);
        if (mlir::failed(first) || mlir::failed(second) || mlir::failed(third) || mlir::failed(fourth)) {
            return mlir::failure();
        }
        if (frontier) {
            auto answer = analysis.frontier(*first, static_cast<fs::PeriodicEventKind>(*second), *third, *fourth);
            results.push_back(
                mlir::succeeded(answer) ? llvm::json::Value(stringify(*answer)) : llvm::json::Value(nullptr));
        } else {
            if (*third < 0 || *third > 2) {
                return mlir::failure();
            }
            auto kind = static_cast<fs::PeriodicEventKind>(static_cast<std::int64_t>(*third));
            auto answer = analysis.reaches(*first, *second, kind, *fourth);
            results.push_back(mlir::succeeded(answer) ? llvm::json::Value(*answer) : llvm::json::Value(nullptr));
        }
    }
    return results;
}

bool empty(const fs::PeriodicDemandReduction& analysis)
{
    return analysis.sites().empty() && analysis.generators().empty() && analysis.retained().empty() &&
           analysis.pipes().empty() && analysis.vertexCount() == 0 && analysis.edgeCount() == 0 &&
           mlir::failed(analysis.threshold(0, 0, fs::PeriodicEventKind::Start));
}
} // namespace

mlir::FailureOr<Object> analyzePeriodic(
    const Object& object, llvm::ArrayRef<const mlir::pto::CompoundInstanceElement*> sites,
    llvm::ArrayRef<fs::RotatingFamily> families, llvm::ArrayRef<fs::RotatingFragment> fragments,
    mlir::MLIRContext& context)
{
    auto supplied = readRecords(object);
    if (mlir::failed(supplied)) {
        return mlir::failure();
    }
    mlir::pto::CompoundInstanceElement warmPhase(
        0, {}, {}, mlir::pto::PipelineType::PIPE_V, mlir::OperationName("test.phase", &context));
    fs::PeriodicDemandReduction analysis;
    if (mlir::failed(analysis.build({&warmPhase}, {{0, 0, llvm::DynamicAPInt(1)}}))) {
        return mlir::failure();
    }
    llvm::SmallVector<const mlir::pto::CompoundInstanceElement*> sequence(sites);
    if (object.getBoolean("null_site").value_or(false) && !sequence.empty()) {
        sequence[0] = nullptr;
    }
    fs::RotatingFootprintAnalysis storage;
    const bool fromStorage = object.getBoolean("from_storage").value_or(false);
    if (fromStorage && mlir::failed(storage.build(sequence, families, fragments))) {
        return mlir::failure();
    }
    auto status = fromStorage ? analysis.build(storage, *supplied) : analysis.build(sequence, *supplied);
    if (mlir::failed(status)) {
        return Object{{"valid", false}, {"empty", empty(analysis)}};
    }
    auto result = dump(analysis);
    auto summaries = prefixes(object, analysis);
    auto answers = queries(object, analysis, false);
    auto frontiers = queries(object, analysis, true);
    if (mlir::failed(summaries) || mlir::failed(answers) || mlir::failed(frontiers)) {
        return mlir::failure();
    }
    result["prefixes"] = std::move(*summaries);
    result["queries"] = std::move(*answers);
    result["frontier_queries"] = std::move(*frontiers);
    if (fromStorage) {
        Array extracted;
        for (const auto& edge : storage.generators()) {
            extracted.push_back(Array{edge.source, edge.consumer, decimal(edge.distance)});
        }
        result["storage"] = std::move(extracted);
    }
    return result;
}
} // namespace frontier_test
