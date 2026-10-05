// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ExplicitAnalysis.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
void queryChecks(const fs::ExplicitAnalysis& analysis, llvm::json::Object& output)
{
    llvm::json::Array rows;
    for (uint32_t a = 0; a < 2 * analysis.occurrences.size(); ++a) {
        llvm::json::Array row;
        for (uint32_t b = 0; b < 2 * analysis.occurrences.size(); ++b) {
            const auto value = fs::explicitEventPrecedes(analysis,
                {a / 2, static_cast<fs::PeriodicEventKind>(a % 2)},
                {b / 2, static_cast<fs::PeriodicEventKind>(b % 2)});
            row.push_back(value ? llvm::json::Value(*value) : llvm::json::Value(nullptr));
        }
        rows.push_back(std::move(row));
    }
    output["event_reachable"] = std::move(rows);
    output["invalid_event_rejected"] = !fs::explicitEventPrecedes(analysis,
        {static_cast<uint32_t>(analysis.occurrences.size())}, {0}).has_value();
}
void spanChecks(const fs::ExplicitAnalysis& analysis, const fs::PhaseIndex& index,
                const pto::SyncInput& input, llvm::json::Object& output)
{
    const auto span = fs::analyzeExplicit(analysis.phases, index, input);
    llvm::json::Array retained;
    for (const auto& edge : span.reduction.retained) {
        retained.push_back(llvm::json::Array{edge.source, edge.target});
    }
    output["span_error"] = span.error;
    output["span_retained"] = std::move(retained);
    if (analysis.phases.size() >= 3) {
        SmallVector<const pto::CompoundInstanceElement*> gap{analysis.phases.front(), analysis.phases.back()};
        output["gap_rejected"] = !fs::analyzeExplicit(gap, index, input).error.empty();
    }
    if (analysis.phases.size() >= 2) {
        SmallVector<const pto::CompoundInstanceElement*> reversed{analysis.phases.back(), analysis.phases.front()};
        output["reverse_rejected"] = !fs::analyzeExplicit(reversed, index, input).error.empty();
    }
}
} // namespace
LogicalResult dumpExplicitAnalysis(func::FuncOp function, const pto::SyncInput& input)
{
    fs::PhaseIndex index;
    if (failed(index.build(function, input)) || function.isDeclaration()) {
        return failure();
    }
    auto result = fs::analyzeExplicit(function.front(), index, input);
    llvm::json::Object output;
    output["function"] = function.getSymName();
    output["error"] = result.error;
    llvm::json::Array occurrences, edges, rows, cells;
    for (const auto& occurrence : result.occurrences) {
        llvm::json::Array accesses;
        for (const auto& access : occurrence.accesses) {
            accesses.push_back(llvm::json::Array{access.atom, access.read, access.write,
                                               static_cast<int64_t>(access.protectionGroup)});
        }
        occurrences.push_back(llvm::json::Object{{"pipe", occurrence.pipe}, {"accesses", std::move(accesses)}});
    }
    for (const auto& edge : result.reduction.retained) {
        edges.push_back(llvm::json::Array{edge.source, edge.target});
    }
    for (const auto& row : result.reduction.startRanks) {
        llvm::json::Array values;
        for (auto value : row) {
            values.push_back(value);
        }
        rows.push_back(std::move(values));
    }
    for (const auto& cell : input.accesses().cells()) {
        cells.push_back(llvm::json::Array{static_cast<int64_t>(cell.space), cell.begin, cell.end});
    }
    llvm::json::Array boundary;
    for (const auto& cell : result.storageBoundary) {
        llvm::json::Array first, last;
        for (const auto& [pipe, payload] : cell.firstReaders) {
            first.push_back(llvm::json::Array{pipe, payload});
        }
        for (const auto& [pipe, payload] : cell.lastReaders) {
            last.push_back(llvm::json::Array{pipe, payload});
        }
        boundary.push_back(llvm::json::Object{{"atom", cell.atom},
            {"first_writer", cell.firstWriter ? llvm::json::Value(*cell.firstWriter) : llvm::json::Value(nullptr)},
            {"last_writer", cell.lastWriter ? llvm::json::Value(*cell.lastWriter) : llvm::json::Value(nullptr)},
            {"first_readers", std::move(first)}, {"last_readers", std::move(last)}});
    }
    output["occurrences"] = std::move(occurrences);
    output["retained"] = std::move(edges);
    output["start_ranks"] = std::move(rows);
    output["cells"] = std::move(cells);
    output["boundary"] = std::move(boundary);
    if (result.error.empty()) {
        queryChecks(result, output);
        spanChecks(result, index, input, output);
    }
    llvm::outs() << llvm::json::Value(std::move(output)) << "\n";
    return success();
}
