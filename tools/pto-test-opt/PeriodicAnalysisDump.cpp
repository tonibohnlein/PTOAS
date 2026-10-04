// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Inspect exact numerical periodic records without claiming executable endpoints.
#include "PTO/Transforms/FrontierSynch/PeriodicAnalysis.h"
#include "llvm/Support/JSON.h"
namespace fs = mlir::pto::frontiersynch;
llvm::json::Object dumpPeriodicAnalysis(const fs::PeriodicAnalysis& result)
{
    llvm::json::Array payloads, generators, retained, sourceRows, localRanks, frontiers;
    for (const auto& payload : result.payloads) {
        payloads.push_back(payload.pipe);
    }
    for (const auto& record : result.generators) {
        generators.push_back(llvm::json::Object{{"source", record.source}, {"target", record.target},
                                               {"displacement", record.displacement}});
    }
    for (auto id : result.retained) {
        retained.push_back(id);
    }
    for (auto row : result.sourceRows) {
        sourceRows.push_back(row);
    }
    for (auto rank : result.localRanks) {
        localRanks.push_back(rank);
    }
    for (const auto& frontier : result.frontiers) {
        llvm::json::Array distances;
        for (auto distance : frontier.distances) {
            distances.push_back(distance ? llvm::json::Value(*distance) : llvm::json::Value(nullptr));
        }
        frontiers.push_back(llvm::json::Object{{"pipe", frontier.pipe}, {"count", frontier.count},
                                              {"distances", std::move(distances)}});
    }
    return llvm::json::Object{{"error", result.error}, {"payloads", std::move(payloads)},
        {"generators", std::move(generators)}, {"retained", std::move(retained)},
        {"source_rows", std::move(sourceRows)}, {"local_ranks", std::move(localRanks)},
        {"frontiers", std::move(frontiers)}, {"graph_edges", result.graphEdges},
        {"minimum_demands_ready", result.error.empty()}, {"completion_queries_ready", result.error.empty()},
        {"interfaces_ready", false}};
}
