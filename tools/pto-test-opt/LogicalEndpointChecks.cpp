// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite matching and canonical-cut data for an independent command oracle.
#include "PTO/Transforms/FrontierSynch/LogicalEndpoints.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <map>
namespace fs = mlir::pto::frontiersynch;
llvm::json::Object dumpLogicalEndpoints(const fs::LogicalEndpointPlan& result);
namespace {
llvm::json::Object instance(const fs::EndpointInstance& value)
{
    return llvm::json::Object{{"kind", static_cast<unsigned>(value.kind)}, {"pipe", value.pipe},
        {"type", value.type}, {"ordinal", value.ordinal}, {"record", value.identity.record},
        {"source_ordinal", value.identity.sourceOrdinal}};
}
llvm::json::Object evaluation(const fs::EndpointEvaluation& value)
{
    return llvm::json::Object{{"error", static_cast<unsigned>(value.error)}, {"active", value.active},
                              {"instance", instance(value.instance)}};
}
bool extraQueries(const llvm::json::Object& input, const fs::LogicalEndpointPlan& plan,
                  llvm::json::Object& result)
{
    if (const auto* queries = input.getArray("endpoint_queries")) {
        if (queries->size() > 128) {
            return false;
        }
        llvm::json::Array answers;
        for (const auto& query : *queries) {
            const auto* tuple = query.getAsArray();
            if (!tuple || tuple->size() != 3) {
                return false;
            }
            auto id = (*tuple)[0].getAsUINT64(), trips = (*tuple)[1].getAsUINT64();
            auto ordinal = (*tuple)[2].getAsUINT64();
            if (!id || *id > UINT32_MAX || !trips || !ordinal) {
                return false;
            }
            answers.push_back(evaluation(plan.evaluate(*id, *trips, *ordinal)));
        }
        result["endpoint_answers"] = std::move(answers);
    }
    if (const auto* queries = input.getArray("counted_loops")) {
        if (queries->size() > 128) {
            return false;
        }
        llvm::json::Array answers;
        for (const auto& query : *queries) {
            const auto* tuple = query.getAsArray();
            if (!tuple || tuple->size() != 4) {
                return false;
            }
            int64_t values[4] = {};
            for (unsigned i = 0; i < 4; ++i) {
                auto value = (*tuple)[i].getAsInteger();
                if (!value) {
                    return false;
                }
                values[i] = *value;
            }
            const auto trips = fs::countedTrips(values[0], values[1], values[2]);
            const auto ordinal = fs::countedOrdinal(values[0], values[1], values[2], values[3]);
            answers.push_back(llvm::json::Object{{"trip_error", static_cast<unsigned>(trips.error)},
                {"trips", trips.value}, {"ordinal_error", static_cast<unsigned>(ordinal.error)},
                {"ordinal", ordinal.value}});
        }
        result["counted_answers"] = std::move(answers);
    }
    return true;
}
} // namespace
bool appendLogicalChecks(const llvm::json::Object& input, const fs::PeriodicAnalysis& analysis,
                         llvm::json::Object& output)
{
    const auto* requested = input.get("logical_trips");
    if (!requested) {
        return true;
    }
    auto trips = requested->getAsUINT64();
    const uint64_t tripLimit = analysis.payloads.size() > 64 ? 4 : 16;
    if (!trips || *trips > tripLimit || analysis.payloads.size() > 512) {
        return false;
    }
    auto plan = fs::buildLogicalEndpoints(analysis);
    auto result = dumpLogicalEndpoints(plan);
    if (!plan.error.empty()) {
        output["logical"] = std::move(result);
        return true;
    }
    const auto types = analysis.payloads.size();
    std::map<uint64_t, std::vector<fs::EndpointInstance>> cuts;
    llvm::json::Array active;
    for (uint32_t id = 0; id < plan.recipes.size(); ++id) {
        for (uint64_t ordinal = 0; ordinal < *trips; ++ordinal) {
            auto endpoint = plan.evaluate(id, *trips, ordinal);
            if (endpoint.error != fs::EndpointError::None) {
                return false;
            }
            if (!endpoint.active) {
                continue;
            }
            const auto& command = endpoint.instance;
            active.push_back(instance(command));
            // This test input is an actual straight-line word; adjacent scalar
            // instruction cuts coincide. Structured IR keys are tested separately.
            const uint64_t cut = ordinal * types + command.type + (command.kind == fs::EndpointKind::Set ? 1 : 0);
            cuts[cut].push_back(command);
        }
    }
    llvm::json::Array ordered;
    for (auto& [cut, commands] : cuts) {
        // Deliberately duplicate and reverse to exercise stable coalescing.
        const auto copy = commands;
        commands.insert(commands.end(), copy.begin(), copy.end());
        std::reverse(commands.begin(), commands.end());
        if (fs::canonicalizeCoincidentCut(commands) != fs::EndpointError::None) {
            return false;
        }
        llvm::json::Array values;
        for (const auto& command : commands) {
            values.push_back(instance(command));
        }
        ordered.push_back(llvm::json::Object{{"cut", cut}, {"commands", std::move(values)}});
    }
    result["active"] = std::move(active);
    result["cuts"] = std::move(ordered);
    if (!extraQueries(input, plan, result)) {
        return false;
    }
    output["logical"] = std::move(result);
    return true;
}
