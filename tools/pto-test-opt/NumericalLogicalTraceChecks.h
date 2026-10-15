// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_NUMERICAL_LOGICAL_TRACE_CHECKS_H
#define PTO_NUMERICAL_LOGICAL_TRACE_CHECKS_H
#include "NumericalRequestOracle.h"
#include "SyncLogicalInsertionChecks.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "mlir/IR/Verifier.h"
#include <map>
#include <optional>
#include <vector>
namespace numerical_request_oracle {
// Test-only command graph, independent of all demand/query producers. Each
// pipe has separate native start/completion chains. WAIT/barrier blocks later
// starts; SET/barrier waits for earlier completions. Textual SET->WAIT order
// alone adds no edge. Matching uses the complete emitted logical identity.
struct LogicalSignature {
    std::vector<std::vector<int64_t>> payloads;
    std::vector<std::vector<bool>> reach;
    bool operator==(const LogicalSignature& other) const { return payloads == other.payloads && reach == other.reach; }
};
inline std::optional<LogicalSignature> logicalSignature(const llvm::json::Object& trace)
{
    auto error = trace.getString("error");
    auto* events = trace.getArray("events");
    auto count = trace.getInteger("payloads");
    if (!error || !error->empty() || !events || !count || *count < 0) { return {}; }
    const auto n = static_cast<std::size_t>(*count);
    struct Item { std::size_t vertex; StringRef kind; };
    std::map<int64_t, std::vector<Item>> rows;
    LogicalSignature result;
    for (const auto& raw : *events) {
        auto* event = raw.getAsObject();
        if (!event) { return {}; }
        auto kind = event->getString("kind"); auto pipe = event->getInteger("pipe");
        if (!kind || !pipe || *pipe < 0 || *pipe > 6) { return {}; }
        if (*kind != "payload") { continue; }
        auto type = event->getInteger("type"); auto* coordinates = event->getArray("coordinates");
        if (!type || !coordinates || *pipe == 6) { return {}; }
        std::vector<int64_t> payload{*type, *pipe};
        for (const auto& value : *coordinates) {
            auto integer = value.getAsInteger(); if (!integer) { return {}; }
            payload.push_back(*integer);
        }
        result.payloads.push_back(std::move(payload)); rows[*pipe];
    }
    if (result.payloads.size() != n) { return {}; }
    const auto vertices = n + events->size();
    std::vector<std::vector<bool>> reach(vertices, std::vector<bool>(vertices));
    std::map<std::vector<int64_t>, std::size_t> sets, waits;
    std::size_t payload = 0, command = 2 * n;
    for (const auto& raw : *events) {
        const auto& event = *raw.getAsObject();
        const auto kind = *event.getString("kind"); const auto pipe = *event.getInteger("pipe");
        if (kind == "payload") {
            reach[2 * payload][2 * payload + 1] = true;
            rows[pipe].push_back({2 * payload++, kind}); continue;
        }
        auto gap = event.getInteger("gap");
        if (!gap || *gap < 0 || static_cast<uint64_t>(*gap) != payload ||
            (kind != "set" && kind != "wait" && kind != "barrier")) { return {}; }
        if (kind != "barrier") {
            std::vector<int64_t> identity;
            for (StringRef field : {"source_pipe", "target_pipe", "plan", "record", "source_ordinal"}) {
                auto value = event.getInteger(field); if (!value) { return {}; }
                identity.push_back(*value);
            }
            if (identity[0] < 0 || identity[0] >= 6 || identity[1] < 0 || identity[1] >= 6 ||
                identity[0] == identity[1] || pipe != identity[kind == "set" ? 0 : 1]) { return {}; }
            if (auto* members = event.getArray("members")) {
                for (const auto& rawMember : *members) {
                    auto member = rawMember.getAsInteger(); if (!member) { return {}; }
                    identity.push_back(*member);
                }
            }
            auto& table = kind == "set" ? sets : waits;
            if (!table.emplace(std::move(identity), command).second) { return {}; }
        }
        if (kind == "barrier" && pipe == 6) {
            for (auto& row : rows) { row.second.push_back({command, kind}); }
        } else { rows[pipe].push_back({command, kind}); }
        ++command;
    }
    if (sets.size() != waits.size()) { return {}; }
    for (const auto& entry : sets) {
        auto found = waits.find(entry.first); if (found == waits.end()) { return {}; }
        reach[entry.second][found->second] = true;
    }
    for (const auto& row : rows) {
        std::optional<std::size_t> previous;
        std::vector<std::size_t> commands, blockers;
        for (const auto& item : row.second) {
            if (item.kind == "payload") {
                if (previous) { reach[*previous][item.vertex] = reach[*previous + 1][item.vertex + 1] = true; }
                for (auto blocker : blockers) { reach[blocker][item.vertex] = true; }
                previous = item.vertex;
            } else {
                const bool completes = item.kind == "set" || item.kind == "barrier";
                if (previous) {
                    reach[*previous][item.vertex] = true;
                    if (completes) { reach[*previous + 1][item.vertex] = true; }
                }
                for (auto blocker : blockers) { reach[blocker][item.vertex] = true; }
                if (completes) { for (auto prior : commands) { reach[prior][item.vertex] = true; } }
                commands.push_back(item.vertex);
                if (item.kind == "wait" || item.kind == "barrier") { blockers.push_back(item.vertex); }
            }
        }
    }
    for (std::size_t middle = 0; middle < vertices; ++middle) {
        for (std::size_t from = 0; from < vertices; ++from) {
            if (!reach[from][middle]) { continue; }
            for (std::size_t to = 0; to < vertices; ++to) { reach[from][to] = reach[from][to] || reach[middle][to]; }
        }
    }
    for (std::size_t i = 0; i < vertices; ++i) { if (reach[i][i]) { return {}; } }
    reach.resize(2 * n); for (auto& row : reach) { row.resize(2 * n); }
    result.reach = std::move(reach);
    return result;
}
inline bool logicalSignatureControls()
{
    auto fixture = llvm::json::parse(R"json({"error":"","payloads":3,"events":[
      {"kind":"payload","pipe":0,"type":0,"coordinates":[]},
      {"kind":"set","pipe":0,"gap":1,"source_pipe":0,"target_pipe":1,
       "plan":0,"record":0,"source_ordinal":0},
      {"kind":"payload","pipe":0,"type":1,"coordinates":[]},
      {"kind":"wait","pipe":1,"gap":2,"source_pipe":0,"target_pipe":1,
       "plan":0,"record":0,"source_ordinal":0},
      {"kind":"payload","pipe":1,"type":2,"coordinates":[]}]})json");
    if (!fixture) { llvm::consumeError(fixture.takeError()); return false; }
    if (!fixture->getAsObject()) { return false; }
    auto& trace = *fixture->getAsObject();
    const auto signature = logicalSignature(trace);
    // SET captures A's completion, not the later same-pipe B completion.
    if (!signature || !signature->reach[1][4] || signature->reach[3][4] || signature->reach[5][0]) { return false; }
    auto* events = trace.getArray("events");
    auto& wait = *(*events)[3].getAsObject();
    wait["source_ordinal"] = 1;
    return !logicalSignature(trace); // An unmatched logical endpoint is never ignored.
}
inline std::optional<std::vector<LogicalSignature>> numericalLogicalTraces(
    func::FuncOp original, pto::GMAliasPolicy policy, std::size_t region, bool warm, StringRef expectedBackend)
{
    // Analyze and insert on a clone: audit work cannot mutate original source
    // or production session counters/owners checked by the caller.
    auto parent = original->getParentOfType<ModuleOp>();
    if (!parent || !logicalSignatureControls()) { return {}; }
    OwningOpRef<ModuleOp> module(cast<ModuleOp>(parent->clone()));
    auto function = module->lookupSymbol<func::FuncOp>(original.getSymName());
    fs::FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return {}; }
    if (warm && !session.analyzeNumericalRegion({region}).mathematical) { return {}; }
    fs::AnalysisRequest request{region}; request.needs.queries = request.needs.selectors = true;
    if (session.analyze(request).status != fs::AnalysisStatus::Ready) { return {}; }
    request.needs.synchronization = true;
    auto selected = session.analyze(request);
    if (!selected.mathematical || selected.mathematical->backend != expectedBackend) { return {}; }
    auto prepared = session.prepareLogical(selected);
    if (selected.status != fs::AnalysisStatus::Ready || failed(prepared) ||
        failed(fs::insertLogicalSynchronization(function, **prepared)) || failed(verify(function))) { return {}; }
    std::vector<LogicalSignature> result;
    for (int64_t bound : {0, 1, 3, 8}) {
        Builder builder(function.getContext());
        function->setAttr("test.trace_arguments", builder.getDenseI64ArrayAttr(
            function.getNumArguments() ? ArrayRef<int64_t>(bound) : ArrayRef<int64_t>{}));
        auto trace = traceStructuredLogicalInsertion(function, session.input()->instructions());
        auto signature = logicalSignature(trace);
        if (!signature) { return {}; }
        result.push_back(std::move(*signature));
    }
    return result;
}
} // namespace numerical_request_oracle
#endif
