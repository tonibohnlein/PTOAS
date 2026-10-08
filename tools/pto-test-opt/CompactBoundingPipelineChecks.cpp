// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Execute prepared endpoint copies through original, independently chosen arms.
// Independent dynamic command graph: no producer demand/certificate is used by
// the oracle. Fixture labels denote whole-cell read/write behavior.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/CompactBoundingInsertion.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "SyncLogicalInsertionChecks.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Graph = std::vector<std::vector<bool>>;
struct Event {
    std::string kind, label, identity, occurrence;
    int64_t pipe = -1, source = -1, target = -1;
    std::size_t node = 0;
};
struct Execution {
    std::vector<Event> events;
    std::vector<std::size_t> payloads;
    Graph order;
    std::string error;
};
std::string render(func::FuncOp function)
{
    std::string result; llvm::raw_string_ostream stream(result); function.print(stream); return result;
}
std::string jsonText(const llvm::json::Value& value)
{
    std::string result; llvm::raw_string_ostream stream(result); stream << value; return result;
}
Execution decode(const llvm::json::Object& trace)
{
    Execution result;
    const auto* events = trace.getArray("events");
    if (!events || !trace.getString("error").value_or("missing status").empty()) {
        result.error = "structured interpretation failed"; return result;
    }
    std::size_t nodes = 0;
    for (const auto& item : *events) {
        const auto* object = item.getAsObject();
        if (!object || !object->getString("kind") || !object->getInteger("pipe")) {
            result.error = "malformed trace event"; return result;
        }
        Event event;
        event.kind = object->getString("kind")->str(); event.pipe = *object->getInteger("pipe");
        event.node = nodes++;
        if (event.kind == "payload") {
            event.label = object->getString("label").value_or("").str();
            const auto* coordinates = object->get("coordinates");
            if (event.label.empty() || !coordinates) { result.error = "missing fixture role"; return result; }
            event.occurrence = event.label + jsonText(*coordinates);
            result.payloads.push_back(result.events.size()); ++nodes;
        } else if (event.kind == "set" || event.kind == "wait") {
            event.source = object->getInteger("source_pipe").value_or(-1);
            event.target = object->getInteger("target_pipe").value_or(-1);
            if (auto physical = object->getInteger("physical_id")) {
                if (*physical < 0 || *physical >= 6) { result.error = "ID outside shared pool"; return result; }
                event.identity = "physical:" + std::to_string(*physical);
            } else {
                for (const auto* name : {"plan", "record", "source_ordinal"}) {
                    auto value = object->getInteger(name);
                    if (!value) { result.error = "missing logical identity"; return result; }
                    event.identity += std::to_string(*value) + ":";
                }
                if (const auto* members = object->get("members")) { event.identity += jsonText(*members); }
            }
        } else if (event.kind != "barrier") {
            result.error = "unknown command"; return result;
        }
        result.events.push_back(std::move(event));
    }
    result.order.assign(nodes, std::vector<bool>(nodes));
    return result;
}
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) {
                graph[a][b] = graph[a][b] || (graph[a][k] && graph[k][b]);
            }
        }
    }
}
bool commandGraph(Execution& run)
{
    if (!run.error.empty()) { return false; }
    auto& graph = run.order;
    std::map<std::string, std::size_t> pending, lastWait;
    std::vector<std::pair<std::size_t, std::size_t>> reuse;
    for (std::size_t i = 0; i < run.events.size(); ++i) {
        const auto& b = run.events[i];
        graph[b.node][b.node] = true;
        if (b.kind == "payload") { graph[b.node][b.node+1] = graph[b.node+1][b.node+1] = true; }
        for (std::size_t j = 0; j < i; ++j) {
            const auto& a = run.events[j];
            if (a.pipe != b.pipe) { continue; }
            if (a.kind == "payload" && b.kind == "payload") {
                graph[a.node][b.node] = graph[a.node+1][b.node+1] = true;
            } else if (a.kind == "payload") {
                graph[a.node][b.node] = true;
                if (b.kind == "set" || b.kind == "barrier") { graph[a.node+1][b.node] = true; }
            } else {
                graph[a.node][b.node] = true;
            }
        }
        if (b.kind == "set") {
            if (!pending.emplace(b.identity, i).second) { run.error = "overwritten live notification"; return false; }
            if (auto previous = lastWait.find(b.identity); previous != lastWait.end()) {
                reuse.emplace_back(run.events[previous->second].node, b.node);
            }
        } else if (b.kind == "wait") {
            auto source = pending.find(b.identity);
            if (source == pending.end()) { run.error = "unmatched WAIT"; return false; }
            const auto& set = run.events[source->second];
            if (set.source != b.source || set.target != b.target) { run.error = "direction mismatch"; return false; }
            graph[set.node][b.node] = true; pending.erase(source); lastWait[b.identity] = i;
        }
    }
    if (!pending.empty()) { run.error = "unconsumed SET"; return false; }
    close(graph);
    for (const auto& pair : reuse) {
        if (!graph[pair.first][pair.second]) { run.error = "reuse lacks causal WAIT before SET"; return false; }
    }
    return true;
}
bool storageCovered(const Execution& run)
{
    for (std::size_t a = 0; a < run.payloads.size(); ++a) {
        const auto& source = run.events[run.payloads[a]];
        for (std::size_t b = a+1; b < run.payloads.size(); ++b) {
            const auto& target = run.events[run.payloads[b]];
            // All fixture writers are on V and all reads on S, at one cell.
            // Thus every pair containing a writer is an original demand.
            const bool writer = source.label.front() == 'W' || target.label.front() == 'W';
            if (writer && !run.order[source.node+1][target.node]) { return false; }
        }
    }
    return true;
}
bool samePayloads(const Execution& a, const Execution& b)
{
    if (a.payloads.size() != b.payloads.size()) { return false; }
    for (std::size_t i = 0; i < a.payloads.size(); ++i) {
        if (a.events[a.payloads[i]].occurrence != b.events[b.payloads[i]].occurrence) { return false; }
    }
    return true;
}
bool sameOrder(const Execution& a, const Execution& b)
{
    if (!samePayloads(a, b)) { return false; }
    for (std::size_t i = 0; i < a.payloads.size(); ++i) {
        for (std::size_t j = 0; j < a.payloads.size(); ++j) {
            for (std::size_t x = 0; x < 2; ++x) {
                for (std::size_t y = 0; y < 2; ++y) {
                    if (a.order[a.events[a.payloads[i]].node+x][a.events[a.payloads[j]].node+y] !=
                        b.order[b.events[b.payloads[i]].node+x][b.events[b.payloads[j]].node+y]) { return false; }
                }
            }
        }
    }
    return true;
}
std::vector<Execution> traces(func::FuncOp function, const pto::SyncInput& input, bool commands)
{
    std::vector<Execution> result;
    for (const auto& counts : std::vector<std::vector<int64_t>>{{0,0}, {1,0}, {1,1}, {2,1}, {2,2}, {3,2}}) {
        SmallVector<int64_t> arguments(function.getNumArguments(), 0);
        for (std::size_t i = 0; i < arguments.size() && i < counts.size(); ++i) { arguments[i] = counts[i]; }
        function->setAttr("test.trace_arguments", DenseI64ArrayAttr::get(function.getContext(), arguments));
        auto run = decode(traceStructuredLogicalInsertion(function, input.instructions()));
        if (commands && !commandGraph(run) && run.error.empty()) { run.error = "command graph failed"; }
        result.push_back(std::move(run));
    }
    return result;
}
bool retainedQueries(const fs::PreparedLogicalPlan& plan)
{
    const auto& owner = plan.compactBoundingOwner;
    if (!owner || !owner->boundary || owner->allocationError.empty() || plan.allocationCertificate) { return false; }
    const auto& bounds = owner->boundary->bounds();
    if (!bounds.upper || !bounds.lower) { return false; }
    const auto& region = bounds.upper->regional();
    if (!region.expressions || region.anchors.empty()) { return false; }
    auto& arena = *region.expressions;
    fs::RegionalEvent event{0, arena.constant(0), fs::PeriodicEventKind::Start};
    if (!region.outerLoops.empty()) { event.visits.assign(region.outerLoops.front().size(), arena.constant(0)); }
    return fs::regionalReachability(region, event, event).has_value();
}
bool missingAllocation(func::FuncOp function)
{
    const auto before = render(function);
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    ScopedDiagnosticHandler capture(function.getContext(), [&](Diagnostic& message) {
        message.print(stream); return success();
    });
    const bool allocated = succeeded(fs::allocatePhysicalEventIds(function, {0,1,2,3,4,5}));
    stream.flush();
    return !allocated && render(function) == before &&
        diagnostic.find("physical allocation requires a supported finite or uniform allocation export; "
                        "allocation adapter not implemented yet") != std::string::npos &&
        diagnostic.find("supplied capacity") == std::string::npos;
}
// An unavailable export must not change the mathematical result or trigger
// a different selected order. This guard is deliberately defined after its SET
// cut and cannot be speculated because division may be undefined.
bool checkUnmetExports(func::FuncOp function, const pto::SyncInput& input)
{
    const auto before = render(function);
    fs::FrontierAnalysis analysis(function);
    if (analysis.hasWholeFunctionMinimumDemands() || failed(analysis.initialize(input.memory().gmPolicy()))) {
        return false;
    }
    auto* demands = analysis.analyzeSequenceFunction();
    if (!demands || !demands->error.empty() || !analysis.hasWholeFunctionMinimumDemands()) { return false; }
    auto region = fs::sequenceRegionalResult(*demands);
    if (region.anchors.size() != 2) { return false; }
    fs::RegionalEvent writer{0, region.expressions->constant(0), fs::PeriodicEventKind::Completion};
    fs::RegionalEvent reader{1, region.expressions->constant(0), fs::PeriodicEventKind::Start};
    const auto query = fs::regionalReachability(region, writer, reader);
    if (!query || region.expressions->constantValue(*query) == 0 ||
        succeeded(fs::prepareSequenceInsertion(*demands)) ||
        !analysis.hasWholeFunctionMinimumDemands() || analysis.analyzeSequenceFunction() != demands ||
        fs::regionalReachability(region, writer, reader) != query) { return false; }
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    ScopedDiagnosticHandler capture(function.getContext(), [&](Diagnostic& message) {
        message.print(stream); return success();
    });
    const auto prepared = fs::prepareFunctionSynchronization(function, input.memory().gmPolicy());
    stream.flush();
    return failed(prepared) && before == render(function) &&
        diagnostic.find("unmet-exports: exact whole-function demands retained") != std::string::npos &&
        diagnostic.find("compact bounding:") == std::string::npos;
}
bool check(func::FuncOp function, const pto::SyncInput& input)
{
    if (function->hasAttr("test.unmet_exports")) { return checkUnmetExports(function, input); }
    if (function->hasAttr("test.exact_priority")) {
        fs::FrontierAnalysis analysis(function);
        if (failed(analysis.initialize(input.memory().gmPolicy())) ||
            failed(analysis.analyzeArithmeticFunction()) || !analysis.hasWholeFunctionMinimumDemands()) {
            return false;
        }
        const auto* exact = analysis.arithmeticDemands();
        const auto* general = analysis.generalArithmeticDemands();
        if (failed(analysis.analyzeArithmeticFunction()) || exact != analysis.arithmeticDemands() ||
            general != analysis.generalArithmeticDemands()) { return false; }
        const auto changedPolicy = input.memory().gmPolicy() == pto::GMAliasPolicy::MayAlias ?
            pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
        if (failed(analysis.initialize(changedPolicy)) || analysis.arithmeticDemands() ||
            analysis.generalArithmeticDemands() || analysis.hasWholeFunctionMinimumDemands() ||
            failed(analysis.analyzeArithmeticFunction()) || !analysis.hasWholeFunctionMinimumDemands()) {
            return false;
        }
    }
    auto original = traces(function, input, false);
    const auto before = render(function);
    FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>> prepared = failure();
    if (function->hasAttr("test.compact_direct")) {
        auto owned = std::make_shared<pto::SyncInput>(input.memory().gmPolicy());
        if (failed(owned->build(function))) { return false; }
        auto compact = fs::prepareCompactBoundingInsertion(function, std::move(owned));
        if (!compact.prepared || !compact.owner || !compact.error.empty() || !compact.exportError.empty()) {
            llvm::errs() << compact.error << "; " << compact.exportError << "\n"; return false;
        }
        prepared = std::move(compact.prepared);
    } else {
        prepared = fs::prepareFunctionSynchronization(function, input.memory().gmPolicy());
    }
    if (succeeded(prepared) && function->hasAttr("test.exact_priority") && (*prepared)->compactBoundingOwner) {
        llvm::errs() << "compact fallback displaced an existing exact route\n"; return false;
    }
    if (succeeded(prepared) && function->hasAttr("test.expect_compact_dispatch") &&
        !(*prepared)->compactBoundingOwner) {
        llvm::errs() << "expected natural exact-route miss to reach compact fallback\n"; return false;
    }
    const bool missingExport = function->hasAttr("test.missing_allocation_export");
    if (succeeded(prepared) && missingExport && !retainedQueries(**prepared)) {
        llvm::errs() << "missing allocation export discarded selected mathematics\n"; return false;
    }
    const bool noNotifications = function->hasAttr("test.no_notifications");
    if (succeeded(prepared) && noNotifications) {
        const auto certificate = (*prepared)->allocationCertificate;
        const auto budget = certificate ? certificate.getAs<IntegerAttr>("budget") : IntegerAttr{};
        const auto entries = certificate ? certificate.getAs<ArrayAttr>("entries") : ArrayAttr{};
        const auto plan = certificate ? certificate.getAs<IntegerAttr>("plan") : IntegerAttr{};
        if (!budget || budget.getInt() != 0 || !entries || !entries.empty() || !plan ||
            plan.getInt() != (*prepared)->planId || llvm::any_of((*prepared)->endpoints, [](const auto& endpoint) {
                return endpoint.kind != fs::LogicalCommandKind::Barrier;
            })) {
            llvm::errs() << "readonly plan lacks its explicit empty allocation proof\n"; return false;
        }
    }
    if (failed(prepared) || render(function) != before ||
        failed(fs::insertLogicalSynchronization(function, **prepared)) || failed(verify(function))) { return false; }
    auto logical = traces(function, input, true);
    for (std::size_t i = 0; i < logical.size(); ++i) {
        if (!logical[i].error.empty() || !samePayloads(original[i], logical[i]) || !storageCovered(logical[i])) {
            llvm::errs() << "logical case " << i << ": " << logical[i].error << "\n"; return false;
        }
    }
    if (missingExport) { return missingAllocation(function); }
    const auto logicalIR = render(function);
    const bool zeroAllocated = succeeded(fs::allocatePhysicalEventIds(function, {}));
    if (zeroAllocated != noNotifications || (!zeroAllocated && render(function) != logicalIR)) { return false; }
    const int64_t eligible[] = {0,1,2,3,4,5};
    if ((!noNotifications && failed(fs::allocatePhysicalEventIds(function, eligible))) || failed(verify(function))) {
        return false;
    }
    pto::SyncInput physicalInput(input.memory().gmPolicy());
    if (failed(physicalInput.build(function))) { return false; }
    auto physical = traces(function, physicalInput, true);
    for (std::size_t i = 0; i < physical.size(); ++i) {
        if (!physical[i].error.empty() || !sameOrder(logical[i], physical[i])) {
            llvm::errs() << "physical case " << i << ": " << physical[i].error << "\n"; return false;
        }
    }
    return true;
}
} // namespace
int runCompactBoundingPipelineChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (!check(function, input)) {
        llvm::errs() << "compact bounding pipeline failed: " << function.getSymName() << "\n"; return 1;
    }
    llvm::outs() << "compact bounding pipeline passed: " << function.getSymName() << "\n"; return 0;
}
