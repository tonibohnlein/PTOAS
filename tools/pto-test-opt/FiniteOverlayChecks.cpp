// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent closure oracle and actual-IR insertion checks for finite overlays.
#include "PTO/Transforms/FrontierSynch/FiniteOverlayInsertion.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingRegional.h"
#include "SyncLogicalInsertionChecks.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <set>
namespace fs = mlir::pto::frontiersynch;
using namespace mlir;
namespace {
using Graph = std::array<std::array<bool, 6>, 6>;
using Pair = std::pair<unsigned, unsigned>;
void close(Graph& graph)
{
    for (unsigned via = 0; via < 6; ++via) {
        for (unsigned from = 0; from < 6; ++from) {
            for (unsigned to = 0; to < 6; ++to) {
                graph[from][to] = graph[from][to] || (graph[from][via] && graph[via][to]);
            }
        }
    }
}
Graph native(bool middle)
{
    Graph graph{};
    for (unsigned payload = 0; payload < 3; ++payload) {
        if (payload == 1 && !middle) { continue; }
        graph[2 * payload][2 * payload] = true;
        graph[2 * payload + 1][2 * payload + 1] = true;
        graph[2 * payload][2 * payload + 1] = true;
    }
    return graph;
}
fs::RegionalEvent event(unsigned number, fs::RegionExpressions::Id zero)
{
    return {number / 2, zero, number % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start};
}
bool valuation(unsigned mask, bool middle, uint64_t& checked)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    const auto zero = arena->constant(0);
    Graph baseGraph = native(middle);
    baseGraph[1][4] = true;
    close(baseGraph);
    auto valid = [arena, zero](fs::RegionalEvent value) {
        return value.type < 3 && value.ordinal == zero;
    };
    fs::RegionalAnalysis base;
    base.expressions = arena;
    base.capabilities.exactQueries = true;
    base.presence = [arena, valid, middle](fs::RegionalEvent value) -> std::optional<fs::RegionExpressions::Id> {
        if (!valid(value)) { return std::nullopt; }
        return arena->boolean(value.type != 1 || middle);
    };
    base.reachability = [arena, valid, baseGraph](fs::RegionalEvent from, fs::RegionalEvent to)
        -> std::optional<fs::RegionExpressions::Id> {
        if (!valid(from) || !valid(to)) { return std::nullopt; }
        const auto a = 2 * from.type + (from.kind == fs::PeriodicEventKind::Completion);
        const auto b = 2 * to.type + (to.kind == fs::PeriodicEventKind::Completion);
        return arena->boolean(baseGraph[a][b]);
    };
    const std::array<Pair, 4> pairs{{{1, 2}, {3, 4}, {1, 2}, {1, 4}}};
    std::vector<fs::FiniteOverlayDemand> added;
    auto expected = baseGraph;
    for (unsigned i = 0; i < pairs.size(); ++i) {
        auto [from, to] = pairs[i];
        const bool active = (mask & (1U << i)) && (i == 3 || middle);
        added.push_back({event(from, zero), event(to, zero), arena->boolean(mask & (1U << i))});
        expected[from][to] = expected[from][to] || active;
    }
    close(expected);
    fs::RegionalOrderQuery order = [arena, valid](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::RegionExpressions::Id> {
        if (!valid(a) || !valid(b)) { return std::nullopt; }
        return arena->boolean(a.type < b.type);
    };
    auto result = fs::analyzeFiniteOverlay(base, added, order);
    if (!result.error.empty() || result.retained.size() != added.size() || result.regional.prepare ||
        result.regional.prepareFiltered || result.regional.capabilities.endpointRecipes) { return false; }
    for (unsigned from = 0; from < 6; ++from) {
        for (unsigned to = 0; to < 6; ++to) {
            auto actual = result.regional.reachability(event(from, zero), event(to, zero));
            if (!actual || arena->constantValue(*actual) != std::optional<uint64_t>(expected[from][to])) {
                return false;
            }
            ++checked;
        }
    }
    std::set<Pair> covers;
    for (unsigned from = 1; from < 6; from += 2) {
        for (unsigned to = 0; to < 6; to += 2) {
            bool cover = expected[from][to];
            for (unsigned via = 0; via < 6; ++via) {
                if (via != from && via != to && expected[from][via] && expected[via][to]) { cover = false; }
            }
            if (cover) { covers.emplace(from, to); }
        }
    }
    std::set<Pair> actual;
    auto baseRetained = result.retainBase(event(1, zero), event(4, zero));
    if (!baseRetained || !arena->constantValue(*baseRetained)) { return false; }
    if (*arena->constantValue(*baseRetained)) { actual.emplace(1, 4); }
    for (unsigned i = 0; i < pairs.size(); ++i) {
        auto retained = arena->constantValue(result.retained[i]);
        if (!retained || (*retained && !actual.insert(pairs[i]).second)) { return false; }
    }
    if (actual != covers) { return false; }
    if (result.regional.reachability({3, zero, fs::PeriodicEventKind::Start}, event(0, zero))) { return false; }
    auto backward = fs::analyzeFiniteOverlay(base,
        {{event(5, zero), event(0, zero), arena->boolean(true)}}, order);
    auto wrongSort = fs::analyzeFiniteOverlay(base, {{event(1, zero), event(4, zero), zero}}, order);
    return !backward.error.empty() && !wrongSort.error.empty();
}
std::string render(func::FuncOp function)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    function.print(stream);
    return text;
}
} // namespace
int runFiniteOverlayChecks()
{
    uint64_t checked = 0;
    for (unsigned mask = 0; mask < 16; ++mask) {
        for (bool middle : {false, true}) {
            if (!valuation(mask, middle, checked)) {
                llvm::errs() << "finite overlay oracle failed at " << mask << ", " << middle << "\n";
                return 1;
            }
        }
    }
    llvm::outs() << "finite overlay checked " << checked << " event pairs across 32 contexts\n";
    return 0;
}
LogicalResult runFiniteOverlayInsertionChecks(func::FuncOp function, pto::GMAliasPolicy policy)
{
    pto::SyncInput input(policy);
    fs::PhaseIndex index;
    if (failed(input.build(function)) || failed(index.build(function, input))) { return failure(); }
    SmallVector<Operation*> roots;
    for (auto& operation : function.front()) { roots.push_back(&operation); }
    const auto before = render(function);
    fs::RegionalAnalysis base;
    std::string baseError;
    scf::ForOp loop;
    for (auto* operation : roots) {
        if (auto candidate = dyn_cast<scf::ForOp>(operation)) {
            if (loop) { return failure(); }
            loop = candidate;
        }
    }
    if (loop) {
        auto recognition = fs::recognizeGuardedRotating(loop, index, input, input.accesses());
        auto periodic = fs::analyzeGuardedRotating(loop, input, recognition);
        baseError = periodic.error;
        if (!baseError.empty()) { llvm::errs() << baseError << "\n"; return failure(); }
        auto exported = fs::guardedRotatingRegionalResult(function, input, periodic, baseError);
        if (failed(exported)) { llvm::errs() << baseError << "\n"; return failure(); }
        base = std::move(*exported);
    } else {
        auto finite = fs::analyzeFiniteGuarded(function, roots, index, input);
        baseError = finite.error;
        base = fs::finiteGuardedRegionalResult(finite);
    }
    if (!baseError.empty() || !base.expressions || base.anchors.size() != 3 ||
        function.getNumArguments() < 2 || !function.getArgument(0).getType().isInteger(1)) { return failure(); }
    auto arena = base.expressions;
    const auto zero = arena->constant(0), guard = arena->input(function.getArgument(0));
    fs::RegionalOrderQuery order = [arena, zero](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::RegionExpressions::Id> {
        if (a.type >= 3 || b.type >= 3 || a.ordinal != zero || b.ordinal != zero) { return std::nullopt; }
        return arena->boolean(a.type < b.type);
    };
    auto overlay = fs::analyzeFiniteOverlay(base, {
        {event(1, zero), event(2, zero), guard}, {event(3, zero), event(4, zero), guard},
        {event(1, zero), event(2, zero), guard}}, order);
    std::string error = overlay.error;
    auto plan = fs::prepareFiniteOverlayInsertion(function, overlay, error);
    const bool unchanged = before == render(function);
    const bool noPhysicalCertificate = succeeded(plan) && !(**plan).allocationCertificate;
    const bool accepted = unchanged && noPhysicalCertificate && succeeded(plan) &&
        succeeded(fs::insertLogicalSynchronization(function, **plan)) && succeeded(verify(function));
    llvm::json::Object report{{"accepted", accepted}, {"unchanged_before_insertion", unchanged},
        {"no_physical_certificate", noPhysicalCertificate}, {"error", error}, {"base_error", baseError}};
    if (accepted) { report["trace"] = traceStructuredLogicalInsertion(function, input.instructions()); }
    llvm::outs() << llvm::json::Value(std::move(report)) << "\n";
    return success(accepted);
}
