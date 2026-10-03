// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite event-graph oracle; production import never unfolds trips.
#include "../../lib/PTO/Transforms/FrontierSynch/RegionalRequests.h"
#include "../../lib/PTO/Transforms/FrontierSynch/StationaryCells.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
namespace {
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
using Graph = SmallVector<SmallVector<unsigned char>>;
std::string printed(Operation* op)
{
    std::string text;
    llvm::raw_string_ostream output(text);
    op->print(output);
    return text;
}
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) { graph[a][b] |= graph[a][k] && graph[k][b]; }
        }
    }
}
bool physicalOverlap(const pto::BaseMemInfo* a, const pto::BaseMemInfo* b)
{
    return a->scope == b->scope && a->baseAddresses.front() < b->baseAddresses.front() + b->allocateSize &&
           b->baseAddresses.front() < a->baseAddresses.front() + a->allocateSize;
}
bool conflict(const pto::CompoundInstanceElement* a, const pto::CompoundInstanceElement* b)
{
    auto overlap = [](auto source, auto target) {
        for (auto* x : source) { for (auto* y : target) { if (physicalOverlap(x, y)) { return true; } } }
        return false;
    };
    return overlap(a->defVec, b->useVec) || overlap(a->useVec, b->defVec) || overlap(a->defVec, b->defVec);
}
bool oracle(const fs::SelectedAnalysis& selected, unsigned trips)
{
    auto phases = selected.periodic.sites();
    std::size_t count = phases.size() * trips;
    Graph native(2 * count, SmallVector<unsigned char>(2 * count));
    DenseMap<pto::PipelineType, std::size_t> previous;
    for (std::size_t a = 0; a < count; ++a) {
        auto* phase = phases[a % phases.size()];
        native[2 * a][2 * a + 1] = 1;
        auto found = previous.find(phase->kPipeValue);
        if (found != previous.end()) {
            native[2 * found->second][2 * a] = 1;
            native[2 * found->second + 1][2 * a + 1] = 1;
        }
        previous[phase->kPipeValue] = a;
    }
    Graph required = native;
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = a + 1; b < count; ++b) {
            if (conflict(phases[a % phases.size()], phases[b % phases.size()])) { required[2 * a + 1][2 * b] = 1; }
        }
    }
    close(native); close(required);
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = a; b < count; ++b) {
            std::size_t source = 2 * a + 1, target = 2 * b;
            bool cover = required[source][target] && !native[source][target];
            for (std::size_t k = 0; k < required.size(); ++k) {
                if (k != source && k != target && required[source][k] && required[k][target]) { cover = false; }
            }
            bool retained = false;
            for (auto id : selected.periodic.retained()) {
                const auto& edge = selected.periodic.generators()[id].edge;
                retained |= edge.source == a % phases.size() && edge.consumer == b % phases.size() &&
                            edge.distance == llvm::DynamicAPInt(b / phases.size() - a / phases.size());
            }
            auto reaches = selected.periodic.reaches(a % phases.size(), b % phases.size(),
                fs::PeriodicEventKind::Start, llvm::DynamicAPInt(b / phases.size() - a / phases.size()));
            if (retained != cover || failed(reaches) || *reaches != static_cast<bool>(required[source][target])) {
                llvm::errs() << "stationary oracle mismatch trips=" << trips << " source=" << a << " target=" << b
                             << " cover=" << cover << " retained=" << retained << "\n";
                return false;
            }
        }
    }
    if (selected.reachability) {
        for (std::size_t a = 0; a < count; ++a) {
            for (std::size_t b = 0; b < count; ++b) {
                for (unsigned sourceKind : {0U, 1U}) {
                    for (unsigned targetKind : {0U, 1U}) {
                        auto kind = [](unsigned value) {
                            return value ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start;
                        };
                        auto queried = selected.reachability->contains(
                            {{selected.sites[a % phases.size()], kind(sourceKind)},
                             {llvm::DynamicAPInt(a / phases.size())}},
                            {{selected.sites[b % phases.size()], kind(targetKind)},
                             {llvm::DynamicAPInt(b / phases.size())}}, {llvm::DynamicAPInt(trips)});
                        bool expected = (a == b && sourceKind == targetKind) ||
                                        required[2 * a + sourceKind][2 * b + targetKind];
                        if (!queried.succeeded() || queried.value != expected) { return false; }
                    }
                }
            }
        }
    }
    return true;
}
bool check(func::FuncOp function)
{
    pto::SyncInput input;
    if (failed(input.build(function))) { return false; }
    auto trace = fs::TraceDemandAnalysis::build(function, input);
    if (failed(trace)) { return false; }
    scf::ForOp loop;
    function.walk([&](scf::ForOp candidate) { loop = candidate; });
    if (!loop) { return false; }
    auto original = printed(function);
    fs::CostLedger costs(true);
    fs::RegionalRequests requests(function, input, *trace, costs);
    auto needs = fs::AnalysisNeeds::modeledCovers();
    auto result = requests.request(loop, requests.contextFor(loop), fs::RegionalMode::Modeled, needs);
    if (!result.analysis || result.analysis->contract.effects != fs::EffectDomain::SharedModeled) { return false; }
    bool fallback = function.getName() == "unknown_bounds";
    if (fallback) {
        if (result.analysis->stationary || result.analysis->route != "regional-periodic-quotient") {
            return false;
        }
    } else {
        if (!result.analysis->stationary || result.analysis->route != "regional-stationary-cells" ||
            result.analysis->stationary->sitePairChecks() !=
                result.analysis->sites.size() * result.analysis->sites.size()) {
            return false;
        }
        for (unsigned trips : {0U, 1U, 2U, 4U}) {
            if (!oracle(*result.analysis, trips)) { return false; }
        }
    }
    auto queryNeeds = needs;
    queryNeeds.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionQueries);
    auto symbolic = requests.request(loop, requests.contextFor(loop), fs::RegionalMode::Modeled, queryNeeds,
                                    fs::RegionalRepresentation::ArithmeticRelations);
    if (!symbolic.analysis || symbolic.analysis->kind != fs::SelectedAnalysis::Kind::Periodic ||
        !symbolic.analysis->reachability || !symbolic.analysis->minimum) { return false; }
    if (!fallback) {
        for (unsigned trips : {0U, 1U, 2U, 4U}) {
            if (!oracle(*symbolic.analysis, trips)) { return false; }
        }
    }
    auto absent = symbolic.analysis->reachability->contains(
        {{symbolic.analysis->sites.front(), fs::PeriodicEventKind::Start}, {llvm::DynamicAPInt(0)}},
        {{symbolic.analysis->sites.front(), fs::PeriodicEventKind::Start}, {llvm::DynamicAPInt(0)}},
        {llvm::DynamicAPInt(0)});
    if (!absent.succeeded() || absent.value) { return false; }
    auto hits = requests.cacheHits();
    auto repeated = requests.request(loop, requests.contextFor(loop), fs::RegionalMode::Modeled, needs);
    if (repeated.analysis != result.analysis || requests.cacheHits() != hits + 1) { return false; }
    std::string reason;
    SmallVector<fs::AnalysisAttempt> attempts;
    auto selected = fs::selectAnalysis(function, input, *trace, attempts, reason, costs);
    bool upperRequired = function.getName() == "rmw";
    // Exact stationary F* remains checked above. Repeated independent V-pipe
    // RMW sites require the sound-upper direct realization: the self-site
    // original demand skips the other payload on this pipe.
    if (!selected || (!fallback && !upperRequired && !selected->stationary) ||
        (upperRequired && (!selected->upper || selected->contract.closure != fs::SelectedClosure::SoundUpper))) {
        llvm::errs() << "stationary selector obligation: " << reason << "\n"; return false;
    }
    if (selected->contract.effects != fs::EffectDomain::SharedModeled || printed(function) != original) {
        return false;
    }
    return true;
}
} // namespace
int runStationaryCellChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
    if (!module) { return 1; }
    unsigned tested = 0;
    for (auto function : module->getOps<mlir::func::FuncOp>()) {
        if (function.isExternal()) { continue; }
        if (!check(function)) { llvm::errs() << "stationary check failed: " << function.getName() << "\n"; return 1; }
        ++tested;
    }
    if (tested != 4) { return 1; }
    llvm::outs() << "verified stationary shared-cell import, modeled hazard certificate, "
                    "independent finite closure/covers, fallback and rollback\n";
    return 0;
}
