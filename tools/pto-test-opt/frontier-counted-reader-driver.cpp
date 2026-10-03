// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Independent finite event closure; production never expands loop trips.
#include "../../lib/PTO/Transforms/FrontierSynch/RegionalRequests.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
namespace {
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
using Graph = SmallVector<SmallVector<unsigned char>>;
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) { graph[a][b] |= graph[a][k] && graph[k][b]; }
        }
    }
}
bool oracle(const fs::SelectedAnalysis& selected, unsigned trips, bool empty)
{
    SmallVector<fs::SignedPoint> word;
    for (unsigned i = 0; i < trips; ++i) {
        word.push_back({{selected.sites[0], std::nullopt}, {llvm::DynamicAPInt(i)}});
        for (unsigned j = 0; j < (empty ? 0 : i); ++j) {
            word.push_back({{selected.sites[1], std::nullopt}, {llvm::DynamicAPInt(i), llvm::DynamicAPInt(j)}});
        }
    }
    Graph native(2 * word.size(), SmallVector<unsigned char>(2 * word.size()));
    for (std::size_t a = 0; a < word.size(); ++a) {
        native[2 * a][2 * a + 1] = 1;
        for (std::size_t b = a + 1; b < word.size(); ++b) {
            if (word[a].tag.site == word[b].tag.site) {
                native[2 * a][2 * b] = 1; native[2 * a + 1][2 * b + 1] = 1;
            }
        }
    }
    auto required = native;
    for (std::size_t a = 0; a < word.size(); ++a) {
        for (std::size_t b = a + 1; b < word.size(); ++b) {
            if (word[a].tag.site == selected.sites[0] || word[b].tag.site == selected.sites[0]) {
                required[2 * a + 1][2 * b] = 1;
            }
        }
    }
    close(native); close(required);
    for (std::size_t a = 0; a < word.size(); ++a) {
        for (std::size_t b = 0; b < word.size(); ++b) {
            for (unsigned s : {0U, 1U}) {
                for (unsigned t : {0U, 1U}) {
                    auto source = word[a], target = word[b];
                    source.tag.kind = s ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start;
                    target.tag.kind = t ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start;
                    auto query = selected.reachability->contains(source, target, {llvm::DynamicAPInt(trips)});
                    bool expected = (a == b && s == t) || required[2 * a + s][2 * b + t];
                    if (!query.succeeded() || query.value != expected) { return false; }
                }
            }
            auto source = word[a], target = word[b];
            source.tag.kind = fs::PeriodicEventKind::Completion; target.tag.kind = fs::PeriodicEventKind::Start;
            bool cover = required[2 * a + 1][2 * b] && !native[2 * a + 1][2 * b];
            for (std::size_t k = 0; k < required.size(); ++k) {
                if (k != 2 * a + 1 && k != 2 * b && required[2 * a + 1][k] && required[k][2 * b]) { cover = false; }
            }
            auto minimum = selected.minimum->contains(source, target, {llvm::DynamicAPInt(trips)});
            if (!minimum.succeeded() || minimum.value != cover) { return false; }
        }
    }
    auto absent = selected.reachability->contains(
        {{selected.sites[0], fs::PeriodicEventKind::Start}, {llvm::DynamicAPInt(0)}},
        {{selected.sites[0], fs::PeriodicEventKind::Start}, {llvm::DynamicAPInt(0)}}, {llvm::DynamicAPInt(0)});
    return absent.succeeded() && !absent.value;
}
bool check(func::FuncOp function)
{
    pto::SyncInput input;
    if (failed(input.build(function))) { return false; }
    auto trace = fs::TraceDemandAnalysis::build(function, input);
    if (failed(trace)) { return false; }
    scf::ForOp outer;
    for (auto loop : function.getOps<scf::ForOp>()) { outer = loop; }
    if (!outer) { return false; }
    std::string before;
    llvm::raw_string_ostream text(before); function.print(text);
    fs::CostLedger costs(true);
    fs::RegionalRequests requests(function, input, *trace, costs);
    auto needs = fs::AnalysisNeeds::modeledCovers();
    needs.interfaces |= fs::interfaceBit(fs::DemandInterface::RegionQueries);
    auto result = requests.request(outer, requests.contextFor(outer), fs::RegionalMode::Modeled, needs);
    bool negative = function.getName() == "atomic_rejected";
    if (negative) {
        for (const auto& attempt : requests.attempts()) {
            if (attempt.route == "regional-counted-readers" && attempt.outcome == "ready") { return false; }
        }
    } else {
        if (!result.analysis || result.analysis->route != "regional-counted-readers" ||
            result.analysis->contract.effects != fs::EffectDomain::SharedModeled) { return false; }
        for (unsigned trips : {0U, 1U, 2U, 4U}) {
            if (!oracle(*result.analysis, trips, function.getName() == "empty")) { return false; }
        }
        auto matching = requests.request(outer, requests.contextFor(outer), fs::RegionalMode::Modeled, needs,
            fs::RegionalRepresentation::ArithmeticRelations, fs::RegionalPreparation::SelectorMatching);
        if (!matching.analysis || matching.analysis->endpoints.empty()) { return false; }
        if (function.getName() == "parent") {
            auto parent = requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs);
            if (!parent.analysis || parent.analysis->route != "structured-composition") { return false; }
            bool used = llvm::any_of(parent.analysis->regionalChildren, [](const auto& child) {
                return child->route == "regional-counted-readers";
            });
            if (!used) { return false; }
            std::optional<std::size_t> producer;
            for (auto [id, site] : llvm::enumerate(trace->sites())) {
                if (site.phase->elementOp->getName().getStringRef() == "pto.tmatmul") { producer = id; }
            }
            if (!producer) { return false; }
            auto* source = trace->sites()[*producer].phase;
            auto* target = trace->sites()[result.analysis->sites[0]].phase;
            pto::DepBaseMemInfoPairVec witnesses;
            if (!input.memory().DepBetween(source->defVec, target->useVec, witnesses) || witnesses.empty()) {
                return false;
            }
            auto ready = requests.request(function, requests.context(), fs::RegionalMode::Modeled, needs,
                fs::RegionalRepresentation::NativeSummaries, fs::RegionalPreparation::SelectorMatching);
            if (!ready.analysis) { return false; }
            auto outgoing = ready.analysis->endpoints.find({source->kPipeValue, target->kPipeValue, true});
            auto incoming = ready.analysis->endpoints.find({source->kPipeValue, target->kPipeValue, false});
            if (outgoing == ready.analysis->endpoints.end() || incoming == ready.analysis->endpoints.end()) {
                return false;
            }
            fs::SymbolicEvent publication{*producer, {}, fs::PeriodicEventKind::Completion};
            for (int trips : {0, 1, 2, 4}) {
                fs::SignedPoint start{{result.analysis->sites[0],
                                      fs::PeriodicEventKind::Start}, {llvm::DynamicAPInt(0)}};
                auto cover = ready.analysis->minimum->contains(
                    {{*producer, fs::PeriodicEventKind::Completion}, {}}, start, {llvm::DynamicAPInt(trips)});
                if (!cover.succeeded() || cover.value != (trips > 0)) { return false; }
                for (int i = 0; i < trips; ++i) {
                    start.coordinates[0] = llvm::DynamicAPInt(i);
                    auto reached = ready.analysis->reachability->contains(
                        {{*producer, fs::PeriodicEventKind::Completion}, {}}, start, {llvm::DynamicAPInt(trips)});
                    auto retained = ready.analysis->minimum->contains(
                        {{*producer, fs::PeriodicEventKind::Completion}, {}}, start, {llvm::DynamicAPInt(trips)});
                    if (!reached.succeeded() || !reached.value || !retained.succeeded() || retained.value != (i == 0)) {
                        return false;
                    }
                }
                auto acquired = outgoing->second->evaluate(publication, {llvm::DynamicAPInt(trips)});
                if (!acquired.succeeded() || static_cast<bool>(acquired.value) != (trips > 0)) { return false; }
                if (trips > 0) {
                    if (acquired.value->site != result.analysis->sites[0] ||
                        acquired.value->kind != fs::PeriodicEventKind::Start ||
                        acquired.value->coordinates.size() != 1 ||
                        acquired.value->coordinates[0] != llvm::DynamicAPInt(0)) { return false; }
                    auto published = incoming->second->evaluate(*acquired.value, {llvm::DynamicAPInt(trips)});
                    if (!published.succeeded() || !published.value || published.value->site != *producer ||
                        published.value->kind != fs::PeriodicEventKind::Completion ||
                        !published.value->coordinates.empty()) {
                        return false;
                    }
                }
            }

        }
    }
    std::string after;
    llvm::raw_string_ostream final(after); function.print(final);
    return before == after;
}
} // namespace
int runCountedReaderChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
    if (!module) { return 1; }
    unsigned count = 0;
    for (auto function : module->getOps<mlir::func::FuncOp>()) {
        if (!check(function)) {
            llvm::errs() << "counted reader check failed: " << function.getName() << "\n"; return 1;
        }
        ++count;
    }
    if (count != 4) { return 1; }
    llvm::outs() << "verified counted reader symbolic queries, first-last maps, zero bypass and parent composition\n";
    return 0;
}
