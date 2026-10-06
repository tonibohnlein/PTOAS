// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RepeatedPhases.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Matrix = std::vector<std::vector<bool>>;
void close(Matrix& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (std::size_t j = 0; j < graph.size(); ++j) {
                graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
            }
        }
    }
}
unsigned mask(unsigned phase, unsigned profile)
{
    if (profile == 1) { return phase % 2 ? 2 : 1; }
    if (profile == 2) { return phase == 0 ? 0 : 3; }
    return profile == 3 ? 0 : 3;
}
bool selected(fs::RegionExpressions& e, const std::vector<fs::RegionalSelector>& values,
              unsigned q, std::optional<unsigned> expected)
{
    std::optional<unsigned> found;
    for (const auto& value : values) {
        auto enabled = e.constantValue(value.present);
        if (!enabled || *enabled > 1) { return false; }
        if (!*enabled) { continue; }
        if (found || value.event.visits.size() != 1) { return false; }
        auto n = e.constantValue(value.event.visits.front());
        if (!n) { return false; }
        found = 2 * (*n * q + value.event.type / 2) + value.event.type % 2;
    }
    return found == expected;
}
bool check(func::FuncOp function, Operation* anchor, scf::ForOp loop,
           unsigned q, unsigned trips, unsigned profile, uint64_t& checked)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    auto zero = e.constant(0);
    std::vector<pto::CompoundInstanceElement> payloads;
    for (auto pipe : {pto::PipelineType::PIPE_MTE1, pto::PipelineType::PIPE_M}) {
        payloads.emplace_back(payloads.size(), SmallVector<const pto::BaseMemInfo*>{},
            SmallVector<const pto::BaseMemInfo*>{}, pipe, anchor->getName());
        payloads.back().elementOp = anchor;
    }
    std::vector<fs::RegionalAnalysis> phases;
    for (unsigned j = 0; j < q; ++j) {
        unsigned active = mask(j, profile);
        fs::RegionalAnalysis view;
        view.expressions = arena; view.capabilities = {true, true, true, false};
        for (unsigned type = 0; type < 2; ++type) {
            view.anchors.push_back({&payloads[type], {}, {anchor->getBlock(), anchor},
                {anchor->getBlock(), anchor->getNextNode()}});
            view.occurrenceLoops.push_back({});
            fs::RegionalSelector selector{{type, zero, fs::PeriodicEventKind::Start}, e.boolean(active & (1U << type))};
            auto pipe = static_cast<uint32_t>(payloads[type].kPipeValue);
            view.firstPayloads[pipe].push_back(selector); view.lastPayloads[pipe].push_back(selector);
        }
        view.presence = [arena, active, zero](fs::RegionalEvent event) -> std::optional<fs::RegionExpressions::Id> {
            if (event.type >= 2 || !event.visits.empty()) { return std::nullopt; }
            return arena->land(arena->boolean(active & (1U << event.type)), arena->eq(event.ordinal, zero));
        };
        view.reachability = [arena, present = view.presence](fs::RegionalEvent a, fs::RegionalEvent b)
            -> std::optional<fs::RegionExpressions::Id> {
            auto pa = present(a), pb = present(b);
            if (!pa || !pb) { return std::nullopt; }
            bool edge = a.type < b.type || (a.type == b.type &&
                (a.kind == fs::PeriodicEventKind::Start || b.kind == fs::PeriodicEventKind::Completion));
            return arena->land(arena->land(*pa, *pb), arena->boolean(edge));
        };
        fs::RegionalStorageBoundary cell;
        cell.cell = {pto::AddressSpace::LEFT, 8 * (j % 2), 8 * (j % 2 + 1)};
        cell.firstWriters.push_back({{0, zero, fs::PeriodicEventKind::Start}, e.boolean(active & 1)});
        cell.lastWriters = cell.firstWriters;
        auto readerPipe = static_cast<uint32_t>(payloads[1].kPipeValue);
        cell.firstReaders[readerPipe].push_back({{1, zero, fs::PeriodicEventKind::Start}, e.boolean(active == 2)});
        cell.lastReaders[readerPipe].push_back({{1, zero, fs::PeriodicEventKind::Start}, e.boolean(active & 2)});
        view.storageBoundary.push_back(cell);
        phases.push_back(std::move(view));
    }
    auto repeated = fs::repeatPhasedRegions(function, loop, std::move(phases), e.constant(trips));
    if (!repeated.error.empty()) { llvm::errs() << repeated.error << "\n"; return false; }
    if (repeated.regional.capabilities.endpointRecipes || repeated.regional.prepare ||
        repeated.regional.prepareWithVisits) { return false; }
    auto active = [&](unsigned occurrence) { return mask((occurrence / 2) % q, profile) & (1U << (occurrence % 2)); };
    const unsigned count = 2 * trips;
    Matrix graph(2 * count, std::vector<bool>(2 * count));
    for (unsigned i = 0; i < count; ++i) {
        if (!active(i)) { continue; }
        graph[2*i][2*i] = graph[2*i+1][2*i+1] = graph[2*i][2*i+1] = true;
        for (unsigned j = i+1; j < count; ++j) {
            if (!active(j)) { continue; }
            if (i%2 == j%2) { graph[2*i][2*j] = graph[2*i+1][2*j+1] = true; }
            if (((i/2)%q)%2 == ((j/2)%q)%2 && (i%2 == 0 || j%2 == 0)) { graph[2*i+1][2*j] = true; }
        }
    }
    close(graph);
    auto event = [&](unsigned vertex) {
        unsigned occurrence = vertex/2, visit = occurrence/2;
        return fs::RegionalEvent{2*(visit%q)+occurrence%2, zero, vertex%2 ? fs::PeriodicEventKind::Completion :
            fs::PeriodicEventKind::Start, {e.constant(visit/q)}};
    };
    for (unsigned i = 0; i < 2*count; ++i) {
        for (unsigned j = 0; j < 2*count; ++j) {
            auto value = fs::regionalReachability(repeated.regional, event(i), event(j));
            if (!value || e.constantValue(*value) != uint64_t(graph[i][j])) {
                llvm::errs() << "phase mismatch q=" << q << " T="
                             << trips << " profile=" << profile << " " << i << "," << j << "\n";
                return false;
            }
            ++checked;
        }
    }
    for (const auto& cell : repeated.regional.storageBoundary) {
        std::optional<unsigned> firstWriter, lastWriter, firstReader, lastReader;
        for (unsigned i = 0; i < count; ++i) {
            if (!active(i) || 8*(((i/2)%q)%2) != cell.cell.begin) { continue; }
            if (i%2 == 0) {
                if (!firstWriter) { firstWriter = i; }
                lastWriter = i; lastReader.reset();
            } else {
                if (!firstWriter && !firstReader) { firstReader = i; }
                lastReader = i;
            }
        }
        auto pipe = static_cast<uint32_t>(payloads[1].kPipeValue);
        auto readers = [&](const auto& table) -> std::vector<fs::RegionalSelector> {
            auto found = table.find(pipe);
            return found == table.end() ? std::vector<fs::RegionalSelector>{} : found->second;
        };
        if (!selected(e, cell.firstWriters, q, firstWriter) || !selected(e, cell.lastWriters, q, lastWriter) ||
            !selected(e, readers(cell.firstReaders), q, firstReader) ||
            !selected(e, readers(cell.lastReaders), q, lastReader)) {
            llvm::errs() << "phase selector mismatch q=" << q << " T="
                             << trips << " profile=" << profile << "\n";
            return false;
        }
    }
    return e.constructionError().empty();
}
} // namespace
bool runRepeatedPhaseChecks(func::FuncOp function)
{
    Operation* anchor = nullptr;
    function.walk([&](Operation* op) { if (op->hasAttr("test.nested")) { anchor = op; } });
    auto loop = anchor ? anchor->getParentOfType<scf::ForOp>() : scf::ForOp();
    if (!loop) { return false; }
    uint64_t checked = 0;
    for (unsigned q : {2U, 3U}) {
        for (unsigned trips : std::set<unsigned>{0, 1, q-1, q, q+1}) {
            for (unsigned profile = 0; profile < 4; ++profile) {
                if (!check(function, anchor, loop, q, trips, profile, checked)) { return false; }
            }
        }
    }
    llvm::errs() << "repeated phase oracle: " << checked << " event queries plus storage selectors\n";
    return true;
}
