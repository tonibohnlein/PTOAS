// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check hierarchical identities through two sequence exports, without insertion.
#include "PTO/Transforms/FrontierSynch/RepeatedRegion.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Matrix = std::vector<std::vector<bool>>;
void closure(Matrix& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (std::size_t j = 0; j < graph.size(); ++j) {
                graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
            }
        }
    }
}
bool check(func::FuncOp function, Operation* anchor, scf::ForOp outer, scf::ForOp inner,
           unsigned trips, unsigned steps, unsigned mask, uint64_t& checked)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    const auto zero = e.constant(0);
    std::vector<pto::CompoundInstanceElement> phases;
    for (auto pipe : {pto::PipelineType::PIPE_MTE1, pto::PipelineType::PIPE_M, pto::PipelineType::PIPE_V}) {
        phases.emplace_back(phases.size(), SmallVector<const pto::BaseMemInfo*>{},
                            SmallVector<const pto::BaseMemInfo*>{}, pipe, anchor->getName());
        phases.back().elementOp = anchor;
    }
    fs::RegionalAnalysis body;
    body.expressions = arena; body.capabilities = {true, true, true, false};
    for (std::size_t t = 0; t < phases.size(); ++t) {
        body.anchors.push_back({&phases[t], {}, {anchor->getBlock(), anchor},
                                {anchor->getBlock(), anchor->getNextNode()}});
        body.occurrenceLoops.push_back({});
        fs::RegionalSelector selected{{uint32_t(t), zero, fs::PeriodicEventKind::Start}, e.boolean(mask & (1U << t))};
        body.firstPayloads[t].push_back(selected); body.lastPayloads[t].push_back(selected);
    }
    Matrix local(6, std::vector<bool>(6));
    for (unsigned t = 0; t < 3; ++t) {
        if (!(mask & (1U << t))) { continue; }
        local[2*t][2*t] = local[2*t+1][2*t+1] = local[2*t][2*t+1] = true;
    }
    local[1][2] = (mask & 3) == 3; closure(local);
    body.presence = [arena, zero, mask](fs::RegionalEvent a) -> std::optional<fs::RegionExpressions::Id> {
        if (a.type >= 3 || !a.visits.empty()) { return std::nullopt; }
        return arena->land(arena->boolean(mask & (1U << a.type)), arena->eq(a.ordinal, zero));
    };
    body.reachability = [arena, local, present = body.presence](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::RegionExpressions::Id> {
        auto pa = present(a), pb = present(b);
        if (!pa || !pb) { return std::nullopt; }
        auto i = 2*a.type + (a.kind == fs::PeriodicEventKind::Completion);
        auto j = 2*b.type + (b.kind == fs::PeriodicEventKind::Completion);
        return arena->land(arena->land(*pa, *pb), arena->boolean(local[i][j]));
    };
    fs::RegionalStorageBoundary cell;
    cell.cell = {pto::AddressSpace::LEFT, 0, 8};
    cell.firstWriters.push_back({{0, zero, fs::PeriodicEventKind::Start}, e.boolean(mask & 1)});
    cell.lastWriters = cell.firstWriters;
    cell.lastReaders[1].push_back({{1, zero, fs::PeriodicEventKind::Start}, e.boolean(mask & 2)});
    cell.firstReaders[1].push_back({{1, zero, fs::PeriodicEventKind::Start}, e.boolean((mask & 3) == 2)});
    body.storageBoundary.push_back(cell);
    auto once = fs::repeatInvariantRegion(function, inner, body, e.constant(steps));
    if (!once.error.empty()) { llvm::errs() << once.error << "\n"; return false; }
    auto twice = fs::repeatInvariantRegion(function, outer, once.regional, e.constant(trips));
    if (!twice.error.empty()) { llvm::errs() << twice.error << "\n"; return false; }
    const unsigned count = 3 * trips * steps;
    Matrix graph(2 * count, std::vector<bool>(2 * count));
    for (unsigned i = 0; i < count; ++i) {
        if (!(mask & (1U << (i % 3)))) { continue; }
        graph[2*i][2*i] = graph[2*i+1][2*i+1] = graph[2*i][2*i+1] = true;
        for (unsigned j = i+1; j < count; ++j) {
            if (!(mask & (1U << (j % 3)))) { continue; }
            if (i%3 == j%3) { graph[2*i][2*j] = graph[2*i+1][2*j+1] = true; }
            if (i%3 < 2 && j%3 < 2 && (i%3 == 0 || j%3 == 0)) { graph[2*i+1][2*j] = true; }
        }
    }
    closure(graph);
    auto event = [&](unsigned vertex) {
        const unsigned occurrence = vertex / 2, visit = occurrence / 3;
        return fs::RegionalEvent{occurrence % 3, zero, vertex % 2 ? fs::PeriodicEventKind::Completion :
            fs::PeriodicEventKind::Start, {e.constant(visit / steps), e.constant(visit % steps)}};
    };
    for (unsigned i = 0; i < 2*count; ++i) {
        for (unsigned j = 0; j < 2*count; ++j) {
            auto got = fs::regionalReachability(twice.regional, event(i), event(j));
            if (!got || e.constantValue(*got) != uint64_t(graph[i][j])) {
                llvm::errs() << "repeat mismatch " << trips << "," << steps << ":" << i << "," << j << "\n";
                return false;
            }
            ++checked;
        }
    }
    auto absent = fs::regionalPresence(twice.regional,
        {0, zero, fs::PeriodicEventKind::Start, {e.constant(trips), zero}});
    return absent && e.constantValue(*absent) == 0 && e.error().empty();
}
} // namespace
bool runRepeatedRegionChecks(func::FuncOp function)
{
    Operation* anchor = nullptr;
    function.walk([&](Operation* op) { if (op->hasAttr("test.nested")) { anchor = op; } });
    if (!anchor) { return false; }
    auto inner = anchor->getParentOfType<scf::ForOp>();
    auto outer = inner ? inner->getParentOfType<scf::ForOp>() : scf::ForOp();
    if (!outer) { return false; }
    uint64_t checked = 0;
    for (unsigned t = 0; t <= 3; ++t) {
        for (unsigned k = 0; k <= 4; ++k) { for (unsigned mask = 0; mask < 8; ++mask) {
            if (!check(function, anchor, outer, inner, t, k, mask, checked)) { return false; }
        } }
    }
    llvm::outs() << "repeated region checked " << checked << " event pairs\n";
    return true;
}
