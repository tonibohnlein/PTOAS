// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/StorageLaneAllocation.h"
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeAllocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <tuple>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Graph = std::vector<std::vector<bool>>;
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (std::size_t j = 0; j < graph.size(); ++j) {
                graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
            }
        }
    }
}
bool rotating(MLIRContext& context, unsigned slots, unsigned stride, unsigned mask, uint64_t& checked)
{
    constexpr unsigned trips = 5, sites = 2, occurrences = trips * sites;
    auto present = [&](unsigned a) { return a % sites == 0 || (mask & (1U << (a / sites))); };
    auto cell = [&](unsigned a) { return (stride * (a / sites)) % slots; };
    Graph graph(2 * occurrences, std::vector<bool>(2 * occurrences));
    for (unsigned a = 0; a < occurrences; ++a) {
        if (!present(a)) { continue; }
        graph[2*a][2*a] = graph[2*a+1][2*a+1] = graph[2*a][2*a+1] = true;
        for (unsigned b = a + 1; b < occurrences; ++b) {
            if (!present(b)) { continue; }
            if (a % sites == b % sites) { graph[2*a][2*b] = graph[2*a+1][2*b+1] = true; }
            if (cell(a) == cell(b) && (!(a % sites) || !(b % sites))) { graph[2*a+1][2*b] = true; }
        }
    }
    close(graph);
    struct Handoff { unsigned source, target; uint64_t lane; };
    std::vector<Handoff> handoffs;
    for (unsigned visit = 0; visit < trips; ++visit) {
        fs::RegionExpressions e;
        Block symbols;
        auto ordinal = e.input(symbols.addArgument(IndexType::get(&context), UnknownLoc::get(&context)));
        fs::LifetimeWindowInput window;
        window.sites = sites; window.span = slots / std::gcd(slots, stride);
        std::vector<fs::StorageLaneCell> cells;
        for (unsigned slot = 0; slot < slots; ++slot) { cells.push_back({slot, 0, 0, 8, slots, stride, slot}); }
        for (unsigned i = 0; i < sites * (window.span + 1); ++i) {
            const bool on = visit + i / sites < trips && present(sites * visit + i);
            window.payloads.push_back({i % sites, e.boolean(on)});
            window.accesses.push_back({i, cell(i), e.boolean(i % sites != 0), e.boolean(i % sites == 0)});
        }
        auto analysis = fs::analyzeLifetimeWindow(e, window);
        auto allocated = fs::buildStorageLaneAllocation(e, window, analysis, cells, ordinal);
        if (!analysis.error.empty() || !allocated.error.empty() || allocated.budget > slots) { return false; }
        fs::RegionExpressions::Substitution atVisit({{ordinal, e.constant(visit)}});
        for (std::size_t r = 0; r < analysis.sourceDemands.size(); ++r) {
            const auto& edge = analysis.sourceDemands[r];
            if (e.constantValue(edge.guard) != 1 || edge.source % sites == edge.target % sites) { continue; }
            auto lane = e.constantValue(e.substitute(allocated.lanes[r], atVisit));
            const auto source = sites * visit + edge.source, target = sites * visit + edge.target;
            if (!lane || *lane != cell(source) || target >= occurrences) { return false; }
            handoffs.push_back({source, target, *lane});
        }
    }
    std::sort(handoffs.begin(), handoffs.end(), [](const auto& a, const auto& b) {
        return std::tie(a.source, a.target) < std::tie(b.source, b.target);
    });
    std::map<uint64_t, unsigned> last;
    for (const auto& handoff : handoffs) {
        auto prior = last.find(handoff.lane);
        if (prior != last.end()) {
            const bool samePipe = prior->second % sites == handoff.source % sites;
            if (!graph[2*prior->second + (samePipe ? 0 : 1)][2*handoff.source]) { return false; }
        }
        last[handoff.lane] = handoff.target; ++checked;
    }
    return true;
}
bool selection(MLIRContext& context)
{
    fs::RegionExpressions e;
    Block symbols;
    auto ordinal = e.input(symbols.addArgument(IndexType::get(&context), UnknownLoc::get(&context)));
    auto condition = e.input(symbols.addArgument(IntegerType::get(&context, 1), UnknownLoc::get(&context)));
    auto yes = e.boolean(true);
    fs::LifetimeWindowInput window;
    window.sites = 2; window.payloads = {{0, yes}, {1, yes}};
    fs::LifetimeWindowAnalysis analysis;
    analysis.sourceDemands = {{0, 1, yes}};
    analysis.sourceWitnesses = {{{0, fs::StorageHazard::RAW, 1, condition},
                                {1, fs::StorageHazard::RAW, 1, yes}}};
    std::vector<fs::StorageLaneCell> cells{{0, 0, 0, 8, 3, 2, 1}, {1, 1, 0, 8, 1, 0, 0}};
    auto allocated = fs::buildStorageLaneAllocation(e, window, analysis, cells, ordinal);
    if (!allocated.error.empty() || allocated.budget != 4) { return false; }
    for (bool choice : {false, true}) {
        for (uint64_t visit : {uint64_t(0), uint64_t(1), UINT64_MAX}) {
            fs::RegionExpressions::Substitution values({{ordinal, e.constant(visit)}, {condition, e.boolean(choice)}});
            auto actual = e.constantValue(e.substitute(allocated.lanes.front(), values));
            if (actual != (choice ? ((visit % 3) * 2 + 1) % 3 : 3)) { return false; }
        }
    }
    analysis.sourceWitnesses.front().pop_back();
    if (fs::buildStorageLaneAllocation(e, window, analysis, cells, ordinal).error.empty()) { return false; }
    auto partial = fs::buildStorageLaneAllocation(e, window, analysis, cells, ordinal, true);
    if (!partial.error.empty() || !partial.records.empty() || partial.budget) { return false; }
    analysis.sourceWitnesses = {{{0, fs::StorageHazard::Supplied, 1, yes}}};
    if (fs::buildStorageLaneAllocation(e, window, analysis, cells, ordinal).error.empty()) { return false; }
    analysis.sourceWitnesses = {{{0, fs::StorageHazard::RAW, 1, yes}}};
    cells.front().slots = 7;
    if (fs::buildStorageLaneAllocation(e, window, analysis, cells, ordinal).error.empty()) { return false; }
    cells.front().slots = 3; cells.back().family = 0; cells.back().begin = 4; cells.back().end = 12;
    cells.back().slots = 3; cells.back().stride = 2;
    return !fs::buildStorageLaneAllocation(e, window, analysis, cells, ordinal).error.empty();
}
bool composite(MLIRContext& context)
{
    fs::RegionExpressions e;
    auto yes = e.boolean(true), no = e.boolean(false);
    fs::LifetimeWindowInput window;
    window.sites = 2; window.span = 1;
    window.payloads = {{0, yes}, {1, yes}, {0, yes}, {1, yes}};
    window.accesses = {{0, 0, no, yes}, {1, 0, yes, no}, {2, 0, no, yes}, {3, 0, yes, no}};
    fs::LifetimeWindowAnalysis analysis;
    analysis.sourceDemands = {{0, 1, yes}, {1, 2, yes}};
    analysis.sourceWitnesses = {{{0, fs::StorageHazard::RAW, 1, yes}},
                                {{0, fs::StorageHazard::Supplied, 0, yes}}};
    auto selected = fs::buildStorageLaneAllocation(
        e, window, analysis, {{0, 0, 0, 8, 1, 0, 0}}, e.constant(0), true);
    if (!selected.error.empty() || selected.records != std::vector<uint32_t>{0} || selected.budget != 1) {
        return false;
    }
    Builder b(&context);
    OwningOpRef<func::FuncOp> function(func::FuncOp::create(
        b.getUnknownLoc(), "storage_lane_composite", b.getFunctionType({}, {})));
    auto storage = fs::storageLaneAllocationCertificate(*function, window, analysis.sourceDemands, selected, 9);
    auto residual = fs::boundedLifetimeAllocationCertificate(*function, e, window, {1, 1},
        {analysis.sourceDemands[1]}, 9, {1});
    auto combined = fs::combineStorageLaneCertificates(storage, residual);
    if (!combined) { return false; }
    auto groups = combined.getAs<ArrayAttr>("groups");
    if (!groups || groups.size() != 2) { return false; }
    for (unsigned i = 0; i < 2; ++i) {
        auto group = cast<DictionaryAttr>(groups[i]);
        auto records = group.getAs<DenseI64ArrayAttr>("records");
        auto rules = group.getAs<ArrayAttr>("tuple_rules");
        auto conflicts = group.getAs<DenseI64ArrayAttr>("conflicts");
        if (!records || records.size() != 1 || records[0] != i || !rules || rules.size() != 1 ||
            !conflicts || conflicts.size() != i || (i && conflicts[0] != 0)) { return false; }
        auto arity = cast<DictionaryAttr>(rules[0]).getAs<IntegerAttr>("coordinate_count");
        if (!arity || arity.getInt() != (i ? 1 : 2)) { return false; }
    }
    // Reusing an original record across proofs must never silently duplicate it.
    return !fs::combineStorageLaneCertificates(storage, storage);
}
} // namespace
int runStorageLaneAllocationChecks()
{
    MLIRContext context;
    context.disableMultithreading(); context.loadDialect<arith::ArithDialect, func::FuncDialect>();
    uint64_t checked = 0;
    for (unsigned slots = 1; slots <= 3; ++slots) {
        for (unsigned stride = 0; stride < slots; ++stride) {
            for (unsigned mask = 0; mask < 32; ++mask) {
                if (!rotating(context, slots, stride, mask, checked)) {
                    llvm::errs() << "storage lane rotating proof mismatch\n"; return 1;
                }
            }
        }
    }
    if (!selection(context) || !composite(context)) {
        llvm::errs() << "storage lane selector/composite validation mismatch\n"; return 1;
    }
    llvm::outs() << "storage lane checked " << checked << " handoffs with independent cell lifetimes\n";
    return 0;
}
