// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/RegionalRelations.h"
#include "PTO/Transforms/FrontierSynch/SequenceAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
struct Occurrence {
    unsigned type;
    DenseMap<Operation*, uint64_t> iterations;
    std::set<unsigned> reads, writes;
};
bool intersects(const std::set<unsigned>& a, const std::set<unsigned>& b)
{
    return llvm::any_of(a, [&](auto value) { return b.count(value); });
}
bool oracle(const fs::RegionalAnalysis& region, func::FuncOp function,
            ArrayRef<scf::ForOp> loops, scf::ForOp inner, uint64_t& checked)
{
    auto& e = *region.expressions;
    for (uint64_t trips : {0, 1, 2, 3}) {
        for (uint64_t length : {3}) {
            std::vector<Occurrence> occurrences;
            for (uint64_t t = 0; t < trips; ++t) {
                for (uint64_t i = 0; i < length; ++i) {
                    DenseMap<Operation*, uint64_t> coordinates{{loops[0], t}, {inner, i}};
                    occurrences.push_back({0, coordinates, {0}, {unsigned(1 + i % 2)}});
                    occurrences.push_back({1, coordinates, {unsigned(1 + i % 2), 3}, {4}});
                }
                occurrences.push_back({2, {{loops[0], t}}, {}, {unsigned(100 + t)}});
            }
            for (uint64_t t = 0; t < trips; ++t) {
                occurrences.push_back({3, {{loops[1], t}}, {unsigned(100 + t)}, {5}});
            }
            for (uint64_t t = 0; t < trips; ++t) {
                occurrences.push_back({4, {{loops[2], t}}, {}, {unsigned(100 + t)}});
            }
            const unsigned count = 2 * occurrences.size();
            std::vector<std::vector<bool>> graph(count, std::vector<bool>(count));
            std::map<uint32_t, unsigned> previous;
            for (unsigned i = 0; i < occurrences.size(); ++i) {
                auto pipe = static_cast<uint32_t>(region.anchors[occurrences[i].type].phase->kPipeValue);
                graph[2*i][2*i+1] = true;
                auto found = previous.find(pipe);
                if (found != previous.end()) {
                    graph[2*found->second][2*i] = true; graph[2*found->second+1][2*i+1] = true;
                }
                previous[pipe] = i;
                for (unsigned j = 0; j < i; ++j) {
                    auto sourcePipe = static_cast<uint32_t>(region.anchors[occurrences[j].type].phase->kPipeValue);
                    if (sourcePipe == 0 && pipe == 0) { continue; } // Shared same-scalar hardware protection.
                    if (intersects(occurrences[j].writes, occurrences[i].reads) ||
                        intersects(occurrences[j].writes, occurrences[i].writes) ||
                        intersects(occurrences[j].reads, occurrences[i].writes)) { graph[2*j+1][2*i] = true; }
                }
            }
            for (unsigned i = 0; i < count; ++i) { graph[i][i] = true; }
            for (unsigned k = 0; k < count; ++k) {
                for (unsigned i = 0; i < count; ++i) {
                    if (!graph[i][k]) { continue; }
                    for (unsigned j = 0; j < count; ++j) { graph[i][j] = graph[i][j] || graph[k][j]; }
                }
            }
            fs::RegionExpressions::Substitution parameters({
                {e.input(function.getArgument(0)), e.constant(trips)},
                {e.input(function.getArgument(1)), e.constant(length)}});
            auto event = [&](unsigned number) {
                const auto& occurrence = occurrences[number / 2];
                fs::RegionalEvent result{occurrence.type, e.constant(0),
                    number % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start};
                auto coordinate = [&](scf::ForOp loop) { return e.constant(occurrence.iterations.lookup(loop)); };
                if (region.occurrenceLoops[occurrence.type]) {
                    result.ordinal = coordinate(region.occurrenceLoops[occurrence.type]);
                }
                if (!region.outerLoops.empty()) {
                    for (auto loop : region.outerLoops[occurrence.type]) { result.visits.push_back(coordinate(loop)); }
                }
                return result;
            };
            for (unsigned i = 0; i < count; ++i) {
                for (unsigned j = 0; j < count; ++j) {
                    auto query = region.reachability(event(i), event(j));
                    auto answer = query ? e.constantValue(e.substitute(*query, parameters)) : std::nullopt;
                    if (!answer || bool(*answer) != graph[i][j]) {
                        llvm::errs() << "mixed symbolic query differs for " << trips << ',' << length
                                     << " events " << i << ',' << j << "\n"; return false;
                    }
                    ++checked;
                }
            }
        }
    }
    return true;
}
} // namespace
int runMixedSymbolicRegionChecks(func::FuncOp function)
{
    fs::FrontierAnalysis analysis(function);
    if (failed(analysis.initialize()) || !analysis.result()) { return 1; }
    const auto& program = *analysis.result();
    auto index = std::make_shared<fs::PhaseIndex>();
    if (failed(index->build(function, *analysis.input()))) { return 1; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    SmallVector<scf::ForOp> loops;
    for (auto loop : function.getBody().front().getOps<scf::ForOp>()) { loops.push_back(loop); }
    if (loops.size() != 3) { return 1; }
    auto original = llvm::find_if(program.nodes, [&](const auto& node) { return node.anchor == loops[0]; });
    if (original == program.nodes.end()) { return 1; }
    auto nested = fs::analyzeSequenceRegion(function, *analysis.input(), program,
        original - program.nodes.begin(), arena, index, false);
    if (!nested.error.empty()) { llvm::errs() << nested.error << "\n"; return 1; }
    auto first = fs::sequenceRegionalResult(nested);
    if (!first.cost.repeatedRegions || first.arithmeticRelations || first.symbolicStorageEffects.empty()) {
        llvm::errs() << "mixed oracle did not select the repeated owned producer\n"; return 1;
    }
    std::string error;
    auto second = fs::analyzeArithmeticRegion({function, loops[1]}, *index, *analysis.input(), arena, error);
    auto third = fs::analyzeArithmeticRegion({function, loops[2]}, *index, *analysis.input(), arena, error);
    if (failed(second) || failed(third)) { llvm::errs() << error << "\n"; return 1; }
    auto pair = fs::composeSymbolicRegionalSequence({first, *second}, function, *analysis.input(), *index, error);
    if (failed(pair) || !pair->relations) { llvm::errs() << "mixed pair: " << error << "\n"; return 1; }
    auto parent = fs::composeSymbolicRegionalSequence({*pair, *third}, function, *analysis.input(), *index, error);
    if (failed(parent) || !parent->relations || parent->anchors.size() != 5) {
        llvm::errs() << "mixed parent: " << error << "\n"; return 1;
    }
    auto inner = *loops[0].getBody()->getOps<scf::ForOp>().begin();
    uint64_t checked = 0;
    if (!oracle(*parent, function, loops, inner, checked)) { return 1; }
    if (!parent->prepare) { return 1; }
    auto prepared = parent->prepare();
    if (failed(prepared)) { llvm::errs() << "mixed original endpoint preparation failed\n"; return 1; }
    llvm::outs() << "mixed repeated/arithmetic symbolic regions: " << checked
                 << " independent event queries and original endpoints passed\n";
    return 0;
}
