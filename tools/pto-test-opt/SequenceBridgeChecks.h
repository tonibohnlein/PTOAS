// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_SEQUENCE_BRIDGE_CHECKS_H
#define PTO_FRONTIER_SEQUENCE_BRIDGE_CHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceAnalysisInternal.h"
namespace mlir::pto::test {
struct BridgeAccess { unsigned group, pipe; bool write; };
inline void bridgeFixture(frontiersynch::SequenceAnalysisState& state,
    ArrayRef<BridgeAccess> accesses, ArrayRef<frontiersynch::RegionExpressions::Id> guards,
    ArrayRef<CompoundInstanceElement> phases)
{
    state.children.resize(guards.size()); state.cells.resize(1);
    state.cells.front().space = AddressSpace::MAT;
    state.boundaries.resize(guards.size(), std::vector<frontiersynch::CellBoundary>(1));
    for (const auto& access : accesses) {
        auto& child = state.children[access.group];
        frontiersynch::TemplateEndpointAnchor anchor;
        anchor.phase = &phases[access.pipe];
        auto type = child.anchors.size(); child.anchors.push_back(anchor);
        auto port = state.port(access.group, type, state.c(0));
        auto selected = frontiersynch::Selected{port, guards[access.group]};
        auto& boundary = state.boundaries[access.group][0];
        if (access.write) {
            boundary.firstWriters.push_back(selected); boundary.lastWriters.push_back(selected);
        } else {
            boundary.firstReaders[access.pipe].push_back(selected);
            boundary.lastReaders[access.pipe].push_back(selected);
        }
    }
    state.bridges();
}
inline bool checkBridgeClosure(frontiersynch::SequenceAnalysisState& state,
    ArrayRef<BridgeAccess> accesses, ArrayRef<frontiersynch::RegionExpressions::Id> guards, unsigned bits)
{
    auto& e = state.expressions;
    SmallVector<std::pair<frontiersynch::Expr, frontiersynch::Expr>> substitutions;
    for (unsigned i = 0; i < guards.size(); ++i) {
        substitutions.emplace_back(guards[i], e.boolean(bits & (1U << i)));
    }
    frontiersynch::RegionExpressions::Substitution binding(substitutions);
    const auto count = accesses.size();
    std::vector<std::vector<bool>> actual(2 * count, std::vector<bool>(2 * count));
    auto active = [&](unsigned i) { return bool(bits & (1U << accesses[i].group)); };
    for (unsigned a = 0; a < count; ++a) {
        if (!active(a)) { continue; }
        actual[2 * a][2 * a + 1] = true;
        for (unsigned b = a + 1; b < count; ++b) {
            if (active(b) && accesses[a].pipe == accesses[b].pipe) {
                actual[2 * a][2 * b] = true; actual[2 * a + 1][2 * b + 1] = true;
            }
        }
    }
    auto required = actual;
    // Dense independent conflict graph, with no frontier or extremum selection.
    for (unsigned a = 0; a < count; ++a) {
        for (unsigned b = a + 1; b < count; ++b) {
            if (active(a) && active(b) && accesses[a].group < accesses[b].group &&
                (accesses[a].write || accesses[b].write)) { required[2 * a + 1][2 * b] = true; }
        }
    }
    for (const auto& edge : state.crossings) {
        auto present = e.constantValue(e.substitute(edge.guard, binding));
        if (!present) { return false; }
        if (*present) { actual[2 * edge.source + 1][2 * edge.target] = true; }
    }
    auto close = [](auto& graph) {
        for (unsigned k = 0; k < graph.size(); ++k) {
            for (unsigned i = 0; i < graph.size(); ++i) {
                for (unsigned j = 0; j < graph.size(); ++j) {
                    graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
                }
            }
        }
    };
    close(actual); close(required);
    return actual == required;
}
inline bool checkSequenceBridgeWork(func::FuncOp function, const SyncInput& input)
{
    if (input.instructions().empty()) { return false; }
    std::array<CompoundInstanceElement, 2> phases = {*input.instructions().front(), *input.instructions().front()};
    phases[0].kPipeValue = static_cast<PipelineType>(PIPE::PIPE_MTE1);
    phases[1].kPipeValue = static_cast<PipelineType>(PIPE::PIPE_M);
    const std::vector<BridgeAccess> accesses = {
        {0, 0, false}, {1, 0, true}, {1, 1, true}, {2, 0, false}, {2, 1, false},
        {3, 1, true}, {4, 0, false}};
    Block inputs;
    auto arena = std::make_shared<frontiersynch::RegionExpressions>();
    std::vector<frontiersynch::Expr> guards;
    for (unsigned i = 0; i < 5; ++i) {
        guards.push_back(arena->input(inputs.addArgument(IntegerType::get(function.getContext(), 1),
                                                        function.getLoc())));
    }
    frontiersynch::SequenceAnalysisState state(function, arena);
    bridgeFixture(state, accesses, guards, phases);
    for (unsigned bits = 0; bits < 32; ++bits) {
        if (!checkBridgeClosure(state, accesses, guards, bits)) { return false; }
    }
    for (unsigned count : {128U, 256U, 1024U}) {
        auto e = std::make_shared<frontiersynch::RegionExpressions>();
        frontiersynch::SequenceAnalysisState chain(function, e);
        std::vector<BridgeAccess> writers;
        for (unsigned i = 0; i < count; ++i) { writers.push_back({i, 0, true}); }
        std::vector<frontiersynch::Expr> present(count, e->boolean(true));
        bridgeFixture(chain, writers, present, phases);
        if (chain.bridgeWork.candidates != count - 1 || chain.bridgeWork.remasked != 0 ||
            chain.bridgeWork.retained != count - 1 || chain.crossings.size() != count - 1) { return false; }
        std::fill(present.begin(), present.end(), e->boolean(false));
        frontiersynch::SequenceAnalysisState empty(function, e);
        bridgeFixture(empty, writers, present, phases);
        if (empty.bridgeWork.candidates || empty.bridgeWork.remasked || empty.bridgeWork.retained) { return false; }
        // Repeated symbolic presence exercises removal after p AND NOT p,
        // rather than only the unconditional clear path above.
        auto shared = e->input(inputs.addArgument(IntegerType::get(function.getContext(), 1), function.getLoc()));
        std::fill(present.begin(), present.end(), shared);
        frontiersynch::SequenceAnalysisState guarded(function, e);
        bridgeFixture(guarded, writers, present, phases);
        if (guarded.bridgeWork.candidates != count - 1 || guarded.bridgeWork.remasked != count - 1 ||
            guarded.bridgeWork.retained != count - 1) { return false; }
    }
    return true;
}
} // namespace mlir::pto::test
#endif
