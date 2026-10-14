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
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/SequenceStorageIndex.h"
namespace mlir::pto::test {
inline bool bridgeCheckFailure(unsigned line)
{
    llvm::errs() << "sequence bridge oracle mismatch at line " << line << "\n";
    return false;
}
struct BridgeAccess { unsigned group, pipe; bool write; };
inline void bridgeFixture(frontiersynch::SequenceAnalysisState& state,
    ArrayRef<BridgeAccess> accesses, ArrayRef<frontiersynch::RegionExpressions::Id> guards,
    ArrayRef<CompoundInstanceElement> phases)
{
    state.children.resize(guards.size()); state.cells.resize(1);
    state.cells.front().space = AddressSpace::MAT;
    state.boundaries.resize(1);
    for (const auto& access : accesses) {
        auto& child = state.children[access.group];
        frontiersynch::TemplateEndpointAnchor anchor;
        anchor.phase = &phases[access.pipe];
        auto type = child.anchors.size(); child.anchors.push_back(anchor);
        auto port = state.port(access.group, type, state.c(0));
        auto selected = frontiersynch::Selected{port, guards[access.group]};
        auto& boundary = state.boundaries[0][access.group];
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
        if (!present) { return bridgeCheckFailure(__LINE__); }
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
inline bool checkSparseBoundaryImport(func::FuncOp function, ArrayRef<CompoundInstanceElement> phases)
{
    namespace fs = frontiersynch;
    OpBuilder builder(function.getContext());
    auto pointer = PtrType::get(function.getContext(), builder.getI8Type(),
        AddressSpaceAttr::get(function.getContext(), AddressSpace::GM));
    OwningOpRef<func::FuncOp> owner(func::FuncOp::create(function.getLoc(),
        "sparse_boundary_bases", builder.getFunctionType({pointer, pointer}, {})));
    auto* bases = owner->addEntryBlock();
    auto baseA = bases->getArgument(0);
    auto baseB = bases->getArgument(1);
    auto arena = std::make_shared<fs::RegionExpressions>();
    fs::SequenceAnalysisState state(function, arena);
    state.children.resize(4);
    struct Range { unsigned child, type; Value base; uint64_t begin, end; int presence; };
    const std::vector<Range> ranges = {
        {0, 0, baseA, 0, 4, 1}, {0, 1, baseA, 4, 6, 1},
        {1, 0, baseA, 2, 5, 1}, {1, 1, baseB, 0, 3, 1},
        {2, 0, baseA, 8, 10, 0}, {2, 1, baseB, 5, 7, -1},
        {3, 0, baseA, 6, 6, 1}, {3, 1, baseA, 1, 3, 1}};
    for (auto& child : state.children) {
        auto& region = child.regional;
        region.expressions = arena;
        region.gmAliasPolicy = GMAliasPolicy::MayNotAlias;
        region.capabilities.completeStorageModel = true;
        region.capabilities.exactQueries = true;
        region.capabilities.exactSelectors = true;
        region.occurrenceLoops.resize(2);
        for (const auto& phase : phases) {
            fs::TemplateEndpointAnchor anchor; anchor.phase = &phase;
            region.anchors.push_back(anchor);
        }
        region.presence = [arena](fs::RegionalEvent) { return std::optional(arena->boolean(true)); };
        region.reachability = [arena](fs::RegionalEvent, fs::RegionalEvent) {
            return std::optional(arena->boolean(false));
        };
    }
    for (const auto& range : ranges) {
        fs::RegionalStorageBoundary boundary;
        boundary.cell = {AddressSpace::GM, range.begin, range.end, range.base};
        if (range.presence >= 0) {
            fs::RegionalSelector selected{{range.type, arena->constant(0), fs::PeriodicEventKind::Start},
                                           arena->boolean(range.presence != 0)};
            boundary.firstWriters.push_back(selected); boundary.lastWriters.push_back(selected);
        }
        state.children[range.child].regional.storageBoundary.push_back(std::move(boundary));
    }
    if (!state.importSummaries(false)) { llvm::errs() << state.error << "\n"; return bridgeCheckFailure(__LINE__); }
    std::vector<SyncStorageCell> expected;
    for (auto base : {baseA, baseB}) {
        std::set<uint64_t> endpoints;
        for (const auto& range : ranges) {
            if (range.base == base && range.begin < range.end) {
                endpoints.insert(range.begin); endpoints.insert(range.end);
            }
        }
        for (auto it = endpoints.begin(); it != endpoints.end() && std::next(it) != endpoints.end(); ++it) {
            const auto begin = *it, end = *std::next(it);
            if (llvm::any_of(ranges, [&](const auto& range) {
                    return range.base == base && range.begin <= begin && range.end >= end;
                })) { expected.push_back({AddressSpace::GM, begin, end, base}); }
        }
    }
    if (state.cells.size() != expected.size() || state.boundaries.size() != expected.size()) {
        return bridgeCheckFailure(__LINE__);
    }
    std::vector<std::pair<unsigned, unsigned>> expectedPorts;
    for (unsigned child = 0; child < state.children.size(); ++child) {
        for (std::size_t cell = 0; cell < expected.size(); ++cell) {
            const auto& wanted = expected[cell]; const auto& actual = state.cells[cell];
            if (!sameStorageDomain(wanted, actual) || wanted.begin != actual.begin || wanted.end != actual.end) {
                return bridgeCheckFailure(__LINE__);
            }
            std::set<unsigned> types;
            for (const auto& range : ranges) {
                if (range.child != child || range.presence != 1 || range.base != wanted.base ||
                    range.begin > wanted.begin || range.end < wanted.end) { continue; }
                types.insert(range.type);
                auto key = std::make_pair(child, range.type);
                if (!llvm::is_contained(expectedPorts, key)) { expectedPorts.push_back(key); }
            }
            auto found = state.boundaries[cell].find(child);
            if (types.empty()) {
                if (found != state.boundaries[cell].end()) {
                    return bridgeCheckFailure(__LINE__);
                }
                continue;
            }
            if (found == state.boundaries[cell].end()) { return bridgeCheckFailure(__LINE__); }
            for (const auto* side : {&found->second.firstWriters, &found->second.lastWriters}) {
                std::set<unsigned> actualTypes;
                for (const auto& selected : *side) {
                    if (selected.port >= state.ports.size() || arena->constantValue(selected.present) != 1) {
                        return bridgeCheckFailure(__LINE__);
                    }
                    actualTypes.insert(state.ports[selected.port].type);
                }
                if (actualTypes != types || side->size() != types.size()) { return bridgeCheckFailure(__LINE__); }
            }
        }
    }
    if (state.ports.size() != expectedPorts.size()) { return bridgeCheckFailure(__LINE__); }
    for (std::size_t id = 0; id < expectedPorts.size(); ++id) {
        if (std::make_pair(state.ports[id].child, state.ports[id].type) != expectedPorts[id]) {
            return bridgeCheckFailure(__LINE__);
        }
    }
    return true;
}
inline bool checkSequenceNativeFrontier(func::FuncOp function)
{
    namespace fs = frontiersynch;
    Block inputs;
    auto makeGuard = [&](fs::RegionExpressions& e) {
        return e.input(inputs.addArgument(IntegerType::get(function.getContext(), 1), function.getLoc()));
    };
    auto populate = [](fs::SequenceAnalysisState& state, ArrayRef<BridgeAccess> sites, ArrayRef<fs::Expr> guards) {
        state.children.resize(guards.size());
        state.nativeFirst.resize(guards.size()); state.nativeLast.resize(guards.size());
        std::vector<unsigned> types(guards.size());
        for (const auto& site : sites) {
            auto port = state.port(site.group, types[site.group]++, state.c(0));
            fs::Selected value{port, guards[site.group]};
            state.nativeFirst[site.group][site.pipe].push_back(value);
            state.nativeLast[site.group][site.pipe].push_back(value);
        }
        state.buildNativeLinks();
    };
    auto arena = std::make_shared<fs::RegionExpressions>();
    std::vector<fs::Expr> guards;
    for (unsigned i = 0; i < 5; ++i) { guards.push_back(makeGuard(*arena)); }
    // Multiple pipes in a group exercise simultaneous envelopes. The group
    // with only pipe 1 leaves pipe 0's pending boundary untouched.
    const std::vector<BridgeAccess> sites = {
        {0, 0, false}, {0, 1, false}, {1, 1, false}, {2, 0, false},
        {2, 1, false}, {3, 0, false}, {4, 0, false}, {4, 1, false}};
    fs::SequenceAnalysisState state(function, arena);
    populate(state, sites, guards);
    for (unsigned bits = 0; bits < 32; ++bits) {
        SmallVector<std::pair<fs::Expr, fs::Expr>> bindings;
        for (unsigned i = 0; i < guards.size(); ++i) {
            bindings.emplace_back(guards[i], arena->boolean(bits & (1U << i)));
        }
        fs::RegionExpressions::Substitution substitution(bindings);
        const auto size = 2 * sites.size();
        std::vector<std::vector<bool>> expected(size, std::vector<bool>(size)), actual = expected;
        auto active = [&](unsigned i) { return bool(bits & (1U << sites[i].group)); };
        for (unsigned i = 0; i < sites.size(); ++i) {
            if (!active(i)) { continue; }
            expected[2 * i][2 * i + 1] = actual[2 * i][2 * i + 1] = true;
            for (unsigned j = 0; j < sites.size(); ++j) {
                if (active(j) && sites[i].group < sites[j].group && sites[i].pipe == sites[j].pipe) {
                    expected[2 * i][2 * j] = expected[2 * i + 1][2 * j + 1] = true;
                }
            }
        }
        for (const auto& [child, links] : state.incoming) {
            for (const auto& link : links) {
                auto present = arena->constantValue(arena->substitute(link.guard, substitution));
                if (!present) { return bridgeCheckFailure(__LINE__); }
                if (*present) { actual[link.source][link.target] = true; }
            }
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
        close(expected); close(actual);
        if (expected != actual) { return bridgeCheckFailure(__LINE__); }
    }
    for (unsigned count : {128U, 256U, 1024U}) {
        auto e = std::make_shared<fs::RegionExpressions>();
        const auto shared = makeGuard(*e);
        std::vector<BridgeAccess> chain;
        for (unsigned i = 0; i < count; ++i) { chain.push_back({i, 0, false}); }
        for (auto guard : {e->boolean(true), e->boolean(false), shared}) {
            fs::SequenceAnalysisState repeated(function, e);
            std::vector<fs::Expr> present(count, guard);
            populate(repeated, chain, present);
            const auto pairs = guard == e->boolean(false) ? 0 : count - 1;
            const auto masks = guard == shared ? count - 1 : 0;
            if (repeated.nativeFrontierWork.candidates != pairs || repeated.nativeFrontierWork.remasked != masks ||
                repeated.nativeFrontierWork.retained != 2 * pairs) { return bridgeCheckFailure(__LINE__); }
        }
    }
    llvm::outs() << "native-frontier checks: 32 guarded closures; linear 128/256/1024-child work\n";
    return true;
}
inline bool checkSequenceBridgeWork(func::FuncOp function, const SyncInput& input)
{
    if (input.instructions().empty()) { return bridgeCheckFailure(__LINE__); }
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
        if (!checkBridgeClosure(state, accesses, guards, bits)) { return bridgeCheckFailure(__LINE__); }
    }
    for (unsigned count : {128U, 256U, 1024U}) {
        auto e = std::make_shared<frontiersynch::RegionExpressions>();
        frontiersynch::SequenceAnalysisState chain(function, e);
        std::vector<BridgeAccess> writers;
        for (unsigned i = 0; i < count; ++i) { writers.push_back({i, 0, true}); }
        std::vector<frontiersynch::Expr> present(count, e->boolean(true));
        bridgeFixture(chain, writers, present, phases);
        if (chain.bridgeWork.candidates != count - 1 || chain.bridgeWork.remasked != 0 ||
            chain.bridgeWork.retained != count - 1 || chain.crossings.size() != count - 1) {
                return bridgeCheckFailure(__LINE__);
            }
        std::fill(present.begin(), present.end(), e->boolean(false));
        frontiersynch::SequenceAnalysisState empty(function, e);
        bridgeFixture(empty, writers, present, phases);
        if (empty.bridgeWork.candidates || empty.bridgeWork.remasked || empty.bridgeWork.retained) {
            return bridgeCheckFailure(__LINE__);
        }
        // Repeated symbolic presence exercises removal after p AND NOT p,
        // rather than only the unconditional clear path above.
        auto shared = e->input(inputs.addArgument(IntegerType::get(function.getContext(), 1), function.getLoc()));
        std::fill(present.begin(), present.end(), shared);
        frontiersynch::SequenceAnalysisState guarded(function, e);
        bridgeFixture(guarded, writers, present, phases);
        if (guarded.bridgeWork.candidates != count - 1 || guarded.bridgeWork.remasked != count - 1 ||
            guarded.bridgeWork.retained != count - 1) { return bridgeCheckFailure(__LINE__); }
        // Each child uses a different cell. Retain one participating entry per
        // cell, including through both bridge and boundary-summary consumers.
        frontiersynch::SequenceAnalysisState sparse(function, e);
        sparse.children.resize(count); sparse.cells.resize(count); sparse.boundaries.resize(count);
        sparse.nativeFirst.resize(count); sparse.nativeLast.resize(count);
        for (unsigned i = 0; i < count; ++i) {
            sparse.cells[i] = {AddressSpace::MAT, uint64_t(2) * i, uint64_t(2) * i + 1, {}};
            frontiersynch::TemplateEndpointAnchor anchor;
            anchor.phase = &phases.front();
            sparse.children[i].anchors.push_back(anchor);
            const auto port = sparse.port(i, 0, sparse.c(0));
            auto& boundary = sparse.boundaries[i][i];
            boundary.firstWriters.push_back({port, e->boolean(true)});
            boundary.lastWriters = boundary.firstWriters;
        }
        sparse.bridges();
        frontiersynch::SequenceAnalysis summary;
        sparse.summarizeBoundaries(summary);
        if (!sparse.crossings.empty() || summary.storageBoundary.size() != count ||
            sparse.boundarySummaryVisits != 4 * count ||
            llvm::any_of(sparse.boundaries, [](const auto& cell) { return cell.size() != 1; })) {
                return bridgeCheckFailure(__LINE__);
            }
        // Compare indexed selection with an independent whole-partition scan.
        const frontiersynch::SequenceStorageIndex partition(sparse.cells);
        for (unsigned begin = 0; begin < 16; ++begin) {
            for (unsigned end = begin; end < 20; ++end) {
                for (auto space : {AddressSpace::MAT, AddressSpace::GM}) {
                    SyncStorageCell range{space, begin, end, {}};
                    std::vector<std::size_t> actual, expected;
                    partition.covered(range, [&](std::size_t cell) { actual.push_back(cell); });
                    for (std::size_t cell = 0; cell < sparse.cells.size(); ++cell) {
                        const auto& value = sparse.cells[cell];
                        if (sameStorageDomain(value, range) && value.begin >= begin && value.end <= end) {
                            expected.push_back(cell);
                        }
                    }
                    if (actual != expected) { return bridgeCheckFailure(__LINE__); }
                }
            }
        }
    }
    return checkSparseBoundaryImport(function, phases) && checkSequenceNativeFrontier(function);
}
} // namespace mlir::pto::test
#endif
