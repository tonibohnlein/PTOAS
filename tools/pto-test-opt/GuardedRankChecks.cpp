// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetime.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/GuardedStorageScan.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/ModuloLaneSpan.h"
#include "RestrictedIncomingChecks.h"
#include "GuardedQueryProjectionChecks.h"
#include "CompactFiniteControlChecks.h"
#include <array>
#include <set>
namespace fs = mlir::pto::frontiersynch;
namespace {
using Graph = std::vector<std::vector<bool>>;
void close(Graph& graph)
{
    for (unsigned z = 0; z < graph.size(); ++z) {
        for (unsigned a = 0; a < graph.size(); ++a) {
            if (!graph[a][z]) {
                continue;
            }
            for (unsigned b = 0; b < graph.size(); ++b) {
                graph[a][b] = graph[a][b] || graph[z][b];
            }
        }
    }
}
bool ranks(mlir::MLIRContext& context, unsigned pattern, fs::GuardedRankLabels labels,
    std::array<uint64_t, 4>& acceptedProjection, bool uniform = false)
{
    constexpr unsigned n = 6;
    mlir::Block block;
    fs::RegionExpressions dag;
    std::vector<fs::RegionExpressions::Id> predicates;
    std::vector<fs::GuardedRankPayload> payloads;
    for (unsigned i = 0; i < n; ++i) {
        auto input = block.addArgument(mlir::IntegerType::get(&context, 1), mlir::UnknownLoc::get(&context));
        predicates.push_back(dag.input(input));
        payloads.push_back({(i + pattern * (i / 2)) % 3, predicates.back()});
    }
    if (uniform) {
        for (auto& payload : payloads) { payload.present = predicates[payload.pipe]; }
    }
    std::vector<fs::GuardedRankEdge> edges, native;
    for (unsigned a = 0; a < n; ++a) {
        for (unsigned b = a + 1; b < n; ++b) {
            const auto index = (a + b + pattern) % n;
            auto guard = predicates[uniform ? payloads[index].pipe : index];
            if ((a + 2 * b + pattern) % 3) {
                edges.push_back({a, b, guard});
            }
            if ((a + 3 * b + pattern) % 7 == 0) {
                native.push_back({a, b, guard});
            }
        }
    }
    if (!edges.empty()) { edges.push_back(edges.front()); }
    if (!native.empty()) { native.push_back(native.front()); }
    auto result = fs::reduceGuardedRanks(dag, payloads, edges, native, labels, true, true);
    if (!result.error.empty()) {
        llvm::errs() << result.error << "\n";
        return false;
    }
    if (uniform && !result.projectionWork.requests) { return false; }
    std::vector<fs::RegionExpressions::Id> queries;
    std::vector<fs::RegionExpressions::Id> projected;
    for (unsigned a = 0; a < 2 * n; ++a) {
        for (unsigned b = 0; b < 2 * n; ++b) {
            auto answer = result.query(
                dag, {a / 2, a % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start},
                {b / 2, b % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start});
            if (!answer) {
                return false;
            }
            queries.push_back(*answer);
            const auto alternative = fs::detail::projectGuardedRankQuery(dag, result,
                {a / 2, a % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start},
                {b / 2, b % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start});
            if (!alternative.answer) { return false; }
            if (alternative.status == fs::detail::GuardedQueryProjectionStatus::Projected) {
                ++acceptedProjection[2 * (a % 2) + b % 2];
            }
            projected.push_back(*alternative.answer);
        }
    }
    for (unsigned mask = 0; mask < (1U << n); ++mask) {
        std::vector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> bindings;
        for (unsigned i = 0; i < n; ++i) {
            bindings.emplace_back(predicates[i], dag.boolean(mask & (1U << i)));
        }
        fs::RegionExpressions::Substitution substitution(bindings);
        auto transaction = fs::RegionExpressions::Transaction(dag);
        auto active = [&](fs::RegionExpressions::Id id) {
            return dag.constantValue(dag.substitute(id, substitution)) == 1;
        };
        auto present = [&](unsigned site) { return bool(mask & (1U << (uniform ? payloads[site].pipe : site))); };
        Graph base(2 * n, std::vector<bool>(2 * n));
        for (unsigned a = 0; a < n; ++a) {
            if (!present(a)) {
                continue;
            }
            base[2 * a][2 * a] = base[2 * a + 1][2 * a + 1] = base[2 * a][2 * a + 1] = true;
            for (unsigned b = a + 1; b < n; ++b) {
                if (present(b) && payloads[a].pipe == payloads[b].pipe) {
                    base[2 * a][2 * b] = base[2 * a + 1][2 * b + 1] = true;
                }
            }
        }
        auto add = [&](Graph& graph, const auto& records) {
            for (auto edge : records) {
                if (present(edge.source) && present(edge.target) && active(edge.guard)) {
                    graph[2 * edge.source + 1][2 * edge.target] = true;
                }
            }
        };
        add(base, native);
        close(base);
        auto full = base;
        add(full, edges);
        close(full);
        auto reduced = base;
        add(reduced, result.retained);
        close(reduced);
        if (full != reduced) {
            llvm::errs() << "guarded rank closure mismatch\n";
            return false;
        }
        for (unsigned a = 0; a < 2 * n; ++a) {
            for (unsigned b = 0; b < 2 * n; ++b) {
                if (active(queries[a * 2 * n + b]) != full[a][b] ||
                    active(projected[a * 2 * n + b]) != full[a][b]) {
                    llvm::errs() << "guarded query mismatch\n";
                    return false;
                }
            }
        }
        for (auto edge : result.retained) {
            bool cover = full[2 * edge.source + 1][2 * edge.target] && !base[2 * edge.source + 1][2 * edge.target];
            for (unsigned z = 2 * edge.source + 2; z < 2 * edge.target; ++z) {
                cover &= !(full[2 * edge.source + 1][z] && full[z][2 * edge.target]);
            }
            if (active(edge.guard) != cover) {
                llvm::errs() << "guarded cover mismatch\n";
                return false;
            }
        }
    }
    return true;
}
bool activityOnlyRows(mlir::MLIRContext& context)
{
    mlir::Block inputs;
    fs::RegionExpressions e;
    const auto type = mlir::IntegerType::get(&context, 1);
    const auto loc = mlir::UnknownLoc::get(&context);
    const auto linked = inputs.addArgument(type, loc), optional = inputs.addArgument(type, loc);
    const auto h = e.input(linked), z = e.input(optional), yes = e.boolean(true);
    // B -> A; optional X and L follow A on its pipe; L -> R under h;
    // C follows R on a third pipe, with an additional A -> C generator.
    // X changes L's executed rank, but cannot affect whether B precedes C.
    const std::vector<fs::GuardedRankPayload> payloads = {
        {1, yes}, {0, yes}, {0, z}, {0, yes}, {2, yes}, {2, yes}};
    const std::vector<fs::GuardedRankEdge> edges = {{0, 1, yes}, {3, 4, h}, {1, 5, yes}};
    const auto result = fs::reduceGuardedRanks(e, payloads, edges);
    if (!result.error.empty()) { return false; }
    const auto row = result.starts[5][result.columns[0]];
    if (llvm::any_of(e.referencedInputs(row), [&](const auto& input) { return input.second == optional; })) {
        llvm::errs() << "reachability row embeds an irrelevant executed-rank decision\n";
        return false;
    }
    for (unsigned mask = 0; mask < 4; ++mask) {
        fs::RegionExpressions::Transaction transaction(e);
        fs::RegionExpressions::Substitution binding({{h, e.boolean(mask & 1)}, {z, e.boolean(mask & 2)}});
        if (e.constantValue(e.substitute(row, binding)) != 1) { return false; }
        for (const auto& edge : result.retained) {
            const bool expected = edge.source == 0 || (edge.source == 3 ? bool(mask & 1) : !(mask & 1));
            if (e.constantValue(e.substitute(edge.guard, binding)) != unsigned(expected)) { return false; }
        }
    }
    return true;
}
bool referenceRankConstruction(mlir::MLIRContext& context)
{
    for (unsigned count : {16U, 64U, 256U}) {
        std::array<std::size_t, 2> sizes{};
        for (unsigned mode = 0; mode < 2; ++mode) {
            mlir::Block block;
            fs::RegionExpressions e;
            std::vector<fs::GuardedRankPayload> payloads;
            for (unsigned i = 0; i < count; ++i) {
                auto value = block.addArgument(mlir::IntegerType::get(&context, 1),
                    mlir::UnknownLoc::get(&context));
                payloads.push_back({0, e.input(value)});
            }
            const auto labels = mode ? fs::GuardedRankLabels::Reference : fs::GuardedRankLabels::Executed;
            auto result = fs::reduceGuardedRanks(e, payloads, {}, {}, labels);
            if (!result.error.empty() || !result.retained.empty()) { return false; }
            if (mode) {
                for (unsigned i = 0; i < count; ++i) {
                    if (e.constantValue(result.ranks[i]) != i + 1) { return false; }
                }
            }
            sizes[mode] = e.size();
        }
        if (sizes[1] >= sizes[0]) {
            llvm::errs() << "reference ranks did not eliminate branch-dependent counters\n";
            return false;
        }
        llvm::outs() << "finite rank construction " << count << ": " << sizes[0] << " -> " << sizes[1] << "\n";
    }
    return true;
}
bool windows(unsigned mode)
{
    // A write every iteration, with independent optional readers. Compare
    // each anchored window to the all-conflict graph of the whole execution.
    for (unsigned mask = 0; mask < 64; ++mask) {
        fs::RegionExpressions dag;
        constexpr unsigned sites = 3, trips = 3, n = sites * trips;
        Graph full(2 * n, std::vector<bool>(2 * n));
        auto pipe = [&](unsigned i) { return mode && mode != 4 ? (i % sites == 2 ? 1U : 0U) : i % sites; };
        auto write = [&](unsigned i) { return i % sites == 0 || (mode && i % sites == 1); };
        auto group = [&](unsigned i) -> uint64_t {
            if (!mode || mode == 4 || !write(i)) {
                return 0;
            }
            return 1 | (mode == 2 && i % sites == 0 ? fs::protectionResetBit : 0);
        };
        auto present = [&](unsigned i) {
            return i % sites == 0 || (mode == 4 && i % sites == 1) ||
                   (mask & (1U << (2 * (i / sites) + i % sites - 1)));
        };
        auto operation = [&](unsigned i) { return 2 * (i / sites) + (i % sites == 2 ? 2 : 1); };
        for (unsigned a = 0; a < n; ++a) {
            if (!present(a)) {
                continue;
            }
            full[2 * a][2 * a + 1] = true;
            for (unsigned b = a + 1; b < n; ++b) {
                if (!present(b)) {
                    continue;
                }
                if (pipe(a) == pipe(b)) {
                    full[2 * a][2 * b] = full[2 * a + 1][2 * b + 1] = true;
                }
                if ((write(a) || write(b)) && !(mode == 3 && pipe(a) == 0 && pipe(b) == 0) &&
                    !(mode == 4 && operation(a) == operation(b)) &&
                    !fs::hardwareProtectsConflict(pipe(a), group(a), pipe(b), group(b))) {
                    full[2 * a + 1][2 * b] = true;
                }
            }
        }
        close(full);
        for (unsigned iteration = 0; iteration < trips; ++iteration) {
            fs::LifetimeWindowInput input;
            input.sites = sites;
            input.span = 1;
            if (mode == 3) {
                input.storageProtection.scalarPipe = 0;
            }
            for (unsigned i = 0; i < 2 * sites; ++i) {
                bool on = iteration + i / sites < trips && present(sites * iteration + i);
                input.payloads.push_back({pipe(i), dag.boolean(on)});
                if (mode == 4) {
                    input.operations.push_back(operation(i));
                }
                input.accesses.push_back({i, 0, dag.boolean(!write(i)), dag.boolean(write(i)), group(i)});
            }
            auto result = fs::analyzeLifetimeWindow(dag, input);
            if (!result.error.empty()) {
                return false;
            }
            std::set<std::pair<unsigned, unsigned>> actual, expected;
            for (const auto& edge : result.sourceDemands) {
                if (dag.constantValue(edge.guard) == 1) {
                    actual.emplace(sites * iteration + edge.source, sites * iteration + edge.target);
                }
            }
            for (unsigned a = sites * iteration; a < sites * (iteration + 1); ++a) {
                for (unsigned b = a + 1; b < n; ++b) {
                    bool cover = full[2 * a + 1][2 * b];
                    for (unsigned z = 2 * a + 2; z < 2 * b; ++z) {
                        cover &= !(full[2 * a + 1][z] && full[z][2 * b]);
                    }
                    if (cover) {
                        expected.emplace(a, b);
                    }
                }
            }
            if (actual != expected) {
                llvm::errs() << "bounded window cover mismatch\n";
                return false;
            }
        }
    }
    std::vector<fs::PeriodicPayload> payloads{{0}, {1}};
    std::vector<fs::RotatingFragment> fragments{{0, 0, 0, 4, 2, 0, false, true, 0}, {1, 0, 0, 4, 2, 1, false, true, 0}};
    if (fs::certifyRotatingRefresh(payloads, fragments, {1, 0}).error.empty()) {
        return false;
    }
    fragments[1].offset = 2;
    auto certificate = fs::certifyRotatingRefresh(payloads, fragments, {1, 0});
    return certificate.error.empty() && certificate.span == 2;
}
bool moduloSpans()
{
    for (uint64_t width = 1; width <= 8; ++width) {
        for (uint64_t stride = 0; stride <= 8; ++stride) {
            for (uint64_t phase : {uint64_t(0), width - 1}) {
                const fs::detail::ModuloLaneDomain domain(stride, phase, width);
                for (uint64_t lane = 0; lane < width; ++lane) {
                    for (uint64_t lower = 0; lower < 8; ++lower) {
                        for (uint64_t upper = 0; upper < 10; ++upper) {
                            fs::RegionExpressions e;
                            auto span = domain.span(e, e.constant(lower), e.constant(upper), e.boolean(true), lane);
                            std::optional<uint64_t> first, last;
                            for (uint64_t n = lower; n <= upper; ++n) {
                                if ((stride * n + phase) % width == lane) {
                                    if (!first) { first = n; }
                                    last = n;
                                }
                            }
                            if (e.constantValue(span.present) != unsigned(bool(first)) ||
                                (first && (e.constantValue(span.first) != first ||
                                           e.constantValue(span.last) != last))) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    for (uint64_t width : {UINT64_MAX, UINT64_MAX - 1}) {
        for (uint64_t stride : {uint64_t(0), uint64_t(2), UINT64_MAX - 2}) {
            fs::detail::ModuloLaneDomain large(stride, width - 1, width);
            for (uint64_t n : {uint64_t(0), uint64_t(1), uint64_t(7), UINT64_MAX}) {
                const auto lane = (llvm::APInt(128, stride) * llvm::APInt(128, n) +
                    llvm::APInt(128, width - 1)).urem(llvm::APInt(128, width)).getZExtValue();
                fs::RegionExpressions e;
                auto span = large.span(e, e.constant(n), e.constant(n), e.boolean(true), lane);
                if (e.constantValue(span.present) != 1 || e.constantValue(span.first) != n ||
                    e.constantValue(span.last) != n) { return false; }
            }
        }
    }
    {
        mlir::MLIRContext context;
        mlir::Block inputs;
        fs::RegionExpressions e;
        const auto upper = e.input(inputs.addArgument(mlir::IndexType::get(&context), mlir::UnknownLoc::get(&context)));
        const auto activity = e.input(inputs.addArgument(mlir::IntegerType::get(&context, 1),
                                                          mlir::UnknownLoc::get(&context)));
        fs::detail::ModuloLaneDomain domain(1, 2, 5);
        const auto span = domain.span(e, e.constant(0), upper, activity, 0);
        if (e.constantValue(span.first) != 3) { return false; }
        for (uint64_t end = 0; end < 12; ++end) {
            for (bool active : {false, true}) {
                fs::RegionExpressions::Transaction transaction(e);
                fs::RegionExpressions::Substitution at({{upper, e.constant(end)}, {activity, e.boolean(active)}});
                const auto present = e.constantValue(e.substitute(span.present, at));
                if (present != unsigned(active && end >= 3)) { return false; }
                if (active && end >= 3 && e.constantValue(e.substitute(span.last, at)) != end - (end - 3) % 5) {
                    return false;
                }
            }
        }
    }
    fs::RegionExpressions e;
    fs::detail::ModuloLaneDomain domain(6, 7, 8);
    for (uint64_t lane = 0; lane < 8; ++lane) {
        const auto span = domain.span(e, e.constant(UINT64_MAX - 5), e.constant(UINT64_MAX), e.boolean(true), lane);
        std::optional<uint64_t> first, last;
        for (uint64_t offset = 0; offset < 6; ++offset) {
            const auto n = UINT64_MAX - 5 + offset;
            if ((6 * (n % 8) + 7) % 8 == lane) {
                if (!first) { first = n; }
                last = n;
            }
        }
        if (e.constantValue(span.present) != unsigned(bool(first)) ||
            (first && (e.constantValue(span.first) != first || e.constantValue(span.last) != last))) { return false; }
        const auto absent = domain.span(e, e.constant(0), e.constant(UINT64_MAX), e.boolean(false), lane);
        if (e.constantValue(absent.present) != 0) { return false; }
    }
    return true;
}
bool storageScans()
{
    // Independent interval oracle: enumerate all pairs and compare the exact
    // intervening-position set, including reverse reader-to-writer scans.
    for (unsigned mask = 0; mask < 256; ++mask) {
        std::vector<bool> writers;
        for (unsigned i = 0; i < 8; ++i) { writers.push_back(mask & (1U << i)); }
        std::set<std::pair<unsigned, unsigned>> seen;
        bool valid = true;
        fs::detail::scanWriterPairs(writers, 0U,
            [&](std::size_t a, std::size_t b, unsigned between) {
                unsigned expected = 0;
                for (unsigned i = a + 1; i < b; ++i) { expected |= 1U << i; }
                valid &= between == expected && seen.emplace(a, b).second;
            }, [](unsigned& between, std::size_t i) { between |= 1U << i; });
        for (unsigned a = 0; a < 8; ++a) {
            for (unsigned b = a + 1; b < 8; ++b) {
                valid &= bool(seen.count({a, b})) == (writers[a] || writers[b]);
            }
        }
        if (!valid) { return false; }
    }
    // Exhaust absent/read/write/RMW modes against all other occurrences.
    // Adjacent pairs are simultaneous envelopes and cannot supersede each other.
    for (unsigned pattern = 0; pattern < 4096; ++pattern) {
        fs::RegionExpressions e;
        std::vector<fs::detail::GuardedStorageMode> modes;
        for (unsigned i = 0; i < 6; ++i) {
            const auto mode = (pattern >> (2 * i)) & 3;
            modes.push_back({i % 3, e.boolean(mode & 1), e.boolean(mode & 2)});
        }
        for (bool tail : {false, true}) {
            fs::detail::GuardedBoundaryScan scan(e);
            bool valid = true;
            for (unsigned step = 0; step < 3; ++step) {
                const auto group = tail ? 2 - step : step;
                std::vector<unsigned> positions{2 * group, 2 * group + 1};
                scan.group(positions, [&](unsigned i) { return modes[i]; },
                    [&](unsigned i, auto writer, auto reader) {
                        bool wantWriter = e.constantValue(modes[i].write) == 1;
                        bool wantReader = e.constantValue(modes[i].read) == 1 && !wantWriter;
                        for (unsigned j = 0; j < 6; ++j) {
                            if (i / 2 == j / 2 || (tail ? j < i : j > i)) { continue; }
                            const bool writes = e.constantValue(modes[j].write) == 1;
                            wantWriter &= !writes;
                            wantReader &= !writes && !(modes[i].pipe == modes[j].pipe &&
                                e.constantValue(modes[j].read) == 1);
                        }
                        valid &= e.constantValue(writer) == unsigned(wantWriter) &&
                                 e.constantValue(reader) == unsigned(wantReader);
                    });
            }
            if (!valid) { return false; }
        }
    }
    for (unsigned count : {32U, 128U, 512U}) {
        for (unsigned writerCount : {0U, 1U, 2U}) {
            fs::RegionExpressions e;
            fs::LifetimeWindowInput input;
            input.sites = count;
            for (unsigned i = 0; i < count; ++i) {
                input.payloads.push_back({i % 3, e.boolean(true)});
                input.accesses.push_back({i, 0, e.boolean(true), e.boolean(i < writerCount)});
            }
            auto result = fs::analyzeLifetimeWindow(e, input);
            const auto expected = writerCount * (count - writerCount) + writerCount * (writerCount - 1) / 2;
            if (!result.error.empty() || result.accessPairs != expected) { return false; }
        }
    }
    return true;
}
} // namespace
int runGuardedRankChecks()
{
    mlir::MLIRContext context;
    if (!compactFiniteControlChecks::run(context) || !moduloSpans() || !storageScans() ||
        !referenceRankConstruction(context) || !activityOnlyRows(context) ||
        !writerPairIndexChecks() || !restrictedIncomingScaling(context) ||
        !guardedQueryProjectionChecks::run(context)) {
        llvm::errs() << "guarded storage scan mismatch\n";
        return 1;
    }
    std::array<uint64_t, 4> acceptedProjection{};
    for (unsigned pattern = 0; pattern < 32; ++pattern) {
        if (!ranks(context, pattern, fs::GuardedRankLabels::Executed, acceptedProjection) ||
            !ranks(context, pattern, fs::GuardedRankLabels::Reference, acceptedProjection) ||
            !ranks(context, pattern, fs::GuardedRankLabels::Reference, acceptedProjection, true)) {
            return 1;
        }
    }
    if (llvm::any_of(acceptedProjection, [](auto count) { return count == 0; })) {
        llvm::errs() << "Boolean query projection did not exercise all event-kind pairs\n"; return 1;
    }
    for (unsigned mode = 0; mode < 5; ++mode) {
        if (!windows(mode)) {
            return 1;
        }
    }
    llvm::outs() << "guarded ranks: 6144 valuations and 960 anchored windows "
                    "(including protected writers/scalars/macros) passed\n";
    return 0;
}
