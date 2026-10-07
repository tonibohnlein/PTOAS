// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/FrontierSynch/BoundedLifetime.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"
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
bool ranks(mlir::MLIRContext& context, unsigned pattern)
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
    std::vector<fs::GuardedRankEdge> edges, native;
    for (unsigned a = 0; a < n; ++a) {
        for (unsigned b = a + 1; b < n; ++b) {
            auto guard = predicates[(a + b + pattern) % n];
            if ((a + 2 * b + pattern) % 3) {
                edges.push_back({a, b, guard});
            }
            if ((a + 3 * b + pattern) % 7 == 0) {
                native.push_back({a, b, guard});
            }
        }
    }
    auto result = fs::reduceGuardedRanks(dag, payloads, edges, native);
    if (!result.error.empty()) {
        llvm::errs() << result.error << "\n";
        return false;
    }
    std::vector<fs::RegionExpressions::Id> queries;
    for (unsigned a = 0; a < 2 * n; ++a) {
        for (unsigned b = 0; b < 2 * n; ++b) {
            auto answer = result.query(
                dag, {a / 2, a % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start},
                {b / 2, b % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start});
            if (!answer) {
                return false;
            }
            queries.push_back(*answer);
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
        Graph base(2 * n, std::vector<bool>(2 * n));
        for (unsigned a = 0; a < n; ++a) {
            if (!(mask & (1U << a))) {
                continue;
            }
            base[2 * a][2 * a] = base[2 * a + 1][2 * a + 1] = base[2 * a][2 * a + 1] = true;
            for (unsigned b = a + 1; b < n; ++b) {
                if ((mask & (1U << b)) && payloads[a].pipe == payloads[b].pipe) {
                    base[2 * a][2 * b] = base[2 * a + 1][2 * b + 1] = true;
                }
            }
        }
        auto add = [&](Graph& graph, const auto& records) {
            for (auto edge : records) {
                if ((mask & (1U << edge.source)) && (mask & (1U << edge.target)) && active(edge.guard)) {
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
                if (active(queries[a * 2 * n + b]) != full[a][b]) {
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
} // namespace
int runGuardedRankChecks()
{
    mlir::MLIRContext context;
    for (unsigned pattern = 0; pattern < 5; ++pattern) {
        if (!ranks(context, pattern)) {
            return 1;
        }
    }
    for (unsigned mode = 0; mode < 5; ++mode) {
        if (!windows(mode)) {
            return 1;
        }
    }
    llvm::outs() << "guarded ranks: 320 valuations and 960 anchored windows "
                   "(including protected writers/scalars/macros) passed\n";
    return 0;
}
