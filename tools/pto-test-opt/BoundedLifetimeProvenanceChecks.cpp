// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/BoundedLifetime.h"
#include "../../lib/PTO/Transforms/FrontierSynch/PhysicalRefreshLimits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <map>
#include <set>
#include <tuple>
namespace fs = mlir::pto::frontiersynch;
namespace {
constexpr unsigned Sites = 6;
constexpr unsigned Predicates = 8;
using Graph = std::array<std::array<bool, 2 * Sites>, 2 * Sites>;
using Witness = std::tuple<uint32_t, fs::StorageHazard, uint32_t>;
using Mode = std::pair<bool, bool>;
using Uses = std::map<unsigned, Mode>;
using Cells = std::map<unsigned, Uses>;
void close(Graph& graph)
{
    for (unsigned k = 0; k < graph.size(); ++k) {
        for (unsigned a = 0; a < graph.size(); ++a) {
            for (unsigned b = 0; b < graph.size(); ++b) {
                graph[a][b] = graph[a][b] || (graph[a][k] && graph[k][b]);
            }
        }
    }
}
bool plain(const fs::LifetimeWindowInput& input, const Uses& uses)
{
    std::set<uint64_t> seen;
    for (const auto& [site, mode] : uses) {
        if (!input.operations.empty() && input.operations[site] && !seen.insert(input.operations[site]).second) {
            return false;
        }
    }
    return true;
}
std::set<Witness> canonical(const fs::LifetimeWindowInput& input, const Cells& cells,
                           unsigned source, unsigned target, bool supplied)
{
    std::set<Witness> result;
    for (const auto& [cell, uses] : cells) {
        if (!plain(input, uses) || !uses.count(source) || !uses.count(target)) { continue; }
        auto [ar, aw] = uses.at(source);
        auto [br, bw] = uses.at(target);
        bool writer = false, reader = false, sourceReader = false, targetReader = false;
        for (const auto& [site, mode] : uses) {
            if (site <= source || site >= target) { continue; }
            writer |= mode.second;
            if (mode.first && !mode.second) {
                reader = true;
                sourceReader |= input.payloads[site].pipe == input.payloads[source].pipe;
                targetReader |= input.payloads[site].pipe == input.payloads[target].pipe;
            }
        }
        if (writer) { continue; }
        if (aw && br && !bw && !targetReader) {
            result.emplace(cell, fs::StorageHazard::RAW, input.payloads[target].pipe);
        }
        if (ar && !aw && bw && !sourceReader) {
            result.emplace(cell, fs::StorageHazard::WAR, input.payloads[source].pipe);
        }
        if (aw && bw && !reader) { result.emplace(cell, fs::StorageHazard::WAW, input.payloads[target].pipe); }
    }
    if (supplied) { result.emplace(0, fs::StorageHazard::Supplied, input.payloads[target].pipe); }
    return result;
}
fs::LifetimeWindowInput makeInput(fs::RegionExpressions& e, const std::vector<fs::RegionExpressions::Id>& p,
                                  unsigned scenario)
{
    fs::LifetimeWindowInput input;
    input.sites = scenario == 4 ? Sites / 2 : Sites;
    input.span = scenario == 4 ? 1 : 0;
    const unsigned pipes[] = {0, 1, 1, 2, 0, 2};
    for (unsigned i = 0; i < Sites; ++i) {
        input.payloads.push_back({pipes[i], p[i]});
        if (scenario == 1) { input.operations.push_back(i < 2 ? 1 : i + 1); }
        if (scenario == 2) { continue; }
        auto write = i == 2 ? p[6] : e.boolean(i == 0 || i == 3 || i == 5);
        auto read = i == 3 ? p[7] : e.boolean(true);
        if (scenario == 3) {
            input.accesses.push_back({i, 0, read, e.boolean(false)});
            input.accesses.push_back({i, 0, e.boolean(false), write});
        } else { input.accesses.push_back({i, 0, read, write}); }
        if (i != 1 && i != 3) {
            input.accesses.push_back({i, 1, e.boolean(i != 0), e.boolean(i == 0 || i == 5)});
        }
    }
    input.prerequisites.push_back({0, 5, p[7]});
    return input;
}
bool valuation(fs::RegionExpressions& e, const fs::LifetimeWindowInput& input,
               const fs::LifetimeWindowAnalysis& analysis, const std::vector<fs::RegionExpressions::Id>& p,
               unsigned mask, unsigned scenario)
{
    std::vector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> bindings;
    for (unsigned i = 0; i < Predicates; ++i) { bindings.emplace_back(p[i], e.boolean(mask & (1U << i))); }
    fs::RegionExpressions::Substitution substitution(bindings);
    auto transaction = fs::RegionExpressions::Transaction(e);
    auto active = [&](fs::RegionExpressions::Id value) {
        return e.constantValue(e.substitute(value, substitution)) == 1;
    };
    Cells cells;
    for (auto access : input.accesses) {
        auto& mode = cells[access.cell][access.payload];
        mode.first |= active(input.payloads[access.payload].present) && active(access.read);
        mode.second |= active(input.payloads[access.payload].present) && active(access.write);
    }
    Graph full{};
    for (unsigned a = 0; a < Sites; ++a) {
        if (!active(input.payloads[a].present)) { continue; }
        full[2 * a][2 * a + 1] = true;
        for (unsigned b = a + 1; b < Sites; ++b) {
            if (!active(input.payloads[b].present)) { continue; }
            if (input.payloads[a].pipe == input.payloads[b].pipe) {
                full[2 * a][2 * b] = full[2 * a + 1][2 * b + 1] = true;
            }
            if (!input.operations.empty() && input.operations[a] == input.operations[b]) { continue; }
            for (const auto& [cell, uses] : cells) {
                if (!uses.count(a) || !uses.count(b)) { continue; }
                auto [ar, aw] = uses.at(a);
                auto [br, bw] = uses.at(b);
                full[2 * a + 1][2 * b] = full[2 * a + 1][2 * b] || (aw && (br || bw)) || (ar && bw);
            }
        }
    }
    bool supplied = active(input.payloads[0].present) && active(input.payloads[5].present) && active(p[7]);
    full[1][10] = full[1][10] || supplied;
    close(full);
    std::set<std::pair<unsigned, unsigned>> expected, actual;
    for (unsigned a = 0; a < input.sites; ++a) {
        for (unsigned b = a + 1; b < Sites; ++b) {
            bool cover = full[2 * a + 1][2 * b];
            for (unsigned k = 2 * a + 2; k < 2 * b; ++k) {
                cover &= !(full[2 * a + 1][k] && full[k][2 * b]);
            }
            if (cover) { expected.emplace(a, b); }
        }
    }
    if (analysis.sourceDemands.size() != analysis.sourceWitnesses.size()) { return false; }
    for (unsigned i = 0; i < analysis.sourceDemands.size(); ++i) {
        const auto& edge = analysis.sourceDemands[i];
        bool retained = active(edge.guard);
        std::set<Witness> witnesses;
        for (auto witness : analysis.sourceWitnesses[i]) {
            if (active(witness.guard)) { witnesses.emplace(witness.cell, witness.hazard, witness.pipe); }
        }
        auto wanted = retained ? canonical(input, cells, edge.source, edge.target,
            supplied && edge.source == 0 && edge.target == 5) : std::set<Witness>{};
        if (witnesses != wanted || (scenario != 1 && retained && witnesses.empty())) { return false; }
        if (retained) { actual.emplace(edge.source, edge.target); }
    }
    return actual == expected;
}
} // namespace
int runBoundedLifetimeProvenanceChecks()
{
    using fs::detail::physicalRefreshWindowFits;
    if (!physicalRefreshWindowFits(1, 1, 0, 32767) || physicalRefreshWindowFits(1, 1, 0, 32768) ||
        !physicalRefreshWindowFits(65535, 1, 0, 0) || physicalRefreshWindowFits(65536, 1, 0, 0) ||
        physicalRefreshWindowFits(UINT64_MAX, 1, 0, 0) ||
        physicalRefreshWindowFits(1, UINT64_MAX, 1, UINT64_MAX) ||
        physicalRefreshWindowFits(0, 1, 0, 0)) {
        llvm::errs() << "physical refresh representation preflight mismatch\n"; return 1;
    }
    mlir::MLIRContext context;
    for (unsigned scenario = 0; scenario < 5; ++scenario) {
        mlir::Block block;
        fs::RegionExpressions e;
        std::vector<fs::RegionExpressions::Id> predicates;
        for (unsigned i = 0; i < Predicates; ++i) {
            predicates.push_back(e.input(block.addArgument(mlir::IntegerType::get(&context, 1),
                                                          mlir::UnknownLoc::get(&context))));
        }
        auto input = makeInput(e, predicates, scenario);
        auto analysis = fs::analyzeLifetimeWindow(e, input);
        if (!analysis.error.empty()) { llvm::errs() << analysis.error << "\n"; return 1; }
        for (unsigned mask = 0; mask < (1U << Predicates); ++mask) {
            if (!valuation(e, input, analysis, predicates, mask, scenario)) {
                llvm::errs() << "bounded provenance mismatch: scenario " << scenario << " mask " << mask << "\n";
                return 1;
            }
        }
    }
    llvm::outs() << "bounded provenance: 1280 independent guard/mode valuations passed\n";
    return 0;
}
