// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite signed-byte oracle for aligned relative storage atoms.
#ifndef PTO_TEST_ALIGNEDMODULARCELLORACLE_H
#define PTO_TEST_ALIGNEDMODULARCELLORACLE_H
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/AlignedModularCells.h"
#include "PTO/Transforms/FrontierSynch/Finite/ExplicitReduction.h"
#include <set>
namespace {
namespace fs = mlir::pto::frontiersynch;
namespace numeric = mlir::pto::frontiersynch::detail;
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
Matrix nativeGraph(llvm::ArrayRef<fs::ExplicitEffects> word)
{
    Matrix result(word.size() * 2, std::vector<bool>(word.size() * 2));
    for (std::size_t i = 0; i < word.size(); ++i) {
        result[2 * i][2 * i + 1] = true;
        for (std::size_t j = i + 1; j < word.size(); ++j) {
            if (word[i].pipe == word[j].pipe) {
                result[2 * i][2 * j] = true; result[2 * i + 1][2 * j + 1] = true;
            }
        }
    }
    return result;
}
bool alignedCellOracle()
{
    uint64_t memberships = 0, eventPairs = 0;
    for (unsigned width : {8U, 16U, 64U}) {
        std::vector<numeric::AlignedModularAccess> accesses;
        const std::vector<uint64_t> offsets{0, 4, 0, 4, 8, 0, 8, 4, 0, 8, 12, 12};
        for (unsigned i = 0; i < offsets.size(); ++i) {
            accesses.push_back({llvm::APInt(width, offsets[i]), {{i % 3, 7 + i % 3}}});
        }
        // Empty intervals inside an otherwise present atom must stay empty.
        accesses.push_back({llvm::APInt(width, 0), {{3, 3}}});
        auto partition = numeric::partitionAlignedModularCells(width, 2, 3, accesses, 65536);
        if (!partition || !partition->references.back().empty()) { return false; }
        std::vector<fs::ExplicitEffects> word;
        for (unsigned i = 0; i < accesses.size(); ++i) {
            fs::ExplicitEffects occurrence{i, i % 3, {}};
            for (auto atom : partition->references[i]) {
                occurrence.accesses.push_back({atom, i % 3 != 0, i % 3 != 1});
            }
            word.push_back(std::move(occurrence));
        }
        const std::vector<fs::StorageGenerator> prerequisites{{0, 5}, {4, 9}};
        const auto scanned = fs::scanStorageLifetimes(word, prerequisites);
        const auto reduced = fs::reduceExplicitDemands(word, scanned.generators);
        if (!scanned.error.empty() || !reduced.error.empty()) { return false; }
        auto actual = nativeGraph(word);
        for (auto edge : reduced.retained) { actual[2 * edge.source + 1][2 * edge.target] = true; }
        close(actual);
        std::vector<uint64_t> origins;
        for (unsigned i = 0; i < 64; ++i) { origins.push_back(i * 4); }
        origins.insert(origins.end(), {uint64_t{1} << 63, (uint64_t{1} << 63) - 4, UINT64_MAX - 3});
        for (auto seed : origins) {
            const llvm::APInt origin(width, seed);
            std::vector<std::set<__int128>> bytes(accesses.size());
            for (unsigned i = 0; i < accesses.size(); ++i) {
                const auto raw = origin + accesses[i].offset;
                const __int128 start = static_cast<__int128>(raw.getSExtValue()) * 3;
                for (auto interval : accesses[i].intervals) {
                    for (auto byte = interval.first; byte < interval.second; ++byte) { bytes[i].insert(start + byte); }
                }
                std::set<__int128> reconstructed;
                for (auto id : partition->references[i]) {
                    const auto& atom = partition->atoms[id];
                    const auto cell = origin + llvm::APInt(width, atom.segment).shl(2);
                    const __int128 base = static_cast<__int128>(cell.getSExtValue()) * 3;
                    for (auto byte = atom.begin; byte < atom.end; ++byte) { reconstructed.insert(base + byte); }
                }
                if (bytes[i] != reconstructed) { return false; }
                memberships += bytes[i].size();
            }
            auto expected = nativeGraph(word);
            for (auto edge : prerequisites) { expected[2 * edge.source + 1][2 * edge.target] = true; }
            for (unsigned i = 0; i < accesses.size(); ++i) {
                for (unsigned j = i + 1; j < accesses.size(); ++j) {
                    const bool writes = i % 3 != 1 || j % 3 != 1;
                    const bool overlap = std::any_of(bytes[i].begin(), bytes[i].end(), [&](auto byte) {
                        return bytes[j].count(byte) != 0;
                    });
                    if (writes && overlap) { expected[2 * i + 1][2 * j] = true; }
                }
            }
            close(expected);
            if (actual != expected) { return false; }
            eventPairs += actual.size() * actual.size();
        }
        // Relative labels and graph outlive all source occurrence descriptors.
        accesses.clear(); accesses.shrink_to_fit();
        if (partition->atoms.empty() || partition->references.size() != word.size()) { return false; }
    }
    std::vector<numeric::AlignedModularAccess> one{{llvm::APInt(8, 0), {{0, 4}}}};
    if (numeric::partitionAlignedModularCells(8, 8, 1, one, 100) ||
        numeric::partitionAlignedModularCells(8, 2, 0, one, 100) ||
        numeric::partitionAlignedModularCells(64, 63, UINT64_MAX, one, 100) ||
        numeric::partitionAlignedModularCells(8, 2, 1, one, 1)) { return false; }
    one[0].offset = llvm::APInt(8, 1);
    if (numeric::partitionAlignedModularCells(8, 2, 1, one, 100)) { return false; }
    one[0].offset = llvm::APInt(8, 0); one[0].intervals = {{0, 5}};
    if (numeric::partitionAlignedModularCells(8, 2, 1, one, 100)) { return false; }
    llvm::outs() << "aligned modular cells: " << memberships << " original byte memberships, "
                 << eventPairs << " independent event pairs and refusal checks passed\n";
    return true;
}
} // namespace
#endif
