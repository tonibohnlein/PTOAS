// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "GraphOracle.h"
#include <functional>
// Finite independent check of the SAME imported cells, payloads and selected
// words. Desired memory/rearming edges are never added to the causal graph.
void finiteRelay(
    const mlir::pto::oahs::Program& input, const mlir::pto::oahs::Commands& baseline,
    const mlir::pto::oahs::Commands& narrow, const std::map<std::string, mlir::pto::oahs::Commands>& mutations)
{
    namespace o = mlir::pto::oahs;
    const auto& g = *input.observed;
    unsigned cases = 0;
    for (const std::vector<unsigned>& lengths : std::vector<std::vector<unsigned>>{{}, {3}, {3, 3}, {1}, {2}, {1, 2}}) {
        std::vector<size_t> expected;
        for (auto n : lengths) {
            for (unsigned i = 0; i < 13; ++i)
                expected.push_back(i);
            for (unsigned j = 0; j < n; ++j)
                for (unsigned i = 13; i < 33; ++i)
                    expected.push_back(i);
            for (unsigned i = 33; i < 43; ++i)
                expected.push_back(i);
        }
        std::vector<o::Cut> path;
        std::set<std::pair<o::Cut, size_t>> active, dead;
        std::function<bool(o::Cut, size_t)> visit = [&](o::Cut at, size_t offset) {
            auto key = std::make_pair(at, offset);
            if (active.count(key) || dead.count(key))
                return false;
            auto op = g.sites[at].operation;
            if (op != o::NoControlId) {
                if (offset == expected.size() || op != expected[offset])
                    return false;
                ++offset;
            }
            path.push_back(at);
            if (at == g.exit && offset == expected.size())
                return true;
            active.insert(key);
            for (auto next : g.sites[at].successors)
                if (visit(next, offset))
                    return true;
            active.erase(key);
            dead.insert(key);
            path.pop_back();
            return false;
        };
        if (!visit(g.entry, 0)) {
            llvm::errs() << "FINITE missing trace\n";
            std::abort();
        }
        unsigned receives = 0, sends = 0;
        for (auto id : expected) {
            bool pop = id == 23 || id == 28 || id == 33 || id == 38;
            bool push = id == 7 || id == 12 || id == 17 || id == 22;
            if (!pop && !push)
                continue;
            unsigned slot = (pop ? receives++ : sends++) % 2;
            bool found = false;
            for (auto a : input.operations[id].accesses)
                if (a.cell == input.cells.size() - 2 + slot)
                    found = true;
            if (!found)
                std::abort();
        }
        auto check = [&](const o::Commands& selected, oahs_oracle::PayloadOrder& order) {
            auto p = input;
            p.observed.reset();
            p.body = {};
            p.operations.clear();
            o::Commands commands(1);
            std::vector<unsigned> visits;
            for (auto at : path) {
                commands.back().insert(commands.back().end(), selected[at].begin(), selected[at].end());
                auto op = g.sites[at].operation;
                if (op == o::NoControlId)
                    continue;
                visits.push_back(p.operations.size());
                p.operations.push_back(input.operations[op]);
                commands.emplace_back();
            }
            return oahs_oracle::graph(p, commands, visits, {}, nullptr, nullptr, &order);
        };
        oahs_oracle::PayloadOrder oldOrder, newOrder;
        auto oldVerdict = check(baseline, oldOrder), newVerdict = check(narrow, newOrder);
        auto emit = [&](auto verdict) {
            llvm::errs() << verdict.hazards << verdict.rearm << verdict.balanced << verdict.acyclic;
        };
        llvm::errs() << "FINITE outer=" << lengths.size() << " payloads=" << expected.size() << " baseline=";
        emit(oldVerdict);
        llvm::errs() << " narrow=";
        emit(newVerdict);
        std::vector<std::pair<unsigned, unsigned>> added, removed;
        std::set_difference(
            newOrder.begin(), newOrder.end(), oldOrder.begin(), oldOrder.end(), std::back_inserter(added));
        std::set_difference(
            oldOrder.begin(), oldOrder.end(), newOrder.begin(), newOrder.end(), std::back_inserter(removed));
        llvm::errs() << " added=" << added.size() << " removed=" << removed.size() << "\n";
        if (!oldVerdict || !newVerdict || !added.empty())
            std::abort();
        // The second QK of each body pair must stop gating the first receive
        // and its two operand preparations. Same-slot FIX completion remains.
        unsigned qkSecond = 0, fixFirst = 0, fixSecond = 0;
        for (unsigned i = 0; i < expected.size(); ++i) {
            if (expected[i] == 17)
                fixFirst = i;
            if (expected[i] == 21)
                qkSecond = i;
            if (expected[i] == 22)
                fixSecond = i;
            if (expected[i] >= 23 && expected[i] <= 25) {
                const auto edge = std::make_pair(2 * qkSecond + 1, 2 * i);
                if (!oldOrder.count(edge) || newOrder.count(edge))
                    std::abort();
            }
            if ((expected[i] == 23 && !newOrder.count({2 * fixFirst + 1, 2 * i})) ||
                (expected[i] == 28 && !newOrder.count({2 * fixSecond + 1, 2 * i})))
                std::abort();
        }
        auto publications = [&](const o::Commands& selected) {
            unsigned count = 0;
            for (auto cut : path)
                for (auto command : selected[cut])
                    count += command.kind == o::Command::Publish;
            return count;
        };
        const auto oldPairs = publications(baseline), newPairs = publications(narrow);
        llvm::errs() << "FINITE pairs=" << oldPairs << "->" << newPairs << "\n";
        if (lengths == std::vector<unsigned>{3}) {
            for (const auto& [name, words] : mutations) {
                oahs_oracle::PayloadOrder order;
                auto verdict = check(words, order);
                std::vector<std::pair<unsigned, unsigned>> added;
                std::set_difference(
                    order.begin(), order.end(), oldOrder.begin(), oldOrder.end(), std::back_inserter(added));
                llvm::errs() << "FINITE " << name << "=";
                emit(verdict);
                llvm::errs() << " added_vs_baseline=" << added.size() << "\n";
                if (name == "late_forwarding") {
                    if (!verdict || added.empty())
                        std::abort();
                } else {
                    if (verdict.hazards || !verdict.balanced || !verdict.acyclic)
                        std::abort();
                    if (name == "missing_later_receipt" && verdict.rearm)
                        std::abort();
                }
            }
        }
        ++cases;
    }
    llvm::errs() << "FINITE cases=" << cases << "\n";
}
