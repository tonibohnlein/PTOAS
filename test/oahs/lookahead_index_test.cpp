// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "../../lib/PTO/Transforms/OAHS/SelectedLookahead.h"
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>

namespace {
using Index = mlir::pto::oahs::selected::LookaheadIndex;
using Id = std::size_t;
using Graph = std::vector<std::vector<Id>>;
void require(bool ok, const char* message)
{
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
bool slowFuture(const Graph& graph, const std::vector<bool>& payload, Id source)
{
    std::vector<bool> seen(graph.size());
    auto todo = graph[source];
    while (!todo.empty()) {
        const auto at = todo.back(); todo.pop_back();
        if (seen[at]) continue;
        seen[at] = true;
        if (payload[at]) return true;
        todo.insert(todo.end(), graph[at].begin(), graph[at].end());
    }
    return false;
}
void compare(const Graph& graph, const std::vector<bool>& payload)
{
    Index index;
    require(index.build(graph, payload, {}), "valid graph rejected");
    for (Id site = 0; site < graph.size(); ++site)
        require(index.mayIssueAfter(site) == slowFuture(graph, payload, site),
                "backward summary differs from forward reachability");
}
void invalidAndBoundary()
{
    Index index;
    require(index.mayIssueAfter(0), "uninitialized summary grants terminality");
    require(index.hasIssueBetween(0, 0, 0, 1), "uninitialized summary grants freshness");
    require(!index.build({{1}}, {true}, {}), "invalid edge accepted");
    require(!index.build({{}}, {}, {}), "invalid dimensions accepted");
    require(!index.build({{}}, {false}, {{1, 0, 0}}), "invalid frame accepted");
    require(!index.build({{}}, {false}, {{0, 1, 0}}), "invalid ordinal accepted");
    require(index.build({{1}, {}}, {true, false}, {{0, 0, 7}, {0, 1, 8}}), "build");
    require(!index.mayIssueAfter(0), "current payload was counted as future");
    require(index.hasIssueBetween(0, 7, 0, 1), "source cut must be included");
    require(!index.hasIssueBetween(0, 8, 0, 1), "consumer cut must be excluded");
    require(!index.hasIssueBetween(0, 7, 0, 0), "empty interval");
    require(index.build({{0}}, {true}, {}), "self-loop build");
    require(index.mayIssueAfter(0), "backedge lost current operation's next occurrence");
    require(!index.build({{1}}, {true}, {}), "bad rebuild");
    require(index.mayIssueAfter(0), "failed rebuild leaked old summary");
    compare({{1, 2}, {3}, {3}, {}}, {false, true, false, false});
    compare({{1}, {2, 3}, {1}, {}}, {false, false, true, false});
}
Id exhaustive(unsigned maxNodes)
{
    Id cases = 0;
    for (unsigned n = 1; n <= maxNodes; ++n) {
        const auto graphs = uint64_t(1) << (n * n);
        for (uint64_t mask = 0; mask < graphs; ++mask) {
            Graph graph(n);
            for (unsigned a = 0; a < n; ++a)
                for (unsigned b = 0; b < n; ++b)
                    if (mask & (uint64_t(1) << (a * n + b))) graph[a].push_back(b);
            for (unsigned bits = 0; bits < (1u << n); ++bits) {
                std::vector<bool> payload(n);
                for (unsigned i = 0; i < n; ++i) payload[i] = bits & (1u << i);
                compare(graph, payload);
                ++cases;
            }
        }
    }
    return cases;
}
Id issueIntervals()
{
    std::mt19937 random(0x2409a6b8);
    Id queries = 0;
    for (unsigned iteration = 0; iteration < 200; ++iteration) {
        constexpr Id size = 17, frames = 4, classes = 13;
        std::vector<Index::ClassIssue> uses;
        for (unsigned j = 0; j < 80; ++j)
            uses.push_back({random() % frames, random() % size, random() % classes});
        uses.push_back(uses.front()); // duplicated RMW/view incidences are harmless
        Index index;
        require(index.build(Graph(size), std::vector<bool>(size), uses), "issue index build");
        for (Id frame = 0; frame < frames; ++frame) {
            for (Id access = 0; access <= classes; ++access) {
                for (Id begin = 0; begin <= size; ++begin) {
                    for (Id end = begin; end <= size; ++end) {
                        const bool slow = std::any_of(uses.begin(), uses.end(), [&](const auto& use) {
                            return use.frame == frame && use.access == access &&
                                   begin <= use.position && use.position < end;
                        });
                        require(index.hasIssueBetween(frame, access, begin, end) == slow,
                                "class index differs from the original interval scan");
                        ++queries;
                    }
                }
            }
        }
    }
    return queries;
}
} // namespace
int main(int argc, char** argv)
{
    const unsigned maxNodes = argc == 2 && std::string(argv[1]) == "--exhaustive" ? 4 : 3;
    invalidAndBoundary();
    const auto cases = exhaustive(maxNodes);
    const auto intervals = issueIntervals();
    std::cout << "lookahead graph/payload cases=" << cases
              << " interval queries=" << intervals
              << " passed\n";
}
