// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Small physical conflicts/selector oracle, independent of lower-fact inference.
#include "PTO/Transforms/FrontierSynch/CompactLowerFacts.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <utility>
namespace fs = mlir::pto::frontiersynch;
namespace {
constexpr auto yes = fs::CompactFactProof::Certified;
using Graph = std::vector<std::vector<bool>>;
Graph native(uint32_t trips)
{
    Graph result(4 * trips, std::vector<bool>(4 * trips));
    for (uint32_t a = 0; a < 2 * trips; ++a) {
        result[2*a][2*a] = result[2*a+1][2*a+1] = result[2*a][2*a+1] = true;
        for (uint32_t b = a + 1; b < 2 * trips; ++b) {
            if (a % 2 == b % 2) { result[2*a][2*b] = result[2*a+1][2*b+1] = true; }
        }
    }
    return result;
}
Graph close(Graph graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) {
                graph[a][b] = graph[a][b] || (graph[a][k] && graph[k][b]);
            }
        }
    }
    return graph;
}
Graph lowerGraph(llvm::ArrayRef<fs::PeriodicRecord> records, uint32_t trips)
{
    auto graph = native(trips);
    for (const auto& record : records) {
        for (uint32_t i = 0; i < trips; ++i) {
            if (record.displacement >= trips - i) { continue; }
            const auto source = 2 * i + record.source;
            const auto target = 2 * (i + record.displacement) + record.target;
            graph[2*source+1][2*target] = true;
        }
    }
    return close(std::move(graph));
}
bool subset(const Graph& small, const Graph& large)
{
    for (std::size_t a = 0; a < small.size(); ++a) {
        for (std::size_t b = 0; b < small.size(); ++b) {
            if (small[a][b] && !large[a][b]) { return false; }
        }
    }
    return true;
}
fs::CompactLowerFactsInput familyInput(uint64_t minimum = 0, uint64_t maximum = 2)
{
    fs::CompactLowerFactsInput input;
    input.payloads = {{13}, {97}};
    input.families.push_back({991, 0, yes, yes, yes});
    input.writers.push_back({0, 0, fs::CompactWriteIndex::Identity, yes});
    input.reads.push_back({0, 1, yes, {{yes, yes, minimum, maximum}}});
    return input;
}
bool familyOracle()
{
    for (uint32_t maximum = 0; maximum <= 3; ++maximum) {
        for (uint32_t minimum = 0; minimum <= maximum; ++minimum) {
            auto input = familyInput(minimum, maximum);
            const auto lower = fs::buildCompactLowerFacts(input);
            if (!lower.error.empty() || lower.records.size() != 1 || lower.records[0].displacement != maximum ||
                lower.provenance[0].group != 991 || lower.cost.writerIncidences != 1 ||
                lower.cost.alternatives != 1 || lower.cost.candidates != 1) { return false; }
            for (uint32_t trips = 0; trips <= 5; ++trips) {
                const auto selected = lowerGraph(lower.records, trips);
                // Independently vary each read among both extremes, including
                // nonaffine choices; startup chooses a valid current cell.
                for (uint32_t choices = 0; choices < (uint32_t(1) << trips); ++choices) {
                    auto actual = native(trips);
                    for (uint32_t j = 0; j < trips; ++j) {
                        const auto distance = j < maximum ? 0 : ((choices >> j) & 1 ? minimum : maximum);
                        actual[4*(j-distance)+1][4*j+2] = true;
                    }
                    if (!subset(selected, close(std::move(actual)))) { return false; }
                }
            }
        }
    }
    auto input = familyInput(1, 2);
    input.reads[0].alternatives.push_back({yes, yes, 2, 7});
    input.reads.push_back(input.reads[0]);
    input.families[0].group = UINT64_MAX;
    auto lower = fs::buildCompactLowerFacts(input);
    return lower.error.empty() && lower.records.size() == 2 && lower.records[0].displacement == 7 &&
        lower.records[1].displacement == 7 && lower.provenance[0].descriptor == 0 &&
        lower.provenance[1].descriptor == 1 && lower.provenance[1].group == UINT64_MAX;
}
bool rejection(fs::CompactLowerFactsInput input, fs::CompactLowerReason reason)
{
    const auto lower = fs::buildCompactLowerFacts(input);
    return lower.error.empty() && lower.records.empty() && lower.decisions.size() == 1 &&
        lower.decisions[0].reason == reason && lower.decisions[0].recordCount == 0;
}
bool familyFailures()
{
    auto input = familyInput();
    input.families[0].completeWriters = fs::CompactFactProof::Unknown;
    if (!rejection(input, fs::CompactLowerReason::IncompleteWriters)) { return false; }
    input = familyInput(); input.writers.push_back({0, 1, fs::CompactWriteIndex::Identity, yes});
    if (!rejection(input, fs::CompactLowerReason::CompetingWriter)) { return false; }
    input = familyInput(); input.writers[0].index = fs::CompactWriteIndex::Other;
    if (!rejection(input, fs::CompactLowerReason::UnprovedIdentity)) { return false; }
    input = familyInput(); input.writers[0].mandatoryCellWrite = fs::CompactFactProof::Unknown;
    if (!rejection(input, fs::CompactLowerReason::MissingMandatoryWriter)) { return false; }
    input = familyInput(); input.reads[0].alternatives.push_back({});
    if (!rejection(input, fs::CompactLowerReason::MissingMandatoryRead)) { return false; }
    input = familyInput(); input.reads[0].alternatives[0].maximum.reset();
    if (!rejection(input, fs::CompactLowerReason::UnprovedSourceRange)) { return false; }
    input = familyInput(); input.reads[0].alternatives[0].validSourceIndex = fs::CompactFactProof::Unknown;
    if (!rejection(input, fs::CompactLowerReason::UnprovedSourceRange)) { return false; }
    input = familyInput(); input.reads[0].site = 0;
    if (!rejection(input, fs::CompactLowerReason::NotForward)) { return false; }
    input = familyInput(); input.families[0].requiresStorageOrder = fs::CompactFactProof::Unknown;
    if (!rejection(input, fs::CompactLowerReason::UnprovedStorageOrder)) { return false; }
    input = familyInput(1, UINT64_MAX);
    auto huge = fs::buildCompactLowerFacts(input);
    return huge.error.empty() && huge.records.size() == 1 && huge.records[0].displacement == UINT64_MAX;
}
fs::CompactRotatingFullWrite rotating(uint64_t slots, uint64_t stride)
{
    return {481, 0, 1, slots, stride, 3, yes, yes, yes, yes, yes};
}
bool rotatingOracle()
{
    for (uint32_t slots = 1; slots <= 4; ++slots) {
        for (uint32_t stride = 0; stride <= 5; ++stride) {
            fs::CompactLowerFactsInput input;
            input.payloads = {{13}, {97}};
            input.rotating.push_back(rotating(slots, stride));
            auto lower = fs::buildCompactLowerFacts(input);
            if (!lower.error.empty() || lower.records.size() != 2 || lower.provenance.size() != 2 ||
                lower.decisions.size() != 1 || lower.decisions[0].recordCount != 2) { return false; }
            for (uint32_t trips = 0; trips <= 6; ++trips) {
                const auto selected = lowerGraph(lower.records, trips);
                for (uint32_t choices = 0; choices < (uint32_t(1) << trips); ++choices) {
                    auto actual = native(trips);
                    for (uint32_t a = 0; a < 2 * trips; ++a) {
                        for (uint32_t b = a + 1; b < 2 * trips; ++b) {
                            const auto sourceSlot = (stride * (a / 2) + 3) % slots;
                            const auto targetSlot = (stride * (b / 2) + 3) % slots;
                            const auto sourceBytes = a % 2 ? (uint32_t(1) << ((choices >> (a / 2)) & 1)) : 3;
                            const auto targetBytes = b % 2 ? (uint32_t(1) << ((choices >> (b / 2)) & 1)) : 3;
                            if (sourceSlot == targetSlot && (sourceBytes & targetBytes) && (!(a % 2) || !(b % 2))) {
                                actual[2*a+1][2*b] = true;
                            }
                        }
                    }
                    if (selected != close(std::move(actual))) { return false; }
                }
            }
        }
    }
    return true;
}
bool fullClassAndMalformed()
{
    auto input = familyInput();
    input.reads[0].mandatory = fs::CompactFactProof::Unknown;
    input.fullClasses.push_back({37, {0, 1, 0}, fs::CompactFullWriter::Source, yes, yes, yes, yes, yes});
    input.fullClasses.push_back({38, {1, 0, 3}, fs::CompactFullWriter::Target, yes, yes, yes, yes, yes});
    auto lower = fs::buildCompactLowerFacts(input);
    if (!lower.error.empty() || lower.records.size() != 2 || lower.decisions.size() != 3 ||
        lower.provenance[0].group != 37 || lower.provenance[1].group != 38 ||
        lower.decisions[1].firstRecord != 0 || lower.decisions[2].firstRecord != 1) { return false; }
    input.fullClasses[0].fullClassWrite = fs::CompactFactProof::Unknown;
    input.fullClasses[1].samePhysicalClass = fs::CompactFactProof::Unknown;
    lower = fs::buildCompactLowerFacts(input);
    if (!lower.error.empty() || !lower.records.empty()) { return false; }
    input = familyInput(); input.rotating.push_back(rotating(UINT64_MAX, 0));
    lower = fs::buildCompactLowerFacts(input);
    if (!lower.error.empty() || lower.records.size() != 3 || lower.records.back().displacement != 1) { return false; }
    input.rotating[0].stride = 1;
    lower = fs::buildCompactLowerFacts(input);
    if (!lower.error.empty() || lower.records.back().displacement != UINT64_MAX || lower.cost.gcdSteps > 2) {
        return false;
    }
    input.rotating[0].slots = 0;
    lower = fs::buildCompactLowerFacts(input);
    if (lower.error.empty() || !lower.records.empty() || !lower.provenance.empty()) { return false; }
    input = familyInput(); input.writers[0].family = 1;
    if (fs::buildCompactLowerFacts(input).error.empty()) { return false; }
    input = familyInput(); input.reads[0].alternatives[0].minimum = 3;
    if (fs::buildCompactLowerFacts(input).error.empty()) { return false; }
    input = familyInput(); input.fullClasses.push_back({});
    lower = fs::buildCompactLowerFacts(input);
    if (!lower.error.empty() || lower.records.size() != 1 ||
        lower.decisions.back().reason != fs::CompactLowerReason::NotForward) { return false; }
    lower = fs::buildCompactLowerFacts({});
    return lower.error.empty() && lower.records.empty() && lower.decisions.empty();
}
} // namespace
int runCompactLowerFactsChecks()
{
    if (!familyOracle() || !familyFailures() || !rotatingOracle() || !fullClassAndMalformed()) {
        llvm::errs() << "compact lower fact checks failed\n";
        return 1;
    }
    llvm::outs() << "compact lower fact checks passed\n";
    return 0;
}
