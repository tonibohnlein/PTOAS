// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent physical-conflict oracle for analysis-only mixed-stride expansion.
#include "PTO/Transforms/FrontierSynch/MixedStrideAnalysis.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "llvm/Support/raw_ostream.h"
namespace fs = mlir::pto::frontiersynch;
namespace {
using Matrix = std::vector<std::vector<bool>>;
Matrix reference(unsigned trips, llvm::ArrayRef<fs::PeriodicPayload> payloads,
                 llvm::ArrayRef<fs::RotatingFragment> fragments)
{
    const auto occurrences = trips * payloads.size();
    Matrix graph(2 * occurrences, std::vector<bool>(2 * occurrences));
    for (std::size_t a = 0; a < occurrences; ++a) {
        graph[2*a][2*a] = graph[2*a+1][2*a+1] = graph[2*a][2*a+1] = true;
        for (std::size_t b = a + 1; b < occurrences; ++b) {
            if (payloads[a % payloads.size()].pipe == payloads[b % payloads.size()].pipe) {
                graph[2*a][2*b] = graph[2*a+1][2*b+1] = true;
            }
            for (const auto& x : fragments) {
                if (x.payload != a % payloads.size()) { continue; }
                for (const auto& y : fragments) {
                    if (y.payload != b % payloads.size() || x.family != y.family || x.atom != y.atom) { continue; }
                    const auto sx = (x.stride * (a / payloads.size()) + x.offset) % x.slots;
                    const auto sy = (y.stride * (b / payloads.size()) + y.offset) % y.slots;
                    if (sx == sy && ((x.write && (y.read || y.write)) || (x.read && y.write))) {
                        graph[2*a+1][2*b] = true;
                    }
                }
            }
        }
    }
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            if (!graph[a][k]) { continue; }
            for (std::size_t b = 0; b < graph.size(); ++b) { graph[a][b] = graph[a][b] || graph[k][b]; }
        }
    }
    return graph;
}
bool compare(unsigned slots, unsigned first, unsigned second, unsigned offset, unsigned trips)
{
    std::vector<fs::PeriodicPayload> payloads{{0}, {1}, {2}};
    std::vector<fs::RotatingFragment> fragments{
        {0, 0, 0, slots, first, 0, false, true, 0},
        {1, 0, 0, slots, second, offset, true, false, 0},
        {2, 0, 0, slots, second, 0, true, true, 0}};
    auto expanded = fs::expandMixedStride(payloads, fragments);
    if (!expanded.error.empty()) { return false; }
    auto extracted = fs::extractRotatingGenerators(expanded.payloads, expanded.fragments);
    if (!extracted.error.empty()) { return false; }
    auto analysis = fs::analyzePeriodicDemands(expanded.payloads, extracted.generators);
    if (!analysis.error.empty()) { return false; }
    auto graph = reference(trips, payloads, fragments);
    for (std::size_t a = 0; a < graph.size(); ++a) {
        for (std::size_t b = 0; b < graph.size(); ++b) {
            const auto source = a / 2, target = b / 2;
            auto answer = analysis.eventPrecedes(
                {static_cast<uint32_t>(source % expanded.payloads.size()),
                    a % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start},
                source / expanded.payloads.size(),
                {static_cast<uint32_t>(target % expanded.payloads.size()),
                    b % 2 ? fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start},
                target / expanded.payloads.size(), trips * payloads.size());
            if (answer.error != fs::PeriodicQueryError::None || answer.value != graph[a][b]) {
                llvm::errs() << "mixed-stride mismatch " << slots << ":" << first << ":" << second
                             << ":" << offset << ":" << trips << " events=" << a << "," << b << "\n";
                return false;
            }
        }
    }
    return true;
}
bool boundaries()
{
    const std::vector<fs::PeriodicPayload> payloads{{0}, {1}};
    const std::vector<fs::RotatingFragment> input{
        {0, 0, 0, 7, 0, 0, false, true, 0}, {1, 0, 0, 7, 1, 0, true, false, 0}};
    if (fs::expandMixedStride(payloads, input, 13).error.empty() ||
        !fs::expandMixedStride(payloads, input, 14).error.empty()) { return false; }
    auto bad = input;
    bad.front().slots = 0;
    if (fs::expandMixedStride(payloads, bad).error.empty()) { return false; }
    bad = input; bad.front().payload = 2;
    if (fs::expandMixedStride(payloads, bad).error.empty()) { return false; }
    bad = input; bad.front().slots = 6;
    if (fs::expandMixedStride(payloads, bad).error.empty()) { return false; }
    auto huge = fs::expandMixedStrideRecords({{0, 1, UINT64_MAX}}, 2, 7);
    if (!huge || huge->size() != 7) { return false; }
    for (uint64_t r = 0; r < 7; ++r) {
        const auto& record = (*huge)[r];
        if (record.source != 2*r || record.target != 2*((r+UINT64_MAX%7)%7)+1 ||
            record.displacement != UINT64_MAX/7+(r+UINT64_MAX%7)/7) { return false; }
    }
    auto protectedInput = input;
    protectedInput[0].protectionGroup = 1 | fs::protectionResetBit;
    protectedInput[1].protectionGroup = 1;
    auto expanded = fs::expandMixedStride(payloads, protectedInput);
    if (!expanded.error.empty() || expanded.period != 7) { return false; }
    const auto a = expanded.fragments[0].protectionGroup;
    const auto b = expanded.fragments[1].protectionGroup;
    return (a & ~fs::protectionResetBit) == b && b != expanded.fragments[3].protectionGroup;
}
} // namespace
int runMixedStrideChecks()
{
    unsigned count = 0;
    for (unsigned slots : {2U, 3U, 4U, 5U}) {
        for (unsigned first : {0U, 1U, 2U}) {
            for (unsigned second : {0U, 1U, 2U}) {
                for (unsigned offset : {0U, 1U}) {
                    for (unsigned trips : {0U, 1U, 2U, 4U, 7U}) {
                        if (!compare(slots, first, second, offset, trips)) { return 1; }
                        ++count;
                    }
                }
            }
        }
    }
    if (!boundaries()) { llvm::errs() << "mixed-stride boundary checks failed\n"; return 1; }
    llvm::outs() << count << " mixed-stride exact prefix comparisons and boundary checks passed\n";
    return 0;
}
