// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ResolvedLifetimeStream.h"
#include <algorithm>
#include <limits>
namespace mlir::pto::frontiersynch {
ResolvedLifetimeStream::ResolvedLifetimeStream(llvm::ArrayRef<uint32_t> pipes,
    uint32_t requestedSpan, uint64_t maximumIterations) : span(requestedSpan)
{
    if (pipes.empty() || pipes.size() > UINT32_MAX || maximumIterations > UINT64_MAX / pipes.size() ||
        uint64_t(span) + 1 > std::numeric_limits<std::size_t>::max() / pipes.size()) {
        constructionError = "resolved lifetime dimensions or rank domain overflow"; return;
    }
    limit = maximumIterations * pipes.size();
    std::vector<uint32_t> labels(pipes.begin(), pipes.end());
    std::sort(labels.begin(), labels.end()); labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    for (auto pipe : pipes) {
        siteColumns.push_back(static_cast<uint32_t>(
            std::lower_bound(labels.begin(), labels.end(), pipe) - labels.begin()));
    }
    const auto slots = pipes.size() * (uint64_t(span) + 1);
    if (slots > history.max_size() || labels.size() > counters.max_size() ||
        slots > std::numeric_limits<std::size_t>::max() / labels.size()) {
        constructionError = "resolved lifetime register storage overflow"; return;
    }
    counters.assign(labels.size(), 0);
    starts.assign(labels.size(), counters); completions = starts;
    history.assign(static_cast<std::size_t>(slots), Slot{UINT64_MAX, 0, counters});
}
ResolvedLifetimeStep ResolvedLifetimeStream::advance(uint64_t occurrence, bool present,
    llvm::ArrayRef<uint64_t> candidates, llvm::ArrayRef<uint64_t> native)
{
    ResolvedLifetimeStep out;
    auto fail = [&](const char* message) { out.error = message; return out; };
    if (!constructionError.empty() || occurrence >= limit || (started && occurrence <= last)) {
        return fail("resolved lifetime occurrence is outside the increasing invocation domain");
    }
    if (!present) {
        if (!candidates.empty() || !native.empty()) { return fail("absent occurrence has resolved incoming edges"); }
        last = occurrence; started = true; return out;
    }
    auto valid = [&](uint64_t source) {
        return source < occurrence && occurrence / siteColumns.size() - source / siteColumns.size() <= span &&
            history[source % history.size()].stamp == source;
    };
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (!valid(candidates[i]) || (i && candidates[i - 1] <= candidates[i])) {
            return fail("resolved lifetime candidates require present recent sources in descending order");
        }
    }
    for (auto source : native) {
        if (!valid(source)) { return fail("resolved native prerequisite has no present recent source"); }
    }
    const auto column = siteColumns[occurrence % siteColumns.size()];
    if (counters[column] == UINT64_MAX) { return fail("resolved lifetime rank overflow"); }
    // Reserve a conservative work bound before mutating persistent registers.
    const auto k = counters.size();
    if (native.size() > UINT64_MAX - 2 || candidates.size() > UINT64_MAX - native.size() - 2 ||
        candidates.size() + native.size() + 2 > UINT64_MAX / k ||
        costs.occurrences == UINT64_MAX || candidates.size() > UINT64_MAX - costs.candidates ||
        candidates.size() > UINT64_MAX - costs.retained || native.size() > UINT64_MAX - costs.native ||
        k * (candidates.size() + native.size() + 2) > UINT64_MAX - costs.rowEntries) {
        return fail("resolved lifetime work counter overflow");
    }
    out.rank = counters[column] + 1;
    out.start = starts[column];
    auto join = [&](std::vector<uint64_t>& destination, const std::vector<uint64_t>& source) {
        for (std::size_t p = 0; p < k; ++p) { destination[p] = std::max(destination[p], source[p]); }
        costs.rowEntries += k;
    };
    for (auto source : native) { join(out.start, history[source % history.size()].completion); }
    for (auto source : candidates) {
        const auto& saved = history[source % history.size()];
        if (out.start[siteColumns[source % siteColumns.size()]] < saved.rank) {
            out.retained.push_back(source); join(out.start, saved.completion);
        }
    }
    out.completion = out.start;
    join(out.completion, completions[column]);
    out.completion[column] = out.rank;
    starts[column] = out.start; completions[column] = out.completion; counters[column] = out.rank;
    history[occurrence % history.size()] = {occurrence, out.rank, out.completion};
    ++costs.occurrences; costs.candidates += candidates.size(); costs.retained += out.retained.size();
    costs.native += native.size();
    last = occurrence; started = true;
    return out;
}
} // namespace mlir::pto::frontiersynch
