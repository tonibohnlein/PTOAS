// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Incoming sources are bucketed in decreasing reference order without sorting.
#include "PTO/Transforms/FrontierSynch/ExplicitReduction.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>
namespace mlir::pto::frontiersynch {
namespace {
using RankRow = std::vector<uint32_t>;
void join(RankRow& destination, const RankRow& source)
{
    for (std::size_t column = 0; column < destination.size(); ++column) {
        destination[column] = std::max(destination[column], source[column]);
    }
}
class Reducer {
public:
    ExplicitReduction output;
    bool run(llvm::ArrayRef<ExplicitEffects> occurrences, llvm::ArrayRef<StorageGenerator> generators)
    {
        if (!index(occurrences) || !bucket(generators)) {
            return false;
        }
        const auto count = occurrences.size(), pipes = output.pipeLabels.size();
        // Each occurrence stores two rank rows. Check the total byte product
        // before allocation, in addition to the uint32_t rank bound in index().
        if (pipes != 0 && count > std::numeric_limits<std::size_t>::max() / (2 * sizeof(uint32_t)) / pipes) {
            output.error = "completion rank storage size overflow";
            return false;
        }
        output.startRanks.assign(count, RankRow(pipes, 0));
        output.completionRanks.assign(count, RankRow(pipes, 0));
        std::vector<std::optional<std::size_t>> previous(pipes);
        for (std::size_t position = 0; position < count; ++position) {
            reduceIncoming(position, previous[output.pipeColumns[position]]);
            previous[output.pipeColumns[position]] = position;
        }
        return true;
    }
private:
    std::unordered_map<uint32_t, std::size_t> positions;
    std::vector<std::vector<std::size_t>> incoming;
    bool index(llvm::ArrayRef<ExplicitEffects> occurrences)
    {
        if (occurrences.size() > std::numeric_limits<uint32_t>::max()) {
            output.error = "payload rank overflow";
            return false;
        }
        std::unordered_map<uint32_t, uint32_t> columns;
        std::vector<uint32_t> ranks;
        for (const auto& occurrence : occurrences) {
            if (!positions.emplace(occurrence.payload, positions.size()).second) {
                output.error = "duplicate payload identity";
                return false;
            }
            auto inserted = columns.emplace(occurrence.pipe, static_cast<uint32_t>(columns.size()));
            if (inserted.second) {
                output.pipeLabels.push_back(occurrence.pipe);
                ranks.push_back(0);
            }
            const auto column = inserted.first->second;
            output.payloads.push_back(occurrence.payload);
            output.pipeColumns.push_back(column);
            output.localRanks.push_back(++ranks[column]);
        }
        return true;
    }
    bool bucket(llvm::ArrayRef<StorageGenerator> generators)
    {
        std::vector<std::vector<std::size_t>> outgoing(positions.size());
        for (const auto& edge : generators) {
            auto source = positions.find(edge.source), target = positions.find(edge.target);
            if (source == positions.end() || target == positions.end() || source->second >= target->second) {
                output.error = "nonforward or absent generator endpoint";
                return false;
            }
            outgoing[source->second].push_back(target->second);
        }
        incoming.resize(positions.size());
        for (std::size_t source = positions.size(); source != 0; --source) {
            for (const auto target : outgoing[source - 1]) {
                incoming[target].push_back(source - 1);
            }
        }
        return true;
    }
    void reduceIncoming(std::size_t target, std::optional<std::size_t> previous)
    {
        auto& starts = output.startRanks[target];
        if (previous) {
            // Starts inherit the previous START's prerequisites. Seeding with
            // its completion row would incorrectly serialize overlapping work.
            starts = output.startRanks[*previous];
        }
        for (const auto source : incoming[target]) {
            if (starts[output.pipeColumns[source]] >= output.localRanks[source]) {
                continue;
            }
            output.retained.push_back({output.payloads[source], output.payloads[target]});
            join(starts, output.completionRanks[source]);
        }
        auto& completions = output.completionRanks[target];
        completions = starts;
        if (previous) {
            join(completions, output.completionRanks[*previous]);
        }
        completions[output.pipeColumns[target]] = output.localRanks[target];
    }
};
} // namespace
ExplicitReduction reduceExplicitDemands(llvm::ArrayRef<ExplicitEffects> occurrences,
                                        llvm::ArrayRef<StorageGenerator> generators)
{
    Reducer reducer;
    if (!reducer.run(occurrences, generators)) {
        ExplicitReduction failure;
        failure.error = reducer.output.error;
        return failure;
    }
    return std::move(reducer.output);
}
} // namespace mlir::pto::frontiersynch
