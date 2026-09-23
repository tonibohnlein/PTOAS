// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Immutable occurrence pairing over original control. Sharing an emitted word
// does not identify dynamic histories. Completion and binding remain separate.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
const OccurrenceCorrespondence& Control::correspondence(Cut publication, Cut acquisition, Id budget) const
{
    static const OccurrenceCorrespondence unknown{
        ProofOutcome::Unknown, "invalid occurrence boundary", NoAnalysisId, 0, {}};
    if (!complete || publication >= canonicalCut.size() || acquisition >= canonicalCut.size()) {
        return unknown;
    }
    const auto key = std::make_tuple(canonicalCut[publication], canonicalCut[acquisition], budget);
    const auto found = correspondences.find(key);
    if (found != correspondences.end()) {
        return found->second;
    }
    auto result = pairOccurrences(std::get<0>(key), std::get<1>(key), budget);
    occurrenceAnalysisSites += result.siteEvaluations;
    return correspondences.emplace(key, std::move(result)).first->second;
}

OccurrenceCorrespondence Control::pairOccurrences(Cut publication, Cut acquisition, Id budget) const
{
    OccurrenceCorrespondence result;
    auto refuse = [&](ProofOutcome outcome, Cut cut, const char* reason) {
        result.outcome = outcome;
        result.reason = reason;
        result.failedAt = cut;
        result.pairs.clear();
        return result;
    };
    std::vector<std::pair<Cut, Cut>> pending{{graph.entry, NoAnalysisId}};
    std::set<std::pair<Cut, Cut>> seen, pairs;
    while (!pending.empty()) {
        auto [at, source] = pending.back();
        pending.pop_back();
        if (!seen.insert({at, source}).second) {
            continue;
        }
        if (result.siteEvaluations == budget) {
            return refuse(ProofOutcome::Unknown, at, "occurrence analysis budget exhausted");
        }
        ++result.siteEvaluations;
        if (canonicalCut[at] == publication) {
            if (source != NoAnalysisId) {
                return refuse(ProofOutcome::Disproved, at, "publication repeats before consumption");
            }
            source = at;
        }
        if (canonicalCut[at] == acquisition) {
            if (source == NoAnalysisId) {
                return refuse(ProofOutcome::Disproved, at, "acquisition has no participating publication");
            }
            pairs.emplace(source, at);
            source = NoAnalysisId;
        }
        if (graph.sites[at].successors.empty() && source != NoAnalysisId) {
            return refuse(ProofOutcome::Disproved, at, "publication reaches exit unconsumed");
        }
        for (auto next : graph.sites[at].successors) {
            pending.emplace_back(next, source);
        }
    }
    if (pairs.empty()) {
        return refuse(ProofOutcome::Unknown, publication, "no participating endpoint occurrence");
    }
    result.pairs.assign(pairs.begin(), pairs.end());
    result.outcome = ProofOutcome::Proved;
    return result;
}
} // namespace mlir::pto::oahs::selected
