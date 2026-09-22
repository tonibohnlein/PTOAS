// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Cross-word publication certificates use original physical-use boundaries.
// Participation, crossed exports, source-time coverage and key legality are
// separate obligations. No proposed return contributes causal credit.
#include "SelectedInternal.h"

namespace mlir::pto::oahs::selected {
bool Ledger::movePublicationTo(Id publication, Cut target, Id offset, const std::vector<Cut>& crossed)
{
    const bool available = publication < endpoints.size() && active(publication) && target < words.size();
    if (!available) {
        return false;
    }
    target = canonical(target);
    auto& endpoint = endpoints[publication];
    const auto original = endpoint.cut;
    if (target == original || endpoint.command.kind != Command::Publish || offset > words[target].size()) {
        return false;
    }
    const bool containsBoundaries = std::find(crossed.begin(), crossed.end(), target) != crossed.end() &&
        std::find(crossed.begin(), crossed.end(), original) != crossed.end();
    if (!containsBoundaries || std::any_of(crossed.begin(), crossed.end(), [&](Cut cut) {
            return cut >= words.size();
        })) {
        return false;
    }
    const auto protectedEnd = protectedPrefixes.find(target);
    if (protectedEnd != protectedPrefixes.end()) {
        const auto at = std::find(words[target].begin(), words[target].end(), protectedEnd->second);
        const bool crossesProtected = at != words[target].end() && offset <= Id(at - words[target].begin());
        if (crossesProtected) {
            return false;
        }
    }
    std::vector<Id> prefix;
    for (Id i = 0; i < offset; ++i) {
        const auto id = words[target][i];
        const auto& command = endpoints[id].command;
        const auto pipe = command.kind == Command::Acquire ? command.observer : command.source;
        if (pipe == endpoint.command.source) {
            prefix.push_back(id);
        }
    }
    auto& oldWord = words[original];
    const auto oldAt = std::find(oldWord.begin(), oldWord.end(), publication);
    const auto next = std::next(oldAt);
    const auto continuation = next == oldWord.end() ? NoAnalysisId : *next;
    indexPublication(publication, false);
    oldWord.erase(oldAt);
    words[target].insert(words[target].begin() + offset, publication);
    endpoint.cut = target;
    indexPublication(publication, true);
    publicationPrefixes[publication] = std::move(prefix);
    PublicationSpan span{publication, original, continuation, {}};
    for (auto cut : crossed) {
        const auto& ids = word(cut);
        const auto end = canonical(cut) == original && continuation != NoAnalysisId ?
            std::find(ids.begin(), ids.end(), continuation) : ids.end();
        span.words.emplace(canonical(cut), std::vector<Id>(ids.begin(), end));
    }
    publicationSpans.push_back(std::move(span));
    protectPublicationPrefix(publication);
    changed.push_back(original);
    changed.push_back(target);
    ++revision;
    return true;
}

Cut Constructor::lifecycleRelease(const SelectedDecision& decision) const
{
    Cut release = NoAnalysisId;
    for (const auto& required : decision.required) {
        bool found = false;
        for (const auto& fact : requirements.at(decision.consumer)) {
            if (fact.access != accessClass(required)) {
                continue;
            }
            if (fact.lifecycleRelease == NoAnalysisId) {
                return NoAnalysisId;
            }
            found = true;
            if (release == NoAnalysisId || control.position[release] < control.position[fact.lifecycleRelease]) {
                release = fact.lifecycleRelease;
            }
        }
        if (!found) {
            return NoAnalysisId;
        }
    }
    return release;
}

bool Constructor::preservePublicationSpan(Id publication, const SelectedDecision& decision)
{
    if (!options.crossWordPrefixes) {
        return true;
    }
    const auto endpoint = ledger.endpoint(publication);
    auto target = lifecycleRelease(decision);
    if (target == NoAnalysisId || target == endpoint.cut || !control.balancedWords(target, endpoint.cut)) {
        target = NoAnalysisId;
        for (auto gap : control.finalReadGaps) {
            const bool qualified = control.balancedWords(gap, endpoint.cut) &&
                finalReadGap(gap, endpoint.command.source, decision.required);
            if (qualified) {
                target = gap;
                break;
            }
        }
    }
    const auto original = endpoint.cut;
    const bool qualified = target != NoAnalysisId && target != original &&
        control.sourceCut(target, endpoint.command.source) &&
        control.wordOccurrences[control.canonicalCut[target]].size() == 1 &&
        control.wordOccurrences[original].size() == 1 && control.balancedWords(target, original);
    if (!qualified) {
        return true;
    }
    // Retain the last source export at the destination. Only its following
    // suffix is crossed; the exact command gap is part of the certificate.
    const auto& word = ledger.word(target);
    Id offset = word.size();
    while (offset && publicationMayCross(endpoint.command, ledger.endpoint(word[offset - 1]).command)) {
        --offset;
    }
    std::vector<Cut> crossed;
    const bool certified = publicationPositionCovers(publication, target, offset, decision.required) &&
        certifyPublicationOrder(program, control, ledger, frontier.keys(), publication, target, offset, crossed);
    if (!certified) {
        return true;
    }
    Ledger proposed = ledger;
    const bool admissible = proposed.movePublicationTo(publication, target, offset, crossed) &&
        publicationEditValid(proposed);
    if (!admissible) {
        return true;
    }
    const bool moved = ledger.movePublicationTo(publication, target, offset, crossed);
    if (!moved || !update()) {
        return false;
    }
    ++result.work.crossWordPublications;
    return true;
}
} // namespace mlir::pto::oahs::selected
