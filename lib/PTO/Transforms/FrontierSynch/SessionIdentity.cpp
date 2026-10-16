// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A transparent Sequence wrapper transports one retained selected graph.
#include "WholeInvocationIdentity.h"
#include <iterator>
namespace mlir::pto::frontiersynch {
bool WholeInvocationIdentity::applies(const SyncInput* expectedInput, const PhaseIndex* expectedIndex,
    const ProgramRecognition* expectedProgram, func::FuncOp expectedFunction) const
{
    if (!input || !index || !program || input.get() != expectedInput || index.get() != expectedIndex ||
        program.get() != expectedProgram ||
        function != expectedFunction || !child || child->input != input || child->region != childRegion ||
        region >= program->nodes.size() || childRegion >= program->nodes.size()) { return false; }
    const auto* original = wholeInvocationLoop(expectedFunction, *input, *program, *index);
    if (!original || original->anchor != loop) { return false; }
    const auto originalRegion = static_cast<std::size_t>(original - program->nodes.data());
    return (region == 0 || region == originalRegion) && childRegion == originalRegion;
}
std::pair<AnalysisOutcome, AnalysisBackend> FrontierAnalysis::selectIdentityChild(
    const AnalysisRequest& request, bool continueSameRegion)
{
    AnalysisOutcome selected;
    AnalysisBackend selectedBackend = AnalysisBackend::Explicit;
    if (!continueSameRegion) {
        selected = analyze(request);
        const auto& attempts = sessionState->attempts[request.region];
        auto found = llvm::find_if(attempts, [&](const auto& entry) {
            return selected.mathematical && entry.second.mathematical == selected.mathematical;
        });
        if (found == attempts.end()) { return {std::move(selected), selectedBackend}; }
        selectedBackend = found->first;
    } else { return continueIdentityChild(request); }
    return {std::move(selected), selectedBackend};
}
std::optional<std::pair<AnalysisOutcome, AnalysisBackend>> retainedIdentityAnswer(
    const AnalysisSessionState& session, const AnalysisRequest& request, ArrayRef<AnalysisBackend> eligible)
{
    const auto region = session.attempts.find(request.region);
    if (region == session.attempts.end() || request.needs.evaluation != AnalysisEvaluation::Uniform) { return {}; }
    auto match = [&](const AnalysisOutcome& answer) -> std::optional<AnalysisBackend> {
        if (answer.status != AnalysisStatus::Ready || !answer.mathematical || answer.completion ||
            answer.mathematical->identityComposition || answer.mathematical->compactDemands ||
            (request.needs.queries && !answer.available.queries) ||
            (request.needs.selectors && !answer.available.selectors) ||
            (request.needs.synchronization && !answer.available.synchronization)) { return {}; }
        for (const auto backend : eligible) {
            auto attempt = region->second.find(backend);
            if (attempt != region->second.end() && attempt->second.mathematical == answer.mathematical) {
                return backend;
            }
        }
        return {};
    };
    const auto key = analysisRequestKey(request);
    const auto identity = std::make_tuple(request.region, key, session.certifying);
    auto exact = session.selectedAnswers.find(identity);
    if (exact != session.selectedAnswers.end()) {
        if (auto backend = match(exact->second)) { return std::make_pair(exact->second, *backend); }
    }
    std::optional<std::pair<AnalysisOutcome, AnalysisBackend>> retained;
    auto begin = session.selectedAnswers.lower_bound({request.region, uint8_t{0}, false});
    for (auto it = begin; it != session.selectedAnswers.end() && std::get<0>(it->first) == request.region; ++it) {
        if ((std::get<1>(it->first) & ~uint8_t{7}) != (key & ~uint8_t{7}) ||
            std::get<2>(it->first) != session.certifying) { continue; }
        if (auto backend = match(it->second)) {
            if (!retained || llvm::find(eligible, *backend) < llvm::find(eligible, retained->second)) {
                retained = std::make_pair(it->second, *backend);
            }
        }
    }
    return retained;
}
std::pair<AnalysisOutcome, AnalysisBackend> FrontierAnalysis::continueIdentityChild(const AnalysisRequest& request)
{
    AnalysisProfileScope profile(sessionState->ledger, "request", request.region, "identity-continuation", true);
    const auto identity = std::make_tuple(request.region, analysisRequestKey(request), sessionState->certifying);
    auto finish = [&](std::pair<AnalysisOutcome, AnalysisBackend> answer) {
        answer.first.ledger = sessionState->ledger;
        answer.first.requestId = profile.requestId();
        if (answer.first.status == AnalysisStatus::Ready && answer.first.mathematical &&
            !answer.first.completion && !answer.first.mathematical->compactDemands &&
            !answer.first.mathematical->identityComposition) {
            sessionState->identityAnswers.emplace(identity, answer);
        }
        return answer;
    };
    if (auto found = sessionState->identityAnswers.find(identity); found != sessionState->identityAnswers.end()) {
        profile.cacheHit(); return finish(found->second);
    }
    // An identity adapter may reuse an already complete child, but must not
    // start a second backend walk while the same region's Sequence producer is
    // active. A cold continuation returns to that producer, allowing compact
    // sequence construction before the outer dispatcher tries other algorithms.
    // Keep the existing eligible-owner restriction; active adapters cannot own
    // their own child, and missing exports do not authorize a weaker result.
    const auto canonical = exactAttemptFamilies(request, AnalysisForm::Original, exactForms(request).size() > 1);
    const auto sequence = llvm::find(canonical, AnalysisBackend::Sequence);
    std::vector<AnalysisBackend> candidates;
    if (sequence != canonical.end()) {
        for (auto family = std::next(sequence); family != canonical.end(); ++family) {
            if (*family == AnalysisBackend::SequenceEndpoints) { continue; }
            const auto methods = *family == AnalysisBackend::ArithmeticPeriodic ? arithmeticMethods(request) :
                std::vector<AnalysisBackend>{*family};
            for (const auto backend : methods) {
                if (!llvm::is_contained(candidates, backend)) { candidates.push_back(backend); }
            }
        }
    }
    if (auto retained = retainedIdentityAnswer(*sessionState, request, candidates)) {
        profile.cacheHit(); return finish(*retained);
    }
    return finish({AnalysisOutcome{}, AnalysisBackend::Explicit});
}
std::shared_ptr<const MathematicalResult> FrontierAnalysis::produceIdentitySequence(
    const AnalysisRequest& request, bool endpoints, std::string& error)
{
    const auto region = request.region;
    const auto* loop = wholeInvocationLoop(function, *storage, *program, *structuralIndex);
    if (!loop) { return {}; }
    const auto childRegion = static_cast<std::size_t>(loop - program->nodes.data());
    if (region != 0 && region != childRegion) { return {}; }
    // Select a new immutable child for the capabilities actually requested.
    // A weak cached answer must not pin a fresh strong owner to an unavailable
    // export. Identity composition still requires an exact child even when its
    // caller permits a fallback; existing owners never replace their child.
    auto childRequest = request;
    childRequest.region = childRegion;
    childRequest.mode = AnalysisMode::MinimumExact;
    childRequest.needs.synchronization |= endpoints;
    auto selection = selectIdentityChild(childRequest, region != 0);
    auto& selected = selection.first;
    const auto selectedBackend = selection.second;
    if (selected.status != AnalysisStatus::Ready || !selected.mathematical) {
        error = "transparent sequence has no ready retained child";
        return {};
    }
    auto owner = std::make_shared<WholeInvocationIdentity>();
    owner->input = storage; owner->index = structuralIndex; owner->program = program;
    owner->child = selected.mathematical; owner->function = function; owner->loop = loop->anchor;
    owner->region = region; owner->childRegion = childRegion; owner->childBackend = selectedBackend;
    auto result = std::make_shared<MathematicalResult>();
    result->input = storage; result->recognition = program; result->region = region;
    result->backend = endpoints ? "identity-sequence-endpoints" : "identity-sequence";
    result->identityComposition = std::move(owner);
    return result;
}
AnalysisOutcome FrontierAnalysis::requestIdentity(const AnalysisRequest& request, BackendAttempt& attempt)
{
    AnalysisOutcome result;
    result.mathematical = attempt.mathematical;
    result.status = AnalysisStatus::UnmetObligation;
    result.stage = AnalysisStage::Demands;
    const auto& owner = *result.mathematical->identityComposition;
    if (result.mathematical->region != owner.region ||
        !owner.applies(storage.get(), structuralIndex.get(), program.get(), function)) {
        result.obligations.push_back({result.stage, "identity composition belongs to another input context"});
        return result;
    }
    auto childRequest = request;
    childRequest.region = owner.childRegion;
    childRequest.needs.synchronization = false;
    auto child = requestBackend(owner.childBackend, childRequest);
    if (child.mathematical != owner.child) {
        result.obligations.push_back({result.stage, "identity composition cannot replace its selected child graph"});
        return result;
    }
    result.available = child.available;
    result.regionalExports = child.regionalExports;
    result.obligations = std::move(child.obligations);
    result.status = child.status; result.stage = child.stage;
    result.available.synchronization = attempt.endpoints.value_or(false);
    if (result.status == AnalysisStatus::Ready && request.needs.synchronization) {
        if (!attempt.endpoints) {
            auto prepared = prepareLogical(result);
            attempt.endpoints = succeeded(prepared);
            if (succeeded(prepared)) { attempt.pendingLogical = std::move(*prepared); }
        }
        result.available.synchronization = *attempt.endpoints;
        if (!*attempt.endpoints) {
            result.status = AnalysisStatus::UnmetObligation;
            result.stage = AnalysisStage::Synchronization;
            if (!attempt.prefixEmissionError.empty()) {
                result.obligations.push_back({result.stage, attempt.prefixEmissionError});
            }
            result.obligations.push_back({result.stage, "retained identity child endpoint export is unavailable"});
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
