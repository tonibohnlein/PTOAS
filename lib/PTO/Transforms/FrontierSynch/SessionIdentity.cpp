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
    } else {
        // Sequence is an identity on this complete original loop. Consume that
        // adapter once, then continue its canonical family order on this node.
        const bool alternative = exactForms(request).size() > 1;
        const auto order = exactAttemptFamilies(request, AnalysisForm::Original, alternative);
        auto sequence = llvm::find(order, AnalysisBackend::Sequence);
        if (sequence == order.end()) { return {std::move(selected), selectedBackend}; }
        for (auto method = std::next(sequence); method != order.end(); ++method) {
            if (*method == AnalysisBackend::SequenceEndpoints) { continue; }
            const bool ready = visitExactFamily(request, *method, [&](AnalysisBackend backend) {
                auto candidate = requestBackend(backend, request, true);
                if (candidate.status != AnalysisStatus::Ready) { return false; }
                selected = std::move(candidate);
                selectedBackend = backend;
                return true;
            });
            if (ready) { break; }
        }
    }
    return {std::move(selected), selectedBackend};
}
std::shared_ptr<const MathematicalResult> FrontierAnalysis::produceIdentitySequence(
    std::size_t region, bool endpoints, std::string& error)
{
    const auto* loop = wholeInvocationLoop(function, *storage, *program, *structuralIndex);
    if (!loop) { return {}; }
    const auto childRegion = static_cast<std::size_t>(loop - program->nodes.data());
    if (region != 0 && region != childRegion) { return {}; }
    AnalysisRequest request;
    request.region = childRegion;
    request.needs.synchronization = endpoints;
    auto selection = selectIdentityChild(request, region != 0);
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
            result.obligations.push_back({result.stage, "retained identity child endpoint export is unavailable"});
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
