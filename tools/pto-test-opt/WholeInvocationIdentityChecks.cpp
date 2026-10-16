// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Verify unary composition ownership and request-sensitive construction.
#include "../../lib/PTO/Transforms/FrontierSynch/WholeInvocationIdentity.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::string printIR(Operation* operation)
{
    std::string result;
    llvm::raw_string_ostream stream(result);
    operation->print(stream);
    return result;
}
std::string printPreparation(const fs::PreparedLogicalPlan& plan)
{
    std::string result;
    llvm::raw_string_ostream stream(result);
    for (const auto& stage : plan.preparation) { stage.code->print(stream); }
    return result;
}
bool freshPlans(const fs::PreparedLogicalPlan& a, const fs::PreparedLogicalPlan& b)
{
    llvm::DenseSet<Operation*> owned;
    llvm::DenseSet<Block*> blocks;
    for (const auto& stage : a.preparation) {
        blocks.insert(stage.code.get());
        stage.code->walk([&](Operation* operation) {
            owned.insert(operation);
            for (auto& region : operation->getRegions()) {
                for (auto& block : region) { blocks.insert(&block); }
            }
        });
    }
    auto borrowed = [&](Value value) {
        if (!value) { return false; }
        if (auto argument = dyn_cast<BlockArgument>(value)) { return blocks.contains(argument.getOwner()); }
        return owned.contains(value.getDefiningOp());
    };
    bool independent = true;
    for (const auto& stage : b.preparation) {
        if (blocks.contains(stage.code.get())) { return false; }
        stage.code->walk([&](Operation* operation) {
            independent &= !owned.contains(operation) && !llvm::any_of(operation->getOperands(), borrowed);
            for (auto& region : operation->getRegions()) {
                for (auto& block : region) { independent &= !blocks.contains(&block); }
            }
        });
    }
    for (const auto& endpoint : b.endpoints) {
        independent &= !borrowed(endpoint.guard) && !borrowed(endpoint.identity) &&
            !llvm::any_of(endpoint.memberCoordinates, borrowed);
    }
    return independent;
}
bool retainedContinuationOwnership()
{
    fs::AnalysisSessionState state;
    fs::AnalysisRequest request;
    request.region = 7;
    auto first = std::make_shared<fs::MathematicalResult>();
    auto second = std::make_shared<fs::MathematicalResult>();
    const std::vector<fs::AnalysisBackend> eligible{fs::AnalysisBackend::VaryingBoundary,
        fs::AnalysisBackend::Arithmetic};
    state.attempts[7][eligible[0]].mathematical = first;
    state.attempts[7][eligible[1]].mathematical = second;
    fs::AnalysisOutcome answer;
    answer.status = fs::AnalysisStatus::Ready;
    answer.mathematical = second;
    answer.available.queries = answer.available.selectors = true;
    state.selectedAnswers[{7, uint8_t{1}, false}] = answer;
    answer.mathematical = first;
    state.selectedAnswers[{7, uint8_t{3}, false}] = answer;
    // Compatible zero-work answers tie in canonical eligible order, rather
    // than the capability-key order used to store the answers.
    auto retained = fs::retainedIdentityAnswer(state, request, eligible);
    if (!retained || retained->second != eligible[0] || retained->first.mathematical != first) { return false; }
    request.needs.queries = true;
    retained = fs::retainedIdentityAnswer(state, request, eligible);
    if (!retained || retained->second != eligible[1]) { return false; } // Exact key has precedence.
    request.needs.synchronization = true;
    if (fs::retainedIdentityAnswer(state, request, eligible)) { return false; }
    request.needs = {};
    request.needs.evaluation = fs::AnalysisEvaluation::Stateful;
    if (fs::retainedIdentityAnswer(state, request, eligible)) { return false; }
    request.needs.evaluation = fs::AnalysisEvaluation::Uniform;
    request.mode = fs::AnalysisMode::Fallback;
    if (fs::retainedIdentityAnswer(state, request, eligible)) { return false; }
    request.mode = fs::AnalysisMode::MinimumExact;
    state.certifying = true;
    if (fs::retainedIdentityAnswer(state, request, eligible)) { return false; }
    state.certifying = false;
    state.attempts[7][eligible[0]].mathematical.reset();
    state.attempts[7][eligible[1]].mathematical.reset();
    // An active Sequence or a pre-Sequence backend cannot authorize a same-node
    // cache hit even if it owns the identical mathematical pointer.
    state.attempts[7][fs::AnalysisBackend::Sequence].mathematical = first;
    state.attempts[7][fs::AnalysisBackend::BoundedLifetime].mathematical = second;
    if (fs::retainedIdentityAnswer(state, request, eligible)) { return false; }
    state.attempts[7][eligible[0]].mathematical = first;
    first->identityComposition = std::make_shared<fs::WholeInvocationIdentity>();
    return !fs::retainedIdentityAnswer(state, request, eligible) && state.identityAnswers.empty() &&
        state.selectedAnswers.size() == 2;
}
bool continuationSelectionProfile(const fs::AnalysisLedger& ledger)
{
    bool observed = false;
    for (const auto& stage : ledger.stages) {
        if (stage.method != "identity-continuation") { continue; }
        observed = true;
        // A same-region identity lookup must not launch another candidate
        // walk. A miss resumes ordinary Sequence construction; the existing
        // outer dispatcher remains responsible for later alternatives.
        for (const auto& selection : ledger.selections) {
            if (selection.request == stage.request) { return false; }
        }
        for (const auto& nested : ledger.stages) {
            if (nested.request == stage.request && nested.stage == "producer-attempt") { return false; }
        }
    }
    return observed;
}
bool coldQueries(func::FuncOp function, pto::GMAliasPolicy policy, fs::AnalysisNeeds queries)
{
    fs::FrontierAnalysis cold(function);
    if (failed(cold.initialize(policy))) { return false; }
    auto first = cold.minimumDemands(0, queries);
    const auto work = cold.constructionCounts().mathematicalAttempts;
    return first.status == fs::AnalysisStatus::Ready && first.mathematical &&
        first.mathematical->identityComposition && cold.minimumDemands(0, queries).mathematical == first.mathematical &&
        cold.constructionCounts().mathematicalAttempts == work &&
        !cold.constructionCounts().arithmeticSelectorBuilds && !cold.constructionCounts().logicalPreparations;
}
bool retainedRequests(fs::FrontierAnalysis& session, const fs::AnalysisOutcome& weak)
{
    const auto counts = session.constructionCounts();
    const auto builds = session.arithmeticRegionConstructions();
    if (counts.logicalPreparations || counts.allocationExports || counts.arithmeticSelectorBuilds ||
        counts.arithmeticQueryBuilds || session.minimumDemands(0).mathematical != weak.mathematical ||
        session.arithmeticRegionConstructions() != builds) { return false; }
    fs::AnalysisNeeds queries;
    queries.queries = true;
    auto strong = session.minimumDemands(0, queries), warm = session.minimumDemands(0, queries);
    return strong.status == fs::AnalysisStatus::Ready && strong.mathematical == weak.mathematical &&
        warm.mathematical == weak.mathematical && strong.regionalExports == warm.regionalExports &&
        strong.available.queries && !session.constructionCounts().arithmeticSelectorBuilds;
}
bool checkPositive(func::FuncOp function, fs::FrontierAnalysis& session, pto::GMAliasPolicy policy)
{
    const auto before = printIR(function);
    session.enableStageProfiling();
    auto weak = session.minimumDemands(0);
    if (weak.status != fs::AnalysisStatus::Ready || !weak.mathematical ||
        !weak.mathematical->identityComposition) { return false; }
    if (!retainedContinuationOwnership() || !continuationSelectionProfile(*session.diagnosticLedger())) {
        return false;
    }
    const auto& proof = *weak.mathematical->identityComposition;
    if (!proof.child || proof.region || !proof.childRegion ||
        !proof.applies(session.input(), proof.index.get(), session.result(), function)) { return false; }
    fs::AnalysisNeeds queries;
    queries.queries = true;
    if (!retainedRequests(session, weak) || !coldQueries(function, policy, queries)) { return false; }
    auto first = session.prepareLogical(weak), second = session.prepareLogical(weak);
    if (failed(first) || failed(second) || !(*first)->preparation.empty() || !(*second)->preparation.empty() ||
        !(*first)->endpoints.empty() || !(*second)->endpoints.empty() ||
        !(*first)->familyRecipe || !(*second)->familyRecipe || first->get() == second->get() ||
        !(*first)->completeInvocation || !(*second)->completeInvocation ||
        (*first)->mathematicalOwner != weak.mathematical || (*second)->mathematicalOwner != weak.mathematical) {
        return false;
    }
    // Immutable recipes may be shared; each consumer must construct its own
    // detached SSA before checking lifetime and allocation independence.
    if (failed(fs::materializeLogicalFamilies(**first)) || failed(fs::materializeLogicalFamilies(**second)) ||
        (*first)->preparation.empty() || (*second)->preparation.empty() ||
        (*first)->endpoints.empty() || (*second)->endpoints.empty() || !freshPlans(**first, **second) ||
        failed(session.attachAllocation(weak, **first)) || !(*first)->allocationCertificate ||
        session.constructionCounts().arithmeticSelectorBuilds || printIR(function) != before) { return false; }
    fs::AnalysisRequest childLogical;
    childLogical.region = proof.childRegion;
    childLogical.needs.synchronization = true;
    const auto preparations = session.constructionCounts().logicalPreparations;
    auto childReady = session.analyze(childLogical);
    if (childReady.status != fs::AnalysisStatus::Ready || childReady.mathematical != proof.child ||
        session.constructionCounts().logicalPreparations != preparations) { return false; }
    const auto surviving = printPreparation(**second);
    first->reset();
    if (printPreparation(**second) != surviving ||
        failed(session.attachAllocation(weak, **second, {0, 1, 2, 3, 4, 5}))) { return false; }
    // Foreign indexes and a rebuilt alias context cannot reuse this owner.
    fs::PhaseIndex foreign;
    if (failed(foreign.build(function, *session.input())) ||
        proof.applies(session.input(), &foreign, session.result(), function)) { return false; }
    const auto other = policy == pto::GMAliasPolicy::MayAlias ?
        pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
    if (failed(session.initialize(other)) || succeeded(session.prepareLogical(weak)) ||
        proof.applies(session.input(), proof.index.get(), session.result(), function)) { return false; }
    auto changed = session.minimumDemands(0);
    return changed.mathematical && changed.mathematical != weak.mathematical && printIR(function) == before;
}
} // namespace
int runWholeInvocationIdentityChecks(func::FuncOp function, pto::GMAliasPolicy policy)
{
    fs::FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return 1; }
    if (function->hasAttr("test.identity_rejected")) {
        fs::PhaseIndex index;
        if (failed(index.build(function, *session.input())) ||
            fs::wholeInvocationLoop(function, *session.input(), *session.result(), index)) { return 1; }
        llvm::outs() << "identity-session: outside payload/control rejected\n";
        return 0;
    }
    if (function->hasAttr("test.identity_early")) {
        auto result = session.minimumDemands(0);
        if (result.status != fs::AnalysisStatus::Ready || !result.mathematical ||
            result.mathematical->identityComposition || result.mathematical->concreteDemands() ||
            session.arithmeticGeneratorConstructions() || session.arithmeticRegionConstructions() ||
            session.result()->arithmetic || !session.result()->arithmeticContracts.empty() ||
            session.constructionCounts().logicalPreparations) { return 1; }
        llvm::outs() << "identity-session: early compact success avoids arithmetic\n";
        return 0;
    }
    if (function->hasAttr("test.identity_explicit")) {
        auto result = session.minimumDemands(0);
        if (result.status != fs::AnalysisStatus::Ready || !result.mathematical->concreteDemands() ||
            session.arithmeticGeneratorConstructions() || session.arithmeticRegionConstructions() ||
            session.constructionCounts().logicalPreparations) { return 1; }
        llvm::outs() << "identity-session: explicit success avoids arithmetic\n";
        return 0;
    }
    if (!checkPositive(function, session, policy)) {
        function.emitError("unary identity ownership or lazy request check failed");
        return 1;
    }
    llvm::outs() << "identity-session: retained child fresh plans query-only allocation isolated context\n";
    return 0;
}
