// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/GuardedPublication.h"
namespace mlir::pto::frontiersynch {
const char* endpointFailureStageName(EndpointFailureStage stage)
{
    switch (stage) {
    case EndpointFailureStage::None: return "none";
    case EndpointFailureStage::ConsumerGuard: return "consumer-guard";
    case EndpointFailureStage::SourceGuard: return "source-guard";
    case EndpointFailureStage::PublicationContract: return "publication-contract";
    case EndpointFailureStage::CleanupGuard: return "cleanup-guard";
    case EndpointFailureStage::MatchingIdentity: return "matching-identity";
    case EndpointFailureStage::CleanupIdentity: return "cleanup-identity";
    }
    return "invalid-stage";
}
namespace {
RegionExpressions::Id structuralPresence(func::FuncOp function, Operation* cut, RegionExpressions& arena)
{
    SmallVector<RegionExpressions::Id> path;
    auto* descendant = cut;
    for (auto* parent = cut->getParentOp(); parent != function; parent = parent->getParentOp()) {
        // The cut qualifier admits only branches and normalized, single-
        // execution physical sections here. Section presence is carried by
        // the original branches materialized by the context coordinator.
        if (auto branch = dyn_cast<scf::IfOp>(parent)) {
            auto condition = arena.input(branch.getCondition());
            auto* region = descendant->getParentRegion();
            path.push_back(region == &branch.getThenRegion() ? condition : arena.lnot(condition));
        }
        descendant = parent;
    }
    // Mask arm-local poison with its enclosing branch before using it at the
    // epilogue. A conjunction does not provide that select semantics.
    auto present = arena.boolean(true);
    for (auto condition : llvm::reverse(path)) {
        present = arena.select(present, condition, arena.boolean(false));
    }
    return present;
}
} // namespace
FailureOr<GuardedPublication> prepareGuardedPublication(func::FuncOp function,
    RegionExpressions& arena, TemplateEndpointCut source, TemplateEndpointCut consumer,
    RegionExpressions::Id sourceGuard, RegionExpressions::Id activity,
    const EndpointExpressionEmitter& emit, std::string& error, Value consumerPredicate,
    std::optional<SelectedPublication> selectedSource, EndpointEmissionFailure* failureInfo)
{
    auto attempt = [&](RegionExpressions::Id expression, Operation* cut, EndpointFailureStage stage) {
        auto value = emit(expression, cut);
        const bool failedAttempt = failed(value) && failureInfo;
        if (failedAttempt) {
            auto diagnostic = failureInfo->expression == expression && failureInfo->cut == cut ?
                failureInfo->diagnostic : arena.lastEmissionError();
            *failureInfo = {stage, expression, cut, std::move(diagnostic)};
        }
        return value;
    };
    if (!function || !source.before || !consumer.before || !emit ||
        sourceGuard >= arena.size() || activity >= arena.size() ||
        !arena.isBoolean(sourceGuard) || !arena.isBoolean(activity)) {
        error = "publication requires original cuts and Boolean activity"; return failure();
    }
    if (selectedSource && (selectedSource->existence >= arena.size() || selectedSource->atCut >= arena.size() ||
        !arena.isBoolean(selectedSource->existence) || !arena.isBoolean(selectedSource->atCut))) {
        error = "selected publication requires Boolean existence and visit predicates"; return failure();
    }
    auto ordinary = attempt(sourceGuard, source.before, EndpointFailureStage::SourceGuard);
    if (succeeded(ordinary)) { return GuardedPublication{*ordinary, activity}; }
    if (!function.getBody().hasOneBlock()) {
        error = "terminal cleanup requires a single-exit invocation"; return failure();
    }
    auto* terminal = function.front().getTerminator();
    TemplateEndpointCut final{terminal->getBlock(), terminal};
    if (failureInfo) {
        *failureInfo = {EndpointFailureStage::PublicationContract, sourceGuard, source.before, {}};
    }
    auto choices = qualifyTerminalCleanupCuts({consumer, final}, consumerPredicate);
    if (!choices || (!selectedSource && !qualifyTerminalCleanupCuts({source, final}))) {
        error = "late SET guard requires single-publication cuts and a terminal cleanup interface";
        return failure();
    }
    auto present = selectedSource ? selectedSource->existence : structuralPresence(function, source.before, arena);
    auto atCut = selectedSource ? selectedSource->atCut : present;
    // Qualify compact exact predicates before creating scalar code. Some
    // unmigrated relational recipes expose their singleton equality only when
    // lowered; retain that existing acceptance path through the same proof.
    const bool selectedBeforeLowering = qualifyTerminalCleanupSelection({source, final}, arena, atCut);
    auto publish = attempt(atCut, source.before, EndpointFailureStage::SourceGuard);
    if (failed(publish)) {
        error = "early publication predicate unavailable: " +
            (failureInfo ? failureInfo->diagnostic : arena.lastEmissionError()); return failure();
    }
    auto cleanup = attempt(arena.select(present, arena.lnot(activity), arena.boolean(false)), terminal,
        EndpointFailureStage::CleanupGuard);
    if (failed(cleanup)) {
        error = "terminal cleanup predicate unavailable: " +
            (failureInfo ? failureInfo->diagnostic : arena.lastEmissionError());
        return failure();
    }
    if (!selectedBeforeLowering && !qualifyTerminalCleanupCuts({source, final}, *publish)) {
        error = "early publication does not select a single original source occurrence"; return failure();
    }
    return GuardedPublication{*publish, present, terminal, *cleanup, std::move(choices)};
}
LogicalResult appendPublicationCleanup(const GuardedPublication& publication,
    uint32_t record, uint32_t sourcePipe, uint32_t targetPipe,
    ArrayRef<RegionExpressions::Id> identity, RegionExpressions& arena,
    const EndpointExpressionEmitter& emit, PreparedLogicalPlan& plan, std::string& error)
{
    if (!publication.cleanupCut) { return success(); }
    if (identity.empty()) { error = "cleanup requires the publication's matching tuple"; return failure(); }
    auto ordinal = emit(identity.back(), publication.cleanupCut);
    auto member = emit(arena.constant(record), publication.cleanupCut);
    if (failed(ordinal) || failed(member)) {
        error = "terminal cleanup matching identity unavailable"; return failure();
    }
    PreparedLogicalEndpoint endpoint{publication.cleanupCut, LogicalCommandKind::Wait,
        sourcePipe, targetPipe, record, publication.cleanupGuard, *ordinal};
    endpoint.records.push_back(record);
    endpoint.piece = plan.endpoints.size();
    endpoint.memberCoordinates.push_back(*member);
    for (auto coordinate : identity.drop_back()) {
        auto value = emit(coordinate, publication.cleanupCut);
        if (failed(value)) { error = "terminal cleanup matching coordinate unavailable"; return failure(); }
        endpoint.memberCoordinates.push_back(*value);
    }
    plan.endpoints.push_back(std::move(endpoint));
    return success();
}
} // namespace mlir::pto::frontiersynch
