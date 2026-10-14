// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/GuardedPublication.h"
#include <map>
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
// Both preparation adapters use this policy. Their only differences are how
// availability and singleton multiplicity are established, never the protocol.
FailureOr<GuardedPublicationRecipe> publicationPolicy(func::FuncOp function,
    RegionExpressions& arena, TemplateEndpointCut source, TemplateEndpointCut consumer,
    RegionExpressions::Id sourceGuard, RegionExpressions::Id activity,
    const std::function<LogicalResult(RegionExpressions::Id, Operation*, EndpointFailureStage)>& attempt,
    const std::function<bool(TemplateEndpointCut)>& consumerSingle,
    const std::function<bool(TemplateEndpointCut, RegionExpressions::Id)>& sourceSingle,
    std::string& error, std::optional<SelectedPublication> selectedSource,
    EndpointEmissionFailure* failureInfo = nullptr)
{
    if (!function || !source.before || !consumer.before || !attempt ||
        sourceGuard >= arena.size() || activity >= arena.size() ||
        !arena.isBoolean(sourceGuard) || !arena.isBoolean(activity)) {
        error = "publication requires original cuts and Boolean activity"; return failure();
    }
    if (selectedSource && (selectedSource->existence >= arena.size() || selectedSource->atCut >= arena.size() ||
        !arena.isBoolean(selectedSource->existence) || !arena.isBoolean(selectedSource->atCut))) {
        error = "selected publication requires Boolean existence and visit predicates"; return failure();
    }
    GuardedPublicationRecipe recipe{&arena, source, consumer, sourceGuard, activity};
    if (succeeded(attempt(sourceGuard, source.before, EndpointFailureStage::SourceGuard))) { return recipe; }
    if (!function.getBody().hasOneBlock()) {
        error = "terminal cleanup requires a single-exit invocation"; return failure();
    }
    auto* terminal = function.front().getTerminator();
    TemplateEndpointCut final{terminal->getBlock(), terminal};
    if (failureInfo) {
        *failureInfo = {EndpointFailureStage::PublicationContract, sourceGuard, source.before, {}};
    }
    if (!consumerSingle(final) || (!selectedSource && !qualifyTerminalCleanupCuts({source, final}))) {
        error = "late SET guard requires single-publication cuts and a terminal cleanup interface";
        return failure();
    }
    recipe.publicationGuard = selectedSource ? selectedSource->existence :
        structuralPresence(function, source.before, arena);
    recipe.sourceGuard = selectedSource ? selectedSource->atCut : recipe.publicationGuard;
    const bool selectedBeforeLowering = qualifyTerminalCleanupSelection({source, final}, arena, recipe.sourceGuard);
    if (failed(attempt(recipe.sourceGuard, source.before, EndpointFailureStage::SourceGuard))) {
        error = "early publication predicate unavailable: " +
            (failureInfo ? failureInfo->diagnostic : arena.lastEmissionError()); return failure();
    }
    recipe.cleanupCut = terminal;
    recipe.cleanupGuard = arena.select(recipe.publicationGuard, arena.lnot(activity), arena.boolean(false));
    if (failed(attempt(recipe.cleanupGuard, terminal, EndpointFailureStage::CleanupGuard))) {
        error = "terminal cleanup predicate unavailable: " +
            (failureInfo ? failureInfo->diagnostic : arena.lastEmissionError()); return failure();
    }
    if (!selectedBeforeLowering && !sourceSingle(final, recipe.sourceGuard)) {
        error = "early publication does not select a single original source occurrence"; return failure();
    }
    return recipe;
}
} // namespace
FailureOr<GuardedPublication> prepareGuardedPublication(func::FuncOp function,
    RegionExpressions& arena, TemplateEndpointCut source, TemplateEndpointCut consumer,
    RegionExpressions::Id sourceGuard, RegionExpressions::Id activity,
    const EndpointExpressionEmitter& emit, std::string& error, Value consumerPredicate,
    std::optional<SelectedPublication> selectedSource, EndpointEmissionFailure* failureInfo)
{
    if (!emit) { error = "publication requires an endpoint emitter"; return failure(); }
    std::map<std::pair<RegionExpressions::Id, Operation*>, Value> values;
    auto attempt = [&](RegionExpressions::Id expression, Operation* cut, EndpointFailureStage stage) {
        auto value = emit(expression, cut);
        if (succeeded(value)) { values[{expression, cut}] = *value; return success(); }
        if (failureInfo) {
            auto diagnostic = failureInfo->expression == expression && failureInfo->cut == cut ?
                failureInfo->diagnostic : arena.lastEmissionError();
            *failureInfo = {stage, expression, cut, std::move(diagnostic)};
        }
        return failure();
    };
    std::shared_ptr<const EndpointCutChoices> choices;
    auto consumerSingle = [&](TemplateEndpointCut final) {
        choices = qualifyTerminalCleanupCuts({consumer, final}, consumerPredicate);
        return bool(choices);
    };
    auto sourceSingle = [&](TemplateEndpointCut final, RegionExpressions::Id guard) {
        return bool(qualifyTerminalCleanupCuts({source, final}, values.at({guard, source.before})));
    };
    auto recipe = publicationPolicy(function, arena, source, consumer, sourceGuard, activity,
        attempt, consumerSingle, sourceSingle, error, selectedSource, failureInfo);
    if (failed(recipe)) { return failure(); }
    return GuardedPublication{values.at({recipe->sourceGuard, source.before}), recipe->publicationGuard,
        recipe->cleanupCut, recipe->cleanupCut ? values.at({recipe->cleanupGuard, recipe->cleanupCut}) : Value(),
        std::move(choices)};
}
FailureOr<std::shared_ptr<const GuardedPublicationRecipe>> preparePublicationRecipe(func::FuncOp function,
    RegionExpressions& arena, TemplateEndpointCut source, TemplateEndpointCut consumer,
    RegionExpressions::Id sourceGuard, RegionExpressions::Id activity, RegionExpressions::Id consumerGuard,
    const EndpointExpressionAvailability& available, std::string& error,
    std::optional<SelectedPublication> selectedSource)
{
    error.clear();
    if (!function || !source.before || !consumer.before || !available ||
        source.block != source.before->getBlock() || consumer.block != consumer.before->getBlock() ||
        source.before->getParentOfType<func::FuncOp>() != function ||
        consumer.before->getParentOfType<func::FuncOp>() != function || !arena.isBoolean(consumerGuard)) {
        error = "publication recipe requires original cuts and Boolean consumer activity"; return failure();
    }
    if (failed(available(consumerGuard, consumer.before))) {
        error = "publication consumer predicate unavailable"; return failure();
    }
    auto attempt = [&](RegionExpressions::Id expression, Operation* cut, EndpointFailureStage) {
        return available(expression, cut);
    };
    auto consumerSingle = [&](TemplateEndpointCut final) {
        return qualifyTerminalCleanupSelection({consumer, final}, arena, consumerGuard);
    };
    auto sourceSingle = [](TemplateEndpointCut, RegionExpressions::Id) { return false; };
    auto recipe = publicationPolicy(function, arena, source, consumer, sourceGuard, activity,
        attempt, consumerSingle, sourceSingle, error, selectedSource);
    if (failed(recipe)) { return failure(); }
    return std::make_shared<const GuardedPublicationRecipe>(std::move(*recipe));
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
