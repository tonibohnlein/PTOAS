// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RegionalRequests.h"
#include "GeneralQueries.h"
#include "FixedBodyUpper.h"
#include "StationaryCells.h"
#include "SingleStreamLoop.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
LogicalResult buildCountedReaderQueries(SelectedAnalysis&, const SignedInputs&, const SyncInput&,
                                       CostLedger&, std::string&);
RegionalRequests::RegionalRequests(
    func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace, CostLedger& costs)
    : function(function), input(input), trace(trace), costs(costs)
{
    auto context = std::shared_ptr<RegionalContext>(new RegionalContext());
    context->root = function;
    context->scope = function;
    CostScope recognition(costs, CostStage::Recognition);
    auto index = std::make_shared<PhaseIndex>();
    if (succeeded(index->build(function, input))) { placements = std::move(index); }
    auto controls = std::make_shared<SmallVector<Value>>();
    function.walk<WalkOrder::PreOrder>([&](Operation* operation) {
        signatures.try_emplace(operation);
        if (operation != function && operation->getNumRegions()) {
            signatures[operation->getParentOp()].regionChildren.insert(operation);
            for (auto* owner = operation; owner; owner = owner->getParentOp()) {
                signatures[owner].control = true;
                if (!isa<scf::IfOp>(operation)) { signatures[owner].loopFree = false; }
                if (owner == function) { break; }
            }
        }
        if (isa<scf::IfOp, scf::ForOp, scf::WhileOp>(operation)) {
            llvm::append_range(*controls, operation->getOperands());
        }
    });
    context->controlValues = controls;
    admitted = context;
    regionalContexts[function] = context;
    for (auto [id, site] : llvm::enumerate(trace.sites())) {
        Operation* child = nullptr;
        for (Operation* owner = site.anchor; owner; owner = owner->getParentOp()) {
            auto& signature = signatures[owner];
            signature.sites.push_back(id);
            if (child) {
                ++membershipQueries;
                if (signature.childMembership.insert(child).second) { signature.children.push_back(child); }
            }
            if (owner == function) { break; }
            child = owner;
        }
    }
}
RegionalContextHandle RegionalRequests::contextFor(Operation* owner)
{
    if (!owner || !signatures.count(owner)) { return {}; }
    auto found = regionalContexts.find(owner);
    if (found != regionalContexts.end()) { return found->second; }
    auto context = std::shared_ptr<RegionalContext>(new RegionalContext(*admitted));
    context->scope = owner;
    for (auto* ancestor = owner; ancestor != function; ancestor = ancestor->getParentOp()) {
        context->entryRegions.push_back(ancestor->getParentRegion());
    }
    regionalContexts[owner] = context;
    return context;
}
void RegionalRequests::record(Operation* owner, StringRef route, StringRef outcome, StringRef reason)
{
    history.push_back({owner == function ? &function.getBody() : owner->getParentRegion(),
                       route.str(), outcome.str(), reason.str(),
                       costs.active() && !attemptStarts.empty() ?
                           std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - attemptStarts.back()).count() : 0});
}
RegionalRequests::Candidate RegionalRequests::explicitRegion(Operation* owner, const Signature& signature)
{
    if (signature.control) {
        return {{}, RegionalStatus::NotApplicable, "region is not a resolved straight-line occurrence sequence"};
    }
    auto result = std::make_shared<SelectedAnalysis>();
    result->route = "regional-explicit";
    result->sites = signature.sites;
    SmallVector<const CompoundInstanceElement*> phases;
    for (auto id : signature.sites) { phases.push_back(trace.sites()[id].phase); }
    std::unique_ptr<StorageAnalysis> storage;
    {
        CostScope effect(costs, CostStage::Effects);
        storage = std::make_unique<StorageAnalysis>(input, phases);
    }
    CostScope backend(costs, CostStage::Backend);
    LifetimeAnalysis lifetimes;
    if (failed(lifetimes.build(phases, *storage))) {
        return {{}, RegionalStatus::UnmetObligation, "regional shared lifetime obligation"};
    }
    llvm::append_range(result->generators, lifetimes.generators());
    for (const auto& baseline : input.target().finiteDrains()) {
        auto source = llvm::find(phases, baseline.previous), target = llvm::find(phases, baseline.consumer);
        if (source == phases.end() || target == phases.end()) { continue; }
        const auto a = static_cast<std::size_t>(source - phases.begin());
        const auto b = static_cast<std::size_t>(target - phases.begin());
        auto existing = llvm::find_if(result->generators, [&](const Demand& edge) {
            return edge.source == a && edge.consumer == b;
        });
        if (existing == result->generators.end()) {
            result->generators.push_back({a, b, {}, baseline.barrier});
        } else { existing->originalBarrier = baseline.barrier; }
    }
    if (failed(result->explicitReduction.build(phases, result->generators))) {
        return {{}, RegionalStatus::UnmetObligation, "regional shared baseline/reduction obligation"};
    }
    result->contract.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
                                  interfaceBit(DemandInterface::CompletionFrontiers);
    return {result, RegionalStatus::Ready, {}};
}
FailureOr<SmallVector<const CompoundInstanceElement*>> RegionalRequests::fixedLoopPhases(
    scf::ForOp loop, const Signature& signature, std::string& reason)
{
    APInt step;
    if (!matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isStrictlyPositive()) {
        reason = "positive constant progression not established"; return failure();
    }
    for (auto& operation : *loop.getBody()) {
        if (operation.getNumRegions()) {
            reason = "whole-loop fixed body has nested control"; return failure();
        }
    }
    SmallVector<const CompoundInstanceElement*> phases;
    for (auto id : signature.sites) {
        auto* phase = trace.sites()[id].phase;
        // A fixed shared record must not depend on an iteration-local SSA view.
        for (auto memories : {ArrayRef<const BaseMemInfo*>(phase->useVec),
                              ArrayRef<const BaseMemInfo*>(phase->defVec)}) {
            for (const auto* memory : memories) {
                auto* definition = memory->baseBuffer ? memory->baseBuffer.getDefiningOp() : nullptr;
                if (!memory->baseBuffer || !definition || loop->isProperAncestor(definition)) {
                    reason = "iteration-dependent storage lacks stationary qualification"; return failure();
                }
            }
        }
        phases.push_back(phase);
    }
    return phases;
}
RegionalRequests::Candidate RegionalRequests::boundaryLoopRegion(
    Operation* owner, const Signature& signature, const AnalysisNeeds& needs)
{
    const auto& original = input.target().originalBarrierChain();
    auto* repeatedOwner = original.frames.empty() ? original.loop : original.frames.front();
    const bool loopRegion = isa<scf::ForOp>(owner) &&
        ((original.barrier && owner == repeatedOwner) ||
         (needs.interfaces & interfaceBit(DemandInterface::RegionBoundaryPorts)));
    if ((!loopRegion && owner != function) || signature.sites.size() < (loopRegion ? 1 : 3)) {
        return {{}, RegionalStatus::NotApplicable, "not a compact loop or invocation boundary sequence"};
    }
    AnalysisContract contract;
    contract.interfaces = interfaceBit(DemandInterface::MinimumRepresentation);
    if (loopRegion) { contract.interfaces |= interfaceBit(DemandInterface::RegionBoundaryPorts); }
    std::string reason;
    if (!contract.accepts(needs, reason)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    FailureOr<std::shared_ptr<const SingleStreamLoop>> imported = std::shared_ptr<const SingleStreamLoop>{};
    SelectedAnalysisHandle child;
    SmallVector<SelectedAnalysisHandle> sequenceChildren;
    if (loopRegion) {
        imported = original.barrier && owner == repeatedOwner ?
            SingleStreamLoop::buildRepeatedRegion(function, input, trace, costs, reason) :
            SingleStreamLoop::buildBodyRegion(function, cast<scf::ForOp>(owner), signature.sites,
                                               input, trace, costs, reason);
    } else if (original.barrier) {
        auto ports = needs;
        ports.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
                           interfaceBit(DemandInterface::RegionBoundaryPorts);
        auto queried = request(repeatedOwner, contextFor(repeatedOwner), RegionalMode::Modeled, ports,
                                   RegionalRepresentation::NativeSummaries, RegionalPreparation::Mathematical,
                                   RegionalRoutePolicy::Economical);
        if (!queried.analysis || !queried.analysis->boundaryLoop) {
            return {{}, RegionalStatus::UnmetObligation, queried.obligation};
        }
        child = queried.analysis;
        imported = SingleStreamLoop::composeRepeatedRegion(function, input, trace, child->boundaryLoop, costs, reason);
    } else {
        SmallVector<scf::ForOp> loops;
        function.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) { loops.push_back(loop); });
        if (loops.size() == 2 && !loops[0]->isAncestor(loops[1]) && !loops[1]->isAncestor(loops[0])) {
            SmallVector<std::shared_ptr<const SingleStreamLoop>> bodies;
            for (auto loop : loops) {
                auto ports = needs;
                ports.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
                                   interfaceBit(DemandInterface::RegionBoundaryPorts);
                auto queried = request(loop, contextFor(loop), RegionalMode::Modeled, ports,
                    RegionalRepresentation::NativeSummaries, RegionalPreparation::Mathematical,
                    RegionalRoutePolicy::Economical);
                if (!queried.analysis || !queried.analysis->boundaryLoop) {
                    return {{}, RegionalStatus::UnmetObligation, queried.obligation};
                }
                bodies.push_back(queried.analysis->boundaryLoop);
                sequenceChildren.push_back(queried.analysis);
            }
            const bool exclusive = bodies[0]->participation && bodies[0]->participation == bodies[1]->participation &&
                                   bodies[0]->activationOutcome != bodies[1]->activationOutcome;
            imported = exclusive ? SingleStreamLoop::composeExclusiveRegions(
                                       function, input, trace, bodies, costs, reason) :
                                   SingleStreamLoop::composeSequentialRegions(
                                       function, input, trace, bodies, costs, reason);
        } else if (loops.size() > 1 && loops.front()->isAncestor(loops.back())) {
            auto ports = needs;
            ports.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
                               interfaceBit(DemandInterface::RegionBoundaryPorts);
            auto queried = request(loops.front(), contextFor(loops.front()), RegionalMode::Modeled, ports,
                RegionalRepresentation::NativeSummaries, RegionalPreparation::Mathematical,
                RegionalRoutePolicy::Economical);
            if (!queried.analysis || !queried.analysis->boundaryLoop) {
                return {{}, RegionalStatus::UnmetObligation, queried.obligation};
            }
            child = queried.analysis;
            imported = SingleStreamLoop::build(function, input, trace, costs, reason, child->boundaryLoop);
        } else { imported = SingleStreamLoop::build(function, input, trace, costs, reason); }
    }
    if (failed(imported) || !*imported) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto selected = std::make_shared<SelectedAnalysis>();
    selected->kind = SelectedAnalysis::Kind::BoundaryLoop;
    selected->route = loopRegion ? "qualified-loop-bound-ports" :
        ((*imported)->protocol == SingleStreamProtocol::ExclusiveArmBoundaries ? "exclusive-arm-port-composition" :
        ((*imported)->protocol == SingleStreamProtocol::SequentialBoundaries ? "sequential-loop-port-composition" :
        ((*imported)->repeatedPair() ? "original-barrier-compact-boundary-composition" :
                                    "correlated-single-stream-boundaries")));
    selected->contract = contract;
    selected->sites = signature.sites;
    selected->boundaryLoop = *imported;
    if (child) { selected->regionalChildren.push_back(std::move(child)); }
    llvm::append_range(selected->regionalChildren, sequenceChildren);
    return {selected, RegionalStatus::Ready, {}};
}
RegionalRequests::Candidate RegionalRequests::stationaryRegion(Operation* owner, const Signature& signature)
{
    auto loop = dyn_cast<scf::ForOp>(owner);
    if (!loop) { return {{}, RegionalStatus::NotApplicable, "region is not a counted loop"}; }
    std::string reason;
    auto phases = fixedLoopPhases(loop, signature, reason);
    if (failed(phases)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto imported = StationaryCellInput::build(input, *phases, costs, reason);
    if (failed(imported)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto result = std::make_shared<SelectedAnalysis>();
    result->kind = SelectedAnalysis::Kind::Periodic;
    result->route = "regional-stationary-cells";
    result->loop = loop;
    result->sites = signature.sites;
    result->stationary = *imported;
    {
        CostScope backend(costs, CostStage::Backend);
        if (failed(result->periodic.build(result->stationary->storage()))) {
            return {{}, RegionalStatus::UnmetObligation, "stationary cell numerical quotient obligation"};
        }
    }
    result->contract.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
                                  interfaceBit(DemandInterface::CompletionFrontiers) |
                                  interfaceBit(DemandInterface::PeriodicThresholds);
    return {result, RegionalStatus::Ready, {}};
}
RegionalRequests::Candidate RegionalRequests::countedRegion(Operation* owner, const Signature& signature)
{
    auto loop = dyn_cast<scf::ForOp>(owner);
    if (!loop || signature.sites.size() != 2) {
        return {{}, RegionalStatus::NotApplicable, "region is not a two-site counted reader nest"};
    }
    std::string reason;
    auto primitives = arithmeticPrimitives(owner, ArithmeticClass::Octagons, reason);
    if (failed(primitives)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto result = std::make_shared<SelectedAnalysis>();
    result->loop = loop; result->sites = signature.sites;
    result->structured = imported.find(owner)->second.first;
    if (failed(buildCountedReaderQueries(*result, *primitives, input, costs, reason))) {
        return {{}, RegionalStatus::UnmetObligation, reason};
    }
    return {result, RegionalStatus::Ready, {}};
}
RegionalRequests::Candidate RegionalRequests::generalCountedRegion(Operation* owner, const Signature& signature)
{
    auto loop = dyn_cast<scf::ForOp>(owner);
    if (!loop || signature.sites.size() != 2) {
        return {{}, RegionalStatus::NotApplicable, "region is not a two-site counted reader nest"};
    }
    std::string reason;
    auto source = arithmeticInput(owner, reason);
    if (failed(source)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto result = std::make_shared<SelectedAnalysis>();
    result->loop = loop; result->sites = signature.sites; result->structured = *source;
    if (failed(buildGeneralCounted(*result, input, costs, reason))) {
        return {{}, RegionalStatus::UnmetObligation, reason};
    }
    return {result, RegionalStatus::Ready, {}};
}
RegionalRequests::Candidate RegionalRequests::upperFixedBodyRegion(
    Operation* owner, const Signature& signature, const AnalysisNeeds& needs)
{
    auto loop = dyn_cast<scf::ForOp>(owner);
    if (!loop) { return {{}, RegionalStatus::NotApplicable, "upper fixed body requires a counted loop"}; }
    std::string reason;
    auto contract = FixedBodyUpper::contract();
    if (!contract.accepts(needs, reason)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    APInt step;
    if (!matchPattern(loop.getStep(), m_ConstantInt(&step)) || !step.isOne()) {
        return {{}, RegionalStatus::UnmetObligation, "upper fixed-body query adapter requires unit progression"};
    }
    for (auto& operation : *loop.getBody()) {
        if (operation.getNumRegions()) {
            return {{}, RegionalStatus::NotApplicable, "upper fixed body contains variable nested participation"};
        }
    }
    SmallVector<const CompoundInstanceElement*> phases;
    for (auto id : signature.sites) {
        auto* phase = trace.sites()[id].phase;
        if (phase->elementOp->getBlock() != loop.getBody()) {
            return {{}, RegionalStatus::UnmetObligation, "upper origin is not in the original fixed body"};
        }
        phases.push_back(phase);
    }
    auto primitives = arithmeticPrimitives(owner, ArithmeticClass::Octagons, reason);
    if (failed(primitives)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto source = imported.find(owner)->second.first;
    if (!source->relations().extras.relation().isIntegerEmpty()) {
        return {{}, RegionalStatus::UnmetObligation, "upper fixed-body extra prerequisite distance is not supplied"};
    }
    auto upper = FixedBodyUpper::build(input, phases, costs, reason);
    if (failed(upper)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto result = std::make_shared<SelectedAnalysis>();
    result->kind = SelectedAnalysis::Kind::Periodic;
    result->route = "regional-fixed-body-upper";
    result->loop = loop; result->sites = signature.sites; result->structured = source;
    result->upper = *upper; result->periodic = (*upper)->selected();
    result->contract = contract;
    if (failed(liftPeriodicQueries(*result, *primitives, reason))) {
        return {{}, RegionalStatus::UnmetObligation, reason};
    }
    return {result, RegionalStatus::Ready, {}};
}
RegionalRequests::Candidate RegionalRequests::periodicRegion(Operation* owner, const Signature& signature)
{
    auto loop = dyn_cast<scf::ForOp>(owner);
    if (!loop) { return {{}, RegionalStatus::NotApplicable, "region is not a whole counted loop"}; }
    std::string reason;
    auto qualified = fixedLoopPhases(loop, signature, reason);
    if (failed(qualified)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto& phases = *qualified;
    CostScope backend(costs, CostStage::Backend);
    SmallVector<PeriodicPrerequisite> generators;
    for (auto [b, target] : llvm::enumerate(phases)) {
        for (auto [a, source] : llvm::enumerate(phases)) {
            DepBaseMemInfoPairVec witnesses;
            if (input.memory().DepBetween(source->defVec, target->useVec, witnesses) ||
                input.memory().DepBetween(source->useVec, target->defVec, witnesses) ||
                input.memory().DepBetween(source->defVec, target->defVec, witnesses)) {
                generators.push_back({a, b, llvm::DynamicAPInt(a < b ? 0 : 1)});
            }
        }
    }
    auto result = std::make_shared<SelectedAnalysis>();
    result->kind = SelectedAnalysis::Kind::Periodic;
    result->route = "regional-periodic-quotient";
    result->loop = loop;
    result->sites = signature.sites;
    if (failed(result->periodic.build(phases, generators))) {
        return {{}, RegionalStatus::UnmetObligation, "whole-loop numerical quotient obligation"};
    }
    result->contract.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
                                  interfaceBit(DemandInterface::CompletionFrontiers) |
                                  interfaceBit(DemandInterface::PeriodicThresholds);
    return {result, RegionalStatus::Ready, {}};
}
FailureOr<StructuredInputHandle> RegionalRequests::arithmeticInput(Operation* owner, std::string& reason)
{
    auto found = imported.find(owner);
    if (found != imported.end()) {
        reason = found->second.second;
        if (!found->second.first) { return failure(); }
        return found->second.first;
    }
    CostScope effect(costs, CostStage::Effects);
    if (!importer) { importer = std::make_unique<EndpointRelationImporter>(function, input, trace, placements); }
    auto result = importer->build(owner, reason);
    imported.emplace(owner, std::make_pair(succeeded(result) ? *result : StructuredInputHandle{}, reason));
    return result;
}
FailureOr<SignedInputs> RegionalRequests::arithmeticPrimitives(
    Operation* owner, ArithmeticClass arithmetic, std::string& reason)
{
    auto key = std::make_pair(owner, arithmetic);
    auto found = primitives.find(key);
    if (found != primitives.end()) {
        reason = found->second.second;
        if (!found->second.first) { return failure(); }
        return *found->second.first;
    }
    auto source = arithmeticInput(owner, reason);
    if (failed(source)) {
        primitives.emplace(key, std::make_pair(std::optional<SignedInputs>{}, reason));
        return failure();
    }
    CostScope backend(costs, CostStage::Backend);
    if (!arithmeticSpace) {
        auto created = SignedSpace::create((*source)->schema(), llvm::DynamicAPInt(1));
        if (!created.succeeded()) { reason = signedDiagnostic(created.status).str(); return failure(); }
        arithmeticSpace = created.value;
    }
    auto result = specializePrimitives(*source, arithmetic, reason, arithmeticSpace);
    primitives.emplace(key, std::make_pair(succeeded(result) ? std::optional<SignedInputs>(*result) :
                                                               std::optional<SignedInputs>{}, reason));
    return result;
}
RegionalRequests::Candidate RegionalRequests::arithmeticRegion(
    Operation* owner, const Signature& signature, ArithmeticClass arithmetic)
{
    std::string reason;
    auto inputs = arithmeticPrimitives(owner, arithmetic, reason);
    if (failed(inputs)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    CostScope backend(costs, CostStage::Backend);
    auto analysis = SignedDemandAnalysis::build(inputs->context->space(), *inputs);
    if (!analysis.succeeded()) {
        return {{}, RegionalStatus::UnmetObligation, signedDiagnostic(analysis.status).str()};
    }
    auto result = std::make_shared<SelectedAnalysis>();
    result->kind = SelectedAnalysis::Kind::Signed;
    result->route = arithmetic == ArithmeticClass::Differences ? "regional-difference-bounds" :
                                                               "regional-integer-octagons";
    result->sites = signature.sites;
    result->structured = imported.find(owner)->second.first;
    result->signedAnalysis = analysis.value;
    result->context = analysis.value->context();
    result->minimum = analysis.value->minimum();
    result->native = analysis.value->native();
    result->reachability = analysis.value->reachability();
    result->contract.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
        interfaceBit(DemandInterface::RegionQueries) | interfaceBit(DemandInterface::UniformMembership);
    return {result, RegionalStatus::Ready, {}};
}
RegionalRequests::Candidate RegionalRequests::composeRegion(
    Operation* owner, const Signature& signature, RegionalMode mode, const AnalysisNeeds& needs,
    RegionalPreparation preparation, RegionalRoutePolicy policy)
{
    if (owner != function || signature.children.size() < 2) {
        return {{}, RegionalStatus::NotApplicable, "no proper function sequence merge"};
    }
    auto queries = needs;
    queries.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
                         interfaceBit(DemandInterface::RegionQueries);
    // Child requests finish before parent-wide arithmetic qualification/import.
    SmallVector<SelectedAnalysisHandle> children;
    for (Operation* child : signature.children) {
        auto built = request(child, contextFor(child), mode, queries, RegionalRepresentation::ArithmeticRelations,
                                 preparation, policy);
        if (!built.analysis) { return {{}, RegionalStatus::UnmetObligation, built.obligation}; }
        children.push_back(built.analysis);
    }
    std::string reason;
    auto inputs = arithmeticPrimitives(owner, ArithmeticClass::Octagons, reason);
    if (failed(inputs)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto merged = composeEndpointRegions(function, imported.find(owner)->second.first, *inputs, history, reason,
                                         costs, this, needs,
                                         preparation == RegionalPreparation::SelectorMatching, policy);
    if (failed(merged)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto result = std::make_shared<SelectedAnalysis>(**merged);
    result->sites = signature.sites;
    result->regionalChildren = std::move(children);
    result->contract.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
        interfaceBit(DemandInterface::RegionQueries) | interfaceBit(DemandInterface::UniformMembership);
    return {result, RegionalStatus::Ready, {}};
}
RegionalRequests::Candidate RegionalRequests::generalComposeRegion(
    Operation* owner, const Signature& signature, RegionalMode mode, const AnalysisNeeds& needs,
    RegionalPreparation preparation, RegionalRoutePolicy policy)
{
    if (owner != function || signature.children.size() < 2) {
        return {{}, RegionalStatus::NotApplicable, "no once-executed function sequence merge"};
    }
    std::string admission;
    if (!generalAnalysisContract().accepts(needs, admission)) {
        return {{}, RegionalStatus::UnmetObligation, admission};
    }
    auto queries = needs;
    queries.interfaces = interfaceBit(DemandInterface::MinimumRepresentation) |
                         interfaceBit(DemandInterface::RegionQueries);
    SmallVector<SelectedAnalysisHandle> children;
    for (auto* child : signature.children) {
        auto built = request(child, contextFor(child), mode, queries, RegionalRepresentation::PresburgerRelations,
                                 preparation, policy);
        if (!built.analysis) { return {{}, RegionalStatus::UnmetObligation, built.obligation}; }
        children.push_back(built.analysis);
    }
    std::string reason;
    auto source = arithmeticInput(owner, reason);
    if (failed(source)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    auto merged = composeGeneralRegions(function, *source, children, costs, reason);
    if (failed(merged)) { return {{}, RegionalStatus::UnmetObligation, reason}; }
    return {std::make_shared<SelectedAnalysis>(**merged), RegionalStatus::Ready, {}};
}
RegionalRequests::Candidate RegionalRequests::guardedRegion(Operation* owner, const Signature& signature)
{
    if (!signature.loopFree) {
        return {{}, RegionalStatus::NotApplicable, "potential occurrences are not loop-free"};
    }
    for (Region* region : contextFor(owner)->entryRegions) {
        if (!isa<func::FuncOp, scf::IfOp>(region->getParentOp())) {
            return {{}, RegionalStatus::NotApplicable, "finite occurrences have enclosing repetition/control"};
        }
    }
    if (!placements) { return {{}, RegionalStatus::UnmetObligation, "shared availability index not established"}; }
    SmallVector<const CompoundInstanceElement*> phases;
    for (auto id : signature.sites) { phases.push_back(trace.sites()[id].phase); }
    std::unique_ptr<StorageAnalysis> storage;
    {
        CostScope effects(costs, CostStage::Effects);
        storage = std::make_unique<StorageAnalysis>(input, phases);
    }
    auto result = std::make_shared<SelectedAnalysis>();
    CostScope backend(costs, CostStage::Backend);
    if (failed(result->guarded.build(owner, *placements, phases, *storage))) {
        return {{}, RegionalStatus::UnmetObligation, "finite guard extraction failed"};
    }
    result->kind = SelectedAnalysis::Kind::Guarded;
    result->route = "finite-guarded";
    result->sites = signature.sites;
    result->contract.interfaces = interfaceBit(DemandInterface::MinimumRepresentation);
    return {result, RegionalStatus::Ready, {}};
}
LogicalResult RegionalRequests::qualifyMatching(Operation* owner, SelectedAnalysis& selected, std::string& reason)
{
    if (!placements) { reason = "shared endpoint availability index not established"; return failure(); }
    if (selected.kind == SelectedAnalysis::Kind::BoundaryLoop) {
        // Source/consumer and first/last participation were established from
        // original SCF and shared storage by the structural producer.
        return success();
    }
    if (selected.kind == SelectedAnalysis::Kind::Periodic) {
        auto phases = selected.periodic.sites();
        DenseMap<PipelineType, std::size_t> last;
        for (auto [id, phase] : llvm::enumerate(phases)) { last[phase->kPipeValue] = id; }
        SmallVector<std::size_t> previous(phases.size());
        for (auto [id, phase] : llvm::enumerate(phases)) {
            previous[id] = last.lookup(phase->kPipeValue); last[phase->kPipeValue] = id;
        }
        for (auto id : selected.periodic.retained()) {
            const auto& edge = selected.periodic.generators()[id].edge;
            if (phases[edge.source]->kPipeValue != phases[edge.consumer]->kPipeValue) { continue; }
            auto adjacent = previous[edge.consumer];
            if (edge.source != adjacent || edge.distance != llvm::DynamicAPInt(adjacent < edge.consumer ? 0 : 1)) {
                reason = "unmet local-adjacency premise for exact direct construction"; return failure();
            }
        }
        if (failed(arithmeticInput(selected.loop, reason))) { return failure(); }
        if (periodicEndpointWidth(selected.loop) > IntegerType::kMaxWidth) {
            reason = "periodic endpoint arithmetic not representable"; return failure();
        }
        selected.structured = imported.find(selected.loop)->second.first;
        return success();
    }
    CostScope selector(costs, CostStage::Selectors);
    if (selected.kind == SelectedAnalysis::Kind::Signed) {
        auto schema = selected.minimum->space()->schema();
        DenseSet<std::size_t> sources, consumers;
        for (const auto& piece : selected.minimum->pieces()) {
            if (piece.domain.site && piece.range.site &&
                schema->sites()[*piece.domain.site].phase->kPipeValue ==
                schema->sites()[*piece.range.site].phase->kPipeValue) {
                sources.insert(*piece.domain.site); consumers.insert(*piece.range.site);
            }
        }
        if (sources.empty()) { return prepareEndpoints(selected, reason); }
        auto inputs = arithmeticPrimitives(owner, ArithmeticClass::Octagons, reason);
        if (failed(inputs)) { return failure(); }
        auto admitted = inputs->reference->restrictContext(inputs->context);
        if (admitted.succeeded()) { admitted = admitted.value->restrictDomain(inputs->present); }
        if (admitted.succeeded()) { admitted = admitted.value->restrictRange(inputs->present); }
        if (!admitted.succeeded()) { reason = "local adjacency context query is not represented"; return failure(); }
        SmallVector<SignedPiece> before, after;
        for (const auto& piece : admitted.value->pieces()) {
            if (piece.domain.site && piece.range.site &&
                schema->sites()[*piece.domain.site].phase->kPipeValue ==
                schema->sites()[*piece.range.site].phase->kPipeValue) {
                // Only endpoints are restricted; every possible intermediate
                // same-pipe occurrence remains available to the witness query.
                if (sources.contains(*piece.domain.site)) { before.push_back(piece); }
                if (consumers.contains(*piece.range.site)) { after.push_back(piece); }
            }
        }
        auto first = SignedRelation::import(selected.minimum->space(), SymbolicTuple::Occurrence,
                                            SymbolicTuple::Occurrence, before);
        auto second = SignedRelation::import(selected.minimum->space(), SymbolicTuple::Occurrence,
                                             SymbolicTuple::Occurrence, after);
        if (!first.succeeded() || !second.succeeded()) {
            reason = "local adjacency reference import is not represented"; return failure();
        }
        auto between = first.value->compose(second.value);
        if (!between.succeeded()) { reason = "local adjacency query is not represented"; return failure(); }
        SmallVector<SignedPiece> pieces;
        for (auto piece : between.value->pieces()) {
            piece.domain.kind = PeriodicEventKind::Completion;
            piece.range.kind = PeriodicEventKind::Start;
            pieces.push_back(std::move(piece));
        }
        auto nonadjacent = SignedRelation::import(selected.minimum->space(), SymbolicTuple::Event,
                                                  SymbolicTuple::Event, pieces);
        auto retained = nonadjacent.succeeded() ? selected.minimum->intersect(nonadjacent.value) : nonadjacent;
        if (!retained.succeeded() || !retained.value->empty()) {
            reason = "unmet local-adjacency premise for exact direct construction"; return failure();
        }
        return prepareEndpoints(selected, reason);
    }
    if (selected.kind == SelectedAnalysis::Kind::General) { return prepareGeneralEndpoints(selected, reason); }
    if (selected.kind == SelectedAnalysis::Kind::Explicit) {
        for (auto id : selected.explicitReduction.retained()) {
            const auto& edge = selected.generators[id];
            auto pipe = trace.sites()[selected.sites[edge.source]].phase->kPipeValue;
            if (pipe != trace.sites()[selected.sites[edge.consumer]].phase->kPipeValue) { continue; }
            for (auto between = edge.source + 1; between < edge.consumer; ++between) {
                if (trace.sites()[selected.sites[between]].phase->kPipeValue == pipe) {
                    reason = "unmet local-adjacency premise for exact direct construction"; return failure();
                }
            }
        }
        return success();
    }
    for (const auto& local : selected.guarded.localDemands()) {
        // Only an arena-proven false nonadjacency permits direct construction.
        // Unknown Boolean feasibility is an unmet premise, not an extra barrier.
        if (local.nonadjacent != 0) {
            reason = "unmet local-adjacency premise for exact direct construction"; return failure();
        }
    }
    auto nodes = selected.guarded.predicates().nodes();
    for (const auto& edge : selected.guarded.retained()) {
        SmallVector<Predicate> pending{edge.predicate};
        DenseSet<Predicate> visited;
        while (!pending.empty()) {
            auto id = pending.pop_back_val();
            if (!visited.insert(id).second) { continue; }
            const auto& node = nodes[id];
            auto* source = selected.guarded.phases()[edge.source];
            auto* target = selected.guarded.phases()[edge.consumer];
            bool local = source->kPipeValue == target->kPipeValue;
            if (node.kind == PredicateKind::Atom &&
                ((!local && !placements->valueAvailable(node.condition, source->elementOp, Boundary::After)) ||
                 !placements->valueAvailable(node.condition, target->elementOp, Boundary::Before))) {
                reason = "endpoint predicate uses an unavailable outcome"; return failure();
            }
            if (node.kind == PredicateKind::Not || node.kind == PredicateKind::And || node.kind == PredicateKind::Or) {
                pending.push_back(node.first);
            }
            if (node.kind == PredicateKind::And || node.kind == PredicateKind::Or) { pending.push_back(node.second); }
        }
    }
    return success();
}
RegionalRequestResult RegionalRequests::request(
    Operation* owner, RegionalContextHandle context, RegionalMode mode, const AnalysisNeeds& needs,
    RegionalRepresentation representation, RegionalPreparation preparation, RegionalRoutePolicy policy)
{
    if (!owner || !context || context->root != function ||
        !signatures.count(owner) ||
        regionalContexts.lookup(context->scope) != context ||
        (context->scope != owner && !context->scope->isProperAncestor(owner))) {
        return {{}, "unmet-obligation", "regional input/context identity mismatch", {}};
    }
    Key key{owner, context.get(), mode, representation, preparation, policy, needs.reduction,
            needs.suppliedExactEffects, needs.allowSoundUpper, needs.interfaces};
    if (auto found = cache.find(key); found != cache.end()) {
        ++hits; return found->second;
    }
    CostScope recognition(costs, CostStage::Recognition);
    attemptStarts.push_back(costs.active() ? std::chrono::steady_clock::now() :
                                           std::chrono::steady_clock::time_point{});
    auto restore = llvm::make_scope_exit([&]() { attemptStarts.pop_back(); });
    contexts.push_back(context);
    if (owner->getNumRegions()) {
        for (auto& region : owner->getRegions()) { visitedRegions.insert(&region); }
    } else {
        visitedRegions.insert(owner->getParentRegion());
    }
    const auto& signature = signatures.find(owner)->second;
    RegionalRequestResult result;
    auto requested = needs;
    if (mode == RegionalMode::MinimumExact) { requested = AnalysisNeeds::minimumExact();
        requested.interfaces = needs.interfaces; }
    auto attempt = [&](StringRef route, Candidate candidate) {
        if (!candidate.analysis) {
            bool unmet = candidate.status == RegionalStatus::UnmetObligation;
            record(owner, route, unmet ? "unmet-obligation" : "not-applicable", candidate.obligation);
            if (unmet) { result.obligation = candidate.obligation; }
            return false;
        }
        std::string reason;
        if (!candidate.analysis->contract.accepts(requested, reason)) {
            record(owner, route, "unmet-obligation", reason); result.obligation = reason; return false;
        }
        if (preparation == RegionalPreparation::SelectorMatching &&
            failed(qualifyMatching(owner, *candidate.analysis, reason))) {
            record(owner, route, "unmet-obligation", reason); result.obligation = reason; return false;
        }
        candidate.analysis->requestContext = context;
        candidate.analysis->regionalContext = contextFor(owner);
        result.analysis = candidate.analysis;
        result.outcome = "ready";
        result.obligation.clear();
        record(owner, route, "ready", ""); return true;
    };
    if (representation == RegionalRepresentation::NativeSummaries) {
        auto explicitResult = explicitRegion(owner, signature);
        bool accepted = attempt("regional-explicit", std::move(explicitResult));
        if (!accepted) {
            accepted = attempt("correlated-single-stream-boundaries", boundaryLoopRegion(owner, signature, requested));
        }
        if (!accepted) { accepted = attempt("regional-counted-readers", countedRegion(owner, signature)); }
        if (!accepted) {
            accepted = attempt("regional-affine-counted-readers", generalCountedRegion(owner, signature));
        }
        if (!accepted) { accepted = attempt("regional-stationary-cells", stationaryRegion(owner, signature)); }
        if (!accepted) { accepted = attempt("regional-periodic-quotient", periodicRegion(owner, signature)); }
        if (accepted) { cache.emplace(key, result); return result; }
    }
    if ((representation == RegionalRepresentation::ArithmeticRelations ||
         representation == RegionalRepresentation::PresburgerRelations) &&
        attempt("regional-counted-readers", countedRegion(owner, signature))) {
        cache.emplace(key, result); return result;
    }
    if (representation == RegionalRepresentation::PresburgerRelations &&
        attempt("regional-affine-counted-readers", generalCountedRegion(owner, signature))) {
        cache.emplace(key, result); return result;
    }
    if (representation == RegionalRepresentation::ArithmeticRelations) {
        auto candidate = stationaryRegion(owner, signature);
        if (!candidate.analysis) { candidate = periodicRegion(owner, signature); }
        if (candidate.analysis) {
            std::string reason;
            auto inputs = arithmeticPrimitives(owner, ArithmeticClass::Octagons, reason);
            auto lifted = [&]() {
                if (failed(inputs)) { return failure(); }
                CostScope backend(costs, CostStage::Backend);
                return liftPeriodicQueries(*candidate.analysis, *inputs, reason);
            }();
            if (failed(lifted)) {
                candidate = {{}, RegionalStatus::UnmetObligation, reason};
            } else {
                candidate.analysis->structured = imported.find(owner)->second.first;
            }
        }
        if (attempt("regional-periodic-symbolic-queries", std::move(candidate))) {
            cache.emplace(key, result); return result;
        }
    }
    // Descent always visits all proper children before parent arithmetic import.
    // A sibling obligation cannot suppress recognition of an independent loop.
    SelectedAnalysisHandle soleChild;
    for (Operation* child : signature.children) {
        if (!signatures.find(child)->second.sites.empty()) {
            auto childResult = request(child, contextFor(child), mode, needs, representation, preparation, policy);
            if (childResult.analysis) {
                result.children.push_back(childResult.analysis);
                if (signature.children.size() == 1) { soleChild = childResult.analysis; }
            }
            llvm::append_range(result.children, childResult.children);
        }
    }
    // A qualified sole child already supplies the entire payload interface.
    // Forward it before importing unrelated parent scalar operations; otherwise
    // a generic arithmetic route would hide the recognized compact fast path.
    if (owner == function && signature.children.size() == 1 && soleChild &&
        signature.regionChildren.size() == 1 && signature.regionChildren.contains(signature.children.front()) &&
        soleChild->sites == signature.sites) {
        result.analysis = soleChild; result.outcome = "ready";
        record(owner, "regional-wrapper", "ready", "");
        cache.emplace(key, result); return result;
    }
    if (representation == RegionalRepresentation::NativeSummaries) {
        if (attempt("regional-sequence-composition",
            composeRegion(owner, signature, mode, requested, preparation, policy))) {
            cache.emplace(key, result); return result;
        }
    }
    // An enabled precision extension can compose admitted General children.
    // Economical requests never force broader symbolic composition. Do not import expensive symbolic siblings after
    // a signed merge rejects an unrelated capability/matching obligation.
    bool generalChild = llvm::any_of(result.children, [](const auto& child) {
        return child->kind == SelectedAnalysis::Kind::General;
    });
    if ((representation == RegionalRepresentation::PresburgerRelations ||
         (representation == RegionalRepresentation::NativeSummaries && generalChild &&
          policy == RegionalRoutePolicy::GeneralExtension)) &&
        attempt("presburger-region-composition",
            generalComposeRegion(owner, signature, mode, requested, preparation, policy))) {
        cache.emplace(key, result); return result;
    }
    {
        if (attempt("regional-difference-bounds", arithmeticRegion(owner, signature, ArithmeticClass::Differences))) {
            cache.emplace(key, result); return result;
        }
        if (representation != RegionalRepresentation::DifferenceRelations &&
            attempt("regional-integer-octagons", arithmeticRegion(owner, signature, ArithmeticClass::Octagons))) {
            cache.emplace(key, result); return result;
        }
    }
    if (representation == RegionalRepresentation::NativeSummaries &&
        attempt("regional-finite-guarded", guardedRegion(owner, signature))) {
        cache.emplace(key, result); return result;
    }
    if (requested.allowSoundUpper && !requested.suppliedExactEffects &&
        representation != RegionalRepresentation::DifferenceRelations &&
        attempt("regional-fixed-body-upper", upperFixedBodyRegion(owner, signature, requested))) {
        cache.emplace(key, result); return result;
    }
    // Preserve the optional general fallback after the cheap candidates, but
    // do not repeat the earlier General-child composition in this request.
    if (representation == RegionalRepresentation::NativeSummaries && !generalChild &&
        policy == RegionalRoutePolicy::GeneralExtension) {
        auto priorObligation = result.obligation;
        if (attempt("presburger-region-composition",
            generalComposeRegion(owner, signature, mode, requested, preparation, policy))) {
            cache.emplace(key, result); return result;
        }
        // The optional precision attempt retains its own diagnostic in the
        // ledger, without hiding a concrete default matching failure.
        if (!priorObligation.empty()) { result.obligation = std::move(priorObligation); }
    }
    result.outcome = "unmet-obligation";
    if (result.obligation.empty()) {
        result.obligation = representation == RegionalRepresentation::ArithmeticRelations ?
            "regional arithmetic representation adapter not supplied" :
            "mixed regional boundary/query and endpoint adapter not supplied";
    }
    record(owner, "regional-composition", result.outcome, result.obligation);
    cache.emplace(key, result);
    return result;
}
} // namespace mlir::pto::frontiersynch
