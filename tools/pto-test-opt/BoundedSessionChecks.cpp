// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check owned exports independently of graph-oracle construction.
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetimeRegional.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::string render(Operation* operation)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    operation->print(stream);
    return text;
}
bool independentPlans(const fs::PreparedLogicalPlan& first, const fs::PreparedLogicalPlan& second)
{
    llvm::DenseSet<Operation*> owned;
    llvm::DenseSet<Block*> blocks;
    for (const auto& stage : first.preparation) {
        blocks.insert(stage.code.get());
        for (auto& operation : *stage.code) { owned.insert(&operation); }
    }
    auto borrowed = [&](Value value) { return value && owned.contains(value.getDefiningOp()); };
    for (const auto& stage : second.preparation) {
        if (blocks.contains(stage.code.get())) { return false; }
        for (auto& operation : *stage.code) {
            if (owned.contains(&operation) || llvm::any_of(operation.getOperands(), borrowed)) { return false; }
        }
    }
    for (const auto& endpoint : second.endpoints) {
        if (borrowed(endpoint.guard) || borrowed(endpoint.identity) ||
            llvm::any_of(endpoint.memberCoordinates, borrowed)) { return false; }
    }
    return true;
}
std::string renderPreparation(const fs::PreparedLogicalPlan& plan)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    for (const auto& stage : plan.preparation) { stage.code->print(stream); }
    return text;
}
bool checkRegionalAllocationFailure(func::FuncOp function, pto::GMAliasPolicy policy, std::size_t region)
{
    auto module = function->getParentOfType<ModuleOp>();
    if (!module) { return false; }
    OwningOpRef<ModuleOp> copy(module.clone());
    auto cloned = copy->lookupSymbol<func::FuncOp>(function.getSymName());
    fs::FrontierAnalysis session(cloned);
    if (failed(session.initialize(policy))) { return false; }
    auto demands = session.minimumDemands(region);
    auto plan = session.prepareLogical(demands);
    if (failed(plan) || (*plan)->mathematicalOwner != demands.mathematical) { return false; }
    const auto endpoints = (*plan)->endpoints.size();
    const auto work = session.constructionCounts().mathematicalAttempts;
    if (succeeded(session.attachAllocation(demands, **plan)) || (*plan)->allocationCertificate ||
        (*plan)->endpoints.size() != endpoints || session.minimumDemands(region).mathematical != demands.mathematical ||
        session.constructionCounts().mathematicalAttempts != work) { return false; }
    if (failed(fs::insertLogicalSynchronization(cloned, **plan)) || failed(verify(*copy))) { return false; }
    const auto before = render(cloned);
    std::string diagnostic;
    ScopedDiagnosticHandler handler(cloned.getContext(), [&](Diagnostic& error) {
        llvm::raw_string_ostream stream(diagnostic); error.print(stream); return success();
    });
    const auto allocated = fs::allocatePhysicalEventIds(cloned, {0, 1, 2, 3, 4, 5});
    if (succeeded(allocated) || render(cloned) != before ||
        diagnostic.find("allocation adapter not implemented yet") == std::string::npos) { return false; }
    llvm::outs() << "bounded regional allocation: unsupported, demands and logical IR retained\n";
    return true;
}
} // namespace

bool checkBoundedSession(func::FuncOp function, pto::GMAliasPolicy policy, std::size_t region, bool requireEndpoint)
{
    auto owner = std::make_unique<fs::FrontierAnalysis>(function);
    auto& session = *owner;
    if (failed(session.initialize(policy))) { return false; }
    auto demands = session.minimumDemands(region);
    if (!demands.mathematical || !demands.mathematical->boundedDemands ||
        session.constructionCounts().boundedQueryBuilds || session.constructionCounts().boundedSelectorBuilds) {
        return false;
    }
    fs::AnalysisNeeds needs; needs.queries = true;
    auto queries = session.minimumDemands(region, needs);
    if (!queries.available.queries || queries.available.selectors || !queries.regionalExports) { return false; }
    needs.selectors = true;
    auto selectors = session.minimumDemands(region, needs);
    const auto reductions = session.constructionCounts().mathematicalAttempts;
    auto retry = session.minimumDemands(region, needs);
    if (!selectors.available.selectors || retry.regionalExports != selectors.regionalExports ||
        retry.mathematical != demands.mathematical || session.constructionCounts().boundedQueryBuilds != 1 ||
        session.constructionCounts().boundedSelectorBuilds != 1 ||
        session.constructionCounts().mathematicalAttempts != reductions ||
        queries.regionalExports->capabilities.exactSelectors || session.constructionCounts().logicalPreparations) {
        llvm::errs() << "bounded retry exports=" << (retry.regionalExports == selectors.regionalExports)
                     << " attempts=" << session.constructionCounts().mathematicalAttempts << "," << reductions
                     << " query-selectors=" << queries.regionalExports->capabilities.exactSelectors
                     << " preparations=" << session.constructionCounts().logicalPreparations << "\n";
        return false;
    }
    if (requireEndpoint) {
        auto root = session.minimumDemands(0, needs);
        if (!root.mathematical || root.mathematical->boundedDemands != demands.mathematical->boundedDemands ||
            root.regionalExports != retry.regionalExports ||
            session.constructionCounts().boundedQueryBuilds != 1 ||
            session.constructionCounts().boundedSelectorBuilds != 1) {
            llvm::errs() << "bounded root export: backend="
                         << (root.mathematical ? root.mathematical->backend : "none")
                         << " same-demands=" << (root.mathematical &&
                             root.mathematical->boundedDemands == demands.mathematical->boundedDemands)
                         << " same-exports=" << (root.regionalExports == retry.regionalExports)
                         << " builds=" << session.constructionCounts().boundedQueryBuilds << ","
                         << session.constructionCounts().boundedSelectorBuilds << "\n";
            return false;
        }
    }
    const auto endpointWork = session.constructionCounts().mathematicalAttempts;
    const auto enclosing = session.result()->nodes[region].loops;
    if (!selectors.regionalExports->capabilities.endpointRecipes ||
        !selectors.regionalExports->prepareWithVisits) { return false; }
    auto firstPlan = selectors.regionalExports->prepareWithVisits(enclosing);
    auto secondPlan = selectors.regionalExports->prepareWithVisits(enclosing);
    auto wrongFrame = enclosing;
    wrongFrame.push_back(demands.mathematical->boundedDemands->loop);
    auto invalidPlan = selectors.regionalExports->prepareWithVisits(wrongFrame);
    const bool available = succeeded(firstPlan);
    if ((requireEndpoint && !available) || available != succeeded(secondPlan) || succeeded(invalidPlan) ||
        session.constructionCounts().mathematicalAttempts != endpointWork) { return false; }
    if (available) {
        if (firstPlan->get() == secondPlan->get() || (*firstPlan)->completeInvocation ||
            !(*firstPlan)->mathematicalOwner || !(*secondPlan)->mathematicalOwner ||
            (requireEndpoint && (*firstPlan)->endpoints.empty()) ||
            !independentPlans(**firstPlan, **secondPlan)) { return false; }
        const auto before = renderPreparation(**secondPlan);
        firstPlan->reset();
        if (renderPreparation(**secondPlan) != before) { return false; }
    } else {
        std::string diagnostic;
        auto unavailable = fs::prepareBoundedLifetimeLogicalResult(
            demands.mathematical->boundedDemands, diagnostic, false);
        if (succeeded(unavailable) ||
            diagnostic.find("bounded endpoint unavailable") == std::string::npos ||
            diagnostic.find("unavailable value outside deterministic scalar replay") == std::string::npos) {
            return false;
        }
        llvm::outs() << "bounded-session endpoint obligation: " << diagnostic << "\n";
    }
    std::weak_ptr<const pto::SyncInput> inputOwner = session.sharedInput();
    auto retained = queries.regionalExports;
    const auto sites = demands.mathematical->boundedDemands->anchors.size();
    demands = {}; queries = {}; selectors = {}; retry = {};
    session.invalidate();
    owner.reset();
    if (inputOwner.expired()) { return false; }
    auto& e = *retained->expressions;
    for (unsigned site = 0; site < sites; ++site) {
        auto present = fs::regionalPresence(*retained, {site, e.constant(0), fs::PeriodicEventKind::Start});
        if (!present || !e.constructionError().empty()) { return false; }
        for (auto [id, input] : e.referencedInputs(*present)) { (void)id; if (!input.getType()) { return false; } }
    }
    retained.reset();
    if (available) {
        if (inputOwner.expired()) { return false; }
        firstPlan->reset();
        secondPlan->reset();
    }
    if (!inputOwner.expired()) { return false; }
    if (requireEndpoint && !checkRegionalAllocationFailure(function, policy, region)) { return false; }
    llvm::outs() << "bounded-session: retained demands, separate cached exports, owned input after invalidation\n";
    return true;
}
