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
#include "../../lib/PTO/Transforms/FrontierSynch/WholeInvocationIdentity.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseSet.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
bool failBoundedSession(StringRef stage)
{
    llvm::errs() << "bounded-session failure stage=" << stage << "\n";
    return false;
}
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
bool primitiveShapeReuse(const fs::ProgramRecognition& program)
{
    fs::ArithmeticProgram form;
    form.sites.resize(2);
    form.primitives.period = 2; form.primitives.pipeCount = 2;
    form.recognition.observedDimensions = 2;
    form.recognition.arithmeticClass = fs::ArithmeticClass::Differences;
    auto rows = fs::IntegerSystem::create(1, {{{fs::BoundInteger(1)}, fs::BoundInteger(10)}});
    if (failed(rows)) { return failBoundedSession("primitive-system"); }
    auto owned = std::make_shared<const fs::IntegerSystem>(*rows);
    for (auto [kind, pieces] : {
            std::make_pair(fs::PrimitiveKind::Reads, 2U), std::make_pair(fs::PrimitiveKind::Writes, 3U),
            std::make_pair(fs::PrimitiveKind::Native, 1U), std::make_pair(fs::PrimitiveKind::Prerequisites, 1U)}) {
        fs::PrimitiveRelation relation;
        relation.kind = kind;
        relation.pieces.resize(pieces);
        for (auto& piece : relation.pieces) { piece.machineSystem = owned; }
        form.primitives.relations.push_back(std::move(relation));
    }
    fs::AnalysisSessionState session;
    // Recognition objects default to Applicable. Establish the missing-premise
    // fixture explicitly before checking that cold screening scans no rows.
    form.extraction.state = form.recognition.state = fs::RecognitionState::MissingPremise;
    // Missing qualification remains unknown and never scans the primitive rows.
    const auto cold = fs::previewRetainedBackendWork(session, program, 0, {},
        fs::AnalysisBackend::Arithmetic, {}, &form);
    if (cold.total || !session.primitiveShapes.empty() || session.primitiveShapeBuilds) {
        return failBoundedSession("primitive-cold-preview");
    }
    form.extraction.state = form.recognition.state = fs::RecognitionState::Applicable;
    for (unsigned request = 0; request < 64; ++request) {
        fs::AnalysisNeeds needs;
        needs.queries = request & 1; needs.selectors = request & 2; needs.synchronization = request & 4;
        const auto backend = request & 8 ? fs::AnalysisBackend::Arithmetic : fs::AnalysisBackend::ArithmeticPeriodic;
        auto preview = fs::previewRetainedBackendWork(session, program, 0, {}, backend, needs, &form);
        if (!preview.total || session.primitiveShapeBuilds != 1 || session.primitiveShapeRelationVisits != 4 ||
            session.primitiveShapePieceVisits != 7) { return failBoundedSession("primitive-shape-reuse"); }
    }
    const auto& shape = session.primitiveShapes.at(&form);
    // Independently: prerequisite + W*(R+W) = 1 + 3*(2+3), one native
    // piece, one row/piece, and three joined descriptions per generator.
    if (shape.pieces != fs::EstimatedCount{16} || shape.native != fs::EstimatedCount{1} ||
        shape.rows != fs::EstimatedCount{51} || shape.domains != fs::EstimatedCount{2} ||
        shape.dimensions != 2 || session.primitiveShapes.size() != 1) {
        return failBoundedSession("primitive-shape-counts");
    }
    const auto distinct = form;
    (void)fs::previewRetainedBackendWork(session, program, 0, {}, fs::AnalysisBackend::Arithmetic, {}, &distinct);
    if (session.primitiveShapeBuilds != 2 || session.primitiveShapes.size() != 2) {
        return failBoundedSession("primitive-distinct-owner");
    }
    // A replacement session models initialize/invalidate: even the same retained
    // form address has no previous construction counts or cached shape there.
    fs::AnalysisSessionState replacement;
    (void)fs::previewRetainedBackendWork(replacement, program, 0, {}, fs::AnalysisBackend::Arithmetic, {}, &form);
    if (replacement.primitiveShapeBuilds != 1 || replacement.primitiveShapeRelationVisits != 4 ||
        replacement.primitiveShapePieceVisits != 7 || replacement.primitiveShapes.size() != 1) {
        return failBoundedSession("primitive-new-session");
    }
    return true;
}
bool retainedWorkDependencies(const fs::ProgramRecognition& program, std::size_t region,
    const fs::AnalysisOutcome& demands, const fs::AnalysisOutcome& queries, const fs::AnalysisOutcome& selectors)
{
    if (!primitiveShapeReuse(program)) { return failBoundedSession("primitive-shape-controls"); }
    fs::AnalysisSessionState model;
    auto& child = model.loopAttempts[{region, fs::AnalysisBackend::BoundedLifetime}];
    child.produced = true; child.mathematical = demands.mathematical;
    auto identity = std::make_shared<fs::WholeInvocationIdentity>();
    identity->childRegion = region; identity->childBackend = fs::AnalysisBackend::BoundedLifetime;
    identity->child = demands.mathematical;
    auto wrapper = std::make_shared<fs::MathematicalResult>();
    wrapper->identityComposition = identity;
    auto& parent = model.attempts[0][fs::AnalysisBackend::Sequence];
    parent.produced = true; parent.mathematical = wrapper;
    const auto size = demands.mathematical->boundedDemands->expressions.size();
    fs::AnalysisNeeds selection; selection.selectors = true;
    for (unsigned repeat = 0; repeat < 64; ++repeat) {
        const auto weak = fs::previewRetainedBackendWork(model, program, 0, region,
            fs::AnalysisBackend::Sequence, {});
        const auto strong = fs::previewRetainedBackendWork(model, program, 0, region,
            fs::AnalysisBackend::Sequence, selection);
        if (weak.required[1] || weak.required[2] || weak.total != fs::EstimatedCount{0} ||
            !strong.required[1] || !strong.required[2] || strong.total ||
            strong.stages[1] != fs::RetainedStage::Missing || strong.stages[2] != fs::RetainedStage::Missing ||
            model.attempts.size() != 1 || model.loopAttempts.size() != 1 ||
            demands.mathematical->boundedDemands->expressions.size() != size) {
            return failBoundedSession("identity-dependency-preview");
        }
    }
    child.boundedQueriesAttempted = true; child.boundedQueries = queries.regionalExports;
    const auto queryReady = fs::previewRetainedBackendWork(model, program, 0, region,
        fs::AnalysisBackend::Sequence, selection);
    if (queryReady.work[1] != fs::EstimatedCount{0} || queryReady.work[2] || queryReady.total ||
        queryReady.stages[1] != fs::RetainedStage::Ready) { return failBoundedSession("query-ready-preview"); }
    child.boundedSelectorsAttempted = true; child.boundedSelectors = selectors.regionalExports;
    const auto allReady = fs::previewRetainedBackendWork(model, program, 0, region,
        fs::AnalysisBackend::Sequence, selection);
    if (allReady.total != fs::EstimatedCount{0} || allReady == queryReady) {
        return failBoundedSession("selector-ready-preview");
    }
    // A retained wrapper around a refused export is not evidence of capability.
    auto expanded = std::make_shared<fs::MathematicalResult>();
    expanded->backend = "expanded-finite-guarded";
    auto& refused = model.attempts[region][fs::AnalysisBackend::ExpandedFinite];
    refused.produced = true; refused.mathematical = expanded;
    refused.expandedQueries = std::make_shared<const fs::RegionalAnalysis>();
    fs::AnalysisNeeds query; query.queries = true;
    const auto unavailable = fs::previewRetainedBackendWork(model, program, region, region,
        fs::AnalysisBackend::ExpandedFinite, query);
    if (unavailable.stages[1] != fs::RetainedStage::Refused || unavailable.total != fs::EstimatedCount{0}) {
        return failBoundedSession("refused-export-preview");
    }
    return true;
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

bool checkBoundedRegionalAllocationRecipe(func::FuncOp function, pto::GMAliasPolicy policy)
{
    fs::FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return false; }
    for (std::size_t region = 1; region < session.result()->nodes.size(); ++region) {
        if (session.result()->nodes[region].kind != fs::StructureKind::Loop) { continue; }
        fs::AnalysisRequest request;
        request.region = region;
        request.needs.queries = request.needs.selectors = request.needs.synchronization = true;
        auto selected = session.analyze(request);
        if (!selected.mathematical || !selected.mathematical->boundedDemands) { continue; }
        auto first = session.prepareLogical(selected), second = session.prepareLogical(selected);
        if (failed(first) || failed(second) || !(*first)->allocationPreparation ||
            (*first)->regionalAllocation || (*second)->regionalAllocation) { return false; }
        const auto work = session.constructionCounts().mathematicalAttempts;
        fs::prepareAllocationSupport(**first);
        fs::prepareAllocationSupport(**second);
        if (!(*first)->regionalAllocation || (*first)->regionalAllocation != (*second)->regionalAllocation ||
            session.constructionCounts().mathematicalAttempts != work) { return false; }
        llvm::outs() << "bounded regional allocation: session recipe consumed, cached export\n";
        return true;
    }
    return false;
}

bool checkBoundedSession(func::FuncOp function, pto::GMAliasPolicy policy, std::size_t region, bool requireEndpoint)
{
    auto owner = std::make_unique<fs::FrontierAnalysis>(function);
    auto& session = *owner;
    session.enableStageProfiling();
    if (failed(session.initialize(policy))) { return failBoundedSession("initialize"); }
    auto demands = session.minimumDemands(region);
    if (!demands.mathematical || !demands.mathematical->boundedDemands ||
        session.constructionCounts().boundedQueryBuilds || session.constructionCounts().boundedSelectorBuilds) {
        llvm::errs() << "bounded-session initial backend="
                     << (demands.mathematical ? demands.mathematical->backend : "none")
                     << " status=" << unsigned(demands.status)
                     << " exports=" << session.constructionCounts().boundedQueryBuilds << ","
                     << session.constructionCounts().boundedSelectorBuilds << "\n";
        return failBoundedSession("demand-only-result");
    }
    fs::AnalysisNeeds needs; needs.queries = true;
    auto queries = session.minimumDemands(region, needs);
    if (!queries.available.queries || queries.available.selectors || !queries.regionalExports) {
        return failBoundedSession("query-only-result");
    }
    needs.selectors = true;
    auto selectors = session.minimumDemands(region, needs);
    if (!retainedWorkDependencies(*session.result(), region, demands, queries, selectors)) {
        return failBoundedSession("retained-work-dependencies");
    }
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
        return failBoundedSession("identical-selector-retry");
    }
    {
        // A new selector-only request reuses the complete bounded export from
        // the preceding query+selector request without walking candidates.
        fs::AnalysisNeeds selectorOnly;
        selectorOnly.selectors = true;
        const auto before = selectors.ledger->stages.size();
        auto selectedOnly = session.minimumDemands(region, selectorOnly);
        if (selectedOnly.status != fs::AnalysisStatus::Ready ||
            selectedOnly.mathematical != selectors.mathematical ||
            selectedOnly.regionalExports != selectors.regionalExports ||
            selectors.ledger->stages.size() != before + 1 || selectors.ledger->stages.back().cacheHits != 1 ||
            session.constructionCounts().boundedQueryBuilds != 1 ||
            session.constructionCounts().boundedSelectorBuilds != 1) {
            return failBoundedSession("selector-only-reuse");
        }
    }
    if (requireEndpoint) {
        auto root = session.minimumDemands(0, needs);
        const auto ranked = llvm::find_if(root.ledger->selections, [&](const auto& selection) {
            return selection.request == root.requestId && selection.region == 0 &&
                selection.method == "bounded-lifetime";
        });
        if (ranked == root.ledger->selections.end() || ranked->unbuiltWork != fs::EstimatedCount{0} ||
            !ranked->attempted || ranked->failureStage ||
            !llvm::is_contained(ranked->reused, "demands") ||
            !llvm::is_contained(ranked->reused, "queries") ||
            !llvm::is_contained(ranked->reused, "selectors")) {
            llvm::errs() << "bounded root selection charged retained canonical capabilities\n";
            return failBoundedSession("root-selection-reuse");
        }
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
            return failBoundedSession("root-export-reuse");
        }
    }
    const auto endpointWork = session.constructionCounts().mathematicalAttempts;
    const auto enclosing = session.result()->nodes[region].loops;
    if (!selectors.regionalExports->capabilities.endpointRecipes ||
        !selectors.regionalExports->prepareWithVisits) { return failBoundedSession("endpoint-recipe-export"); }
    auto firstPlan = selectors.regionalExports->prepareWithVisits(enclosing);
    auto secondPlan = selectors.regionalExports->prepareWithVisits(enclosing);
    auto wrongFrame = enclosing;
    wrongFrame.push_back(demands.mathematical->boundedDemands->loop);
    auto invalidPlan = selectors.regionalExports->prepareWithVisits(wrongFrame);
    const bool available = succeeded(firstPlan);
    if ((requireEndpoint && !available) || available != succeeded(secondPlan) || succeeded(invalidPlan) ||
        session.constructionCounts().mathematicalAttempts != endpointWork) {
        return failBoundedSession("endpoint-frame-outcomes");
    }
    if (available) {
        if (firstPlan->get() == secondPlan->get() || (*firstPlan)->completeInvocation ||
            !(*firstPlan)->mathematicalOwner || !(*secondPlan)->mathematicalOwner ||
            (requireEndpoint && (*firstPlan)->endpoints.empty()) ||
            !independentPlans(**firstPlan, **secondPlan)) { return failBoundedSession("independent-endpoint-plans"); }
        const auto before = renderPreparation(**secondPlan);
        firstPlan->reset();
        if (renderPreparation(**secondPlan) != before) { return failBoundedSession("destroy-first-plan"); }
    } else {
        std::string diagnostic;
        auto unavailable = fs::prepareBoundedLifetimeLogicalResult(
            demands.mathematical->boundedDemands, diagnostic, false);
        if (succeeded(unavailable) ||
            diagnostic.find("bounded endpoint unavailable") == std::string::npos ||
            diagnostic.find("stage=publication-contract") == std::string::npos ||
            diagnostic.find("single-publication cuts and a terminal cleanup interface") == std::string::npos) {
            llvm::errs() << "bounded-session actual endpoint diagnostic: " << diagnostic << "\n";
            return failBoundedSession("precise-endpoint-refusal");
        }
        llvm::outs() << "bounded-session endpoint obligation: " << diagnostic << "\n";
    }
    std::weak_ptr<const pto::SyncInput> inputOwner = session.sharedInput();
    auto retained = queries.regionalExports;
    const auto sites = demands.mathematical->boundedDemands->anchors.size();
    demands = {}; queries = {}; selectors = {}; retry = {};
    session.invalidate();
    owner.reset();
    if (inputOwner.expired()) { return failBoundedSession("retained-input-owner"); }
    auto& e = *retained->expressions;
    for (unsigned site = 0; site < sites; ++site) {
        auto present = fs::regionalPresence(*retained, {site, e.constant(0), fs::PeriodicEventKind::Start});
        if (!present || !e.constructionError().empty()) { return failBoundedSession("post-invalidation-presence"); }
        for (auto [id, input] : e.referencedInputs(*present)) {
            (void)id;
            if (!input.getType()) { return failBoundedSession("post-invalidation-input"); }
        }
    }
    retained.reset();
    if (available) {
        if (inputOwner.expired()) { return failBoundedSession("plan-input-owner"); }
        firstPlan->reset();
        secondPlan->reset();
    }
    if (!inputOwner.expired()) { return failBoundedSession("final-owner-release"); }
    if (requireEndpoint && !checkRegionalAllocationFailure(function, policy, region)) {
        return failBoundedSession("regional-allocation-refusal");
    }
    llvm::outs() << "bounded-session: retained demands, separate cached exports, owned input after invalidation\n";
    return true;
}
