// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CompactStorageBoundary.h"
#include "CompactWriterReaderInputInternal.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
using Kind = PeriodicEventKind;
bool charge(uint64_t& value)
{
    if (value == UINT64_MAX) { return false; }
    ++value; return true;
}
void nativeSelectors(RegionalAnalysis& out, llvm::ArrayRef<PeriodicPayload> word, Expr trips)
{
    auto& arena = *out.expressions;
    const auto zero = arena.constant(0), one = arena.constant(1);
    const auto active = arena.lt(zero, trips);
    const auto last = arena.select(active, arena.sub(trips, one), zero);
    for (uint32_t site = 0; site < word.size(); ++site) {
        RegionalSelector first{{site, zero, Kind::Start}, active};
        RegionalSelector end{{site, last, Kind::Completion}, active};
        out.firstSitePayloads[site] = {first};
        if (!out.firstPayloads.count(word[site].pipe)) { out.firstPayloads[word[site].pipe] = {first}; }
        out.lastPayloads[word[site].pipe] = {end};
    }
    out.capabilities.completeStorageModel = false;
    out.capabilities.exactSelectors = false;
}
std::optional<RegionalOrderView> finiteView(FiniteRequirements frame, std::string& error)
{
    auto context = frame->original()->context();
    RegionalOrderQueries queries;
    queries.reachability = [frame, context](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
        const auto pa = regionalPresence(context->domain(), a), pb = regionalPresence(context->domain(), b);
        if (!pa || !pb) { return std::nullopt; }
        const auto value = explicitEventPrecedes(frame->analysis(), {a.type, a.kind}, {b.type, b.kind});
        if (!value) { return std::nullopt; }
        auto& arena = *context->expressions();
        return arena.land(arena.land(*pa, *pb), arena.boolean(*value));
    };
    auto result = makeRegionalOrderView(context, std::move(queries), error);
    return succeeded(result) ? std::optional<RegionalOrderView>(*result) : std::nullopt;
}
Operation* rootAnchor(Operation* operation, Block* block)
{
    while (operation && operation->getBlock() != block) { operation = operation->getParentOp(); }
    return operation;
}
} // namespace
CompactClasses withCompactClassPreparation(CompactClasses original,
    std::function<FailureOr<std::unique_ptr<PreparedLogicalPlan>>(ArrayRef<scf::ForOp>)> preparation)
{
    if (!original || !preparation) { return original; }
    auto result = std::shared_ptr<CompactClassBoundary>(new CompactClassBoundary(*original));
    result->selectors.prepareWithVisits = std::move(preparation);
    result->selectors.prepare = [prepare = result->selectors.prepareWithVisits]() { return prepare({}); };
    result->selectors.capabilities.endpointRecipes = true;
    return result;
}
CompactClasses captureCompactClassBoundary(scf::ForOp loop, const PhaseIndex& index,
    std::shared_ptr<const CompactFixedBodyContext> domain,
    const CompactWriterReaderBindings& bindings, std::string& error)
{
    error.clear();
    if (!domain || !loop) { error = "class boundary requires a qualified compact occurrence domain"; return {}; }
    const auto& input = domain->context()->input();
    auto source = buildConditionalCompactInput(loop, input, index, bindings);
    if (!source.error.empty()) { error = source.error; return {}; }
    const auto& canonical = domain->context()->domain();
    if (source.storage.payloads.size() != domain->payloads().size() ||
        canonical.anchors.size() != source.storage.payloads.size()) {
        error = "class boundary word differs from its qualified occurrence context"; return {};
    }
    for (std::size_t i = 0; i < source.storage.payloads.size(); ++i) {
        const auto& alternatives = source.storage.phaseAlternatives[i];
        if (source.storage.payloads[i].pipe != domain->payloads()[i].pipe || canonical.occurrenceLoops[i] != loop ||
            (alternatives.size() == 1 ? canonical.anchors[i].phase != alternatives.front() :
                                       canonical.anchors[i].phase != nullptr)) {
            error = "class boundary alternatives differ from the captured slot namespace"; return {};
        }
    }
    auto result = std::shared_ptr<CompactClassBoundary>(new CompactClassBoundary());
    result->compactInput = std::make_shared<const CompactWriterReaderInput>(std::move(source.storage));
    result->compactOrders = std::make_shared<const CompactOrderBounds>(
        buildCompactOrderBounds(domain, source.mathematical, {}, 0));
    result->order = result->compactOrders->bounds;
    result->unavailable = result->compactOrders->exportError;
    result->selectors = canonical;
    nativeSelectors(result->selectors, domain->payloads(), domain->trips());
    result->effects = result->compactInput->classEffects;
    auto& arena = *domain->context()->expressions();
    const auto zero = arena.constant(0), active = arena.lt(zero, domain->trips());
    const auto last = arena.select(active, arena.sub(domain->trips(), arena.constant(1)), zero);
    for (std::size_t id = 0; id < result->compactInput->accesses.size(); ++id) {
        const auto& access = result->compactInput->accesses[id];
        result->sites.push_back({access.storageClass, domain->payloads()[access.site].pipe,
            result->compactInput->accessEffects[id], access.read, access.write,
            {{access.site, zero, Kind::Start}, active}, {{access.site, last, Kind::Completion}, active}});
    }
    for (const auto& alternatives : result->compactInput->phaseAlternatives) {
        result->originalPhases.insert(result->originalPhases.end(), alternatives.begin(), alternatives.end());
    }
    result->firstAnchor = result->lastAnchor = loop;
    result->invocation = domain->invocationBlock();
    auto function = loop->getParentOfType<func::FuncOp>();
    if (!function || loop->getBlock() != result->invocation) {
        result->unavailable = "class composition needs an enclosing conditional-presence adapter";
    }
    if (llvm::any_of(canonical.anchors, [](const auto& anchor) { return !anchor.phase; })) {
        result->unavailable = "class composition needs a balanced abstract-slot pipe/cut adapter";
    }
    if (!arena.constructionError().empty()) { result->unavailable = arena.constructionError(); }
    return result;
}
CompactClasses captureFiniteClassBoundary(FiniteRequirements frame, FiniteSelection selection, std::string& error)
{
    error.clear();
    if (!frame || (selection && selection->frame() != frame)) {
        error = "finite class boundary requires its original shared scan and certified selection"; return {};
    }
    auto result = std::shared_ptr<CompactClassBoundary>(new CompactClassBoundary());
    result->finite = frame; result->selectedFinite = selection;
    result->invocation = frame->invocationBlock();
    result->order.context = frame->original()->context();
    result->order.provenance = selection ? selection->snapshot() : frame->original();
    auto query = finiteView(frame, error);
    if (!query) { return {}; }
    result->order.lower = *query; result->order.upper = *query;
    result->order.guarantee = InputOrderGuarantee::InputOrderEquivalent;
    result->order.reduction = ReductionQuality::Generators;
    result->selectors = result->order.context->domain();
    auto& arena = *result->order.context->expressions();
    const auto zero = arena.constant(0), yes = arena.boolean(true);
    const auto& analysis = frame->analysis();
    const auto& model = result->order.context->input().accesses();
    std::vector<PeriodicPayload> word;
    for (uint32_t site = 0; site < analysis.phases.size(); ++site) {
        const auto* phase = analysis.phases[site];
        word.push_back({static_cast<uint32_t>(phase->kPipeValue)});
        result->originalPhases.push_back(phase);
        for (auto effect : model.effectsFor(phase)) {
            const auto& access = model.effects()[effect];
            if (access.rangesMaterialized && access.ranges.empty()) { continue; }
            if (result->effects.size() == UINT32_MAX) {
                error = "finite class identity exceeds representation"; return {};
            }
            const auto identity = static_cast<uint32_t>(result->effects.size());
            result->effects.push_back({effect});
            result->sites.push_back({identity, static_cast<uint32_t>(phase->kPipeValue), {effect},
                access.mode == SyncAccessMode::Read,
                access.mode == SyncAccessMode::Write, {{site, zero, Kind::Start}, yes},
                {{site, zero, Kind::Completion}, yes}});
        }
    }
    nativeSelectors(result->selectors, word, arena.constant(1));
    if (!analysis.phases.empty()) {
        result->firstAnchor = analysis.phases.front()->elementOp;
        result->lastAnchor = analysis.phases.back()->elementOp;
        auto function = result->firstAnchor->getParentOfType<func::FuncOp>();
        if (!function || result->firstAnchor->getBlock() != result->invocation) {
            result->unavailable = "finite class composition needs an enclosing conditional-presence adapter";
        }
    }
    return result;
}
CompactClasses captureGuardedClassBoundary(func::FuncOp function, Block& invocation,
    ArrayRef<Operation*> roots, const PhaseIndex& index, const SyncInput& input,
    std::shared_ptr<RegionExpressions> expressions, std::string& error)
{
    error.clear();
    if (roots.empty() || llvm::any_of(roots, [&](Operation* op) {
            return !op || op->getBlock() != &invocation;
        })) {
        error = "guarded class roots must belong to their original invocation";
        return {};
    }
    auto analysis = analyzeFiniteGuarded(function, roots, index, input, std::move(expressions));
    if (!analysis.error.empty()) { error = analysis.error; return {}; }
    auto domain = finiteGuardedRegionalResult(analysis);
    auto context = captureRegionalOrderContext(input, domain, error);
    if (failed(context)) { return {}; }
    auto view = makeRegionalOrderView(*context, {domain.reachability, domain.numerical}, error);
    if (failed(view)) { return {}; }
    auto result = std::shared_ptr<CompactClassBoundary>(new CompactClassBoundary());
    result->order.context = *context;
    result->order.lower = *view;
    result->order.upper = *view;
    result->order.guarantee = InputOrderGuarantee::InputOrderEquivalent;
    result->order.reduction = ReductionQuality::Covers;
    result->invocation = &invocation;
    result->firstAnchor = roots.front();
    result->lastAnchor = roots.back();
    result->selectors = domain;
    auto& arena = *domain.expressions;
    const auto zero = arena.constant(0);
    for (uint32_t site = 0; site < domain.anchors.size(); ++site) {
        const auto* phase = domain.anchors[site].phase;
        const RegionalEvent event{site, zero, Kind::Start};
        auto present = regionalPresence(domain, event);
        if (!present) { error = "guarded occurrence presence unavailable"; return {}; }
        result->originalPhases.push_back(phase);
        result->selectors.firstSitePayloads[site] = {{event, *present}};
        for (auto effect : input.accesses().effectsFor(phase)) {
            const auto& access = input.accesses().effects()[effect];
            if (access.rangesMaterialized && access.ranges.empty()) { continue; }
            if (result->effects.size() == UINT32_MAX) {
                error = "guarded class identity exceeds representation"; return {};
            }
            const auto identity = static_cast<uint32_t>(result->effects.size());
            result->effects.push_back({effect});
            result->sites.push_back({identity, static_cast<uint32_t>(phase->kPipeValue), {effect},
                access.mode == SyncAccessMode::Read, access.mode == SyncAccessMode::Write,
                {event, *present}, {{site, zero, Kind::Completion}, *present}});
        }
    }
    // The leaf is one relative invocation. Enclosing repetition prefixes the
    // original coordinates; it must not bind this leaf to a different nest.
    SmallVector<scf::ForOp> enclosing;
    for (auto* parent = invocation.getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) { enclosing.push_back(loop); }
    }
    std::reverse(enclosing.begin(), enclosing.end());
    result->selectors.prepareWithVisits = [prepare = domain.prepare, enclosing](ArrayRef<scf::ForOp> loops)
        -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
        if (loops != ArrayRef<scf::ForOp>(enclosing)) { return failure(); }
        return prepare();
    };
    return result;
}
CompactClassCrossings collectCompactClassCrossings(llvm::ArrayRef<CompactClasses> children)
{
    CompactClassCrossings result;
    OrderContext common;
    for (const auto& child : children) {
        if (!child || !child->bounds().context) { result.error = "missing class schema context"; return result; }
        const auto& context = child->bounds().context;
        if (common && (&common->input() != &context->input() || common->expressions() != context->expressions() ||
            common->gmAliasPolicy() != context->gmAliasPolicy())) {
            result.error = "class schemas use different shared-model contexts"; return result;
        }
        common = context;
    }
    if (children.size() > UINT32_MAX) { result.error = "class child identity exceeds representation"; return result; }
    if (!common) { return result; }
    auto& arena = *common->expressions();
    const auto& model = common->input().accesses();
    const auto protection = structuredProtection(model);
    for (uint32_t a = 0; a < children.size(); ++a) {
        for (uint32_t b = a + 1; b < children.size(); ++b) {
            for (const auto& x : children[a]->accesses()) {
                for (const auto& y : children[b]->accesses()) {
                    if (!charge(result.accessPairs)) {
                        result.error = "class access-pair count overflow"; return result;
                    }
                    if ((!x.write && !y.write) || (!x.read && !x.write) || (!y.read && !y.write)) { continue; }
                    if (ptoStorageProtection().protectsScalar(x.pipe, y.pipe)) { continue; }
                    bool protectedAll = true;
                    for (auto left : x.contributors) {
                        for (auto right : y.contributors) {
                            if (!charge(result.effectPairs)) {
                                result.error = "class protection-pair count overflow"; return result;
                            }
                            const auto& source = model.effects()[left]; const auto& target = model.effects()[right];
                            if (source.mode == SyncAccessMode::Read && target.mode == SyncAccessMode::Read) {
                                continue;
                            }
                            const auto sg = source.memory && source.memory->scope == AddressSpace::ACC ?
                                protection.within(source.phase, {}) : 0;
                            const auto tg = target.memory && target.memory->scope == AddressSpace::ACC ?
                                protection.within(target.phase, {}) : 0;
                            protectedAll &= hardwareProtectsConflict(x.pipe, sg, y.pipe, tg);
                        }
                    }
                    if (protectedAll) { continue; }
                    bool overlap = false;
                    for (auto left : children[a]->classes()[x.storageClass]) {
                        for (auto right : children[b]->classes()[y.storageClass]) {
                            if (!charge(result.effectPairs)) {
                                result.error = "class effect-pair count overflow"; return result;
                            }
                            overlap |= detail::compactEffectsMayMeet(model, left, right);
                        }
                    }
                    if (!overlap) { continue; }
                    const auto& source = x.last.event; const auto& target = y.first.event;
                    result.upper.push_back({{a, source.type, source.ordinal, Kind::Completion, source.visits},
                        {b, target.type, target.ordinal, Kind::Start, target.visits},
                        arena.land(x.last.present, y.first.present), {}});
                }
            }
        }
    }
    if (!arena.constructionError().empty()) { result.error = arena.constructionError(); }
    return result;
}
CompactClassComposition composeCompactClassBoundariesInBlock(
    func::FuncOp function, Block& invocation, const SyncInput& input, std::vector<CompactClasses> children)
{
    CompactClassComposition result;
    result.original = children;
    BoundingSequenceInput specification;
    if (!function || function.isDeclaration() || !function.getBody().hasOneBlock()) {
        result.error = "class composition needs a defined single-block function";
    }
    for (auto* ancestor = invocation.getParentOp(); ancestor; ancestor = ancestor->getParentOp()) {
        if (auto enclosing = dyn_cast<scf::ForOp>(ancestor)) { specification.enclosing.push_back(enclosing); }
    }
    std::reverse(specification.enclosing.begin(), specification.enclosing.end());
    specification.lower.bridges = specification.upper.bridges = BoundingSequenceBridges::SuppliedCrossings;
    specification.guarantee = InputOrderGuarantee::InputOrderCovering;
    if (!invocation.getParentOp() || (function && invocation.getParentOp() != function.getOperation() &&
        !function->isProperAncestor(invocation.getParentOp()))) {
        result.error = "class invocation block is outside its original function";
    }
    llvm::DenseSet<const CompoundInstanceElement*> members;
    Operation* first = nullptr; Operation* last = nullptr;
    uint64_t types = 0, classes = 0;
    for (const auto& child : children) {
        if (!child || !child->bounds().context || &child->bounds().context->input() != &input) {
            result.error = "class composition requires original shared schema owners"; return result;
        }
        BoundingSequenceChild selected;
        selected.bounds = child->bounds();
        selected.mathematicalOwner = child;
        selected.placementMayStrengthen = true;
        selected.lowerExports = selected.upperExports = child->nativeExports();
        selected.lowerExports->prepare = {}; selected.lowerExports->prepareWithVisits = {};
        selected.lowerExports->prepareFiltered = {}; selected.lowerExports->capabilities.endpointRecipes = false;
        specification.children.push_back(std::move(selected));
        if (child->invocationBlock() != &invocation) { result.error = "class children have different invocations"; }
        if (!child->exportError().empty() && result.error.empty()) { result.error = child->exportError(); }
        types += child->bounds().context->domain().anchors.size(); classes += child->classes().size();
        if (types > UINT32_MAX || classes > UINT32_MAX) { result.error = "class composition identity overflow"; }
        for (auto* phase : child->originalPhases) {
            if (!members.insert(phase).second) { result.error = "class composition repeats an original occurrence"; }
        }
        if (!child->firstAnchor) { continue; }
        if (!function || function.isDeclaration() || child->firstAnchor->getBlock() != &invocation ||
            child->lastAnchor->getBlock() != &invocation ||
            (last && (last->getBlock() != child->firstAnchor->getBlock() ||
                      !last->isBeforeInBlock(child->firstAnchor)))) {
            result.error = "class composition requires ordered sibling spans in one original invocation";
        }
        if (!first) { first = child->firstAnchor; }
        last = child->lastAnchor;
    }
    if (result.error.empty() && function && !function.isDeclaration() && first && last) {
        for (const auto* phase : input.instructions()) {
            auto* root = rootAnchor(phase->elementOp, &invocation);
            if (root && (root == first || first->isBeforeInBlock(root)) &&
                (root == last || root->isBeforeInBlock(last)) && !members.contains(phase)) {
                result.error = "class composition omitted an intervening original payload";
            }
        }
    }
    if (!result.error.empty()) {
        auto saved = std::make_shared<BoundingSequenceResult>();
        saved->original = std::make_shared<const BoundingSequenceInput>(std::move(specification));
        result.mathematical = std::move(saved); return result;
    }
    result.crossings = collectCompactClassCrossings(children);
    if (!result.crossings.error.empty()) { result.error = result.crossings.error; return result; }
    specification.upper.crossings = result.crossings.upper;
    auto composed = composeBoundingSequence(function, input, std::move(specification));
    result.mathematical = std::make_shared<const BoundingSequenceResult>(std::move(composed));
    if (!result.mathematical->error.empty()) { result.error = result.mathematical->error; return result; }
    if (!result.mathematical->bounds.upper || !result.mathematical->bounds.lower) {
        result.error = !result.mathematical->upper.exportError.empty() ? result.mathematical->upper.exportError :
            result.mathematical->lower.exportError;
        return result;
    }
    auto out = std::shared_ptr<CompactClassBoundary>(new CompactClassBoundary());
    out->composition = result.mathematical; out->order = result.mathematical->bounds;
    out->invocation = &invocation;
    out->selectors = *result.mathematical->upper.regional;
    out->firstAnchor = first; out->lastAnchor = last;
    uint32_t typeOffset = 0, classOffset = 0;
    for (const auto& child : children) {
        out->effects.insert(out->effects.end(), child->effects.begin(), child->effects.end());
        for (auto access : child->sites) {
            access.storageClass += classOffset;
            access.first.event.type += typeOffset; access.last.event.type += typeOffset;
            out->sites.push_back(std::move(access));
        }
        out->originalPhases.insert(out->originalPhases.end(),
            child->originalPhases.begin(), child->originalPhases.end());
        typeOffset += child->bounds().context->domain().anchors.size(); classOffset += child->classes().size();
    }
    result.boundary = std::move(out);
    return result;
}
CompactClassComposition composeCompactClassBoundaries(
    func::FuncOp function, const SyncInput& input, std::vector<CompactClasses> children)
{
    if (!function || function.isDeclaration()) {
        CompactClassComposition result;
        result.original = std::move(children); result.error = "class composition needs a defined function";
        return result;
    }
    return composeCompactClassBoundariesInBlock(function, function.front(), input, std::move(children));
}
} // namespace mlir::pto::frontiersynch
