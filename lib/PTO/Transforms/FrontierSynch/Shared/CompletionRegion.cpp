// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
#include "PTO/Transforms/InsertSync/SyncMacroModel.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "RecognitionInternal.h"
#include "ClosedCompletionChild.h"
#include "CompletionInlineChild.h"
#include "CompletionStorage.h"
#include "CompletionReadiness.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
std::string text(Operation* operation)
{
    std::string result;
    llvm::raw_string_ostream stream(result); operation->print(stream); return result;
}
std::vector<Operation*> operations(func::FuncOp function)
{
    std::vector<Operation*> result;
    function.walk([&](Operation* operation) { result.push_back(operation); });
    return result;
}
// Flat compact CFG of original cuts: structured loops contribute backedges,
// never repeated visits. Qualified calls supply owned closed-child summaries;
// original payloads otherwise remain individually closed primitive children.
class GraphBuilder {
public:
    GraphBuilder(
        const PhaseIndex& index, const DenseMap<Operation*, std::shared_ptr<const ClosedCompletionChild>>& children,
        ArrayRef<std::shared_ptr<const CompletionInlineChild>> inlinePlans, const CompletionStorage& storage,
        const CompletionReadiness& readiness)
        : index(index), storage(storage), readiness(readiness), children(children)
    {
        for (const auto& child : inlinePlans) { inlineEnds[child->roots().back()] = child.get(); }
    }
    std::vector<CompletionVertex> graph;
    std::vector<Operation*> anchors, lastAnchors;
    std::vector<bool> eventUse, pendingEvents;
    std::vector<const CompletionInlineChild*> resourceChild;
    std::vector<CompletionLossAtom> loss;
    std::size_t vertex(Operation* anchor)
    {
        auto id = graph.size();
        auto empty = CompletionFacts(storage.size(), 0);
        graph.push_back({{empty, empty, empty, true}, {}, false});
        anchors.push_back(anchor);
        lastAnchors.push_back(anchor);
        eventUse.push_back(false);
        pendingEvents.push_back(false);
        resourceChild.push_back(nullptr);
        loss.push_back({id, 0, true});
        if (!anchor) { return id; }
        graph[id].requiresEntryCompletion = readiness.protects(anchor);
        if (auto child = children.find(anchor); child != children.end()) {
            graph[id].summary = storage.liftCoarse(child->second->summary());
            loss[id] = {id, std::nullopt, false};
            eventUse[id] = child->second->resourceClosure() != ClosedCompletionChild::ResourceClosure::Empty;
            return id;
        }
        const auto count = index.phasesFor(anchor).size();
        loss[id] = {id, count, count <= 1};
        auto modes = storage.effects(index.phasesFor(anchor));
        graph[id].summary = {modes, modes, modes, true};
        if (auto barrier = dyn_cast<BarrierOp>(anchor); barrier && barrier.getPipe().getPipe() == PIPE::PIPE_ALL) {
            graph[id].summary.transparent = false;
        }
        return id;
    }
    std::size_t block(Block& block, std::size_t continuation, bool skipTerminator = false)
    {
        SmallVector<Operation*> roots;
        for (Operation& operation : block) {
            if (!skipTerminator || &operation != block.getTerminator()) { roots.push_back(&operation); }
        }
        return sequence(roots, continuation);
    }
    std::size_t sequence(ArrayRef<Operation*> roots, std::size_t continuation)
    {
        auto next = continuation;
        std::size_t at = roots.size();
        while (at) {
            auto* op = roots[--at];
            if (auto child = inlineEnds.find(op); child != inlineEnds.end()) {
                const auto* imported = child->second;
                at -= imported->roots().size() - 1;
                auto id = vertex(imported->roots().front());
                graph[id].summary = imported->classifiedSummary(storage, index);
                loss[id] = {id, imported->payloadCount(), imported->exactRealizedOrder()};
                lastAnchors[id] = imported->roots().back();
                resourceChild[id] = imported;
                eventUse[id] = imported->usesEvents();
                pendingEvents[id] = imported->returnsPendingEvents();
                graph[id].successors = {next};
                next = id;
                continue;
            }
            auto id = vertex(op);
            if (auto branch = dyn_cast<scf::IfOp>(op)) {
                auto yes = this->block(branch.getThenRegion().front(), next);
                auto no = branch.getElseRegion().empty() ? next : this->block(branch.getElseRegion().front(), next);
                graph[id].successors = {yes, no};
            } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
                auto header = vertex(nullptr);
                auto body = this->block(*loop.getBody(), header);
                graph[header].successors = {body, next};
                graph[id].successors = {header};
            } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
                auto condition = vertex(loop.getBefore().front().getTerminator());
                auto before = this->block(loop.getBefore().front(), condition, true);
                auto after = this->block(loop.getAfter().front(), before);
                graph[condition].successors = {after, next};
                graph[id].successors = {before};
            } else {
                graph[id].successors = {next};
            }
            next = id;
        }
        return next;
    }
private:
    DenseMap<Operation*, const CompletionInlineChild*> inlineEnds;
    const PhaseIndex& index;
    const CompletionStorage& storage;
    const CompletionReadiness& readiness;
    const DenseMap<Operation*, std::shared_ptr<const ClosedCompletionChild>>& children;
};
bool contains(const DenseSet<Operation*>& roots, Operation* operation)
{
    for (auto* op = operation; op; op = op->getParentOp()) {
        if (roots.contains(op)) { return true; }
    }
    return false;
}
bool adjacent(func::FuncOp function, ArrayRef<Operation*> roots)
{
    Operation* previous = nullptr;
    for (auto* root : roots) {
        if (!root || !function->isProperAncestor(root) || root->hasTrait<OpTrait::IsTerminator>() ||
            (previous && previous->getNextNode() != root)) { return false; }
        previous = root;
    }
    return true;
}
bool validChildren(func::FuncOp function, const SyncInput* input, ArrayRef<Operation*> roots,
    ArrayRef<std::shared_ptr<const CompletionInlineChild>> children)
{
    DenseSet<Operation*> scope(roots.begin(), roots.end()), used;
    for (const auto& child : children) {
        if (!child || child->input() != input || child->roots().empty() || !adjacent(function, child->roots())) {
            return false;
        }
        for (auto* root : child->roots()) {
            if (!contains(scope, root) || !used.insert(root).second) { return false; }
        }
    }
    for (auto* root : used) {
        if (contains(used, root->getParentOp())) { return false; }
    }
    return true;
}
} // namespace
// A source seal belongs to the immutable analysis session, not to each scope.
class CompletionSourceSnapshot {
public:
    func::FuncOp function;
    std::shared_ptr<const SyncInput> input;
    // Construction-time identity: source facts must remain unchanged. Never
    // dereferenced after capture; the snapshot does not retain the index.
    const PhaseIndex* index = nullptr;
    std::shared_ptr<const PhaseSourceFacts> facts;
    std::unique_ptr<const CompletionStorage> storage;
    std::string body;
    std::vector<Operation*> inventory;
    DenseMap<Operation*, uint64_t> ordinals;
    Operation* originalModule = nullptr;
    DictionaryAttr moduleAttributes;
    bool unchanged() const
    {
        return function->getParentOp() == originalModule && text(function) == body &&
            operations(function) == inventory &&
            function->getParentOp()->getAttrDictionary() == moduleAttributes;
    }
};
std::shared_ptr<const CompletionSourceSnapshot> CompletionRegion::captureSource(func::FuncOp function,
    std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error)
{
    error.clear();
    if (!function || !input || function.isDeclaration() || !function.getBody().hasOneBlock() ||
        !isa<ModuleOp>(function->getParentOp()) || failed(verify(function))) {
        error = "completion fallback requires one verified original structured function"; return {};
    }
    DenseSet<const CompoundInstanceElement*> remaining;
    for (const auto* phase : input->instructions()) {
        if (!phase || !phase->elementOp || !function->isProperAncestor(phase->elementOp) ||
            !remaining.insert(phase).second) {
            error = "completion input has mismatched original phases"; return {};
        }
    }
    function.walk([&](Operation* op) {
        for (const auto* phase : index.phasesFor(op)) {
            if (!remaining.erase(phase)) { error = "completion index belongs to different source facts"; }
        }
    });
    if (!remaining.empty() || !error.empty()) {
        error = "completion index does not cover the exact shared input"; return {};
    }
    auto source = std::make_shared<CompletionSourceSnapshot>();
    source->function = function; source->input = std::move(input); source->index = &index;
    source->facts = index.sourceFacts();
    source->storage = std::make_unique<CompletionStorage>(source->input->instructions());
    source->body = text(function); source->inventory = operations(function);
    for (auto [id, op] : llvm::enumerate(source->inventory)) { source->ordinals[op] = id; }
    source->originalModule = function->getParentOp();
    source->moduleAttributes = source->originalModule->getAttrDictionary();
    return source;
}
std::shared_ptr<const CompletionRegion> CompletionRegion::buildRegion(func::FuncOp function,
    std::shared_ptr<const SyncInput> input, const PhaseIndex& index, ArrayRef<Operation*> roots,
    ArrayRef<std::shared_ptr<const CompletionInlineChild>> children, std::string& error,
    std::shared_ptr<const CompletionSourceSnapshot> source)
{
    ClosedCompletionImportContext context;
    context.active.insert(function);
    return buildImpl(function, std::move(input), index, error, false, context,
        roots, children, false, std::move(source));
}
std::shared_ptr<const CompletionRegion> CompletionRegion::buildWithChildren(func::FuncOp function,
    std::shared_ptr<const SyncInput> input, const PhaseIndex& index,
    ArrayRef<std::shared_ptr<const CompletionInlineChild>> children, std::string& error,
    std::shared_ptr<const CompletionSourceSnapshot> source)
{
    ClosedCompletionImportContext context;
    context.active.insert(function);
    return buildImpl(function, std::move(input), index, error, false, context, {}, children, true, std::move(source));
}
std::shared_ptr<const CompletionRegion> CompletionRegion::buildCoarse(
    func::FuncOp function, GMAliasPolicy policy, std::string& error)
{
    auto input = std::make_shared<SyncInput>(policy);
    PhaseIndex index;
    if (!function ||
        failed(input->build(function, SyncInstructionView::PipeEnvelopes, SyncInputDetail::CoarseCompletion)) ||
        failed(index.build(function, *input))) {
        error = "completion request could not recover coarse phase/effect facts";
        return {};
    }
    return build(function, std::move(input), index, error);
}
std::shared_ptr<const CompletionRegion> CompletionRegion::build(func::FuncOp function,
    std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error)
{
    ClosedCompletionImportContext context;
    context.active.insert(function);
    return buildImpl(function, std::move(input), index, error, false, context);
}
FailureOr<CompletionSummary> CompletionRegion::summarizeEventFree(func::FuncOp function,
    std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error)
{
    if (!function || hasManualOnCoreSynchronization(function) || function.walk([](Operation* op) {
            return isa<func::CallOp, LogicalFamilyOp, LogicalSetOp, LogicalWaitOp>(op) ||
                getSyncMacroModel(op).has_value() ||
                (isPreservedSyncProtocol(op) && !isa<BarrierOp>(op)) ?
                WalkResult::interrupt() : WalkResult::advance();
        }).wasInterrupted()) {
        error = "event-free child has event resources or transitive calls";
        return failure();
    }
    ClosedCompletionImportContext context;
    context.active.insert(function);
    auto qualified = buildImpl(function, std::move(input), index, error, true, context);
    if (!qualified) { return failure(); }
    return qualified->result.summary;
}
std::shared_ptr<const CompletionRegion> CompletionRegion::qualifyExistingChild(func::FuncOp function,
    std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error,
    ClosedCompletionImportContext& context)
{
    if (!function || !function.isPrivate() || hasManualOnCoreSynchronization(function) ||
        function.walk([](Operation* op) {
            return isa<LogicalFamilyOp, LogicalSetOp, LogicalWaitOp>(op) || getSyncMacroModel(op).has_value() ||
                (isPreservedSyncProtocol(op) && !isa<BarrierOp>(op)) ?
                WalkResult::interrupt() : WalkResult::advance();
        }).wasInterrupted()) {
        error = "completion child has local event resources or hidden protocols";
        return {};
    }
    return buildImpl(function, std::move(input), index, error, true, context);
}
std::shared_ptr<const CompletionRegion> CompletionRegion::buildImpl(func::FuncOp function,
    std::shared_ptr<const SyncInput> input, const PhaseIndex& index, std::string& error, bool existingOnly,
    ClosedCompletionImportContext& context, ArrayRef<Operation*> requestedRoots,
    ArrayRef<std::shared_ptr<const CompletionInlineChild>> inlinePlans, bool wholeInvocation,
    std::shared_ptr<const CompletionSourceSnapshot> source)
{
    error.clear();
    if (!source) { source = captureSource(function, input, index, error); }
    if (!source || source->function != function || source->input != input || source->index != &index ||
        source->facts != index.sourceFacts()) {
        if (error.empty()) { error = "completion scope has a foreign source snapshot"; }
        return {};
    }
    SmallVector<Operation*> roots(requestedRoots);
    if (wholeInvocation) {
        if (!roots.empty()) { error = "whole invocation cannot supply a partial scope"; return {}; }
        for (auto& op : function.front().without_terminator()) { roots.push_back(&op); }
    }
    if (!adjacent(function, roots) || !validChildren(function, input.get(), roots, inlinePlans)) {
        error = "completion scope or retained children are foreign, overlapping or noncontiguous"; return {};
    }
    auto core = recoverSyncPhysicalCore(function);
    // A2/A3 PipeBarrier contract: ALL completes prior interfaces on every
    // pipeline of this core. Scalar-to-scalar ordering is hardware-provided.
    // https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/PipeBarrier_ISASI.html
    if (getTargetArch(function) != PTOArch::A3 ||
        (core != SyncPhysicalCore::AIC && core != SyncPhysicalCore::AIV)) {
        error = "completion fallback lacks a qualified single-core A3 protocol"; return {};
    }
    auto readiness = CompletionReadiness::qualify(roots, index);
    if (failed(readiness)) {
        error = "completion fallback lacks a forward scalar prerequisite certificate"; return {};
    }
    for (const auto& child : inlinePlans) {
        bool hiddenCut = false;
        for (auto* root : child->roots()) {
            root->walk([&](Operation* op) { hiddenCut |= readiness->protects(op); });
        }
        if (hiddenCut) {
            error = "completion child hides a required scalar readiness cut"; return {};
        }
    }
    ConstructionWork work;
    RecognitionResult leaves;
    DenseMap<Operation*, std::shared_ptr<const ClosedCompletionChild>> imports, byCallee;
    auto inspect = [&](Operation* op) {
        ++work.scopedOperations;
        work.scopedPhases += index.phasesFor(op).size();
        if (!error.empty()) { return; }
        if (recoverSyncPhysicalCore(op) != core || isa<LogicalFamilyOp, LogicalSetOp, LogicalWaitOp>(op) ||
            (isPreservedSyncProtocol(op) && !isa<BarrierOp>(op))) {
            error = "completion child lacks closed same-core execution"; return;
        }
        if (auto call = dyn_cast<func::CallOp>(op)) {
            auto callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
            auto found = byCallee.find(callee);
            auto child = found != byCallee.end() ? found->second : ClosedCompletionChild::import(call, error, context);
            if (!child) {
                return;
            }
            imports[op] = child;
            byCallee[callee] = std::move(child);
            return;
        }
        if (op->getNumRegions()) {
            if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op) ||
                llvm::any_of(op->getRegions(), [](Region& region) {
                    return !region.empty() && !region.hasOneBlock();
                })) { error = "completion fallback has unsupported control"; }
            return;
        }
        detail::inspectLeaf(*op, index, leaves);
        for (const auto* phase : index.phasesFor(op)) {
            auto present = getSyncPipelinePresence("a3", core, static_cast<PIPE>(phase->kPipeValue));
            if (present != std::optional<bool>(true)) {
                error = "completion child requires an unavailable pipe";
            }
        }
    };
    for (auto* root : roots) { root->walk(inspect); }
    if (!error.empty()) { return {}; }
    if (leaves.state != RecognitionState::Applicable) {
        error = "completion fallback has unmodeled effects, macro matching or prerequisites"; return {};
    }
    auto owner = std::shared_ptr<CompletionRegion>(new CompletionRegion);
    owner->summaryOnly = existingOnly;
    owner->wholeInvocation = wholeInvocation;
    owner->roots = roots;
    owner->inlinePlans.assign(inlinePlans.begin(), inlinePlans.end());
    owner->original = function; owner->input = std::move(input);
    owner->source = std::move(source);
    for (const auto& child : byCallee) {
        owner->children.push_back(child.second);
        owner->childFunctions.push_back(cast<func::FuncOp>(child.first));
    }
    for (const auto& child : imports) {
        owner->importedCalls.insert(child.first);
    }
    GraphBuilder builder(index, imports, inlinePlans, *owner->source->storage, *readiness);
    auto exit = builder.vertex(nullptr);
    auto entry = builder.sequence(roots, exit);
    work.graphVertices = builder.graph.size();
    work.readinessCuts = readiness->size();
    work.readinessOperationVisits = readiness->operationVisits();
    work.readinessUseVisits = readiness->useVisits();
    owner->work = work;
    for (auto [id, atom] : llvm::enumerate(builder.loss)) {
        if (builder.anchors[id]) {
            atom.firstOperation = owner->source->ordinals.find(builder.anchors[id])->second;
            atom.lastOperation = owner->source->ordinals.find(builder.lastAnchors[id])->second;
        }
    }
    std::vector<std::vector<std::size_t>> successors;
    for (const auto& vertex : builder.graph) { successors.push_back(vertex.successors); }
    owner->loss = std::make_shared<const CompletionLossCertificate>(
        std::move(builder.loss), std::move(successors), entry, exit);
    if (!owner->loss->valid()) { error = "completion loss partition is malformed"; return {}; }
    owner->result = buildCompletionTransfer(builder.graph, entry, exit, owner->source->storage->size());
    if (!owner->result.error.empty()) { error = owner->result.error; return {}; }
    if (existingOnly) {
        if (llvm::any_of(owner->result.protectEntry, [](bool protect) { return protect; })) {
            error = imports.empty() ? "event-free child requires additional internal completion" :
                "closed-descendant child requires additional internal completion";
            return {};
        }
        return owner;
    }
    for (std::size_t v = 0; v < builder.graph.size(); ++v) {
        if (!owner->result.protectEntry[v]) { continue; }
        if (!builder.anchors[v]) { error = "completion placement lacks an original command cut"; return {}; }
        owner->cuts.push_back(builder.anchors[v]);
    }
    // Closed matching proves issuance, not consumption. Propagate pending
    // notification use over the original CFG and the ALREADY selected common
    // completions. Never add a resource-driven completion here. Existing calls
    // with completed resources may share the pool only after the same cut check.
    using Child = const CompletionInlineChild*;
    std::map<Child, std::optional<CompletionInlineChild::ResourceFootprint>> footprints;
    DenseMap<Child, std::size_t> childIndices;
    for (auto [i, child] : llvm::enumerate(inlinePlans)) {
        footprints.emplace(child.get(), child->resourceFootprint());
        childIndices[child.get()] = i;
    }
    std::set<std::pair<std::size_t, std::size_t>> separate;
    auto independent = [&](Child pending, Child current) {
        if (!pending || !current) { return false; }
        const auto& a = footprints.at(pending);
        const auto& b = footprints.at(current);
        if (!a || !b) { return false; }
        bool sharedDirection = false;
        for (auto [direction, mask] : *a) {
            auto other = b->find(direction);
            if (other == b->end()) { continue; }
            if (mask & other->second) { return false; }
            sharedDirection = true;
        }
        if (sharedDirection) {
            auto first = childIndices.lookup(pending), second = childIndices.lookup(current);
            separate.insert(std::minmax(first, second));
        }
        return true;
    };
    // Each pending owner propagates independently. Union at joins is a may-use
    // union, so mutually exclusive paths cannot incorrectly prove reuse.
    std::set<std::pair<std::size_t, Child>> resourceStates;
    SmallVector<std::pair<std::size_t, Child>> resourceWork{{entry, nullptr}};
    bool resourceConflict = false;
    while (!resourceWork.empty()) {
        auto [v, pending] = resourceWork.pop_back_val();
        if (!resourceStates.insert({v, pending}).second) { continue; }
        ++owner->work.resourceStateVisits;
        auto all = dyn_cast_or_null<BarrierOp>(builder.anchors[v]);
        if (owner->result.protectEntry[v] || (all && all.getPipe().getPipe() == PIPE::PIPE_ALL)) {
            pending = nullptr;
        }
        Child current = builder.resourceChild[v];
        if (builder.eventUse[v] && pending && !independent(pending, current)) {
            resourceConflict = true;
            break;
        }
        Child newlyPending = builder.pendingEvents[v] ? current : nullptr;
        if (v == exit) { owner->pendingEvents |= pending != nullptr || newlyPending != nullptr; }
        for (auto next : builder.graph[v].successors) {
            resourceWork.push_back({next, pending ? pending : newlyPending});
            if (pending && newlyPending) { resourceWork.push_back({next, newlyPending}); }
        }
    }
    owner->separatePalettes.assign(separate.begin(), separate.end());
    if (resourceConflict) {
        // These eventful exports were provisional until the parent supplied
        // resource separation. Preserve the prior event-free composition route
        // when that additional capability is absent; do not modify child code
        // or install a barrier merely to qualify its physical resources.
        SmallVector<std::shared_ptr<const CompletionInlineChild>> qualified;
        for (const auto& child : inlinePlans) {
            if (!child->usesEvents()) { qualified.push_back(child); }
        }
        if (qualified.size() == inlinePlans.size()) {
            error = "completion children lack consumption-before-republication evidence";
            return {};
        }
        auto retry = buildImpl(function, owner->input, index, error, existingOnly, context, requestedRoots,
                               qualified, wholeInvocation, owner->source);
        if (!retry) { return {}; }
        auto retained = std::shared_ptr<CompletionRegion>(new CompletionRegion(*retry));
        retained->work.resourceStateVisits += owner->work.resourceStateVisits;
        retained->work.provisionalGraphVertices += owner->work.graphVertices;
        retained->work.provisionalScopeVisits += owner->work.scopedOperations;
        retained->work.provisionalReadinessVisits +=
            owner->work.readinessOperationVisits + owner->work.readinessUseVisits;
        return retained;
    }
    // The complete invocation drains after every child's work and before return.
    if (wholeInvocation) {
        owner->pendingEvents = false;
        owner->cuts.push_back(function.front().getTerminator());
        owner->result.summary.suffix.assign(owner->source->storage->size(), 0);
        owner->result.summary.transparent = false;
    }
    return owner;
}
bool CompletionRegion::usesEventResources() const
{
    return llvm::any_of(inlinePlans, [](const auto& child) { return child->usesEvents(); }) ||
        llvm::any_of(children, [](const auto& child) {
            return child->resourceClosure() != ClosedCompletionChild::ResourceClosure::Empty;
        });
}
uint64_t CompletionRegion::eventfulInlineChildren() const
{
    return llvm::count_if(inlinePlans, [](const auto& child) { return child->usesEvents(); });
}
bool CompletionRegion::ownsClosedCall(Operation* call) const { return importedCalls.contains(call); }
CompletionRegion::ChildContracts CompletionRegion::childContracts() const
{
    ChildContracts result;
    for (const auto& child : children) {
        switch (child->resourceClosure()) {
            case ClosedCompletionChild::ResourceClosure::Empty: ++result.eventFree; break;
            case ClosedCompletionChild::ResourceClosure::CompletedExit: ++result.completedExit; break;
            case ClosedCompletionChild::ResourceClosure::ClosedDescendants: ++result.closedDescendants; break;
        }
    }
    return result;
}
bool CompletionRegion::validateClosedChildren() const
{
    DenseSet<const ClosedCompletionChild*> checked;
    return validateClosedChildren(checked);
}
bool CompletionRegion::validateClosedChildren(DenseSet<const ClosedCompletionChild*>& checked) const
{
    return llvm::all_of(children, [&](const auto& child) { return child->unchanged(checked); });
}
LogicalResult CompletionRegion::append(PreparedLogicalPlan& destination, IRMapping& mapping,
    SmallVectorImpl<Operation*>& completionCuts, DenseSet<const CompletionSourceSnapshot*>& checkedSources,
    DenseSet<const ClosedCompletionChild*>& checkedCalls, CompletionChildAssembly& assembly) const
{
    if (!validateClosedChildren(checkedCalls) ||
        (checkedSources.insert(source.get()).second && !source->unchanged())) { return failure(); }
    SmallVector<std::pair<std::size_t, std::size_t>> groupRanges;
    for (const auto& child : inlinePlans) {
        const auto begin = assembly.allocationGroups.size();
        if (failed(child->append(destination, mapping, completionCuts, checkedSources, checkedCalls, assembly))) {
            return failure();
        }
        groupRanges.push_back({begin, assembly.allocationGroups.size()});
    }
    Builder builder(original->getContext());
    for (auto [first, second] : separatePalettes) {
        for (auto a = groupRanges[first].first; a < groupRanges[first].second; ++a) {
            for (auto b = groupRanges[second].first; b < groupRanges[second].second; ++b) {
                auto left = cast<DictionaryAttr>(assembly.allocationGroups[a]);
                auto right = cast<DictionaryAttr>(assembly.allocationGroups[b]);
                if (left.get("source") != right.get("source") || left.get("target") != right.get("target")) {
                    continue;
                }
                SmallVector<int64_t> conflicts(right.getAs<DenseI64ArrayAttr>("conflicts").asArrayRef());
                conflicts.push_back(a);
                NamedAttrList attributes(right);
                attributes.set("conflicts", builder.getDenseI64ArrayAttr(conflicts));
                assembly.allocationGroups[b] = attributes.getDictionary(original->getContext());
            }
        }
    }
    for (auto* cut : cuts) {
        auto* mapped = mapping.lookupOrNull(cut);
        if (!mapped) { return failure(); }
        completionCuts.push_back(mapped);
    }
    return success();
}
LogicalResult CompletionRegion::insert(func::FuncOp function, PreparedLogicalPlan& plan) const
{
    const bool clean = !summaryOnly && wholeInvocation && function == original && plan.completionOwner.get() == this &&
                       plan.completeInvocation == wholeInvocation &&
                       !plan.mathematicalOwner && !plan.privateState && !plan.compactBoundingOwner &&
                       !plan.regionalAllocation && !plan.allocationCertificate && !plan.allocationPreparation &&
                       !plan.familyRecipe && !plan.familyProtocol && !plan.physicalOutput && plan.preparation.empty() &&
                       plan.endpoints.empty() && plan.families.empty() && !plan.groupedFamilies &&
                       !plan.nestedIdentities && !plan.independentPieces;
    DenseSet<const CompletionSourceSnapshot*> checkedSources;
    DenseSet<const ClosedCompletionChild*> checkedCalls;
    if (!clean || !validateClosedChildren(checkedCalls) || !source->unchanged()) {
        return function.emitError("completion plan no longer owns its unchanged original invocation");
    }
    checkedSources.insert(source.get());
    IRMapping mapping;
    OwningOpRef<ModuleOp> staged = cast<ModuleOp>(function->getParentOp()->clone(mapping));
    auto clone = cast<func::FuncOp>(mapping.lookup(function.getOperation()));
    PreparedLogicalPlan retained(plan.planId);
    SmallVector<Operation*> completionCuts;
    CompletionChildAssembly assembly;
    if (failed(append(retained, mapping, completionCuts, checkedSources, checkedCalls, assembly))) {
        return function.emitError("completion child no longer has a qualified retained recipe");
    }
    auto namespaces = familyRecordNamespaces(retained.families);
    for (auto& endpoint : retained.endpoints) { endpoint.record = namespaces.at(endpoint.records.front()); }
    if (!assembly.allocationGroups.empty()) {
        Builder b(function.getContext());
        retained.allocationCertificate = b.getDictionaryAttr({
            b.getNamedAttr("version", b.getI64IntegerAttr(2)),
            b.getNamedAttr("kind", b.getStringAttr("finite")),
            b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
            b.getNamedAttr("plan", b.getI64IntegerAttr(plan.planId)),
            b.getNamedAttr("groups", b.getArrayAttr(assembly.allocationGroups))});
    }
    if (!inlinePlans.empty() && failed(insertLogicalSynchronization(clone, retained))) {
        return function.emitError("completion child failed shared staged insertion");
    }
    auto all = PipeAttr::get(function.getContext(), PIPE::PIPE_ALL);
    for (auto* mapped : completionCuts) {
        auto previous = dyn_cast_or_null<BarrierOp>(mapped->getPrevNode());
        if (previous && previous.getPipe() == all) { continue; }
        OpBuilder builder(mapped);
        auto barrier = builder.create<BarrierOp>(mapped->getLoc(), all);
        barrier->setAttr("pto.completion_boundary", builder.getUnitAttr());
    }
    Builder attributes(function.getContext());
    clone->setAttr("pto.completion_protocol",
        attributes.getStringAttr("a3-single-core-all; conservative-storage-classes"));
    // Eventful children keep their actual families and supplied allocation
    // proof. Only the notification-free case uses the ordinary empty contract.
    if (!retained.allocationCertificate) {
        clone->setAttr("pto.endpoint_families", attributes.getDictionaryAttr({
        attributes.getNamedAttr("version", attributes.getI64IntegerAttr(1)),
        attributes.getNamedAttr("plan", attributes.getI64IntegerAttr(plan.planId)),
        attributes.getNamedAttr("families", attributes.getArrayAttr({})),
        attributes.getNamedAttr("pieces", attributes.getArrayAttr({}))}));
    }
    if (retained.allocationCertificate) {
        clone->setAttr("pto.finite_allocation", retained.allocationCertificate);
    }
    if (plan.recognitionReport) { clone->setAttr("pto.frontier_recognition", plan.recognitionReport); }
    if (plan.completionPhysicalIds && failed(allocatePhysicalEventIds(clone, *plan.completionPhysicalIds))) {
        return function.emitError("completion children do not fit the requested physical pool");
    }
    if (failed(verify(*staged))) { return function.emitError("completion plan failed staged verification"); }
    function.getBody().takeBody(clone.getBody());
    function->setAttrs(clone->getAttrs());
    return success();
}
} // namespace mlir::pto::frontiersynch
