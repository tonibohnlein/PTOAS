// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Realize retained Boolean evaluations without evaluating opaque producers again.
#include "PrefixExpression.h"
#include "MappedCutInputs.h"
#include "PrefixExpressionProfile.h"
#include "PrefixExpressionSession.h"
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "../../InsertSync/SyncScalarReplay.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
struct Binding {
    Expr input;
    scf::ForOp loop;
    Operation* definition;
    int64_t lower;
    Expr ordinal;
    std::size_t valueSlot, validSlot;
};
std::optional<int64_t> literal(Value value)
{
    APInt integer;
    if (!matchPattern(value, m_ConstantInt(&integer)) || !integer.isSignedIntN(64)) { return {}; }
    return integer.getSExtValue();
}
bool sameFrame(scf::ForOp loop, Operation* cut, DominanceInfo& dominance)
{
    if (!loop->isProperAncestor(cut) && !dominance.properlyDominates(loop.getOperation(), cut)) { return false; }
    for (auto* parent = loop->getParentOp(); !isa<func::FuncOp>(parent); parent = parent->getParentOp()) {
        if (isa<scf::ForOp, scf::WhileOp>(parent) && !parent->isProperAncestor(cut)) { return false; }
    }
    return true;
}
std::vector<Binding> collectBindings(
    const std::shared_ptr<RegionExpressions>& owner, ArrayRef<std::pair<Expr, Value>> inputs, Operation* cut,
    InvocationPrefixCollection& collection)
{
    auto& arena = *owner;
    std::vector<Binding> bindings;
    DominanceInfo dominance(collection.function());
    for (auto [id, unused] : inputs) {
        auto evaluation = arena.evaluation(id);
        if (!evaluation) { continue; }
        auto loop = dyn_cast_or_null<scf::ForOp>(evaluation->loop);
        auto* definition = evaluation->original.getDefiningOp();
        if (!loop || !definition || definition->getBlock() != loop.getBody() ||
            !collection.function()->isProperAncestor(loop) || !sameFrame(loop, cut, dominance)) { continue; }
        const auto lower = literal(loop.getLowerBound()), step = literal(loop.getStep());
        // Unit-step signed loops cannot wrap on the final increment: IV<upper
        // implies IV+1<=upper<=INT64_MAX. Other strides need a retained proof.
        if (!lower || step != 1 || !loop.getInductionVar().getType().isIndex()) { continue; }
        std::string error;
        auto slots = collection.snapshotPredicate(evaluation->original, loop, owner, evaluation->ordinal, error);
        if (failed(slots)) { continue; }
        bindings.push_back({id, loop, definition, *lower, evaluation->ordinal, slots->first, slots->second});
    }
    return bindings;
}
bool inactive(RegionExpressions& arena, Expr root, PrefixExpressionProfile::Entry& work)
{
    RegionExpressions::ThresholdProofCost cost;
    const bool result = arena.inactiveByThresholds(root, &cost);
    accumulateCost(work.thresholdDiscovery, cost.dagVisits);
    accumulateCost(work.thresholdDependencies, cost.dependencyVisits);
    accumulateCost(work.thresholdPreparationHits, cost.preparationHits);
    accumulateCost(work.thresholdReused, cost.reusedNodes);
    accumulateCost(work.thresholdSkipped, cost.skippedBranches);
    accumulateCost(work.thresholdSubstitution, cost.substitutionVisits);
    accumulateCost(work.proofVisits, cost.proofVisits);
    return result;
}
// Each proof leaves every other historical input unconstrained. The premise
// contains only numeric control bounds, so the certificates compose when
// several unavailable evaluations are replaced simultaneously.
bool independentWhenUnavailable(
    RegionExpressions& arena, Expr root, Operation* cut, Binding binding, PrefixExpressionProfile::Entry& work)
{
    RegionExpressions::Transaction scratch(arena);
    auto available = arena.boolean(false), domain = arena.boolean(true);
    const auto selected = binding.ordinal;
    const auto lower = arena.constant(static_cast<uint64_t>(binding.lower));
    const auto upper = arena.input(binding.loop.getUpperBound());
    const auto exists = arena.land(arena.slt(lower, upper), arena.lt(selected, arena.sub(upper, lower)));
    if (binding.loop->isProperAncestor(cut)) {
        const auto current = arena.input(binding.loop.getInductionVar());
        const auto ordinal = arena.sub(current, lower);
        available = arena.lt(selected, ordinal);
        DominanceInfo dominance(cut->getParentOfType<func::FuncOp>());
        if (dominance.properlyDominates(binding.definition, cut)) { available = arena.le(selected, ordinal); }
        domain = arena.land(arena.sle(arena.constant(static_cast<uint64_t>(binding.lower)), current),
                            arena.slt(current, arena.input(binding.loop.getUpperBound())));
    } else {
        available = exists;
    }
    // Keep a simple signed comparison when an absolute selected IV is known.
    // This is equivalent on an executed visit, and avoids asking the proof
    // engine to rediscover the unsigned-distance identity for constant visits.
    if (auto ordinal = arena.constantValue(selected)) {
        const APInt actual = APInt(128, static_cast<uint64_t>(binding.lower), true) + APInt(128, *ordinal);
        if (actual.isSignedIntN(64)) {
            const auto induction = arena.constant(static_cast<uint64_t>(actual.getSExtValue()));
            if (binding.loop->isProperAncestor(cut)) {
                const auto current = arena.input(binding.loop.getInductionVar());
                DominanceInfo dominance(cut->getParentOfType<func::FuncOp>());
                available = dominance.properlyDominates(binding.definition, cut) ?
                    arena.sle(induction, current) : arena.slt(induction, current);
            } else { available = arena.slt(induction, upper); }
        }
    }
    if (arena.constantUnder(domain, available, &work.proofVisits) == 1) {
        return true;
    }
    RegionExpressions::Substitution no({{binding.input, arena.boolean(false)}});
    RegionExpressions::Substitution yes({{binding.input, arena.boolean(true)}});
    const auto a = arena.substitute(root, no, &work.historySubstitution);
    const auto b = arena.substitute(root, yes, &work.historySubstitution);
    const auto differs = arena.land(domain, arena.land(arena.lnot(available), arena.lnot(arena.eq(a, b))));
    return arena.constantValue(differs) == 0 || inactive(arena, differs, work);
}
struct InputBatch {
    SmallVector<std::pair<Expr, Value>> inputs;
    DominanceInfo dominance;
    InputBatch(SmallVector<std::pair<Expr, Value>> inputs, func::FuncOp function)
        : inputs(std::move(inputs)), dominance(function)
    {}
};
struct ExpressionRecipe {
    std::shared_ptr<RegionExpressions> arena;
    Expr root;
    Operation* cut;
    std::shared_ptr<const std::vector<Binding>> bindings;
    std::shared_ptr<InputBatch> inputBatch;
    std::optional<bool> historyProved;
    std::optional<bool> inactiveProved;
    std::shared_ptr<PrefixExpressionProfile> profile;
    std::shared_ptr<PrefixExpressionSession> session;
    PrefixExpressionProfile::Entry* work = nullptr;
    FailureOr<Value> emit(OpBuilder& builder, const IRMapping& mapping,
                          ArrayRef<Value> state, std::string& error)
    {
        auto& e = *arena;
        accumulateCost(work->emissions, 1);
        Block scratch;
        OpBuilder detached(builder.getContext()); detached.setInsertionPointToEnd(&scratch);
        auto attempt = [&]() {
            accumulateCost(work->attempts, 1);
            accumulateCost(work->constructed, scratch.getOperations().size());
        };
        RegionExpressions::CutEmission context;
        RegionExpressions::CutEmission* accepted = nullptr;
        auto lower = [&](PrefixExpressionSession::Environment environment) -> FailureOr<Value> {
            auto* shared = session->bindings(&e, cut, builder.getInsertionBlock(), mapping, std::move(environment));
            if (shared) {
                if (auto value = shared->values.lookup(root)) {
                    accumulateCost(work->sharedReturns, 1);
                    accepted = shared;
                    return value;
                }
            }
            PrefixExpressionSession::OriginalBindings local;
            auto* original = session->originalBindings(&e, inputBatch.get(), cut, builder.getInsertionBlock(), mapping);
            if (!original) {
                original = &local;
            }
            if (!original->prepared) {
                uint64_t traversed = 0;
                detail::bindMappedCutInputList(
                    inputBatch->inputs, cut, mapping, inputBatch->dominance, original->context, &traversed);
                original->prepared = true;
                accumulateCost(work->bindingNodes, traversed);
            }
            // Consult shared layers on demand. Do not seed every output memo
            // with every original or previously generated value in the batch.
            auto lookupValue = [&](Expr id) {
                auto value = shared ? shared->values.lookup(id) : Value{};
                return value ? value : original->context.values.lookup(id);
            };
            auto lookupInput = [&](Value input) {
                auto value = shared ? shared->inputs.lookup(input) : Value{};
                return value ? value : original->context.inputs.lookup(input);
            };
            auto hasValue = [&](Expr id) { return bool(lookupValue(id)); };
            RegionExpressions::SharedCutPlanning planning{hasValue};
            RegionExpressions::SharedCutBindings emitted{lookupValue, lookupInput};
            auto recipe = e.prepareContextual(root, cut, context, &planning);
            if (failed(recipe)) { return failure(); }
            auto result = e.emitPreparedContextual(**recipe, detached, cut, context, &emitted);
            if (succeeded(result)) { accepted = shared; }
            return result;
        };
        auto value = lower({});
        attempt();
        if (failed(value)) {
            accumulateCost(work->directFailures, 1);
            const std::string directError = e.lastEmissionError();
            scratch.clear(); detached.setInsertionPointToEnd(&scratch);
            if (!inactiveProved) {
                accumulateCost(work->inactiveAttempts, 1);
                inactiveProved = inactive(e, root, *work);
                if (*inactiveProved) {
                    accumulateCost(work->inactiveSuccesses, 1);
                }
            }
            if (*inactiveProved) {
                RegionExpressions::CutEmission inactive;
                value = e.emitContextual(e.boolean(false), detached, cut, inactive);
                attempt();
            } else {
                if (!historyProved) {
                    historyProved = !bindings->empty() && llvm::all_of(*bindings, [&](const Binding& binding) {
                        return independentWhenUnavailable(e, root, cut, binding, *work);
                    });
                }
                if (!*historyProved) { error = directError; return failure(); }
                accumulateCost(work->historyAttempts, 1);
                context = RegionExpressions::CutEmission();
                PrefixExpressionSession::Environment environment;
                for (const auto& binding : *bindings) {
                    if (binding.validSlot >= state.size() || !state[binding.valueSlot].getType().isInteger(1) ||
                        !state[binding.validSlot].getType().isInteger(1)) {
                        error = "historical expression lost its invocation state"; return failure();
                    }
                    // The independent cofactor proof justifies any defined bit
                    // while invalid. Validity is retained, never assumed true.
                    context.values[binding.input] = state[binding.valueSlot];
                    context.cofactors[binding.input] = binding.input;
                    environment.emplace_back(binding.input, state[binding.valueSlot].getAsOpaquePointer());
                }
                value = lower(std::move(environment));
                attempt();
            }
            if (failed(value)) { error = e.lastEmissionError(); return failure(); }
        }
        const auto kept = scratch.getOperations().size();
        accumulateCost(work->retained, kept);
        work->maximumRetained = std::max(work->maximumRetained, static_cast<uint64_t>(kept));
        builder.getInsertionBlock()->getOperations().splice(builder.getInsertionPoint(), scratch.getOperations());
        if (accepted) {
            for (auto item : context.values) { accepted->values.try_emplace(item.first, item.second); }
            for (auto item : context.inputs) { accepted->inputs.try_emplace(item.first, item.second); }
        }
        return *value;
    }
};
llvm::json::Object coordinateReport(const PrefixExpressionProfile::Coordinate& coordinate)
{
    llvm::json::Object result{{"expression", coordinate.expression}, {"nodes", coordinate.nodes},
        {"input_count", coordinate.inputCount}};
    if (coordinate.constant) { result["constant_u64"] = *coordinate.constant; }
    llvm::json::Array inputs;
    for (const auto& input : coordinate.inputs) {
        llvm::json::Object item{{"expression", input.expression}, {"original", input.original}};
        if (input.evaluationOrdinal) {
            item["evaluation_ordinal"] = *input.evaluationOrdinal;
            item["evaluation_loop"] = input.evaluationLoop;
        }
        inputs.push_back(std::move(item));
    }
    result["inputs"] = std::move(inputs);
    return result;
}
llvm::json::Object queryEventReport(const PrefixExpressionProfile::QueryEvent& event)
{
    llvm::json::Object result{{"type", event.type}, {"kind", event.kind}, {"anchor", event.anchor},
        {"occurrence_loop", event.occurrenceLoop}, {"ordinal", coordinateReport(event.ordinal)},
        {"visit_count", event.visitCount}, {"outer_loop_count", event.outerLoopCount}};
    llvm::json::Array visits, loops;
    for (const auto& visit : event.visits) { visits.push_back(coordinateReport(visit)); }
    for (const auto& loop : event.outerLoops) { loops.push_back(loop); }
    result["visits"] = std::move(visits); result["outer_loops"] = std::move(loops);
    return result;
}
} // namespace
PrefixExpressionProfile::Entry& PrefixExpressionProfile::request(
    const void* arena, uint32_t root, Operation* cut, const std::string& producer, uint64_t distance)
{
    auto [position, added] = entries.try_emplace({arena, root, cut, producer, distance});
    auto& entry = position->second;
    if (added) {
        entry.arena = arenas.try_emplace(arena, arenas.size()).first->second;
        entry.root = root; entry.producer = producer; entry.distance = distance;
        entry.cut = cut->getName().getStringRef().str();
        llvm::raw_string_ostream location(entry.location);
        cut->getLoc().print(location);
    }
    accumulateCost(entry.requests, 1);
    return entry;
}
void PrefixExpressionProfile::annotateSequence(
    const void* arena, uint32_t root, Operation* cut, SequenceLineage lineage)
{
    auto found = entries.find({arena, root, cut, "sequence-incoming", 0});
    if (found == entries.end()) { return; }
    auto& entry = found->second;
    accumulateCost(entry.lineageRequests, 1);
    if (entry.sequenceLineage.size() < 4) { entry.sequenceLineage.push_back(std::move(lineage)); }
}
void PrefixExpressionProfile::recordOccurrenceRestriction(const PrefixOccurrenceRestrictionWork& work)
{
    accumulateCost(incomingRequests, work.requests);
    accumulateCost(incomingChoices, work.choices);
    accumulateCost(incomingPruned, work.pruned);
    accumulateCost(incomingResidual, work.residual);
    accumulateCost(incomingProofVisits, work.proofVisits);
}
void PrefixExpressionProfile::resetEmission()
{
    for (auto& [key, entry] : entries) {
        entry.emissions = entry.attempts = entry.directFailures = entry.inactiveAttempts = entry.historyAttempts = 0;
        entry.constructed = entry.retained = entry.maximumRetained = entry.inactiveSuccesses = entry.sharedReturns =
            entry.bindingNodes = 0;
        entry.thresholdDiscovery = entry.thresholdSubstitution = entry.proofVisits = entry.historySubstitution = 0;
        entry.thresholdDependencies = entry.thresholdReused = entry.thresholdSkipped =
            entry.thresholdPreparationHits = 0;
        // Bounded translation and input slicing happen inside the staged
        // materialization transaction; regional batch discovery precedes it.
        if (entry.producer == "bounded") {
            entry.preparationNodes = entry.preparationInputs = 0;
        }
    }
}
llvm::json::Object PrefixExpressionProfile::report() const
{
    Entry totals;
    std::vector<const Entry*> largest;
    for (const auto& [key, entry] : entries) {
        accumulateCost(totals.requests, entry.requests);
        accumulateCost(totals.emissions, entry.emissions);
        accumulateCost(totals.attempts, entry.attempts);
        accumulateCost(totals.directFailures, entry.directFailures);
        accumulateCost(totals.inactiveAttempts, entry.inactiveAttempts);
        accumulateCost(totals.historyAttempts, entry.historyAttempts);
        accumulateCost(totals.inactiveSuccesses, entry.inactiveSuccesses);
        accumulateCost(totals.preparationNodes, entry.preparationNodes);
        accumulateCost(totals.preparationInputs, entry.preparationInputs);
        accumulateCost(totals.sharedReturns, entry.sharedReturns);
        accumulateCost(totals.bindingNodes, entry.bindingNodes);
        accumulateCost(totals.thresholdDiscovery, entry.thresholdDiscovery);
        accumulateCost(totals.thresholdDependencies, entry.thresholdDependencies);
        accumulateCost(totals.thresholdPreparationHits, entry.thresholdPreparationHits);
        accumulateCost(totals.thresholdReused, entry.thresholdReused);
        accumulateCost(totals.thresholdSkipped, entry.thresholdSkipped);
        accumulateCost(totals.thresholdSubstitution, entry.thresholdSubstitution);
        accumulateCost(totals.proofVisits, entry.proofVisits);
        accumulateCost(totals.historySubstitution, entry.historySubstitution);
        accumulateCost(totals.constructed, entry.constructed);
        accumulateCost(totals.retained, entry.retained);
        largest.push_back(&entry);
    }
    llvm::sort(largest, [](const Entry* a, const Entry* b) {
        if (a->constructed != b->constructed) { return a->constructed > b->constructed; }
        return std::tie(a->arena, a->root, a->location) < std::tie(b->arena, b->root, b->location);
    });
    auto counts = [](const Entry& entry) {
        return llvm::json::Object{
            {"requests", entry.requests},
            {"emissions", entry.emissions},
            {"attempts", entry.attempts},
            {"direct_failures", entry.directFailures},
            {"inactive_attempts", entry.inactiveAttempts},
            {"inactive_successes", entry.inactiveSuccesses},
            {"history_attempts", entry.historyAttempts},
            {"preparation_nodes", entry.preparationNodes},
            {"preparation_inputs", entry.preparationInputs},
            {"shared_returns", entry.sharedReturns},
            {"binding_node_visits", entry.bindingNodes},
            {"threshold_discovery_visits", entry.thresholdDiscovery},
            {"threshold_dependency_visits", entry.thresholdDependencies},
            {"threshold_preparation_cache_hits", entry.thresholdPreparationHits},
            {"threshold_reused_nodes", entry.thresholdReused},
            {"threshold_skipped_branches", entry.thresholdSkipped},
            {"threshold_substitution_visits", entry.thresholdSubstitution},
            {"proof_traversal_visits", entry.proofVisits},
            {"history_substitution_visits", entry.historySubstitution},
            {"constructed_operations", entry.constructed},
            {"retained_operations", entry.retained}};
    };
    auto result = counts(totals);
    result["incoming_capture_restriction"] = llvm::json::Object{
        {"requests", incomingRequests}, {"choice_nodes", incomingChoices}, {"pruned_choices", incomingPruned},
        {"residual_choices", incomingResidual}, {"proof_visits", incomingProofVisits}};
    result["unique_recipes"] = entries.size();
    result["scope"] = "latest-materialization-attempt";
    llvm::json::Array top;
    for (const auto* entry : ArrayRef<const Entry*>(largest).take_front(16)) {
        auto item = counts(*entry);
        item["arena"] = entry->arena; item["root"] = entry->root;
        item["producer"] = entry->producer; item["distance"] = entry->distance;
        item["cut"] = entry->cut; item["location"] = entry->location;
        item["maximum_retained_operations"] = entry->maximumRetained;
        if (entry->lineageRequests) {
            item["sequence_lineage_requests"] = entry->lineageRequests;
            llvm::json::Array lineage;
            for (const auto& source : entry->sequenceLineage) {
                llvm::json::Object record{{"origin", source.origin}, {"anchor", source.anchor},
                    {"source_cut", source.source}, {"source_dag_nodes", source.sourceNodes},
                    {"cached_queries", source.cachedQueries}, {"reachable_cached_query_pairs", source.reachableQueries},
                    {"query_group_count", source.queryGroups}, {"omitted_query_pairs", source.omittedQueryPairs},
                    {"composition_entries", source.entries}, {"composition_prefix_queries", source.prefixQueries},
                    {"composition_suffix_queries", source.suffixQueries}};
                if (source.region) { record["region"] = *source.region; }
                llvm::json::Array frame;
                for (const auto& loop : source.frame) { frame.push_back(loop); }
                record["enclosing_loops"] = std::move(frame);
                llvm::json::Array groups;
                for (const auto& group : source.groups) {
                    groups.push_back(llvm::json::Object{{"child", group.child}, {"producer", group.producer},
                        {"event_kinds", group.kinds}, {"reachable_cached_query_pairs", group.outputs}});
                }
                record["query_groups"] = std::move(groups);
                record["omitted_query_details"] = source.omittedQueryDetails;
                llvm::json::Array queries;
                for (const auto& query : source.queries) {
                    queries.push_back(llvm::json::Object{{"child", query.child}, {"output", query.output},
                        {"producer", query.producer}, {"source", queryEventReport(query.source)},
                        {"target", queryEventReport(query.target)}});
                }
                record["query_pairs"] = std::move(queries);
                lineage.push_back(std::move(record));
            }
            item["sequence_lineage_samples"] = std::move(lineage);
        }
        top.push_back(std::move(item));
    }
    result["largest_recipes"] = std::move(top);
    return result;
}
std::vector<PrefixIncomingSelector> preparePrefixExpressions(
    std::shared_ptr<RegionExpressions> arena, ArrayRef<Expr> roots, Operation* cut,
    InvocationPrefixCollection& collection, StringRef producer)
{
    if (roots.empty()) {
        return {};
    }
    uint64_t nodes = 0;
    auto inputs = arena->referencedInputs(roots, &nodes);
    auto bindings = std::make_shared<const std::vector<Binding>>(collectBindings(arena, inputs, cut, collection));
    auto inputBatch = std::make_shared<InputBatch>(std::move(inputs), collection.function());
    auto& profile = collection.recipe().expressionProfile;
    if (!profile) { profile = std::make_shared<PrefixExpressionProfile>(); }
    auto& session = collection.recipe().expressionSession;
    if (!session) { session = std::make_shared<PrefixExpressionSession>(); }
    std::vector<PrefixIncomingSelector> result;
    for (auto root : roots) {
        auto recipe = std::make_shared<ExpressionRecipe>();
        recipe->arena = arena;
        recipe->root = root;
        recipe->cut = cut;
        recipe->profile = profile;
        recipe->session = session;
        recipe->bindings = bindings;
        recipe->inputBatch = inputBatch;
        recipe->work = &profile->request(arena.get(), root, cut, producer.str());
        if (result.empty()) {
            accumulateCost(recipe->work->preparationNodes, nodes);
            accumulateCost(recipe->work->preparationInputs, inputBatch->inputs.size());
        }
        result.push_back([recipe](OpBuilder& b, const IRMapping& mapping, ArrayRef<Value> state, std::string& error) {
            return recipe->emit(b, mapping, state, error);
        });
    }
    return result;
}
PrefixIncomingSelector preparePrefixExpression(
    std::shared_ptr<RegionExpressions> arena, Expr root, Operation* cut, InvocationPrefixCollection& collection,
    StringRef producer)
{
    return preparePrefixExpressions(std::move(arena), ArrayRef<Expr>(root), cut, collection, producer).front();
}
} // namespace mlir::pto::frontiersynch
