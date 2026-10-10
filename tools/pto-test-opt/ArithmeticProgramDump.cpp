// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Structured test output for independent finite checks of derived relations.
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/ArithmeticProgramInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/FiniteExpansionPlan.h"
#include "../../lib/PTO/Transforms/FrontierSynch/FiniteGuardedInternal.h"
#include "../../lib/PTO/Transforms/FrontierSynch/RegionalRelationsInternal.h"
#include "PTO/Transforms/FrontierSynch/ProgramRecognition.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
void dumpArithmeticJSON(func::FuncOp function, const fs::ArithmeticProgram& program)
{
    llvm::json::Array relations;
    for (auto [id, relation] : llvm::enumerate(program.primitives.relations)) {
        llvm::json::Array pieces;
        for (const auto& normalized : program.recognition.pieces) {
            if (normalized.relation != id) {
                continue;
            }
            llvm::json::Array rows, residues;
            for (auto residue : relation.pieces[normalized.piece].residues) {
                residues.push_back(residue);
            }
            for (const auto& row : normalized.rows) {
                llvm::json::Array coefficients;
                for (auto coefficient : row.coefficients) {
                    coefficients.push_back(coefficient);
                }
                rows.push_back(llvm::json::Object{{"coefficients", std::move(coefficients)},
                                                {"constant", row.constant}, {"equality", row.equality}});
            }
            pieces.push_back(llvm::json::Object{{"empty", normalized.empty}, {"residues", std::move(residues)},
                                               {"rows", std::move(rows)}});
        }
        relations.push_back(llvm::json::Object{
            {"kind", static_cast<unsigned>(relation.kind)},
            {"source", relation.sourceSite ? static_cast<int64_t>(*relation.sourceSite) : -1},
            {"target", relation.targetSite ? static_cast<int64_t>(*relation.targetSite) : -1},
            {"source_event", static_cast<unsigned>(relation.sourceEvent)},
            {"target_event", static_cast<unsigned>(relation.targetEvent)},
            {"source_dimensions", relation.sourceDimensions}, {"target_dimensions", relation.targetDimensions},
            {"space", relation.storageSpace ? static_cast<int64_t>(*relation.storageSpace) : -1},
            {"base_argument", relation.storageBase
                ? static_cast<int64_t>(cast<BlockArgument>(relation.storageBase).getArgNumber()) : -1},
            {"dimensions", relation.dimensions}, {"pieces", std::move(pieces)}});
    }
    llvm::json::Array sites, parameters;
    for (const auto& site : program.sites) {
        llvm::json::Array fixed;
        for (auto coordinate : site.fixedCoordinates) { fixed.push_back(coordinate.induction); }
        sites.push_back(llvm::json::Object{{"depth", site.loops.size()}, {"fixed", std::move(fixed)},
            {"op", site.phase->elementOp->getName().getStringRef()},
            {"pipe", static_cast<unsigned>(site.phase->kPipeValue)}});
    }
    for (auto parameter : program.parameters) {
        auto argument = dyn_cast<BlockArgument>(parameter);
        parameters.push_back(argument ? static_cast<int64_t>(argument.getArgNumber()) : -1);
    }
    llvm::json::Object document{{"function", function.getSymName()}, {"period", program.primitives.period},
                                {"sites", std::move(sites)}, {"parameters", std::move(parameters)},
                                {"relations", std::move(relations)}};
    if (program.context.root && program.context.root != function.getOperation()) {
        auto label = program.context.root->getAttrOfType<StringAttr>("frontier.test_arithmetic_region");
        document["region"] = label ? label.getValue() : StringRef("anonymous");
        document["incoming_prerequisites"] = program.incomingPrerequisites.size();
        document["discharged_effects"] = program.extraction.dischargedEffects.size();
        llvm::json::Array issues;
        for (const auto& diagnostic : program.extraction.diagnostics) {
            issues.push_back(fs::recognitionName(diagnostic.issue));
        }
        llvm::json::Array arithmeticIssues;
        for (const auto& diagnostic : fs::summarizeArithmeticDiagnostics(program.recognition.diagnostics)) {
            issues.push_back(fs::recognitionName(diagnostic.issue));
            arithmeticIssues.push_back(llvm::json::Object{{"issue", fs::recognitionName(diagnostic.issue)},
                {"outside_class", diagnostic.outsideClass}, {"count", diagnostic.count},
                {"relation", diagnostic.relation}, {"piece", diagnostic.piece}, {"witness", "first"}});
        }
        document["issues"] = std::move(issues);
        document["arithmetic_issue_counts"] = std::move(arithmeticIssues);
        llvm::json::Array bindings;
        for (Value parameter : program.parameters) {
            auto argument = dyn_cast<BlockArgument>(parameter);
            StringRef kind = argument ? (argument.getOwner() == &function.front() ? "function" : "enclosing") :
                                        "entry-value";
            bindings.push_back(kind);
        }
        document["parameter_kinds"] = std::move(bindings);
    }
    llvm::outs() << "arithmetic-json " << llvm::json::Value(std::move(document)) << "\n";
}

namespace {
bool checkFiniteTranslation(const fs::ArithmeticProgram& form)
{
    auto normalized = fs::detail::normalizeFiniteDemandAccesses(form);
    const bool small = normalized && form.parameters.size() <= 3;
    if (!small) { return false; }
    const auto protection = fs::structuredProtection(form.modeledInput->accesses());
    auto original = fs::analyzeGeneralArithmeticGenerators(form, &protection);
    auto shifted = fs::analyzeGeneralArithmeticGenerators(*normalized, &protection);
    const bool valid = original.analysis().error.empty() && shifted.analysis().error.empty();
    if (!valid) { return false; }
    fs::RegionExpressions arena;
    unsigned samples = 1;
    for (unsigned i = 0; i < form.parameters.size(); ++i) { samples *= 6; }
    using Key = std::tuple<std::size_t, fs::ArithmeticEvent, std::size_t, fs::ArithmeticEvent>;
    bool concrete = true;
    auto evaluate = [&](const fs::GeneralArithmeticRelation& relation,
                        ArrayRef<fs::RegionExpressions::Id> parameters) {
        std::map<Key, bool> answers;
        for (const auto& [key, pieces] : relation) {
            for (const auto& piece : pieces) {
                auto guard = arena.integerPredicate(piece, parameters, form.primitives.period, key.parameterResidues);
                auto answer = arena.constantValue(guard);
                concrete &= answer.has_value();
                const bool active = answer == 1;
                if (active) {
                    answers[{key.source.site, key.source.event, key.target.site, key.target.event}] = true;
                }
            }
        }
        return answers;
    };
    for (unsigned sample = 0; sample < samples; ++sample) {
        SmallVector<fs::RegionExpressions::Id> parameters;
        auto digits = sample;
        for (unsigned i = 0; i < form.parameters.size(); ++i) {
            parameters.push_back(arena.constant(static_cast<int64_t>(digits % 6) - 2));
            digits /= 6;
        }
        const bool generators = evaluate(original.analysis().generators, parameters) ==
            evaluate(shifted.analysis().generators, parameters);
        const bool native = evaluate(original.analysis().nativeOrder, parameters) ==
            evaluate(shifted.analysis().nativeOrder, parameters);
        if (!generators || !native || !concrete) {
            return false;
        }
    }
    auto capped = form;
    capped.expansionFragmentLimit = normalized->expandedFragments / 2;
    uint64_t attempted = 0;
    const bool refused = !fs::detail::normalizeFiniteDemandAccesses(capped, &attempted);
    if (!refused || !attempted) { return false; }
    auto missing = form;
    missing.finiteAccessRecipes.clear();
    if (fs::detail::normalizeFiniteDemandAccesses(missing)) { return false; }
    auto mismatched = form;
    bool changed = false;
    for (auto& recipe : mismatched.finiteAccessRecipes) {
        if (form.primitives.relations[recipe.relation].storageSpace == pto::AddressSpace::LEFT) {
            recipe.translation = recipe.translation + 1;
            changed = true;
            break;
        }
    }
    auto alternative = fs::detail::normalizeFiniteDemandAccesses(mismatched);
    if (!changed || !alternative) { return false; }
    for (auto [id, relation] : llvm::enumerate(form.primitives.relations)) {
        if (relation.storageSpace != pto::AddressSpace::LEFT) { continue; }
        const auto& retained = alternative->primitives.relations[id];
        const bool sameSize = retained.pieces.size() == relation.pieces.size();
        if (!sameSize) { return false; }
        for (auto [a, b] : llvm::zip(relation.pieces, retained.pieces)) {
            if (a.system != b.system || a.residues != b.residues) { return false; }
        }
    }
    return arena.constructionError().empty();
}
bool checkStoragePreflight(func::FuncOp function, fs::RegionExpressions& retained)
{
    Block inputs;
    auto parameter = inputs.addArgument(IndexType::get(function.getContext()), function.getLoc());
    auto second = inputs.addArgument(IndexType::get(function.getContext()), function.getLoc());
    fs::BoundInteger huge(INT64_MAX);
    huge *= huge;
    auto domain = fs::IntegerSystem::create(2,
        {{{huge, fs::BoundInteger(1)}, fs::BoundInteger(0)}});
    if (failed(domain)) { return false; }
    const auto before = retained.size();
    const auto previousError = retained.constructionError();
    std::string nativeError, byteError;
    const bool native = fs::detail::preflightFiniteStoragePredicate(function, *domain,
        {parameter, second}, false, 2, {0, 0}, {}, nativeError);
    const bool physical = fs::detail::preflightFiniteStoragePredicate(function, *domain,
        {parameter}, true, 2, {0, 0}, {}, byteError);
    return !native && !physical && !nativeError.empty() && !byteError.empty() &&
        retained.size() == before && retained.constructionError() == previousError;
}
llvm::json::Array dumpEventQueries(const fs::RegionalAnalysis& region, std::size_t sites,
    const std::function<uint64_t(fs::RegionExpressions::Id)>& evaluate)
{
    llvm::json::Array result;
    if (sites > 64) { return result; }
    const auto zero = region.expressions->constant(0);
    for (uint32_t source = 0; source < 2 * sites; ++source) {
        llvm::json::Array row;
        for (uint32_t target = 0; target < 2 * sites; ++target) {
            auto event = [&](uint32_t id) {
                return fs::RegionalEvent{id / 2, zero, id % 2 ?
                    fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start};
            };
            auto answer = fs::regionalReachability(region, event(source), event(target));
            row.push_back(answer ? evaluate(*answer) : UINT64_MAX);
        }
        result.push_back(std::move(row));
    }
    return result;
}
llvm::json::Object dumpFiniteStorage(const fs::RegionalAnalysis& region,
    const std::function<uint64_t(fs::RegionExpressions::Id)>& evaluate)
{
    auto choices = [&](const std::vector<fs::RegionalSelector>& selectors) {
        llvm::json::Array result;
        for (auto selector : selectors) {
            result.push_back(llvm::json::Array{selector.event.type, evaluate(selector.present)});
        }
        return result;
    };
    auto pipes = [&](const std::map<uint32_t, std::vector<fs::RegionalSelector>>& selectors) {
        llvm::json::Array result;
        for (const auto& [pipe, selected] : selectors) {
            result.push_back(llvm::json::Array{pipe, choices(selected)});
        }
        return result;
    };
    llvm::json::Array storage;
    for (auto space : {pto::AddressSpace::MAT, pto::AddressSpace::LEFT,
                       pto::AddressSpace::RIGHT, pto::AddressSpace::ACC}) {
        for (int64_t byte : {-1, 0, 511, 512, 1023, 1024, 2047, 2048}) {
            fs::RegionalByteAddress address{space, {}, region.expressions->constant(byte)};
            auto selected = region.storageSelectors(address);
            auto member = region.expressions->boolean(false);
            bool membershipAvailable = true;
            const std::vector<fs::RegionalStorageFamily> empty;
            for (const auto& family : region.symbolicStorage ? region.symbolicStorage->families : empty) {
                auto present = family.membership(address);
                membershipAvailable &= present.has_value();
                if (present) { member = region.expressions->lor(member, *present); }
            }
            if (!selected) { continue; }
            storage.push_back(llvm::json::Object{{"space", static_cast<unsigned>(space)}, {"byte", byte},
                {"member", membershipAvailable ? evaluate(member) : UINT64_MAX},
                {"first_writers", choices(selected->firstWriters)},
                {"last_writers", choices(selected->lastWriters)},
                {"first_readers", pipes(selected->firstReaders)}, {"last_readers", pipes(selected->lastReaders)}});
        }
    }
    return llvm::json::Object{{"storage", std::move(storage)}, {"first_payloads", pipes(region.firstPayloads)},
        {"last_payloads", pipes(region.lastPayloads)}, {"first_sites", pipes(region.firstSitePayloads)}};
}
} // namespace

// Exercise the regional API only on explicit test annotations. Production
// dispatch does not claim this extraction supplies a composable plan.
void dumpRegionalArithmetic(func::FuncOp function, const fs::PhaseIndex& index,
                            const pto::SyncInput& input)
{
    function.walk([&](Operation* root) {
        const bool body = root->hasAttr("frontier.test_expanded_body");
        const bool requested = root->hasAttr("frontier.test_expanded_region") || body;
        if (requested) {
            SmallVector<Operation*> roots;
            auto arm = root->getAttrOfType<IntegerAttr>("frontier.test_expanded_body");
            const auto selected = arm ? arm.getInt() : 0;
            const bool validArm = selected >= 0 && static_cast<uint64_t>(selected) < root->getNumRegions();
            if (body && validArm && root->getRegion(selected).hasOneBlock()) {
                for (auto& operation : root->getRegion(selected).front()) {
                    if (!operation.hasTrait<OpTrait::IsTerminator>()) { roots.push_back(&operation); }
                }
                auto skip = root->getAttrOfType<IntegerAttr>("frontier.test_expanded_skip");
                auto take = root->getAttrOfType<IntegerAttr>("frontier.test_expanded_take");
                const bool validSkip = skip && skip.getInt() >= 0 &&
                    static_cast<uint64_t>(skip.getInt()) <= roots.size();
                if (validSkip) {
                    roots.erase(roots.begin(), roots.begin() + skip.getInt());
                }
                const bool validTake = take && take.getInt() >= 0 &&
                    static_cast<uint64_t>(take.getInt()) <= roots.size();
                if (validTake) {
                    roots.resize(take.getInt());
                }
            } else if (!body) { roots.push_back(root); }
            fs::ArithmeticRegionContext context{function, roots.empty() ? nullptr : roots.front()};
            context.roots = roots;
            const auto plan = fs::preflightFiniteExpansion(context, index, input);
            auto expanded = fs::analyzeExpandedFinite(plan, index, input);
            llvm::json::Object document{{"function", function.getSymName()}, {"error", expanded.error}};
            if (expanded.state) {
                document["preflight_sites"] = plan.sites.size();
                document["preflight_visits"] = plan.visits.size();
                pto::SyncInput foreign;
                auto refused = fs::materializeFiniteExpansion(plan, index, foreign);
                document["foreign_preflight_rejected"] = refused.extraction.state != fs::RecognitionState::Applicable;
                fs::PhaseIndex foreignIndex;
                auto indexRefused = fs::materializeFiniteExpansion(plan, foreignIndex, input);
                document["foreign_index_rejected"] = indexRefused.extraction.state != fs::RecognitionState::Applicable;
                auto& state = *expanded.state;
                const auto& form = *expanded.expandedProgram;
                dumpArithmeticJSON(function, form);
                const auto compatibility = fs::expandFiniteArithmeticProgram(
                    form.context, index, input, input.accesses());
                bool mappings = compatibility.sites.size() == form.sites.size() &&
                    compatibility.expandedVisits == form.expandedVisits &&
                    compatibility.expandedFoldOperations == form.expandedFoldOperations &&
                    compatibility.expandedPrunedArms == form.expandedPrunedArms &&
                    compatibility.parameters == form.parameters &&
                    compatibility.primitives.period == form.primitives.period;
                for (auto [a, b] : llvm::zip(compatibility.sites, form.sites)) {
                    mappings &= a.phase == b.phase && a.loops == b.loops && a.guards.size() == b.guards.size() &&
                        a.fixedCoordinates.size() == b.fixedCoordinates.size();
                    for (auto [x, y] : llvm::zip(a.guards, b.guards)) {
                        mappings &= x.branch == y.branch && x.takeThen == y.takeThen && x.provenTaken == y.provenTaken;
                    }
                    for (auto [x, y] : llvm::zip(a.fixedCoordinates, b.fixedCoordinates)) {
                        mappings &= x.loop == y.loop && x.induction == y.induction;
                    }
                }
                document["preflight_mapping_matches"] = mappings;
                document["root_count"] = form.context.roots.size();
                document["incoming_prerequisites"] = form.incomingPrerequisites.size();
                bool invalidRejected = true;
                for (SmallVector<Operation*> malformed : {SmallVector<Operation*>{nullptr, root},
                                                         SmallVector<Operation*>{root, root},
                                                         SmallVector<Operation*>{root, nullptr}}) {
                    fs::ArithmeticRegionContext context{function, root};
                    context.roots = std::move(malformed);
                    auto invalid = fs::expandFiniteArithmeticProgram(context, index, input, input.accesses());
                    invalidRejected &= invalid.extraction.state != fs::RecognitionState::Applicable;
                }
                document["invalid_roots_rejected"] = invalidRejected;
                if (form.primitives.period == 2) {
                    fs::FiniteExpansionLimits limits;
                    limits.fragments = form.expandedFragments / 3;
                    auto smaller = fs::expandFiniteArithmeticProgram(
                        form.context, index, input, input.accesses(), limits);
                    document["period_budget_fallback"] = smaller.extraction.state == fs::RecognitionState::Applicable &&
                        smaller.recognition.state == fs::RecognitionState::Applicable && smaller.primitives.period == 1;
                }
                document["visits"] = form.expandedVisits;
                document["fold_operations"] = form.expandedFoldOperations;
                document["pruned_arms"] = form.expandedPrunedArms;
                document["translation_fragments"] = form.expandedTranslationFragments;
                if (function->hasAttr("test.translation_compare")) {
                    document["translation_checked"] = checkFiniteTranslation(form);
                }
                document["fragments"] = form.expandedFragments;
                document["overlap_joins"] = state.cost.crossingCandidates;
                document["circuit_nodes"] = state.cost.expressionNodes;
                document["sites"] = form.sites.size();
                auto queries = fs::expandedFiniteRegionalQueries(expanded);
                document["exact_queries"] = queries.capabilities.exactQueries;
                std::optional<fs::RegionalAnalysis> selectedStorage, relationStorage;
                if (function->hasAttr("test.expanded_storage")) {
                    document["preflight_failure_retained"] = checkStoragePreflight(function, *state.arena);
                    std::string error;
                    uint64_t checks = 0;
                    auto selected = fs::expandedFiniteRegionalSelectors(expanded, queries, error, {}, &checks);
                    document["selector_checks"] = checks;
                    document["storage_translation_families"] = state.expandedStorageTranslations.size();
                    document["selector_error"] = error;
                    document["exact_selectors"] = succeeded(selected);
                    if (succeeded(selected)) {
                        selectedStorage = std::move(*selected);
                        document["relation_available"] = bool(selectedStorage->relations);
                        document["relation_error"] = state.expandedRelationError;
                        document["relation_projections"] = state.expandedRelationCost.projections;
                        if (selectedStorage->relations) {
                            auto carrier = std::make_shared<fs::RegionalRelations>(*selectedStorage->relations);
                            auto native = fs::exportRegionalRelationData(carrier, state.arena, error);
                            if (succeeded(native) && native->capabilities.exactSelectors) {
                                relationStorage = std::move(*native);
                            }
                        }
                        const auto invalid = fs::RegionExpressions::invalid;
                        const auto boolean = state.arena->boolean(true);
                        document["invalid_storage_rejected"] =
                            !selectedStorage->storageSelectors({pto::AddressSpace::LEFT, {}, invalid}) &&
                            !selectedStorage->storageSelectors({pto::AddressSpace::LEFT, {}, boolean});
                    }
                    document["query_snapshot_unchanged"] = !queries.capabilities.exactSelectors &&
                        !queries.capabilities.completeStorageModel && !queries.storageSelectors &&
                        !queries.symbolicStorage;
                    bool aliasChecked = true;
                    unsigned aliasQueryCount = 0;
                    if (selectedStorage) {
                        for (const auto& family : selectedStorage->symbolicStorage->families) {
                            if (family.space != pto::AddressSpace::GM) { continue; }
                            for (auto argument : function.getArguments()) {
                                const bool foreign = isa<pto::PtrType>(argument.getType()) && argument != family.base;
                                if (!foreign) { continue; }
                                ++aliasQueryCount;
                                fs::RegionalByteAddress address{pto::AddressSpace::GM, argument,
                                    state.arena->constant(0)};
                                auto member = family.membership(address);
                                auto selected = selectedStorage->storageSelectors(address);
                                const bool comparable = input.memory().gmPolicy() == pto::GMAliasPolicy::MayNotAlias;
                                aliasChecked &= comparable ? member && state.arena->constantValue(*member) == 0 &&
                                    selected && selected->firstWriters.empty() && selected->lastWriters.empty() :
                                    !member && !selected;
                            }
                        }
                    }
                    document["alias_queries_checked"] = aliasChecked;
                    document["alias_query_count"] = aliasQueryCount;
                }
                const auto zero = state.arena->constant(0);
                bool invalidQueries = true;
                if (!form.sites.empty()) {
                    fs::RegionalEvent external{0, zero, fs::PeriodicEventKind::Start, {zero}};
                    fs::RegionalEvent invalid{static_cast<uint32_t>(form.sites.size()), zero,
                        fs::PeriodicEventKind::Start};
                    fs::RegionalEvent boolean{0, state.arena->boolean(true), fs::PeriodicEventKind::Start};
                    invalidQueries = !queries.presence(external) && !queries.reachability(external, external) &&
                        !queries.presence(invalid) && !queries.presence(boolean);
                }
                document["invalid_queries_rejected"] = invalidQueries;
                llvm::json::Array samples;
                const auto count = form.parameters.size();
                if (count <= 4) {
                    const bool signedSamples = function->hasAttr("test.sample_signed") && count <= 2 &&
                        llvm::all_of(form.parameters, [](Value value) { return value.getType().isIndex(); });
                    unsigned sampleCount = 1;
                    for (unsigned i = 0; i < count; ++i) { sampleCount *= signedSamples ? 6 : 2; }
                    for (unsigned mask = 0; mask < sampleCount; ++mask) {
                        unsigned digits = mask;
                        SmallVector<std::pair<fs::RegionExpressions::Id, fs::RegionExpressions::Id>> bindings;
                        llvm::json::Array values;
                        for (unsigned i = 0; i < count; ++i) {
                            const int64_t value = signedSamples ? static_cast<int64_t>(digits % 6) - 2 : digits % 2;
                            digits /= signedSamples ? 6 : 2;
                            values.push_back(value);
                            auto expression = state.arena->input(form.parameters[i]);
                            auto fixed = form.parameters[i].getType().isInteger(1) ?
                                state.arena->boolean(value) : state.arena->constant(value);
                            bindings.push_back({expression, fixed});
                        }
                        fs::RegionExpressions::Substitution substitution(bindings);
                        auto evaluate = [&](fs::RegionExpressions::Id id) {
                            auto value = state.arena->substitute(id, substitution);
                            return state.arena->constantValue(value).value_or(UINT64_MAX);
                        };
                        llvm::json::Array presence, edges, native, retained;
                        for (auto guard : state.presence) { presence.push_back(evaluate(guard)); }
                        auto append = [&](const auto& source, llvm::json::Array& target) {
                            for (auto edge : source) {
                                target.push_back(llvm::json::Array{edge.source, edge.target, evaluate(edge.guard)});
                            }
                        };
                        append(state.guardedResidual, edges);
                        append(state.guardedNative, native);
                        append(state.retained, retained);
                        auto eventQueries = dumpEventQueries(queries, form.sites.size(), evaluate);
                        llvm::json::Object sample{{"parameters", std::move(values)},
                            {"presence", std::move(presence)}, {"generators", std::move(edges)},
                            {"native", std::move(native)}, {"retained", std::move(retained)},
                            {"event_queries", std::move(eventQueries)}};
                        if (selectedStorage) {
                            auto storage = dumpFiniteStorage(*selectedStorage, evaluate);
                            for (auto& entry : storage) { sample[entry.first] = std::move(entry.second); }
                        }
                        if (relationStorage) {
                            sample["relation_event_queries"] =
                                dumpEventQueries(*relationStorage, form.sites.size(), evaluate);
                            sample["relation_storage"] = dumpFiniteStorage(*relationStorage, evaluate);
                        }
                        samples.push_back(std::move(sample));
                    }
                }
                document["samples"] = std::move(samples);
                document["exports_blocked"] = !fs::finiteGuardedRegionalResult(expanded).capabilities.exactQueries &&
                    failed(fs::prepareFiniteGuardedLogicalInsertion(expanded));
            }
            llvm::outs() << "expanded-json " << llvm::json::Value(std::move(document)) << "\n";
        }
        if (!root->hasAttr("frontier.test_arithmetic_region")) { return; }
        auto program = fs::recognizeArithmeticProgram({function, root}, index, input,
                                                      input.accesses(), {8, 8, 2, 8});
        dumpArithmeticJSON(function, program);
    });
}
