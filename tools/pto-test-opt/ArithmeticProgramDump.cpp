// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Structured test output for independent finite checks of derived relations.
#include "PTO/Transforms/FrontierSynch/ArithmeticProgram.h"
#include "../../lib/PTO/Transforms/FrontierSynch/FiniteGuardedInternal.h"
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
            auto expanded = fs::analyzeExpandedFinite(function, roots, index, input);
            llvm::json::Object document{{"function", function.getSymName()}, {"error", expanded.error}};
            if (expanded.state) {
                auto& state = *expanded.state;
                const auto& form = *expanded.expandedProgram;
                dumpArithmeticJSON(function, form);
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
                document["fragments"] = form.expandedFragments;
                document["overlap_joins"] = state.cost.crossingCandidates;
                document["circuit_nodes"] = state.cost.expressionNodes;
                document["sites"] = form.sites.size();
                auto queries = fs::expandedFiniteRegionalQueries(expanded);
                document["exact_queries"] = queries.capabilities.exactQueries;
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
                        llvm::json::Array eventQueries;
                        constexpr uint32_t maxDumpSites = 64;
                        const bool dumpQueries = form.sites.size() <= maxDumpSites;
                        if (dumpQueries) {
                            for (uint32_t source = 0; source < 2 * form.sites.size(); ++source) {
                                llvm::json::Array row;
                                for (uint32_t target = 0; target < 2 * form.sites.size(); ++target) {
                                    auto event = [&](uint32_t id) {
                                        return fs::RegionalEvent{id / 2, zero, id % 2 ?
                                            fs::PeriodicEventKind::Completion : fs::PeriodicEventKind::Start};
                                    };
                                    auto answer = fs::regionalReachability(queries, event(source), event(target));
                                    row.push_back(answer ? evaluate(*answer) : UINT64_MAX);
                                }
                                eventQueries.push_back(std::move(row));
                            }
                        }
                        samples.push_back(llvm::json::Object{{"parameters", std::move(values)},
                            {"presence", std::move(presence)}, {"generators", std::move(edges)},
                            {"native", std::move(native)}, {"retained", std::move(retained)},
                            {"event_queries", std::move(eventQueries)}});
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
