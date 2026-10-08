// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PhaseNormalization.h"
#include "BoundarySlices.h"
#include "CountedLoop.h"
#include "RecognitionInternal.h"
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingRegional.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/NumericTemplateRegional.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "PTO/Transforms/FrontierSynch/RepeatedPhases.h"
namespace mlir::pto::frontiersynch {
namespace {
void specialize(GuardedRotatingAnalysis& analysis, RegionExpressions::Substitution& bindings)
{
    auto rewrite = [&](Expr& expression) { expression = analysis.expressions->substitute(expression, bindings); };
    auto payloads = [&](auto& values) { for (auto& value : values) { rewrite(value.presence); } };
    auto records = [&](auto& values) {
        for (auto& value : values) { rewrite(value.active); rewrite(value.displacement); }
    };
    payloads(analysis.payloads);
    payloads(analysis.periodic.payloads);
    records(analysis.generators);
    records(analysis.periodic.records);
    records(analysis.periodic.nativePrerequisites);
    for (auto& value : analysis.periodic.retained) { rewrite(value); }
    for (auto& frontier : analysis.periodic.frontiers) {
        for (auto& threshold : frontier.starts) {
            rewrite(threshold.reachable); rewrite(threshold.distance);
        }
        for (auto& threshold : frontier.completions) {
            rewrite(threshold.reachable); rewrite(threshold.distance);
        }
    }
    for (auto& fragment : analysis.fragments) {
        rewrite(fragment.offset); rewrite(fragment.read); rewrite(fragment.write);
    }
}
} // namespace
bool SequenceAnalysisState::phasedChild(const StructureNode& node, Expr trips)
{
    RegionExpressions::Transaction transaction(expressions);
    auto unavailable = [&](const std::string& reason) {
        if (!repeatedAttempt.empty()) { repeatedAttempt += "; "; }
        repeatedAttempt += "q-phase repeat: " + reason;
        return false;
    };
    auto outer = dyn_cast<scf::ForOp>(node.anchor);
    if (!outer || index.hasRelevantCarriedState(outer) || node.children.size() != 1) {
        return unavailable("single body without relevant carried state required");
    }
    if (llvm::any_of(input->instructions(), [&](auto* phase) {
            return phase->macroOpInstanceId >= 0 && outer->isProperAncestor(phase->elementOp);
        })) {
        return unavailable("repeated macro envelope and hidden-event adapter not implemented yet");
    }
    const auto& sequence = program->nodes[node.children.front()];
    PhaseNormalization normalizer(outer, index, expressions);
    uint64_t period = 1;
    bool nested = false, uniformControl = true, periodFits = true;
    auto addPeriod = [&](uint64_t slots) {
        if (!slots) { periodFits = false; return; }
        auto factor = slots / std::gcd(period, slots);
        if (factor > maxRegionalSlotVisits / period) { periodFits = false; return; }
        period *= factor;
    };
    outer.getBody()->walk([&](Operation* operation) {
        if (auto inner = dyn_cast<scf::ForOp>(operation)) {
            nested = true;
            uniformControl &= normalizer.independent(inner.getLowerBound()) &&
                normalizer.independent(inner.getUpperBound()) && normalizer.independent(inner.getStep());
        }
        if (operation->getNumResults() == 1 && index.isRelevant(operation->getResult(0)) &&
            !normalizer.independent(operation->getResult(0))) {
            if (auto divisor = PhaseNormalization::modulus(operation->getResult(0))) { addPeriod(*divisor); }
        }
    });
    if (!nested) {
        return unavailable("no nested body");
    }
    if (!uniformControl) {
        return unavailable("nested bounds must be invariant across outer visits");
    }
    std::string sliceError;
    std::vector<BoundarySlice> intervals;
    bool boundaryControl = false;
    outer.getBody()->walk([&](scf::IfOp branch) { boundaryControl |= !normalizer.independent(branch.getCondition()); });
    if (boundaryControl) {
        auto sliced = collectBoundarySlices(outer, index, expressions, trips, DenseMap<Value, Expr>(), sliceError);
        if (!sliced) { return unavailable(sliceError); }
        intervals = std::move(*sliced);
    } else { intervals.push_back({{c(0), trips}, DenseMap<Value, Expr>()}); }
    // Keep the native regional providers: specialize their parameters, not a
    // flattened or atomic replacement of their payloads. Arithmetic providers
    // reconstruct physical byte selectors under these bindings; substituting
    // only endpoint expressions would leave the wrong physical cell labels.
    std::map<std::size_t, GuardedRotatingAnalysis> rotating;
    std::map<std::size_t, SmallVector<Value>> rotatingParameters;
    for (auto id : sequence.children) {
        const auto& child = program->nodes[id];
        if (child.kind != StructureKind::Loop) { continue; }
        auto inner = dyn_cast<scf::ForOp>(child.anchor);
        if (!inner) { continue; }
        auto recognition = recognizeGuardedRotating(inner, index, *input, input->accesses());
        if (recognition.result.state != RecognitionState::Applicable) { continue; }
        DenseSet<Value> seen;
        for (const auto& access : recognition.result.accesses) {
            bool varies = false;
            for (Value parameter : access.parameters) {
                if (normalizer.independent(parameter)) { continue; }
                varies = true;
                if (seen.insert(parameter).second) { rotatingParameters[id].push_back(parameter); }
            }
            if (varies) { addPeriod(access.slots); }
        }
        auto analyzed = analyzeGuardedRotating(inner, *input, recognition, arena);
        if (!analyzed.error.empty()) { return unavailable(analyzed.error); }
        rotating.emplace(id, std::move(analyzed));
    }
    if (!periodFits || (period <= 1 && !boundaryControl)) {
        return unavailable("no supported outer bank period or finite control boundary");
    }
    bool uniformDescriptors = true;
    outer.getBody()->walk([&](Operation* operation) {
        if (operation->getNumRegions() || !index.phasesFor(operation).empty()) { return; }
        auto storage = [](Type type) { return isa<TileBufType, MultiTileBufType, PtrType, MemRefType>(type); };
        if (!llvm::any_of(operation->getOperandTypes(), storage) &&
            !llvm::any_of(operation->getResultTypes(), storage)) { return; }
        for (Value operand : operation->getOperands()) {
            if (operand.getType().isIntOrIndex()) { uniformDescriptors &= normalizer.periodic(operand, period); }
        }
    });
    if (!uniformDescriptors) { return unavailable("a descriptor is not invariant under the outer bank period"); }
    for (const auto& effect : input->accesses().effects()) {
        if (!effect.phase || !outer->isProperAncestor(effect.phase->elementOp)) { continue; }
        const auto effectId = static_cast<std::size_t>(&effect - input->accesses().effects().data());
        if (detail::dischargeGlobalReadOnlyEffect(effectId, input->accesses())) { continue; }
        if (effect.selection && !normalizer.periodic(effect.selection->selector, period)) {
            return unavailable("storage selection does not repeat under the outer bank period");
        }
        for (const auto& region : effect.regions) {
            if (!normalizer.periodic(region, period)) {
                return unavailable("physical storage map needs an owned-family or boundary phase adapter");
            }
        }
    }
    auto writersExported = [&](const RegionalAnalysis& view) {
        DenseSet<std::size_t> exported;
        for (const auto& access : view.accessBoundary) { exported.insert(access.effect); }
        for (const auto& payload : view.anchors) {
            for (auto effect : input->accesses().effectsFor(payload.phase)) {
                if (input->accesses().effects()[effect].mode == SyncAccessMode::Write && !exported.count(effect)) {
                    return false;
                }
            }
        }
        return true;
    };
    std::vector<RegionalAnalysis> intervalViews;
    for (const auto& interval : intervals) {
        std::vector<RegionalAnalysis> phases;
        for (uint64_t phase = 0; phase < period; ++phase) {
            auto previousAttempt = repeatedAttempt;
            std::string compactFailure;
            bool invalidExpression = false;
            auto buildCompactBody = [&]() -> std::optional<RegionalAnalysis> {
                // Destruction order matters: discard providers and their query
                // caches, then this local normalizer, before rolling back arena
                // nodes. The enclosing normalizer never observes attempt IDs.
                RegionExpressions::Transaction compactTransaction(expressions);
                PhaseNormalization normalizer(outer, index, expressions);
                std::vector<RegionalAnalysis> parts;
                std::function<bool(std::size_t)> appendChild = [&](std::size_t id) {
                    const auto& child = program->nodes[id];
                    if (!child.payloadCount) { return true; }
                    if (child.kind == StructureKind::Sequence) {
                        for (auto nested : child.children) { if (!appendChild(nested)) { return false; } }
                        return true;
                    }
                    if (child.kind == StructureKind::Conditional) {
                        auto branch = dyn_cast_or_null<scf::IfOp>(child.anchor);
                        if (!branch || index.hasRelevantResults(branch) || index.needsValuePrerequisite(branch)) {
                            return unavailable("phase conditional needs result-free mapped control");
                        }
                        auto condition = boundaryGuard(
                            branch.getCondition(), normalizer, interval.bindings, expressions);
                        if (!condition) { return unavailable("phase conditional has no invariant interval predicate"); }
                        for (auto armId : child.children) {
                            const auto& arm = program->nodes[armId];
                            bool takeThen = arm.region == &branch.getThenRegion();
                            auto guard = takeThen ? *condition : expressions.lnot(*condition);
                            if (expressions.constantValue(guard) == 0) { continue; }
                            auto first = parts.size();
                            if (!appendChild(armId)) { return false; }
                            if (expressions.constantValue(guard) == 1) { continue; }
                            for (std::size_t part = first; part < parts.size(); ++part) {
                                parts[part] = guardRegionalArm(std::move(parts[part]), guard);
                            }
                        }
                        return true;
                    }
                    std::string diagnostic;
                    if (auto inner = dyn_cast_or_null<scf::ForOp>(child.anchor);
                        child.kind == StructureKind::Loop && inner) {
                        // A compact child with an invariant complete interface can
                        // reuse its own route recursively. Every nontrivial outer
                        // dependence stays with the explicit phase specializer.
                        bool invariant = true;
                        inner->walk([&](Operation* operation) {
                            if (auto nested = dyn_cast<scf::ForOp>(operation)) {
                                invariant &= normalizer.independent(nested.getLowerBound()) &&
                                    normalizer.independent(nested.getUpperBound()) &&
                                    normalizer.independent(nested.getStep());
                            } else if (auto branch = dyn_cast<scf::IfOp>(operation)) {
                                invariant &= normalizer.independent(branch.getCondition());
                            }
                        });
                        for (const auto& effect : input->accesses().effects()) {
                            if (!effect.phase || !inner->isProperAncestor(effect.phase->elementOp)) { continue; }
                            auto effectId = static_cast<std::size_t>(&effect - input->accesses().effects().data());
                            if (detail::dischargeGlobalReadOnlyEffect(effectId, input->accesses())) { continue; }
                            if (effect.selection) { invariant &= normalizer.independent(effect.selection->selector); }
                            for (const auto& region : effect.regions) {
                                if (region.base) { invariant &= normalizer.independent(region.base); }
                                for (Value symbol : region.symbols) { invariant &= normalizer.independent(symbol); }
                            }
                        }
                        if (invariant) {
                            auto analyzed = analyzeSequenceRegion(function, *input, *program, id, arena, indexOwner);
                            if (!analyzed.error.empty()) {
                                return unavailable("invariant nested phase: " + analyzed.error);
                            }
                            parts.push_back(sequenceRegionalResult(analyzed));
                            return true;
                        }
                    }
                    if (auto found = rotating.find(id); found != rotating.end() && interval.bindings.empty()) {
                        SmallVector<std::pair<Expr, Expr>> replacements;
                        for (Value parameter : rotatingParameters[id]) {
                            auto value = normalizer.residue(parameter, phase, period);
                            if (!value) { return unavailable("fixed-width bank residue normalization is unproved"); }
                            replacements.emplace_back(expressions.input(parameter), *value);
                        }
                        auto specialized = found->second;
                        RegionExpressions::Substitution substitution(replacements);
                        specialize(specialized, substitution);
                        auto view = guardedRotatingRegionalResult(function, *input, specialized, diagnostic);
                        if (failed(view)) { return unavailable("child phase export: " + diagnostic); }
                        parts.push_back(std::move(*view));
                        return true;
                    }
                    if (auto inner = dyn_cast_or_null<scf::ForOp>(child.anchor);
                        child.kind == StructureKind::Loop && inner) {
                        auto domain = CountedLoop::get(inner);
                        if (!domain) {
                            return unavailable("nested sliced child requires a representable counted domain");
                        }
                        auto innerTrips = domain->trips(expressions);
                        auto childSlices = collectBoundarySlices(
                            inner, index, expressions, innerTrips, interval.bindings, diagnostic);
                        if (!childSlices) { return unavailable("nested boundary: " + diagnostic); }
                        std::vector<RegionalAnalysis> childParts;
                        for (auto& slice : *childSlices) {
                            PhaseNormalization childNormalizer(inner, index, expressions);
                            DenseMap<Value, bool> choices;
                            SmallVector<Value> guards;
                            inner.getBody()->walk([&](scf::IfOp branch) {
                                auto bound = boundaryGuard(
                                    branch.getCondition(), childNormalizer, slice.bindings, expressions);
                                if (!bound) { return; }
                                slice.bindings[branch.getCondition()] = *bound;
                                if (auto known = expressions.constantValue(*bound)) {
                                    choices[branch.getCondition()] = *known != 0;
                                }
                                else { guards.push_back(branch.getCondition()); }
                            });
                            auto recognized = detail::recognizeRotatingSlice(inner, index, *input, choices, guards);
                            if (recognized.result.state != RecognitionState::Applicable) {
                                std::string reason = "nested slice does not supply a rotating regional interface";
                                for (const auto& issue : recognized.result.diagnostics) {
                                    reason += " / " + recognitionName(issue.issue).str();
                                }
                                return unavailable(reason);
                            }
                            auto analyzed = analyzeGuardedRotating(inner, *input, recognized, arena, slice.bindings);
                            if (!analyzed.error.empty()) { return unavailable(analyzed.error); }
                            SmallVector<std::pair<Expr, Expr>> replacements;
                            DenseSet<Value> seen;
                            for (const auto& access : recognized.result.accesses) {
                                for (Value parameter : access.parameters) {
                                    if (normalizer.independent(parameter) || !seen.insert(parameter).second) {
                                        continue;
                                    }
                                    auto value = normalizer.residue(parameter, phase, period);
                                    if (!value) {
                                        return unavailable("nested phase parameter lacks a residue certificate");
                                    }
                                    replacements.emplace_back(expressions.input(parameter), *value);
                                }
                            }
                            RegionExpressions::Substitution substitution(replacements);
                            specialize(analyzed, substitution);
                            auto view = guardedRotatingRegionalResult(
                                function, *input, analyzed, diagnostic, slice.interval);
                            if (failed(view)) { return unavailable("nested slice export: " + diagnostic); }
                            childParts.push_back(std::move(*view));
                        }
                        if (childParts.empty()) { return true; }
                        auto composed = composeRegionalSequenceWithin(
                            function, arena, std::move(childParts), true, false, child.loops);
                        if (!composed.error.empty()) {
                            return unavailable("nested slice composition: " + composed.error);
                        }
                        composed.state->completeInvocation = false;
                        composed.state->requiredOuterLoops.assign(child.loops.begin(), child.loops.end());
                        parts.push_back(sequenceRegionalResult(composed));
                        return true;
                    }
                    auto appendArithmetic = [&](Operation* root) {
                        bool guardContract = true;
                        root->walk([&](scf::IfOp branch) {
                            auto condition = branch.getCondition();
                            auto* definition = condition.getDefiningOp();
                            // Entry Boolean parameters receive the certified slice
                            // bindings directly. A predicate defined inside this root
                            // would instead be expanded by the arithmetic producer;
                            // its outer-IV comparisons cannot use a phase representative.
                            if (definition && root->isProperAncestor(definition) &&
                                !normalizer.independent(condition)) {
                                guardContract = false;
                            }
                        });
                        if (!guardContract) {
                            diagnostic = "root-local outer control requires a recursively selected regional arm";
                            return false;
                        }
                        auto view = analyzeArithmeticRegion({function, root}, index, *input, arena, diagnostic,
                            [&](Value parameter) -> std::optional<Expr> {
                                if (parameter.getType().isInteger(1)) {
                                    if (auto bound = boundaryGuard(
                                            parameter, normalizer, interval.bindings, expressions)) {
                                        return bound;
                                    }
                                }
                                return normalizer.atPhase(parameter, phase, period);
                            });
                        if (failed(view)) { return false; }
                        parts.push_back(std::move(*view));
                        return true;
                    };
                    if (child.kind == StructureKind::ExplicitRun) {
                        bool invariant = true;
                        for (auto payload : child.payloads) {
                            for (auto effectId : program->payloads[payload].effects) {
                                const auto& effect = input->accesses().effects()[effectId];
                                if (effect.selection) {
                                    invariant &= normalizer.independent(effect.selection->selector);
                                }
                                for (const auto& region : effect.regions) {
                                    if (region.base) { invariant &= normalizer.independent(region.base); }
                                    for (Value symbol : region.symbols) { invariant &= normalizer.independent(symbol); }
                                }
                            }
                        }
                        if (invariant) {
                            auto analyzed = analyzeSequenceRegion(function, *input, *program, id, arena, indexOwner);
                            if (!analyzed.error.empty()) {
                                return unavailable("invariant explicit phase: " + analyzed.error);
                            }
                            parts.push_back(sequenceRegionalResult(analyzed));
                            return true;
                        }
                        auto explicitPhase = [&]() -> FailureOr<RegionalAnalysis> {
                            RegionExpressions::Transaction runTransaction(expressions);
                            PhaseNormalization runNormalizer(outer, index, expressions);
                            SmallVector<scf::ForOp> enclosing(node.loops.begin(), node.loops.end());
                            enclosing.push_back(outer);
                            auto view = specializedExplicitRunRegional(function, outer, child.operations, index,
                                *input, arena, enclosing, [&](Value value) -> std::optional<int64_t> {
                                    auto bound = runNormalizer.atPhase(value, phase, period);
                                    if (!bound) { return std::nullopt; }
                                    auto literal = expressions.constantValue(*bound);
                                    return literal ? std::optional<int64_t>(APInt(64, *literal).getSExtValue()) :
                                                     std::nullopt;
                                }, diagnostic);
                            if (failed(view)) { return failure(); }
                            runTransaction.commit();
                            return view;
                        }();
                        if (succeeded(explicitPhase)) {
                            parts.push_back(std::move(*explicitPhase));
                            return true;
                        }
                        for (auto* operation : child.operations) {
                            if (!index.phasesFor(operation).empty() && !appendArithmetic(operation)) {
                                return unavailable("explicit phase export: " + diagnostic);
                            }
                        }
                    } else if (!appendArithmetic(child.anchor)) {
                        return unavailable("compact phase export: " + diagnostic);
                    }
                    return true;
                };
                for (auto id : sequence.children) {
                    if (!appendChild(id)) {
                        compactFailure = repeatedAttempt;
                        invalidExpression = !expressions.constructionError().empty();
                        return std::nullopt;
                    }
                }
                // Empty selected arms still export exact absent selectors.
                SmallVector<scf::ForOp> protectionContext(node.loops.begin(), node.loops.end());
                protectionContext.push_back(outer);
                auto composed = composeRegionalSequenceWithin(function, arena, std::move(parts), true, false,
                                                               protectionContext);
                if (!composed.error.empty()) {
                    compactFailure = "phase body composition: " + composed.error;
                    invalidExpression = !expressions.constructionError().empty();
                    return std::nullopt;
                }
                composed.state->completeInvocation = false;
                composed.state->requiredOuterLoops.assign(node.loops.begin(), node.loops.end());
                composed.state->requiredOuterLoops.push_back(outer);
                auto view = sequenceRegionalResult(composed);
                if (!writersExported(view)) {
                    compactFailure = "discharged writer lacks an outer re-entry storage interface";
                    return std::nullopt;
                }
                compactTransaction.commit();
                return view;
            };
            auto selected = buildCompactBody();
            repeatedAttempt = previousAttempt;
            if (invalidExpression) { return unavailable(compactFailure); }
            if (!selected) {
                // A bounded numeric inner body is an explicit word, including
                // genuine moving subregions. Bind geometry only after the complete
                // periodic-map proof above; control receives independent interval
                // facts and actual enumerated inner coordinates, never the phase IV.
                auto finite = recognizeSpecializedNumericBody(outer, index, *input,
                    [&](Value value) -> std::optional<int64_t> {
                        auto bound = normalizer.atPhase(value, phase, period);
                        if (!bound) { return std::nullopt; }
                        auto literal = expressions.constantValue(*bound);
                        return literal ? std::optional<int64_t>(APInt(64, *literal).getSExtValue()) : std::nullopt;
                    }, [&](Value value) -> std::optional<bool> {
                        auto bound = boundaryGuard(value, normalizer, interval.bindings, expressions);
                        if (!bound) { return std::nullopt; }
                        auto literal = expressions.constantValue(*bound);
                        return literal ? std::optional<bool>(*literal != 0) : std::nullopt;
                    });
                if (finite.result.state != RecognitionState::Applicable) {
                    std::string reason = compactFailure + "; finite body expansion";
                    for (const auto& issue : finite.result.diagnostics) {
                        reason += " / " + recognitionName(issue.issue).str();
                    }
                    return unavailable(reason);
                }
                SmallVector<scf::ForOp> enclosing(node.loops.begin(), node.loops.end());
                enclosing.push_back(outer);
                std::string diagnostic;
                auto body = numericBodyRegionalResult(function, *input, finite, arena, enclosing, diagnostic);
                if (failed(body)) { return unavailable(compactFailure + "; finite phase body: " + diagnostic); }
                selected = std::move(*body);
            }
            auto view = std::move(*selected);
            if (!writersExported(view)) {
                return unavailable("discharged writer lacks an outer re-entry storage interface");
            }
            phases.push_back(std::move(view));
        }
        auto repeated = repeatPhasedRegions(function, outer, std::move(phases), interval.interval.end,
                                           node.loops, interval.interval.begin, interval.maximumLength);
        if (!repeated.error.empty()) { return unavailable(repeated.error); }
        intervalViews.push_back(std::move(repeated.regional));
    }
    if (intervalViews.empty()) { return unavailable("empty phase boundary partition"); }
    Child child;
    if (intervalViews.size() == 1) { child.regional = std::move(intervalViews.front()); }
    else {
        // Each interval has already connected every body SSA prerequisite.
        // The result-free outer loop has no carried bindings: an SSA producer
        // inside one visit cannot feed a different visit/interval. Repeating
        // those original-site edges here would invent cross-visit uses. Storage
        // and native boundary crossings are still constructed and reduced.
        auto composed = composeRegionalSequenceWithin(
            function, arena, std::move(intervalViews), false, false, node.loops);
        if (!composed.error.empty()) { return unavailable("outer slice composition: " + composed.error); }
        composed.state->completeInvocation = false;
        composed.state->requiredOuterLoops.assign(node.loops.begin(), node.loops.end());
        child.regional = sequenceRegionalResult(composed);
    }
    child.anchors = child.regional.anchors;
    children.push_back(std::move(child));
    transaction.commit();
    repeatedAttempt.clear();
    return true;
}
} // namespace mlir::pto::frontiersynch
