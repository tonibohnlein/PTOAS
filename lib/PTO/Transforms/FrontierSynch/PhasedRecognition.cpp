// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Compose original nested regions after proving their storage and counted
// domains repeat under one common outer phase period. Phase-specialized trip
// circuits govern queries, storage boundaries and endpoint filters together.
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
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
namespace mlir::pto::frontiersynch {
namespace {
// After joint storage certification, owned-cell translation is a graph
// isomorphism. Bind only the body's parameter context, protecting caller query
// coordinates with owned placeholders before substituting that context.
struct PhaseLogicalView {
    RegionalAnalysis original;
    Block symbols;
    SmallVector<std::pair<Expr, Expr>> parameters;
    SmallVector<Expr> coordinates;
    unsigned width = 1;
    PhaseLogicalView(RegionalAnalysis original, ArrayRef<std::pair<Expr, Expr>> parameters,
                     scf::ForOp outer)
        : original(std::move(original)), parameters(parameters.begin(), parameters.end())
    {
        for (const auto& loops : this->original.outerLoops) { width = std::max(width, unsigned(loops.size() + 1)); }
        for (unsigned i = 0; i < 2 * width; ++i) {
            coordinates.push_back(this->original.expressions->input(
                symbols.addArgument(IndexType::get(outer.getContext()), outer.getLoc())));
        }
    }
    RegionalEvent bind(RegionalEvent event, unsigned start, SmallVectorImpl<std::pair<Expr, Expr>>& values)
    {
        auto& e = *original.expressions;
        RegionExpressions::Substitution context(parameters);
        auto protect = [&](Expr& coordinate, unsigned position) {
            // Keep already invariant coordinates, especially literal explicit
            // ordinals, in the form understood by the child's query provider.
            if (e.substitute(coordinate, context) == coordinate) { return; }
            values.emplace_back(coordinates[position], coordinate);
            coordinate = coordinates[position];
        };
        protect(event.ordinal, start);
        for (unsigned i = 0; i < event.visits.size(); ++i) { protect(event.visits[i], start + i + 1); }
        return event;
    }
    std::optional<Expr> query(RegionalEvent a, std::optional<RegionalEvent> b, bool reference)
    {
        if (a.visits.size() >= width || (b && b->visits.size() >= width)) { return std::nullopt; }
        SmallVector<std::pair<Expr, Expr>> values;
        auto first = bind(a, 0, values);
        std::optional<Expr> result;
        if (!b) { result = regionalPresence(original, first); }
        else {
            auto second = bind(*b, width, values);
            result = reference ? regionalReferenceBefore(original, first, second) :
                                 regionalReachability(original, first, second);
        }
        if (!result) { return std::nullopt; }
        auto& e = *original.expressions;
        RegionExpressions::Substitution context(parameters), arguments(values);
        return e.substitute(e.substitute(*result, context), arguments);
    }
};
void bindLogicalPhase(RegionalAnalysis& view, ArrayRef<std::pair<Expr, Expr>> parameters, scf::ForOp outer)
{
    auto state = std::make_shared<PhaseLogicalView>(view, parameters, outer);
    view.presence = [state](RegionalEvent a) { return state->query(a, std::nullopt, false); };
    view.reachability = [state](RegionalEvent a, RegionalEvent b) { return state->query(a, b, false); };
    view.referenceBefore = [state](RegionalEvent a, RegionalEvent b) { return state->query(a, b, true); };
    RegionExpressions::Substitution bindings(parameters);
    auto& e = *view.expressions;
    auto rewrite = [&](RegionalSelector& value) {
        value.present = e.substitute(value.present, bindings);
        value.event.ordinal = e.substitute(value.event.ordinal, bindings);
        for (auto& visit : value.event.visits) { visit = e.substitute(visit, bindings); }
    };
    for (auto* side : {&view.firstPayloads, &view.lastPayloads, &view.firstSitePayloads}) {
        for (auto& [pipe, values] : *side) { for (auto& value : values) { rewrite(value); } }
    }
    for (auto& cell : view.storageBoundary) {
        for (auto* side : {&cell.firstWriters, &cell.lastWriters}) { for (auto& value : *side) { rewrite(value); } }
        for (auto* side : {&cell.firstReaders, &cell.lastReaders}) {
            for (auto& [pipe, values] : *side) { for (auto& value : values) { rewrite(value); } }
        }
    }
    for (auto* side : {&view.accessBoundary, &view.deferredAccessBoundary}) {
        for (auto& access : *side) { rewrite(access.first); rewrite(access.last); }
    }
    view.arithmeticRelations.reset(); view.relations.reset(); view.symbolicStorage.reset(); view.numerical.reset();
}
std::optional<Expr> phaseTripCount(scf::ForOp loop, PhaseNormalization& normalizer,
    RegionExpressions& expressions, uint64_t phase, uint64_t period,
    SmallVectorImpl<std::pair<Expr, Expr>>& bindings)
{
    auto domain = CountedLoop::get(loop);
    if (!domain) { return std::nullopt; }
    for (Value bound : {loop.getLowerBound(), loop.getUpperBound(), loop.getStep()}) {
        if (normalizer.independent(bound)) { continue; }
        // A congruence alone cannot replace a trip bound by a representative.
        if (!normalizer.periodic(bound, period)) { return std::nullopt; }
        auto value = normalizer.atPhase(bound, phase, period);
        if (!value) { return std::nullopt; }
        bindings.emplace_back(expressions.input(bound), *value);
    }
    RegionExpressions::Substitution substitution(bindings);
    return expressions.substitute(domain->trips(expressions), substitution);
}
void specialize(GuardedRotatingAnalysis& analysis, RegionExpressions::Substitution& bindings)
{
    rewriteGuardedRotatingExpressions(analysis, [&](Expr expression) {
        return analysis.expressions->substitute(expression, bindings);
    });
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
    bool nested = false, periodFits = true;
    SmallVector<Value> innerBounds;
    auto addPeriod = [&](uint64_t slots) {
        if (!slots) { periodFits = false; return; }
        auto factor = slots / std::gcd(period, slots);
        if (factor > maxRegionalSlotVisits / period) { periodFits = false; return; }
        period *= factor;
    };
    outer.getBody()->walk([&](Operation* operation) {
        if (auto inner = dyn_cast<scf::ForOp>(operation)) {
            nested = true;
            innerBounds.append({inner.getLowerBound(), inner.getUpperBound(), inner.getStep()});
        }
        if (operation->getNumResults() == 1 && index.isRelevant(operation->getResult(0)) &&
            !normalizer.independent(operation->getResult(0))) {
            if (auto divisor = PhaseNormalization::modulus(operation->getResult(0))) { addPeriod(*divisor); }
        }
    });
    if (!nested) {
        return unavailable("no nested body");
    }
    // Bound-only remainder values need not participate in a physical address.
    // Discover their periods before checking whole-value domain invariance.
    SmallVector<Value> pending(innerBounds.begin(), innerBounds.end());
    DenseSet<Value> seenBounds;
    while (!pending.empty()) {
        auto value = pending.pop_back_val();
        if (!seenBounds.insert(value).second || normalizer.independent(value)) { continue; }
        if (auto divisor = PhaseNormalization::modulus(value)) { addPeriod(*divisor); }
        auto* definition = value.getDefiningOp();
        if (definition && outer->isProperAncestor(definition)) {
            llvm::append_range(pending, definition->getOperands());
        }
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
        std::string guardedError;
        GuardedRotatingSpecialization request; request.loop = inner;
        auto cached = resolveOriginal.specializedGuarded ?
            resolveOriginal.specializedGuarded(request, arena, guardedError) : nullptr;
        if (resolveOriginal.specializedGuarded && !cached) { unavailable(guardedError); continue; }
        auto recognition = cached ? cached->recognition :
            recognizeGuardedRotating(inner, index, *input, input->accesses());
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
        auto analyzed = cached ? cached->demands :
            analyzeGuardedRotating(inner, *input, recognition, index, arena);
        const bool guardedFailed = !guardedError.empty() || !analyzed.error.empty();
        if (guardedFailed) {
            unavailable(guardedError.empty() ? analyzed.error : guardedError); continue;
        }
        rotating.emplace(id, std::move(analyzed));
    }
    if (!periodFits || (period <= 1 && !boundaryControl)) {
        return unavailable("no supported outer bank period or finite control boundary");
    }
    for (Value bound : innerBounds) {
        if (!normalizer.periodic(bound, period)) {
            return unavailable("nested bounds do not repeat under the common outer phase period");
        }
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
    bool evolvingStorage = !uniformDescriptors;
    for (const auto& effect : input->accesses().effects()) {
        if (!effect.phase || !outer->isProperAncestor(effect.phase->elementOp)) { continue; }
        const auto effectId = static_cast<std::size_t>(&effect - input->accesses().effects().data());
        if (detail::dischargeGlobalReadOnlyEffect(effectId, input->accesses())) { continue; }
        if (effect.selection && !normalizer.periodic(effect.selection->selector, period)) {
            evolvingStorage = true;
        }
        for (const auto& region : effect.regions) {
            if (!normalizer.periodic(region, period)) {
                evolvingStorage = true;
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
                            SequenceRegionResolver specialized;
                            specialized.specializedDemands = resolveOriginal.specializedDemands;
                            specialized.specializedNumeric = resolveOriginal.specializedNumeric;
                            specialized.specializedGuarded = resolveOriginal.specializedGuarded;
                            auto analyzed = analyzeSequenceRegionWithResolver(function, *input, *program, id,
                                arena, indexOwner, requireEndpoints, std::move(specialized));
                            if (!analyzed.error.empty()) {
                                return unavailable("invariant nested phase: " + analyzed.error);
                            }
                            parts.push_back(sequenceRegionalResult(analyzed));
                            return true;
                        }
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
                        auto parameterBinding = [&](Value parameter) -> std::optional<Expr> {
                            if (parameter.getType().isInteger(1)) {
                                if (auto bound = boundaryGuard(
                                        parameter, normalizer, interval.bindings, expressions)) {
                                    return bound;
                                }
                            }
                            // A true owned address must retain its actual
                            // outer coordinate. A later joint family proof
                            // justifies projecting its inter-visit effects.
                            if (evolvingStorage && !normalizer.periodic(parameter, period)) {
                                return expressions.input(parameter);
                            }
                            return normalizer.atPhase(parameter, phase, period);
                        };
                        FailureOr<RegionalAnalysis> view = failure();
                        if (resolveOriginal.specializedDemands) {
                            ArithmeticEntryConstant constants = [&](Value value) -> std::optional<int64_t> {
                                auto bound = parameterBinding(value);
                                auto literal = bound ? expressions.constantValue(*bound) : std::nullopt;
                                return literal ? std::optional<int64_t>(APInt(64, *literal).getSExtValue()) :
                                    std::nullopt;
                            };
                            auto mathematics = resolveOriginal.specializedDemands(
                                {function, root}, constants, diagnostic);
                            if (mathematics) {
                                view = exportSpecializedArithmeticRegion(
                                    *mathematics, arena, true, diagnostic, parameterBinding);
                            }
                        } else {
                            view = analyzeArithmeticRegionWithProfiles({function, root}, index, *input, arena,
                                diagnostic, program->regionalArithmeticProfiles, parameterBinding);
                        }
                        if (failed(view)) { return false; }
                        parts.push_back(std::move(*view));
                        return true;
                    };
                    if (evolvingStorage && child.kind == StructureKind::Loop) {
                        bool ownedMaps = false;
                        for (const auto& effect : input->accesses().effects()) {
                            if (!effect.phase || !child.anchor->isProperAncestor(effect.phase->elementOp)) { continue; }
                            for (const auto& region : effect.regions) {
                                ownedMaps |= !normalizer.periodic(region, period);
                            }
                        }
                        if (ownedMaps) {
                            if (!appendArithmetic(child.anchor)) {
                                return unavailable("owned nested phase export: " + diagnostic);
                            }
                            return true;
                        }
                    }
                    if (auto found = rotating.find(id); found != rotating.end() && interval.bindings.empty()) {
                        SmallVector<std::pair<Expr, Expr>> replacements;
                        auto phaseTrips = phaseTripCount(found->second.loop, normalizer, expressions,
                                                        phase, period, replacements);
                        if (!phaseTrips) {
                            return unavailable("phase-specialized nested counted-domain adapter unavailable");
                        }
                        const bool specializedDomain = !replacements.empty();
                        for (Value parameter : rotatingParameters[id]) {
                            auto inputExpression = expressions.input(parameter);
                            if (llvm::any_of(replacements, [&](const auto& binding) {
                                    return binding.first == inputExpression;
                                })) { continue; }
                            auto value = normalizer.residue(parameter, phase, period);
                            if (!value) { return unavailable("fixed-width bank residue normalization is unproved"); }
                            replacements.emplace_back(inputExpression, *value);
                        }
                        auto specialized = found->second;
                        RegionExpressions::Substitution substitution(replacements);
                        specialize(specialized, substitution);
                        // The surrounding phase supplies the domain equality.
                        // Keep original ordinal identities and cuts; every query
                        // and endpoint filter sees this same specialized length.
                        auto slice = specializedDomain ?
                            std::optional<PeriodicSlice>(PeriodicSlice{c(0), *phaseTrips}) :
                                                         std::nullopt;
                        auto view = guardedRotatingRegionalResult(function, *input, specialized, diagnostic, slice);
                        if (failed(view)) { return unavailable("child phase export: " + diagnostic); }
                        parts.push_back(std::move(*view));
                        return true;
                    }
                    if (auto inner = dyn_cast_or_null<scf::ForOp>(child.anchor);
                        child.kind == StructureKind::Loop && inner) {
                        SmallVector<std::pair<Expr, Expr>> domainBindings;
                        auto innerTrips = phaseTripCount(inner, normalizer, expressions, phase, period, domainBindings);
                        if (!innerTrips) {
                            return unavailable("nested sliced child requires a phase-specialized counted domain");
                        }
                        auto childSlices = collectBoundarySlices(
                            inner, index, expressions, *innerTrips, interval.bindings, diagnostic);
                        if (!childSlices) { return unavailable("nested boundary: " + diagnostic); }
                        std::vector<RegionalAnalysis> childParts;
                        for (auto& slice : *childSlices) {
                            RegionExpressions::Substitution domainSubstitution(domainBindings);
                            slice.interval.begin = expressions.substitute(slice.interval.begin, domainSubstitution);
                            slice.interval.end = expressions.substitute(slice.interval.end, domainSubstitution);
                            for (auto& binding : slice.bindings) {
                                binding.second = expressions.substitute(binding.second, domainSubstitution);
                            }
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
                            GuardedRotatingSpecialization request{inner, true, choices, guards, slice.bindings, arena};
                            auto cached = resolveOriginal.specializedGuarded ?
                                resolveOriginal.specializedGuarded(request, arena, diagnostic) : nullptr;
                            if (resolveOriginal.specializedGuarded && !cached) { return unavailable(diagnostic); }
                            auto recognized = cached ? cached->recognition :
                                detail::recognizeRotatingSlice(inner, index, *input, choices, guards);
                            if (recognized.result.state != RecognitionState::Applicable) {
                                std::string reason = "nested slice does not supply a rotating regional interface";
                                for (const auto& issue : recognized.result.diagnostics) {
                                    reason += " / " + recognitionName(issue.issue).str();
                                }
                                return unavailable(reason);
                            }
                            auto analyzed = cached ? cached->demands :
                                analyzeGuardedRotating(inner, *input, recognized, index, arena, slice.bindings);
                            const bool guardedFailed = !diagnostic.empty() || !analyzed.error.empty();
                            if (guardedFailed) {
                                return unavailable(diagnostic.empty() ? analyzed.error : diagnostic);
                            }
                            SmallVector<std::pair<Expr, Expr>> replacements(
                                domainBindings.begin(), domainBindings.end());
                            DenseSet<Value> seen;
                            for (const auto& access : recognized.result.accesses) {
                                for (Value parameter : access.parameters) {
                                    if (normalizer.independent(parameter) || !seen.insert(parameter).second) {
                                        continue;
                                    }
                                    auto inputExpression = expressions.input(parameter);
                                    if (llvm::any_of(replacements, [&](const auto& binding) {
                                            return binding.first == inputExpression;
                                        })) { continue; }
                                    auto value = normalizer.residue(parameter, phase, period);
                                    if (!value) {
                                        return unavailable("nested phase parameter lacks a residue certificate");
                                    }
                                    replacements.emplace_back(inputExpression, *value);
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
                            SequenceRegionResolver specialized;
                            specialized.specializedDemands = resolveOriginal.specializedDemands;
                            specialized.specializedNumeric = resolveOriginal.specializedNumeric;
                            specialized.specializedGuarded = resolveOriginal.specializedGuarded;
                            auto analyzed = analyzeSequenceRegionWithResolver(function, *input, *program, id,
                                arena, indexOwner, requireEndpoints, std::move(specialized));
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
                                    if (evolvingStorage && !runNormalizer.periodic(value, period)) {
                                        return std::nullopt;
                                    }
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
                if (!evolvingStorage && !writersExported(view)) {
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
                if (evolvingStorage) {
                    return unavailable(compactFailure + "; evolving phase requires an exact original-map adapter");
                }
                // A bounded numeric inner body is an explicit word, including
                // genuine moving subregions. Bind geometry only after the complete
                // periodic-map proof above; control receives independent interval
                // facts and actual enumerated inner coordinates, never the phase IV.
                TemplateGeometryConstant geometry = [&](Value value) -> std::optional<int64_t> {
                    auto bound = normalizer.atPhase(value, phase, period);
                    if (!bound) { return std::nullopt; }
                    auto literal = expressions.constantValue(*bound);
                    return literal ? std::optional<int64_t>(APInt(64, *literal).getSExtValue()) : std::nullopt;
                };
                TemplateControlConstant control = [&](Value value) -> std::optional<bool> {
                    auto bound = boundaryGuard(value, normalizer, interval.bindings, expressions);
                    if (!bound) { return std::nullopt; }
                    auto literal = expressions.constantValue(*bound);
                    return literal ? std::optional<bool>(*literal != 0) : std::nullopt;
                };
                std::string diagnostic;
                std::shared_ptr<const NumericBodyMathematics> mathematics;
                if (resolveOriginal.specializedNumeric) {
                    auto normalized = resolveOriginal.normalized ? resolveOriginal.normalized(
                        static_cast<std::size_t>(&node - program->nodes.data())) : nullptr;
                    mathematics = resolveOriginal.specializedNumeric(
                        outer, geometry, control, std::move(normalized), diagnostic);
                } else {
                    auto finite = recognizeSpecializedNumericBody(outer, index, *input, geometry, control, {},
                        resolveOriginal.normalized ? resolveOriginal.normalized(
                            static_cast<std::size_t>(&node - program->nodes.data())) : nullptr);
                    auto demands = analyzeNumericBody(finite, diagnostic);
                    if (demands) {
                        mathematics = std::make_shared<const NumericBodyMathematics>(
                            NumericBodyMathematics{std::move(finite), std::move(demands), {}, {}, {}});
                    }
                }
                if (!mathematics) {
                    return unavailable(compactFailure + "; finite body expansion: " + diagnostic);
                }
                SmallVector<scf::ForOp> enclosing(node.loops.begin(), node.loops.end());
                enclosing.push_back(outer);
                auto body = exportNumericBody(function, *input, *mathematics, arena, enclosing, diagnostic);
                if (failed(body)) { return unavailable(compactFailure + "; finite phase body: " + diagnostic); }
                selected = std::move(*body);
            }
            auto view = std::move(*selected);
            if (!evolvingStorage && !writersExported(view)) {
                return unavailable("discharged writer lacks an outer re-entry storage interface");
            }
            phases.push_back(std::move(view));
        }
        std::shared_ptr<const RepeatedStorageTypesResult> storage;
        if (evolvingStorage) {
            for (auto& phase : phases) {
                llvm::append_range(phase.accessBoundary, phase.deferredAccessBoundary);
                phase.deferredAccessBoundary.clear();
            }
            for (uint64_t phase = 0; phase < period; ++phase) {
                SmallVector<std::pair<Expr, Expr>> bindings;
                bool mapped = true;
                auto bind = [&](Value value) {
                    if (!value.getType().isIntOrIndex() || value.getType().isInteger(1) ||
                        normalizer.independent(value) || !index.isRelevant(value)) { return; }
                    auto representative = normalizer.atPhase(value, phase, period);
                    if (!representative) { mapped = false; return; }
                    bindings.emplace_back(expressions.input(value), *representative);
                };
                bind(outer.getInductionVar());
                outer.getBody()->walk([&](Operation* operation) {
                    for (Value value : operation->getResults()) { bind(value); }
                });
                if (!mapped) { return unavailable("owned phase logical parameter normalization unavailable"); }
                bindLogicalPhase(phases[phase], bindings, outer);
            }
            // The transformation is tentative until the joint proof succeeds.
            // Its physical reconstruction still consumes original shared maps;
            // retaining these canonical logical callbacks also prevents owner
            // byte queries from capturing one currently executing outer IV.
            auto proof = std::make_shared<RepeatedStorageTypesResult>(
                recognizeRepeatedStorageTypes(phases, outer, interval.interval.end, &index, period));
            if (!proof->error.empty()) { return unavailable("joint phase storage: " + proof->error); }
            storage = std::move(proof);
        }
        auto repeated = repeatPhasedRegions(function, outer, std::move(phases), interval.interval.end,
                                           node.loops, interval.interval.begin, interval.maximumLength,
                                           std::move(storage));
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
