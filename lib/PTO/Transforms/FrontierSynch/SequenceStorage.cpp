// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "SequenceAnalysisInternal.h"
#include "CountedLoop.h"
#include "PTO/Transforms/FrontierSynch/RotatingRegion.h"
#include "PTO/Transforms/FrontierSynch/VaryingRotatingRegional.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingRegional.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "llvm/ADT/MapVector.h"
namespace mlir::pto::frontiersynch {
std::optional<int64_t> sequenceInteger(Value value)
{
    APInt number;
    if (!matchPattern(value, m_ConstantInt(&number)) || !number.isSignedIntN(64)) { return std::nullopt; }
    return number.getSExtValue();
}
FailureOr<RegionalAnalysis> SequenceAnalysisState::originalRegion(std::size_t node, std::string& diagnostic)
{
    if (resolveOriginal) { return resolveOriginal(node, requireEndpoints, diagnostic); }
    auto analyzed = analyzeSequenceRegion(function, *input, *program, node, arena, indexOwner, requireEndpoints);
    if (!analyzed.error.empty()) { diagnostic = analyzed.error; return failure(); }
    return sequenceRegionalResult(analyzed);
}
bool SequenceAnalysisState::resolvedChild(std::size_t node)
{
    std::string diagnostic;
    auto regional = originalRegion(node, diagnostic);
    if (failed(regional)) { return fail(diagnostic); }
    Child child;
    child.regional = std::move(*regional);
    child.originalNode = node;
    child.anchors = child.regional.anchors;
    children.push_back(std::move(child));
    return true;
}
bool SequenceAnalysisState::explicitChild(const StructureNode& node)
{

    bool envelopes = llvm::any_of(node.payloads, [&](auto id) {
        return program->payloads[id].phase->macroOpInstanceId >= 0;
    });
    if (envelopes) {
        auto finite = analyzeFiniteGuarded(function, node.operations, index, *input, arena);
        if (!finite.error.empty()) { return fail(finite.error); }
        Child child;
        child.regional = finiteGuardedRegionalResult(finite);
        children.push_back(std::move(child));
        return true;
    }
    if (!node.explicitResult || node.explicitResult->state != RecognitionState::Applicable) {
        return fail("sequence explicit child has an unsupported control or prerequisite obligation");
    }
    SmallVector<const CompoundInstanceElement*> phases;
    for (auto id : node.payloads) { phases.push_back(program->payloads[id].phase); }
    if (phases.empty()) { return true; }
    Child child;
    child.trips = c(1);
    child.explicitAnalysis = analyzeExplicit(phases, index, *input, true);
    if (!child.explicitAnalysis.error.empty()) { return fail(child.explicitAnalysis.error); }
    child.dischargedEffects = child.explicitAnalysis.dischargedEffects;
    for (uint32_t type = 0; type < phases.size(); ++type) {
        auto* phase = phases[type];
        auto* op = phase->elementOp;
        child.anchors.push_back({phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
        for (auto id : input->accesses().effectsFor(phase)) {
            const auto& effect = input->accesses().effects()[id];
            if (llvm::is_contained(child.explicitAnalysis.dischargedEffects, id)) {
                // The explicit adapter already proved these phases conflict-free
                // against every other phase in the shared input.
                continue;
            }
            for (const auto& range : effect.ranges) {
                child.patterns.push_back({range, type, 0, 1,
                    effect.mode == SyncAccessMode::Read, effect.mode == SyncAccessMode::Write});
            }
        }
    }
    children.push_back(std::move(child));
    return true;
}
bool SequenceAnalysisState::numericPatterns(Child& child, const NumericTemplate& numeric)
{
    for (uint32_t type = 0; type < numeric.payloads.size(); ++type) {
        for (const auto& effect : numeric.payloads[type].effects) {
            if (effect.discharge != TemplateDischarge::None) {
                if (!llvm::is_contained(child.dischargedEffects, effect.sourceEffect)) {
                    child.dischargedEffects.push_back(effect.sourceEffect);
                }
                continue;
            }
            for (const auto& range : effect.ranges) {
                child.patterns.push_back({range, type, 0, 1,
                    effect.mode == SyncAccessMode::Read, effect.mode == SyncAccessMode::Write});
            }
        }
    }
    return true;
}
bool SequenceAnalysisState::rotatingPatterns(Child& child, const RotatingAnalysis& analysis,
                                const RecognitionResult& recognized)
{
    uint64_t slotVisits = 0;
    for (const auto& access : recognized.accesses) {
        if (access.slots > maxRegionalSlotVisits - slotVisits) {
            return fail("regional physical slot expansion exceeds the supported visit limit");
        }
        slotVisits += access.slots;
    }
    child.dischargedEffects = recognized.dischargedEffects;
    DenseMap<const CompoundInstanceElement*, uint32_t> types;
    for (uint32_t type = 0; type < analysis.phases.size(); ++type) { types[analysis.phases[type]] = type; }
    for (const auto& access : recognized.accesses) {
        if (!access.atom || !access.slots || access.parameterOffset || access.guard) {
            return fail("sequence rotating selectors require fixed slot families");
        }
        const auto& effect = input->accesses().effects()[access.effect];
        auto ranges = mlir::pto::detail::physicalSlotRanges(*input, *effect.memory);
        SmallVector<uint64_t> origins;
        if (access.firstPhysicalSlot) {
            for (uint64_t slot = 0; slot < access.slots; ++slot) {
                origins.push_back(access.firstPhysicalSlot->begin + slot * access.physicalSlotStride);
            }
        } else if (effect.selection) { origins = effect.selection->addresses; }
        else { for (const auto& range : ranges) { origins.push_back(range.begin); } }
        if (origins.size() != access.slots) { return fail("rotating slot addresses are not enumerable"); }
        auto period = access.slots / std::gcd(access.stride, access.slots);
        // The finite slot-table cost is explicit; no dependence-distance window
        // or runtime loop trip count is enumerated.
        for (uint64_t residue = 0; residue < period; ++residue) {
            ++child.costs.rotatingResidues;
            auto slot = (APInt(128, residue) * APInt(128, access.stride) + APInt(128, access.offset))
                            .urem(APInt(128, access.slots)).getZExtValue();
            auto base = origins[slot];
            if (access.atom->second > UINT64_MAX - base) { return fail("rotating physical range overflow"); }
            child.patterns.push_back({{effect.memory->scope, base + access.atom->first,
                                      base + access.atom->second}, types.lookup(effect.phase),
                                      residue, period, access.reads, access.writes});
        }
    }
    return true;
}
bool SequenceAnalysisState::loopChild(const StructureNode& node)
{
    repeatedAttempt.clear();
    Child child;
    child.loop = dyn_cast<scf::ForOp>(node.anchor);
    if (!child.loop || index.hasRelevantCarriedState(child.loop)) {
        return fail("sequence loop interface has relevant carried state");
    }
    auto domain = CountedLoop::get(child.loop);
    if (!domain) { return fail("sequence loop requires a representable positive-step ordinal domain"); }
    child.trips = domain->trips(expressions);
    // A proved empty domain needs no body interface.
    if (expressions.constantValue(child.trips) == 0) { return true; }
    // Each export attempt owns its candidate. Failed storage or endpoint
    // adapters cannot leave partial patterns in the child selected later.
    auto obligation = [&](const std::string& reason) {
        if (!repeatedAttempt.empty()) { repeatedAttempt += "; "; }
        repeatedAttempt += reason;
        error.clear();
    };
    if (node.rotatingResult && node.rotatingResult->state == RecognitionState::Applicable) {
        Child candidate;
        candidate.loop = child.loop;
        candidate.trips = child.trips;
        const auto id = static_cast<std::size_t>(&node - program->nodes.data());
        auto retained = resolveOriginal.demands ? resolveOriginal.demands(id, AnalysisBackend::Rotating) : nullptr;
        RotatingAnalysis analysis;
        if (retained && retained->rotatingDemands) {
            analysis = *retained->rotatingDemands;
            SmallVector<TemplateEndpointAnchor> anchors;
            for (auto* phase : analysis.phases) {
                auto* operation = phase->elementOp;
                anchors.push_back({phase, {}, {operation->getBlock(), operation},
                                   {operation->getBlock(), operation->getNextNode()}});
            }
            analysis.endpoints = bindPeriodicEndpoints(analysis.loop, anchors, analysis.periodic);
        } else if (resolveOriginal.demands) { analysis.error = "cached rotating demands unavailable"; }
        else { analysis = analyzeRotating(child.loop, index, *input, *node.rotatingResult); }
        if (!analysis.error.empty()) { obligation(analysis.error); }
        else if (!rotatingPatterns(candidate, analysis, *node.rotatingResult)) { obligation(error); }
        else if (!analysis.periodic.error.empty() || !analysis.endpoints.logical.error.empty()) {
            obligation("rotating child cannot export exact endpoints");
        } else {
            candidate.periodic = std::move(analysis.periodic);
            candidate.endpoints = std::move(analysis.endpoints);
            candidate.anchors = candidate.endpoints.anchors;
            children.push_back(std::move(candidate));
            return true;
        }
    }
    if (node.guardedRotatingResult &&
        node.guardedRotatingResult->result.state == RecognitionState::Applicable) {
        const auto id = static_cast<std::size_t>(&node - program->nodes.data());
        auto retained = resolveOriginal.demands ?
            resolveOriginal.demands(id, AnalysisBackend::GuardedRotating) : nullptr;
        std::shared_ptr<GuardedRotatingAnalysis> analysis;
        if (retained) { analysis = retained->guardedRotatingDemands; }
        else if (!resolveOriginal.demands) {
            analysis = std::make_shared<GuardedRotatingAnalysis>(
                analyzeGuardedRotating(child.loop, *input, *node.guardedRotatingResult, index, arena));
        }
        if (!analysis) { obligation("cached guarded rotating demands unavailable"); }
        else if (!analysis->error.empty()) { obligation(analysis->error); }
        else {
            std::string exportError;
            auto regional = guardedRotatingRegionalResult(function, *input, *analysis, exportError);
            if (failed(regional)) { obligation(exportError); }
            else {
                child.regional = std::move(*regional);
                child.anchors = child.regional.anchors;
                children.push_back(std::move(child));
                return true;
            }
        }
    }
    {
        if (node.boundedLifetime &&
            node.boundedLifetime->skeleton.result.state == RecognitionState::Applicable) {
            std::string diagnostic;
            auto bounded = cachedBoundedLifetimeRegion(function, node, index, *input, diagnostic);
            if (succeeded(bounded)) {
                node.boundedExportError =
                    "bounded demands available; global query/native/storage selectors require a regional provider";
            } else if (!diagnostic.empty()) { repeatedAttempt = diagnostic; }
        }
        // Existing numerical routes can already allocate physical IDs. Preserve
        // that capability for statically enumerable inner bodies, including
        // regional candidates with prologues/epilogues. Dynamic inner bounds do
        // not trigger expansion and go directly to repetition.
        bool staticInnerBounds = true;
        DenseMap<Value, bool> numericDependencies;
        std::function<bool(Value)> mayEnumerate = [&](Value value) {
            if (auto found = numericDependencies.find(value); found != numericDependencies.end()) {
                return found->second;
            }
            numericDependencies[value] = false;
            if (auto argument = dyn_cast<BlockArgument>(value)) {
                auto inner = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
                return numericDependencies[value] = inner && child.loop->isProperAncestor(inner) &&
                    argument == inner.getInductionVar();
            }
            auto* definition = value.getDefiningOp();
            return numericDependencies[value] = definition && !definition->getNumRegions() &&
                llvm::all_of(definition->getOperands(), mayEnumerate);
        };
        child.loop.getBody()->walk([&](scf::ForOp inner) {
            staticInnerBounds &= mayEnumerate(inner.getLowerBound()) &&
                                 mayEnumerate(inner.getUpperBound()) && mayEnumerate(inner.getStep());
        });
        std::shared_ptr<const MathematicalResult> retainedNumeric;
        if (resolveOriginal.demands) {
            const auto id = static_cast<std::size_t>(&node - program->nodes.data());
            retainedNumeric = resolveOriginal.demands(id, AnalysisBackend::NumericalPeriodic);
        }
        std::optional<NumericTemplate> cachedNumeric;
        const bool originalNumeric = staticInnerBounds && node.numericTemplate &&
            !node.numericTemplate->specializedBody &&
            node.numericTemplate->result.state == RecognitionState::Applicable;
        if (retainedNumeric && retainedNumeric->numericalDemands) {
            cachedNumeric = retainedNumeric->numericalDemands->form;
        } else if (staticInnerBounds && !resolveOriginal.demands) {
            // A successful whole-invocation template already proves the
            // regional context: no external payload can conflict with it.
            // Preserve its original-coordinate mapping and expanded word.
            cachedNumeric = originalNumeric ? *node.numericTemplate :
                recognizeRegionalNumericTemplate(child.loop, index, *input);
        }
        if (!cachedNumeric || cachedNumeric->result.state != RecognitionState::Applicable) {
            auto rotating = recognizeRotatingRegion(function, *input, *program,
                static_cast<std::size_t>(&node - program->nodes.data()), index, arena);
            if (rotating.error.empty()) {
                child.regional = std::move(rotating.regional); child.anchors = child.regional.anchors;
                children.push_back(std::move(child)); return true;
            }
            if (!repeatedAttempt.empty()) { repeatedAttempt += "; "; }
            repeatedAttempt += "rotating compact child: " + rotating.error;
        }
        if ((!cachedNumeric || cachedNumeric->result.state != RecognitionState::Applicable) &&
            repeatedChild(node, child.trips)) { return true; }
        if ((!cachedNumeric || cachedNumeric->result.state != RecognitionState::Applicable) &&
            phasedChild(node, child.trips)) { return true; }
        if (node.varyingRotating && node.varyingRotating->result.state == RecognitionState::Applicable) {
            std::string diagnostic;
            const auto id = static_cast<std::size_t>(&node - program->nodes.data());
            FailureOr<RegionalAnalysis> regional = failure();
            if (resolveOriginal.exports) {
                regional = resolveOriginal.exports(id, AnalysisBackend::VaryingBoundary, diagnostic);
            } else {
                regional = varyingRotatingRegionalResult(function, *node.varyingRotating, index, *input, arena,
                    diagnostic, node.varyingDemands ? &*node.varyingDemands : nullptr);
            }
            if (succeeded(regional)) {
                child.regional = std::move(*regional); child.anchors = child.regional.anchors;
                children.push_back(std::move(child)); return true;
            }
            obligation(diagnostic);
        }
        if (boundaryLoop(child.loop)) { return true; }
        // Reuse even a failed cached recognition; diagnostics must not rerun it.
        if (!cachedNumeric) {
            if (resolveOriginal.demands) {
                cachedNumeric.emplace();
                cachedNumeric->result.note(RecognitionIssue::TemplateContext, child.loop, true);
            } else { cachedNumeric = recognizeRegionalNumericTemplate(child.loop, index, *input); }
        }
        auto numeric = std::move(*cachedNumeric);
        if (numeric.result.state != RecognitionState::Applicable) {
            std::string arithmeticError;
            FailureOr<RegionalAnalysis> regional = failure();
            if (resolveOriginal.exports) {
                const auto id = static_cast<std::size_t>(&node - program->nodes.data());
                regional = resolveOriginal.exports(id, AnalysisBackend::Arithmetic, arithmeticError);
            } else {
                regional = analyzeArithmeticRegionWithProfiles({function, child.loop}, index, *input, arena,
                    arithmeticError, program->regionalArithmeticProfiles);
            }
            if (succeeded(regional)) {
                if (node.boundedDemands && child.loop->getParentOp() == function) {
                    // Both constructions derive their modeled graph from this
                    // original loop, input, guards and prerequisite index. The
                    // arithmetic provider supplies global queries/selectors;
                    // bounded local rows are never used as global queries.
                    auto demands = node.boundedDemands;
                    regional->cost.physicalFragments += demands->window.accesses.size();
                    regional->cost.retainedExpressionNodes += demands->expressions.size();
                    regional->cost.selectorComparisons += demands->analysis.accessPairs;
                    auto originalPrepare = regional->prepareWithVisits;
                    regional->prepareWithVisits = [demands, originalPrepare](ArrayRef<scf::ForOp> enclosing)
                        -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
                        // Preserve a working arithmetic recipe and its regional
                        // allocation certificate. Bounded recipes fill endpoint
                        // gaps; they do not replace successful physical exports.
                        if (originalPrepare) {
                            auto prepared = originalPrepare(enclosing);
                            if (succeeded(prepared)) { return prepared; }
                        }
                        if (!enclosing.empty()) { return failure(); }
                        std::string diagnostic;
                        return prepareBoundedLifetimeResult(demands, diagnostic);
                    };
                    regional->prepare = [prepare = regional->prepareWithVisits]() { return prepare({}); };
                    node.boundedExportError.clear();
                }
                child.regional = std::move(*regional);
                child.anchors = child.regional.anchors;
                children.push_back(std::move(child));
                error.clear();
                return true;
            }
            if (!arithmeticError.empty()) {
                if (!repeatedAttempt.empty()) { repeatedAttempt += "; "; }
                repeatedAttempt += "regional arithmetic: " + arithmeticError;
            }
            if (!repeatedAttempt.empty()) { repeatedAttempt += "; "; }
            repeatedAttempt += "numeric repeat recognition: exact regional template unavailable";
            for (const auto& diagnostic : numeric.result.diagnostics) {
                repeatedAttempt += " / " + recognitionName(diagnostic.issue).str();
            }
            return fail(repeatedAttempt);
        }
        child.costs.numericVisits = numeric.countedVisits;
        if (!numericPatterns(child, numeric)) { return false; }
        if (retainedNumeric && retainedNumeric->numericalDemands) {
            const auto& owner = retainedNumeric->numericalDemands;
            child.periodic = owner->analysis;
            child.numericTemplate = std::shared_ptr<const NumericTemplate>(owner, &owner->form);
        } else {
            child.periodic = originalNumeric && node.periodicAnalysis ? *node.periodicAnalysis :
                analyzeNumericTemplate(numeric);
            child.numericTemplate = std::make_shared<const NumericTemplate>(std::move(numeric));
        }
        auto anchors = numericTemplateAnchors(*child.numericTemplate, child.periodic, error);
        if (failed(anchors)) { return false; }
        child.anchors = std::move(*anchors);
    }
    if (!child.periodic.error.empty() || !child.endpoints.logical.error.empty()) {
        return fail("sequence periodic child cannot export exact endpoints");
    }
    if (!child.numericTemplate) { child.anchors = child.endpoints.anchors; }
    children.push_back(std::move(child));
    return true;
}
bool SequenceAnalysisState::collect(std::size_t rootNode)
{
    if ((!indexReady && failed(index.build(function, *input))) || program->nodes.empty()) {
        return fail("sequence structure unavailable");
    }
    indexReady = true;
    if (rootNode >= program->nodes.size()) { return fail("regional sequence node is invalid"); }
    const auto& root = program->nodes[rootNode];
    if (root.kind == StructureKind::ExplicitRun) { return explicitChild(root); }
    if (root.kind == StructureKind::Loop) { return loopChild(root); }
    if (root.kind == StructureKind::Conditional) { return conditionalChild(root); }
    if (root.kind != StructureKind::Sequence || !root.region ||
        (root.region->getParentOp() != function && !function->isProperAncestor(root.region->getParentOp()))) {
        return fail("regional sequence must belong to the original function");
    }
    for (std::size_t position = 0; position < root.children.size();) {
        const auto& node = program->nodes[root.children[position]];
        if (node.kind == StructureKind::Loop) {
            const bool accepted = resolveOriginal ? resolvedChild(root.children[position]) : loopChild(node);
            if (!accepted) { return false; }
            ++position; continue;
        }
        auto containsLoop = [](const StructureNode& child) {
            bool found = false;
            if (child.kind == StructureKind::Conditional) {
                child.anchor->walk([&](scf::ForOp) { found = true; return WalkResult::interrupt(); });
            }
            return found;
        };
        if (containsLoop(node)) {
            const bool accepted = resolveOriginal ? resolvedChild(root.children[position]) : conditionalChild(node);
            if (!accepted) { return false; }
            ++position; continue;
        }
        SmallVector<Operation*> roots;
        auto end = position;
        bool guarded = false;
        while (end < root.children.size()) {
            const auto& child = program->nodes[root.children[end]];
            if (child.kind == StructureKind::ExplicitRun) {
                roots.append(child.operations.begin(),child.operations.end());
            } else if (child.kind == StructureKind::Conditional) {
                if (containsLoop(child)) { break; }
                roots.push_back(child.anchor); guarded = true;
            } else { break; }
            ++end;
        }
        if (end == position) { return fail("sequence child lacks an exact regional interface"); }
        if (guarded) {
            auto analysis = analyzeFiniteGuarded(function,roots,index,*input,arena);
            if (!analysis.error.empty()) { return fail(analysis.error); }
            Child child;
            child.regional = finiteGuardedRegionalResult(analysis);
            child.anchors = child.regional.anchors;
            children.push_back(std::move(child));
        } else {
            StructureNode fused;
            fused.kind = StructureKind::ExplicitRun;
            fused.operations = roots;
            fused.explicitResult = RecognitionResult{};
            for (auto current = position; current < end; ++current) {
                const auto& part = program->nodes[root.children[current]];
                if (!part.explicitResult || part.explicitResult->state != RecognitionState::Applicable) {
                    return fail("adjacent explicit child has an unmet form obligation");
                }
                llvm::append_range(fused.payloads, part.payloads);
            }
            if (!explicitChild(fused)) { return false; }
        }
        position = end;
    }
    if (children.size() > UINT32_MAX) { return fail("sequence child identity overflow"); }
    return true;
}
bool SequenceAnalysisState::partition()
{
    std::map<AddressSpace, llvm::MapVector<Value, std::map<uint64_t, int64_t>>> changes;
    SmallVector<SyncStorageCell> identities;
    for (const auto& child : children) {
        for (const auto& boundary : child.regional.storageBoundary) { identities.push_back(boundary.cell); }
        for (const auto& pattern : child.patterns) {
            ++costs.physicalFragments;
            identities.push_back(pattern.range);
            if (pattern.range.begin > pattern.range.end) { return fail("invalid physical interval"); }
            if (pattern.range.begin == pattern.range.end) { continue; }
            ++changes[pattern.range.space][pattern.range.base][pattern.range.begin];
            --changes[pattern.range.space][pattern.range.base][pattern.range.end];
        }
    }
    for (const auto& [space, bases] : changes) {
        for (const auto& [base, points] : bases) {
            int64_t active = 0;
            uint64_t previous = 0;
            for (auto [point, delta] : points) {
                if (active > 0 && point > previous) { cells.push_back({space, previous, point, base}); }
                active += delta;
                previous = point;
            }
        }
    }
    if (cells.size() > UINT32_MAX) { return fail("sequence physical cell identity overflow"); }
    return true;
}

void SequenceAnalysisState::summarize()
{
    boundaries.resize(children.size(), std::vector<CellBoundary>(cells.size()));
    for (uint32_t childId = 0; childId < children.size(); ++childId) {
        auto& child = children[childId];
        if (child.regional.presence) { continue; }
        const auto comparisonsBefore = costs.selectorComparisons;
        for (uint32_t cellId = 0; cellId < cells.size(); ++cellId) {
            const auto& cell = cells[cellId];
            auto& summary = boundaries[childId][cellId];
            std::vector<Selected> firstWrites, lastWrites, firstReads, lastReads;
            for (const auto& pattern : child.patterns) {
                if (!sameStorageDomain(pattern.range, cell) || pattern.range.begin > cell.begin ||
                    pattern.range.end < cell.end) { continue; }
                Expr exists = expressions.lt(c(pattern.residue), child.trips);
                auto last = expressions.sub(child.trips, c(1));
                last = expressions.sub(last, expressions.rem(expressions.sub(last, c(pattern.residue)),
                                                               c(pattern.period)));
                Selected first{port(childId, pattern.type, c(pattern.residue)), exists};
                Selected final{port(childId, pattern.type, last), exists};
                if (pattern.write) { firstWrites.push_back(first); lastWrites.push_back(final); }
                if (pattern.read) { firstReads.push_back(first); lastReads.push_back(final); }
            }
            auto extremal = [&](const std::vector<Selected>& candidates, bool first) {
                std::vector<Selected> result;
                for (auto selected : candidates) {
                    for (auto other : candidates) {
                        auto earlier = first ? before(other.port, selected.port) : before(selected.port, other.port);
                        selected.present = both(selected.present, negate(both(other.present, earlier)));
                    }
                    result.push_back(selected);
                }
                return result;
            };
            summary.firstWriters = extremal(firstWrites, true);
            summary.lastWriters = extremal(lastWrites, false);
            std::map<uint32_t, std::vector<Selected>> readsByPipe;
            for (auto selected : firstReads) {
                for (auto writer : firstWrites) {
                    selected.present = both(selected.present,
                        negate(both(writer.present, negate(before(selected.port, writer.port)))));
                }
                readsByPipe[pipe(selected.port)].push_back(selected);
            }
            for (const auto& [p, reads] : readsByPipe) { summary.firstReaders[p] = extremal(reads, true); }
            readsByPipe.clear();
            for (auto selected : lastReads) {
                for (auto writer : lastWrites) {
                    selected.present = both(selected.present,
                        negate(both(writer.present, negate(before(writer.port, selected.port)))));
                }
                readsByPipe[pipe(selected.port)].push_back(selected);
            }
            for (const auto& [p, reads] : readsByPipe) { summary.lastReaders[p] = extremal(reads, false); }
        }
        // Native boundary ports are required even for payloads without effects.
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> positions;
        for (uint32_t type = 0; type < child.anchors.size(); ++type) {
            auto p = static_cast<uint32_t>(child.anchors[type].phase->kPipeValue);
            auto [it, added] = positions.emplace(p, std::make_pair(type, type));
            it->second.second = type;
        }
        for (const auto& [p, pair] : positions) {
            port(childId, pair.first, c(0));
            port(childId, pair.second, expressions.sub(child.trips, c(1)));
        }
        child.costs.selectorComparisons = costs.selectorComparisons - comparisonsBefore;
    }
}
} // namespace mlir::pto::frontiersynch
