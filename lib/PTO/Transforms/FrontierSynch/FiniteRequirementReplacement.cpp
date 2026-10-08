// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FiniteRequirementReplacement.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include <algorithm>
#include <limits>
#include <map>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Expr = RegionExpressions::Id;
using Matrix = std::vector<std::vector<Expr>>;
bool boolean(const RegionExpressions& arena, Expr value)
{
    return value < arena.size() && arena.isBoolean(value);
}
bool eventValid(const RegionExpressions& arena, std::size_t count, const RegionalEvent& event)
{
    return event.type < count && event.visits.empty() && event.ordinal < arena.size() &&
           !arena.isBoolean(event.ordinal) && (event.kind == PeriodicEventKind::Start ||
                                               event.kind == PeriodicEventKind::Completion);
}
void add(Matrix& graph, std::size_t source, std::size_t target, Expr guard, RegionExpressions& arena)
{
    graph[source][target] = arena.lor(graph[source][target], guard);
}
void close(Matrix& graph, RegionExpressions& arena, uint64_t& updates)
{
    // Boolean Floyd on the finite event frame. Native/reflexive paths and ALL
    // original group memberships are inserted before closure, so a reduced or
    // strengthened upper graph cannot accidentally become the proof's baseline.
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t a = 0; a < graph.size(); ++a) {
            for (std::size_t b = 0; b < graph.size(); ++b) {
                graph[a][b] = arena.lor(graph[a][b], arena.land(graph[a][k], graph[k][b]));
                ++updates;
            }
        }
    }
}
bool sameOrigin(const RequirementSnapshot& original, const RequirementSnapshot& snapshot)
{
    return snapshot && snapshot->context() == original->context() &&
        snapshot->originalMemberships().data() == original->originalMemberships().data() &&
        snapshot->originalMemberships().size() == original->originalMemberships().size() &&
        snapshot->groups().data() == original->groups().data() &&
        snapshot->groups().size() == original->groups().size();
}
} // namespace
FiniteRequirements captureFiniteRequirements(func::FuncOp function, const SyncInput& input,
    llvm::ArrayRef<const CompoundInstanceElement*> phases, std::shared_ptr<RegionExpressions> expressions,
    uint64_t producer, llvm::ArrayRef<RequirementGroupId> atomGroups,
    RequirementGroupId prerequisiteGroup, std::string& error)
{
    error.clear();
    if (!function || !expressions || !expressions->constructionError().empty() ||
        atomGroups.size() != input.accesses().cells().size()) {
        error = "finite requirements need a valid arena and a group for every shared cell";
        return {};
    }
    for (const auto* phase : phases) {
        if (!phase || !phase->elementOp || phase->elementOp->getParentOfType<func::FuncOp>() != function) {
            error = "finite requirement occurrence is outside the supplied function";
            return {};
        }
        for (auto* parent = phase->elementOp->getParentOp(); parent && parent != function.getOperation();
             parent = parent->getParentOp()) {
            if (isa<LoopLikeOpInterface>(parent)) {
                error = "finite requirement frame needs an enclosing-visit adapter for repeated occurrences";
                return {};
            }
        }
    }
    PhaseIndex index;
    if (failed(index.build(function, input))) {
        error = "finite requirements could not index the unchanged shared input";
        return {};
    }
    auto result = std::shared_ptr<FiniteRequirementFrame>(new FiniteRequirementFrame());
    result->analyzed = analyzeExplicit(phases, index, input);
    if (!result->analyzed.error.empty()) { error = result->analyzed.error; return {}; }
    const auto mapped = index.mapPrerequisites(phases);
    if (!mapped.error.empty()) { error = mapped.error; return {}; }
    result->nativeEdges = mapped.native;
    RegionalAnalysis domain;
    domain.expressions = expressions;
    domain.accessModel = &input.accesses();
    domain.gmAliasPolicy = input.memory().gmPolicy();
    domain.occurrenceLoops.resize(phases.size());
    for (const auto* phase : phases) {
        TemplateEndpointAnchor anchor;
        anchor.phase = phase;
        anchor.before = {phase->elementOp->getBlock(), phase->elementOp};
        anchor.after = {phase->elementOp->getBlock(), phase->elementOp->getNextNode()};
        domain.anchors.push_back(std::move(anchor));
    }
    const auto count = phases.size();
    domain.presence = [expressions, count](RegionalEvent event) -> std::optional<Expr> {
        if (!eventValid(*expressions, count, event)) { return std::nullopt; }
        return expressions->eq(event.ordinal, expressions->constant(0));
    };
    domain.referenceBefore = [expressions, count](RegionalEvent a, RegionalEvent b) -> std::optional<Expr> {
        if (!eventValid(*expressions, count, a) || !eventValid(*expressions, count, b)) { return std::nullopt; }
        return expressions->land(expressions->boolean(a.type < b.type), expressions->land(
            expressions->eq(a.ordinal, expressions->constant(0)),
            expressions->eq(b.ordinal, expressions->constant(0))));
    };
    auto owner = captureRegionalOrderContext(input, domain, error);
    if (failed(owner)) { return {}; }
    result->provenance = captureStorageRequirementProvenance(*owner, producer, result->analyzed.scan,
        atomGroups, prerequisiteGroup, error);
    return result->provenance ? result : FiniteRequirements{};
}
FiniteReplacementProof certifyFiniteRequirementReplacement(FiniteRequirements frame,
    RequirementSnapshot snapshot, RequirementGroupId group, Expr beta,
    std::vector<FiniteReplacementRecord> candidate)
{
    FiniteReplacementProof result;
    if (!frame || !sameOrigin(frame->original(), snapshot)) {
        result.error = "finite replacement requires its certified original ownership and context";
        return result;
    }
    auto& arena = *snapshot->context()->expressions();
    if (!arena.constructionError().empty() || !boolean(arena, beta)) {
        result.error = "finite replacement domain is not a valid Boolean expression";
        return result;
    }
    const auto groups = snapshot->groups();
    const auto selected = std::find_if(groups.begin(), groups.end(),
        [group](const auto& entry) { return entry.id == group; });
    if (selected == groups.end() || selected->scope != RequirementScope::Internal) {
        result.error = "finite replacement only certifies original internal storage groups";
        return result;
    }
    const auto& analysis = frame->analysis();
    const auto n = analysis.occurrences.size();
    const auto limit = std::numeric_limits<uint64_t>::max();
    if (n > std::numeric_limits<std::size_t>::max() / 2) {
        result.error = "finite event count exceeds representation"; return result;
    }
    const auto vertices = n * 2;
    if (vertices && (vertices > limit / vertices || vertices * vertices > limit / vertices / 2)) {
        result.error = "finite closure work count exceeds representation"; return result;
    }
    std::set<std::pair<uint64_t, uint64_t>> keys;
    for (const auto& member : snapshot->originalMemberships()) {
        keys.emplace(member.record.producer, member.record.originalRecord);
    }
    for (const auto& member : snapshot->memberships()) {
        keys.emplace(member.record.producer, member.record.originalRecord);
    }
    for (const auto& member : snapshot->lowerFacts()) {
        keys.emplace(member.record.producer, member.record.originalRecord);
    }
    for (const auto& record : candidate) {
        if (!keys.emplace(record.key.producer, record.key.originalRecord).second ||
            record.edge.source >= record.edge.target || record.edge.target >= n || !boolean(arena, record.guard)) {
            result.error = "finite replacement needs fresh unique keys, forward edges and Boolean guards";
            return result;
        }
    }
    RegionExpressions::Transaction transaction(arena);
    const auto yes = arena.boolean(true), no = arena.boolean(false);
    Matrix original(vertices, std::vector<Expr>(vertices, no));
    std::map<uint32_t, uint32_t> previous;
    for (uint32_t i = 0; i < n; ++i) {
        original[2 * i][2 * i] = yes;
        original[2 * i + 1][2 * i + 1] = yes;
        original[2 * i][2 * i + 1] = yes;
        auto found = previous.find(analysis.occurrences[i].pipe);
        if (found != previous.end()) {
            original[2 * found->second][2 * i] = yes;
            original[2 * found->second + 1][2 * i + 1] = yes;
        }
        previous[analysis.occurrences[i].pipe] = i;
    }
    for (const auto& edge : frame->native()) { original[2 * edge.source + 1][2 * edge.target] = yes; }
    std::set<RequirementGroupId> fixed;
    for (const auto& entry : groups) {
        if (entry.scope == RequirementScope::FixedPrerequisite) { fixed.insert(entry.id); }
    }
    const auto& generators = analysis.scan.generators;
    for (const auto& member : snapshot->originalMemberships()) {
        if (!fixed.count(member.group)) { continue; }
        const auto& edge = generators[member.record.originalRecord];
        add(original, 2 * edge.source + 1, 2 * edge.target, member.guard, arena);
    }
    auto replacement = original;
    for (const auto& member : snapshot->originalMemberships()) {
        if (member.group != group) { continue; }
        const auto& edge = generators[member.record.originalRecord];
        add(original, 2 * edge.source + 1, 2 * edge.target, member.guard, arena);
    }
    for (const auto& record : candidate) {
        add(replacement, 2 * record.edge.source + 1, 2 * record.edge.target, record.guard, arena);
    }
    close(original, arena, result.closureUpdates);
    close(replacement, arena, result.closureUpdates);
    for (std::size_t a = 0; a < vertices; ++a) {
        for (std::size_t b = 0; b < vertices; ++b) {
            const auto equal = arena.land(arena.lor(arena.lnot(original[a][b]), replacement[a][b]),
                                          arena.lor(arena.lnot(replacement[a][b]), original[a][b]));
            ++result.equalityChecks;
            if (!arena.implies(beta, equal)) {
                result.error = "finite candidate closure is not proved equal to all original group requirements";
                return result;
            }
        }
    }
    if (!arena.constructionError().empty()) { result.error = arena.constructionError(); return result; }
    std::vector<RequirementExactRecord> records;
    for (const auto& record : candidate) { records.push_back({record.key, record.guard}); }
    result.evidence = bindTrustedRequirementReplacement(snapshot, group, beta, std::move(records), result.error);
    if (!result.evidence) { return result; }
    result.frame = std::move(frame);
    result.records = std::make_shared<const std::vector<FiniteReplacementRecord>>(std::move(candidate));
    transaction.commit();
    return result;
}
FiniteSelection applyCertifiedFiniteReplacement(FiniteRequirements frame, FiniteSelection previous,
    RequirementGroupId group, Expr beta, std::vector<FiniteReplacementRecord> candidate, std::string& error)
{
    error.clear();
    if (!frame || (previous && previous->frame() != frame)) {
        error = "finite selected requirements have a foreign original frame"; return {};
    }
    auto snapshot = previous ? previous->snapshot() : frame->original();
    auto proof = certifyFiniteRequirementReplacement(frame, snapshot, group, beta, std::move(candidate));
    if (!proof.evidence) { error = proof.error; return {}; }
    auto selected = applyRequirementReplacement(snapshot, proof.evidence, error);
    if (!selected) { return {}; }
    auto result = std::shared_ptr<FiniteRequirementSelection>(new FiniteRequirementSelection());
    result->origin = std::move(frame); result->selected = std::move(selected);
    if (previous) { result->records = previous->records; }
    if (proof.records->size() > result->records.max_size() - result->records.size()) {
        error = "finite selected definition count exceeds representation"; return {};
    }
    result->records.insert(result->records.end(), proof.records->begin(), proof.records->end());
    return result;
}
} // namespace mlir::pto::frontiersynch
