// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent Boolean adjacency oracle for concrete original-scan certificates.
#include "PTO/Transforms/FrontierSynch/FiniteRequirementReplacement.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Bits = std::vector<std::vector<bool>>;
Bits native(const fs::FiniteRequirementFrame& frame)
{
    const auto& occurrences = frame.analysis().occurrences;
    Bits result(2 * occurrences.size(), std::vector<bool>(2 * occurrences.size(), false));
    for (std::size_t a = 0; a < occurrences.size(); ++a) {
        result[2 * a][2 * a] = result[2 * a + 1][2 * a + 1] = result[2 * a][2 * a + 1] = true;
        for (std::size_t b = a + 1; b < occurrences.size(); ++b) {
            if (occurrences[a].pipe == occurrences[b].pipe) {
                result[2 * a][2 * b] = result[2 * a + 1][2 * b + 1] = true;
            }
        }
    }
    for (const auto& edge : frame.native()) { result[2 * edge.source + 1][2 * edge.target] = true; }
    std::set<fs::RequirementGroupId> fixed;
    for (const auto& group : frame.original()->groups()) {
        if (group.scope == fs::RequirementScope::FixedPrerequisite) { fixed.insert(group.id); }
    }
    for (const auto& member : frame.original()->originalMemberships()) {
        if (fixed.count(member.group)) {
            const auto& edge = frame.analysis().scan.generators[member.record.originalRecord];
            result[2 * edge.source + 1][2 * edge.target] = true;
        }
    }
    return result;
}
Bits closure(const Bits& edges)
{
    // Per-source graph search, independent of production's guarded Floyd DAG.
    Bits result = edges;
    for (std::size_t source = 0; source < edges.size(); ++source) {
        std::vector<std::size_t> todo{source};
        std::vector<bool> visited(edges.size(), false);
        visited[source] = true;
        while (!todo.empty()) {
            const auto at = todo.back(); todo.pop_back();
            for (std::size_t next = 0; next < edges.size(); ++next) {
                if (edges[at][next] && !visited[next]) { visited[next] = true; todo.push_back(next); }
            }
        }
        result[source] = std::move(visited);
    }
    return result;
}
std::vector<fs::FiniteReplacementRecord> groupRecords(const fs::FiniteRequirementFrame& frame,
    fs::RequirementGroupId group, fs::RegionExpressions::Id guard)
{
    std::vector<fs::FiniteReplacementRecord> result;
    std::set<uint64_t> seen;
    for (const auto& member : frame.original()->originalMemberships()) {
        if (member.group == group && seen.insert(member.record.originalRecord).second) {
            result.push_back({{987, result.size()},
                frame.analysis().scan.generators[member.record.originalRecord], guard});
        }
    }
    return result;
}
bool checkGroup(fs::FiniteRequirements frame, fs::RequirementGroupId group,
                fs::RegionExpressions::Id predicate, uint64_t& cases)
{
    auto& arena = *frame->original()->context()->expressions();
    const auto yes = arena.boolean(true), no = arena.boolean(false);
    auto actual = groupRecords(*frame, group, yes);
    auto expectedGraph = native(*frame);
    for (const auto& record : actual) { expectedGraph[2 * record.edge.source + 1][2 * record.edge.target] = true; }
    const auto expected = closure(expectedGraph);
    std::vector<fs::StorageGenerator> all;
    for (uint32_t a = 0; a < frame->analysis().occurrences.size(); ++a) {
        for (uint32_t b = a + 1; b < frame->analysis().occurrences.size(); ++b) { all.push_back({a, b}); }
    }
    if (all.size() > 10) { return false; }
    for (uint64_t mask = 0; mask < (uint64_t(1) << all.size()); ++mask) {
        auto alternativeGraph = native(*frame);
        std::vector<fs::FiniteReplacementRecord> alternative;
        for (std::size_t edge = 0; edge < all.size(); ++edge) {
            if (!(mask & (uint64_t(1) << edge))) { continue; }
            alternative.push_back({{988, edge}, all[edge], yes});
            alternativeGraph[2 * all[edge].source + 1][2 * all[edge].target] = true;
        }
        const bool equal = closure(alternativeGraph) == expected;
        const auto proof = fs::certifyFiniteRequirementReplacement(frame, frame->original(), group, yes, alternative);
        if (bool(proof.evidence) != equal || proof.error.empty() != equal) { return false; }
        ++cases;
    }
    auto guarded = actual;
    for (auto& record : guarded) { record.guard = predicate; }
    auto proof = fs::certifyFiniteRequirementReplacement(frame, frame->original(), group, predicate, guarded);
    if (!proof.evidence || !proof.records || proof.frame != frame) { return false; }
    std::string error;
    auto applied = fs::applyRequirementReplacement(frame->original(), proof.evidence, error);
    if (!applied || applied->originalMemberships().data() != frame->original()->originalMemberships().data() ||
        applied->lowerFacts().size() != guarded.size()) { return false; }
    const auto old = frame->original()->memberships();
    for (std::size_t i = 0; i < old.size(); ++i) {
        if (old[i].group != group && applied->memberships()[i].guard != old[i].guard) { return false; }
        if (old[i].group == group && applied->memberships()[i].guard != arena.lnot(predicate)) { return false; }
    }
    if (fs::applyRequirementReplacement(applied, proof.evidence, error)) { return false; }
    // A second proof on a revised snapshot must still compare ORIGINAL members.
    for (auto& record : actual) { record.key.producer = 989; }
    auto second = fs::certifyFiniteRequirementReplacement(frame, applied, group, arena.lnot(predicate), actual);
    if (!second.evidence || !fs::applyRequirementReplacement(applied, second.evidence, error)) { return false; }
    auto bad = actual;
    if (!bad.empty()) {
        bad.front().edge = {0, 0};
        if (fs::certifyFiniteRequirementReplacement(frame, frame->original(), group, yes, bad).evidence) {
            return false;
        }
        bad = actual;
        bad.front().key = frame->original()->originalMemberships().front().record;
        if (fs::certifyFiniteRequirementReplacement(frame, frame->original(), group, yes, bad).evidence) {
            return false;
        }
    }
    const auto size = arena.size();
    auto empty = fs::certifyFiniteRequirementReplacement(frame, frame->original(), group, yes, {});
    if (bool(empty.evidence) != (closure(native(*frame)) == expected) || (!empty.evidence && arena.size() != size)) {
        return false;
    }
    auto chosen = fs::applyCertifiedFiniteReplacement(frame, {}, group, predicate, guarded, error);
    if (!chosen || chosen->snapshot() == frame->original() || chosen->definitions().size() != guarded.size()) {
        return false;
    }
    auto revised = fs::applyCertifiedFiniteReplacement(frame, chosen, group, arena.lnot(predicate), actual, error);
    if (!revised || revised->definitions().size() != guarded.size() + actual.size()) { return false; }
    if (!actual.empty()) {
        auto malformed = actual; malformed.front().edge = {0, 0};
        if (fs::applyCertifiedFiniteReplacement(frame, revised, group, yes, malformed, error)) { return false; }
    }
    auto vacuous = fs::certifyFiniteRequirementReplacement(frame, frame->original(), group, no, {});
    return bool(vacuous.evidence);
}
} // namespace
int runFiniteRequirementReplacementChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (!function->hasAttr("test.finite_replacement")) { return 0; }
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return 1; }
    auto phases = index.explicitSequence(function.front());
    if (failed(phases) || phases->size() > 5 || function.getNumArguments() != 1) { return 1; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    const auto predicate = arena->input(function.getArgument(0));
    std::vector<fs::RequirementGroupId> groups(input.accesses().cells().size());
    for (std::size_t i = 0; i < groups.size(); ++i) { groups[i] = 1 + i % 2; }
    std::string error;
    auto frame = fs::captureFiniteRequirements(function, input, *phases, arena, 17, groups, 3, error);
    if (!frame || !error.empty() || frame->analysis().scan.generators.empty()) {
        llvm::errs() << "finite capture: " << error << "\n"; return 1;
    }
    uint64_t cases = 0;
    for (const auto& group : frame->original()->groups()) {
        if (group.scope == fs::RequirementScope::Internal && !checkGroup(frame, group.id, predicate, cases)) {
            llvm::errs() << "finite group oracle failed: " << group.id << "\n"; return 1;
        }
    }
    auto foreign = fs::captureFiniteRequirements(function, input, *phases, arena, 17, groups, 3, error);
    if (!foreign || fs::certifyFiniteRequirementReplacement(frame, foreign->original(), 1,
        arena->boolean(true), {}).evidence || fs::certifyFiniteRequirementReplacement(frame, frame->original(), 3,
        arena->boolean(true), {}).evidence) { return 1; }
    // Public representation-only provenance construction cannot forge the private scan binding.
    auto original = frame->original();
    auto forged = fs::createRequirementProvenance(original->context(),
        std::vector<fs::RequirementGroup>(original->groups().begin(), original->groups().end()),
        std::vector<fs::RequirementMembership>(original->originalMemberships().begin(),
                                             original->originalMemberships().end()), error);
    if (!forged || fs::certifyFiniteRequirementReplacement(frame, forged, 1, arena->boolean(true), {}).evidence) {
        return 1;
    }
    llvm::outs() << "finite replacement checks passed: " << function.getSymName() << " cases=" << cases << "\n";
    return 0;
}
