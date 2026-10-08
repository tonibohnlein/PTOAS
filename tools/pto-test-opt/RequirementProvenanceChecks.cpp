// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic ownership/binding tests, not a proof checker for producer equivalence.
#include "PTO/Transforms/FrontierSynch/RequirementProvenance.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Expr = fs::RegionExpressions::Id;
fs::OrderContext context(const pto::SyncInput& input, std::shared_ptr<fs::RegionExpressions> arena)
{
    fs::RegionalAnalysis domain;
    domain.expressions = std::move(arena);
    domain.accessModel = &input.accesses();
    domain.gmAliasPolicy = input.memory().gmPolicy();
    std::string error;
    auto captured = fs::captureRegionalOrderContext(input, domain, error);
    return succeeded(captured) ? *captured : fs::OrderContext{};
}
bool value(fs::RegionExpressions& arena, Expr guard, Expr x, Expr p, uint64_t index, bool predicate, bool expected)
{
    const std::array<std::pair<Expr, Expr>, 2> bindings{{
        {x, arena.constant(index)}, {p, arena.boolean(predicate)}}};
    fs::RegionExpressions::Substitution substitution(bindings);
    return arena.constantValue(arena.substitute(guard, substitution)) == uint64_t(expected);
}
bool guarded(const fs::OrderContext& owner, Expr x, Expr p)
{
    auto& arena = *owner->expressions();
    const auto q = arena.eq(arena.rem(x, arena.constant(2)), arena.constant(0));
    const auto beta = arena.lt(x, arena.constant(2));
    const auto yes = arena.boolean(true), no = arena.boolean(false);
    const fs::RequirementRecordKey shared{11, 7};
    const std::vector<fs::RequirementGroup> groups{{1, fs::RequirementScope::Internal},
        {2, fs::RequirementScope::Boundary}, {3, fs::RequirementScope::FixedPrerequisite}};
    const std::vector<fs::RequirementMembership> original{
        {1, shared, p}, {1, {11, 9}, q}, {2, shared, arena.lnot(p)}, {3, shared, yes}};
    std::string error;
    auto source = fs::createRequirementProvenance(owner, groups, original, error);
    std::vector<fs::RequirementExactRecord> records{{{22, 7}, arena.lor(p, q)}, {{11, 9}, q}};
    auto evidence = fs::bindTrustedRequirementReplacement(source, 1, beta, records, error);
    // Mutation of caller-owned request data cannot change the bound replacement.
    records[0].guard = no;
    auto result = fs::applyRequirementReplacement(source, evidence, error);
    if (!source || !evidence || !result || !error.empty() || result->memberships().size() != 6 ||
        result->lowerFacts().size() != 2 || result->originalMemberships().size() != original.size()) { return false; }
    for (std::size_t i = 0; i < original.size(); ++i) {
        if (source->memberships()[i].guard != original[i].guard ||
            result->originalMemberships()[i].guard != original[i].guard ||
            !(result->memberships()[i].record == original[i].record)) { return false; }
    }
    for (uint64_t index = 0; index < 4; ++index) {
        for (bool predicate : {false, true}) {
            const bool even = index % 2 == 0, covered = index < 2;
            const std::array<bool, 6> expected{{predicate && !covered, even && !covered,
                !predicate, true, (predicate || even) && covered, even && covered}};
            for (std::size_t i = 0; i < expected.size(); ++i) {
                if (!value(arena, result->memberships()[i].guard, x, p, index, predicate, expected[i])) {
                    return false;
                }
            }
            for (std::size_t i = 0; i < result->lowerFacts().size(); ++i) {
                if (!value(arena, result->lowerFacts()[i].guard, x, p, index, predicate, expected[i + 4])) {
                    return false;
                }
            }
        }
    }
    // A proof token cannot be replayed on a revised or merely equal snapshot.
    auto copied = fs::createRequirementProvenance(owner, groups, original, error);
    if (fs::applyRequirementReplacement(result, evidence, error) ||
        fs::applyRequirementReplacement(copied, evidence, error)) { return false; }
    auto second = fs::bindTrustedRequirementReplacement(result, 1, arena.lnot(beta), {}, error);
    auto twice = fs::applyRequirementReplacement(result, second, error);
    if (!twice || twice->lowerFacts().size() != 2 || twice->originalMemberships().data() !=
        source->originalMemberships().data()) { return false; }
    for (uint64_t index = 0; index < 4; ++index) {
        if (!value(arena, twice->memberships()[0].guard, x, p, index, true, false) ||
            !value(arena, twice->memberships()[4].guard, x, p, index, true, index < 2)) { return false; }
    }
    auto vacuous = fs::bindTrustedRequirementReplacement(source, 1, no, {{{33, 0}, yes}}, error);
    auto unchanged = fs::applyRequirementReplacement(source, vacuous, error);
    return unchanged && unchanged->memberships()[0].guard == p &&
           arena.constantValue(unchanged->memberships().back().guard) == 0;
}
bool invalid(const fs::OrderContext& owner, const fs::OrderContext& other)
{
    auto& arena = *owner->expressions();
    const auto yes = arena.boolean(true), integer = arena.constant(0);
    const std::vector<fs::RequirementGroup> groups{{1, fs::RequirementScope::Internal},
                                                 {2, fs::RequirementScope::FixedPrerequisite}};
    std::string error;
    auto source = fs::createRequirementProvenance(owner, groups, {{1, {0, 0}, yes}}, error);
    auto foreign = fs::createRequirementProvenance(other, groups, {{1, {0, 0}, yes}}, error);
    auto evidence = fs::bindTrustedRequirementReplacement(source, 1, yes, {}, error);
    if (!source || !foreign || !evidence || fs::applyRequirementReplacement(foreign, evidence, error) ||
        fs::bindTrustedRequirementReplacement(source, 2, yes, {}, error) ||
        fs::bindTrustedRequirementReplacement(source, 3, yes, {}, error) ||
        fs::bindTrustedRequirementReplacement(source, 1, integer, {}, error) ||
        fs::bindTrustedRequirementReplacement(source, 1, yes, {{{0, 1}, integer}}, error) ||
        fs::bindTrustedRequirementReplacement(source, 1, fs::RegionExpressions::invalid, {}, error) ||
        fs::applyRequirementReplacement(source, {}, error)) { return false; }
    if (fs::createRequirementProvenance({}, {}, {}, error) ||
        fs::createRequirementProvenance(owner, {{1}, {1}}, {}, error) ||
        fs::createRequirementProvenance(owner, {{1, static_cast<fs::RequirementScope>(99)}}, {}, error) ||
        fs::createRequirementProvenance(owner, groups, {{3, {0, 0}, yes}}, error) ||
        fs::createRequirementProvenance(owner, groups, {{1, {0, 0}, integer}}, error)) { return false; }
    return bool(fs::createRequirementProvenance(owner, {}, {}, error));
}
bool scan(const fs::OrderContext& owner)
{
    const std::vector<fs::ExplicitEffects> occurrences{
        {0, 0, {{0, false, true}, {1, false, true}}},
        {1, 1, {{0, true, false}, {1, true, false}}},
        {2, 2, {{0, false, true}, {1, false, true}}}};
    const auto original = fs::scanStorageLifetimes(occurrences, {{0, 1}});
    std::string error;
    auto captured = fs::captureStorageRequirementProvenance(owner, 17, original, {10, 11}, 99, error);
    if (!captured || original.generators.size() != 3 || captured->memberships().size() != 7) { return false; }
    std::set<fs::RequirementGroupId> owners;
    for (const auto& member : captured->memberships()) {
        if (member.record.producer != 17 || member.record.originalRecord >= original.generators.size() ||
            owner->expressions()->constantValue(member.guard) != 1) { return false; }
        const auto& edge = original.generators[member.record.originalRecord];
        if (edge.source == 0 && edge.target == 1) { owners.insert(member.group); }
    }
    if (owners != std::set<fs::RequirementGroupId>({10, 11, 99})) { return false; }
    auto evidence = fs::bindTrustedRequirementReplacement(captured, 10, owner->expressions()->boolean(true), {}, error);
    auto replaced = fs::applyRequirementReplacement(captured, evidence, error);
    if (!replaced || replaced->memberships().size() != original.witnesses.size()) { return false; }
    for (const auto& member : replaced->memberships()) {
        if (owner->expressions()->constantValue(member.guard) != uint64_t(member.group != 10)) { return false; }
    }
    auto malformed = original;
    malformed.generators.push_back({2, 3});
    if (fs::captureStorageRequirementProvenance(owner, 17, malformed, {10, 11}, 99, error)) { return false; }
    malformed = original;
    malformed.witnesses.front().generator = UINT32_MAX;
    if (fs::captureStorageRequirementProvenance(owner, 17, malformed, {10, 11}, 99, error)) { return false; }
    malformed = original;
    malformed.witnesses.front().atom = UINT32_MAX;
    if (fs::captureStorageRequirementProvenance(owner, 17, malformed, {10, 11}, 99, error)) { return false; }
    malformed = original;
    malformed.witnesses.front().hazard = static_cast<fs::StorageHazard>(99);
    if (fs::captureStorageRequirementProvenance(owner, 17, malformed, {10, 11}, 99, error)) { return false; }
    malformed = original;
    malformed.error = "failed scan";
    if (fs::captureStorageRequirementProvenance(owner, 17, malformed, {10, 11}, 99, error) ||
        fs::captureStorageRequirementProvenance(owner, 17, original, {99, 11}, 99, error)) { return false; }
    return bool(fs::captureStorageRequirementProvenance(owner, 17, {}, {}, 99, error));
}
} // namespace
int runRequirementProvenanceChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (function.getNumArguments() < 2 || !function.getArgument(1).getType().isInteger(1)) { return 1; }
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto owner = context(input, arena), other = context(input, arena);
    auto x = arena->input(function.getArgument(0)), p = arena->input(function.getArgument(1));
    if (!owner || !other || !guarded(owner, x, p) || !invalid(owner, other) || !scan(owner)) {
        llvm::errs() << "guarded requirement provenance binding/ownership check failed\n";
        return 1;
    }
    llvm::outs() << "guarded requirement provenance binding/ownership checks passed\n";
    return 0;
}
