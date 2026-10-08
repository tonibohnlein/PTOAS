// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent matching enumeration and actual original-branch cut checks.
#include "PTO/Transforms/FrontierSynch/GuardedCompactMatching.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Id = fs::RegionExpressions::Id;
using Proof = fs::GuardedCompactProof;
uint64_t evaluate(fs::RegionExpressions& arena, Id root, Id ordinal, Id trips, uint64_t i, uint64_t count)
{
    const std::pair<Id, Id> pairs[]{{ordinal, arena.constant(i)}, {trips, arena.constant(count)}};
    fs::RegionExpressions::Substitution substitution(pairs);
    return arena.constantValue(arena.substitute(root, substitution)).value_or(UINT64_MAX);
}
bool algebra(MLIRContext& context)
{
    for (uint32_t mask = 0; mask < 64; ++mask) {
        for (uint64_t shift = 0; shift <= 3; ++shift) {
            Block parameters;
            fs::RegionExpressions arena;
            auto ordinal = arena.input(parameters.addArgument(IndexType::get(&context), UnknownLoc::get(&context)));
            auto trips = arena.input(parameters.addArgument(IndexType::get(&context), UnknownLoc::get(&context)));
            auto guard = [&arena, mask](uint32_t site, Id at) -> std::optional<fs::GuardedCompactPresence> {
                auto result = arena.boolean(site == 0);
                if (site) {
                    for (uint64_t j = 0; j < 6; ++j) {
                        if ((mask >> j) & 1U) { result = arena.lor(result, arena.eq(at, arena.constant(j))); }
                    }
                }
                return fs::GuardedCompactPresence{result, {}};
            };
            const fs::PeriodicRecord records[]{{0, 1, shift}, {0, 1, shift}};
            auto matching = fs::buildGuardedCompactMatching(arena, ordinal, trips, {{0}, {1}}, records, guard);
            if (!matching.error.empty() || !matching.sourcePresenceComplete || matching.records.size() != 2 ||
                matching.guardEvaluations != 4 || matching.records[0].proof != Proof::MandatorySource) { return false; }
            for (uint64_t count = 0; count <= 6; ++count) {
                std::set<std::pair<std::size_t, uint64_t>> sets, waits, expected;
                for (std::size_t record = 0; record < matching.records.size(); ++record) {
                    const auto& recipe = matching.records[record];
                    for (uint64_t i = 0; i < count; ++i) {
                        if (evaluate(arena, recipe.publication, ordinal, trips, i, count)) {
                            sets.emplace(record, evaluate(arena, recipe.sourceIdentity, ordinal, trips, i, count));
                        }
                        if (((mask >> i) & 1U) && evaluate(arena, recipe.acquisition, ordinal, trips, i, count)) {
                            waits.emplace(record, evaluate(arena, recipe.targetIdentity, ordinal, trips, i, count));
                        }
                        if (i >= shift && ((mask >> i) & 1U)) { expected.emplace(record, i - shift); }
                    }
                }
                if (sets != expected || waits != expected) { return false; }
            }
        }
    }
    return true;
}
bool proofCases(MLIRContext& context)
{
    Block parameters;
    fs::RegionExpressions arena;
    auto value = [&parameters, &arena, &context](Type type) {
        return arena.input(parameters.addArgument(type, UnknownLoc::get(&context)));
    };
    const auto ordinal = value(IndexType::get(&context)), trips = value(IndexType::get(&context));
    const auto g = value(IntegerType::get(&context, 1)), h = value(IntegerType::get(&context, 1));
    auto enclosing = [&arena, g, h](uint32_t site, Id) -> std::optional<fs::GuardedCompactPresence> {
        return site ? fs::GuardedCompactPresence{arena.land(g, h), {g, h}} : fs::GuardedCompactPresence{g, {g}};
    };
    auto result = fs::buildGuardedCompactMatching(arena, ordinal, trips, {{0}, {1}}, {{0, 1, 0}}, enclosing);
    if (!result.sourcePresenceComplete || result.records[0].proof != Proof::EnclosingBranch) { return false; }
    auto shifted = [&arena](uint32_t site, Id at) -> std::optional<fs::GuardedCompactPresence> {
        auto coordinate = site ? at : arena.add(at, arena.constant(2));
        return fs::GuardedCompactPresence{arena.eq(arena.rem(coordinate, arena.constant(3)), arena.constant(1)), {}};
    };
    result = fs::buildGuardedCompactMatching(arena, ordinal, trips, {{0}, {1}}, {{0, 1, 2}}, shifted);
    if (!result.sourcePresenceComplete || result.records[0].proof != Proof::IdenticalGuard) { return false; }
    auto unknown = [&arena, g](uint32_t site, Id) -> std::optional<fs::GuardedCompactPresence> {
        return fs::GuardedCompactPresence{site ? arena.boolean(true) : g, {}};
    };
    result = fs::buildGuardedCompactMatching(arena, ordinal, trips, {{0}, {1}}, {{0, 1, 0}}, unknown);
    if (result.sourcePresenceComplete || !result.error.empty() || result.records.size() != 1) { return false; }
    result = fs::buildGuardedCompactMatching(arena, ordinal, trips, {{0}, {0}}, {{0, 1, 0}}, enclosing);
    if (result.sourcePresenceComplete || result.records[0].proof != Proof::Local) { return false; }
    auto malformed = [&arena, g](uint32_t, Id) -> std::optional<fs::GuardedCompactPresence> {
        return fs::GuardedCompactPresence{arena.boolean(true), {g}};
    };
    result = fs::buildGuardedCompactMatching(arena, ordinal, trips, {{0}, {1}}, {{0, 1, 0}}, malformed);
    if (result.error.empty() || result.records.size() != 1) { return false; }
    auto mandatory = [&arena](uint32_t, Id) -> std::optional<fs::GuardedCompactPresence> {
        return fs::GuardedCompactPresence{arena.boolean(true), {}};
    };
    result = fs::buildGuardedCompactMatching(arena, ordinal, trips, {{0}, {1}}, {{0, 1, UINT64_MAX}}, mandatory);
    if (!result.sourcePresenceComplete ||
        evaluate(arena, result.records[0].publication, ordinal, trips, 1, UINT64_MAX) != 0 ||
        evaluate(arena, result.records[0].publication, ordinal, trips, 0, UINT64_MAX) != 0) { return false; }
    result = fs::buildGuardedCompactMatching(arena, ordinal, trips, {{0}, {1}}, {{1, 0, 0}}, mandatory);
    return !result.error.empty() && result.records.size() == 1;
}
bool quality(MLIRContext& context)
{
    Block parameters;
    fs::RegionExpressions arena;
    const auto ordinal = arena.input(parameters.addArgument(IndexType::get(&context), UnknownLoc::get(&context)));
    const auto trips = arena.input(parameters.addArgument(IndexType::get(&context), UnknownLoc::get(&context)));
    for (uint64_t delta = 0; delta <= 7; ++delta) {
        for (uint64_t count = 0; count <= 6; ++count) {
            for (uint32_t mask = 0; mask < 64; ++mask) {
                uint64_t sum = 0, actualPairs = 0;
                for (uint64_t j = 0; j < count; ++j) {
                    const bool active = (mask >> j) & 1U;
                    const auto bound = fs::optionalIndirectReadExcess(
                        arena, ordinal, arena.boolean(active), delta, count);
                    sum += evaluate(arena, bound.contribution, ordinal, trips, j, count);
                    // Actual worst-case f(j)=max(0,j-delta). Count excess source
                    // completions separately; earlier readers have nondecreasing f.
                    if (active) {
                        const auto selected = j > delta ? j - delta : 0;
                        for (uint64_t source = selected + 1; source <= j; ++source) { ++actualPairs; }
                    }
                    if (bound.allPresent.ugt(bound.deltaTimesTrips)) { return false; }
                }
                if (sum != actualPairs) { return false; }
            }
        }
    }
    const auto huge = fs::optionalIndirectReadExcess(arena, ordinal, arena.boolean(true), UINT64_MAX, UINT64_MAX);
    return huge.allPresent.getActiveBits() > 64 && huge.deltaTimesTrips.getActiveBits() == 128;
}
std::string print(func::FuncOp function)
{
    std::string text;
    llvm::raw_string_ostream output(text);
    function.print(output);
    return text;
}
bool originalCuts(func::FuncOp function, const pto::SyncInput& input)
{
    auto kind = function->getAttrOfType<IntegerAttr>("test.guard_case");
    auto shift = function->getAttrOfType<IntegerAttr>("test.guard_shift");
    if (!kind || !shift || shift.getInt() < 0) { return false; }
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
    scf::ForOp loop;
    for (auto candidate : function.getOps<scf::ForOp>()) { if (loop) { return false; } loop = candidate; }
    std::vector<const pto::CompoundInstanceElement*> phases(2);
    function.walk([&index, &phases](Operation* operation) {
        if (auto site = operation->getAttrOfType<IntegerAttr>("test.guard_site")) {
            if (site.getInt() >= 0 && site.getInt() < 2 && index.phasesFor(operation).size() == 1) {
                phases[site.getInt()] = index.phasesFor(operation)[0];
            }
        }
    });
    if (!loop || !phases[0] || !phases[1]) { return false; }
    const auto before = print(function);
    auto prepared = fs::prepareGuardedCompactMatching(function, loop, index, phases,
        {{0, 1, static_cast<uint64_t>(shift.getInt())}});
    if (before != print(function) || prepared.matching.records.size() != 1) { return false; }
    if (kind.getInt() >= 3) {
        if (prepared.plan || prepared.placementError.empty()) { return false; }
        if (kind.getInt() == 3) { return !prepared.matching.sourcePresenceComplete; }
        if (kind.getInt() == 4) { return prepared.matching.sourcePresenceComplete; }
        return prepared.matching.records[0].proof == Proof::Local;
    }
    if (!prepared.plan || !prepared.placementError.empty() || prepared.plan->endpoints.size() != 2 ||
        prepared.matching.records[0].proof != static_cast<Proof>(kind.getInt())) { return false; }
    auto& plan = *prepared.plan;
    if (plan.endpoints[0].before != phases[0]->elementOp->getNextNode() ||
        plan.endpoints[1].before != phases[1]->elementOp || plan.endpoints[0].memberCoordinates.size() != 1 ||
        plan.endpoints[1].memberCoordinates.size() != 1) { return false; }
    // Commit only after detached success; production never mutates on a gap.
    return succeeded(fs::insertLogicalSynchronization(function, plan)) && succeeded(verify(function));
}
} // namespace
int runGuardedCompactMatchingChecks(func::FuncOp function, const pto::SyncInput& input)
{
    auto kind = function->getAttrOfType<IntegerAttr>("test.guard_case");
    if (!kind || (kind.getInt() == 0 && (!algebra(*function.getContext()) || !proofCases(*function.getContext()) ||
        !quality(*function.getContext()))) || !originalCuts(function, input)) {
        llvm::errs() << "guarded compact matching checks failed: " << function.getSymName() << "\n";
        return 1;
    }
    llvm::outs() << "guarded compact matching checks passed: " << function.getSymName() << "\n";
    return 0;
}
