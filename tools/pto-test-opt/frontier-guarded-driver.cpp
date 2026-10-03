// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bounded synthetic predicate inputs at the production analysis boundary.
#include "frontier-guarded-driver.h"
#include "frontier-guarded-dump.h"
#include "mlir/IR/Builders.h"
#include <cstdint>
namespace frontier_test {
namespace {
namespace fs = mlir::pto::frontiersynch;
constexpr std::size_t MaxAtoms = 32;
constexpr std::size_t MaxSites = 32;
struct Guards {
    mlir::Block arguments;
    fs::PredicateArena arena;
    llvm::SmallVector<fs::Predicate> atoms;
};
mlir::LogicalResult initialize(Guards& guards, const llvm::json::Object& object, mlir::MLIRContext& context)
{
    auto count = object.getInteger("atoms");
    if (!count || *count < 0 || static_cast<std::uint64_t>(*count) > MaxAtoms) {
        return mlir::failure();
    }
    for (std::int64_t id = 0; id < *count; ++id) {
        auto value = guards.arguments.addArgument(mlir::IntegerType::get(&context, 1), mlir::UnknownLoc::get(&context));
        auto atom = guards.arena.atom(value);
        if (mlir::failed(atom)) {
            return mlir::failure();
        }
        guards.atoms.push_back(*atom);
    }
    return mlir::success();
}
mlir::FailureOr<fs::Predicate> predicate(const llvm::json::Value& value, Guards& guards)
{
    const auto* literals = value.getAsArray();
    if (!literals || literals->size() > MaxAtoms) {
        return mlir::failure();
    }
    fs::Predicate result = 1;
    const auto bound = static_cast<std::int64_t>(guards.atoms.size());
    for (const auto& literal : *literals) {
        auto id = literal.getAsInteger();
        if (!id || *id == 0 || *id < -bound || *id > bound) {
            return mlir::failure();
        }
        auto atom = guards.atoms[static_cast<std::size_t>((*id > 0 ? *id : -*id) - 1)];
        result = guards.arena.conjunction(result, *id > 0 ? atom : guards.arena.negate(atom));
    }
    return result;
}
mlir::FailureOr<llvm::SmallVector<fs::GuardedDemand>> additional(
    const llvm::json::Object& object, Guards& guards, std::size_t count)
{
    llvm::SmallVector<fs::GuardedDemand> result;
    const auto* edges = object.getArray("guarded_extra");
    if (!edges) {
        return result;
    }
    if (edges->size() > MaxSites * MaxSites) {
        return mlir::failure();
    }
    for (const auto& edge : *edges) {
        const auto* row = edge.getAsArray();
        if (!row || row->size() != 3) {
            return mlir::failure();
        }
        auto source = (*row)[0].getAsInteger();
        auto consumer = (*row)[1].getAsInteger();
        if (!source || !consumer || *source < 0 || *consumer < 0 || static_cast<std::uint64_t>(*source) > count ||
            static_cast<std::uint64_t>(*consumer) > count) {
            return mlir::failure();
        }
        auto guard = predicate((*row)[2], guards);
        if (mlir::failed(guard)) {
            return mlir::failure();
        }
        result.push_back({static_cast<std::size_t>(*source), static_cast<std::size_t>(*consumer), *guard, {}});
    }
    return result;
}
} // namespace
mlir::FailureOr<llvm::json::Object> analyzeGuarded(
    const llvm::json::Object& object, mlir::ArrayRef<const mlir::pto::CompoundInstanceElement*> sequence,
    mlir::ArrayRef<fs::StorageFootprint> footprints, mlir::ArrayRef<fs::StorageAlias> aliases,
    mlir::MLIRContext& context)
{
    auto* rows = object.getArray("guards");
    if (!rows || rows->size() != sequence.size() || sequence.size() > MaxSites) {
        return mlir::failure();
    }
    Guards guards;
    if (mlir::failed(initialize(guards, object, context))) {
        return mlir::failure();
    }
    llvm::SmallVector<fs::Predicate> occurrences;
    for (const auto& row : *rows) {
        auto guard = predicate(row, guards);
        if (mlir::failed(guard)) {
            return mlir::failure();
        }
        occurrences.push_back(*guard);
    }
    auto extra = additional(object, guards, sequence.size());
    if (mlir::failed(extra)) {
        return mlir::failure();
    }
    fs::GuardedDemandAnalysis analysis;
    llvm::SmallVector<fs::Predicate> present(sequence.size(), 1);
    if (mlir::failed(analysis.build(sequence, present, footprints, {}, guards.arena))) {
        return mlir::failure();
    }
    // Invalidate AFTER warming the result so production failure atomicity is checked.
    auto invalidPredicate = object.getBoolean("invalid_predicate");
    if (invalidPredicate.value_or(false) && !occurrences.empty()) {
        occurrences.front() = guards.arena.nodes().size();
    }
    if (object.getBoolean("invalid_count").value_or(false) && !occurrences.empty()) {
        occurrences.pop_back();
    }
    llvm::SmallVector<const mlir::pto::CompoundInstanceElement*> candidate(sequence);
    if (object.getBoolean("duplicate_phase").value_or(false) && candidate.size() > 1) {
        candidate[1] = candidate[0];
    }
    if (object.getBoolean("null_phase").value_or(false) && !candidate.empty()) {
        candidate[0] = nullptr;
    }
    if (mlir::failed(analysis.build(candidate, occurrences, footprints, aliases, std::move(guards.arena), *extra))) {
        const bool empty = analysis.phases().empty() && analysis.occurrences().empty() &&
                           analysis.generators().empty() && analysis.retained().empty() &&
                           analysis.localDemands().empty() && analysis.predicates().nodes().size() == 2 &&
                           !analysis.completionBeforeStart(0, 0);
        return llvm::json::Object{{"valid", false}, {"empty", empty}};
    }
    return dump(analysis);
}
} // namespace frontier_test
