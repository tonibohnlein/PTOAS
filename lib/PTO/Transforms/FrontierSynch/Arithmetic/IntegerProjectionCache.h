// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_INTEGERPROJECTIONCACHE_H
#define PTO_FRONTIERSYNCH_INTEGERPROJECTIONCACHE_H
#include "PTO/Transforms/FrontierSynch/AnalysisCost.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticStorageSelectors.h"
#include "llvm/ADT/Hashing.h"
#include <map>
#include <tuple>
#include <unordered_map>
namespace mlir::pto::frontiersynch {
// Numerical queries in one unchanged builder context. Every assumption is in
// the owned system; tags, physical bases and coordinate residues stay outside.
// Structural equality needs no mathematical query. Map-owned operation values
// remain stable while new systems are interned; failures are never cached.
class IntegerProjectionCache {
public:
    bool isEmpty(const IntegerSystem& system)
    {
        accumulateCost(counts.emptinessRequests, 1);
        const auto id = identity(system);
        auto found = empty.find(id);
        if (found != empty.end()) {
            accumulateCost(counts.emptinessHits, 1); return found->second;
        }
        accumulateCost(counts.emptinessBuilds, 1);
        const bool value = systems[id].isEmpty();
        empty.emplace(id, value);
        return value;
    }
    FailureOr<llvm::ArrayRef<IntegerSystem>> project(const IntegerSystem& system, llvm::ArrayRef<unsigned> keep)
    {
        accumulateCost(counts.projectionRequests, 1);
        const auto id = identity(system);
        const auto key = std::make_pair(id, keep.vec());
        if (auto found = projected.find(key); found != projected.end()) {
            accumulateCost(counts.projectionHits, 1); return llvm::ArrayRef<IntegerSystem>(found->second);
        }
        accumulateCost(counts.projectionBuilds, 1);
        auto value = systems[id].project(keep);
        if (failed(value)) { return failure(); }
        accumulateCost(counts.projectionPieces, value->size());
        auto position = projected.emplace(key, std::move(*value)).first;
        return llvm::ArrayRef<IntegerSystem>(position->second);
    }
    FailureOr<const IntegerSystem*> remap(const IntegerSystem& system, unsigned dimensions,
                                         llvm::ArrayRef<unsigned> columns)
    {
        accumulateCost(counts.remapRequests, 1);
        const auto key = std::make_tuple(identity(system), dimensions, columns.vec());
        if (auto found = remapped.find(key); found != remapped.end()) {
            accumulateCost(counts.remapHits, 1); return &found->second;
        }
        accumulateCost(counts.remapBuilds, 1);
        auto value = system.remap(dimensions, columns);
        if (failed(value)) { return failure(); }
        return &remapped.emplace(key, std::move(*value)).first->second;
    }
    FailureOr<const IntegerSystem*> intersect(const IntegerSystem& left, const IntegerSystem& right)
    {
        accumulateCost(counts.intersectionRequests, 1);
        const auto leftID = identity(left), rightID = identity(right);
        const auto key = std::make_pair(leftID, rightID);
        if (auto found = intersected.find(key); found != intersected.end()) {
            accumulateCost(counts.intersectionHits, 1); return &found->second;
        }
        accumulateCost(counts.intersectionBuilds, 1);
        auto value = left.intersect(right);
        if (failed(value)) { return failure(); }
        return &intersected.emplace(key, std::move(*value)).first->second;
    }
    FailureOr<llvm::ArrayRef<IntegerTupleWitness>> witnesses(
        const IntegerSystem& system, unsigned inputs, llvm::ArrayRef<uint64_t> outputResidues)
    {
        accumulateCost(counts.witnessRequests, 1);
        const auto key = std::make_tuple(identity(system), inputs, outputResidues.vec());
        if (auto found = witnessed.find(key); found != witnessed.end()) {
            accumulateCost(counts.witnessHits, 1); return llvm::ArrayRef<IntegerTupleWitness>(found->second);
        }
        accumulateCost(counts.witnessBuilds, 1);
        auto value = buildIntegerTupleWitnesses(system, inputs, outputResidues);
        if (failed(value)) { return failure(); }
        accumulateCost(counts.witnessPieces, value->size());
        auto position = witnessed.emplace(key, std::move(*value)).first;
        return llvm::ArrayRef<IntegerTupleWitness>(position->second);
    }
    const ArithmeticStorageSelectorCost& cost() const { return counts; }
private:
    ArithmeticStorageSelectorCost counts;
    std::vector<IntegerSystem> systems;
    std::unordered_map<std::size_t, std::vector<std::size_t>> buckets;
    std::map<std::size_t, bool> empty;
    std::map<std::pair<std::size_t, std::vector<unsigned>>, std::vector<IntegerSystem>> projected;
    std::map<std::tuple<std::size_t, unsigned, std::vector<unsigned>>, IntegerSystem> remapped;
    std::map<std::pair<std::size_t, std::size_t>, IntegerSystem> intersected;
    std::map<std::tuple<std::size_t, unsigned, std::vector<uint64_t>>,
             std::vector<IntegerTupleWitness>> witnessed;
    static bool same(const IntegerSystem& a, const IntegerSystem& b)
    {
        if (a.dimensions() != b.dimensions() || a.constraints().size() != b.constraints().size() ||
            a.congruences().size() != b.congruences().size()) { return false; }
        for (std::size_t i = 0; i < a.constraints().size(); ++i) {
            const auto& x = a.constraints()[i]; const auto& y = b.constraints()[i];
            if (x.coefficients != y.coefficients || x.bound != y.bound) { return false; }
        }
        for (std::size_t i = 0; i < a.congruences().size(); ++i) {
            const auto& x = a.congruences()[i]; const auto& y = b.congruences()[i];
            if (x.coefficients != y.coefficients || x.residue != y.residue || x.modulus != y.modulus) { return false; }
        }
        return true;
    }
    std::size_t identity(const IntegerSystem& system)
    {
        std::size_t hash = system.dimensions();
        for (const auto& row : system.constraints()) {
            hash = llvm::hash_combine(hash,
                llvm::hash_combine_range(row.coefficients.begin(), row.coefficients.end()), row.bound);
        }
        hash = llvm::hash_combine(hash, system.constraints().size());
        for (const auto& row : system.congruences()) {
            hash = llvm::hash_combine(hash,
                llvm::hash_combine_range(row.coefficients.begin(), row.coefficients.end()), row.residue, row.modulus);
        }
        auto& bucket = buckets[hash];
        for (auto id : bucket) { if (same(systems[id], system)) { return id; } }
        const auto id = systems.size(); systems.push_back(system); bucket.push_back(id); return id;
    }
};
} // namespace mlir::pto::frontiersynch
#endif
