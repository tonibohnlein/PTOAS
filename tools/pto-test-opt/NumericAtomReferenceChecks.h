// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independently check partition-local reuse, owned backing and protection parity.
#ifndef PTO_TEST_NUMERICATOMREFERENCECHECKS_H
#define PTO_TEST_NUMERICATOMREFERENCECHECKS_H
#include "../../lib/PTO/Transforms/FrontierSynch/Structured/NumericTemplateAtomReferences.h"
namespace {
namespace atomchecks {
using namespace mlir;
namespace fs = pto::frontiersynch;
using Cache = fs::detail::NumericAtomReferences;
struct Budget {
    uint64_t limit = 65536, charged = 0, attempted = 0;
    bool charge(uint64_t count)
    {
        if (charged > limit || count > limit - charged) { return false; }
        charged += count; return true;
    }
    Cache::Backing get(Cache& cache, ArrayRef<pto::SyncStorageCell> ranges)
    {
        return cache.get(ranges, [&](auto count) { return charge(count); },
                         [&](auto count) { attempted += count; });
    }
};
bool independentMembership(ArrayRef<pto::SyncStorageCell> partition, ArrayRef<pto::SyncStorageCell> ranges,
    ArrayRef<std::size_t> references)
{
    for (std::size_t id = 0; id < partition.size(); ++id) {
        for (uint64_t byte = partition[id].begin; byte < partition[id].end; ++byte) {
            bool touched = false;
            for (const auto& range : ranges) {
                touched |= pto::sameStorageDomain(range, partition[id]) && byte >= range.begin && byte < range.end;
            }
            if (touched != llvm::is_contained(references, id)) { return false; }
        }
    }
    return true;
}
bool cacheReuse(MLIRContext* context)
{
    Block bases;
    auto first = bases.addArgument(IndexType::get(context), UnknownLoc::get(context));
    auto second = bases.addArgument(IndexType::get(context), UnknownLoc::get(context));
    SmallVector<pto::SyncStorageCell> partition{{pto::AddressSpace::VEC, 0, 4}, {pto::AddressSpace::VEC, 4, 8},
        {pto::AddressSpace::ACC, 0, 4}, {pto::AddressSpace::GM, 0, 4, first},
        {pto::AddressSpace::GM, 0, 4, second}};
    for (uint64_t i = 0; i < 66; ++i) {
        partition.push_back({pto::AddressSpace::VEC, 8 + 4 * i, 12 + 4 * i});
    }
    Cache cache(partition, 65536);
    Budget budget;
    SmallVector<pto::SyncStorageCell> wide{{pto::AddressSpace::VEC, 0, 8}};
    auto original = budget.get(cache, wide);
    if (!original || *original != std::vector<std::size_t>({0, 1})) { return false; }
    for (unsigned i = 0; i < 1000; ++i) {
        if (budget.get(cache, wide) != original) { return false; }
    }
    if (cache.cost().expansions != 1 || cache.cost().expandedIds != 2 || cache.cost().hits != 1000 ||
        budget.charged != 1003) { return false; }
    SmallVector<pto::SyncStorageCell> split{{pto::AddressSpace::VEC, 4, 8}, {pto::AddressSpace::VEC, 0, 4}};
    if (budget.get(cache, split) != original || cache.cost().expansions != 1 || cache.cost().hits != 1001) {
        return false;
    }
    for (const auto& cell : partition) {
        auto value = budget.get(cache, {cell});
        if (!value || !independentMembership(partition, {cell}, *value)) { return false; }
    }
    for (uint64_t i = 0; i < 66; ++i) {
        if (!budget.get(cache, {{pto::AddressSpace::VEC, 8 + 4 * i, 12 + 4 * i}})) { return false; }
    }
    return *original == std::vector<std::size_t>({0, 1});
}
bool cacheContextsAndFailure()
{
    SmallVector<pto::SyncStorageCell> first{{pto::AddressSpace::VEC, 0, 4}, {pto::AddressSpace::ACC, 0, 4}};
    SmallVector<pto::SyncStorageCell> second{{pto::AddressSpace::ACC, 0, 4}, {pto::AddressSpace::VEC, 0, 4}};
    Cache a(first, 65536), b(second, 65536);
    Budget budget;
    auto x = budget.get(a, {{pto::AddressSpace::VEC, 0, 4}});
    auto y = budget.get(b, {{pto::AddressSpace::VEC, 0, 4}});
    if (!x || !y || *x != std::vector<std::size_t>({0}) || *y != std::vector<std::size_t>({1})) { return false; }
    Cache failed(first, 65536);
    Budget refusal{0, 0, 0};
    if (refusal.get(failed, {{pto::AddressSpace::VEC, 0, 4}}) || !refusal.attempted) { return false; }
    refusal.limit = 65536;
    if (!refusal.get(failed, {{pto::AddressSpace::VEC, 0, 4}}) || failed.cost().expansions != 2) { return false; }
    if (budget.get(a, {{pto::AddressSpace::VEC, 4, 0}}) ||
        budget.get(a, {{pto::AddressSpace::VEC, 1, 4}}) ||
        budget.get(a, {{pto::AddressSpace::VEC, 0, 3}}) ||
        budget.get(a, {{pto::AddressSpace::VEC, 0, 8}})) { return false; }
    SmallVector<pto::SyncStorageCell> overlapping{{pto::AddressSpace::VEC, 0, 4},
        {pto::AddressSpace::VEC, 2, 6}};
    Cache malformed(overlapping, 65536);
    return !budget.get(malformed, {{pto::AddressSpace::VEC, 0, 4}});
}
bool ownedProtection(MLIRContext* context)
{
    pto::CompoundInstanceElement phase(0, {}, {}, pto::PipelineType::PIPE_M, OperationName("func.call", context));
    fs::NumericTemplate legacy;
    legacy.atoms.push_back({pto::AddressSpace::ACC, 0, 4});
    for (uint64_t group : {7, 9}) {
        fs::TemplatePayload payload;
        payload.phase = &phase; payload.invocationProtection = group;
        for (auto mode : {pto::SyncAccessMode::Read, pto::SyncAccessMode::Write}) {
            fs::TemplateEffect effect;
            effect.mode = mode; effect.atoms.push_back(0);
            payload.effects.push_back(std::move(effect));
        }
        legacy.payloads.push_back(std::move(payload));
    }
    auto shared = legacy;
    std::weak_ptr<const std::vector<std::size_t>> weak;
    {
        Cache cache(shared.atoms, 65536);
        Budget budget;
        for (auto& payload : shared.payloads) {
            for (auto& effect : payload.effects) {
                effect.sharedAtoms = budget.get(cache, shared.atoms);
                effect.atoms.clear(); weak = effect.sharedAtoms;
            }
        }
    }
    auto retained = shared;
    shared = fs::NumericTemplate{};
    if (weak.expired()) { return false; }
    auto expected = fs::numericTemplateOccurrences(legacy, 2);
    auto actual = fs::numericTemplateOccurrences(retained, 2);
    if (failed(expected) || failed(actual) || expected->size() != actual->size()) { return false; }
    for (auto [a, b] : llvm::zip(*expected, *actual)) {
        if (a.pipe != b.pipe || a.payload != b.payload || a.accesses.size() != b.accesses.size()) { return false; }
        for (auto [x, y] : llvm::zip(a.accesses, b.accesses)) {
            if (x.atom != y.atom || x.read != y.read || x.write != y.write ||
                x.protectionGroup != y.protectionGroup || !y.protectionGroup) { return false; }
        }
    }
    retained.payloads.front().effects.front().atoms.push_back(0);
    if (succeeded(fs::numericTemplateOccurrences(retained))) { return false; }
    retained = fs::NumericTemplate{};
    return weak.expired();
}
} // namespace atomchecks
bool numericAtomReferenceChecks(mlir::MLIRContext* context)
{
    return atomchecks::cacheReuse(context) && atomchecks::cacheContextsAndFailure() &&
        atomchecks::ownedProtection(context);
}
} // namespace
#endif
