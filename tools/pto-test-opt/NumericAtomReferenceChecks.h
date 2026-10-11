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
#include "PTO/Transforms/FrontierSynch/Finite/LifetimeScanInput.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
namespace {
namespace atomchecks {
using namespace mlir;
namespace fs = pto::frontiersynch;
using Cache = fs::detail::NumericAtomReferences;
struct Budget {
    uint64_t limit = 65536, charged = 0, attempted = 0;
    bool charge(uint64_t count)
    {
        if (charged > limit || count > limit - charged) {
            return false;
        }
        charged += count;
        return true;
    }
    Cache::Backing get(Cache& cache, ArrayRef<pto::SyncStorageCell> ranges)
    {
        return cache.get(ranges, [&](auto count) { return charge(count); }, [&](auto count) { attempted += count; });
    }
};
bool independentMembership(
    ArrayRef<pto::SyncStorageCell> partition, ArrayRef<pto::SyncStorageCell> ranges, ArrayRef<std::size_t> references)
{
    for (std::size_t id = 0; id < partition.size(); ++id) {
        for (uint64_t byte = partition[id].begin; byte < partition[id].end; ++byte) {
            bool touched = false;
            for (const auto& range : ranges) {
                touched |= pto::sameStorageDomain(range, partition[id]) && byte >= range.begin && byte < range.end;
            }
            if (touched != llvm::is_contained(references, id)) {
                return false;
            }
        }
    }
    return true;
}
bool cacheReuse(MLIRContext* context)
{
    Block bases;
    auto first = bases.addArgument(IndexType::get(context), UnknownLoc::get(context));
    auto second = bases.addArgument(IndexType::get(context), UnknownLoc::get(context));
    SmallVector<pto::SyncStorageCell> partition{
        {pto::AddressSpace::VEC, 0, 4},
        {pto::AddressSpace::VEC, 4, 8},
        {pto::AddressSpace::ACC, 0, 4},
        {pto::AddressSpace::GM, 0, 4, first},
        {pto::AddressSpace::GM, 0, 4, second}};
    for (uint64_t i = 0; i < 66; ++i) {
        partition.push_back({pto::AddressSpace::VEC, 8 + 4 * i, 12 + 4 * i});
    }
    Cache cache(partition, 65536);
    Budget budget;
    SmallVector<pto::SyncStorageCell> wide{{pto::AddressSpace::VEC, 0, 8}};
    auto original = budget.get(cache, wide);
    if (!original || *original != std::vector<std::size_t>({0, 1})) {
        return false;
    }
    for (unsigned i = 0; i < 1000; ++i) {
        if (budget.get(cache, wide) != original) {
            return false;
        }
    }
    if (cache.cost().expansions != 1 || cache.cost().expandedIds != 2 || cache.cost().hits != 1000 ||
        budget.charged != 1003) {
        return false;
    }
    SmallVector<pto::SyncStorageCell> split{{pto::AddressSpace::VEC, 4, 8}, {pto::AddressSpace::VEC, 0, 4}};
    if (budget.get(cache, split) != original || cache.cost().expansions != 1 || cache.cost().hits != 1001) {
        return false;
    }
    for (const auto& cell : partition) {
        auto value = budget.get(cache, {cell});
        if (!value || !independentMembership(partition, {cell}, *value)) {
            return false;
        }
    }
    for (uint64_t i = 0; i < 66; ++i) {
        if (!budget.get(cache, {{pto::AddressSpace::VEC, 8 + 4 * i, 12 + 4 * i}})) {
            return false;
        }
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
    if (!x || !y || *x != std::vector<std::size_t>({0}) || *y != std::vector<std::size_t>({1})) {
        return false;
    }
    Cache failed(first, 65536);
    Budget refusal{0, 0, 0};
    if (refusal.get(failed, {{pto::AddressSpace::VEC, 0, 4}}) || !refusal.attempted) {
        return false;
    }
    refusal.limit = 65536;
    if (!refusal.get(failed, {{pto::AddressSpace::VEC, 0, 4}}) || failed.cost().expansions != 2) {
        return false;
    }
    if (budget.get(a, {{pto::AddressSpace::VEC, 4, 0}}) || budget.get(a, {{pto::AddressSpace::VEC, 1, 4}}) ||
        budget.get(a, {{pto::AddressSpace::VEC, 0, 3}}) || budget.get(a, {{pto::AddressSpace::VEC, 0, 8}})) {
        return false;
    }
    SmallVector<pto::SyncStorageCell> overlapping{{pto::AddressSpace::VEC, 0, 4}, {pto::AddressSpace::VEC, 2, 6}};
    Cache malformed(overlapping, 65536);
    return !budget.get(malformed, {{pto::AddressSpace::VEC, 0, 4}});
}
bool sameScan(const fs::StorageScanResult& a, const fs::StorageScanResult& b, bool witnesses = true)
{
    if (a.error != b.error || a.protectedHazards != b.protectedHazards ||
        !llvm::equal(
            a.generators, b.generators, [](auto x, auto y) { return x.source == y.source && x.target == y.target; })) {
        return false;
    }
    return !witnesses || llvm::equal(a.witnesses, b.witnesses, [](auto x, auto y) {
        return x.generator == y.generator && x.atom == y.atom && x.hazard == y.hazard;
    });
}
bool borrowedScanBoundaries()
{
    std::vector<fs::ExplicitEffects> word{{17, 0, {{3, false, true}}}, {29, 1, {{3, true, false}}}};
    auto metadata = [&](std::size_t id) { return fs::StorageOccurrenceMetadata{word[id].payload, word[id].pipe}; };
    auto accesses = [&](std::size_t id, fs::StorageAccessConsumer consume) {
        for (const auto& access : word[id].accesses) {
            if (!consume(access)) {
                return false;
            }
        }
        return true;
    };
    auto check = [&](llvm::ArrayRef<fs::StorageGenerator> prerequisites, const char* error) {
        const auto result =
            fs::scanStorageLifetimes(fs::StorageScanInput{word.size(), metadata, accesses}, prerequisites, {}, false);
        return result.error == error && result.witnesses.empty() &&
               sameScan(result, fs::scanStorageLifetimes(word, prerequisites), false);
    };
    word[1].accesses.front().atom = 4;
    const auto supplied =
        fs::scanStorageLifetimes(fs::StorageScanInput{word.size(), metadata, accesses}, {{17, 29}}, {}, false);
    if (!supplied.error.empty() || supplied.generators.size() != 1 || supplied.generators.front().source != 17 ||
        supplied.generators.front().target != 29 || !supplied.witnesses.empty()) {
        return false;
    }
    word[1].accesses.front().atom = 3;
    if (!check({{17, 29}}, "") || !check({{29, 17}}, "nonforward or absent prerequisite endpoint") ||
        !check({{17, 30}}, "nonforward or absent prerequisite endpoint")) {
        return false;
    }
    word[1].payload = 17;
    if (!check({}, "duplicate payload identity")) {
        return false;
    }
    word[1].payload = 29;
    word[0].accesses = {{3, false, false}};
    if (!check({}, "empty access mode")) {
        return false;
    }
    word[0].accesses = {{3, false, true, 1}, {3, true, false, 2}};
    if (!check({}, "inconsistent access protection")) {
        return false;
    }
    word[0].accesses = {{3, true, false, 1}};
    if (!check({}, "protection requires writers on one pipe")) {
        return false;
    }
    word[0].accesses = {{3, false, true}};
    word.push_back({41, 2, {{3, false, true}}});
    auto unavailable = [&](std::size_t id, fs::StorageAccessConsumer consume) {
        return id < 2 && accesses(id, consume);
    };
    fs::StorageScanWork work;
    work.incidenceVisits = 99;
    auto failed =
        fs::scanStorageLifetimes(fs::StorageScanInput{word.size(), metadata, unavailable}, {}, {}, false, &work);
    if (failed.error != "invalid access input" || !failed.generators.empty() || !failed.witnesses.empty() ||
        work.incidenceVisits != 2 || work.consolidatedVisits != 2 || !check({}, "")) {
        return false;
    }
    word.clear();
    return check({}, "");
}
bool ownedProtection(MLIRContext* context)
{
    pto::CompoundInstanceElement phase(0, {}, {}, pto::PipelineType::PIPE_M, OperationName("func.call", context));
    fs::NumericTemplate legacy;
    legacy.atoms.push_back({pto::AddressSpace::ACC, 0, 4});
    for (uint64_t group : {7, 9}) {
        fs::TemplatePayload payload;
        payload.phase = &phase;
        payload.invocationProtection = group;
        for (auto mode : {pto::SyncAccessMode::Read, pto::SyncAccessMode::Write}) {
            fs::TemplateEffect effect;
            effect.mode = mode;
            effect.atoms.push_back(0);
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
                effect.atoms.clear();
                weak = effect.sharedAtoms;
            }
        }
    }
    auto retained = shared;
    shared = fs::NumericTemplate{};
    if (weak.expired()) {
        return false;
    }
    auto expected = fs::numericTemplateOccurrences(legacy, 2);
    auto actual = fs::numericTemplateOccurrences(retained, 2);
    if (failed(expected) || failed(actual) || expected->size() != actual->size()) {
        return false;
    }
    for (auto [a, b] : llvm::zip(*expected, *actual)) {
        if (a.pipe != b.pipe || a.payload != b.payload || a.accesses.size() != b.accesses.size()) {
            return false;
        }
        for (auto [x, y] : llvm::zip(a.accesses, b.accesses)) {
            if (x.atom != y.atom || x.read != y.read || x.write != y.write || x.protectionGroup != y.protectionGroup ||
                !y.protectionGroup) {
                return false;
            }
        }
    }
    for (unsigned visit = 0; visit < 2; ++visit) {
        for (unsigned site = 0; site < 2; ++site) {
            for (const auto& access : (*actual)[2 * visit + site].accesses) {
                if (access.protectionGroup != 2 * visit + site + 1) {
                    return false;
                }
            }
        }
    }
    auto baseline = fs::scanStorageLifetimes(*actual, {}, fs::ptoStorageProtection());
    auto streamed = fs::scanNumericTemplateStorage(retained, 2);
    if (!sameScan(baseline, streamed)) {
        return false;
    }
    for (auto group : {fs::invocationProtectionBit | 7, fs::protectionResetBit | 7}) {
        auto scoped = retained;
        scoped.payloads[0].invocationProtection = group;
        scoped.payloads[1].invocationProtection = group & ~fs::protectionResetBit;
        auto word = fs::numericTemplateOccurrences(scoped, 2);
        if (failed(word) || !sameScan(
                                fs::scanStorageLifetimes(*word, {}, fs::ptoStorageProtection()),
                                fs::scanNumericTemplateStorage(scoped, 2))) {
            return false;
        }
        for (unsigned visit = 0; visit < 2; ++visit) {
            for (unsigned site = 0; site < 2; ++site) {
                auto expectedGroup = (group & fs::invocationProtectionBit) ? group : uint64_t(visit + 1);
                if (site == 0) {
                    expectedGroup |= group & fs::protectionResetBit;
                }
                for (const auto& access : (*word)[2 * visit + site].accesses) {
                    if (access.protectionGroup != expectedGroup) {
                        return false;
                    }
                }
            }
        }
    }
    retained.payloads.front().effects.front().atoms.push_back(0);
    if (succeeded(fs::numericTemplateOccurrences(retained))) {
        return false;
    }
    retained = fs::NumericTemplate{};
    return weak.expired();
}
bool sharedScanFamilies(MLIRContext* context)
{
    pto::CompoundInstanceElement phase(0, {}, {}, pto::PipelineType::PIPE_V, OperationName("func.call", context));
    for (unsigned atoms : {1U, 3U, 16U, 64U}) {
        for (unsigned effects : {2U, 4U, 16U}) {
            fs::NumericTemplate form;
            auto backing = std::make_shared<std::vector<std::size_t>>();
            for (unsigned id = 0; id < atoms; ++id) {
                form.atoms.push_back({pto::AddressSpace::VEC, 4 * id, 4 * id + 4});
                backing->push_back(id);
            }
            for (unsigned id = 0; id < effects; ++id) {
                fs::TemplatePayload payload;
                payload.phase = &phase;
                for (auto mode : {pto::SyncAccessMode::Read, pto::SyncAccessMode::Write}) {
                    fs::TemplateEffect effect;
                    effect.mode = mode;
                    effect.sharedAtoms = backing;
                    payload.effects.push_back(std::move(effect));
                }
                form.payloads.push_back(std::move(payload));
            }
            auto explicitWord = fs::numericTemplateOccurrences(form, 2);
            if (failed(explicitWord)) {
                return false;
            }
            const auto baseline = fs::scanStorageLifetimes(*explicitWord, {}, fs::ptoStorageProtection());
            fs::StorageScanWork full, compact;
            const auto actual = fs::scanNumericTemplateStorage(form, 2, true, &full);
            const auto generators = fs::scanNumericTemplateStorage(form, 2, false, &compact);
            const uint64_t c = 4ULL * atoms * effects;
            if (!sameScan(baseline, actual) || !sameScan(actual, generators, false) || !generators.witnesses.empty() ||
                full.incidenceVisits != c || compact.incidenceVisits != c || full.consolidatedVisits != c / 2 ||
                compact.consolidatedVisits != c / 2 || full.hazardWitnesses != 2ULL * atoms * (2 * effects - 1) ||
                full.retainedWitnesses != full.hazardWitnesses || compact.retainedWitnesses != 0 ||
                compact.hazardWitnesses != full.hazardWitnesses || actual.generators.size() != 2 * effects - 1 ||
                full.counterOverflow || compact.counterOverflow) {
                return false;
            }
            for (const auto& payload : form.payloads) {
                for (const auto& effect : payload.effects) {
                    if (effect.sharedAtoms != backing || !effect.atoms.empty()) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}
} // namespace atomchecks
bool numericAtomReferenceChecks(mlir::MLIRContext* context)
{
    return atomchecks::cacheReuse(context) && atomchecks::cacheContextsAndFailure() &&
           atomchecks::borrowedScanBoundaries() && atomchecks::ownedProtection(context) &&
           atomchecks::sharedScanFamilies(context);
}
} // namespace
#endif
