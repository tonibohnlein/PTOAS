// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/BoundedLifetime.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include <map>
#include <numeric>
#include <set>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
struct Mode {
    Id read, write;
    uint64_t group;
};
using Witnesses = std::map<std::pair<uint32_t, uint32_t>, std::vector<LifetimeStorageWitness>>;
// Prefixes contain only accesses strictly between the source and target.
// A read-modify-write access closes a lifetime; it is never a read-only user.
class WitnessPrefix {
public:
    explicit WitnessPrefix(RegionExpressions& expressions) : e(expressions), noReaders(e.boolean(true)) {}
    void append(Witnesses& witnesses, uint32_t cell, uint32_t source, uint32_t target,
                uint32_t sourcePipe, uint32_t targetPipe, Mode a, Mode b, Id generator)
    {
        auto sourceRead = e.land(a.read, e.lnot(a.write));
        auto targetRead = e.land(b.read, e.lnot(b.write));
        auto add = [&](StorageHazard hazard, uint32_t pipe, Id mode, Id prefix) {
            auto guard = e.land(generator, e.land(mode, prefix));
            if (e.constantValue(guard) != 0) {
                witnesses[{source, target}].push_back({cell, hazard, pipe, guard});
            }
        };
        add(StorageHazard::RAW, targetPipe, e.land(a.write, targetRead), absent(targetPipe));
        add(StorageHazard::WAR, sourcePipe, e.land(sourceRead, b.write), absent(sourcePipe));
        add(StorageHazard::WAW, targetPipe, e.land(a.write, b.write), noReaders);
        auto notReader = e.lnot(targetRead);
        readers[targetPipe] = e.land(absent(targetPipe), notReader);
        noReaders = e.land(noReaders, notReader);
    }
private:
    RegionExpressions& e;
    Id noReaders;
    std::map<uint32_t, Id> readers;
    Id absent(uint32_t pipe) const
    {
        auto found = readers.find(pipe);
        return found == readers.end() ? e.boolean(true) : found->second;
    }
};
void retainWitnesses(RegionExpressions& e, const LifetimeWindowInput& input,
                     const Witnesses& witnesses, LifetimeWindowAnalysis& out)
{
    for (auto edge : out.window.retained) {
        if (edge.source >= input.sites) { continue; }
        out.sourceDemands.push_back(edge);
        auto& selected = out.sourceWitnesses.emplace_back();
        auto found = witnesses.find({edge.source, edge.target});
        if (found != witnesses.end()) {
            for (auto witness : found->second) {
                witness.guard = e.land(edge.guard, witness.guard);
                if (e.constantValue(witness.guard) != 0) { selected.push_back(witness); }
            }
        }
        for (auto supplied : input.prerequisites) {
            if (supplied.source == edge.source && supplied.target == edge.target) {
                auto guard = e.land(edge.guard, supplied.guard);
                if (e.constantValue(guard) != 0) {
                    selected.push_back({0, StorageHazard::Supplied, input.payloads[edge.target].pipe, guard});
                }
            }
        }
    }
}
} // namespace
LifetimeWindowAnalysis analyzeLifetimeWindow(RegionExpressions& e, const LifetimeWindowInput& input)
{
    LifetimeWindowAnalysis out;
    if (!input.sites || input.span >= UINT32_MAX || input.sites > UINT32_MAX / (input.span + 1) ||
        input.payloads.size() != input.sites * (input.span + 1) ||
        (!input.operations.empty() && input.operations.size() != input.payloads.size())) {
        out.error = "bounded window has invalid potential occurrence dimensions";
        return out;
    }
    std::map<uint32_t, std::map<uint32_t, Mode>> cells;
    for (const auto& access : input.accesses) {
        if (access.payload >= input.payloads.size() || access.read >= e.size() || access.write >= e.size() ||
            !e.isBoolean(access.read) || !e.isBoolean(access.write)) {
            out.error = "invalid bounded-window access predicate";
            return out;
        }
        auto [found, added] =
            cells[access.cell].emplace(access.payload, Mode{access.read, access.write, access.protectionGroup});
        if (!added) {
            found->second = {
                e.lor(found->second.read, access.read), e.lor(found->second.write, access.write),
                found->second.group == access.protectionGroup ? access.protectionGroup : 0};
        }
    }
    // Validate before using occurrence predicates in storage guards.
    for (const auto& payload : input.payloads) {
        if (payload.present >= e.size() || !e.isBoolean(payload.present)) {
            out.error = "invalid bounded-window presence";
            return out;
        }
    }
    // The shared protection contract only removes same-pipe hazards. When
    // that removes one side of a writer-chain witness, native completion order
    // carries readiness forward and native start order carries release forward.
    // Arbitrary pair exemptions would not justify the no-writer test below.
    std::vector<GuardedRankEdge> generators = input.prerequisites;
    Witnesses witnesses;
    auto sameOperation = [&](uint32_t a, uint32_t b) {
        return !input.operations.empty() && input.operations[a] && input.operations[a] == input.operations[b];
    };
    for (auto& [cell, uses] : cells) {
        std::set<uint64_t> operations;
        bool plain = true;
        for (auto& [id, mode] : uses) {
            if (!input.operations.empty() && input.operations[id]) {
                plain &= operations.insert(input.operations[id]).second;
            }
            auto present = input.payloads[id].present;
            mode = {e.land(present, mode.read), e.land(present, mode.write), mode.group};
        }
        for (auto a = uses.begin(); a != uses.end(); ++a) {
            auto noWriter = e.boolean(true);
            WitnessPrefix prefix(e);
            for (auto b = std::next(a); b != uses.end(); ++b) {
                auto [ar, aw, ag] = a->second;
                auto [br, bw, bg] = b->second;
                auto conflict = e.lor(e.land(aw, e.lor(br, bw)), e.land(ar, bw));
                if (sameOperation(a->first, b->first)) {
                    conflict = e.boolean(false);
                }
                const auto p = input.payloads[a->first].pipe, q = input.payloads[b->first].pipe;
                if (input.storageProtection.protectsScalar(p, q)) {
                    conflict = e.boolean(false);
                }
                if (hardwareProtectsConflict(p, ag, q, bg)) {
                    // The certified accumulation contract protects writer
                    // pairs, including their RMW input. Read-only users keep
                    // readiness/release; resets never consume protection.
                    conflict = e.land(conflict, e.lnot(e.land(aw, bw)));
                }
                auto generator = e.land(conflict, noWriter);
                generators.push_back({a->first, b->first, generator});
                if (plain && a->first < input.sites) {
                    prefix.append(witnesses, cell, a->first, b->first, p, q, a->second, b->second, generator);
                }
                // Simultaneous envelopes do not form a writer chain. As in
                // finite guarded analysis, keep raw pairs for such cells.
                if (plain) {
                    noWriter = e.land(noWriter, e.lnot(bw));
                }
                ++out.accessPairs;
            }
        }
    }
    out.window = reduceGuardedRanks(e, input.payloads, generators, input.nativePrerequisites);
    out.error = out.window.error;
    if (!out.error.empty()) {
        return out;
    }
    retainWitnesses(e, input, witnesses, out);
    return out;
}
RefreshCertificate certifyRotatingRefresh(
    llvm::ArrayRef<PeriodicPayload> payloads, llvm::ArrayRef<RotatingFragment> fragments,
    llvm::ArrayRef<uint8_t> unconditional, uint64_t prerequisiteSpan)
{
    RefreshCertificate result;
    if (unconditional.size() != payloads.size()) {
        result.error = "refresh presence list does not match payloads";
        return result;
    }
    std::vector<uint8_t> coveredWriters;
    for (const auto& fragment : fragments) {
        if (fragment.payload >= unconditional.size()) {
            result.error = "refresh fragment has no payload"; return result;
        }
        coveredWriters.push_back(fragment.write && unconditional[fragment.payload]);
    }
    return certifyRotatingRefreshCoverage(payloads, fragments, coveredWriters, prerequisiteSpan);
}
RefreshCertificate certifyRotatingRefreshCoverage(
    llvm::ArrayRef<PeriodicPayload> payloads, llvm::ArrayRef<RotatingFragment> fragments,
    llvm::ArrayRef<uint8_t> coveredWriters, uint64_t prerequisiteSpan)
{
    RefreshCertificate result;
    if (coveredWriters.size() != fragments.size()) {
        result.error = "refresh coverage list does not match fragments"; return result;
    }
    auto normalized = extractRotatingGenerators(payloads, fragments);
    if (!normalized.error.empty()) {
        result.error = normalized.error;
        return result;
    }
    using Orbit = std::tuple<uint32_t, uint32_t, uint64_t>;
    std::set<Orbit> writable, refreshed;
    result.span = prerequisiteSpan;
    for (std::size_t i = 0; i < fragments.size(); ++i) {
        const auto& f = fragments[i];
        if (f.protectionGroup) {
            result.error = "refresh certificate requires ordinary storage generators";
            return result;
        }
        if (!f.write) {
            continue;
        }
        const auto divisor = std::gcd(f.stride % f.slots, f.slots);
        Orbit orbit{f.family, f.atom, f.offset % divisor};
        writable.insert(orbit);
        if (coveredWriters[i]) {
            refreshed.insert(orbit);
            result.span = std::max(result.span, f.slots / divisor);
        }
    }
    for (const auto& orbit : writable) {
        if (!refreshed.count(orbit)) {
            result.error = "a writable orbit has no collectively complete refresh map";
            return result;
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
