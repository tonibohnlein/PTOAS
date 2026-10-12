// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A constant local effect word refreshes every written atom within one visit.
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplateAnalysis.h"
#include "NumericTemplateRecurringGlobal.h"
#include "PTO/Transforms/FrontierSynch/Finite/LifetimeScanInput.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include <limits>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
PeriodicAnalysis reject(const char* reason)
{
    PeriodicAnalysis output;
    output.error = reason;
    return output;
}
// The template owns atom memberships; visits add only ordinals and protection
// metadata. No complete incidence word is copied before or during the scan.
class NumericOccurrences {
public:
    NumericOccurrences(const NumericTemplate& source, unsigned copies) : input(source)
    {
        if (copies == 0 || copies > 2 || input.payloads.size() > UINT32_MAX / copies ||
            (input.recurringGlobal && (!input.recurringGlobal->applies(input) ||
                                       input.atoms.size() > UINT32_MAX - input.recurringGlobal->atomCount))) {
            return;
        }
        std::map<std::pair<unsigned, uint64_t>, uint64_t> scopes;
        metadata.reserve(input.payloads.size() * copies);
        protection.reserve(input.payloads.size() * copies);
        for (unsigned visit = 0; visit < copies; ++visit) {
            for (const auto& payload : input.payloads) {
                if (!payload.phase || payload.phase->macroOpInstanceId >= 0) {
                    return;
                }
                metadata.push_back(
                    {static_cast<uint32_t>(metadata.size()), static_cast<uint32_t>(payload.phase->kPipeValue)});
                auto group = payload.invocationProtection;
                if (group && !(group & invocationProtectionBit)) {
                    const auto reset = group & protectionResetBit;
                    const auto key = std::make_pair(visit, group & ~protectionResetBit);
                    group = scopes.emplace(key, scopes.size() + 1).first->second | reset;
                }
                protection.push_back(group);
            }
        }
        valid = true;
    }
    bool valid = false;
    std::vector<StorageOccurrenceMetadata> metadata;
    bool accesses(std::size_t ordinal, StorageAccessConsumer consume) const
    {
        const auto site = ordinal % input.payloads.size();
        const auto& payload = input.payloads[site];
        for (auto [effectId, effect] : llvm::enumerate(payload.effects)) {
            if (effect.pendingGlobalGeometry || !effect.validAtomReferences()) {
                return false;
            }
            const auto* relative = input.recurringGlobal ? input.recurringGlobal->find(site, effectId) : nullptr;
            if (relative) {
                for (auto atom : relative->atoms) {
                    const CellAccess access{
                        static_cast<uint32_t>(input.atoms.size() + atom), effect.mode == SyncAccessMode::Read,
                        effect.mode == SyncAccessMode::Write};
                    if (!consume(access)) {
                        return false;
                    }
                }
                continue;
            }
            if (effect.discharge != TemplateDischarge::None) {
                if (!effect.atomReferences().empty()) {
                    return false;
                }
                continue;
            }
            for (const auto& range : effect.ranges) {
                if (range.space == AddressSpace::GM) {
                    return false;
                }
            }
            for (auto atom : effect.atomReferences()) {
                if (atom >= input.atoms.size() || atom > UINT32_MAX || input.atoms[atom].space == AddressSpace::GM) {
                    return false;
                }
                const CellAccess access{
                    static_cast<uint32_t>(atom), effect.mode == SyncAccessMode::Read,
                    effect.mode == SyncAccessMode::Write,
                    input.atoms[atom].space == AddressSpace::ACC ? protection[ordinal] : 0};
                if (!consume(access)) {
                    return false;
                }
            }
        }
        return true;
    }

private:
    const NumericTemplate& input;
    std::vector<uint64_t> protection;
};
} // namespace
FailureOr<std::vector<ExplicitEffects>> numericTemplateOccurrences(const NumericTemplate& input, unsigned copies)
{
    NumericOccurrences source(input, copies);
    if (!source.valid) {
        return failure();
    }
    std::vector<ExplicitEffects> output;
    output.reserve(source.metadata.size());
    for (std::size_t id = 0; id < source.metadata.size(); ++id) {
        const auto occurrence = source.metadata[id];
        ExplicitEffects effects{occurrence.payload, occurrence.pipe, {}};
        if (!source.accesses(id, [&](const CellAccess& access) {
                effects.accesses.push_back(access);
                return true;
            })) {
            return failure();
        }
        output.push_back(std::move(effects));
    }
    return output;
}
bool withNumericTemplateStorage(const NumericTemplate& input, unsigned copies,
                               llvm::function_ref<bool(const StorageScanInput&)> visitor)
{
    NumericOccurrences source(input, copies);
    if (!source.valid) { return false; }
    auto occurrence = [&](std::size_t id) { return source.metadata[id]; };
    auto accesses = [&](std::size_t id, StorageAccessConsumer consume) { return source.accesses(id, consume); };
    return visitor(StorageScanInput{source.metadata.size(), occurrence, accesses});
}
StorageScanResult scanNumericTemplateStorage(
    const NumericTemplate& input, unsigned copies, bool retainWitnesses, StorageScanWork* work)
{
    if (work) { *work = {}; }
    StorageScanResult result;
    const bool valid = withNumericTemplateStorage(input, copies, [&](const StorageScanInput& source) {
        result = scanStorageLifetimes(source, {}, ptoStorageProtection(), retainWitnesses, work);
        return result.error.empty();
    });
    if (!valid && (result.error.empty() || result.error == "invalid access input")) {
        result.error = "numeric template has invalid physical atom references";
    }
    return result;
}
PeriodicAnalysis analyzeNumericTemplate(const NumericTemplate& input)
{
    const bool certified =
        input.result.state == RecognitionState::Applicable && input.period == 1 && input.refresh == 1;
    if (!certified || input.payloads.size() > std::numeric_limits<uint32_t>::max() / 2) {
        return reject("numeric template lacks period-one certificate");
    }
    const auto count = static_cast<uint32_t>(input.payloads.size());
    std::vector<PeriodicPayload> payloads;
    payloads.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (!input.payloads[i].phase) {
            return reject("numeric template has invalid physical atom references");
        }
        payloads.push_back({static_cast<uint32_t>(input.payloads[i].phase->kPipeValue)});
    }
    auto scanned = scanNumericTemplateStorage(input, 2, false, nullptr);
    if (!scanned.error.empty()) {
        return reject(
            scanned.error == "numeric template has invalid physical atom references" ?
                scanned.error.c_str() :
                "numeric template lifetime scan failed");
    }
    std::vector<PeriodicRecord> records;
    for (const auto& generator : scanned.generators) {
        if (generator.source < count) {
            records.push_back({generator.source, generator.target % count, generator.target / count});
        }
    }
    for (const auto& [a, b] : input.uniformConflicts) {
        if (a >= payloads.size() || b >= payloads.size()) {
            return reject("numeric template conflict has invalid payload reference");
        }
        if (ptoStorageProtection().protectsScalar(payloads[a].pipe, payloads[b].pipe)) {
            continue;
        }
        records.push_back({a, b, a < b ? 0U : 1U});
        if (a != b) {
            records.push_back({b, a, 1});
        }
    }
    // Every written atom has the same writer one visit later. Two visits
    // therefore contain every source-zero lifetime generator. Read-only atoms
    // yield none. Endpoint truncation handles all finite trip counts, including 0.
    std::vector<PeriodicRecord> native;
    for (const auto& edge : input.nativePrerequisites) {
        native.push_back({edge.first, edge.second, 0});
    }
    for (const auto& edge : input.valueDemands) {
        records.push_back({edge.first, edge.second, 0});
    }
    return analyzePeriodicDemands(payloads, records, native);
}
} // namespace mlir::pto::frontiersynch
