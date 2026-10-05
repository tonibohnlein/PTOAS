// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A constant local effect word refreshes every written atom within one visit.
#include "PTO/Transforms/FrontierSynch/NumericTemplateAnalysis.h"
#include "PTO/Transforms/FrontierSynch/LifetimeScan.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
PeriodicAnalysis reject(const char* reason)
{
    PeriodicAnalysis output;
    output.error = reason;
    return output;
}
bool localEffects(const NumericTemplate& input, std::vector<ExplicitEffects>& word)
{
    for (std::size_t i = 0; i < input.payloads.size(); ++i) {
        const auto& payload = input.payloads[i];
        if (!payload.phase) {
            return false;
        }
        ExplicitEffects occurrence;
        occurrence.payload = static_cast<uint32_t>(i);
        occurrence.pipe = static_cast<uint32_t>(payload.phase->kPipeValue);
        for (const auto& effect : payload.effects) {
            if (effect.discharge != TemplateDischarge::None) {
                if (!effect.atoms.empty()) {
                    return false;
                }
                continue; // Certified whole-base GM discharge retained in input.
            }
            // Undischarged effects are exactly partitioned absolute local ranges.
            for (const auto& range : effect.ranges) {
                if (range.space == AddressSpace::GM) {
                    return false;
                }
            }
            for (auto atom : effect.atoms) {
                if (atom >= input.atoms.size() || atom > std::numeric_limits<uint32_t>::max() ||
                    input.atoms[atom].space == AddressSpace::GM) {
                    return false;
                }
                occurrence.accesses.push_back({static_cast<uint32_t>(atom),
                    effect.mode == SyncAccessMode::Read, effect.mode == SyncAccessMode::Write});
            }
        }
        word.push_back(std::move(occurrence));
    }
    return true;
}
} // namespace
FailureOr<std::vector<ExplicitEffects>> numericTemplateOccurrences(const NumericTemplate& input, unsigned copies)
{
    if (copies == 0 || copies > 2 || input.payloads.size() > std::numeric_limits<uint32_t>::max() / copies) {
        return failure();
    }
    std::vector<ExplicitEffects> word;
    if (!localEffects(input, word)) {
        return failure();
    }
    std::vector<ExplicitEffects> output;
    output.reserve(word.size() * copies);
    HardwareProtectionBuilder protection;
    for (unsigned visit = 0; visit < copies; ++visit) {
        protection.endScope();
        for (std::size_t i = 0; i < word.size(); ++i) {
            auto occurrence = word[i];
            occurrence.payload = static_cast<uint32_t>(output.size());
            SmallVector<uint32_t> accumulatorAtoms;
            for (const auto& access : occurrence.accesses) {
                if (input.atoms[access.atom].space == AddressSpace::ACC) {
                    accumulatorAtoms.push_back(access.atom);
                }
            }
            protection.observe(input.payloads[i].phase->elementOp, occurrence, accumulatorAtoms);
            output.push_back(std::move(occurrence));
        }
    }
    return output;
}
PeriodicAnalysis analyzeNumericTemplate(const NumericTemplate& input)
{
    const bool certified = input.result.state == RecognitionState::Applicable &&
        input.period == 1 && input.refresh == 1;
    if (!certified || input.payloads.size() > std::numeric_limits<uint32_t>::max() / 2) {
        return reject("numeric template lacks period-one certificate");
    }
    auto visits = numericTemplateOccurrences(input, 2);
    if (failed(visits)) {
        return reject("numeric template has invalid physical atom references");
    }
    const auto count = static_cast<uint32_t>(input.payloads.size());
    std::vector<PeriodicPayload> payloads;
    payloads.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        payloads.push_back({(*visits)[i].pipe});
    }
    auto scanned = scanStorageLifetimes(*visits);
    if (!scanned.error.empty()) {
        return reject("numeric template lifetime scan failed");
    }
    std::vector<PeriodicRecord> records;
    for (const auto& generator : scanned.generators) {
        if (generator.source < count) {
            records.push_back({generator.source, generator.target % count, generator.target / count});
        }
    }
    // Every written atom has the same writer one visit later. Two visits
    // therefore contain every source-zero lifetime generator. Read-only atoms
    // yield none. Endpoint truncation handles all finite trip counts, including 0.
    return analyzePeriodicDemands(payloads, records);
}
} // namespace mlir::pto::frontiersynch
