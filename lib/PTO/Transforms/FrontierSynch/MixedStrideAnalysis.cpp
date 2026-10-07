// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A common iteration period makes mixed access strides equal modulo their family size.
#include "PTO/Transforms/FrontierSynch/MixedStrideAnalysis.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "llvm/ADT/APInt.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <tuple>
namespace mlir::pto::frontiersynch {
namespace {
uint64_t modular(uint64_t a, uint64_t b, uint64_t c, uint64_t modulus)
{
    return (llvm::APInt(128, a) * llvm::APInt(128, b) + llvm::APInt(128, c))
        .urem(llvm::APInt(128, modulus)).getZExtValue();
}
bool periodFor(ArrayRef<RotatingFragment> fragments, uint64_t maximum, uint64_t& period, std::string& error)
{
    std::map<uint32_t, std::pair<uint64_t, uint64_t>> families;
    for (const auto& fragment : fragments) {
        if (!fragment.slots) { error = "mixed-stride family has zero slots"; return false; }
        const auto stride = fragment.stride % fragment.slots;
        auto [entry, fresh] = families.try_emplace(fragment.family, fragment.slots, stride);
        if (entry->second.first != fragment.slots) {
            error = "mixed-stride family has inconsistent slot counts"; return false;
        }
        // Comparing every stride with one family representative suffices:
        // p*(s-t)==0 mod b follows by subtracting their equations with that representative.
        if (!fresh) {
            const auto reference = entry->second.second;
            const auto difference = stride >= reference ? stride - reference : reference - stride;
            const auto pairPeriod = fragment.slots / std::gcd(difference, fragment.slots);
            const auto factor = pairPeriod / std::gcd(period, pairPeriod);
            if (factor > maximum / period) {
                error = "mixed-stride numerical phase expansion exceeds representation"; return false;
            }
            period *= factor;
        }
    }
    return true;
}
uint64_t protection(uint64_t original, uint64_t phase,
                    std::map<std::tuple<uint64_t, bool, uint64_t>, uint64_t>& groups)
{
    if (!original) { return 0; }
    const auto flags = original & (invocationProtectionBit | protectionResetBit);
    const auto base = original & ~(invocationProtectionBit | protectionResetBit);
    const bool invocation = original & invocationProtectionBit;
    const auto key = std::make_tuple(base, invocation, invocation ? 0 : phase);
    const auto id = groups.try_emplace(key, groups.size() + 1).first->second;
    return id | flags;
}
} // namespace
MixedStrideExpansion expandMixedStride(ArrayRef<PeriodicPayload> payloads,
    ArrayRef<RotatingFragment> fragments, uint64_t maximumExpandedPayloads, uint64_t maximumExpandedFragments)
{
    MixedStrideExpansion out;
    if (payloads.empty() || !maximumExpandedPayloads || payloads.size() > maximumExpandedPayloads ||
        payloads.size() > UINT32_MAX / 3 || fragments.size() > UINT32_MAX / 2) {
        out.error = "mixed-stride input exceeds payload representation"; return out;
    }
    for (const auto& fragment : fragments) {
        if (fragment.payload >= payloads.size()) { out.error = "invalid mixed-stride payload"; return out; }
    }
    const auto maximumPeriod = std::min(maximumExpandedPayloads / payloads.size(),
                                       uint64_t(UINT32_MAX / 3) / payloads.size());
    if (!periodFor(fragments, maximumPeriod, out.period, out.error)) { return out; }
    if (!fragments.empty() && out.period >
        std::min(maximumExpandedFragments, uint64_t(UINT32_MAX / 2)) / fragments.size()) {
        out.error = "mixed-stride fragment expansion exceeds representation"; return out;
    }
    out.payloads.reserve(payloads.size() * out.period);
    out.fragments.reserve(fragments.size() * out.period);
    std::map<std::tuple<uint64_t, bool, uint64_t>, uint64_t> groups;
    for (uint64_t phase = 0; phase < out.period; ++phase) {
        out.payloads.insert(out.payloads.end(), payloads.begin(), payloads.end());
        for (auto fragment : fragments) {
            fragment.payload += static_cast<uint32_t>(phase * payloads.size());
            fragment.offset = modular(fragment.stride, phase, fragment.offset, fragment.slots);
            fragment.stride = modular(fragment.stride, out.period, 0, fragment.slots);
            fragment.protectionGroup = protection(fragment.protectionGroup, phase, groups);
            out.fragments.push_back(fragment);
        }
    }
    return out;
}
std::optional<std::vector<PeriodicRecord>> expandMixedStrideRecords(
    ArrayRef<PeriodicRecord> records, uint64_t payloadCount, uint64_t period)
{
    if (!payloadCount || !period || period > UINT32_MAX / payloadCount ||
        (!records.empty() && period > UINT32_MAX / records.size())) { return std::nullopt; }
    std::vector<PeriodicRecord> out;
    out.reserve(records.size() * period);
    for (const auto& record : records) {
        if (record.source >= payloadCount || record.target >= payloadCount ||
            (!record.displacement && record.source >= record.target)) { return std::nullopt; }
        for (uint64_t phase = 0; phase < period; ++phase) {
            // Divide first, keeping phase addition below 2*period and avoiding
            // overflow for an original distance near UINT64_MAX.
            const auto quotient = record.displacement / period, residue = record.displacement % period;
            const auto targetPhase = phase + residue;
            const auto carry = targetPhase / period;
            if (carry > UINT64_MAX - quotient) { return std::nullopt; }
            out.push_back({static_cast<uint32_t>(phase * payloadCount + record.source),
                static_cast<uint32_t>((targetPhase % period) * payloadCount + record.target), quotient + carry});
        }
    }
    return out;
}
} // namespace mlir::pto::frontiersynch
