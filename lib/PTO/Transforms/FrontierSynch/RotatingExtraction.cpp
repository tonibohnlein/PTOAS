// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Sort accesses to one representative slot by modular phase and body order.
#include "PTO/Transforms/FrontierSynch/RotatingExtraction.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "llvm/ADT/APInt.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <tuple>
#include <unordered_map>
namespace mlir::pto::frontiersynch {
namespace {
uint64_t inverse(uint64_t value, uint64_t modulus)
{
    uint64_t oldRemainder = modulus, remainder = value;
    // 256-bit signed intermediates contain products of a 64-bit quotient and
    // the extended-Euclidean coefficients, without host signed overflow.
    llvm::APInt oldCoefficient(256, 0), coefficient(256, 1);
    while (remainder != 0) {
        const auto quotient = oldRemainder / remainder;
        const auto nextRemainder = oldRemainder % remainder;
        const auto nextCoefficient = oldCoefficient - llvm::APInt(256, quotient) * coefficient;
        oldRemainder = remainder;
        remainder = nextRemainder;
        oldCoefficient = coefficient;
        coefficient = nextCoefficient;
    }
    const llvm::APInt bound(256, modulus);
    auto result = oldCoefficient.srem(bound);
    if (result.isNegative()) {
        result += bound;
    }
    return result.getZExtValue();
}
uint64_t modularPhase(uint64_t offset, uint64_t inverseStride, uint64_t period)
{
    if (period == 1) {
        return 0;
    }
    const auto product = llvm::APInt(128, offset) * llvm::APInt(128, inverseStride);
    const auto phase = product.urem(llvm::APInt(128, period)).getZExtValue();
    return phase == 0 ? 0 : period - phase;
}
struct Family {
    uint64_t slots;
    uint64_t stride;
    uint64_t divisor;
    uint64_t period;
    uint64_t inverseStride;
};
struct Access {
    RotatingFragment fragment;
    uint64_t residue;
    uint64_t phase;
    uint64_t period;
};
auto key(const Access& access)
{
    return std::make_tuple(access.fragment.family, access.fragment.atom, access.residue,
                           access.phase, access.fragment.payload);
}
bool sameGroup(const Access& left, const Access& right)
{
    return left.fragment.family == right.fragment.family && left.fragment.atom == right.fragment.atom &&
        left.residue == right.residue;
}
uint64_t distance(const Access& source, const Access& target)
{
    if (source.phase < target.phase) {
        return target.phase - source.phase;
    }
    if (source.phase > target.phase) {
        return source.period - (source.phase - target.phase);
    }
    return source.fragment.payload < target.fragment.payload ? 0 : source.period;
}
class Extractor {
public:
    RotatingExtraction output;
    explicit Extractor(llvm::ArrayRef<PeriodicPayload> input) : payloads(input) {}
    bool run(llvm::ArrayRef<RotatingFragment> fragments)
    {
        const auto maxID = std::numeric_limits<uint32_t>::max();
        if (payloads.size() > maxID / 3 || fragments.size() > maxID / 2) {
            output.error = "rotating generator identity overflow";
            return false;
        }
        for (const auto& fragment : fragments) {
            if (!append(fragment)) {
                return false;
            }
        }
        if (!consolidate()) {
            return false;
        }
        for (std::size_t first = 0; first < accesses.size();) {
            std::size_t end = first + 1;
            while (end < accesses.size() && sameGroup(accesses[first], accesses[end])) {
                ++end;
            }
            sweep(first, end);
            first = end;
        }
        deduplicate();
        return true;
    }
private:
    llvm::ArrayRef<PeriodicPayload> payloads;
    std::unordered_map<uint32_t, Family> families;
    std::unordered_map<uint64_t, uint32_t> groupPipes;
    std::vector<Access> accesses;
    bool append(RotatingFragment fragment)
    {
        if (fragment.payload >= payloads.size() || fragment.slots == 0 || (!fragment.read && !fragment.write)) {
            output.error = "absent rotating payload, zero slots, or empty access mode";
            return false;
        }
        fragment.stride %= fragment.slots;
        fragment.offset %= fragment.slots;
        if (!family(fragment) || !protection(fragment)) {
            return false;
        }
        const auto& descriptor = families.at(fragment.family);
        const auto residue = fragment.offset % descriptor.divisor;
        const auto phase = modularPhase(fragment.offset / descriptor.divisor,
                                        descriptor.inverseStride, descriptor.period);
        accesses.push_back({fragment, residue, phase, descriptor.period});
        return true;
    }
    bool family(const RotatingFragment& fragment)
    {
        auto found = families.find(fragment.family);
        if (found != families.end()) {
            if (found->second.slots != fragment.slots || found->second.stride != fragment.stride) {
                output.error = "inconsistent rotating family slots or stride";
                return false;
            }
            return true;
        }
        const auto divisor = std::gcd(fragment.stride, fragment.slots);
        const auto period = fragment.slots / divisor;
        const auto inverseStride = period == 1 ? 0 : inverse(fragment.stride / divisor, period);
        families.emplace(fragment.family, Family{fragment.slots, fragment.stride, divisor, period, inverseStride});
        output.refreshBound = std::max(output.refreshBound, period);
        return true;
    }
    bool protection(const RotatingFragment& fragment)
    {
        if (fragment.protectionGroup == 0) {
            return true;
        }
        const auto pipe = payloads[fragment.payload].pipe;
        const auto found = groupPipes.emplace(fragment.protectionGroup, pipe).first;
        if (!fragment.write || found->second != pipe) {
            output.error = "rotating protection requires writers on one pipe";
            return false;
        }
        return true;
    }
    bool consolidate()
    {
        std::sort(accesses.begin(), accesses.end(), [](const Access& a, const Access& b) { return key(a) < key(b); });
        std::size_t kept = 0;
        for (const auto& access : accesses) {
            if (kept != 0 && key(accesses[kept - 1]) == key(access)) {
                auto& previous = accesses[kept - 1].fragment;
                if (previous.protectionGroup != access.fragment.protectionGroup) {
                    output.error = "inconsistent rotating access protection";
                    return false;
                }
                previous.read |= access.fragment.read;
                previous.write |= access.fragment.write;
            } else {
                accesses[kept++] = access;
            }
        }
        accesses.resize(kept);
        return true;
    }
    void emit(const Access& source, const Access& target)
    {
        const auto advance = distance(source, target);
        // Local groups protect one visit; invocation proofs also cover wraps.
        const bool sameScope = advance == 0 || (source.fragment.protectionGroup & invocationProtectionBit);
        if (sameScope && source.fragment.write && target.fragment.write &&
            hardwareProtectsConflict(payloads[source.fragment.payload].pipe, source.fragment.protectionGroup,
                                      payloads[target.fragment.payload].pipe, target.fragment.protectionGroup)) {
            ++output.protectedHazards;
            return;
        }
        output.generators.push_back({source.fragment.payload, target.fragment.payload, advance});
    }
    void sweep(std::size_t first, std::size_t end)
    {
        auto previous = end;
        for (auto i = first; i < end; ++i) {
            if (accesses[i].fragment.write) {
                previous = i;
            }
        }
        if (previous == end) {
            return;
        }
        for (auto i = first; i < end; ++i) {
            emit(accesses[previous], accesses[i]);
            if (accesses[i].fragment.write) {
                previous = i;
            }
        }
        auto next = first;
        while (!accesses[next].fragment.write) {
            ++next;
        }
        for (auto i = end; i != first; --i) {
            const auto current = i - 1;
            if (accesses[current].fragment.write) {
                next = current;
            } else {
                emit(accesses[current], accesses[next]);
            }
        }
    }
    void deduplicate()
    {
        auto recordKey = [](const PeriodicRecord& record) {
            return std::make_tuple(record.source, record.target, record.displacement);
        };
        std::sort(output.generators.begin(), output.generators.end(), [&](const auto& a, const auto& b) {
            return recordKey(a) < recordKey(b);
        });
        output.generators.erase(std::unique(output.generators.begin(), output.generators.end(), [&](const auto& a,
                                                                                                  const auto& b) {
            return recordKey(a) == recordKey(b);
        }), output.generators.end());
    }
};
} // namespace
RotatingExtraction extractRotatingGenerators(llvm::ArrayRef<PeriodicPayload> payloads,
                                             llvm::ArrayRef<RotatingFragment> fragments)
{
    Extractor extractor(payloads);
    if (!extractor.run(fragments)) {
        RotatingExtraction failure;
        failure.error = extractor.output.error;
        return failure;
    }
    return std::move(extractor.output);
}
} // namespace mlir::pto::frontiersynch
