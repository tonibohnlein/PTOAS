// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// The circular access order is computed arithmetically for each exact atom and
// offset residue. Only static fragments are visited, regardless of slot counts.
#include "PTO/Transforms/FrontierSynch/RotatingFootprintAnalysis.h"
#include <algorithm>
#include <tuple>

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::frontiersynch;
using llvm::DynamicAPInt;

namespace {
bool reads(RotatingAccessMode mode) { return mode != RotatingAccessMode::Write; }
bool writes(RotatingAccessMode mode) { return mode != RotatingAccessMode::Read; }

LogicalResult validate(
    ArrayRef<const CompoundInstanceElement*> sites, ArrayRef<RotatingFamily> families,
    ArrayRef<RotatingFragment> fragments)
{
    for (const auto* site : sites) {
        if (!site) {
            return failure();
        }
    }
    for (const auto& family : families) {
        if (family.slots <= 0 || family.atoms == 0) {
            return failure();
        }
    }
    for (const auto& fragment : fragments) {
        if (fragment.site >= sites.size() || fragment.family >= families.size()) {
            return failure();
        }
        if (fragment.atom >= families[fragment.family].atoms ||
            (fragment.mode != RotatingAccessMode::Read && fragment.mode != RotatingAccessMode::Write &&
             fragment.mode != RotatingAccessMode::ReadWrite)) {
            return failure();
        }
    }
    return success();
}

auto fragmentKey(const RotatingFragment& fragment)
{
    return std::tie(fragment.family, fragment.atom, fragment.offset, fragment.site);
}

SmallVector<RotatingFragment> normalize(ArrayRef<RotatingFragment> fragments, ArrayRef<RotatingFamily> families)
{
    SmallVector<RotatingFragment> sorted(fragments);
    for (auto& fragment : sorted) {
        fragment.offset = mod(fragment.offset, families[fragment.family].slots);
    }
    llvm::sort(sorted, [](const auto& left, const auto& right) { return fragmentKey(left) < fragmentKey(right); });
    SmallVector<RotatingFragment> result;
    for (const auto& fragment : sorted) {
        if (!result.empty() && fragmentKey(result.back()) == fragmentKey(fragment)) {
            if (result.back().mode != fragment.mode) {
                result.back().mode = RotatingAccessMode::ReadWrite;
            }
        } else {
            result.push_back(fragment);
        }
    }
    return result;
}

// Extended Euclid, used only with coprime 0 < stride < modulus and modulus > 1.
DynamicAPInt inverse(const DynamicAPInt& stride, const DynamicAPInt& modulus)
{
    DynamicAPInt remainder = modulus, nextRemainder = stride;
    DynamicAPInt coefficient(0), nextCoefficient(1);
    while (nextRemainder != 0) {
        DynamicAPInt quotient = remainder / nextRemainder;
        DynamicAPInt saved = remainder - quotient * nextRemainder;
        remainder = nextRemainder;
        nextRemainder = saved;
        saved = coefficient - quotient * nextCoefficient;
        coefficient = nextCoefficient;
        nextCoefficient = saved;
    }
    return mod(coefficient, modulus);
}

RotatingFamilyFacts preprocess(const RotatingFamily& family)
{
    DynamicAPInt stride = mod(family.stride, family.slots);
    DynamicAPInt divisor = gcd(stride, family.slots);
    DynamicAPInt refresh = family.slots / divisor;
    DynamicAPInt inverseStride = refresh == 1 ? DynamicAPInt(0) : inverse(stride / divisor, refresh);
    return {family, divisor, refresh, inverseStride};
}

struct AccessKey {
    std::size_t fragment = 0;
    std::size_t atom = 0;
    DynamicAPInt residue;
    DynamicAPInt phase;
    std::size_t site = 0;
};
SmallVector<AccessKey> accessKeys(
    ArrayRef<RotatingFragment> fragments, std::size_t begin, std::size_t end, const RotatingFamilyFacts& arithmetic)
{
    SmallVector<AccessKey> keys;
    for (std::size_t id = begin; id < end; ++id) {
        const auto& fragment = fragments[id];
        DynamicAPInt residue = mod(fragment.offset, arithmetic.divisor);
        DynamicAPInt quotient = (fragment.offset - residue) / arithmetic.divisor;
        DynamicAPInt phase = mod(-arithmetic.inverseStride * quotient, arithmetic.refresh);
        keys.push_back({id, fragment.atom, residue, phase, fragment.site});
    }
    llvm::sort(keys, [](const auto& left, const auto& right) {
        return std::tie(left.atom, left.residue, left.phase, left.site) <
               std::tie(right.atom, right.residue, right.phase, right.site);
    });
    return keys;
}

void emit(
    const AccessKey& source, const AccessKey& consumer, const DynamicAPInt& refresh,
    ArrayRef<RotatingFragment> fragments, SmallVectorImpl<PeriodicStorageDemand>& edges)
{
    DynamicAPInt distance = mod(consumer.phase - source.phase, refresh);
    if (distance == 0 && source.site >= consumer.site) {
        distance = refresh;
    }
    PeriodicStorageDemand edge{source.site, consumer.site, distance, {}};
    const auto sourceMode = fragments[source.fragment].mode;
    const auto consumerMode = fragments[consumer.fragment].mode;
    if (writes(sourceMode) && reads(consumerMode)) {
        edge.witnesses.push_back({Hazard::RAW, source.fragment, consumer.fragment});
    }
    if (reads(sourceMode) && writes(consumerMode)) {
        edge.witnesses.push_back({Hazard::WAR, source.fragment, consumer.fragment});
    }
    if (writes(sourceMode) && writes(consumerMode)) {
        edge.witnesses.push_back({Hazard::WAW, source.fragment, consumer.fragment});
    }
    edges.push_back(std::move(edge));
}

void scanGroup(
    ArrayRef<AccessKey> keys, const DynamicAPInt& refresh, ArrayRef<RotatingFragment> fragments,
    SmallVectorImpl<PeriodicStorageDemand>& edges)
{
    // Seed each circular scan with the extreme writer. Updating after emitting
    // excludes the current RMW occurrence even when it is the sole writer.
    const AccessKey* first = nullptr;
    const AccessKey* last = nullptr;
    for (const auto& key : keys) {
        if (writes(fragments[key.fragment].mode)) {
            if (!first) {
                first = &key;
            }
            last = &key;
        }
    }
    if (!first) {
        return;
    }
    for (const auto& key : keys) {
        emit(*last, key, refresh, fragments, edges);
        if (writes(fragments[key.fragment].mode)) {
            last = &key;
        }
    }
    for (const auto& key : llvm::reverse(keys)) {
        if (writes(fragments[key.fragment].mode)) {
            first = &key;
        } else {
            emit(key, *first, refresh, fragments, edges);
        }
    }
}

void scanFamily(
    ArrayRef<AccessKey> keys, const DynamicAPInt& refresh, ArrayRef<RotatingFragment> fragments,
    SmallVectorImpl<PeriodicStorageDemand>& edges)
{
    std::size_t begin = 0;
    while (begin < keys.size()) {
        std::size_t end = begin + 1;
        while (end < keys.size() && keys[end].atom == keys[begin].atom && keys[end].residue == keys[begin].residue) {
            ++end;
        }
        scanGroup(keys.slice(begin, end - begin), refresh, fragments, edges);
        begin = end;
    }
}

auto endpointKey(const PeriodicStorageDemand& edge) { return std::tie(edge.source, edge.consumer, edge.distance); }

SmallVector<PeriodicStorageDemand> deduplicate(SmallVector<PeriodicStorageDemand> edges)
{
    llvm::stable_sort(
        edges, [](const auto& left, const auto& right) { return endpointKey(left) < endpointKey(right); });
    SmallVector<PeriodicStorageDemand> result;
    for (auto& edge : edges) {
        if (!result.empty() && endpointKey(result.back()) == endpointKey(edge)) {
            result.back().witnesses.append(edge.witnesses.begin(), edge.witnesses.end());
        } else {
            result.push_back(std::move(edge));
        }
    }
    return result;
}
} // namespace

LogicalResult RotatingFootprintAnalysis::build(
    ArrayRef<const CompoundInstanceElement*> sequence, ArrayRef<RotatingFamily> families,
    ArrayRef<RotatingFragment> fragments)
{
    // Build separately so callers may also supply views of the previous result.
    RotatingFootprintAnalysis result;
    if (failed(validate(sequence, families, fragments))) {
        *this = std::move(result);
        return failure();
    }
    result.phaseSites.assign(sequence.begin(), sequence.end());
    result.normalized = normalize(fragments, families);
    for (const auto& family : families) {
        result.familyFacts.push_back(preprocess(family));
    }
    std::size_t begin = 0;
    while (begin < result.normalized.size()) {
        const auto family = result.normalized[begin].family;
        std::size_t end = begin + 1;
        while (end < result.normalized.size() && result.normalized[end].family == family) {
            ++end;
        }
        const auto& arithmetic = result.familyFacts[family];
        result.familyCertificates.push_back({family, arithmetic.divisor, arithmetic.refresh});
        result.bound = std::max(result.bound, arithmetic.refresh);
        const auto keys = accessKeys(result.normalized, begin, end, arithmetic);
        scanFamily(keys, arithmetic.refresh, result.normalized, result.edges);
        begin = end;
    }
    result.edges = deduplicate(std::move(result.edges));
    result.initialized = true;
    *this = std::move(result);
    return success();
}
