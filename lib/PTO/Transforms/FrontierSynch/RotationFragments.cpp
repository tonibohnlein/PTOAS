// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Make the exact common within-slot partition before generator extraction.
#include "RecognitionInternal.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <tuple>
#include <vector>

namespace mlir::pto::frontiersynch::detail {
void normalizeFragments(RecognitionResult& result, const SyncStorageEffects& effects)
{
    DenseMap<Value, SmallVector<uint64_t>> endpoints;
    for (const auto& access : result.accesses) {
        if (access.atom) {
            auto& points = endpoints[access.family];
            points.push_back(access.atom->first);
            points.push_back(access.atom->second);
        }
    }
    for (auto& entry : endpoints) {
        llvm::sort(entry.second);
        entry.second.erase(std::unique(entry.second.begin(), entry.second.end()), entry.second.end());
    }
    SmallVector<RotatingAccess, 0> normalized;
    DenseMap<Value, std::size_t> valueIds;
    DenseMap<AffineExpr, std::size_t> expressionIds;
    DenseMap<const CompoundInstanceElement*, std::size_t> phaseIds;
    auto valueId = [&](Value value) { return valueIds.try_emplace(value, valueIds.size()).first->second; };
    using Key = std::tuple<std::size_t, std::size_t, uint64_t, uint64_t,
                           uint64_t, uint64_t, std::size_t, std::vector<std::size_t>>;
    std::map<Key, std::size_t> positions;
    // Only merge fragments of the same payload and selector. Distinct payloads
    // remain separate even when their byte sets coincide.
    for (const auto& access : result.accesses) {
        if (!access.atom) {
            normalized.push_back(access);
            continue;
        }
        const auto& points = endpoints[access.family];
        auto first = llvm::lower_bound(points, access.atom->first);
        auto last = llvm::lower_bound(points, access.atom->second);
        for (auto it = first; it != last; ++it) {
            RotatingAccess fragment = access;
            fragment.atom = std::make_pair(*it, *std::next(it));
            std::vector<std::size_t> parameters;
            for (Value parameter : fragment.parameters) {
                parameters.push_back(valueId(parameter));
            }
            auto phase = effects.effects()[fragment.effect].phase;
            auto phaseId = phaseIds.try_emplace(phase, phaseIds.size()).first->second;
            auto expressionId = expressionIds.try_emplace(fragment.parameterOffset, expressionIds.size()).first->second;
            Key key{phaseId, valueId(fragment.family),
                    fragment.atom->first, fragment.atom->second, fragment.stride, fragment.offset,
                    expressionId,
                    std::move(parameters)};
            auto [position, inserted] = positions.emplace(std::move(key), normalized.size());
            if (inserted) {
                normalized.push_back(std::move(fragment));
            } else {
                auto found = normalized.begin() + position->second;
                found->reads |= fragment.reads;
                found->writes |= fragment.writes;
                for (auto id : fragment.effects) {
                    if (!llvm::is_contained(found->effects, id)) {
                        found->effects.push_back(id);
                    }
                }
            }
        }
    }
    result.accesses = std::move(normalized);
}
} // namespace mlir::pto::frontiersynch::detail
