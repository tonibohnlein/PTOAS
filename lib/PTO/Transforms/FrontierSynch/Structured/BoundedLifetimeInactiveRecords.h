// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Sufficient uniform guard proofs; unknown guards retain their original obligations.
#ifndef PTO_FRONTIERSYNCH_BOUNDEDLIFETIMEINACTIVERECORDS_H
#define PTO_FRONTIERSYNCH_BOUNDEDLIFETIMEINACTIVERECORDS_H
#include "PTO/Transforms/FrontierSynch/Structured/BoundedLifetime.h"
#include <map>
#include <optional>
#include <vector>
namespace mlir::pto::frontiersynch::detail {
// The caller validates the window schema, presence IDs, and unconditional extent.
class BoundedInactiveRecordProof {
public:
    BoundedInactiveRecordProof(RegionExpressions& expressions, const LifetimeWindowInput& window,
        llvm::ArrayRef<uint8_t> unconditional) : e(expressions)
    {
        std::vector<RegionExpressions::Id> presence;
        for (uint32_t site = 0; site < window.sites; ++site) {
            if (!unconditional[site]) { continue; }
            for (uint32_t d = 0; d <= window.span; ++d) {
                presence.push_back(window.payloads[d * window.sites + site].present);
            }
            break;
        }
        for (std::size_t i = 1; i < presence.size(); ++i) {
            if (!e.implies(presence[i], presence[i - 1])) { return; }
        }
        if (!presence.empty()) { makePatterns(presence); }
    }
    bool proves(RegionExpressions::Id guard)
    {
        if (auto found = proofs.find(guard); found != proofs.end()) { return found->second; }
        bool inactive = e.constantValue(guard) == 0 || e.constantUnder(e.boolean(true), guard) == 0;
        if (!inactive && !patterns.empty()) {
            inactive = true;
            for (auto pattern : patterns) {
                if (e.constantUnder(pattern, guard) != 0) { inactive = false; break; }
            }
        }
        proofs.emplace(guard, inactive);
        return inactive;
    }
private:
    void makePatterns(llvm::ArrayRef<RegionExpressions::Id> presence)
    {
        // Certified monotonicity makes these prefix vectors exhaustive, including
        // none/all present. Opaque branch predicates are never enumerated or fixed.
        std::vector<RegionExpressions::Id> prefix{e.boolean(true)};
        for (auto present : presence) { prefix.push_back(e.land(prefix.back(), present)); }
        std::vector<RegionExpressions::Id> suffix(presence.size() + 1, e.boolean(true));
        for (std::size_t i = presence.size(); i > 0; --i) {
            suffix[i - 1] = e.land(e.lnot(presence[i - 1]), suffix[i]);
        }
        for (std::size_t count = 0; count <= presence.size(); ++count) {
            patterns.push_back(e.land(prefix[count], suffix[count]));
        }
    }
    RegionExpressions& e;
    std::vector<RegionExpressions::Id> patterns;
    std::map<RegionExpressions::Id, bool> proofs;
};
// Emission proofs are uniform over the original window. They do not borrow
// domain-conditioned allocation flags or retain temporary arena expressions.
inline std::optional<std::vector<uint8_t>> boundedInactiveEmissionRecipes(
    RegionExpressions& e, const LifetimeWindowInput& window,
    llvm::ArrayRef<uint8_t> unconditional, llvm::ArrayRef<GuardedRankEdge> records)
{
    const bool dimensions = window.sites && window.span < UINT32_MAX &&
        window.sites <= UINT32_MAX / (window.span + 1) &&
        window.payloads.size() == window.sites * (window.span + 1) &&
        unconditional.size() == window.sites;
    if (!dimensions) { return std::nullopt; }
    for (auto value : unconditional) { if (value > 1) { return std::nullopt; } }
    for (auto payload : window.payloads) {
        const bool valid = payload.present < e.size() && e.isBoolean(payload.present);
        if (!valid) { return std::nullopt; }
    }
    for (auto edge : records) {
        const bool valid = edge.source < window.sites && edge.source < edge.target &&
            edge.target < window.payloads.size() && edge.guard < e.size() && e.isBoolean(edge.guard);
        if (!valid) { return std::nullopt; }
    }
    std::vector<uint8_t> inactive;
    {
        RegionExpressions::Transaction transaction(e);
        BoundedInactiveRecordProof proof(e, window, unconditional);
        for (auto edge : records) { inactive.push_back(proof.proves(edge.guard)); }
    }
    return inactive;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
