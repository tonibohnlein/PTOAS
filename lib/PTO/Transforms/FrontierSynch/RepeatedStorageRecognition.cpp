// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Generic constant-stride proposals from shared, non-wrapping mathematical maps.
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
#include "RepeatedStorageProjection.h"
#include "../InsertSync/SyncRegionArithmetic.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"
namespace mlir::pto::frontiersynch {
namespace {
namespace geometry = mlir::pto::detail;
std::optional<int64_t> integer(Value value)
{
    APInt number;
    if (!matchPattern(value, m_ConstantInt(&number)) || !number.isSignedIntN(64)) { return std::nullopt; }
    return number.getSExtValue();
}
bool pruneSymbols(SyncAccessRegion& origin)
{
    SmallVector<Value> symbols;
    SmallVector<AffineExpr> replacements;
    auto* context = origin.byteOffset.getContext();
    for (unsigned i = 0; i < origin.symbols.size(); ++i) {
        if (origin.byteOffset.isFunctionOfSymbol(i)) {
            replacements.push_back(getAffineSymbolExpr(symbols.size(), context));
            symbols.push_back(origin.symbols[i]);
        } else { replacements.push_back(getAffineConstantExpr(0, context)); }
    }
    origin.byteOffset = geometry::substitute(origin.byteOffset, {}, replacements);
    origin.symbols = std::move(symbols);
    return bool(origin.byteOffset);
}
std::optional<RepeatedStorageFamily> propose(const SyncStorageEffect& effect, const SyncAccessRegion& physical,
                                              scf::ForOp loop, std::size_t id)
{
    if (!effect.memory || effect.regions.empty()) { return std::nullopt; }
    auto lower = integer(loop.getLowerBound()), step = integer(loop.getStep());
    if (!physical.byteOffset || !lower || *lower < 0 || !step || *step <= 0) { return std::nullopt; }
    auto* context = loop.getContext();
    auto local = physical.byteOffset;
    int64_t coefficient = 0, stride = 0;
    auto varying = llvm::find(physical.symbols, loop.getInductionVar());
    if (varying != physical.symbols.end()) {
        auto split = geometry::splitTranslation(local, physical.extents.size(), physical.symbols.size(),
                                               varying - physical.symbols.begin());
        if (!split) { return std::nullopt; }
        local = split->first; coefficient = split->second;
    }
    if (coefficient < 0 || llvm::MulOverflow(coefficient, *step, stride) ||
        (!stride && effect.mode == SyncAccessMode::Write)) { return std::nullopt; }
    auto shift = geometry::checkedMul(getAffineConstantExpr(coefficient, context),
                                      getAffineConstantExpr(*lower, context));
    local = geometry::checkedAdd(local, shift);
    SmallVector<AffineExpr> zeroDims(physical.extents.size(), getAffineConstantExpr(0, context));
    auto origin = local ? simplifyAffineExpr(local.replaceDims(zeroDims), 0, physical.symbols.size()) : AffineExpr();
    if (!origin) { return std::nullopt; }
    SmallVector<AffineExpr> zeroSymbols(physical.symbols.size(), getAffineConstantExpr(0, context));
    auto constant = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(origin.replaceSymbols(zeroSymbols), 0, 0));
    if (!constant || constant.getValue() < 0) { return std::nullopt; }
    // Align the constant part of an owned origin to its reservation. This merges
    // overlapping views of the same reservation regardless of effect order.
    auto remove = stride ? constant.getValue() % stride : constant.getValue();
    origin = geometry::checkedAdd(origin, getAffineConstantExpr(-remove, context));
    if (!origin) { return std::nullopt; }
    RepeatedStorageFamily family;
    family.kind = stride ? RepeatedStorageKind::VisitOwned : RepeatedStorageKind::SharedReadOnly;
    family.space = effect.memory->scope; family.stride = stride; family.effects.push_back(id);
    family.origin.base = physical.base; family.origin.symbols = physical.symbols;
    family.origin.byteOffset = simplifyAffineExpr(origin, 0, physical.symbols.size());
    return pruneSymbols(family.origin) ? std::optional<RepeatedStorageFamily>(std::move(family)) : std::nullopt;
}
bool sameFamily(const RepeatedStorageFamily& a, const RepeatedStorageFamily& b)
{
    return a.kind == b.kind && a.space == b.space && a.stride == b.stride &&
        a.origin.base == b.origin.base && a.origin.byteOffset == b.origin.byteOffset &&
        a.origin.symbols == b.origin.symbols;
}
} // namespace
RepeatedStorageResult recognizeRepeatedStorage(const RegionalAnalysis& body, scf::ForOp loop,
                                               RegionExpressions::Id trips, const PhaseIndex* phaseIndex,
                                               uint64_t phasePeriod)
{
    if (!loop || !body.accessModel) { return {"evolving storage has no shared effect model", {}}; }
    llvm::DenseSet<std::size_t> visited;
    std::vector<RepeatedStorageFamily> families;
    for (const auto& anchor : body.anchors) {
        for (auto id : body.accessModel->effectsFor(anchor.phase)) {
            if (!visited.insert(id).second) { continue; }
            const auto& effect = body.accessModel->effects()[id];
            const bool represented = llvm::any_of(body.accessBoundary, [id](const auto& access) {
                return access.effect == id && access.representedByCells;
            });
            // Regional cells may already resolve a rotating selector even when
            // the original access record has no materialized static ranges.
            // completeEffects still checks invariance of every unclassified map.
            if (represented) { continue; }
            auto projected = detail::projectAccesses(body, loop, id);
            if (!projected || projected->empty()) { continue; }
            auto candidate = propose(effect, projected->front().region, loop, id);
            if (!candidate) { continue; } // Remaining effects must still satisfy completeEffects.
            auto existing = llvm::find_if(families, [&](const auto& family) { return sameFamily(family, *candidate); });
            if (existing == families.end()) { families.push_back(std::move(*candidate)); }
            else { existing->effects.push_back(id); }
        }
    }
    if (families.empty()) { return {"no constant-stride evolving storage family was constructed", {}}; }
    return buildRepeatedStorage(body, loop, trips, families, phaseIndex, phasePeriod);
}
} // namespace mlir::pto::frontiersynch
