// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Coordinate projection for storage summaries: invert certified bounded affine
// inner runs symbolically, otherwise use the finite projection adapter. Payloads
// and their reachability stay in the child.
#ifndef PTO_FRONTIERSYNCH_REPEATEDSTORAGEPROJECTION_H
#define PTO_FRONTIERSYNCH_REPEATEDSTORAGEPROJECTION_H
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
#include "../InsertSync/SyncScalarEvolution.h"
#include "../InsertSync/SyncEffectRanges.h"
#include "llvm/Support/MathExtras.h"
namespace mlir::pto::frontiersynch::detail {
struct ProjectedAccess {
    SyncAccessRegion region;
    RegionalAccessBoundary boundary;
    std::optional<RepeatedStorageInnerRun> inner;
};
inline std::optional<SyncAccessRegion> projectMap(const SyncAccessRegion& source, scf::ForOp loop,
                                                 const DenseMap<Value, int64_t>& bindings)
{
    SyncAccessRegion out = source;
    out.symbols.clear();
    mlir::pto::detail::ScalarEvolution scalar(loop.getContext(), loop);
    SmallVector<AffineExpr> replacements, dims;
    auto symbol = [&](Value value) -> AffineExpr {
        if (auto found = bindings.find(value); found != bindings.end()) {
            return getAffineConstantExpr(found->second, loop.getContext());
        }
        auto found = llvm::find(out.symbols, value);
        if (found == out.symbols.end()) { out.symbols.push_back(value); found = std::prev(out.symbols.end()); }
        return getAffineSymbolExpr(found - out.symbols.begin(), loop.getContext());
    };
    for (auto value : source.symbols) { replacements.push_back(scalar.value(value, symbol)); }
    for (unsigned i = 0; i < source.extents.size(); ++i) { dims.push_back(getAffineDimExpr(i, loop.getContext())); }
    out.byteOffset = mlir::pto::detail::substitute(source.byteOffset, dims, replacements);
    if (!out.byteOffset) { return std::nullopt; }
    for (auto& extent : out.extents) {
        extent = mlir::pto::detail::substitute(extent, {}, replacements);
        if (!extent) { return std::nullopt; }
    }
    mlir::pto::detail::compactRegionSymbols(out);
    return out;
}
// Recover an inner affine translation without unfolding its iteration domain.
// The current adapter binds one coordinate; additional nonconstant coordinates
// remain obligations, rather than being erased from the physical map.
inline std::optional<SmallVector<ProjectedAccess, 0>> projectInnerRun(
    const RegionalAnalysis& body, scf::ForOp outer, const SyncStorageEffect& effect,
    const RegionalAccessBoundary& access, const DenseMap<Value, int64_t>& fixed,
    ArrayRef<scf::ForOp> frames, bool leaf)
{
    if (frames.size() != 1 || !leaf) { return std::nullopt; }
    auto loop = frames.front();
    APInt lower, step;
    if (!loop || !outer->isProperAncestor(loop) ||
        !matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) || !lower.isSignedIntN(64) ||
        lower.isNegative() || !matchPattern(loop.getStep(), m_ConstantInt(&step)) ||
        !step.isSignedIntN(64) || !step.isStrictlyPositive()) { return std::nullopt; }
    auto upper = loop.getUpperBound();
    auto* owner = upper.getDefiningOp();
    if (auto arg = dyn_cast<BlockArgument>(upper)) { owner = arg.getOwner()->getParentOp(); }
    // The storage reservation and invocation selectors must repeat unchanged.
    if (!owner || owner == outer || outer->isProperAncestor(owner)) { return std::nullopt; }
    mlir::pto::detail::ScalarEvolution scalar(loop.getContext(), loop);
    auto range = scalar.signedRange(upper);
    if (!range || range->first < 0) { return std::nullopt; }
    auto a = lower.sextOrTrunc(128), b = APInt(128, range->second, true), stride = step.sextOrTrunc(128);
    const auto count = b.sgt(a) ? (b - a - 1).udiv(stride) + 1 : APInt(128, 0);
    if (count.getActiveBits() > 63 || count.isZero()) { return std::nullopt; }
    auto& e = *body.expressions;
    auto low = e.constant(lower.getZExtValue()), high = e.input(upper);
    // Divide (distance-1) only on nonempty domains; the circuits are total and
    // the select masks empty domains. No upper+step expression can overflow.
    auto trips = e.select(e.lt(low, high),
        e.add(e.div(e.sub(e.sub(high, low), e.constant(1)), e.constant(step.getZExtValue())), e.constant(1)),
        e.constant(0));
    SmallVector<ProjectedAccess, 0> result;
    for (const auto& region : effect.regions) {
        auto map = projectMap(region, outer, fixed);
        if (!map) { return std::nullopt; }
        auto variable = llvm::find(map->symbols, loop.getInductionVar());
        if (variable == map->symbols.end()) { return std::nullopt; }
        auto split = mlir::pto::detail::splitTranslation(map->byteOffset, map->extents.size(),
                                                       map->symbols.size(), variable - map->symbols.begin());
        int64_t byteStride = 0;
        if (!split || split->second <= 0 || llvm::MulOverflow(split->second, step.getSExtValue(), byteStride) ||
            byteStride <= 0) { return std::nullopt; }
        // Extent dependence cannot be replaced by a reservation hull.
        for (auto extent : map->extents) {
            if (extent.isFunctionOfSymbol(variable - map->symbols.begin())) { return std::nullopt; }
        }
        auto bindings = fixed;
        bindings[loop.getInductionVar()] = lower.getSExtValue();
        auto projected = projectMap(region, outer, bindings);
        if (!projected) { return std::nullopt; }
        result.push_back({std::move(*projected), access,
                          RepeatedStorageInnerRun{uint64_t(byteStride), count.getZExtValue(), trips, 0}});
    }
    return result;
}
inline std::optional<SmallVector<ProjectedAccess, 0>> projectAccesses(
    const RegionalAnalysis& body, scf::ForOp loop, std::size_t effectId)
{
    if (!loop || !body.expressions || !body.accessModel || effectId >= body.accessModel->effects().size()) {
        return std::nullopt;
    }
    const auto& effect = body.accessModel->effects()[effectId];
    SmallVector<ProjectedAccess, 0> out;
    auto& e = *body.expressions;
    for (const auto& access : body.accessBoundary) {
        if (access.effect != effectId) { continue; }
        if (!validRegionalEvent(body, access.first.event) || !validRegionalEvent(body, access.last.event)) {
            return std::nullopt;
        }
        const auto type = access.first.event.type;
        if (type >= body.anchors.size() || access.last.event.type != type) { return std::nullopt; }
        auto nested = [&](scf::ForOp frame) { return frame && loop->isProperAncestor(frame); };
        if ((body.occurrenceLoops[type] && !nested(body.occurrenceLoops[type])) ||
            (!body.outerLoops.empty() && !llvm::all_of(body.outerLoops[type], nested))) { return std::nullopt; }
        DenseMap<Value, int64_t> fixed;
        for (auto coordinate : body.anchors[type].coordinates) {
            if (!coordinate.loop || !loop->isProperAncestor(coordinate.loop)) { return std::nullopt; }
            fixed[coordinate.loop.getInductionVar()] = coordinate.induction;
        }
        auto append = [&](const DenseMap<Value, int64_t>& bindings, const RegionalAccessBoundary& boundary) {
            for (const auto& region : effect.regions) {
                auto mapped = projectMap(region, loop, bindings);
                if (!mapped || out.size() >= maxRegionalSlotVisits) { return false; }
                out.push_back({std::move(*mapped), boundary});
            }
            return true;
        };
        bool varies = false;
        for (const auto& region : effect.regions) {
            auto mapped = projectMap(region, loop, fixed);
            if (!mapped) { return std::nullopt; }
            for (auto symbol : mapped->symbols) {
                auto* owner = symbol.getDefiningOp();
                if (auto arg = dyn_cast<BlockArgument>(symbol)) { owner = arg.getOwner()->getParentOp(); }
                varies |= symbol != loop.getInductionVar() && owner && loop->isProperAncestor(owner);
            }
        }
        if (!varies) {
            if (!append(fixed, access)) { return std::nullopt; }
            continue;
        }
        SmallVector<scf::ForOp> frames;
        if (!body.outerLoops.empty()) { frames.append(body.outerLoops[type].begin(), body.outerLoops[type].end()); }
        if (!body.outerDivisors.empty() && llvm::any_of(body.outerDivisors[type],
            [](uint64_t divisor) { return divisor != 1; })) { return std::nullopt; }
        const bool leaf = bool(body.occurrenceLoops[type]);
        if (leaf) { frames.push_back(body.occurrenceLoops[type]); }
        if (frames.empty()) { return std::nullopt; }
        if (auto symbolic = projectInnerRun(body, loop, effect, access, fixed, frames, leaf)) {
            llvm::append_range(out, std::move(*symbolic));
            continue;
        }
        if (frames.size() > 64) { return std::nullopt; }
        struct Domain { APInt lower, step; uint64_t trips; };
        SmallVector<Domain, 4> domains;
        bool empty = false;
        for (auto frame : frames) {
            APInt lower, upper, step;
            if (!frame || !loop->isProperAncestor(frame) ||
                !matchPattern(frame.getLowerBound(), m_ConstantInt(&lower)) ||
                !matchPattern(frame.getUpperBound(), m_ConstantInt(&upper)) ||
                !matchPattern(frame.getStep(), m_ConstantInt(&step)) || !lower.isSignedIntN(64) ||
                !upper.isSignedIntN(64) || !step.isSignedIntN(64) || !step.isStrictlyPositive()) {
                return std::nullopt;
            }
            auto a = lower.sextOrTrunc(128), b = upper.sextOrTrunc(128), stride = step.sextOrTrunc(128);
            const auto count = b.sgt(a) ? (b - a - 1).udiv(stride) + 1 : APInt(128, 0);
            if (count.ugt(maxRegionalSlotVisits)) { return std::nullopt; }
            domains.push_back({a, stride, count.getZExtValue()});
            empty |= count.isZero();
        }
        if (empty) { continue; }
        uint64_t product = 1;
        for (const auto& domain : domains) {
            if (domain.trips > maxRegionalSlotVisits / product) { return std::nullopt; }
            product *= domain.trips;
        }
        RegionalEvent event{type, e.constant(0), PeriodicEventKind::Start};
        event.visits.resize(frames.size() - unsigned(leaf), e.constant(0));
        std::function<bool(unsigned)> enumerate = [&](unsigned depth) {
            if (depth == frames.size()) {
                auto present = regionalPresence(body, event);
                if (!present) { return false; }
                if (e.constantValue(*present) == uint64_t(0)) { return true; }
                RegionalSelector selected{event, *present};
                return append(fixed, {effectId, selected, selected, false});
            }
            const auto& domain = domains[depth];
            for (uint64_t ordinal = 0; ordinal < domain.trips; ++ordinal) {
                fixed[frames[depth].getInductionVar()] =
                    (domain.lower + APInt(128, ordinal) * domain.step).getSExtValue();
                if (depth < event.visits.size()) { event.visits[depth] = e.constant(ordinal); }
                else { event.ordinal = e.constant(ordinal); }
                if (!enumerate(depth + 1)) { return false; }
            }
            fixed.erase(frames[depth].getInductionVar());
            return true;
        };
        if (!enumerate(0)) { return std::nullopt; }
    }
    return out;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
