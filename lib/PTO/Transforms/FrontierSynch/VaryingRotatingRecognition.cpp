// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/VaryingRotatingRecognition.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "PhaseNormalization.h"
#include "RecognitionInternal.h"
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
#include "ArithmeticRows.h"
#include "../InsertSync/SyncScalarEvolution.h"
#include "llvm/Support/MathExtras.h"
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
std::optional<int64_t> integer(Value value)
{
    APInt number;
    if (!matchPattern(value, m_ConstantInt(&number)) || !number.isSignedIntN(64)) {
        return std::nullopt;
    }
    return number.getSExtValue();
}
} // namespace
VaryingRotatingRecognition recognizeVaryingRotating(scf::ForOp outer, const PhaseIndex& index, const SyncInput& input)
{
    VaryingRotatingRecognition out;
    out.outer = outer;
    auto reject = [&](RecognitionIssue issue, Operation* anchor, bool mismatch = false) {
        out.result.note(issue, anchor, mismatch);
        return out;
    };
    if (!outer || index.hasRelevantCarriedState(outer)) {
        return reject(RecognitionIssue::LoopCarriedState, outer);
    }
    for (Operation& operation : *outer.getBody()) {
        if (auto inner = dyn_cast<scf::ForOp>(operation)) {
            if (out.inner) {
                return reject(RecognitionIssue::StructuredBody, &operation, true);
            }
            out.inner = inner;
        } else if (operation.getNumRegions() || !index.phasesFor(&operation).empty()) {
            return reject(RecognitionIssue::StructuredBody, &operation, true);
        } else {
            detail::inspectLeaf(operation, index, out.result);
        }
    }
    if (out.result.state != RecognitionState::Applicable) {
        return out;
    }
    if (!out.inner) {
        return reject(RecognitionIssue::StructuredBody, outer, true);
    }
    auto lower = integer(outer.getLowerBound()), step = integer(outer.getStep());
    auto innerLower = integer(out.inner.getLowerBound()), innerStep = integer(out.inner.getStep());
    if (!lower || *lower < 0 || !step || *step <= 0 || !innerLower || *innerLower < 0 || !innerStep ||
        *innerStep <= 0) {
        return reject(RecognitionIssue::LoopDomain, out.inner);
    }
    mlir::pto::detail::ScalarEvolution evolution(outer.getContext(), outer);
    SmallVector<Value> unknown;
    auto expression = evolution.value(out.inner.getUpperBound(), [&](Value value) {
        if (value == outer.getInductionVar()) {
            return getAffineDimExpr(0, outer.getContext());
        }
        auto found = llvm::find(unknown, value);
        if (found == unknown.end()) {
            unknown.push_back(value);
            found = std::prev(unknown.end());
        }
        return getAffineSymbolExpr(found - unknown.begin(), outer.getContext());
    });
    LinearRow row;
    if (!expression || !unknown.empty()) {
        return reject(RecognitionIssue::IndexArithmetic, out.inner);
    }
    if (detail::collectRow(expression, 1, 0, row) || row.coefficients.size() != 1 || row.coefficients[0] <= 0) {
        return reject(RecognitionIssue::IndexArithmetic, out.inner, true);
    }
    int64_t slope, intercept;
    if (llvm::MulOverflow(row.coefficients[0], *step, slope) ||
        llvm::MulOverflow(row.coefficients[0], *lower, intercept) ||
        llvm::AddOverflow(intercept, row.constant, intercept) || llvm::SubOverflow(intercept, *innerLower, intercept) ||
        intercept < 0 || slope % *innerStep || intercept % *innerStep) {
        return reject(RecognitionIssue::LoopDomain, out.inner);
    }
    out.slope = slope / *innerStep;
    out.intercept = intercept / *innerStep;
    out.child = recognizeRotating(out.inner, index, input, input.accesses());
    if (out.child.state != RecognitionState::Applicable) {
        out.result = out.child;
        return out;
    }
    RegionExpressions arena;
    PhaseNormalization invariance(outer, index, arena);
    for (const auto& access : out.child.accesses) {
        const auto& effect = input.accesses().effects()[access.effect];
        if (effect.selection && !invariance.independent(effect.selection->selector)) {
            return reject(RecognitionIssue::SlotExpression, effect.phase->elementOp);
        }
        for (const auto& region : effect.regions) {
            if (!invariance.periodic(region, 1)) {
                return reject(RecognitionIssue::SymbolicGeometry, effect.phase->elementOp);
            }
        }
    }
    if (!out.child.dischargedEffects.empty()) {
        // A within-visit discharge does not establish cross-visit independence.
        return reject(RecognitionIssue::GMDischarge, outer);
    }
    return out;
}
AffineRotatingVisits analyzeVaryingRotating(
    const VaryingRotatingRecognition& recognized, const PhaseIndex& index, const SyncInput& input)
{
    AffineRotatingVisits failure;
    if (recognized.result.state != RecognitionState::Applicable) {
        failure.error = "varying rotating contract not established";
        return failure;
    }
    auto child = analyzeRotating(recognized.inner, index, input, recognized.child, false);
    if (!child.error.empty()) {
        failure.error = child.error;
        return failure;
    }
    std::vector<std::pair<uint32_t, uint32_t>> uniformCrossings;
    if (input.accesses().hasUniformRelationships(child.phases)) {
        for (uint32_t source = 0; source < child.phases.size(); ++source) {
            for (uint32_t target = 0; target < child.phases.size(); ++target) {
                bool conflict = false;
                for (auto a : input.accesses().effectsFor(child.phases[source])) {
                    for (auto b : input.accesses().effectsFor(child.phases[target])) {
                        conflict |= input.accesses().uniformConflict(a, b);
                    }
                }
                if (conflict) {
                    uniformCrossings.emplace_back(source, target);
                }
            }
        }
    }
    // Inner invocation protection need not extend across re-entry. Project the
    // same shared facts into the outer scope for the crossing witnesses only.
    const auto protection = structuredProtection(input.accesses());
    for (std::size_t i = 0; i < child.fragments.size(); ++i) {
        auto& fragment = child.fragments[i];
        const auto& effect = input.accesses().effects()[recognized.child.accesses[i].effect];
        fragment.protectionGroup = effect.memory->scope == AddressSpace::ACC && fragment.stride == 0 ?
            protection.inLoop(effect.phase, recognized.outer) : 0;
    }
    std::vector<RotatingBoundaryCell> cells;
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (const auto& fragment : child.fragments) {
        if (!seen.emplace(fragment.family, fragment.atom).second) {
            continue;
        }
        if (fragment.slots > maxRegionalSlotVisits || cells.size() > maxRegionalSlotVisits - fragment.slots) {
            failure.error = "varying boundary cell expansion exceeds representation";
            return failure;
        }
        for (uint64_t slot = 0; slot < fragment.slots; ++slot) {
            cells.push_back({fragment.family, fragment.atom, slot});
        }
    }
    auto certificate = buildRotatingBoundaryCertificate(
        std::move(child.periodic), child.fragments, cells, uniformCrossings, maxRegionalSlotVisits,
        ptoStorageProtection());
    return analyzeAffineRotatingVisits(
        std::move(certificate), recognized.slope, recognized.intercept, maxRegionalSlotVisits);
}
} // namespace mlir::pto::frontiersynch
