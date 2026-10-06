// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PhaseNormalization.h"
#include "SequenceAnalysisInternal.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingRegional.h"
#include "PTO/Transforms/FrontierSynch/RepeatedPhases.h"
namespace mlir::pto::frontiersynch {
namespace {
bool affineResidueCompatible(AffineExpr expression, uint64_t period)
{
    bool valid = true;
    if (!expression) { return true; }
    expression.walk([&](AffineExpr term) {
        auto binary = dyn_cast<AffineBinaryOpExpr>(term);
        if (!binary) { return; }
        if (binary.getKind() == AffineExprKind::Add || binary.getKind() == AffineExprKind::Mul) { return; }
        auto divisor = dyn_cast<AffineConstantExpr>(binary.getRHS());
        valid &= binary.getKind() == AffineExprKind::Mod && divisor && divisor.getValue() > 0 &&
                 period % static_cast<uint64_t>(divisor.getValue()) == 0;
    });
    return valid;
}
void specialize(GuardedRotatingAnalysis& analysis, RegionExpressions::Substitution& bindings)
{
    auto rewrite = [&](Expr& expression) { expression = analysis.expressions->substitute(expression, bindings); };
    auto payloads = [&](auto& values) { for (auto& value : values) { rewrite(value.presence); } };
    auto records = [&](auto& values) {
        for (auto& value : values) { rewrite(value.active); rewrite(value.displacement); }
    };
    payloads(analysis.payloads);
    payloads(analysis.periodic.payloads);
    records(analysis.generators);
    records(analysis.periodic.records);
    records(analysis.periodic.nativePrerequisites);
    for (auto& value : analysis.periodic.retained) { rewrite(value); }
    for (auto& threshold : analysis.periodic.thresholds) {
        rewrite(threshold.reachable); rewrite(threshold.distance);
    }
    for (auto& fragment : analysis.fragments) {
        rewrite(fragment.offset); rewrite(fragment.read); rewrite(fragment.write);
    }
}
} // namespace
bool SequenceAnalysisState::phasedChild(const StructureNode& node, Expr trips)
{
    auto outer = dyn_cast<scf::ForOp>(node.anchor);
    if (!outer || outer.getNumRegionIterArgs() || node.children.size() != 1) { return false; }
    auto unavailable = [&](const std::string& reason) { repeatedAttempt = "q-phase repeat: " + reason; return false; };
    const auto& sequence = program->nodes[node.children.front()];
    const StructureNode* innerNode = nullptr;
    for (auto id : sequence.children) {
        const auto& child = program->nodes[id];
        if (child.kind == StructureKind::Loop && !innerNode) { innerNode = &child; }
        else if (child.payloadCount) {
            return unavailable("child phase-specialization interface unavailable for this sequence");
        }
    }
    if (!innerNode) { return false; }
    auto inner = dyn_cast<scf::ForOp>(innerNode->anchor);
    PhaseNormalization normalizer(outer, index, expressions);
    if (!inner || !normalizer.independent(inner.getLowerBound()) ||
        !normalizer.independent(inner.getUpperBound()) || !normalizer.independent(inner.getStep())) {
        return unavailable("period-invariant child bounds have not been established");
    }
    auto recognition = recognizeGuardedRotating(inner, index, *input, input->accesses());
    if (recognition.result.state != RecognitionState::Applicable) {
        return unavailable("exact guarded-rotating child specialization is unavailable");
    }
    for (const auto& guard : recognition.guards) {
        if (!normalizer.independent(guard.condition)) {
            return unavailable("outer-dependent child control needs a phase-control adapter");
        }
    }
    uint64_t period = 1;
    SmallVector<Value> parameters;
    DenseSet<Value> seen;
    for (const auto& access : recognition.result.accesses) {
        bool varies = false;
        for (Value parameter : access.parameters) {
            if (normalizer.independent(parameter)) { continue; }
            varies = true;
            if (seen.insert(parameter).second) { parameters.push_back(parameter); }
        }
        if (!varies) { continue; }
        auto factor = access.slots / std::gcd(period, access.slots);
        if (!access.slots || factor > maxRegionalSlotVisits / period) {
            return unavailable("checked bank-period LCM exceeds the supported explicit phase count");
        }
        period *= factor;
    }
    if (period <= 1) { return unavailable("no nontrivial certified outer bank period"); }
    for (const auto& access : recognition.result.accesses) {
        if (llvm::any_of(access.parameters, [&](Value parameter) { return !normalizer.independent(parameter); }) &&
            !affineResidueCompatible(access.parameterOffset, period)) {
            return unavailable("bank map needs an arithmetic phase-normalization proof");
        }
    }
    // Discharge certificates belong to the unchanged invocation. A varying
    // discharged map needs the symbolic-family adapter, not phase relabeling.
    for (auto id : recognition.result.dischargedEffects) {
        const auto& effect = input->accesses().effects()[id];
        if (effect.selection && !normalizer.independent(effect.selection->selector)) {
            return unavailable("varying discharged storage requires a symbolic-family interface");
        }
        for (const auto& region : effect.regions) {
            for (Value symbol : region.symbols) {
                if (!normalizer.independent(symbol)) {
                    return unavailable("varying discharged byte map lacks a phase interface");
                }
            }
        }
    }
    auto original = analyzeGuardedRotating(inner, *input, recognition, arena);
    if (!original.error.empty()) { return unavailable(original.error); }
    std::vector<RegionalAnalysis> phases;
    for (uint64_t phase = 0; phase < period; ++phase) {
        SmallVector<std::pair<Expr, Expr>> replacements;
        for (Value parameter : parameters) {
            auto value = normalizer.residue(parameter, phase, period);
            if (!value) { return unavailable("fixed-width bank residue normalization is unproved"); }
            replacements.emplace_back(expressions.input(parameter), *value);
        }
        auto specialized = original;
        RegionExpressions::Substitution substitution(replacements);
        specialize(specialized, substitution);
        if (!expressions.constructionError().empty()) { return unavailable(expressions.constructionError()); }
        std::string diagnostic;
        auto view = guardedRotatingRegionalResult(function, *input, specialized, diagnostic);
        if (failed(view)) { return unavailable("child phase export: " + diagnostic); }
        DenseSet<std::size_t> exported;
        for (const auto& access : view->accessBoundary) { exported.insert(access.effect); }
        for (const auto& payload : view->anchors) {
            for (auto effect : input->accesses().effectsFor(payload.phase)) {
                if (input->accesses().effects()[effect].mode == SyncAccessMode::Write && !exported.count(effect)) {
                    return unavailable("discharged writer lacks an outer re-entry storage interface");
                }
            }
        }
        phases.push_back(std::move(*view));
    }
    auto repeated = repeatPhasedRegions(function, outer, std::move(phases), trips, node.loops);
    if (!repeated.error.empty()) { return unavailable(repeated.error); }
    Child child;
    child.regional = std::move(repeated.regional);
    child.anchors = child.regional.anchors;
    children.push_back(std::move(child));
    repeatedAttempt.clear();
    return true;
}
} // namespace mlir::pto::frontiersynch
