// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Late, charged expansion into a periodic body; no demands or IR insertion.
#include "NumericTemplateInternal.h"
#include "RecognitionInternal.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
namespace {
std::optional<int64_t> constant(Value value)
{
    APInt number;
    if (!matchPattern(value, m_ConstantInt(&number)) || !number.isSignedIntN(64)) {
        return std::nullopt;
    }
    return number.getSExtValue();
}
bool scope(NumericTemplate& output, const PhaseIndex& index, const SyncInput& input, bool regional)
{
    auto outer = output.outer;
    auto function = dyn_cast<func::FuncOp>(outer->getParentOp());
    if (!function || !function.getBody().hasOneBlock()) {
        output.result.note(RecognitionIssue::TemplateContext, outer);
        return false;
    }
    for (const auto* phase : input.instructions()) {
        if (!regional && !outer->isProperAncestor(phase->elementOp)) {
            output.result.note(RecognitionIssue::TemplateContext, phase->elementOp);
            return false;
        }
    }
    function.walk([&](Operation* op) {
        if (op == function.getOperation() || (regional && op != outer && !outer->isProperAncestor(op))) {
            return;
        }
        if (index.needsValuePrerequisite(op)) {
            output.result.note(RecognitionIssue::AdditionalPrerequisite, op);
        }
        if (isa<scf::ForOp, scf::IfOp>(op)) {
            for (auto& region : op->getRegions()) {
                if (!region.empty() && !region.hasOneBlock()) {
                    output.result.note(RecognitionIssue::UnsupportedControl, op, true);
                }
            }
        } else {
            detail::inspectLeaf(*op, index, output.result);
        }
    });
    return output.result.state == RecognitionState::Applicable;
}
void clear(NumericTemplate& output)
{
    output.payloads.clear();
    output.atoms.clear();
    output.fragments = 0;
    output.period = 0;
    output.refresh = 0;
}
NumericTemplate recognizeTemplate(scf::ForOp outer, const PhaseIndex& index,
                                  const SyncInput& input, NumericTemplateLimits limits, bool regional)
{
    NumericTemplate output;
    output.outer = outer;
    output.limits = limits;
    auto lower = constant(outer.getLowerBound()), step = constant(outer.getStep());
    const bool valid = lower && step && *lower >= 0 && *step > 0 && limits.depth &&
        limits.visits && limits.payloads && limits.fragments &&
        limits.visits <= NumericTemplateLimits{}.visits && limits.payloads <= NumericTemplateLimits{}.payloads &&
        limits.fragments <= NumericTemplateLimits{}.fragments && limits.depth <= NumericTemplateLimits{}.depth;
    if (!valid) {
        output.result.note(RecognitionIssue::LoopDomain, outer, true);
        return output;
    }
    output.lower = *lower;
    output.step = *step;
    if (!scope(output, index, input, regional)) {
        return output;
    }
    auto upper = constant(outer.getUpperBound());
    if (upper && *upper <= output.lower) {
        // Scope and the positive step are established. No body occurrence can
        // execute, so no effect substitution or inner-visit expansion is needed.
        // This certificate is tied to these bounds, not a periodic body schema.
        output.emptyInvocation = true;
        output.period = 1;
        output.refresh = 1;
        return output;
    }
    detail::TemplateBuilder builder{output, index, input, DenseMap<Value, int64_t>(), {}};
    for (auto state : outer.getRegionIterArgs()) {
        if (!builder.scalar(state)) {
            output.result.note(RecognitionIssue::LoopCarriedState, outer);
            return output;
        }
    }
    if (!builder.block(*outer.getBody(), false, 0)) {
        return output;
    }
    output.countedVisits = builder.visits;
    output.countedPayloads = builder.phases;
    builder.visits = 0;
    builder.phases = 0;
    builder.coordinates.clear();
    builder.path.clear();
    if (!builder.block(*outer.getBody(), true, 0) || !detail::prepareTemplateEffects(builder)) {
        clear(output);
        return output;
    }
    if (regional) {
        const auto effects = input.accesses().effects();
        llvm::DenseSet<std::size_t> checkedSources;
        for (const auto& payload : output.payloads) {
            for (const auto& effect : payload.effects) {
                if (effect.discharge == TemplateDischarge::None ||
                    !checkedSources.insert(effect.sourceEffect).second) {
                    continue;
                }
                for (std::size_t external = 0; external < effects.size(); ++external) {
                    if (!outer->isProperAncestor(effects[external].phase->elementOp) &&
                        input.accesses().mayConflict(effect.sourceEffect, external)) {
                        output.result.note(RecognitionIssue::TemplateContext, effects[external].phase->elementOp);
                        clear(output);
                        return output;
                    }
                }
            }
        }
    }
    output.period = 1;
    output.refresh = 1;
    return output;
}
} // namespace
NumericTemplate recognizeNumericTemplate(scf::ForOp outer, const PhaseIndex& index,
                                         const SyncInput& input, NumericTemplateLimits limits)
{
    return recognizeTemplate(outer, index, input, limits, false);
}
NumericTemplate recognizeRegionalNumericTemplate(scf::ForOp outer, const PhaseIndex& index,
                                                 const SyncInput& input, NumericTemplateLimits limits)
{
    return recognizeTemplate(outer, index, input, limits, true);
}
} // namespace mlir::pto::frontiersynch
