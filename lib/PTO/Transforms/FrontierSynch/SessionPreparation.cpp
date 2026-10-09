// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Detached preparation consumes retained mathematics. Producer and reducer
// work is never repeated after a missing synchronization export.
#include "AnalysisSessionInternal.h"
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingInsertion.h"
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticInsertion.h"
#include "PTO/Transforms/FrontierSynch/PeriodicSharedCertificate.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
namespace mlir::pto::frontiersynch {
FailureOr<std::unique_ptr<PreparedLogicalPlan>> FrontierAnalysis::prepareRetained(const MathematicalResult& demands)
{
    std::string error;
    if (demands.explicitDemands) {
        auto plan = prepareExplicitInsertion(function, *demands.explicitDemands, false);
        const bool nativeScalar = succeeded(plan) && demands.backend == "native-scalar";
        if (nativeScalar) {
            (*plan)->completeInvocation = !storage->instructions().empty();
        }
        return plan;
    }
    if (demands.numericNode || demands.rotatingDemands) {
        const auto bits = DataLayout::closest(function).getTypeSizeInBits(IndexType::get(function.getContext()));
        const bool supported = !bits.isScalable() && bits.getFixedValue() == 64;
        if (!supported) { return failure(); }
    }
    if (demands.numericNode) {
        // Original nested numerical coordinates need a regional insertion
        // adapter; the whole-invocation emitter cannot stand in for it.
        if (demands.region != 0) { return failure(); }
        auto& node = program->nodes[*demands.numericNode];
        if (!node.logicalEndpoints) {
            node.logicalEndpoints = buildNumericTemplateEndpoints(*node.numericTemplate, *node.periodicAnalysis);
        }
        if (!node.logicalEndpoints->logical.error.empty()) { return failure(); }
        return prepareNumericTemplateInsertion(function, *program);
    }
    if (demands.rotatingDemands) {
        const auto& rotating = *demands.rotatingDemands;
        auto& recipe = sessionState->attempts[demands.region][AnalysisBackend::Rotating].periodicEndpoints;
        if (!recipe) {
            SmallVector<TemplateEndpointAnchor> anchors;
            for (auto* phase : rotating.phases) {
                auto* op = phase->elementOp;
                anchors.push_back({phase, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
            }
            recipe = std::make_shared<const NumericTemplateEndpoints>(
                bindPeriodicEndpoints(rotating.loop, anchors, rotating.periodic));
        }
        const auto& endpoints = *recipe;
        if (!endpoints.logical.error.empty()) { return failure(); }
        auto plan = std::make_unique<PreparedLogicalPlan>(0);
        plan->completeInvocation = demands.region == 0 && !rotating.phases.empty();
        if (failed(prepareCountedEndpointCode(function, endpoints, *plan))) { return failure(); }
        plan->allocationCertificate = encodePeriodicSharedAllocation(rotating.periodic, plan->planId,
                                                                     function.getContext());
        return plan;
    }
    if (demands.guardedRotatingDemands) {
        auto plan = prepareGuardedRotatingEndpoints(function, *demands.guardedRotatingDemands, error);
        const bool regional = succeeded(plan) && demands.region != 0;
        if (regional) { (*plan)->completeInvocation = false; }
        return plan;
    }
    if (demands.boundedDemands) {
        return prepareBoundedLifetimeResult(demands.boundedDemands, error, demands.region == 0);
    }
    if (demands.sequenceDemands) { return prepareSequenceInsertion(*demands.sequenceDemands); }
    if (demands.regionalDemands) {
        const auto& regional = *demands.regionalDemands;
        if (regional.prepare) { return regional.prepare(); }
        return failure();
    }
    if (demands.arithmeticPeriodicDemands) { return prepareArithmeticPeriodicFunction(); }
    if (demands.arithmeticDemands) {
        return prepareArithmeticInsertion(function, *program->arithmetic, *demands.arithmeticDemands, error);
    }
    if (demands.generalArithmeticDemands) {
        return prepareGeneralArithmeticInsertion(function, *program->arithmetic,
                                                 *demands.generalArithmeticDemands, error);
    }
    if (demands.finiteGuardedDemands) { return prepareFiniteGuardedInsertion(*demands.finiteGuardedDemands); }
    return failure();
}
} // namespace mlir::pto::frontiersynch
