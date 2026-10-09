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
#include "PTO/Transforms/FrontierSynch/MixedStrideAnalysis.h"
#include "PTO/Transforms/FrontierSynch/CompactBoundingInsertion.h"
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicInsertion.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticPeriodicConversion.h"
#include "PTO/Transforms/FrontierSynch/BoundedLifetimeInsertion.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ArithmeticInsertion.h"
#include "PTO/Transforms/FrontierSynch/PeriodicSharedCertificate.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
namespace mlir::pto::frontiersynch {
namespace {
bool sameOriginalCoordinates(const RegionalAnalysis& a, const RegionalAnalysis& b)
{
    const bool sameDomain = a.expressions == b.expressions && a.anchors.size() == b.anchors.size() &&
        a.occurrenceLoops == b.occurrenceLoops && a.accessModel == b.accessModel &&
        a.gmAliasPolicy == b.gmAliasPolicy;
    if (!sameDomain) { return false; }
    // Specialized phase views need their own explicit coordinate adapter.
    if (a.firstOrdinal != b.firstOrdinal || a.endpointSiteGuard != b.endpointSiteGuard ||
        a.endpointInvocationGuard != b.endpointInvocationGuard) { return false; }
    for (std::size_t i = 0; i < a.anchors.size(); ++i) {
        const auto& x = a.anchors[i];
        const auto& y = b.anchors[i];
        if (x.phase != y.phase || x.before.block != y.before.block || x.before.before != y.before.before ||
            x.after.block != y.after.block || x.after.before != y.after.before ||
            x.coordinates.size() != y.coordinates.size()) { return false; }
        for (std::size_t j = 0; j < x.coordinates.size(); ++j) {
            if (x.coordinates[j].loop != y.coordinates[j].loop ||
                x.coordinates[j].induction != y.coordinates[j].induction) { return false; }
        }
        const auto ax = a.outerLoops.empty() ? ArrayRef<scf::ForOp>() : ArrayRef<scf::ForOp>(a.outerLoops[i]);
        const auto bx = b.outerLoops.empty() ? ArrayRef<scf::ForOp>() : ArrayRef<scf::ForOp>(b.outerLoops[i]);
        if (ax != bx) { return false; }
        for (std::size_t j = 0; j < ax.size(); ++j) {
            const auto ad = a.outerDivisors.empty() ? 1 : a.outerDivisors[i][j];
            const auto bd = b.outerDivisors.empty() ? 1 : b.outerDivisors[i][j];
            if (ad != bd) { return false; }
        }
    }
    return true;
}
} // namespace
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
        return prepareNumericTemplateLogicalInsertion(function, *program);
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
        return plan;
    }
    if (demands.guardedRotatingDemands) {
        const auto& analysis = *demands.guardedRotatingDemands;
        GuardedPeriodicEndpointInput input{analysis.loop, analysis.expressions, analysis.phases,
            analysis.payloads, analysis.generators, &analysis.periodic};
        auto plan = prepareGuardedPeriodicEndpoints(function, input, error);
        const bool regional = succeeded(plan) && demands.region != 0;
        if (regional) { (*plan)->completeInvocation = false; }
        return plan;
    }
    if (demands.mixedStrideDemands) {
        return prepareMixedStrideLogicalInsertion(function, *demands.mixedStrideDemands, error);
    }
    if (demands.compactDemands) {
        return prepareCompactBoundingLogicalInsertion(function, demands.compactDemands, error);
    }
    if (demands.boundedDemands) {
        return prepareBoundedLifetimeLogicalResult(demands.boundedDemands, error, demands.region == 0);
    }
    if (demands.varyingBoundaryDemands) {
        auto regional = varyingExports(demands, false, error);
        if (!regional || !regional->prepareWithVisits) { return failure(); }
        auto result = regional->prepareWithVisits(program->nodes[demands.region].loops);
        if (succeeded(result)) { (*result)->completeInvocation = demands.region == 0; }
        return result;
    }
    if (demands.sequenceDemands) {
        SequenceEndpointResolver resolver = [this](std::size_t region, const RegionalAnalysis& retained,
            ArrayRef<scf::ForOp> enclosing) -> FailureOr<std::unique_ptr<PreparedLogicalPlan>> {
            const bool sameFrame = ArrayRef<scf::ForOp>(program->nodes[region].loops) == enclosing;
            if (!sameFrame) { return failure(); }
            AnalysisRequest request;
            request.region = region;
            request.needs.queries = request.needs.selectors = request.needs.synchronization = true;
            auto selected = analyze(request);
            auto exports = selected.regionalExports ? selected.regionalExports :
                (selected.mathematical ? selected.mathematical->regionalDemands : nullptr);
            const bool available = selected.status == AnalysisStatus::Ready && selected.mathematical &&
                                   exports;
            if (!available) { return failure(); }
            // Both results certify the exact order for this unchanged original
            // region. Reuse crossing queries only with identical event identities;
            // query callbacks may differ while describing the same exact order.
            if (!sameOriginalCoordinates(retained, *exports)) { return failure(); }
            return prepareLogical(selected);
        };
        return prepareSequenceLogicalInsertion(*demands.sequenceDemands, program->nodes[demands.region].loops,
                                                resolver);
    }
    if (demands.regionalDemands) {
        const auto& regional = *demands.regionalDemands;
        const auto& enclosing = program->nodes[demands.region].loops;
        if (regional.prepareWithVisits) { return regional.prepareWithVisits(enclosing); }
        const bool flatRecipe = enclosing.empty() && regional.prepare;
        if (flatRecipe) { return regional.prepare(); }
        return failure();
    }
    if (demands.arithmeticPeriodicDemands) {
        const auto& converted = *demands.arithmeticPeriodicDemands;
        const auto& conversion = converted.conversion;
        std::vector<uint64_t> residues;
        for (const auto& site : converted.sites) { residues.push_back(site.residue); }
        if (conversion.numerical) {
            NumericalPeriodicEndpointInput input{converted.loop, conversion.expressions, converted.phases,
                &*conversion.numerical, converted.period, residues};
            return prepareNumericalPeriodicEndpoints(function, input, error);
        }
        if (!conversion.guarded) { return failure(); }
        GuardedPeriodicEndpointInput input{converted.loop, conversion.expressions, converted.phases,
            conversion.payloads, conversion.generators, &*conversion.guarded, converted.period, residues};
        return prepareGuardedPeriodicEndpoints(function, input, error);
    }
    if (demands.arithmeticDemands) {
        auto& records = sessionState->attempts[demands.region][AnalysisBackend::Arithmetic].arithmeticRecords;
        return prepareArithmeticLogicalInsertion(function, *program->arithmetic, *demands.arithmeticDemands,
                                                 error, records);
    }
    if (demands.generalArithmeticDemands) {
        auto& records = sessionState->attempts[demands.region][AnalysisBackend::Arithmetic].arithmeticRecords;
        return prepareGeneralArithmeticLogicalInsertion(function, *program->arithmetic,
                                                        *demands.generalArithmeticDemands, error, records);
    }
    if (demands.finiteGuardedDemands) { return prepareFiniteGuardedLogicalInsertion(*demands.finiteGuardedDemands); }
    return failure();
}
} // namespace mlir::pto::frontiersynch
