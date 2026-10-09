// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/CompactBoundingInsertion.h"
#include "PTO/Transforms/FrontierSynch/CompactBoundingAllocation.h"
#include "PTO/Transforms/FrontierSynch/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/FiniteAllocation.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
namespace mlir::pto::frontiersynch {
namespace {
using Preparation = FailureOr<std::unique_ptr<PreparedLogicalPlan>>;
bool matchesInvocation(Block* block, ArrayRef<scf::ForOp> context)
{
    if (!block) { return false; }
    SmallVector<scf::ForOp> actual;
    for (auto* parent = block->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) { actual.push_back(loop); }
    }
    std::reverse(actual.begin(), actual.end());
    return ArrayRef<scf::ForOp>(actual) == context;
}
CompactClasses attachLeaf(func::FuncOp function, const PhaseIndex& index, CompactClasses source,
                          const std::shared_ptr<std::string>& error)
{
    if (!source) { return source; }
    if (auto frame = source->finiteFrame()) {
        // No replacement is selected by this automatic route. A future selected
        // finite producer must emit its selected definitions, not original covers.
        if (source->finiteSelection()) { *error = "selected finite endpoint records unavailable"; return source; }
        return withCompactClassPreparation(source,
            [function, frame, error](ArrayRef<scf::ForOp> context) -> Preparation {
            if (!matchesInvocation(frame->invocationBlock(), context)) {
                *error = "finite endpoint context differs from its original invocation"; return failure();
            }
            auto prepared = prepareExplicitInsertion(function, frame->analysis(), false);
            if (succeeded(prepared)) {
                (*prepared)->completeInvocation = false;
                (*prepared)->allocationPreparation = [frame, context = function->getContext()](
                    PreparedLogicalPlan& plan) {
                    plan.allocationCertificate = explicitAllocationCertificate(
                        frame->analysis(), plan.planId, context);
                };
            }
            else { *error = "finite selected endpoint preparation unavailable"; }
            return prepared;
        });
    }
    auto compact = source->compact();
    if (!compact || !compact->upperGraph || !source->bounds().upper) { return source; }
    const auto& domain = compact->domain->context()->domain();
    if (domain.occurrenceLoops.empty()) { return source; }
    auto body = std::make_shared<const BalancedCompactBody>(
        recognizeBalancedCompactBody(domain.occurrenceLoops.front(), compact->domain->context()->input(), index));
    if (!body->error.empty()) { *error = body->error; return source; }
    auto region = source->bounds().upper->regional();
    return withCompactClassPreparation(source,
        [function, compact, body, region, error](ArrayRef<scf::ForOp> context) -> Preparation {
            auto prepared = prepareBalancedCompactInsertion(function, *body,
                compact->upperGraph->analysis(), 0, *error, context);
            if (failed(prepared)) { return failure(); }
            // This optional proof uses the weaker selected order, hence stays
            // sound when actual local barrier placement adds ordering.
            (*prepared)->allocationPreparation = [region, compact](PreparedLogicalPlan& plan) {
                std::string allocationError;
                attachCompactBoundingAllocation(region, compact->upperGraph->analysis(),
                    compact->domain->trips(), plan, allocationError);
            };
            return prepared;
        });
}
} // namespace
CompactBoundingPreparation analyzeCompactBounding(
    func::FuncOp function, std::shared_ptr<const SyncInput> input, const PhaseIndex& index)
{
    CompactBoundingPreparation result;
    auto owner = std::make_shared<CompactBoundingOwner>(); owner->input = std::move(input);
    result.owner = owner;
    if (!function || function.isDeclaration() || !llvm::hasSingleElement(function.getBody()) || !owner->input) {
        result.error = "compact fallback needs one original function block and shared input"; return result;
    }
    auto arena = std::make_shared<RegionExpressions>();
    for (const auto* phase : owner->input->instructions()) { arena->forbidRecomputation(phase->elementOp); }
    auto preparationError = std::make_shared<std::string>();
    owner->preparationError = preparationError;
    auto built = captureCompactClassSequence(function, function.front(), *owner->input, arena,
        [function, &index, preparationError](CompactClasses leaf) {
            return attachLeaf(function, index, std::move(leaf), preparationError);
        });
    owner->boundary = built.boundary; owner->captured = std::move(built.captured);
    if (!built.error.empty() || !owner->boundary) {
        result.error = built.error.empty() ? "compact selected class construction unavailable" : built.error;
        return result;
    }
    bool missingPrerequisite = false;
    function.walk([&](Operation* operation) {
        if (!index.phasesFor(operation).empty()) { return; }
        missingPrerequisite |= index.needsValuePrerequisite(operation);
        for (const auto& prerequisite : index.prerequisitesFor(operation)) {
            missingPrerequisite |= !prerequisite.native;
        }
    });
    if (missingPrerequisite) {
        result.error = "phase-less completion prerequisite needs a compact order/endpoint interface"; return result;
    }
    return result;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareCompactBoundingLogicalInsertion(
    func::FuncOp function, std::shared_ptr<const CompactBoundingOwner> owner, std::string& error)
{
    if (!owner || !owner->boundary || !owner->input) { return failure(); }
    const auto& region = owner->boundary->nativeExports();
    if (!region.prepare && !region.prepareWithVisits) {
        error = owner->boundary->exportError().empty() ?
            "compact selected mathematics has no endpoint export" : owner->boundary->exportError();
        return failure();
    }
    auto prepared = region.prepareWithVisits ? region.prepareWithVisits({}) : region.prepare();
    if (failed(prepared)) {
        error = !owner->preparationError || owner->preparationError->empty() ?
            "compact selected endpoint preparation unavailable" : *owner->preparationError;
        return failure();
    }
    (*prepared)->completeInvocation = !owner->input->instructions().empty();
    (*prepared)->compactBoundingOwner = std::move(owner);
    return prepared;
}
DictionaryAttr compactBoundingAllocationCertificate(const CompactBoundingOwner& owner, PreparedLogicalPlan& plan)
{
    prepareAllocationSupport(plan);
    auto certificate = plan.allocationCertificate;
    if (!certificate && plan.regionalAllocation) {
        auto selected = owner.boundary->nativeExports();
        if (owner.boundary->bounds().upper) {
            selected.reachability = owner.boundary->bounds().upper->regional().reachability;
        }
        certificate = regionalAllocationCertificate(selected, plan);
    }
    const bool notifications = llvm::any_of(plan.endpoints, [](const auto& endpoint) {
        return endpoint.kind != LogicalCommandKind::Barrier;
    });
    owner.allocationError = notifications && !certificate ? "selected compact allocation certificate unavailable" : "";
    return certificate;
}
CompactBoundingPreparation prepareCompactBoundingInsertion(
    func::FuncOp function, std::shared_ptr<const SyncInput> input)
{
    CompactBoundingPreparation result;
    const bool valid = function && !function.isDeclaration() && llvm::hasSingleElement(function.getBody()) && input;
    if (!valid) {
        result.error = "compact fallback needs one original function block and shared input"; return result;
    }
    PhaseIndex index;
    if (failed(index.build(function, *input))) {
        result.error = "compact fallback shared phase index unavailable"; return result;
    }
    result = analyzeCompactBounding(function, std::move(input), index);
    if (!result.error.empty()) { return result; }
    auto prepared = prepareCompactBoundingLogicalInsertion(function, result.owner, result.exportError);
    if (failed(prepared)) { return result; }
    result.prepared = std::move(*prepared);
    result.prepared->allocationCertificate = compactBoundingAllocationCertificate(*result.owner, *result.prepared);
    return result;
}
} // namespace mlir::pto::frontiersynch
