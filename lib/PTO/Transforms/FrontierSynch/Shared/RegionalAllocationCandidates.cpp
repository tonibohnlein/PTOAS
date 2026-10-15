// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "RegionalAllocationCandidates.h"
#include "CertifiedAllocationCandidates.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/SaveAndRestore.h"
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
bool fits(func::FuncOp function, DictionaryAttr certificate, ArrayRef<int64_t> eligibleIds)
{
    if (!certificate) { return false; }
    // Candidate refusal is expected. The caller reports whole-request failure;
    // a rejected sufficient assignment is not a minimum-capacity lower bound.
    ScopedDiagnosticHandler diagnostics(function.getContext(), [](Diagnostic&) { return success(); });
    return succeeded(decodeRegionalAllocation(function, certificate, eligibleIds));
}
}
LogicalResult selectRegionalAllocationCandidate(func::FuncOp function,
    const RegionalAnalysis& region, PreparedLogicalPlan& plan, ArrayRef<int64_t> eligibleIds,
    AllocationProfile* profile)
{
    if (!function || plan.privateState || plan.completionOwner ||
        llvm::any_of(region.anchors, [&](const TemplateEndpointAnchor& anchor) {
            return !anchor.before.before || anchor.before.before->getParentOfType<func::FuncOp>() != function;
        })) { return failure(); }
    std::set<int64_t> unique;
    for (auto id : eligibleIds) {
        if (id < 0 || id > 5 || !unique.insert(id).second) { return failure(); }
    }
    if (failed(normalizeLogicalFamilies(plan))) {
        return failure();
    }
    const auto incumbent = plan.regionalAllocation;
    const auto certificate = plan.allocationCertificate;
    auto prior = certifiedAllocationCandidates(certificate);
    if (failed(prior)) { return failure(); }
    SmallVector<DictionaryAttr> available = std::move(*prior);
    if (certificate) { available.push_back(withoutAllocationInventory(certificate)); }
    auto finish = [&](const std::shared_ptr<RegionalAllocationSummary>& selected, DictionaryAttr evidence) {
        for (const auto& completed : plan.allocationPreparation.retainedCertificates(plan, eligibleIds)) {
            available.push_back(completed.evidence);
        }
        plan.regionalAllocation = selected;
        plan.allocationCertificate = withAllocationInventory(evidence, available);
        plan.allocationPreparation.preserveCertificates(plan);
        // Only completed proof data crosses the saved-artifact boundary.
        plan.allocationPreparation = AllocationPreparation{};
        return success();
    };
    if (fits(function, certificate, eligibleIds)) { return finish(incumbent, certificate); }
    // A previous required parent export may already have the exact proof for
    // this protocol. Reading it never invokes a summary or proof producer.
    for (const auto& completed : plan.allocationPreparation.retainedCertificates(plan, eligibleIds)) {
        if (fits(function, completed.evidence, eligibleIds)) {
            return finish(completed.owner, completed.evidence);
        }
    }
    std::shared_ptr<RegionalAllocationSummary> previous;
    for (auto strength : {AllocationStrength::Independent, AllocationStrength::Shared}) {
        auto candidate = plan.allocationPreparation.retained(strength);
        if (!candidate) { candidate = requestAllocationSummary(plan, strength, profile); }
        if (!candidate || candidate == previous) { continue; }
        previous = candidate;
        DictionaryAttr evidence;
        bool accepted = false;
        {
            llvm::SaveAndRestore<std::shared_ptr<RegionalAllocationSummary>> selected(
                plan.regionalAllocation, candidate);
            for (const auto& completed : plan.allocationPreparation.retainedCertificates(plan, eligibleIds)) {
                if (completed.owner == candidate) { evidence = completed.evidence; }
            }
            if (!evidence) {
                evidence = regionalBaseAllocationCertificate(region, plan, profile, eligibleIds);
                plan.allocationPreparation.retainCertificate(plan, candidate, evidence, eligibleIds);
            }
            accepted = fits(function, evidence, eligibleIds);
            if (!accepted) {
                auto stronger = strengthenRegionalAllocationCertificate(region, plan, evidence, profile, eligibleIds);
                if (stronger != evidence) {
                    evidence = stronger;
                    plan.allocationPreparation.retainCertificate(plan, candidate, evidence, eligibleIds);
                    accepted = fits(function, evidence, eligibleIds);
                }
            }
        }
        if (accepted) { return finish(candidate, evidence); }
    }
    plan.regionalAllocation = incumbent;
    plan.allocationCertificate = certificate;
    return failure();
}
} // namespace mlir::pto::frontiersynch
