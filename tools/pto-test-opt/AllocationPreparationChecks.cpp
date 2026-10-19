// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Parent requests must retain access to stronger child sharing on one graph.
#include "PTO/Transforms/FrontierSynch/Shared/AllocationPreparation.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalAllocationCandidates.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/AllocationInputCache.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SaveAndRestore.h"
#include <algorithm>
namespace {
namespace fs = mlir::pto::frontiersynch;
using Summary = std::shared_ptr<fs::RegionalAllocationSummary>;
using Completed = fs::AllocationPreparation::CompletedCertificate;
bool sameCertificates(llvm::ArrayRef<Completed> left, llvm::ArrayRef<Completed> right)
{
    return left.size() == right.size() &&
        std::is_permutation(left.begin(), left.end(), right.begin(), [&](const auto& entry, const auto& other) {
            return entry.owner == other.owner && entry.evidence == other.evidence;
        });
}
bool preservesCertificates(llvm::ArrayRef<Completed> before, llvm::ArrayRef<Completed> after)
{
    return llvm::all_of(before, [&](const auto& entry) {
        auto same = [&](const auto& other) { return entry.owner == other.owner && entry.evidence == other.evidence; };
        return std::count_if(before.begin(), before.end(), same) == std::count_if(after.begin(), after.end(), same);
    });
}
Summary palette(uint64_t budget)
{
    auto result = std::make_shared<fs::RegionalAllocationSummary>();
    fs::RegionalAllocationGroup group;
    group.sourcePipe = 1; group.targetPipe = 2; group.budget = budget;
    result->groups.push_back(std::move(group));
    return result;
}
bool inputCacheChecks()
{
    fs::AllocationInputCache cache;
    auto first = palette(1), distinct = palette(1);
    unsigned builds = 0;
    bool reused = true;
    auto unavailable = [&]() { ++builds; return Summary{}; };
    if (cache.request({first}, unavailable, &reused) || reused || builds != 1) { return false; }
    if (cache.request({first}, unavailable, &reused) || !reused || builds != 1) { return false; }
    auto available = [&]() { ++builds; return palette(2); };
    auto result = cache.request({distinct}, available, &reused);
    if (!result || reused || builds != 2) { return false; }
    if (cache.request({distinct}, available, &reused) != result || !reused || builds != 2) { return false; }
    if (!cache.request({distinct, first}, available, &reused) || reused || builds != 3) { return false; }
    return true;
}
}
bool runAllocationPreparationChecks()
{
    if (!inputCacheChecks()) { return false; }
    fs::PreparedLogicalPlan child(0), parent(0);
    unsigned cheap = 0, shared = 0;
    child.allocationPreparation = fs::AllocationPreparation::candidates(
        [&](fs::PreparedLogicalPlan&, fs::AllocationStrength strength) {
        if (strength == fs::AllocationStrength::Independent) { ++cheap; return palette(2); }
        ++shared; return palette(1);
    });
    parent.allocationPreparation = fs::AllocationPreparation::candidates(
        [&](fs::PreparedLogicalPlan&, fs::AllocationStrength strength) {
        auto result = fs::requestAllocationSummary(child, strength);
        if (!result) { return Summary{}; }
        auto combined = std::make_shared<fs::RegionalAllocationSummary>(*result);
        combined->groups.front().budget += 5;
        return combined;
    });
    const auto independent = fs::AllocationStrength::Independent;
    const auto stronger = fs::AllocationStrength::Shared;
    if (parent.allocationPreparation.retained(independent) ||
        parent.allocationPreparation.retained(stronger) || cheap || shared) { return false; }
    auto first = fs::requestAllocationSummary(parent, independent);
    if (!first || first->groups.front().budget != 7 || cheap != 1 || shared || parent.regionalAllocation ||
        !parent.allocationPreparation || !child.allocationPreparation) { return false; }
    if (parent.allocationPreparation.retained(independent) != first ||
        parent.allocationPreparation.retained(stronger) ||
        fs::requestAllocationSummary(parent, independent) != first || cheap != 1) { return false; }
    auto second = fs::requestAllocationSummary(parent, stronger);
    if (!second || second->groups.front().budget != 6 || cheap != 1 || shared != 1 ||
        parent.allocationPreparation || child.allocationPreparation) { return false; }
    if (fs::requestAllocationSummary(parent, stronger) != second ||
        fs::requestAllocationSummary(parent, independent) != first || shared != 1) { return false; }
    // Failed independent construction is cached but never blocks a stronger
    // request. Requesting Shared first never constructs a weaker answer later.
    fs::PreparedLogicalPlan failed(0), direct(0);
    unsigned attempts = 0;
    failed.allocationPreparation = fs::AllocationPreparation::candidates(
        [&](fs::PreparedLogicalPlan&, fs::AllocationStrength strength) {
        ++attempts; return strength == independent ? Summary{} : palette(1);
    });
    if (fs::requestAllocationSummary(failed, independent) ||
        fs::requestAllocationSummary(failed, independent) || attempts != 1 ||
        !fs::requestAllocationSummary(failed, stronger) || attempts != 2) { return false; }
    direct.allocationPreparation = fs::AllocationPreparation::candidates(
        [&](fs::PreparedLogicalPlan&, fs::AllocationStrength strength) {
        ++attempts; return strength == stronger ? palette(1) : Summary{};
    });
    auto selected = fs::requestAllocationSummary(direct, stronger);
    if (!selected || direct.allocationPreparation.retained(independent) != selected ||
        direct.allocationPreparation.retained(stronger) != selected ||
        fs::requestAllocationSummary(direct, independent) != selected || attempts != 3) { return false; }
    // Legacy producers stay one-shot. Moving a capability into a regional view
    // preserves its ownership and does not materialize deferred scalar code.
    fs::PreparedLogicalPlan legacy(0);
    unsigned legacyCalls = 0;
    legacy.allocationPreparation = [&](fs::PreparedLogicalPlan& plan) {
        ++legacyCalls; plan.regionalAllocation = palette(1);
    };
    auto view = fs::takeRegionalAllocationView(legacy, true);
    if (!view || legacy.allocationPreparation) { return false; }
    auto answer = fs::requestAllocationSummary(*view, independent);
    if (!answer || fs::requestAllocationSummary(*view, stronger) != answer || legacyCalls != 1 ||
        view->allocationPreparation || !view->preparation.empty()) { return false; }
    mlir::MLIRContext context;
    fs::PreparedLogicalPlan certificateOnly(0);
    auto evidence = mlir::DictionaryAttr::get(&context);
    certificateOnly.allocationPreparation = [&](fs::PreparedLogicalPlan& plan) {
        ++legacyCalls; plan.allocationCertificate = evidence;
    };
    if (fs::requestAllocationSummary(certificateOnly, independent) || certificateOnly.allocationCertificate ||
        legacyCalls != 2) { return false; }
    fs::prepareAllocationSupport(certificateOnly);
    if (certificateOnly.allocationCertificate != evidence || legacyCalls != 2) { return false; }
    return true;
}

bool runRegionalAllocationCandidateChecks(mlir::func::FuncOp function,
    fs::FrontierAnalysis& analysis, const fs::AnalysisOutcome& outcome, fs::PreparedLogicalPlan& plan)
{
    auto region = outcome.regionalExports;
    if (!region && outcome.mathematical) {
        region = outcome.mathematical->regionalDemands;
        if (!region && outcome.mathematical->sequenceDemands) {
            region = std::make_shared<fs::RegionalAnalysis>(
                fs::sequenceRegionalResult(*outcome.mathematical->sequenceDemands));
        }
    }
    if (!region || !plan.familyRecipe) {
        llvm::errs() << "candidate test requires retained regional recipes\n";
        return false;
    }
    auto render = [&]() {
        std::string text;
        llvm::raw_string_ostream stream(text);
        function.print(stream);
        return text;
    };
    const auto original = render();
    if (mlir::failed(fs::normalizeLogicalFamilies(plan))) { return false; }
    const auto attempts = analysis.constructionCounts().mathematicalAttempts;
    const auto recipe = plan.familyRecipe;
    const auto prepared = plan.preparation.size();
    const auto endpoints = plan.endpoints.size();
    auto cheap = fs::requestAllocationSummary(plan, fs::AllocationStrength::Independent);
    if (fs::requestAllocationSummary(plan, fs::AllocationStrength::Independent) != cheap) { return false; }
    if (!cheap || !plan.allocationPreparation.retainedCertificates(plan).empty()) { return false; }
    mlir::DictionaryAttr completed;
    {
        llvm::SaveAndRestore<Summary> selected(plan.regionalAllocation, cheap);
        // This explicitly requested parent export performs the real proof.
        // Subsequent candidate selection must reuse its completed evidence.
        completed = fs::regionalBaseAllocationCertificate(*region, plan);
    }
    const auto retained = plan.allocationPreparation.retainedCertificates(plan);
    if (!completed || retained.size() != 1 || retained.front().owner != cheap ||
        retained.front().evidence != completed) { return false; }
    mlir::NamedAttrList otherScope(completed);
    otherScope.set("test_completed_scope", mlir::UnitAttr::get(function.getContext()));
    const auto scoped = otherScope.getDictionary(function.getContext());
    plan.allocationPreparation.retainCertificate(plan, palette(1), scoped);
    if (plan.allocationPreparation.retainedCertificates(plan).size() != 1) { return false; }
    plan.allocationPreparation.retainCertificate(plan, cheap, scoped, {0, 1});
    if (plan.allocationPreparation.retainedCertificates(plan).size() != 1 ||
        plan.allocationPreparation.retainedCertificates(plan, {0, 1}).size() != 2 ||
        plan.allocationPreparation.retainedCertificates(plan, {1, 0}).size() != 1) { return false; }
    {
        llvm::SaveAndRestore<bool> changed(plan.completeInvocation, !plan.completeInvocation);
        if (!plan.allocationPreparation.retainedCertificates(plan).empty()) { return false; }
    }
    if (plan.allocationPreparation.retainedCertificates(plan).size() != 1) { return false; }
    // Keep a separate incumbent on the same retained mathematical result.
    // It establishes that the default split-pipeline request still works.
    auto incumbent = analysis.prepareLogical(outcome);
    if (mlir::failed(incumbent) || mlir::failed(analysis.attachAllocation(outcome, **incumbent))) { return false; }
    const auto priorUnrestricted = plan.allocationPreparation.retainedCertificates(plan);
    const auto priorScoped = plan.allocationPreparation.retainedCertificates(plan, {0, 1});
    const auto priorReversed = plan.allocationPreparation.retainedCertificates(plan, {1, 0});
    const llvm::SmallVector<int64_t> ids{0, 1, 2, 3, 4, 5};
    const auto priorSelected = plan.allocationPreparation.retainedCertificates(plan, ids);
    const auto originalSummary = plan.regionalAllocation;
    const auto originalCertificate = plan.allocationCertificate;
    if (mlir::succeeded(analysis.attachAllocation(outcome, plan, {8})) ||
        mlir::succeeded(analysis.attachAllocation(outcome, plan, {0, 0})) ||
        mlir::succeeded(fs::selectRegionalAllocationCandidate(function, *region, plan, {8})) ||
        mlir::succeeded(fs::selectRegionalAllocationCandidate(function, *region, plan, {})) ||
        plan.regionalAllocation != originalSummary || plan.allocationCertificate != originalCertificate) {
        return false;
    }
    // Both the unrestricted export and failed empty-pool placement can finish
    // valid lane proofs. Failure preserves the selected answer, but completed
    // evidence remains available. Snapshot only after these required attempts.
    const auto beforeSelection = plan.allocationPreparation.retainedCertificates(plan);
    bool retainedOriginal = false;
    for (const auto& entry : beforeSelection) {
        if (entry.owner != cheap || !entry.evidence ||
            mlir::failed(fs::decodeRegionalAllocation(function, entry.evidence, ids))) {
            llvm::errs() << "candidate archive: invalid completed owner or certificate\n";
            return false;
        }
        if (entry.evidence == completed) { retainedOriginal = true; continue; }
        mlir::NamedAttrList base(entry.evidence);
        if (!base.erase("lane_evidence") || base.getDictionary(function.getContext()) != completed) {
            llvm::errs() << "candidate archive: completed proof changed original base\n";
            return false;
        }
    }
    if (!retainedOriginal) {
        llvm::errs() << "candidate archive: original completed base lost\n";
        return false;
    }
    auto scopedExpected = beforeSelection;
    scopedExpected.push_back({cheap, scoped});
    const auto beforeScoped = plan.allocationPreparation.retainedCertificates(plan, {0, 1});
    const auto beforeReversed = plan.allocationPreparation.retainedCertificates(plan, {1, 0});
    const auto beforeSelected = plan.allocationPreparation.retainedCertificates(plan, ids);
    if (!sameCertificates(beforeScoped, scopedExpected) || !sameCertificates(beforeReversed, beforeSelection) ||
        !sameCertificates(beforeSelected, beforeSelection)) {
        llvm::errs() << "candidate archive: ordered pool isolation changed\n";
        return false;
    }
    if (!preservesCertificates(priorUnrestricted, beforeSelection) ||
        !preservesCertificates(priorScoped, beforeScoped) ||
        !preservesCertificates(priorReversed, beforeReversed) ||
        !preservesCertificates(priorSelected, beforeSelected)) {
        llvm::errs() << "candidate archive: failed selection discarded completed evidence\n";
        return false;
    }
    if (mlir::failed(analysis.attachAllocation(outcome, plan, ids))) {
        llvm::errs() << "regional allocation candidate did not fit fixture\n";
        return false;
    }
    if (!cheap || plan.regionalAllocation != cheap || plan.familyRecipe != recipe ||
        plan.preparation.size() != prepared ||
        plan.endpoints.size() != endpoints || plan.allocationPreparation || render() != original) { return false; }
    // The callback is gone, but its proof-only archive survives. Capacity
    // partitions and original source owners remain exact after this transfer.
    const auto archived = plan.allocationPreparation.retainedCertificates(plan);
    if (!sameCertificates(archived, beforeSelection) ||
        !sameCertificates(plan.allocationPreparation.retainedCertificates(plan, {0, 1}), beforeScoped) ||
        !sameCertificates(plan.allocationPreparation.retainedCertificates(plan, {1, 0}), beforeReversed) ||
        !sameCertificates(plan.allocationPreparation.retainedCertificates(plan, ids), beforeSelected)) {
        llvm::errs() << "candidate archive: callback consumption changed completed inventory\n";
        return false;
    }
    auto unrestricted = analysis.prepareLogical(outcome);
    if (mlir::failed(unrestricted) || mlir::failed(analysis.attachAllocation(outcome, **unrestricted)) ||
        (*unrestricted)->allocationCertificate != (*incumbent)->allocationCertificate ||
        analysis.constructionCounts().mathematicalAttempts != attempts) { return false; }
    auto consumed = analysis.prepareLogical(outcome);
    if (mlir::failed(consumed) || mlir::failed(fs::normalizeLogicalFamilies(**consumed))) { return false; }
    auto consumedSource = fs::requestAllocationSummary(**consumed, fs::AllocationStrength::Independent);
    mlir::DictionaryAttr consumedEvidence;
    {
        llvm::SaveAndRestore<Summary> selected((*consumed)->regionalAllocation, consumedSource);
        consumedEvidence = fs::regionalBaseAllocationCertificate(*region, **consumed);
    }
    if (!consumedEvidence) { return false; }
    fs::prepareAllocationSupport(**consumed);
    const auto preserved = (*consumed)->allocationPreparation.retainedCertificates(**consumed);
    if ((*consumed)->allocationPreparation || !llvm::any_of(preserved, [&](const auto& entry) {
            return entry.owner == consumedSource && entry.evidence == consumedEvidence;
        }) || render() != original) { return false; }
    llvm::errs() << "regional-allocation-candidates: independent=" << bool(cheap)
                 << "; selected certificate without scalar materialization\n";
    return true;
}
