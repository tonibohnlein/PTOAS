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
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAllocation.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/SequenceAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/RegionalAllocationCandidates.h"
#include "llvm/Support/raw_ostream.h"
namespace {
namespace fs = mlir::pto::frontiersynch;
using Summary = std::shared_ptr<fs::RegionalAllocationSummary>;
Summary palette(uint64_t budget)
{
    auto result = std::make_shared<fs::RegionalAllocationSummary>();
    fs::RegionalAllocationGroup group;
    group.sourcePipe = 1; group.targetPipe = 2; group.budget = budget;
    result->groups.push_back(std::move(group));
    return result;
}
}
bool runAllocationPreparationChecks()
{
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
    auto first = fs::requestAllocationSummary(parent, independent);
    if (!first || first->groups.front().budget != 7 || cheap != 1 || shared || parent.regionalAllocation ||
        !parent.allocationPreparation || !child.allocationPreparation) { return false; }
    if (fs::requestAllocationSummary(parent, independent) != first || cheap != 1) { return false; }
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
    if (!selected || fs::requestAllocationSummary(direct, independent) != selected || attempts != 3) { return false; }
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
    const auto recipe = plan.familyRecipe;
    const auto prepared = plan.preparation.size();
    const auto endpoints = plan.endpoints.size();
    auto cheap = fs::requestAllocationSummary(plan, fs::AllocationStrength::Independent);
    if (fs::requestAllocationSummary(plan, fs::AllocationStrength::Independent) != cheap) { return false; }
    // Keep a separate incumbent on the same retained mathematical result.
    // It establishes that the default split-pipeline request still works.
    auto incumbent = analysis.prepareLogical(outcome);
    if (mlir::failed(incumbent) || mlir::failed(analysis.attachAllocation(outcome, **incumbent))) { return false; }
    const auto originalSummary = plan.regionalAllocation;
    const auto originalCertificate = plan.allocationCertificate;
    if (mlir::succeeded(fs::selectRegionalAllocationCandidate(function, *region, plan, {6})) ||
        mlir::succeeded(fs::selectRegionalAllocationCandidate(function, *region, plan, {})) ||
        plan.regionalAllocation != originalSummary || plan.allocationCertificate != originalCertificate) {
        return false;
    }
    const llvm::SmallVector<int64_t> ids{0, 1, 2, 3, 4, 5};
    if (mlir::failed(fs::selectRegionalAllocationCandidate(function, *region, plan, ids))) {
        llvm::errs() << "regional allocation candidate did not fit fixture\n";
        return false;
    }
    if (plan.familyRecipe != recipe || plan.preparation.size() != prepared ||
        plan.endpoints.size() != endpoints || plan.allocationPreparation || render() != original) { return false; }
    llvm::errs() << "regional-allocation-candidates: independent=" << bool(cheap)
                 << "; selected certificate without scalar materialization\n";
    return true;
}
