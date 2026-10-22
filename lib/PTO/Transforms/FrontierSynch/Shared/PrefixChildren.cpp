// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Explicit provider capability request; no backend selection policy is added.
#include "PrefixPreparation.h"
#include "CompletionStorage.h"
#include "PTO/Transforms/FrontierSynch/Shared/RetainedInventory.h"
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include <set>
#include <chrono>
namespace mlir::pto::frontiersynch::prefix {
bool Preparation::insideChild(Operation* operation) const
{
    return boundary && boundary->invocation->isProperAncestor(operation);
}
LogicalResult Preparation::prepareChild(fs::PrefixInput& description, fs::ServiceMask& masks)
{
    if (!childRegion && !childProvider) {
        return success();
    }
    if (childRegion && childProvider) {
        return refuse(fs::PrefixRefusal::Certificate, "ambiguous child provider/frame request");
    }
    if (pruning) {
        return refuse(fs::PrefixRefusal::Certificate, "child-aware pruning is not certified");
    }
    if (llvm::any_of(description.sites, [](const auto& site) { return !site.refreshed.empty(); })) {
        return refuse(fs::PrefixRefusal::Certificate, "child entry union requires the raw no-KILL baseline");
    }
    const auto start = std::chrono::steady_clock::now();
    std::string backend = "supplied-closed-provider";
    Operation* requestedInvocation = nullptr;
    if (childProvider) {
        child = childProvider(function);
    } else {
        fs::FrontierAnalysis session(function);
        if (failed(session.initialize(aliasPolicy))) {
            return refuse(fs::PrefixRefusal::Admission, "child analysis input unavailable");
        }
        const auto* program = session.result();
        if (!program || *childRegion >= program->nodes.size() ||
            program->nodes[*childRegion].kind != fs::StructureKind::Loop ||
            !isa_and_nonnull<scf::ForOp>(program->nodes[*childRegion].anchor)) {
            return refuse(fs::PrefixRefusal::Certificate,
                "prefix child request requires a whole original counted-loop region");
        }
        requestedInvocation = program->nodes[*childRegion].anchor;
        fs::AnalysisRequest request;
        request.region = *childRegion;
        auto answer = session.analyze(request);
        if (!answer.mathematical || !answer.mathematical->completionChild) {
            return refuse(fs::PrefixRefusal::Certificate, "retained mathematics has no closed child export");
        }
        backend = answer.mathematical->backend;
        work["child_retained_inventory"] = fs::retainedAnalysisInventory(*answer.mathematical);
        work["child_analysis_seconds"] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto qualificationStart = std::chrono::steady_clock::now();
        child = answer.mathematical->completionChild();
        work["child_provider_qualification_seconds"] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - qualificationStart).count();
    }
    work["child_analysis_and_qualification_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto exported = std::chrono::steady_clock::now();
    if (!child) {
        return refuse(fs::PrefixRefusal::Certificate, "closed child qualification unavailable");
    }
    if (!child->input() || child->input()->memory().gmPolicy() != aliasPolicy) {
        return refuse(fs::PrefixRefusal::Admission, "child access-model alias policy differs from parent");
    }
    fs::CompletionStorage storage(input.instructions());
    boundary = child->prefixBoundary(storage, index);
    if (!boundary || boundary->invocation->getParentOfType<func::FuncOp>() != function || !child->usesEvents()) {
        return refuse(fs::PrefixRefusal::Certificate, "whole-loop notification boundary unavailable");
    }
    const auto physicalStart = std::chrono::steady_clock::now();
    if (requestedInvocation && boundary->invocation.getOperation() != requestedInvocation) {
        return refuse(fs::PrefixRefusal::Certificate, "child boundary belongs to a different original invocation");
    }
    auto prepared = child->preparePrefixImport(pool);
    work["child_map_verification_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - physicalStart).count();
    if (failed(prepared)) {
        return refuse(fs::PrefixRefusal::Certificate, "child assignment-preserving preparation unavailable");
    }
    childPlan = std::move(*prepared);
    std::map<uint32_t, std::set<uint32_t>> sources;
    for (auto id : description.order) {
        if (!insideChild(input.instructions()[id]->elementOp)) {
            continue;
        }
        auto target = description.sites[id].pipe;
        sources[target].insert(masks[id].begin(), masks[id].end());
        masks[id].clear();
    }
    // Explicit previous-consumption -> entry-publication -> new-publication
    // paths, for each actual reused direction. No metadata closure is taken.
    for (const auto& entry : boundary->palette) {
        sources[entry.first.first].insert(entry.first.second);
    }
    llvm::json::Array palettes;
    for (const auto& [direction, mask] : boundary->palette) {
        palettes.push_back(
            llvm::json::Object{{"source", direction.first}, {"target", direction.second}, {"id_mask", mask}});
    }
    work["child_palettes"] = std::move(palettes);
    llvm::json::Array summaries;
    for (const auto& [pipe, modes] : boundary->pendingByPipe) {
        auto& row = sources[pipe];
        entryPipes.push_back(pipe);
        entryMasks.emplace_back(row.begin(), row.end());
        description.sites.push_back({pipe, {modes.begin(), modes.end()}, {}});
        masks.push_back(entryMasks.back());
        llvm::json::Array facts;
        for (auto mode : modes) {
            facts.push_back(mode);
        }
        summaries.push_back(llvm::json::Object{{"pipe", pipe}, {"pending_modes", std::move(facts)}});
    }
    if (childRegion) {
        work["child_region"] = *childRegion;
    }
    work["child_backend"] = backend;
    work["child_families"] = childPlan->families.size();
    work["child_endpoint_sites"] = childPlan->endpoints.size();
    work["child_pending_effects"] = std::move(summaries);
    work["child_closed_notifications"] = true;
    work["child_payload_completion"] = false;
    work["child_interface_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - exported).count();
    return success();
}
} // namespace mlir::pto::frontiersynch::prefix
