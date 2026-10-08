// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RequirementProvenance.h"
#include <algorithm>
#include <set>
#include <utility>
namespace mlir::pto::frontiersynch {
namespace {
bool validGuard(const OrderContext& context, RegionExpressions::Id guard)
{
    return context && context->expressions() && context->expressions()->constructionError().empty() &&
           guard < context->expressions()->size() && context->expressions()->isBoolean(guard);
}
bool validScope(RequirementScope scope)
{
    return scope == RequirementScope::Internal || scope == RequirementScope::Boundary ||
           scope == RequirementScope::FixedPrerequisite;
}
bool replaceableGroup(const RequirementProvenance& snapshot, RequirementGroupId group)
{
    auto groups = snapshot.groups();
    auto found = std::find_if(groups.begin(), groups.end(),
        [group](const auto& entry) { return entry.id == group; });
    return found != groups.end() && found->scope != RequirementScope::FixedPrerequisite;
}
bool validMemberships(const OrderContext& context, llvm::ArrayRef<RequirementGroup> groups,
                      llvm::ArrayRef<RequirementMembership> memberships)
{
    if (!context || !context->expressions() || !context->expressions()->constructionError().empty()) { return false; }
    std::set<RequirementGroupId> ids;
    for (const auto& group : groups) {
        if (!validScope(group.scope) || !ids.insert(group.id).second) { return false; }
    }
    for (const auto& membership : memberships) {
        if (!ids.count(membership.group) || !validGuard(context, membership.guard)) { return false; }
    }
    return true;
}
} // namespace
RequirementSnapshot createRequirementProvenance(OrderContext context,
    std::vector<RequirementGroup> groups, std::vector<RequirementMembership> memberships, std::string& error)
{
    error.clear();
    if (!validMemberships(context, groups, memberships)) {
        error = "requirement provenance needs a valid context, unique groups and Boolean membership guards";
        return {};
    }
    // Private construction prevents snapshots whose origin and selected members
    // disagree. Public access is const; revisions share the immutable origin.
    auto result = std::shared_ptr<RequirementProvenance>(new RequirementProvenance());
    result->owner = std::move(context);
    result->groupDefinitions = std::make_shared<const std::vector<RequirementGroup>>(std::move(groups));
    result->original = std::make_shared<const std::vector<RequirementMembership>>(std::move(memberships));
    result->selected = *result->original;
    return result;
}
RequirementReplacement bindTrustedRequirementReplacement(RequirementSnapshot snapshot,
    RequirementGroupId group, RegionExpressions::Id domain,
    std::vector<RequirementExactRecord> replacement, std::string& error)
{
    error.clear();
    if (!snapshot || !replaceableGroup(*snapshot, group) || !validGuard(snapshot->context(), domain)) {
        error = "replacement needs an existing nonfixed group and a Boolean domain in its snapshot";
        return {};
    }
    for (const auto& record : replacement) {
        if (!validGuard(snapshot->context(), record.guard)) {
            error = "replacement record guard does not belong to the snapshot's Boolean frame";
            return {};
        }
    }
    auto evidence = std::shared_ptr<RequirementReplacementEvidence>(new RequirementReplacementEvidence());
    evidence->snapshot = snapshot;
    evidence->context = snapshot->context();
    evidence->group = group;
    evidence->domain = domain;
    evidence->replacement = std::move(replacement);
    return evidence;
}
RequirementSnapshot captureStorageRequirementProvenance(OrderContext context, uint64_t producer,
    const StorageScanResult& scan, llvm::ArrayRef<RequirementGroupId> atomGroups,
    RequirementGroupId prerequisiteGroup, std::string& error)
{
    error.clear();
    if (!context || !context->expressions() || !context->expressions()->constructionError().empty() ||
        !scan.error.empty()) {
        error = "storage provenance requires a valid context and successful original lifetime scan";
        return {};
    }
    std::set<RequirementGroupId> ids(atomGroups.begin(), atomGroups.end());
    if (ids.count(prerequisiteGroup)) {
        error = "fixed prerequisites require a group distinct from storage groups";
        return {};
    }
    std::vector<RequirementGroup> groups;
    for (auto id : ids) { groups.push_back({id, RequirementScope::Internal}); }
    groups.push_back({prerequisiteGroup, RequirementScope::FixedPrerequisite});
    std::vector<uint8_t> covered(scan.generators.size(), 0);
    std::vector<RequirementMembership> memberships;
    const auto enabled = context->expressions()->boolean(true);
    for (const auto& witness : scan.witnesses) {
        const bool supplied = witness.hazard == StorageHazard::Supplied;
        const bool storage = witness.hazard == StorageHazard::RAW || witness.hazard == StorageHazard::WAR ||
                             witness.hazard == StorageHazard::WAW;
        if (witness.generator >= covered.size() || (!supplied && (!storage || witness.atom >= atomGroups.size()))) {
            error = "original storage witness has an invalid generator, hazard or atom";
            return {};
        }
        covered[witness.generator] = 1;
        const auto group = supplied ? prerequisiteGroup : atomGroups[witness.atom];
        memberships.push_back({group, {producer, witness.generator}, enabled});
    }
    if (std::find(covered.begin(), covered.end(), uint8_t(0)) != covered.end()) {
        error = "original lifetime generator has no requirement ownership witness";
        return {};
    }
    return createRequirementProvenance(std::move(context), std::move(groups), std::move(memberships), error);
}
RequirementSnapshot applyRequirementReplacement(
    RequirementSnapshot snapshot, RequirementReplacement evidence, std::string& error)
{
    error.clear();
    if (!snapshot || !evidence || evidence->snapshot != snapshot || evidence->context != snapshot->context() ||
        !replaceableGroup(*snapshot, evidence->group) || !validGuard(snapshot->context(), evidence->domain)) {
        error = "replacement evidence is not bound to this immutable snapshot, context and nonfixed group";
        return {};
    }
    for (const auto& record : evidence->replacement) {
        if (!validGuard(snapshot->context(), record.guard)) {
            error = "replacement expression frame changed after evidence was bound";
            return {};
        }
    }
    if (evidence->replacement.size() > snapshot->selected.max_size() - snapshot->selected.size() ||
        evidence->replacement.size() > snapshot->definite.max_size() - snapshot->definite.size()) {
        error = "replacement membership count exceeds representation";
        return {};
    }
    auto& expressions = *snapshot->context()->expressions();
    RegionExpressions::Transaction transaction(expressions);
    auto result = std::shared_ptr<RequirementProvenance>(new RequirementProvenance(*snapshot));
    const auto outside = expressions.lnot(evidence->domain);
    for (auto& membership : result->selected) {
        if (membership.group == evidence->group) {
            membership.guard = expressions.land(membership.guard, outside);
        }
    }
    for (const auto& record : evidence->replacement) {
        const auto guard = expressions.land(record.guard, evidence->domain);
        RequirementMembership member{evidence->group, record.record, guard};
        result->selected.push_back(member);
        result->definite.push_back(member);
    }
    if (!expressions.constructionError().empty()) {
        error = expressions.constructionError();
        return {};
    }
    transaction.commit();
    return result;
}
} // namespace mlir::pto::frontiersynch
