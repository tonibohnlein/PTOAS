// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Shared/AllocationPreparation.h"
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "llvm/Support/SaveAndRestore.h"
#include <array>
#include <map>
#include <vector>
namespace mlir::pto::frontiersynch {
class AllocationProofArchive {
public:
    using Protocol = std::vector<uint64_t>;
    using Key = std::pair<Protocol, std::vector<int64_t>>;
    struct Sources {
        std::shared_ptr<const FamilyEndpointRecipe> protocol;
        SmallVector<AllocationPreparation::Summary> owners;
    };
    std::map<Protocol, Sources> sources;
    std::map<Key, SmallVector<AllocationPreparation::CompletedCertificate>> certificates;
};
struct AllocationPreparation::State {
    Legacy legacy;
    Builder build;
    std::array<Summary, 2> answers;
    std::array<bool, 2> complete{false, false};
    bool active = false;
    bool oneShot = false;
    DictionaryAttr certificate;
    std::shared_ptr<AllocationProofArchive> archive = std::make_shared<AllocationProofArchive>();
};
AllocationPreparation& AllocationPreparation::operator=(Legacy prepare)
{
    if (!prepare) { state.reset(); return *this; }
    state = std::make_shared<State>();
    state->legacy = std::move(prepare);
    state->oneShot = true;
    return *this;
}
AllocationPreparation AllocationPreparation::candidates(Builder build)
{
    AllocationPreparation result;
    result.state = std::make_shared<State>();
    result.state->build = std::move(build);
    return result;
}
AllocationPreparation::Summary AllocationPreparation::retained(AllocationStrength strength) const
{
    if (!state) { return {}; }
    const unsigned index = strength == AllocationStrength::Independent ? 0 : 1;
    if (state->complete[index]) { return state->answers[index]; }
    return state->complete[1] ? state->answers[1] : Summary{};
}
void AllocationPreparation::preserveCertificates(PreparedLogicalPlan& plan) const
{
    if (!state || !plan.familyProtocol || !validFamilyEndpointProtocol(plan)) { return; }
    const auto key = normalizedFamilyProtocolKey(*plan.familyProtocol);
    if (key.empty()) { return; }
    const std::vector<uint64_t> protocol(key.begin(), key.end());
    auto& sources = state->archive->sources[protocol];
    sources.protocol = plan.familyProtocol;
    for (unsigned i = 0; i < state->answers.size(); ++i) {
        if (state->complete[i] && state->answers[i] && !llvm::is_contained(sources.owners, state->answers[i])) {
            sources.owners.push_back(state->answers[i]);
        }
    }
    if (!plan.allocationProofs) { plan.allocationProofs = state->archive; return; }
    if (plan.allocationProofs == state->archive) { return; }
    for (const auto& [identity, source] : state->archive->sources) {
        auto& target = plan.allocationProofs->sources[identity];
        target.protocol = source.protocol;
        for (const auto& owner : source.owners) {
            if (!llvm::is_contained(target.owners, owner)) { target.owners.push_back(owner); }
        }
    }
    for (const auto& [identity, values] : state->archive->certificates) {
        auto& target = plan.allocationProofs->certificates[identity];
        for (const auto& value : values) {
            if (llvm::none_of(target, [&](const auto& old) { return old.evidence == value.evidence; })) {
                target.push_back(value);
            }
        }
    }
}
void AllocationPreparation::retainCertificate(const PreparedLogicalPlan& plan, const Summary& owner,
    DictionaryAttr evidence, ArrayRef<int64_t> eligibleIds) const
{
    if (!owner || !evidence || !plan.familyProtocol || !validFamilyEndpointProtocol(plan)) { return; }
    const auto key = normalizedFamilyProtocolKey(*plan.familyProtocol);
    if (key.empty()) { return; }
    const std::vector<uint64_t> protocol(key.begin(), key.end());
    auto archive = plan.allocationProofs ? plan.allocationProofs : state ? state->archive : nullptr;
    if (!archive) { return; }
    const bool retained = state && ((state->complete[0] && state->answers[0] == owner) ||
        (state->complete[1] && state->answers[1] == owner));
    auto found = archive->sources.find(protocol);
    if (!retained && (found == archive->sources.end() || !llvm::is_contained(found->second.owners, owner))) { return; }
    auto& sources = archive->sources[protocol];
    sources.protocol = plan.familyProtocol;
    if (!llvm::is_contained(sources.owners, owner)) { sources.owners.push_back(owner); }
    auto& values = archive->certificates[{protocol, std::vector<int64_t>(eligibleIds.begin(), eligibleIds.end())}];
    if (llvm::none_of(values, [&](const auto& value) { return value.evidence == evidence; })) {
        values.push_back({owner, evidence});
    }
}
SmallVector<AllocationPreparation::CompletedCertificate> AllocationPreparation::retainedCertificates(
    const PreparedLogicalPlan& plan, ArrayRef<int64_t> eligibleIds) const
{
    if (!plan.familyProtocol || !validFamilyEndpointProtocol(plan)) { return {}; }
    const auto key = normalizedFamilyProtocolKey(*plan.familyProtocol);
    if (key.empty()) { return {}; }
    const std::vector<uint64_t> protocol(key.begin(), key.end());
    SmallVector<CompletedCertificate> result;
    for (const auto& archive : {state ? state->archive : nullptr, plan.allocationProofs}) {
        if (!archive) { continue; }
        // Unrestricted proofs remain reusable, but capacity-specific exports
        // never enter the unrestricted cache or a different ordered ID pool.
        const auto requested = std::vector<int64_t>(eligibleIds.begin(), eligibleIds.end());
        for (const auto& pool : {std::vector<int64_t>{}, requested}) {
            const auto found = archive->certificates.find({protocol, pool});
            if (found == archive->certificates.end()) { continue; }
            for (const auto& value : found->second) {
                if (llvm::none_of(result, [&](const auto& old) { return old.evidence == value.evidence; })) {
                    result.push_back(value);
                }
            }
        }
    }
    return result;
}
AllocationPreparation::Summary AllocationPreparation::request(
    PreparedLogicalPlan& plan, AllocationStrength strength)
{
    if (!state) { return plan.regionalAllocation; }
    auto owner = state; // A compatibility callback can replace the plan's owner.
    const unsigned index = strength == AllocationStrength::Independent ? 0 : 1;
    if (owner->complete[index]) { return owner->answers[index]; }
    if (owner->complete[1]) { return owner->answers[1]; }
    if (owner->active) { return {}; }
    llvm::SaveAndRestore<Summary> summary(plan.regionalAllocation);
    llvm::SaveAndRestore<DictionaryAttr> certificate(plan.allocationCertificate);
    owner->active = true;
    if (owner->legacy) {
        auto prepare = std::move(owner->legacy);
        prepare(plan);
        owner->answers[0] = owner->answers[1] = plan.regionalAllocation;
        owner->certificate = plan.allocationCertificate;
        owner->complete = {true, true};
    } else {
        owner->answers[index] = owner->build ? owner->build(plan, strength) : Summary{};
        owner->complete[index] = true;
        if (index == 1) { owner->build = {}; }
    }
    owner->active = false;
    return owner->answers[index];
}
void AllocationPreparation::operator()(PreparedLogicalPlan& plan)
{
    auto owner = state;
    auto selected = request(plan, AllocationStrength::Shared);
    if (owner && owner->oneShot && owner->complete[1]) { plan.allocationCertificate = owner->certificate; }
    if (selected) { plan.regionalAllocation = std::move(selected); }
}
AllocationPreparation::operator bool() const
{
    return state && (state->legacy || state->build);
}
std::shared_ptr<RegionalAllocationSummary> requestAllocationSummary(
    PreparedLogicalPlan& plan, AllocationStrength strength, AllocationProfile* profile)
{
    if (plan.privateState || plan.completionOwner || failed(normalizeLogicalFamilies(plan))) {
        return {};
    }
    llvm::SaveAndRestore<AllocationProfile*> profiling(plan.allocationProfile, profile);
    return plan.allocationPreparation.request(plan, strength);
}
} // namespace mlir::pto::frontiersynch
