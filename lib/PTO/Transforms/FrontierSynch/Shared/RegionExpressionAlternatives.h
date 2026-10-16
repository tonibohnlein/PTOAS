// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_REGIONEXPRESSIONALTERNATIVES_H
#define PTO_FRONTIER_REGIONEXPRESSIONALTERNATIVES_H
#include "RegionTotality.h"
#include "RegionAlternativeWork.h"
#include "llvm/ADT/ScopeExit.h"
namespace mlir::pto::frontiersynch::detail {
inline void alternateWork(uint64_t* work, uint64_t amount = 1)
{
    if (work) { *work = amount > UINT64_MAX - *work ? UINT64_MAX : *work + amount; }
}
inline RegionAlternativeWork& RegionConstructionAccess::alternativeConstructionWork(RegionExpressions& e)
{
    if (!e.ownAlternativeWork) { e.ownAlternativeWork = std::make_shared<RegionAlternativeWork>(); }
    e.alternativeWorkOwners.try_emplace(e.ownAlternativeWork.get(), e.ownAlternativeWork);
    return *e.ownAlternativeWork;
}
inline std::vector<std::shared_ptr<const RegionAlternativeWork>>
RegionConstructionAccess::alternativeConstructionOwners(const RegionExpressions& e)
{
    std::vector<std::shared_ptr<const RegionAlternativeWork>> result;
    for (const auto& item : e.alternativeWorkOwners) { result.push_back(item.second); }
    return result;
}
inline void RegionConstructionAccess::inheritAlternativeConstruction(
    RegionExpressions& e, const RegionExpressions& source)
{
    for (const auto& item : source.alternativeWorkOwners) {
        e.alternativeWorkOwners.try_emplace(item.first, item.second);
    }
}
inline bool RegionConstructionAccess::retainAlternative(
    RegionExpressions& e, RegionExpressions::Id original, RegionExpressions::Id alternate, uint64_t* work)
{
    auto& counts = alternativeConstructionWork(e);
    RegionAlternativeWork::add(counts.registrationAttempts);
    uint64_t visited = 0;
    auto account = llvm::make_scope_exit([&] {
        RegionAlternativeWork::add(counts.registrationVisits, visited);
        alternateWork(work, visited);
    });
    if (!e.valid(original) || !e.valid(alternate) || !e.isBoolean(original) || !e.isBoolean(alternate) ||
        original == alternate || e.equivalentAlternatives.count(original) || !e.constructionMessage.empty()) {
        return false;
    }
    uint64_t totalNodes = 0, oldNodes = 0, newNodes = 0;
    const RegionExpressions::Id roots[]{original, alternate};
    const bool total = totalScalarDag(e, roots, totalNodes);
    alternateWork(&visited, totalNodes);
    if (!total) { return false; }
    auto before = e.referencedInputs(ArrayRef<RegionExpressions::Id>(original), &oldNodes);
    auto after = e.referencedInputs(ArrayRef<RegionExpressions::Id>(alternate), &newNodes);
    alternateWork(&visited, oldNodes); alternateWork(&visited, newNodes);
    if (newNodes >= oldNodes) { return false; }
    llvm::DenseSet<RegionExpressions::Id> permitted;
    for (auto input : before) { permitted.insert(input.first); }
    for (auto input : after) { if (!permitted.count(input.first)) { return false; } }
    // Append the journal before publishing the map entry. If either container
    // allocation throws, erase/pop-only cleanup preserves exact ownership.
    e.alternativeInsertions.push_back(original);
    auto rollback = llvm::make_scope_exit([&] { e.alternativeInsertions.pop_back(); });
    e.equivalentAlternatives.emplace(original, alternate);
    rollback.release();
    RegionAlternativeWork::add(counts.registrations);
    return true;
}
inline std::optional<RegionExpressions::Id> RegionConstructionAccess::alternative(
    RegionExpressions& e, RegionExpressions::Id original, uint64_t* work)
{
    if (e.equivalentAlternatives.empty() || !e.valid(original)) { return {}; }
    auto& counts = alternativeConstructionWork(e);
    RegionAlternativeWork::add(counts.finalLookups);
    SmallVector<std::pair<RegionExpressions::Id, RegionExpressions::Id>> bindings;
    const auto visited = e.visitReferenced(ArrayRef<RegionExpressions::Id>(original), [&](auto id) {
        auto found = e.equivalentAlternatives.find(id);
        if (found != e.equivalentAlternatives.end()) { bindings.push_back(*found); }
    });
    alternateWork(work, visited);
    RegionAlternativeWork::add(counts.finalLookupVisits, visited);
    RegionAlternativeWork::add(counts.finalReached, bindings.size());
    if (bindings.empty()) { return {}; }
    const auto previous = e.transportingAlternatives;
    e.transportingAlternatives = true;
    auto restore = llvm::make_scope_exit([&] { e.transportingAlternatives = previous; });
    RegionExpressions::Transaction scratch(e);
    RegionExpressions::Substitution replacement(bindings);
    auto result = e.substitute(original, replacement, work);
    if (!e.valid(result) || !e.constructionMessage.empty() || result == original) { return {}; }
    scratch.commit();
    RegionAlternativeWork::add(counts.finalRebuilt);
    return result;
}
// The numerical producer's original mixed substitution remains authoritative.
// A composite binding is redundant only when the same atomic Input replacement
// reconstructs its exact target. Also check the whole bound root, so overlapping
// replacements and existing memo cutoffs cannot silently change the proof.
inline void RegionConstructionAccess::transportQualifiedAlternative(RegionExpressions& e,
    RegionExpressions::Id original, RegionExpressions::Id bound,
    const RegionExpressions::Substitution& substitution)
{
    const auto& bindings = substitution.bindings;
    if (substitution.owner != &e || e.transportingAlternatives || !e.valid(original) || !e.valid(bound) ||
        !e.constructionMessage.empty()) { return; }
    auto found = e.equivalentAlternatives.find(original);
    if (found == e.equivalentAlternatives.end()) { return; }
    const auto alternate = found->second;
    auto& counts = alternativeConstructionWork(e);
    auto outcome = [&](const char* reason) { RegionAlternativeWork::add(counts.outcomes[reason]); };
    outcome("binding-source-registered");
    if (e.equivalentAlternatives.count(bound)) { outcome("binding-already-retained"); return; }
    uint64_t visits = 0;
    const auto before = e.size();
    RegionExpressions::Transaction scratch(e);
    auto account = llvm::make_scope_exit([&] {
        RegionAlternativeWork::add(counts.bindingVisits, visits);
        RegionAlternativeWork::add(counts.bindingConstructed, e.size() - before);
    });
    const auto previous = e.transportingAlternatives;
    e.transportingAlternatives = true;
    auto restore = llvm::make_scope_exit([&] { e.transportingAlternatives = previous; });
    SmallVector<std::pair<RegionExpressions::Id, RegionExpressions::Id>> inputs;
    for (auto binding : bindings) {
        alternateWork(&visits);
        if (!e.valid(binding.first) || !e.valid(binding.second) ||
            e.isBoolean(binding.first) != e.isBoolean(binding.second)) {
            outcome("binding-invalid"); return;
        }
        if (e.nodes[binding.first].kind == RegionExpressions::Kind::Input) { inputs.push_back(binding); }
    }
    RegionExpressions::Substitution inputOnly(inputs);
    for (auto binding : bindings) {
        alternateWork(&visits);
        if (e.nodes[binding.first].kind == RegionExpressions::Kind::Input) { continue; }
        if (e.substitute(binding.first, inputOnly, &visits) != binding.second ||
            !e.constructionMessage.empty()) { outcome("binding-nonredundant"); return; }
        outcome("binding-redundant-composite");
    }
    if (e.substitute(original, inputOnly, &visits) != bound || !e.constructionMessage.empty()) {
        outcome("binding-root-mismatch"); return;
    }
    const auto rewritten = e.substitute(alternate, inputOnly, &visits);
    if (!retainAlternative(e, bound, rewritten, &visits)) { outcome("binding-unqualified-alternate"); return; }
    outcome("binding-retained");
    scratch.commit();
}
inline void RegionConstructionAccess::transportAlternatives(RegionExpressions& e,
    const RegionExpressions::Substitution& substitution, ArrayRef<RegionExpressions::Id> visited, uint64_t* work)
{
    if (e.transportingAlternatives || e.equivalentAlternatives.empty()) { return; }
    auto& counts = alternativeConstructionWork(e);
    const bool mixed = llvm::any_of(substitution.bindings, [&](auto binding) {
        alternateWork(work);
        RegionAlternativeWork::add(counts.bindingVisits);
        return e.nodes[binding.first].kind != RegionExpressions::Kind::Input;
    });
    if (mixed) {
        RegionAlternativeWork::add(counts.outcomes["transport-mixed-bindings"]);
        for (auto id : visited) {
            alternateWork(work);
            RegionAlternativeWork::add(counts.bindingVisits);
            if (e.equivalentAlternatives.count(id)) {
                RegionAlternativeWork::add(counts.outcomes["transport-mixed-reached-registration"]);
            }
        }
        return;
    }
    // Internal-node replacement is not congruence of whole equivalent queries.
    // Input-only simultaneous replacement preserves the certified relation.
    const auto previous = e.transportingAlternatives;
    e.transportingAlternatives = true;
    auto restore = llvm::make_scope_exit([&] { e.transportingAlternatives = previous; });
    for (auto id : visited) {
        alternateWork(work);
        auto found = e.equivalentAlternatives.find(id);
        if (found == e.equivalentAlternatives.end()) { continue; }
        const auto original = substitution.memo.lookup(id), alternate = found->second;
        if (e.equivalentAlternatives.count(original)) { continue; }
        RegionExpressions::Transaction scratch(e);
        RegionExpressions::Substitution fresh(substitution.bindings);
        const auto rewritten = e.substitute(alternate, fresh, work);
        if (retainAlternative(e, original, rewritten, work)) { scratch.commit(); }
    }
}
inline void RegionConstructionAccess::importAlternatives(RegionExpressions& e, const RegionExpressions& source,
    ArrayRef<RegionExpressions::Id> visited,
    const llvm::DenseMap<RegionExpressions::Id, RegionExpressions::Id>& imported, uint64_t* work)
{
    if (e.transportingAlternatives || source.equivalentAlternatives.empty()) { return; }
    const auto previous = e.transportingAlternatives;
    e.transportingAlternatives = true;
    auto restore = llvm::make_scope_exit([&] { e.transportingAlternatives = previous; });
    for (auto id : visited) {
        alternateWork(work);
        auto found = source.equivalentAlternatives.find(id);
        if (found == source.equivalentAlternatives.end()) { continue; }
        const auto original = imported.lookup(id);
        if (e.equivalentAlternatives.count(original)) { continue; }
        RegionExpressions::Transaction scratch(e);
        const auto alternate = e.import(source, ArrayRef<RegionExpressions::Id>(found->second), work);
        if (succeeded(alternate) && retainAlternative(e, original, alternate->front(), work)) { scratch.commit(); }
    }
}
} // namespace mlir::pto::frontiersynch::detail
#endif
