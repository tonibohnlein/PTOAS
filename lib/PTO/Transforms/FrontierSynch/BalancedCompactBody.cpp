// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/BalancedCompactBody.h"
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "llvm/ADT/DenseSet.h"
#include <iterator>
#include <utility>
namespace mlir::pto::frontiersynch {
namespace {
using Slots = std::vector<BalancedCompactSlot>;
class Builder {
public:
    Builder(scf::ForOp loop, const SyncInput& input, const PhaseIndex& index)
        : input(input), index(index), shared(input.instructions().begin(), input.instructions().end())
    {
        result.loop = loop;
    }
    BalancedCompactBody result;
    bool run();
private:
    bool fail(BalancedCompactIssue issue, const char* message)
    {
        result.issue = issue; result.error = message; return false;
    }
    bool block(Block& block, SmallVector<Region*>& path, Slots& slots);
    bool branch(scf::IfOp conditional, SmallVector<Region*>& path, Slots& slots);
    bool leaf(Operation& operation, ArrayRef<Region*> path, Slots& slots);
    void prerequisites(func::FuncOp function);
    const SyncInput& input;
    const PhaseIndex& index;
    llvm::DenseSet<const CompoundInstanceElement*> shared, seen;
};
bool Builder::leaf(Operation& operation, ArrayRef<Region*> path, Slots& slots)
{
    Operation* pointer = &operation;
    const auto contract = recognizeExplicitRun(ArrayRef<Operation*>(pointer), index, input.accesses());
    for (const auto& diagnostic : contract.diagnostics) {
        if (diagnostic.issue == RecognitionIssue::AdditionalPrerequisite) {
            result.unresolvedPrerequisites.push_back(&operation);
        } else { return fail(BalancedCompactIssue::LeafContract, "balanced body has an unsupported shared leaf"); }
    }
    const auto phases = index.phasesFor(&operation);
    if (phases.empty()) { return true; }
    if (phases.size() != 1 || !shared.contains(phases.front()) || !seen.insert(phases.front()).second) {
        return fail(BalancedCompactIssue::InvalidBinding, "balanced body needs one distinct original phase per leaf");
    }
    if (!operation.getNextNode() || slots.size() == UINT32_MAX) {
        return fail(BalancedCompactIssue::InvalidBinding, "balanced payload has no legal following cut or slot ID");
    }
    BalancedCompactAlternative alternative;
    alternative.phase = phases.front();
    alternative.before = {operation.getBlock(), &operation};
    alternative.after = {operation.getBlock(), operation.getNextNode()};
    alternative.path.append(path.begin(), path.end());
    if (path.size() > UINT64_MAX - result.pathElements) {
        return fail(BalancedCompactIssue::InvalidBinding, "balanced path metadata exceeds representation");
    }
    result.pathElements += path.size();
    slots.push_back({static_cast<uint32_t>(phases.front()->kPipeValue), {std::move(alternative)}});
    return true;
}
bool Builder::branch(scf::IfOp conditional, SmallVector<Region*>& path, Slots& slots)
{
    Slots arms[2];
    for (unsigned arm = 0; arm < 2; ++arm) {
        auto& region = conditional->getRegion(arm);
        if (region.empty()) { continue; }
        if (!region.hasOneBlock()) {
            return fail(BalancedCompactIssue::UnsupportedStructure, "balanced branch requires single-block arms");
        }
        path.push_back(&region);
        const bool valid = block(region.front(), path, arms[arm]);
        path.pop_back();
        if (!valid) { return false; }
    }
    if (arms[0].size() != arms[1].size()) {
        return fail(BalancedCompactIssue::DifferentSignatures, "branch arms have different payload counts");
    }
    for (std::size_t slot = 0; slot < arms[0].size(); ++slot) {
        ++result.signatureComparisons;
        if (arms[0][slot].pipe != arms[1][slot].pipe) {
            return fail(BalancedCompactIssue::DifferentSignatures, "branch arms have different payload pipe words");
        }
        auto& alternatives = arms[0][slot].alternatives;
        auto& other = arms[1][slot].alternatives;
        alternatives.insert(alternatives.end(), std::make_move_iterator(other.begin()),
                            std::make_move_iterator(other.end()));
    }
    if (arms[0].size() > UINT32_MAX - slots.size()) {
        return fail(BalancedCompactIssue::InvalidBinding, "balanced slot count exceeds representation");
    }
    slots.insert(slots.end(), std::make_move_iterator(arms[0].begin()), std::make_move_iterator(arms[0].end()));
    return true;
}
bool Builder::block(Block& block, SmallVector<Region*>& path, Slots& slots)
{
    for (auto& operation : block) {
        ++result.syntaxNodes;
        if (auto conditional = dyn_cast<scf::IfOp>(operation)) {
            if (!index.phasesFor(&operation).empty()) {
                return fail(BalancedCompactIssue::InvalidBinding, "balanced control operation has payload phases");
            }
            if (!branch(conditional, path, slots)) { return false; }
            if (index.needsValuePrerequisite(&operation)) { result.unresolvedPrerequisites.push_back(&operation); }
        } else if (operation.getNumRegions()) {
            return fail(BalancedCompactIssue::UnsupportedStructure,
                        "balanced body supports if/else but no inner loops");
        } else if (!leaf(operation, path, slots)) { return false; }
    }
    return true;
}
void Builder::prerequisites(func::FuncOp function)
{
    result.carriedPrerequisites = index.hasRelevantCarriedState(result.loop);
    if (index.needsValuePrerequisite(result.loop)) { result.unresolvedPrerequisites.push_back(result.loop); }
    function.walk([&](Operation* operation) {
        for (const auto& edge : index.prerequisitesFor(operation)) {
            if (result.phaseSlots.contains(edge.producer) || operation == result.loop ||
                result.loop->isProperAncestor(operation)) { result.prerequisites.push_back(edge); }
        }
    });
}
bool Builder::run()
{
    if (!result.loop || !result.loop.getBody()) {
        return fail(BalancedCompactIssue::InvalidBinding, "balanced body requires an original counted loop");
    }
    auto function = result.loop->getParentOfType<func::FuncOp>();
    if (!function) { return fail(BalancedCompactIssue::InvalidBinding, "balanced loop has no original function"); }
    SmallVector<Region*> path;
    if (!block(*result.loop.getBody(), path, result.slots)) { return false; }
    for (uint32_t slot = 0; slot < result.slots.size(); ++slot) {
        for (const auto& alternative : result.slots[slot].alternatives) {
            result.phaseSlots.try_emplace(alternative.phase, slot);
        }
    }
    for (const auto* phase : input.instructions()) {
        if (result.loop->isProperAncestor(phase->elementOp) && !seen.contains(phase)) {
            return fail(BalancedCompactIssue::InvalidBinding, "balanced body omitted a shared phase");
        }
    }
    prerequisites(function);
    return true;
}
} // namespace
BalancedCompactBody recognizeBalancedCompactBody(scf::ForOp loop, const SyncInput& input, const PhaseIndex& index)
{
    Builder builder(loop, input, index);
    builder.run();
    return std::move(builder.result);
}
} // namespace mlir::pto::frontiersynch
