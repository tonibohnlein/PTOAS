// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve branch predicates and arm identity in a shared guard DAG. A guarded
// loop requires predicates defined before that loop, so participation repeats.
#include "RecognitionInternal.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <tuple>

namespace mlir::pto::frontiersynch {
namespace {
void collect(ArrayRef<Operation*> roots, Operation* entry, const PhaseIndex& index, GuardedRecognition& output,
             bool requireInvariant, const DenseMap<Value, bool>& choices = DenseMap<Value, bool>(),
             ArrayRef<Value> sliceGuards = {})
{
    // An explicit stack avoids host recursion on deeply nested if/else trees.
    struct Work {
        Operation* operation = nullptr;
        std::optional<std::size_t> guard;
    };
    SmallVector<Work> stack;
    auto append = [&](Region& region, std::optional<std::size_t> guard) {
        if (region.empty()) {
            return;
        }
        if (!region.hasOneBlock()) {
            output.result.note(RecognitionIssue::UnsupportedControl, region.getParentOp(), true);
            return;
        }
        for (Operation& op : llvm::reverse(region.front())) {
            stack.push_back({&op, guard});
        }
    };
    for (auto* root : llvm::reverse(roots)) { stack.push_back({root, std::nullopt}); }
    while (!stack.empty()) {
        const auto work = stack.pop_back_val();
        Operation& op = *work.operation;
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
            if (index.needsValuePrerequisite(&op)) {
                output.result.note(RecognitionIssue::AdditionalPrerequisite, &op);
            }
            auto choice = choices.find(branch.getCondition());
            if (choice != choices.end()) {
                append(choice->second ? branch.getThenRegion() : branch.getElseRegion(), work.guard);
                continue;
            }
            const bool available = entry && index.valueAvailable(branch.getCondition(), entry, Boundary::Before);
            output.entryGuardsAvailable &= available;
            if (requireInvariant && !available && !llvm::is_contained(sliceGuards, branch.getCondition()) &&
                !detail::entryExpression(branch.getCondition(), entry, index, output.entryExpressions)) {
                output.result.note(RecognitionIssue::GuardInvariance, &op);
            }
            const auto thenGuard = output.guards.size();
            output.guards.push_back({work.guard, branch.getCondition(), true});
            const auto elseGuard = output.guards.size();
            output.guards.push_back({work.guard, branch.getCondition(), false});
            append(branch.getElseRegion(), elseGuard);
            append(branch.getThenRegion(), thenGuard);
            continue;
        }
        if (op.getNumRegions()) {
            output.result.note(RecognitionIssue::UnsupportedControl, &op, true);
            continue;
        }
        detail::inspectLeaf(op, index, output.result, /*allowEnvelopes=*/!requireInvariant);
        for (const auto* phase : index.phasesFor(&op)) {
            output.phases.push_back({phase, work.guard});
        }
    }
}
// Coverage concerns one identical physical map, not merely its slot orbit.
// A guard node is covered when some matching writer always executes under it.
// Complementary covered children establish their parent's coverage; nested
// parents occur earlier in the DAG, so one reverse pass suffices.
bool coversEveryVisit(const GuardedRecognition& skeleton, ArrayRef<std::size_t> writers)
{
    std::vector<uint8_t> covered(skeleton.guards.size());
    for (auto writer : writers) {
        if (writer >= skeleton.phases.size()) { return false; }
        auto guard = skeleton.phases[writer].guard;
        if (!guard) { return true; }
        if (*guard >= covered.size()) { return false; }
        covered[*guard] = 1;
    }
    std::vector<DenseMap<Value, uint8_t>> branches(skeleton.guards.size() + 1);
    for (std::size_t i = skeleton.guards.size(); i > 0; --i) {
        const auto& guard = skeleton.guards[i - 1];
        if (!guard.condition || (guard.parent && *guard.parent >= i - 1)) { return false; }
        if (!covered[i - 1]) { continue; }
        auto parent = guard.parent.value_or(skeleton.guards.size());
        auto& arms = branches[parent][guard.condition];
        arms |= guard.takeThen ? 1 : 2;
        if (arms != 3) { continue; }
        if (!guard.parent) { return true; }
        covered[*guard.parent] = 1;
    }
    return false;
}
} // namespace

GuardedRecognition recognizeFiniteGuarded(Region& region, const PhaseIndex& index,
                                          const SyncStorageEffects& effects)
{
    GuardedRecognition output;
    SmallVector<Operation*> roots;
    if (region.empty()) { return output; }
    if (!region.hasOneBlock()) {
        output.result.note(RecognitionIssue::UnsupportedControl, region.getParentOp(), true);
        return output;
    }
    for (Operation& op : region.front()) { roots.push_back(&op); }
    return recognizeFiniteGuarded(roots, index, effects);
}
GuardedRecognition recognizeFiniteGuarded(ArrayRef<Operation*> roots, const PhaseIndex& index,
                                          const SyncStorageEffects& effects)
{
    GuardedRecognition output;
    Operation* entry = roots.empty() ? nullptr : roots.front();
    Operation* previous = nullptr;
    for (auto* root : roots) {
        if (!root || !root->getBlock() || (previous && previous->getNextNode() != root)) {
            output.result.note(RecognitionIssue::UnsupportedControl, root, true);
            return output;
        }
        previous = root;
    }
    collect(roots, entry, index, output, false);
    return output;
}

GuardedRecognition detail::recognizeRotatingSlice(scf::ForOp loop, const PhaseIndex& index,
    const SyncInput& input, const DenseMap<Value, bool>& choices, ArrayRef<Value> sliceGuards)
{
    const auto& effects = input.accesses();
    GuardedRecognition output;
    if (!detail::checkRotatingDomain(loop, index, output.result)) {
        return output;
    }
    if (index.needsValuePrerequisite(loop)) {
        output.result.note(RecognitionIssue::AdditionalPrerequisite, loop);
    }
    SmallVector<Operation*> roots;
    for (Operation& op : loop.getBody()->getOperations()) { roots.push_back(&op); }
    collect(roots, loop, index, output, true, choices, sliceGuards);
    SmallVector<const CompoundInstanceElement*> phases;
    DenseMap<const CompoundInstanceElement*, std::optional<std::size_t>> guards;
    for (const auto& item : output.phases) {
        phases.push_back(item.phase);
        guards[item.phase] = item.guard;
    }
    detail::inspectRotatingPhases(loop, phases, input, effects, output.result, index, true);
    for (auto& access : output.result.accesses) {
        for (auto parameter : access.parameters) {
            if (!detail::entryExpression(parameter, loop, index, output.entryExpressions)) {
                output.result.note(RecognitionIssue::GuardInvariance, loop);
            }
        }
        access.guard = guards.lookup(effects.effects()[access.effect].phase);
    }
    return output;
}
GuardedRecognition recognizeGuardedRotating(scf::ForOp loop, const PhaseIndex& index,
                                            const SyncInput& input, const SyncStorageEffects& effects)
{
    return detail::recognizeRotatingSlice(loop, index, input, DenseMap<Value, bool>());
}
BoundedLifetimeRecognition recognizeBoundedLifetime(scf::ForOp loop, const PhaseIndex& index, const SyncInput& input)
{
    BoundedLifetimeRecognition out;
    auto& skeleton = out.skeleton;
    if (!detail::checkRotatingDomain(loop, index, skeleton.result)) {
        return out;
    }
    SmallVector<Operation*> roots;
    for (Operation& operation : *loop.getBody()) {
        roots.push_back(&operation);
    }
    collect(roots, loop, index, skeleton, false);
    if (skeleton.result.state != RecognitionState::Applicable) {
        return out;
    }
    SmallVector<const CompoundInstanceElement*> phases;
    DenseMap<const CompoundInstanceElement*, uint32_t> positions;
    std::vector<PeriodicPayload> payloads;
    for (const auto& item : skeleton.phases) {
        positions[item.phase] = phases.size();
        phases.push_back(item.phase);
        payloads.push_back({static_cast<uint32_t>(item.phase->kPipeValue)});
    }
    detail::inspectRotatingPhases(loop, phases, input, input.accesses(), skeleton.result, index);
    if (skeleton.result.state != RecognitionState::Applicable) {
        return out;
    }
    if (input.accesses().hasUniformRelationships(phases)) {
        out.refresh.error = "bounded refresh needs a uniform-relationship adapter";
        skeleton.result.note(RecognitionIssue::RefreshBound, loop);
        return out;
    }
    auto prerequisites = index.mapPrerequisites(phases);
    if (!prerequisites.error.empty()) {
        out.refresh.error = prerequisites.error;
        skeleton.result.note(RecognitionIssue::AdditionalPrerequisite, loop);
        return out;
    }
    DenseMap<Value, uint32_t> families;
    std::map<std::tuple<uint32_t, uint64_t, uint64_t>, uint32_t> atoms;
    std::vector<RotatingFragment> fragments;
    for (const auto& access : skeleton.result.accesses) {
        if (!access.atom || access.parameterOffset) {
            out.refresh.error = "refresh orbit producer requires numerical normalized fragments";
            skeleton.result.note(RecognitionIssue::RefreshBound, loop);
            return out;
        }
        auto found = positions.find(input.accesses().effects()[access.effect].phase);
        if (found == positions.end()) {
            out.refresh.error = "refresh access has no payload";
            skeleton.result.note(RecognitionIssue::RefreshBound, loop);
            return out;
        }
        auto family = families.try_emplace(access.family, families.size()).first->second;
        auto atom =
            atoms.emplace(std::make_tuple(family, access.atom->first, access.atom->second), atoms.size()).first->second;
        fragments.push_back(
            {found->second, family, atom, access.slots, access.stride, access.offset, access.reads, access.writes, 0});
    }
    using Map = std::tuple<uint32_t, uint32_t, uint64_t, uint64_t, uint64_t>;
    std::map<Map, SmallVector<std::size_t>> writerMaps;
    for (std::size_t i = 0; i < fragments.size(); ++i) {
        const auto& fragment = fragments[i];
        if (!fragment.write) { continue; }
        if (!fragment.slots) {
            out.refresh.error = "collective refresh requires a nonzero slot modulus";
            skeleton.result.note(RecognitionIssue::RefreshBound, loop);
            return out;
        }
        writerMaps[{fragment.family, fragment.atom, fragment.slots,
                    fragment.stride % fragment.slots, fragment.offset % fragment.slots}].push_back(i);
    }
    std::vector<uint8_t> coveredWriters(fragments.size());
    for (const auto& [map, members] : writerMaps) {
        SmallVector<std::size_t> writers;
        for (auto member : members) { writers.push_back(fragments[member].payload); }
        if (!coversEveryVisit(skeleton, writers)) { continue; }
        for (auto member : members) { coveredWriters[member] = 1; }
    }
    out.refresh = certifyRotatingRefreshCoverage(payloads, fragments, coveredWriters);
    if (!out.refresh.error.empty()) { skeleton.result.note(RecognitionIssue::RefreshBound, loop); }
    return out;
}
} // namespace mlir::pto::frontiersynch
