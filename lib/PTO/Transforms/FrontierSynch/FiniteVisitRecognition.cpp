// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite type selection is kept separate from invariant type interiors.
#include "PTO/Transforms/FrontierSynch/FiniteVisitRecognition.h"
#include "PhaseNormalization.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "CountedLoop.h"
#include "llvm/ADT/DenseSet.h"
namespace mlir::pto::frontiersynch {
namespace {
void obligation(ProgramContractCandidate& candidate, StringRef name, ContractStatus status)
{
    for (auto& entry : candidate.obligations) {
        if (entry.name == name) { entry.status = status; return; }
    }
    candidate.obligations.push_back({name.str(), status});
}
bool alternatives(const ProgramRecognition& program, std::size_t body, FiniteVisitRecognition& result,
                  const PhaseIndex& index)
{
    std::vector<FiniteVisitAlternative> pending{{body, {}}};
    while (!pending.empty()) {
        auto current = std::move(pending.back()); pending.pop_back();
        const auto& sequence = program.nodes[current.node];
        std::optional<std::size_t> choice;
        bool onlyChoice = true;
        for (auto id : sequence.children) {
            const auto& child = program.nodes[id];
            if (child.kind == StructureKind::Conditional && !choice) { choice = id; }
            else if (child.payloadCount || child.kind != StructureKind::ExplicitRun) { onlyChoice = false; }
        }
        if (!choice || !onlyChoice) {
            result.alternatives.push_back(std::move(current)); continue;
        }
        // Scalar choice scaffolding is still original program input. A run
        // without payload phases must satisfy the shared leaf contract too;
        // zero payload count does not discharge unknown effects or commands.
        for (auto id : sequence.children) {
            const auto& child = program.nodes[id];
            if (child.kind == StructureKind::ExplicitRun &&
                (!child.explicitResult || child.explicitResult->state != RecognitionState::Applicable)) {
                result.error = "finite type selection scaffolding lacks a complete shared leaf contract";
                return false;
            }
        }
        const auto& node = program.nodes[*choice];
        auto branch = dyn_cast_or_null<scf::IfOp>(node.anchor);
        if (!branch || node.children.size() != 2 || branch.getElseRegion().empty() ||
            index.hasRelevantResults(branch) || index.needsValuePrerequisite(branch)) {
            result.error = "finite type selection needs exhaustive arms and mapped result/control prerequisites";
            return false;
        }
        for (auto id : llvm::reverse(node.children)) {
            auto path = current.selection;
            path.push_back({branch, program.nodes[id].region == &branch.getThenRegion()});
            pending.push_back({id, std::move(path)});
        }
    }
    if (result.alternatives.size() < 2) {
        result.error = "finite alternative adapter requires an exhaustive original decision tree"; return false;
    }
    return true;
}
bool qualifyInterior(const SyncInput& input, const ProgramRecognition& program, const PhaseIndex& index,
                     FiniteVisitRecognition& result)
{
    RegionExpressions arena;
    PhaseNormalization scalar(result.loop, index, arena);
    DenseSet<Operation*> choices;
    for (const auto& type : result.alternatives) {
        for (auto arm : type.selection) { choices.insert(arm.branch.getOperation()); }
    }
    bool uniform = true, prerequisites = !index.hasRelevantCarriedState(result.loop);
    result.loop.getBody()->walk([&](Operation* op) {
        if (auto nested = dyn_cast<scf::ForOp>(op)) {
            uniform &= scalar.independent(nested.getLowerBound()) && scalar.independent(nested.getUpperBound()) &&
                       scalar.independent(nested.getStep());
            prerequisites &= !index.hasRelevantCarriedState(nested);
        } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
            if (!choices.count(op)) { uniform &= scalar.independent(branch.getCondition()); }
        }
        if (index.phasesFor(op).empty() && index.needsValuePrerequisite(op)) { prerequisites = false; }
    });
    if (!prerequisites) {
        result.error = "finite visit adapter lacks carried or phase-less adjacent prerequisite templates"; return false;
    }
    if (!uniform) {
        result.error = "finite type interior control varies with the outer visit; specialization is unavailable";
        return false;
    }
    for (const auto& effect : input.accesses().effects()) {
        if (!effect.phase || !result.loop->isProperAncestor(effect.phase->elementOp)) { continue; }
        if (effect.selection && !scalar.independent(effect.selection->selector)) { uniform = false; }
        // Complete fixed shared ranges already describe the modeled set; they
        // need no second affine geometry certificate. A supplied selection is
        // still checked above, and maps retain all actually used symbols.
        if (effect.regions.empty() && !effect.rangesMaterialized) { uniform = false; }
        for (const auto& region : effect.regions) { uniform &= scalar.periodic(region, 1); }
    }
    if (!uniform) {
        result.error = "finite type physical-map invariance or visit-owned storage projection is unavailable";
        return false;
    }
    // Every structural leaf is a whole arm invocation. Its surrounding choice
    // is deliberately excluded from the child analysis, never shared by two
    // distinct visits. Only inner invariant predicates remain in its graph.
    for (const auto& type : result.alternatives) {
        if (program.nodes[type.node].unsupportedContext) {
            result.error = "finite type arm has an unsupported original invocation context"; return false;
        }
    }
    return true;
}
} // namespace
FiniteVisitRecognition recognizeFiniteVisitLoop(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node, const PhaseIndex& index)
{
    FiniteVisitRecognition out;
    out.contract.kind = ContractClass::FiniteVisitTypes; out.contract.node = node;
    out.contract.membership = ContractStatus::Unproved;
    for (const char* name : {"exhaustive-original-type-selection", "invariant-type-interiors",
        "complete-adjacent-prerequisite-coverage", "exact-child-demand-query-selector-interfaces",
        "common-mandatory-pipes-and-persistent-refresh"}) {
        obligation(out.contract, name, ContractStatus::Unproved);
    }
    auto fail = [&](StringRef message) {
        out.error = message.str(); out.contract.implementationError = out.error; return out;
    };
    if (!function || node >= program.nodes.size() || program.nodes[node].kind != StructureKind::Loop ||
        program.nodes[node].children.size() != 1 || program.nodes[node].unsupportedContext) {
        return fail("finite visit adapter requires an original single-body loop");
    }
    out.loop = dyn_cast_or_null<scf::ForOp>(program.nodes[node].anchor);
    if (!out.loop || out.loop->getParentOfType<func::FuncOp>() != function || !CountedLoop::get(out.loop)) {
        return fail("finite visit counted occurrence domain is unavailable");
    }
    if (!alternatives(program, program.nodes[node].children.front(), out, index)) { return fail(out.error); }
    obligation(out.contract, "exhaustive-original-type-selection", ContractStatus::Established);
    if (!qualifyInterior(input, program, index, out)) { return fail(out.error); }
    obligation(out.contract, "invariant-type-interiors", ContractStatus::Established);
    obligation(out.contract, "complete-adjacent-prerequisite-coverage", ContractStatus::Established);
    return out;
}
FiniteVisitAnalysis analyzeFiniteVisitLoop(func::FuncOp function, const SyncInput& input,
    const ProgramRecognition& program, std::size_t node)
{
    FiniteVisitAnalysis out;
    auto index = std::make_shared<PhaseIndex>();
    if (failed(index->build(function, input))) {
        out.recognition.error = "finite visit phase bindings are unavailable";
        out.recognition.contract.kind = ContractClass::FiniteVisitTypes;
        out.recognition.contract.node = node;
        out.recognition.contract.membership = ContractStatus::Unproved;
        out.recognition.contract.implementationError = out.recognition.error;
        return out;
    }
    out.recognition = recognizeFiniteVisitLoop(function, input, program, node, *index);
    auto& contract = out.recognition.contract;
    if (!out.recognition.error.empty()) { return out; }
    auto arena = std::make_shared<RegionExpressions>();
    FiniteVisitInput types;
    types.enclosing.assign(program.nodes[node].loops.begin(), program.nodes[node].loops.end());
    // The selected loop advances between the two visits. Only its enclosing
    // invocations are fixed; including it would incorrectly extend scoped
    // hardware protection across the very boundary being reduced.
    for (const auto& alternative : out.recognition.alternatives) {
        out.children.push_back(analyzeSequenceRegion(function, input, program, alternative.node, arena, index, false));
        if (!out.children.back().error.empty()) {
            contract.demands = ContractImplementation::Unavailable;
            contract.implementationError = "finite type child interface: " + out.children.back().error;
            return out;
        }
        types.types.push_back({sequenceRegionalResult(out.children.back()), {}});
    }
    obligation(contract, "exact-child-demand-query-selector-interfaces", ContractStatus::Established);
    out.demands = buildFiniteVisitDemands(function, std::move(types));
    if (!out.demands->error.empty()) {
        contract.demands = ContractImplementation::Unavailable;
        contract.implementationError = "finite visit qualification/reduction: " + out.demands->error;
        return out;
    }
    obligation(contract, "common-mandatory-pipes-and-persistent-refresh", ContractStatus::Established);
    contract.membership = ContractStatus::Established;
    contract.demands = ContractImplementation::Available;
    contract.endpointRecipes = ContractImplementation::Unavailable;
    contract.allocation = ContractImplementation::NotRequested;
    contract.implementationError = "finite type demand library retained; arbitrary-word endpoint selection unavailable";
    return out;
}
} // namespace mlir::pto::frontiersynch
