// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite type selection is kept separate from invariant type interiors.
#include "PTO/Transforms/FrontierSynch/FiniteVisitRecognition.h"
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
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
    struct Pending {
        FiniteVisitAlternative type;
        std::vector<std::size_t> remaining;
    };
    std::vector<Pending> pending{{{}, {body}}};
    result.typeDescriptions = 1;
    result.nodeReferences = 1;
    while (!pending.empty()) {
        auto current = std::move(pending.back());
        pending.pop_back();
        bool forked = false;
        while (!current.remaining.empty()) {
            const auto id = current.remaining.back();
            current.remaining.pop_back();
            const auto& node = program.nodes[id];
            if (node.unsupportedContext) {
                result.error = "finite type has an unsupported original invocation context"; return false;
            }
            if (node.kind == StructureKind::Sequence) {
                for (auto child : llvm::reverse(node.children)) { current.remaining.push_back(child); }
                result.nodeReferences += node.children.size();
                continue;
            }
            if (node.kind == StructureKind::Conditional) {
                auto branch = dyn_cast_or_null<scf::IfOp>(node.anchor);
                if (!branch || node.children.size() != 2 || branch.getElseRegion().empty() ||
                    index.hasRelevantResults(branch) || index.needsValuePrerequisite(branch)) {
                    result.error =
                        "finite type selection needs exhaustive arms and mapped result/control prerequisites";
                    return false;
                }
                // One SSA condition has one value throughout this visit.
                // Repeated tests must follow the already selected polarity;
                // copying both arms would invent impossible whole-visit types.
                auto known = llvm::find_if(current.type.selection, [&](auto choice) {
                    return choice.branch.getCondition() == branch.getCondition();
                });
                if (known != current.type.selection.end()) {
                    const bool takeThen = known->takeThen;
                    for (auto child : node.children) {
                        if ((program.nodes[child].region == &branch.getThenRegion()) == takeThen) {
                            current.type.selection.push_back({branch, takeThen});
                            current.remaining.push_back(child);
                            ++result.nodeReferences;
                            break;
                        }
                    }
                    continue;
                }
                // Every pending prefix produces at least one complete type.
                // Bound the explicit product before copying either branch.
                if (pending.size() + result.alternatives.size() + 2 > maxRegionalSlotVisits) {
                    result.error = "finite whole-visit type expansion exceeds producer limit"; return false;
                }
                for (auto child : llvm::reverse(node.children)) {
                    auto next = current;
                    next.type.selection.push_back({branch, program.nodes[child].region == &branch.getThenRegion()});
                    next.remaining.push_back(child);
                    ++result.typeDescriptions;
                    result.nodeReferences += next.type.nodes.size() + next.remaining.size();
                    pending.push_back(std::move(next));
                }
                forked = true;
                break;
            }
            // Payload-free scalar scaffolding still needs a complete original
            // leaf contract. It is not an independent synchronization child.
            if (node.kind == StructureKind::ExplicitRun && !node.payloadCount) {
                if (!node.explicitResult || node.explicitResult->state != RecognitionState::Applicable) {
                    result.error = "finite type selection scaffolding lacks a complete shared leaf contract";
                    return false;
                }
                continue;
            }
            current.type.nodes.push_back(id);
            ++result.nodeReferences;
        }
        if (!forked) { result.alternatives.push_back(std::move(current.type)); }
    }
    if (result.alternatives.size() < 2) {
        result.error = "finite alternative adapter requires an exhaustive original decision tree"; return false;
    }
    return true;
}
bool qualifyInterior(const SyncInput& input, const PhaseIndex& index,
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
    result.storageProjectionRequired = !uniform;
    obligation(result.contract, "physical-map-invariance-or-joint-storage-projection",
        uniform ? ContractStatus::Established : ContractStatus::Unproved);
    return true;
}
bool projectStorage(FiniteVisitInput& types, FiniteVisitRecognition& recognition)
{
    auto& contract = recognition.contract;
    const bool projection = recognition.storageProjectionRequired ||
        llvm::any_of(types.types, [](const auto& type) {
            const auto& body = type.body;
            return body.storageSelectors || !body.symbolicStorageEffects.empty() ||
                !body.deferredAccessBoundary.empty() ||
                llvm::any_of(body.accessBoundary, [](const auto& access) { return !access.representedByCells; });
        });
    if (projection) {
        obligation(contract, "physical-map-invariance-or-joint-storage-projection", ContractStatus::Unproved);
        std::vector<RegionalAnalysis> bodies;
        for (const auto& type : types.types) { bodies.push_back(type.body); }
        auto domain = CountedLoop::get(recognition.loop);
        if (!domain) {
            contract.demands = ContractImplementation::Unavailable;
            contract.implementationError = "finite visit storage projection has no counted domain";
            return false;
        }
        auto trips = domain->trips(*types.types.front().body.expressions);
        auto proof = std::make_shared<RepeatedStorageTypesResult>(
            recognizeRepeatedStorageTypes(bodies, recognition.loop, trips));
        if (!proof->error.empty()) {
            contract.demands = ContractImplementation::Unavailable;
            contract.implementationError = "finite visit joint storage projection: " + proof->error;
            return false;
        }
        for (auto& type : types.types) {
            type.projectedStorage = FiniteVisitStorageProjection{type.body.storageBoundary, proof};
        }
        obligation(contract, "physical-map-invariance-or-joint-storage-projection", ContractStatus::Established);
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
        "complete-adjacent-prerequisite-coverage", "physical-map-invariance-or-joint-storage-projection",
        "exact-child-demand-query-selector-interfaces",
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
    if (!qualifyInterior(input, index, out)) { return fail(out.error); }
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
    std::map<std::size_t, RegionalAnalysis> regions;
    auto within = types.enclosing;
    within.push_back(out.recognition.loop);
    for (const auto& alternative : out.recognition.alternatives) {
        std::vector<RegionalAnalysis> parts;
        for (auto child : alternative.nodes) {
            auto found = out.cachedNodes.find(child);
            if (found == out.cachedNodes.end()) {
                auto analyzed = analyzeSequenceRegion(function, input, program, child, arena, index, false);
                found = out.cachedNodes.emplace(child, std::move(analyzed)).first;
            }
            if (!found->second.error.empty()) {
                contract.demands = ContractImplementation::Unavailable;
                contract.implementationError = "finite type child interface: " + found->second.error;
                return out;
            }
            auto view = regions.find(child);
            if (view == regions.end()) {
                view = regions.emplace(child, sequenceRegionalResult(found->second)).first;
            }
            parts.push_back(view->second);
        }
        // A type is a complete unmasked visit. Its original selection path is
        // retained for consumers, never reused as an inter-visit predicate.
        out.children.push_back(composeRegionalSequenceWithin(
            function, arena, std::move(parts), true, false, within));
        if (!out.children.back().error.empty()) {
            contract.demands = ContractImplementation::Unavailable;
            contract.implementationError = "finite whole-visit composition: " + out.children.back().error;
            return out;
        }
        types.types.push_back({sequenceRegionalResult(out.children.back()), {}});
    }
    obligation(contract, "exact-child-demand-query-selector-interfaces", ContractStatus::Established);
    if (!projectStorage(types, out.recognition)) { return out; }
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
