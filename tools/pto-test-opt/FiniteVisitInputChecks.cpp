// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/FiniteVisitRecognition.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
int runFiniteVisitInputChecks(func::FuncOp function)
{
    fs::FrontierAnalysis analysis(function);
    if (failed(analysis.initialize()) || failed(analysis.analyzeFiniteVisitCandidates())) {
        llvm::errs() << "finite visit original-input analysis failed\n"; return 1;
    }
    const auto& results = analysis.finiteVisitDemands();
    if (function->hasAttr("test.type_expansion_limit")) {
        for (const auto& candidate : analysis.result()->finiteVisitContracts) {
            if (candidate.implementationError == "finite whole-visit type expansion exceeds producer limit" &&
                candidate.membership == fs::ContractStatus::Unproved &&
                candidate.demands != fs::ContractImplementation::Available && results.empty()) {
                llvm::outs() << "finite visit type expansion: explicit producer limit retained\n";
                return 0;
            }
        }
        llvm::errs() << "finite visit expansion limit lost its producer diagnostic\n"; return 1;
    }
    if (results.size() != 1) { llvm::errs() << "expected one original finite visit loop\n"; return 1; }
    const auto& result = *results.begin()->second;
    const auto& contract = result.recognition.contract;
    if (!result.demands || !result.demands->error.empty()) {
        llvm::errs() << "finite visit demands: " <<
            (result.demands ? result.demands->error : result.recognition.error) << "\n"; return 1;
    }
    const bool wholeVisit = function->hasAttr("test.whole_visit");
    const std::size_t count = wholeVisit ? 4 : 2;
    if (result.children.size() != count || result.demands->boundaries.size() != count * count ||
        result.demands->cost.pairReductions != count * count ||
        contract.membership != fs::ContractStatus::Established ||
        contract.demands != fs::ContractImplementation::Available ||
        contract.endpointRecipes != fs::ContractImplementation::Unavailable ||
        contract.allocation != fs::ContractImplementation::NotRequested) {
        llvm::errs() << "finite visit stage contract lost\n"; return 1;
    }
    std::set<std::size_t> nodes;
    for (const auto& alternative : result.recognition.alternatives) {
        nodes.insert(alternative.nodes.begin(), alternative.nodes.end());
    }
    if (nodes.size() != result.cachedNodes.size() ||
        result.recognition.typeDescriptions < count || result.recognition.nodeReferences < nodes.size()) {
        llvm::errs() << "finite visit node reuse or expansion cost lost\n"; return 1;
    }
    if (wholeVisit) {
        if (nodes.size() != 7) { llvm::errs() << "whole-visit shared nodes were not reused\n"; return 1; }
        for (std::size_t type = 0; type < count; ++type) {
            const auto& alternative = result.recognition.alternatives[type];
            const auto& body = result.demands->original->types[type].body;
            std::vector<StringRef> expected{
                "prefix", type < 2 ? "first_then" : "first_else", "middle",
                type % 2 == 0 ? "second_then" : "second_else", "suffix"};
            if (function->hasAttr("test.owned_prefix")) {
                expected.insert(expected.begin(), "owned");
                const auto& projection = result.demands->original->types[type].projectedStorage;
                if (!projection || !projection->owner) {
                    llvm::errs() << "whole-visit ownership projection lost its joint proof\n"; return 1;
                }
            }
            if (alternative.nodes.size() != 5 || alternative.selection.size() != 2 ||
                body.anchors.size() != expected.size()) {
                llvm::errs() << "whole-visit prefix/choice/suffix shape lost\n"; return 1;
            }
            auto& arena = *body.expressions;
            for (std::size_t site = 0; site < expected.size(); ++site) {
                auto label = body.anchors[site].phase->elementOp->getAttrOfType<StringAttr>("test.label");
                auto present = fs::regionalPresence(body,
                    {static_cast<uint32_t>(site), arena.constant(0), fs::PeriodicEventKind::Start});
                if (!label || label.getValue() != expected[site] ||
                    !present || arena.constantValue(*present) != 1) {
                    llvm::errs() << "whole-visit occurrence order or independent choice presence lost\n"; return 1;
                }
            }
        }
    }
    if (auto repeats = function->getAttrOfType<IntegerAttr>("test.repeated_choices")) {
        const auto length = repeats.getInt();
        if (length <= 1 || nodes.size() != 2 * static_cast<uint64_t>(length) ||
            result.recognition.typeDescriptions != 3) {
            llvm::errs() << "repeated condition expanded syntactic combinations\n"; return 1;
        }
        for (std::size_t type = 0; type < count; ++type) {
            const auto& alternative = result.recognition.alternatives[type];
            const auto& body = result.demands->original->types[type].body;
            if (alternative.selection.size() != static_cast<uint64_t>(length) ||
                body.anchors.size() != static_cast<uint64_t>(length)) {
                llvm::errs() << "repeated condition lost original decisions or payloads\n"; return 1;
            }
            auto firstBranch = alternative.selection.front().branch;
            auto condition = firstBranch.getCondition();
            for (std::size_t i = 0; i < body.anchors.size(); ++i) {
                auto choice = alternative.selection[i];
                const StringRef expected = (i % 2 == 0) == (type == 0) ? "pto.textract" : "pto.tmatmul";
                if (choice.branch.getCondition() != condition || choice.takeThen != (type == 0) ||
                    body.anchors[i].phase->elementOp->getName().getStringRef() != expected) {
                    llvm::errs() << "repeated SSA condition has a contradictory visit type\n"; return 1;
                }
            }
        }
    }
    // Distinct visit choices must retain nonempty cross-arm storage demands.
    for (const auto& boundary : result.demands->boundaries) {
        if (boundary.demands.empty()) { llvm::errs() << "missing finite visit boundary demands\n"; return 1; }
    }
    const auto* before = results.begin()->second.get();
    if (failed(analysis.analyzeFiniteVisitCandidates()) ||
        analysis.finiteVisitDemands().begin()->second.get() != before) {
        llvm::errs() << "finite visit child/pair analysis was not cached\n"; return 1;
    }
    bool audited = false;
    for (const auto& candidate : analysis.result()->contractAudit) {
        if (candidate.kind == fs::ContractClass::FiniteVisitTypes &&
            candidate.membership == fs::ContractStatus::Established &&
            candidate.demands == fs::ContractImplementation::Available) { audited = true; }
    }
    if (!audited) { llvm::errs() << "finite visit success missing from contract audit\n"; return 1; }
    llvm::outs() << "finite visit original input: " << count << " types, " << count * count
                 << " cached boundaries, " << nodes.size() << " analyzed nodes, demand-only exports\n";
    return 0;
}
