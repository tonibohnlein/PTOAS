// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check shared normalized input across different mathematical adapters.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/FiniteGuardedAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/NormalizedControl.h"
#include "../../lib/PTO/Transforms/FrontierSynch/NumericTemplatePlan.h"
#include "../../lib/PTO/Transforms/FrontierSynch/FiniteExpansionPlan.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
LogicalResult checkNormalizedControlSession(func::FuncOp function, pto::GMAliasPolicy policy)
{
    using namespace pto::frontiersynch;
    FrontierAnalysis session(function);
    if (failed(session.initialize(policy))) { return failure(); }
    auto node = llvm::find_if(session.result()->nodes, [&](const auto& value) {
        return value.kind == StructureKind::Loop && value.anchor->getParentOp() == function;
    });
    const bool fresh = node != session.result()->nodes.end() && !session.normalizationConstructions();
    if (!fresh) { return failure(); }
    const auto id = static_cast<std::size_t>(node - session.result()->nodes.begin());
    std::string originalIR;
    llvm::raw_string_ostream originalStream(originalIR);
    function.print(originalStream);
    const auto numerical = session.analyzeNumericalRegion({id});
    const bool rejectsNumerical = function->hasAttr("test.normalization_runtime_guard");
    const bool expectedNumerical = rejectsNumerical ? !numerical.mathematical :
        numerical.mathematical && numerical.mathematical->normalizedInput;
    const bool oneInput = expectedNumerical && session.normalizationConstructions() == 1;
    if (!oneInput) {
        return function.emitError("numerical adapter did not retain exactly one normalized input");
    }
    const auto finite = session.analyzeFiniteExpansion({id});
    const bool available = finite.status == AnalysisStatus::Ready && finite.mathematical &&
        finite.mathematical->finiteGuardedDemands && finite.mathematical->normalizedInput;
    if (!available) { return function.emitError("finite adapter cannot consume retained normalized input"); }
    const auto owner = finite.mathematical->normalizedInput;
    const bool shared = session.normalizationConstructions() == 1 &&
        (!numerical.mathematical || numerical.mathematical->normalizedInput == owner) &&
        session.analyzeFiniteExpansion({id}).mathematical == finite.mathematical &&
        session.analyzeNumericalRegion({id}).mathematical == numerical.mathematical &&
        !session.constructionCounts().logicalPreparations && !session.constructionCounts().allocationExports;
    if (!shared) { return function.emitError("mathematical adapters rebuilt normalization or constructed exports"); }
    SmallVector<int64_t> coordinates;
    for (const auto& normalized : owner->nodes) {
        const bool extract = normalized.original &&
            normalized.original->getName().getStringRef() == "pto.textract";
        if (!extract) { continue; }
        for (auto fixed : normalized.fixedCoordinates) {
            const bool inner = fixed.loop.getOperation() != node->anchor;
            if (inner) { coordinates.push_back(fixed.induction); }
        }
    }
    const auto expected = function->getAttrOfType<DenseI64ArrayAttr>("test.normalization_coordinates");
    const bool mapped = expected && ArrayRef<int64_t>(coordinates) == expected.asArrayRef() &&
        (owner->expandedLoops || function->hasAttr("test.normalization_compact")) &&
        owner->result.state == RecognitionState::Applicable;
    if (!mapped) { return function.emitError("normalized visits lost zero/one/non-unit original coordinates"); }
    const auto& index = *owner->index;
    const auto& input = *owner->input;
    auto malformed = owner->context;
    malformed.roots = {malformed.root, malformed.root};
    auto duplicate = normalizeSmallCountControl(malformed, index, input);
    malformed.roots = {malformed.root->getNextNode()};
    auto displaced = normalizeSmallCountControl(malformed, index, input);
    malformed.function = {}; malformed.roots = {malformed.root, malformed.root};
    auto missingFunction = normalizeSmallCountControl(malformed, index, input);
    const bool invalidRoots = missingFunction->result.state != RecognitionState::Applicable &&
        missingFunction->nodes.empty() && duplicate->result.state != RecognitionState::Applicable &&
        displaced->result.state != RecognitionState::Applicable && duplicate->nodes.empty() && displaced->nodes.empty();
    if (!invalidRoots) { return function.emitError("normalization accepted malformed root selections"); }
    FiniteExpansionLimits small; small.visits = 1; small.payloads = 1;
    auto compact = normalizeSmallCountControl(owner->context, index, input, small);
    const bool linear = compact->result.state == RecognitionState::Applicable &&
        compact->expandedLoops == 0 && compact->planningVisits == compact->nodes.size() &&
        owner->planningVisits == compact->planningVisits;
    if (!linear) { return function.emitError("optional expansion did not retain a linearly planned compact input"); }
    uint64_t compactPayloads = 0, compactEffects = 0;
    for (const auto& source : compact->nodes) {
        for (auto* phase : index.phasesFor(source.original)) {
            ++compactPayloads;
            compactEffects += input.accesses().effectsFor(phase).size();
        }
    }
    if (compact->payloads != compactPayloads || compact->effects != compactEffects) {
        return function.emitError("compact fallback lost original payload/effect counts");
    }
    const auto direct = preflightFiniteExpansion(owner->context, index, input, {}, compact);
    if (direct.result.state != RecognitionState::Applicable) {
        return function.emitError("compact fallback lost finite enumeration or dead-arm pruning");
    }
    FiniteExpansionLimits exact;
    exact.visits = direct.visits.size();
    auto boundary = preflightFiniteExpansion(owner->context, index, input, exact, owner);
    --exact.visits;
    auto exhausted = preflightFiniteExpansion(owner->context, index, input, exact, owner);
    const bool bounded = boundary.result.state == RecognitionState::Applicable &&
        exhausted.result.state != RecognitionState::Applicable;
    if (!bounded) { return function.emitError("supplied input bypassed finite visit boundaries"); }
    if (!rejectsNumerical) {
        NumericTemplateLimits tiny; tiny.visits = 1;
        const auto rejected = preflightNumericTemplate(cast<scf::ForOp>(node->anchor), index, input,
            tiny, true, {}, {}, owner);
        if (rejected.form.result.state == RecognitionState::Applicable) {
            return function.emitError("supplied input bypassed numerical visit limits");
        }
        if (function->hasAttr("test.normalization_deep")) {
            tiny = {}; tiny.depth = 1;
            const auto shallow = preflightNumericTemplate(cast<scf::ForOp>(node->anchor), index, input,
                tiny, true, {}, {}, owner);
            if (shallow.form.result.state == RecognitionState::Applicable) {
                return function.emitError("supplied input bypassed numerical depth limits");
            }
        }
    }
    AnalysisRequest queries; queries.region = id; queries.needs.queries = true;
    const auto exported = session.analyzeFiniteExpansion(queries);
    const bool retained = exported.mathematical == finite.mathematical && session.normalizationConstructions() == 1;
    if (!retained) { return function.emitError("stronger exports replaced the normalized alternative"); }
    const auto alternate = policy == pto::GMAliasPolicy::MayAlias ?
        pto::GMAliasPolicy::MayNotAlias : pto::GMAliasPolicy::MayAlias;
    const bool reinitialized = succeeded(session.initialize(alternate)) && !session.normalizationConstructions();
    if (!reinitialized) { return failure(); }
    const auto isolated = session.analyzeFiniteExpansion({id});
    const bool independent = isolated.mathematical && isolated.mathematical->normalizedInput != owner &&
        session.normalizationConstructions() == 1 && owner->result.state == RecognitionState::Applicable;
    std::string unchangedIR;
    llvm::raw_string_ostream unchangedStream(unchangedIR);
    function.print(unchangedStream);
    if (!independent || unchangedIR != originalIR) {
        return function.emitError("normalized context isolation failed");
    }
    llvm::outs() << "normalized-control-session: shared-input numerical-to-finite "
                 << "original-coordinates isolated-context\n";
    return success();
}
