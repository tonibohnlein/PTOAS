// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Supplied finite protocol fixture: two distinct banks, one grouped SET/WAIT
// site, fixed child visits j=0,1, and an enclosing invocation coordinate.
// Its two-ID certificate is independently checked in the Python event oracle.
#include "PTO/Transforms/FrontierSynch/FrontierAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionInlineChild.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Matchers.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
bool fixedTwoVisits(scf::ForOp loop)
{
    APInt lower, upper, step;
    return matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) && lower == 0 &&
        matchPattern(loop.getUpperBound(), m_ConstantInt(&upper)) && upper == 2 &&
        matchPattern(loop.getStep(), m_ConstantInt(&step)) && step == 1;
}
DictionaryAttr suppliedCertificate(MLIRContext* context, bool badArity)
{
    Builder b(context);
    auto rule = b.getDictionaryAttr({b.getNamedAttr("coordinate_count", b.getI64IntegerAttr(badArity ? 2 : 1)),
        b.getNamedAttr("base", b.getI64IntegerAttr(0)),
        b.getNamedAttr("terms", b.getArrayAttr({b.getDenseI64ArrayAttr({0, 1, 0, 2, 1})}))});
    auto group = b.getDictionaryAttr({b.getNamedAttr("source", b.getI64IntegerAttr(4)),
        b.getNamedAttr("target", b.getI64IntegerAttr(3)), b.getNamedAttr("budget", b.getI64IntegerAttr(2)),
        b.getNamedAttr("records", b.getDenseI64ArrayAttr({0, 1})),
        b.getNamedAttr("strides", b.getDenseI64ArrayAttr({0, 0})),
        b.getNamedAttr("phases", b.getDenseI64ArrayAttr({0, 0})),
        b.getNamedAttr("conflicts", b.getDenseI64ArrayAttr({})),
        b.getNamedAttr("tuple_rules", b.getArrayAttr({rule, rule}))});
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
        b.getNamedAttr("plan", b.getI64IntegerAttr(0)),
        b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
        b.getNamedAttr("groups", b.getArrayAttr({group}))});
}
std::unique_ptr<fs::PreparedLogicalPlan> suppliedPlan(scf::ForOp loop, Operation* cut, bool badArity)
{
    auto plan = std::make_unique<fs::PreparedLogicalPlan>(0);
    plan->groupedFamilies = plan->nestedIdentities = plan->independentPieces = true;
    fs::EndpointFamily family;
    family.sourcePipe = 4; family.targetPipe = 3;
    family.sourceCut = family.targetCut = {cut->getBlock(), cut};
    for (uint32_t record : {0u, 1u}) {
        family.members.push_back({record, 0, 1, {{loop, record}}, {{loop, record}}});
    }
    plan->families.push_back(std::move(family));
    OpBuilder b(cut->getContext());
    b.setInsertionPointToEnd(&plan->addPreparation(cut));
    auto active = b.create<arith::ConstantIntOp>(cut->getLoc(), 1, 1);
    for (auto kind : {fs::LogicalCommandKind::Set, fs::LogicalCommandKind::Wait}) {
        fs::PreparedLogicalEndpoint endpoint;
        endpoint.before = cut; endpoint.kind = kind; endpoint.sourcePipe = 4; endpoint.targetPipe = 3;
        endpoint.guard = active; endpoint.identity = loop.getInductionVar();
        endpoint.memberCoordinates.push_back(loop.getInductionVar());
        endpoint.records = {0, 1}; endpoint.piece = plan->endpoints.size();
        plan->endpoints.push_back(std::move(endpoint));
    }
    plan->allocationCertificate = suppliedCertificate(cut->getContext(), badArity);
    return plan;
}
} // namespace
LogicalResult runGroupedCompletionParent(func::FuncOp function,
    std::shared_ptr<const fs::MathematicalResult> mathematics)
{
    scf::ForOp loop;
    Operation* cut = nullptr;
    function.walk([&](scf::ForOp candidate) { if (candidate->hasAttr("test.grouped_visit")) { loop = candidate; } });
    if (!loop || !fixedTwoVisits(loop) || !mathematics || !mathematics->input) { return failure(); }
    for (auto& op : loop.getBody()->without_terminator()) {
        if (op.getName().getStringRef() == "pto.textract") { cut = &op; }
    }
    if (!cut) { return failure(); }
    auto index = std::make_shared<fs::PhaseIndex>();
    if (failed(index->build(function, *mathematics->input))) { return failure(); }
    auto prepare = [loop, cut, bad = function->hasAttr("test.bad_child_arity")]() {
        return FailureOr<std::unique_ptr<fs::PreparedLogicalPlan>>(suppliedPlan(loop, cut, bad));
    };
    auto child = fs::CompletionInlineChild::qualifyVisit(mathematics, mathematics->input, loop, prepare);
    function->setAttr("test.imported_grouped_child", BoolAttr::get(function.getContext(), bool(child)));
    const bool pendingPayload = child && llvm::any_of(child->summary().suffix, [](uint8_t mode) { return mode; });
    const bool transparent = child && child->summary().transparent;
    SmallVector<std::shared_ptr<const fs::CompletionInlineChild>> children;
    SmallVector<Operation*> prefix;
    for (auto& operation : function.getBody().front()) {
        if (operation.hasAttr("test.prefix_payload")) { prefix.push_back(&operation); }
    }
    auto source = fs::finiteFunctionInput(function, mathematics->input, index);
    auto finite = fs::analyzeFinite(fs::finiteRegionInput(source, prefix), fs::FiniteStrategy::Concrete);
    auto leading = fs::CompletionInlineChild::qualify(finite);
    if (!leading) { return function.emitError("prefix child qualification failed"); }
    children.push_back(std::move(leading));
    if (child) { children.push_back(child); }
    std::string error;
    auto parent = fs::CompletionRegion::buildWithChildren(function, mathematics->input, *index, children, error);
    if (!parent) { return function.emitError("explicit conservative parent: ") << error; }
    const auto retained = parent->eventfulInlineChildren();
    fs::PreparedLogicalPlan plan(0);
    plan.completeInvocation = true; plan.completionOwner = parent;
    if (failed(fs::insertLogicalSynchronization(function, plan))) { return failure(); }
    function->setAttr("test.retained_eventful_children",
        IntegerAttr::get(IntegerType::get(function.getContext(), 64), retained));
    function->setAttr("test.pending_child_payloads", BoolAttr::get(function.getContext(), pendingPayload));
    function->setAttr("test.transparent_child", BoolAttr::get(function.getContext(), transparent));
    return verify(function);
}
