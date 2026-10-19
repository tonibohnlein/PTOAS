// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ResultGuarantees.h"
#include "PTO/Transforms/FrontierSynch/AnalysisRequest.h"
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticRegional.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompactBoundingInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "WholeInvocationIdentity.h"
#include "mlir/IR/Builders.h"
namespace mlir::pto::frontiersynch {
namespace {
ResultGuarantees mathematicalGuarantees(const MathematicalResult* source)
{
    ResultGuarantees result;
    if (!source) { return result; }
    result.input = source->input;
    while (source && source->identityComposition) { source = source->identityComposition->child.get(); }
    if (!source) { return result; }
    if (source->compactDemands) {
        if (source->compactDemands->boundary) {
            const auto& bounds = source->compactDemands->boundary->bounds();
            result.modeledOrder = bounds.guarantee;
            result.reduction = bounds.reduction;
            result.selectedOrderBounds = bounds.lower.has_value() && bounds.upper.has_value();
        }
    } else {
        result.modeledOrder = InputOrderGuarantee::InputOrderEquivalent;
        result.reduction = ReductionQuality::Covers;
    }
    return result;
}
ResultGuarantees completionGuarantees(const CompletionRegion& completion)
{
    ResultGuarantees result;
    result.input = completion.inputOwner();
    result.completion = true;
    result.modeledOrder = InputOrderGuarantee::InputOrderCovering;
    result.realization = RealizationOrder::CompletionCoverage;
    result.completionClasses = completion.transfer().summary.all.size();
    result.completedExit = completion.completesInvocation();
    result.realizedLoss = completion.lossReport();
    result.inlineChildren = completion.inlineChildren();
    result.eventfulInlineChildren = completion.eventfulInlineChildren();
    result.readinessCuts = completion.constructionWork().readinessCuts;
    const auto children = completion.childContracts();
    result.eventFreeChildren = children.eventFree;
    result.completedChildren = children.completedExit;
    result.descendantClosedChildren = children.closedDescendants;
    return result;
}
const char* realizationName(RealizationOrder value)
{
    switch (value) {
        case RealizationOrder::Unestablished: return "not-established";
        case RealizationOrder::CoversSelectedOrder: return "covers-selected-order";
        case RealizationOrder::Equivalent: return "equivalent-to-selected-order";
        case RealizationOrder::MayStrengthen: return "may-strengthen-selected-order";
        case RealizationOrder::CompletionCoverage: return "covers-modeled-order";
    }
    llvm_unreachable("invalid realization guarantee");
}
} // namespace
RealizationOrder explicitRealizationOrder(const ExplicitAnalysis& analysis)
{
    const auto& ranks = analysis.reduction;
    for (const auto& edge : ranks.retained) {
        if (edge.source >= ranks.pipeColumns.size() || edge.target >= ranks.pipeColumns.size() ||
            edge.source >= ranks.localRanks.size() || edge.target >= ranks.localRanks.size()) {
            return RealizationOrder::CoversSelectedOrder;
        }
        if (ranks.pipeColumns[edge.source] == ranks.pipeColumns[edge.target] &&
            uint64_t(ranks.localRanks[edge.target]) != uint64_t(ranks.localRanks[edge.source]) + 1) {
            return RealizationOrder::MayStrengthen;
        }
    }
    return RealizationOrder::Equivalent;
}
ResultGuarantees describeResultGuarantees(const AnalysisOutcome& outcome)
{
    if (outcome.completion) { return completionGuarantees(*outcome.completion); }
    return mathematicalGuarantees(outcome.mathematical.get());
}
ResultGuarantees describeResultGuarantees(const PreparedLogicalPlan& plan)
{
    if (plan.completionOwner) { return completionGuarantees(*plan.completionOwner); }
    auto result = mathematicalGuarantees(plan.mathematicalOwner.get());
    result.completedExit = plan.completeInvocation;
    if (!plan.mathematicalOwner) { return result; }
    result.realization = RealizationOrder::CoversSelectedOrder;
    auto* source = plan.mathematicalOwner.get();
    while (source && source->identityComposition) { source = source->identityComposition->child.get(); }
    if (!source) { return result; }
    if (plan.privateState) { return result; } // Prefix matching alone is not an order-equality certificate.
    if (source->compactDemands) {
        result.realization = source->compactDemands->placementMayStrengthen ?
            RealizationOrder::MayStrengthen : RealizationOrder::CoversSelectedOrder;
    } else if (source->concreteDemands()) {
        result.realization = explicitRealizationOrder(*source->concreteDemands());
    } else if ((source->arithmeticDemands && source->arithmeticDemands->adjacentLocalDemands) ||
               (source->generalArithmeticDemands && source->generalArithmeticDemands->adjacentLocalDemands) ||
               (source->arithmeticRegionalDemands &&
                source->arithmeticRegionalDemands->analysis.adjacentLocalDemands)) {
        result.realization = RealizationOrder::Equivalent;
    }
    return result;
}
DictionaryAttr encodeResultGuarantees(MLIRContext* context, const ResultGuarantees& value)
{
    Builder b(context);
    NamedAttrList report;
    report.set("version", b.getI64IntegerAttr(1));
    report.set("basis", b.getStringAttr(value.input ? "shared-modeled-effects" : "unavailable"));
    report.set("original_effect_precision", b.getStringAttr("not-independently-certified"));
    report.set("selected_order", b.getStringAttr(!value.modeledOrder ? "unavailable" :
        *value.modeledOrder == InputOrderGuarantee::InputOrderEquivalent ? "equivalent" : "covering"));
    report.set("reduction", b.getStringAttr(!value.reduction ? "unavailable" :
        *value.reduction == ReductionQuality::Covers ? "covers" :
        *value.reduction == ReductionQuality::Partial ? "partial" : "generators"));
    report.set("realization_order", b.getStringAttr(realizationName(value.realization)));
    report.set("selected_order_bounds", b.getBoolAttr(value.selectedOrderBounds));
    report.set("realized_loss_bound", value.realizedLoss ? Attribute(value.realizedLoss) :
        Attribute(b.getStringAttr("unavailable")));
    report.set("invocation_exit", b.getStringAttr(value.completedExit ? "completed" : "not-certified-completed"));
    if (value.completion) {
        NamedAttrList completion;
        completion.set("placement", b.getStringAttr("scc-local"));
        completion.set("protocol", b.getStringAttr("qualified-a3-all-barrier"));
        completion.set("storage_classes", b.getI64IntegerAttr(value.completionClasses));
        completion.set("parent_event_ids", b.getI64IntegerAttr(0));
        completion.set("storage_model", b.getStringAttr("proved-spaces-and-reservations-with-may-alias"));
        completion.set("inline_children", b.getI64IntegerAttr(value.inlineChildren));
        completion.set("eventful_inline_children", b.getI64IntegerAttr(value.eventfulInlineChildren));
        completion.set("scalar_readiness_cuts", b.getI64IntegerAttr(value.readinessCuts));
        completion.set("scalar_readiness", b.getStringAttr("qualified-same-block-first-use"));
        completion.set("child_payload_completion", b.getStringAttr("independent-of-matching-closure"));
        completion.set("event_free_children", b.getI64IntegerAttr(value.eventFreeChildren));
        completion.set("completed_exit_children", b.getI64IntegerAttr(value.completedChildren));
        completion.set("closed_descendant_children", b.getI64IntegerAttr(value.descendantClosedChildren));
        report.set("completion_contract", completion.getDictionary(context));
    }
    return report.getDictionary(context);
}
void attachResultGuarantees(PreparedLogicalPlan& plan, MLIRContext* context)
{
    NamedAttrList report(plan.recognitionReport);
    report.set("guarantees", encodeResultGuarantees(context, describeResultGuarantees(plan)));
    plan.recognitionReport = report.getDictionary(context);
}
} // namespace mlir::pto::frontiersynch
