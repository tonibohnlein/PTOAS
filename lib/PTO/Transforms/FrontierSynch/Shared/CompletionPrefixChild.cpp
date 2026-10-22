// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Provider-owned whole-invocation boundary and assignment-preserving adapter.
#include "CompletionInlineChild.h"
#include "CompletionStorage.h"
#include "PTO/Transforms/FrontierSynch/Shared/FamilyInsertion.h"
namespace mlir::pto::frontiersynch {
std::optional<CompletionInlineChild::PrefixBoundary> CompletionInlineChild::prefixBoundary(
    const CompletionStorage& storage, const PhaseIndex& index) const
{
    // Nested SCC owners do not yet export closed event boundaries. A complete
    // loop is eligible; one open body iteration is deliberately not inferred.
    if (completion || roots().size() != 1 || !input()) {
        return {};
    }
    auto loop = dyn_cast<scf::ForOp>(roots().front());
    auto palette = resourceFootprint();
    if (!loop || !palette) {
        return {};
    }
    PrefixBoundary result{loop, {}, *palette};
    loop->walk([&](Operation* operation) {
        for (const auto* phase : index.phasesFor(operation)) {
            auto facts = storage.effects({phase});
            auto& modes = result.pendingByPipe[static_cast<uint32_t>(phase->kPipeValue)];
            if (modes.empty()) {
                modes.resize(storage.size());
            }
            for (std::size_t i = 0; i < modes.size(); ++i) {
                modes[i] |= facts[i];
            }
        }
    });
    for (const auto& entry : *palette) {
        if (!result.pendingByPipe.count(entry.first.first) || !result.pendingByPipe.count(entry.first.second)) {
            return {};
        }
    }
    return result;
}
namespace {
DictionaryAttr prefixCertificate(MLIRContext* context, ArrayRef<Attribute> groups)
{
    Builder b(context);
    return b.getDictionaryAttr(
        {b.getNamedAttr("version", b.getI64IntegerAttr(2)), b.getNamedAttr("kind", b.getStringAttr("finite")),
         b.getNamedAttr("strategy", b.getStringAttr("regional-palettes")),
         b.getNamedAttr("plan", b.getI64IntegerAttr(0)), b.getNamedAttr("groups", b.getArrayAttr(groups))});
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prefixPlan(const CompletionInlineChild& child)
{
    auto function = child.roots().front()->getParentOfType<func::FuncOp>();
    IRMapping mapping;
    function.walk([&](Operation* op) { mapping.map(op, op); });
    auto result = std::make_unique<PreparedLogicalPlan>(0);
    SmallVector<Operation*> cuts;
    DenseSet<const CompletionSourceSnapshot*> sources;
    DenseSet<const ClosedCompletionChild*> calls;
    CompletionChildAssembly assembly;
    if (failed(child.append(*result, mapping, cuts, sources, calls, assembly)) || !cuts.empty()) {
        return failure();
    }
    auto namespaces = familyRecordNamespaces(result->families);
    for (auto& endpoint : result->endpoints) {
        auto found = endpoint.records.empty() ? namespaces.end() : namespaces.find(endpoint.records.front());
        if (found == namespaces.end()) {
            return failure();
        }
        endpoint.record = found->second;
    }
    result->allocationCertificate = prefixCertificate(function.getContext(), assembly.allocationGroups);
    return result;
}
bool sameRecordMap(
    const PreparedLogicalEndpoint& endpoint, PhysicalRecordAllocation original,
    const PhysicalRecordAllocation& converted, bool translatedVisit, DominanceInfo& dominance)
{
    if (endpoint.memberCoordinates.empty()) {
        return false;
    }
    // qualifyVisit stores an absolute first ID for appendChildPalettes. Recover
    // the table-local offset; physical tuple coordinates never include labels.
    if (translatedVisit && original.tupleRule) {
        if (original.ids.empty() || original.tupleRule->base < uint64_t(original.ids.front())) {
            return false;
        }
        original.tupleRule->base -= original.ids.front();
    }
    const auto arity = endpoint.memberCoordinates.size();
    auto before = planPhysicalEndpointMap(endpoint.before, {original}, {{}}, arity, true, dominance);
    auto after = planPhysicalEndpointMap(endpoint.before, {converted}, {{}}, arity, true, dominance);
    return succeeded(before) && succeeded(after) && equivalentPhysicalMaps(**before, **after);
}
bool sameAssignment(
    func::FuncOp function, const PreparedLogicalPlan& plan, const PhysicalAllocationPlan& original,
    const PhysicalAllocationPlan& decoded, bool translatedVisit)
{
    if (decoded.records.size() != original.records.size()) {
        return false;
    }
    std::map<int64_t, const PhysicalRecordAllocation*> records;
    for (const auto& record : decoded.records) {
        if (!records.emplace(record.record, &record).second) {
            return false;
        }
    }
    std::map<int64_t, const PreparedLogicalEndpoint*> endpoints;
    for (const auto& endpoint : plan.endpoints) {
        if (endpoint.kind == LogicalCommandKind::Barrier) {
            continue;
        }
        for (auto label : endpoint.records) {
            endpoints.try_emplace(label, &endpoint);
        }
    }
    DominanceInfo dominance(function);
    // Total singleton-map equivalence for every possible grouped label.
    for (const auto& record : original.records) {
        auto found = records.find(record.record);
        auto endpoint = endpoints.find(record.record);
        if (found == records.end() || endpoint == endpoints.end() ||
            !sameRecordMap(*endpoint->second, record, *found->second, translatedVisit, dominance)) {
            return false;
        }
    }
    return true;
}
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> CompletionInlineChild::preparePrefixImport(
    ArrayRef<int64_t> eligibleIds) const
{
    if (completion || !input() || !eventAssignment || roots().size() != 1) {
        return failure();
    }
    auto function = roots().front()->getParentOfType<func::FuncOp>();
    if (!function || function->hasAttr("pto.closed_completion")) {
        return failure();
    }
    auto result = prefixPlan(*this);
    if (failed(result)) {
        return failure();
    }
    auto decoded = decodePhysicalAllocationCertificate(function, (*result)->allocationCertificate, eligibleIds);
    if (failed(decoded) || !sameAssignment(function, **result, *eventAssignment, *decoded, bool(visitPreparation))) {
        return failure();
    }
    return std::move(*result);
}
} // namespace mlir::pto::frontiersynch
