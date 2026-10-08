// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Reject non-partitions and malformed alternative metadata before mutation.
#include "PTO/Transforms/FrontierSynch/BalancedCompactBody.h"
#include "PTO/Transforms/FrontierSynch/CompactBoundingAllocation.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::string render(func::FuncOp function)
{
    std::string result; llvm::raw_string_ostream stream(result); function.print(stream); return result;
}
bool partitionChecks(const fs::BalancedCompactBody& body)
{
    if (fs::qualifyEndpointCutChoices(body.loop, {})) { return false; }
    for (const auto& slot : body.slots) {
        SmallVector<fs::TemplateEndpointCut> cuts;
        for (const auto& alternative : slot.alternatives) { cuts.push_back(alternative.before); }
        auto proof = fs::qualifyEndpointCutChoices(body.loop, cuts);
        if (!proof || proof->cuts().size() != cuts.size()) { return false; }
        auto duplicate = cuts; duplicate.push_back(cuts.front());
        if (fs::qualifyEndpointCutChoices(body.loop, duplicate)) { return false; }
        if (cuts.size() > 1 && fs::qualifyEndpointCutChoices(
            body.loop, ArrayRef<fs::TemplateEndpointCut>(cuts).drop_front())) {
            return false;
        }
        auto invalid = cuts; invalid.front().block = nullptr;
        if (fs::qualifyEndpointCutChoices(body.loop, invalid)) { return false; }
        if (fs::qualifyEndpointCutChoices({}, cuts)) { return false; }
    }
    // Two slots execute sequentially; their union cannot be one alternative set.
    SmallVector<fs::TemplateEndpointCut> both;
    for (const auto& slot : body.slots) {
        for (const auto& alternative : slot.alternatives) { both.push_back(alternative.before); }
    }
    return !fs::qualifyEndpointCutChoices(body.loop, both);
}
bool check(func::FuncOp function, const pto::SyncInput& input)
{
    if (function->hasAttr("test.balanced_reject")) { return true; }
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return false; }
    scf::ForOp loop;
    function.walk([&](scf::ForOp candidate) { if (!loop) { loop = candidate; } });
    auto body = fs::recognizeBalancedCompactBody(loop, input, index);
    if (!body.error.empty() || body.slots.size() != 2 || !partitionChecks(body)) { return false; }
    auto graph = fs::analyzePeriodicDemands({{body.slots[0].pipe}, {body.slots[1].pipe}},
                                           {{0, 1, 0}, {1, 0, 1}});
    std::string error;
    auto prepared = fs::prepareBalancedCompactInsertion(function, body, graph, 97, error);
    if (failed(prepared) || !error.empty()) { return false; }
    fs::RegionalAnalysis region;
    region.expressions = std::make_shared<fs::RegionExpressions>();
    region.anchors.resize(body.slots.size());
    region.occurrenceLoops.assign(body.slots.size(), body.loop);
    const auto trips = region.expressions->constant(5);
    auto emptyGraph = fs::analyzePeriodicDemands(graph.payloads, {});
    auto emptyPlan = fs::prepareBalancedCompactInsertion(function, body, emptyGraph, 97, error);
    if (failed(emptyPlan) || !(*emptyPlan)->endpoints.empty() ||
        !fs::attachCompactBoundingAllocation(region, emptyGraph, trips, **emptyPlan, error)) { return false; }
    auto emptyCertificate = (*emptyPlan)->allocationCertificate;
    auto budget = emptyCertificate ? emptyCertificate.getAs<IntegerAttr>("budget") : IntegerAttr{};
    auto entries = emptyCertificate ? emptyCertificate.getAs<ArrayAttr>("entries") : ArrayAttr{};
    if (!budget || budget.getInt() != 0 || !entries || !entries.empty()) { return false; }
    auto& plan = **prepared;
    auto* saved = plan.endpoints.front().before;
    plan.endpoints.front().before = nullptr;
    if (fs::attachCompactBoundingAllocation(region, graph, trips, plan, error)) { return false; }
    plan.endpoints.front().before = saved;
    plan.families.push_back(plan.families.front());
    if (fs::attachCompactBoundingAllocation(region, graph, trips, plan, error)) { return false; }
    plan.families.pop_back();
    if (!fs::attachCompactBoundingAllocation(region, graph, trips, plan, error) ||
        !plan.allocationCertificate || !plan.regionalAllocation ||
        failed(fs::insertLogicalSynchronization(function, plan)) || failed(verify(function))) { return false; }
    // A zero-handoff proof must never authorize this nonempty notification plan.
    const auto originalCertificate = function->getAttr(fs::CyclicAllocationAttr);
    function->setAttr(fs::CyclicAllocationAttr, emptyCertificate);
    const auto malformed = render(function);
    {
        ScopedDiagnosticHandler suppress(function.getContext(), [](Diagnostic&) { return success(); });
        if (succeeded(fs::allocatePhysicalEventIds(function, {})) || render(function) != malformed) { return false; }
    }
    function->setAttr(fs::CyclicAllocationAttr, originalCertificate);
    Builder builder(function.getContext());
    const auto metadata = function->getAttrOfType<DictionaryAttr>("pto.endpoint_families");
    auto families = metadata.getAs<ArrayAttr>("families");
    bool checked = false;
    for (std::size_t i = 0; i < families.size(); ++i) {
        auto family = cast<DictionaryAttr>(families[i]);
        auto choices = family.getAs<DenseI64ArrayAttr>("source_choices");
        if (!choices || choices.empty()) { continue; }
        for (bool wrongLoop : {false, true}) {
            SmallVector<int64_t> malformed(choices.asArrayRef());
            if (wrongLoop) { malformed[0] = INT64_MAX; }
            else { malformed[2] = malformed[1]; }
            NamedAttrList changedFamily(family);
            changedFamily.set("source_choices", builder.getDenseI64ArrayAttr(malformed));
            SmallVector<Attribute> changedFamilies(families.begin(), families.end());
            changedFamilies[i] = changedFamily.getDictionary(function.getContext());
            NamedAttrList changedMetadata(metadata);
            changedMetadata.set("families", builder.getArrayAttr(changedFamilies));
            function->setAttr("pto.endpoint_families", changedMetadata.getDictionary(function.getContext()));
            const auto before = render(function);
            ScopedDiagnosticHandler suppress(function.getContext(), [](Diagnostic&) { return success(); });
            if (succeeded(fs::allocatePhysicalEventIds(function, {0, 1, 2, 3, 4, 5})) ||
                render(function) != before) { return false; }
            function->setAttr("pto.endpoint_families", metadata);
        }
        checked = true; break;
    }
    if (body.slots[0].alternatives.size() > 1 && !checked) { return false; }
    return succeeded(fs::allocatePhysicalEventIds(function, {0, 1, 2, 3, 4, 5})) && succeeded(verify(function));
}
} // namespace
int runCompactBoundingAllocationChecks(func::FuncOp function, const pto::SyncInput& input)
{
    if (!check(function, input)) {
        llvm::errs() << "compact bounding allocation checks failed: " << function.getSymName() << "\n";
        return 1;
    }
    llvm::outs() << "compact bounding allocation checks passed: " << function.getSymName() << "\n";
    return 0;
}
