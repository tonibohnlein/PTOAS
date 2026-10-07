// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Phase-guarded endpoints on original cuts; only the numerical analysis body is expanded.
#include "PTO/Transforms/FrontierSynch/MixedStrideAnalysis.h"
#include "PTO/Transforms/FrontierSynch/PeriodicSharedCertificate.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "CircuitEndpoints.h"
#include "CountedLoop.h"
#include "RecognitionInternal.h"
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
std::optional<RecognitionResult> mixedContract(const RecognitionResult& original)
{
    if (original.diagnostics.empty() || llvm::any_of(original.diagnostics, [](const auto& diagnostic) {
            return diagnostic.issue != RecognitionIssue::CommonStride;
        })) { return std::nullopt; }
    auto result = original;
    result.diagnostics.clear();
    result.state = RecognitionState::Applicable;
    return result;
}
const StructureNode* wholeLoop(func::FuncOp function, const SyncInput& input, const ProgramRecognition& program)
{
    if (!function || function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) { return nullptr; }
    const StructureNode* selected = nullptr;
    for (const auto& node : program.nodes) {
        if (node.kind != StructureKind::Loop || node.anchor->getParentOp() != function) { continue; }
        if (selected || !node.rotatingResult || !mixedContract(*node.rotatingResult)) { return nullptr; }
        selected = &node;
    }
    if (!selected || llvm::any_of(input.instructions(), [&](const auto* phase) {
            return !selected->anchor->isProperAncestor(phase->elementOp);
        })) { return nullptr; }
    return selected;
}
DictionaryAttr remapCertificate(const PeriodicAnalysis& periodic, MLIRContext* context)
{
    auto certificate = encodePeriodicSharedAllocation(periodic, 0, context);
    if (!certificate) { return {}; }
    std::map<uint32_t, uint32_t> records;
    for (auto [index, record] : llvm::enumerate(periodic.retained)) {
        records.emplace(record, static_cast<uint32_t>(index));
    }
    SmallVector<Attribute> entries;
    for (auto raw : certificate.getAs<ArrayAttr>("entries")) {
        auto entry = cast<DictionaryAttr>(raw);
        auto id = entry.getAs<IntegerAttr>("record").getInt();
        if (id < 0 || static_cast<uint64_t>(id) > UINT32_MAX) { return {}; }
        auto found = records.find(static_cast<uint32_t>(id));
        if (found == records.end()) { return {}; }
        NamedAttrList attributes(entry);
        attributes.set("record", IntegerAttr::get(IntegerType::get(context, 64), found->second));
        entries.push_back(attributes.getDictionary(context));
    }
    NamedAttrList attributes(certificate);
    attributes.set("entries", ArrayAttr::get(context, entries));
    return attributes.getDictionary(context);
}
struct EndpointBuilder {
    RegionExpressions arena;
    std::vector<TemplateEndpointAnchor> anchors;
    uint64_t period;
    Id trips, ordinal, generation, phase;
    std::vector<Id> counts;
    EndpointBuilder(const CountedLoop& loop, ArrayRef<const CompoundInstanceElement*> phases, uint64_t period)
        : period(period)
    {
        trips = loop.trips(arena);
        auto original = loop.loop;
        ordinal = arena.div(arena.sub(arena.input(original.getInductionVar()),
            arena.input(original.getLowerBound())), arena.constant(loop.step));
        generation = arena.div(ordinal, arena.constant(period));
        phase = arena.rem(ordinal, arena.constant(period));
        for (const auto* payload : phases) {
            auto* op = payload->elementOp;
            anchors.push_back({payload, {}, {op->getBlock(), op}, {op->getBlock(), op->getNextNode()}});
        }
        for (uint64_t r = 0; r < period; ++r) {
            const auto present = arena.lt(arena.constant(r), trips);
            const auto end = arena.select(present,
                arena.sub(arena.sub(trips, arena.constant(r)), arena.constant(1)), arena.constant(0));
            counts.push_back(arena.select(present,
                arena.add(arena.div(end, arena.constant(period)), arena.constant(1)), arena.constant(0)));
        }
    }
    bool add(CircuitEndpoints& emit, const PeriodicRecord& record)
    {
        const auto source = record.source % anchors.size(), target = record.target % anchors.size();
        const auto sourcePhase = record.source / anchors.size(), targetPhase = record.target / anchors.size();
        if (sourcePhase >= period || targetPhase >= period) { return false; }
        const auto d = arena.constant(record.displacement);
        auto publish = arena.land(arena.eq(phase, arena.constant(sourcePhase)),
                                  arena.lt(generation, counts[sourcePhase]));
        publish = arena.land(publish, arena.land(arena.lt(d, counts[targetPhase]),
            arena.lt(generation, arena.sub(counts[targetPhase], d))));
        auto consume = arena.land(arena.eq(phase, arena.constant(targetPhase)),
                                  arena.lt(generation, counts[targetPhase]));
        consume = arena.land(consume, arena.land(arena.le(d, generation),
            arena.lt(arena.sub(generation, d), counts[sourcePhase])));
        return emit.add(static_cast<uint32_t>(source), static_cast<uint32_t>(target), publish, consume,
                        {generation}, {arena.sub(generation, d)});
    }
};
} // namespace
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareMixedStrideInsertion(
    func::FuncOp function, const SyncInput& input, const ProgramRecognition& program, std::string& error)
{
    error.clear();
    auto* selected = wholeLoop(function, input, program);
    if (!selected) { error = "mixed-stride adapter requires one complete fixed-body invocation"; return failure(); }
    auto loop = dyn_cast<scf::ForOp>(selected->anchor);
    auto domain = CountedLoop::get(loop);
    if (!domain) { error = "mixed-stride counted domain unavailable"; return failure(); }
    PhaseIndex index;
    if (failed(index.build(function, input))) { error = "mixed-stride phase index unavailable"; return failure(); }
    RecognitionResult outside;
    for (auto& op : function.front()) {
        if (&op == loop.getOperation()) { continue; }
        detail::inspectLeaf(op, index, outside);
        if (op.getNumRegions()) { error = "mixed-stride invocation has an unmodeled outer region"; return failure(); }
    }
    if (outside.state != RecognitionState::Applicable) {
        error = "mixed-stride invocation has unmodeled outside effects"; return failure();
    }
    auto contract = mixedContract(*selected->rotatingResult);
    auto primitives = collectRotatingPrimitives(loop, index, input, *contract);
    if (!primitives.error.empty()) { error = primitives.error; return failure(); }
    auto expanded = expandMixedStride(primitives.payloads, primitives.fragments);
    if (!expanded.error.empty()) { error = expanded.error; return failure(); }
    auto demands = expandMixedStrideRecords(primitives.prerequisites, primitives.payloads.size(), expanded.period);
    auto native = expandMixedStrideRecords(primitives.nativePrerequisites, primitives.payloads.size(), expanded.period);
    if (!demands || !native) { error = "mixed-stride prerequisite phase expansion unavailable"; return failure(); }
    auto extracted = extractRotatingGenerators(expanded.payloads, expanded.fragments, ptoStorageProtection());
    if (!extracted.error.empty()) { error = extracted.error; return failure(); }
    llvm::append_range(extracted.generators, *demands);
    auto periodic = analyzePeriodicDemands(expanded.payloads, extracted.generators, *native);
    if (!periodic.error.empty()) { error = periodic.error; return failure(); }
    EndpointBuilder endpoints(*domain, primitives.phases, expanded.period);
    CircuitEndpoints emit(function, endpoints.arena, endpoints.anchors);
    for (auto id : periodic.retained) {
        if (id >= periodic.generators.size() || !endpoints.add(emit, periodic.generators[id])) {
            error = "mixed-stride phase endpoint unavailable: " + endpoints.arena.lastEmissionError();
            return failure();
        }
    }
    auto plan = emit.take();
    // The only source coordinate is the superperiod ordinal. The original
    // record selector is a namespace operand, not another dynamic coordinate.
    plan->nestedIdentities = false;
    plan->completeInvocation = !primitives.phases.empty();
    plan->allocationCertificate = remapCertificate(periodic, function.getContext());
    return plan;
}
} // namespace mlir::pto::frontiersynch
