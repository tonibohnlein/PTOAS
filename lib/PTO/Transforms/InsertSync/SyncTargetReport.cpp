// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Optional serialization and durable profile identity use the already recovered
// shared records. No external documents are read during compilation.
#include "PTO/Transforms/InsertSync/SyncTargetInfo.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::pto {
namespace {
StringRef coreName(SyncPhysicalCore core)
{
    switch (core) {
    case SyncPhysicalCore::AIC: return "aic";
    case SyncPhysicalCore::AIV: return "aiv";
    case SyncPhysicalCore::Conflict: return "conflict";
    case SyncPhysicalCore::Unknown: return "unknown";
    }
    return "unknown";
}
ArrayAttr terminatorDescriptors(Builder& builder, const DenseMap<Operation*, int64_t>& ids)
{
    SmallVector<std::pair<int64_t, Operation*>> terminals;
    for (const auto& entry : ids) {
        if (entry.first->hasTrait<OpTrait::IsTerminator>()) {
            terminals.emplace_back(entry.second, entry.first);
        }
    }
    llvm::sort(terminals, [](const auto& a, const auto& b) { return a.first < b.first; });
    SmallVector<Attribute> descriptors;
    for (const auto& entry : terminals) {
        Operation* op = entry.second;
        unsigned blockIndex = 0;
        for (Block& block : *op->getParentRegion()) {
            if (&block == op->getBlock()) {
                break;
            }
            ++blockIndex;
        }
        descriptors.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("source", builder.getI64IntegerAttr(entry.first)),
            builder.getNamedAttr("owner", builder.getI64IntegerAttr(ids.lookup(op->getParentOp()))),
            builder.getNamedAttr("region", builder.getI64IntegerAttr(op->getParentRegion()->getRegionNumber())),
            builder.getNamedAttr("block", builder.getI64IntegerAttr(blockIndex)),
            builder.getNamedAttr("operation", builder.getStringAttr(op->getName().getStringRef()))}));
    }
    return builder.getArrayAttr(descriptors);
}
} // namespace
DictionaryAttr SyncTargetInfo::activationAttribute() const
{
    if (!function) { return {}; }
    Builder builder(function->getContext());
    const auto& witness = participation().witness;
    NamedAttrList attributes;
    attributes.set("origin", builder.getStringAttr(witness.origin == SyncActivationOrigin::Argument ? "argument" :
        witness.origin == SyncActivationOrigin::ComparisonResult ? "comparison-result" : "none"));
    attributes.set("argument", builder.getI64IntegerAttr(witness.argument ?
        static_cast<int64_t>(cast<BlockArgument>(witness.argument).getArgNumber()) : -1));
    attributes.set("result_source", builder.getI64IntegerAttr(witness.comparison ?
        sourceIds.lookup(witness.comparison) : -1));
    attributes.set("result_index", builder.getI64IntegerAttr(witness.comparison ? 0 : -1));
    if (witness.comparison) {
        attributes.set("predicate", builder.getI64IntegerAttr(witness.predicate));
        attributes.set("argument_first", builder.getBoolAttr(witness.argumentFirst));
        attributes.set("constant_source", builder.getI64IntegerAttr(sourceIds.lookup(witness.constantSource)));
        attributes.set("constant", witness.constant);
    }
    return attributes.getDictionary(function->getContext());
}
DictionaryAttr SyncTargetInfo::attribute() const
{
    if (!function) {
        return {};
    }
    Builder builder(function->getContext());
    NamedAttrList result;
    result.set("schema", builder.getStringAttr("shared-sync-target-v1"));
    if (participation().owner) {
        result.set("common_participation", builder.getDictionaryAttr({
            builder.getNamedAttr("source", builder.getI64IntegerAttr(sourceIds.lookup(participation().owner))),
            builder.getNamedAttr("activation_binding", activationAttribute())}));
    }
    if (originalChain.barrier) {
        SmallVector<int64_t> computeSources;
        for (const auto* phase : originalChain.computes) {
            computeSources.push_back(sourceIds.lookup(phase->elementOp));
        }
        auto memoryIndices = [&](ArrayRef<std::pair<const BaseMemInfo*, const BaseMemInfo*>> pairs,
                                 ArrayRef<const BaseMemInfo*> sources, ArrayRef<const BaseMemInfo*> targets) {
            SmallVector<int64_t> indices;
            for (const auto& pair : pairs) {
                indices.push_back(llvm::find(sources, pair.first) - sources.begin());
                indices.push_back(llvm::find(targets, pair.second) - targets.begin());
            }
            return builder.getDenseI64ArrayAttr(indices);
        };
        SmallVector<Attribute> lexical;
        for (const auto& witness : originalChain.lexical) {
            const bool war = witness.kind == SyncBarrierHazard::WriteAfterRead;
            const bool raw = witness.kind == SyncBarrierHazard::ReadAfterWrite;
            const auto& sources = war ? witness.source->useVec : witness.source->defVec;
            const auto& targets = raw ? witness.consumer->useVec : witness.consumer->defVec;
            lexical.push_back(builder.getDictionaryAttr({
                builder.getNamedAttr("source", builder.getI64IntegerAttr(sourceIds.lookup(witness.source->elementOp))),
                builder.getNamedAttr("consumer",
                    builder.getI64IntegerAttr(sourceIds.lookup(witness.consumer->elementOp))),
                builder.getNamedAttr("hazard", builder.getStringAttr(war ? "WAR" : raw ? "RAW" : "WAW")),
                builder.getNamedAttr("original_memory_vector_indices",
                    memoryIndices(witness.pairs, sources, targets))}));
        }
        SmallVector<int64_t> frameSources;
        for (auto* frame : originalChain.frames) { frameSources.push_back(sourceIds.lookup(frame)); }
        result.set("original_barrier_chain", builder.getDictionaryAttr({
            builder.getNamedAttr("barrier_source", builder.getI64IntegerAttr(sourceIds.lookup(originalChain.barrier))),
            builder.getNamedAttr("loop_source", builder.getI64IntegerAttr(sourceIds.lookup(originalChain.loop))),
            builder.getNamedAttr("frame_sources", builder.getDenseI64ArrayAttr(frameSources)),
            builder.getNamedAttr("producer_source",
                builder.getI64IntegerAttr(sourceIds.lookup(originalChain.producer->elementOp))),
            builder.getNamedAttr("consumer_source",
                builder.getI64IntegerAttr(sourceIds.lookup(originalChain.consumer->elementOp))),
            builder.getNamedAttr("release_producer_source",
                builder.getI64IntegerAttr(sourceIds.lookup(originalChain.releaseProducer->elementOp))),
            builder.getNamedAttr("compute_sources", builder.getDenseI64ArrayAttr(computeSources)),
            builder.getNamedAttr("lexical_witnesses", builder.getArrayAttr(lexical)),
            builder.getNamedAttr("ready_memory_vector_indices", memoryIndices(originalChain.ready,
                originalChain.producer->defVec, originalChain.consumer->useVec)),
            builder.getNamedAttr("release_memory_vector_indices", memoryIndices(originalChain.release,
                originalChain.releaseProducer->useVec, originalChain.producer->defVec)),
            builder.getNamedAttr("final_memory_vector_indices", memoryIndices(originalChain.finalReads,
                originalChain.releaseProducer->defVec, records.back().phase->useVec)),
            builder.getNamedAttr("prefix_source", builder.getI64IntegerAttr(originalChain.prefix ?
                sourceIds.lookup(originalChain.prefix->elementOp) : -1)),
            builder.getNamedAttr("participation_source", builder.getI64IntegerAttr(originalChain.participationContext ?
                sourceIds.lookup(participation().owner) : -1)),
            builder.getNamedAttr("activation_binding", originalChain.participationContext ? activationAttribute() :
                builder.getDictionaryAttr({builder.getNamedAttr("origin", builder.getStringAttr("none"))})),
            builder.getNamedAttr("activation_argument", builder.getI64IntegerAttr(
                originalChain.participationContext && participation().witness.origin == SyncActivationOrigin::Argument ?
                static_cast<int64_t>(cast<BlockArgument>(participation().condition).getArgNumber()) : -1)),
            builder.getNamedAttr("pipe", builder.getStringAttr("MTE2")),
            builder.getNamedAttr("core", builder.getStringAttr("aiv")),
            builder.getNamedAttr("requirement_domain", builder.getStringAttr(
                originalChain.requirementDomain == SyncBarrierRequirementDomain::SharedUnion ?
                    "shared-union" : "unqualified")),
            builder.getNamedAttr("native_proof",
                builder.getStringAttr("original drain and gate, independent of inserted flags")),
            builder.getNamedAttr("model_equivalence",
                builder.getStringAttr("original union READY/lexical/RELEASE plus captured final read "
                                      "implies original barrier projected payload prerequisites"))}));
    }
    SmallVector<Attribute> finiteBaselines;
    for (const auto& drain : originalFiniteDrains) {
        finiteBaselines.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("barrier_source", builder.getI64IntegerAttr(sourceIds.lookup(drain.barrier))),
            builder.getNamedAttr("previous_source",
                builder.getI64IntegerAttr(sourceIds.lookup(drain.previous->elementOp))),
            builder.getNamedAttr("consumer_source",
                builder.getI64IntegerAttr(sourceIds.lookup(drain.consumer->elementOp))),
            builder.getNamedAttr("core", builder.getStringAttr(coreName(drain.core))),
            builder.getNamedAttr("pipe", builder.getI64IntegerAttr(static_cast<unsigned>(drain.consumer->kPipeValue))),
            builder.getNamedAttr("domain", builder.getStringAttr("original-command mandatory baseline"))}));
    }
    result.set("original_finite_drains", builder.getArrayAttr(finiteBaselines));
    result.set("architecture", builder.getStringAttr(arch));
    auto effective = getTargetArch(function.operator->());
    result.set("compiler_effective_architecture", builder.getStringAttr(effective == PTOArch::A5 ? "a5" : "a3"));
    result.set("id_pools", builder.getStringAttr(
        "R1 classic identity=(physical core,directed template,numeric ID); IDs0-5 minus shared hidden "
        "reservations; IDs6/7 reserved; no aggregate-six pool; other profiles unmet"));
    result.set("invocation_abi",
        builder.getStringAttr("closed-event ABI: used directed flags clear on entry, matched consumption and "
                              "terminal PIPE_ALL on exit; compatible external header/runtime binding required"));
    result.set("source_id_namespace", builder.getStringAttr("pto.frontier.source"));
    result.set("source_id_order", builder.getStringAttr("mlir-postorder-v1"));
    result.set("function_source", builder.getI64IntegerAttr(sourceIds.lookup(function.operator->())));
    const auto& ids = sourceIds;
    // SCF custom assembly can elide implicit terminators and their attributes.
    // Owner/region/block coordinates preserve their identity across text roundtrips.
    result.set("source_terminators", terminatorDescriptors(builder, ids));
    SmallVector<Attribute> phases;
    for (const auto& record : records) {
        NamedAttrList phase;
        phase.set("index", builder.getI64IntegerAttr(record.phase->GetIndex()));
        phase.set("anchor", builder.getI64IntegerAttr(ids.lookup(record.phase->elementOp)));
        phase.set("pipe", builder.getI64IntegerAttr(static_cast<unsigned>(record.phase->kPipeValue)));
        phase.set("core", builder.getStringAttr(coreName(record.context.core)));
        phase.set("shared_core_type", builder.getI64IntegerAttr(static_cast<unsigned>(record.phase->compoundCoreType)));
        phase.set("reads", builder.getI64IntegerAttr(record.phase->useVec.size()));
        phase.set("writes", builder.getI64IntegerAttr(record.phase->defVec.size()));
        NamedAttrList controls;
        for (auto named : record.phase->elementOp->getAttrs()) {
            std::string text;
            llvm::raw_string_ostream os(text);
            named.getValue().print(os);
            controls.set(named.getName(), builder.getStringAttr(os.str()));
        }
        phase.set("lowering_controls", controls.getDictionary(function->getContext()));
        phase.set("macro_phase", builder.getI64IntegerAttr(record.phase->macroOpInstanceId));
        phase.set("anchor_phase_count", builder.getI64IntegerAttr(record.anchorPhaseCount));
        phase.set("external_before", builder.getI64IntegerAttr(ids.lookup(record.before)));
        phase.set("external_after", builder.getI64IntegerAttr(record.after ? ids.lookup(record.after) : -1));
        SmallVector<Attribute> guards, events;
        for (auto* region : record.context.regions) {
            guards.push_back(builder.getDictionaryAttr({
                builder.getNamedAttr("owner", builder.getI64IntegerAttr(ids.lookup(region->getParentOp()))),
                builder.getNamedAttr("region", builder.getI64IntegerAttr(region->getRegionNumber()))}));
        }
        for (const auto& event : record.hiddenEvents) {
            SmallVector<Attribute> eventIds;
            for (auto id : event.eventIds) {
                eventIds.push_back(builder.getI64IntegerAttr(id));
            }
            events.push_back(builder.getDictionaryAttr({
                builder.getNamedAttr("source_pipe", builder.getI64IntegerAttr(static_cast<unsigned>(event.srcPipe))),
                builder.getNamedAttr("consumer_pipe", builder.getI64IntegerAttr(static_cast<unsigned>(event.dstPipe))),
                builder.getNamedAttr("modeled_ids", builder.getArrayAttr(eventIds))}));
        }
        phase.set("execution_regions", builder.getArrayAttr(guards));
        phase.set("hidden_events", builder.getArrayAttr(events));
        phases.push_back(phase.getDictionary(function->getContext()));
    }
    result.set("phases", builder.getArrayAttr(phases));
    return result.getDictionary(function->getContext());
}
} // namespace mlir::pto
