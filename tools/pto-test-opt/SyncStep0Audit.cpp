// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Validate Step 0 against the shared translator and original MLIR interfaces.
// The step1 mode uses production pipe envelopes and stops before recognition.
// This checks preservation, not native footprint completeness or F* readiness.
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/AsmState.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
using namespace mlir::pto;

namespace {
bool sameDeclaration(const SyncMemoryEffect& a, const SyncMemoryEffect& b)
{
    return a.getEffect() == b.getEffect() && a.getValue() == b.getValue() &&
        a.getEffectValue<OpOperand*>() == b.getEffectValue<OpOperand*>() &&
        a.getResource() == b.getResource() && a.getParameters() == b.getParameters() &&
        a.getStage() == b.getStage() && a.getEffectOnFullRegion() == b.getEffectOnFullRegion() &&
        a.getSymbolRef() == b.getSymbolRef();
}

bool sameLoops(const SyncAccessRegion& region, Operation* anchor)
{
    SmallVector<scf::ForOp> loops;
    for (auto* parent = anchor->getParentOp(); parent; parent = parent->getParentOp()) {
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            loops.push_back(loop);
        }
    }
    std::reverse(loops.begin(), loops.end());
    if (loops.size() != region.iterations.size()) {
        return false;
    }
    for (auto [loop, saved] : llvm::zip(loops, region.iterations)) {
        if (saved.induction != loop.getInductionVar() || saved.lower != loop.getLowerBound() ||
            saved.upper != loop.getUpperBound() || saved.step != loop.getStep()) {
            return false;
        }
    }
    return true;
}

std::string printed(AffineExpr expression)
{
    std::string text;
    llvm::raw_string_ostream stream(text);
    expression.print(stream);
    return text;
}
llvm::json::Object valueIdentity(Value value, AsmState& state)
{
    if (!value) { return {}; }
    std::string name, location;
    llvm::raw_string_ostream names(name), locations(location);
    value.printAsOperand(names, state);
    auto argument = dyn_cast<BlockArgument>(value);
    auto* owner = argument ? argument.getOwner()->getParentOp() : value.getDefiningOp();
    owner->getLoc().print(locations);
    return llvm::json::Object{{"ssa", name}, {"location", location}, {"block_argument", bool(argument)},
            {"number", argument ? argument.getArgNumber() : cast<OpResult>(value).getResultNumber()}};
}
llvm::json::Object regionDescription(const SyncAccessRegion& region, AsmState& state)
{
    llvm::json::Array extents, symbols, loops;
    for (auto extent : region.extents) { extents.push_back(printed(extent)); }
    for (auto symbol : region.symbols) { symbols.push_back(valueIdentity(symbol, state)); }
    for (const auto& loop : region.iterations) {
        loops.push_back(llvm::json::Object{{"induction", valueIdentity(loop.induction, state)},
            {"lower", valueIdentity(loop.lower, state)}, {"upper", valueIdentity(loop.upper, state)},
            {"step", valueIdentity(loop.step, state)}});
    }
    return llvm::json::Object{{"base", valueIdentity(region.base, state)}, {"offset", printed(region.byteOffset)},
            {"element_bytes", region.elementBytes}, {"extents", std::move(extents)},
            {"symbols", std::move(symbols)}, {"loops", std::move(loops)}};
}
llvm::json::Array effectDescriptions(func::FuncOp function, const SyncInput& input)
{
    const auto& storage = input.accesses();
    AsmState state(function);
    llvm::json::Array result;
    for (auto [id, effect] : llvm::enumerate(storage.effects())) {
        const auto& memory = *effect.memory;
        std::string location;
        llvm::raw_string_ostream stream(location);
        effect.phase->elementOp->getLoc().print(stream);
        llvm::json::Array regions, ranges;
        for (const auto& region : effect.regions) { regions.push_back(regionDescription(region, state)); }
        for (const auto& range : effect.ranges) {
            ranges.push_back(llvm::json::Object{{"begin", range.begin}, {"end", range.end},
                                               {"base", valueIdentity(range.base, state)}});
        }
        unsigned selectedDeclarations = 0;
        for (const auto& declaration : input.effectsFor(*effect.phase)) {
            auto parameters = dyn_cast_or_null<DictionaryAttr>(declaration.getParameters());
            const bool sameMode = effect.mode == SyncAccessMode::Read ?
                isa<MemoryEffects::Read>(declaration.getEffect()) : isa<MemoryEffects::Write>(declaration.getEffect());
            const bool selection = sameMode && declaration.getValue() == memory.baseBuffer && parameters &&
                                   parameters.get("pto.access_region");
            if (selection) { ++selectedDeclarations; }
        }
        llvm::json::Object entry{{"declared_selections", selectedDeclarations}, {"id", id},
            {"operation", effect.phase->opName.getStringRef()},
            {"location", location}, {"pipe", static_cast<unsigned>(effect.phase->kPipeValue)},
            {"mode", effect.mode == SyncAccessMode::Read ? "read" : "write"},
            {"operand", valueIdentity(memory.baseBuffer, state)}, {"root", valueIdentity(memory.rootBuffer, state)},
            {"space", static_cast<unsigned>(memory.scope)}, {"allocation_bytes", memory.allocateSize},
            {"known_physical_address", memory.hasKnownPhysicalAddresses},
            {"ranges_materialized", effect.rangesMaterialized},
            {"regions", std::move(regions)}, {"ranges", std::move(ranges)}};
        if (effect.descriptorRegion) { entry["descriptor"] = regionDescription(*effect.descriptorRegion, state); }
        result.push_back(std::move(entry));
    }
    return result;
}

struct Audit {
    const SyncInput& input;
    const frontiersynch::PhaseIndex& index;
    const SyncStorageEffects& storage;
    bool valid = true;
    int64_t declarations = 0, unphased = 0, operations = 0, expected = 0;
    int64_t controlled = 0, descriptors = 0, symbolic = 0, selections = 0, planned = 0;
    int64_t intervals = 0, maps = 0, unresolved = 0;
    llvm::json::Object unphasedKinds, pipes;
    int64_t loops = 0, branches = 0, prerequisites = 0;

    void operation(Operation* op)
    {
        ++operations;
        loops += isa<scf::ForOp>(op);
        branches += isa<scf::IfOp>(op);
        prerequisites += index.prerequisitesFor(op).size();
        auto interface = dyn_cast<MemoryEffectOpInterface>(op);
        if (!interface) {
            return;
        }
        SmallVector<SyncMemoryEffect> original;
        interface.getEffects(original);
        auto saved = input.effectsFor(op);
        if (original.size() != saved.size()) {
            valid = false;
            return;
        }
        for (auto [a, b] : llvm::zip(original, saved)) {
            valid &= sameDeclaration(a, b);
        }
        declarations += saved.size();
        if (!saved.empty() && index.phasesFor(op).empty()) {
            unphased += saved.size();
            auto name = op->getName().getStringRef();
            unphasedKinds[name] = unphasedKinds.getInteger(name).value_or(0) + int64_t(saved.size());
        }
    }

    void effect(const SyncStorageEffect& effect)
    {
        planned += effect.memory->hasKnownPhysicalAddresses;
        intervals += effect.rangesMaterialized;
        maps += !effect.rangesMaterialized && !effect.regions.empty();
        unresolved += !effect.rangesMaterialized && effect.regions.empty();
        if (effect.descriptorRegion) {
            ++descriptors;
            symbolic += !effect.descriptorRegion->symbols.empty() || bool(effect.descriptorRegion->base);
            valid &= sameLoops(*effect.descriptorRegion, effect.phase->elementOp);
            for (Value symbol : effect.descriptorRegion->symbols) {
                valid &= index.valueAvailable(symbol, effect.phase->elementOp, frontiersynch::Boundary::Before);
            }
        }
        if (effect.selection) {
            ++selections;
            valid &= bool(effect.selection->selector) && !effect.selection->addresses.empty();
        }
    }

    void phase(const CompoundInstanceElement* phase)
    {
        valid &= llvm::is_contained(index.phasesFor(phase->elementOp), phase);
        SmallVector<Region*> sourcePath;
        for (auto* region = phase->elementOp->getParentRegion();
             region && !isa<func::FuncOp>(region->getParentOp()); region = region->getParentRegion()) {
            if (isa<RegionBranchOpInterface>(region->getParentOp())) { sourcePath.push_back(region); }
        }
        std::reverse(sourcePath.begin(), sourcePath.end());
        valid &= sourcePath == index.controlPath(*phase);
        const auto pipe = std::to_string(static_cast<unsigned>(phase->kPipeValue));
        pipes[pipe] = pipes.getInteger(pipe).value_or(0) + 1;
        expected += phase->useVec.size() + phase->defVec.size();
        controlled += !index.controlPath(*phase).empty();
        auto effects = storage.effectsFor(phase);
        if (effects.size() != phase->useVec.size() + phase->defVec.size()) {
            valid = false;
            return;
        }
        for (std::size_t i = 0; i < effects.size(); ++i) {
            const auto& record = storage.effects()[effects[i]];
            bool read = i < phase->useVec.size();
            auto* memory = read ? phase->useVec[i] : phase->defVec[i - phase->useVec.size()];
            valid &= record.phase == phase && record.memory == memory &&
                record.mode == (read ? SyncAccessMode::Read : SyncAccessMode::Write);
            effect(record);
        }
    }

    void emit(func::FuncOp function, bool envelopes)
    {
        llvm::json::Object result{
            {"function", function.getSymName()}, {envelopes ? "step1" : "step0", "preserved"},
            {"instruction_view", envelopes ? "pipe-envelopes" : "translator-stages"},
            {"loops", loops}, {"branches", branches}, {"value_prerequisites", prerequisites},
            {"pipes", std::move(pipes)},
            {"operations", operations}, {"phases", int64_t(input.instructions().size())},
            {"effects", expected}, {"declarations", declarations},
            {"unphased_declarations", unphased}, {"unphased_kinds", std::move(unphasedKinds)},
            {"controlled_phases", controlled}, {"planned_effects", planned},
            {"descriptor_maps", descriptors}, {"symbolic_maps", symbolic},
            {"slot_selections", selections}, {"materialized_accesses", intervals},
            {"symbolic_accesses", maps}, {"unresolved_accesses", unresolved}
        };
        if (envelopes) { result["access_records"] = effectDescriptions(function, input); }
        llvm::outs() << llvm::json::Value(std::move(result)) << "\n";
    }
};
} // namespace

LogicalResult auditSyncStep0(func::FuncOp function, const SyncInput& input, bool envelopes)
{
    frontiersynch::PhaseIndex index;
    const auto& storage = input.accesses();
    if (failed(index.build(function, input))) {
        return failure();
    }
    Audit audit{input, index, storage};
    function.walk([&](Operation* op) { audit.operation(op); });
    for (const auto* phase : input.instructions()) {
        audit.phase(phase);
    }
    if (!audit.valid || audit.expected != int64_t(storage.effects().size())) {
        return function.emitError("Step 0 lost shared access metadata or occurrence context");
    }
    audit.emit(function, envelopes);
    return success();
}
