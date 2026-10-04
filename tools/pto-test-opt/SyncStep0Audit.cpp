// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Validate Step 0 against the shared translator and original MLIR interfaces.
// This checks preservation, not native footprint completeness or F* readiness.
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

struct Audit {
    const SyncInput& input;
    const frontiersynch::PhaseIndex& index;
    const SyncStorageEffects& storage;
    bool valid = true;
    int64_t declarations = 0, unphased = 0, operations = 0, expected = 0;
    int64_t controlled = 0, descriptors = 0, symbolic = 0, selections = 0, planned = 0;
    int64_t exact = 0, upper = 0, unknown = 0;
    llvm::json::Object unphasedKinds;

    void operation(Operation* op)
    {
        ++operations;
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
        exact += effect.precision == SyncAccessPrecision::Exact;
        upper += effect.precision == SyncAccessPrecision::UpperBound;
        unknown += effect.precision == SyncAccessPrecision::Unknown;
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

    void emit(func::FuncOp function)
    {
        llvm::json::Object result{
            {"function", function.getSymName()}, {"step0", "preserved"},
            {"operations", operations}, {"phases", int64_t(input.instructions().size())},
            {"effects", expected}, {"declarations", declarations},
            {"unphased_declarations", unphased}, {"unphased_kinds", std::move(unphasedKinds)},
            {"controlled_phases", controlled}, {"planned_effects", planned},
            {"descriptor_maps", descriptors}, {"symbolic_maps", symbolic},
            {"slot_selections", selections}, {"exact_accesses", exact},
            {"upper_bound_accesses", upper}, {"unknown_accesses", unknown}
        };
        llvm::outs() << llvm::json::Value(std::move(result)) << "\n";
    }
};
} // namespace

LogicalResult auditSyncStep0(func::FuncOp function, const SyncInput& input)
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
    audit.emit(function);
    return success();
}
