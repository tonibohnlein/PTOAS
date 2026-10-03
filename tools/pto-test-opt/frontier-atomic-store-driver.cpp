// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Genuine operation effects and RMW incidence; no hardware/native certificate.
#include "PTO/Transforms/FrontierSynch/DemandAnalysis.h"
#include "PTO/Transforms/InsertSync/SyncStorageBounds.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
namespace {
using namespace mlir;
using namespace mlir::pto;
bool checkInterface(TStoreOp store, bool atomic)
{
    SmallVector<MemoryEffects::EffectInstance, 0> effects;
    store.getEffects(effects);
    unsigned sourceReads = 0, destinationReads = 0, destinationWrites = 0;
    for (const auto& effect : effects) {
        auto* operand = effect.getEffectValue<OpOperand*>();
        if (!operand || operand->getOwner() != store.getOperation()) { return false; }
        if (isa<MemoryEffects::Read>(effect.getEffect())) {
            if (operand == &store.getSrcMutable()) { ++sourceReads; }
            else if (operand == &store.getDstMutable()) { ++destinationReads; }
            else { return false; }
        } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
            if (operand != &store.getDstMutable()) { return false; }
            ++destinationWrites;
        } else { return false; }
    }
    return sourceReads == 1 && destinationReads == static_cast<unsigned>(atomic) && destinationWrites == 1;
}
bool checkShared(const frontiersynch::StorageAnalysis& storage,
                 const CompoundInstanceElement* phase, TStoreOp store, bool atomic)
{
    unsigned reads = 0, writes = 0, incidences = 0;
    for (const auto* memory : phase->useVec) { reads += memory->baseBuffer == store.getDst(); }
    for (const auto* memory : phase->defVec) { writes += memory->baseBuffer == store.getDst(); }
    for (const auto& footprint : storage.footprints()) {
        if (footprint.memory->baseBuffer != store.getDst()) { continue; }
        for (const auto& access : footprint.accesses) {
            if (access.phase != phase) { continue; }
            if (access.read != atomic || !access.write) { return false; }
            ++incidences;
        }
    }
    return reads == static_cast<unsigned>(atomic) && writes == 1 &&
           incidences == 1;
}
bool checkRMW(const SyncInput& input, const frontiersynch::StorageAnalysis& storage)
{
    // Actual first three phases: write, RMW, write on one destination.
    // Old-state acquisition must precede writer replacement. The independent
    // modeled oracle requires RAW0->1 then WAW1->2; it supplies no native proof.
    auto sequence = input.instructions().take_front(3);
    frontiersynch::LifetimeAnalysis lifetimes;
    if (failed(lifetimes.build(sequence, storage.footprints(), storage.aliases())) ||
        lifetimes.generators().size() != 2) { return false; }
    bool covered[2] = {false, false};
    for (const auto& demand : lifetimes.generators()) {
        if (demand.source >= 2 || covered[demand.source]) { return false; }
        covered[demand.source] = true;
        if (demand.consumer != demand.source + 1 || demand.witnesses.size() != 1) { return false; }
        auto expected = demand.source == 0 ? frontiersynch::Hazard::RAW : frontiersynch::Hazard::WAW;
        if (demand.witnesses.front().hazard != expected) { return false; }
    }
    return covered[0] && covered[1];
}
} // namespace
int runAtomicStoreChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
    if (!module) { return 1; }
    auto function = module->lookupSymbol<mlir::func::FuncOp>("atomic_store_effects");
    mlir::pto::SyncInput input;
    if (!function || failed(input.build(function)) || input.instructions().size() != 5) { return 1; }
    mlir::pto::frontiersynch::StorageAnalysis storage(input);
    unsigned atomics = 0;
    for (const auto* phase : input.instructions()) {
        auto store = dyn_cast<mlir::pto::TStoreOp>(phase->elementOp);
        if (!store) { return 1; }
        bool atomic = store.getAtomicType() == mlir::pto::AtomicType::AtomicAdd;
        if (!checkInterface(store, atomic) || !checkShared(storage, phase, store, atomic)) { return 1; }
        atomics += atomic;
    }
    if (atomics != 2 || !checkRMW(input, storage)) { return 1; }
    llvm::outs() << "verified AtomicAdd destination read, shared RMW identity/incidence and modeled old-state graph\n";
    return 0;
}
