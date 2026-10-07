// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exercise the temporary phase copies used by the loop-backedge traversal.
#ifndef PTO_TEST_SYNC_PHASE_COPY_CHECKS_H
#define PTO_TEST_SYNC_PHASE_COPY_CHECKS_H
#include "PTO/Transforms/InsertSync/SyncStorageEffects.h"

inline bool checkPhaseCopies(const mlir::pto::SyncInput& input)
{
    using namespace mlir::pto;
    const auto& storage = input.accesses();
    llvm::SmallVector<std::unique_ptr<CompoundInstanceElement>> copies;
    for (const auto* phase : input.instructions()) {
        auto copy = std::make_unique<CompoundInstanceElement>(phase->GetIndex(), phase->defVec,
            phase->useVec, phase->kPipeValue, phase->opName);
        copy->elementOp = phase->elementOp;
        copy->compoundCoreType = phase->compoundCoreType;
        // Deliberately use the same fields as InsertBackForSync. An operation
        // pointer alone must not merge distinct macro phases.
        if (storage.effectsFor(copy.get()) != storage.effectsFor(phase)) {
            return false;
        }
        copies.push_back(std::move(copy));
    }
    for (std::size_t i = 0; i < copies.size(); ++i) {
        for (std::size_t j = 0; j < copies.size(); ++j) {
            for (auto leftMode : {SyncAccessMode::Read, SyncAccessMode::Write}) {
                for (auto rightMode : {SyncAccessMode::Read, SyncAccessMode::Write}) {
                    DepBaseMemInfoPairVec original, cloned, mixed;
                    bool expected = storage.dependencies(input.instructions()[i], leftMode,
                        input.instructions()[j], rightMode, original);
                    bool actual = storage.dependencies(copies[i].get(), leftMode,
                        copies[j].get(), rightMode, cloned);
                    bool partial = storage.dependencies(input.instructions()[i], leftMode,
                        copies[j].get(), rightMode, mixed);
                    if (expected != actual || expected != partial || original != cloned || original != mixed) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}
// The legacy pass must keep every original macro stage and its effect lookup.
// Frontier's view is the exact per-call/per-pipe union of those same records.
inline bool checkMacroEnvelopeView(mlir::func::FuncOp function,
                                  const mlir::pto::SyncInput& legacy)
{
    using namespace mlir::pto;
    SyncInput envelopes(legacy.memory().gmPolicy());
    if (mlir::failed(envelopes.build(function, SyncInstructionView::PipeEnvelopes)) ||
        legacy.ir().size() != envelopes.ir().size()) { return false; }
    for (std::size_t i = 0; i < legacy.ir().size(); ++i) {
        auto* before = llvm::dyn_cast<CompoundInstanceElement>(legacy.ir()[i].get());
        auto* after = llvm::dyn_cast<CompoundInstanceElement>(envelopes.ir()[i].get());
        if (bool(before) != bool(after)) { return false; }
        if (!before) { continue; }
        if (before->defVec.size() != after->defVec.size() || before->useVec.size() != after->useVec.size() ||
            before->GetIndex() != after->GetIndex() || before->macroOpInstanceId != after->macroOpInstanceId) {
            return false;
        }
        for (auto* memory : before->defVec) {
            if (!llvm::any_of(legacy.accesses().effectsFor(before), [&](auto id) {
                    return legacy.accesses().effects()[id].memory == memory &&
                        legacy.accesses().effects()[id].mode == SyncAccessMode::Write;
                })) { return false; }
        }
        for (auto* memory : before->useVec) {
            if (!llvm::any_of(legacy.accesses().effectsFor(before), [&](auto id) {
                    return legacy.accesses().effects()[id].memory == memory &&
                        legacy.accesses().effects()[id].mode == SyncAccessMode::Read;
                })) { return false; }
        }
    }
    for (const auto* phase : envelopes.instructions()) {
        llvm::SmallVector<mlir::Value> writes, reads;
        for (const auto* raw : legacy.instructions()) {
            if (raw->elementOp != phase->elementOp || raw->kPipeValue != phase->kPipeValue) { continue; }
            for (auto* memory : raw->defVec) { writes.push_back(memory->baseBuffer); }
            for (auto* memory : raw->useVec) { reads.push_back(memory->baseBuffer); }
        }
        auto same = [](auto& values, auto& memories) {
            return llvm::all_of(values, [&](auto value) {
                return llvm::any_of(memories, [&](auto* memory) { return memory->baseBuffer == value; });
            }) && llvm::all_of(memories, [&](auto* memory) { return llvm::is_contained(values, memory->baseBuffer); });
        };
        if (!same(writes, phase->defVec) || !same(reads, phase->useVec)) { return false; }
    }
    return checkPhaseCopies(envelopes);
}
#endif
