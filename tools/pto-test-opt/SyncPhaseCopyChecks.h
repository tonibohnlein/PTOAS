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
#endif
