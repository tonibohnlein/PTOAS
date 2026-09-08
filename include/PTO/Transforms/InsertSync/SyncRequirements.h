// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCREQUIREMENTS_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCREQUIREMENTS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <memory>

namespace mlir::pto {
struct LocalStorageRequirements;
// Requirements retain payload/SSA identities across construction, placement and
// allocation. The justification is deliberately typed: access disjointness,
// intrinsic ordering, member ownership and full completion are different facts.
// These records are obligations to reconstruct, never verifier certificates.
struct SyncRequirement {
    enum class Kind { SlotDisjoint, GlobalDisjoint, MmadOrder, FullCompletion, Lifecycle, DirectRepair };
    Kind kind;
    Operation *source = nullptr, *target = nullptr;
    Value sourceAccess, targetAccess;
    bool carried = false;
    unsigned channel = ~0u;
    unsigned repairGroup = ~0u;
};

class SyncRequirements {
    SmallVector<SyncRequirement> entries;
    llvm::DenseMap<Operation*, SmallVector<unsigned, 2>> actions;
public:
    // Immutable input requirements. Unlike entries (resolution witnesses), this
    // survives alreadySync, failed recipe selection, and changes of event IDs.
    std::shared_ptr<const LocalStorageRequirements> local;
    void retain(SyncRequirement requirement) {
        bool phaseWide = requirement.kind == SyncRequirement::Kind::FullCompletion ||
                         requirement.kind == SyncRequirement::Kind::MmadOrder;
        if (std::none_of(entries.begin(), entries.end(), [&](const auto& old) {
            return old.kind == requirement.kind && old.source == requirement.source &&
                   old.target == requirement.target && (phaseWide ||
                   (old.sourceAccess == requirement.sourceAccess && old.targetAccess == requirement.targetAccess &&
                    old.carried == requirement.carried && old.channel == requirement.channel &&
                    old.repairGroup == requirement.repairGroup));
        })) entries.push_back(requirement);
    }
    // Codegen binds the stable logical repair group AFTER motion and allocation.
    // Several groups can share an emitted action; ownership never follows an ID
    // guessed from the output. Bindings expire if their action is deleted.
    void bind(unsigned group, Operation* action) {
        if (action && !llvm::is_contained(actions[action], group)) actions[action].push_back(group);
    }
    bool owns(Operation* action) const { return actions.count(action); }
    ArrayRef<unsigned> groups(Operation* action) const {
        auto found = actions.find(action);
        return found == actions.end() ? ArrayRef<unsigned>() : ArrayRef<unsigned>(found->second);
    }
    ArrayRef<SyncRequirement> all() const { return entries; }
    unsigned count(SyncRequirement::Kind kind) const {
        return std::count_if(entries.begin(), entries.end(), [&](const auto& r) { return r.kind == kind; });
    }
};
} // namespace mlir::pto
#endif
