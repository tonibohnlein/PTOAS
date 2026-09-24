// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License. Original physical/control records extracted from OAHS Plan.h. The translated phases and
// source IR are borrowed from SyncInput and must outlive this result.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALSTRUCTURE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALSTRUCTURE_H
#include "PTO/Transforms/FrontierSynch/OriginalProgramPoints.h"
#include "PTO/Transforms/FrontierSynch/SyncTileDescriptorState.h"
#include "PTO/Transforms/FrontierSynch/WriteCoverage.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <limits>
#include <string>
#include <vector>

namespace mlir::pto::frontiersynch {
struct Access {
    std::size_t cell = 0;
    bool read = false, write = false;
    // Full-cell overwrite requires a separate effect-coverage proof.
    bool definiteWrite = false;
    // Original translated effect identity; the cell alone does not identify
    // which independent physical selector produced this use.
    const BaseMemInfo* memory = nullptr;
    // The specific qualified address relation for this translated effect.
    std::size_t physicalRelation = NoControlId;
    // Independent of read/write modes and retained even when the current cell
    // is too coarse to receive a strong update. Never a completion certificate.
    WriteCoverage coverage = {};
};
struct Cell {
    bool exclusive = false;
    std::string addressSpace;
    std::string provenance;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    bool unknownRange = false;
    enum class Storage { Abstract, CanonicalInterval, OverlapWitness };
    Storage storage = Storage::Abstract;
    std::string coordinateSpace;
    std::vector<std::size_t> storageOrigins;
};
struct PhysicalOperation {
    const CompoundInstanceElement* instruction = nullptr;
    std::size_t original = NoControlId;
    std::vector<Access> accesses;
    // Only the outer boundaries of a multi-phase original instruction are
    // directly executable in the unchanged IR. Internal phase cuts need a
    // separate lowering contract before they can host synchronization.
    bool beforeExecutable = false, afterExecutable = false;
    std::size_t enclosingAfter = NoControlId;
};
// An address relation records physical alternatives only. Neither its finite
// period nor its root identity establishes a predecessor-use correspondence.
struct PhysicalAddressRelation {
    std::size_t owner = NoControlId;
    Value address;
    const BaseMemInfo* memory = nullptr;
    std::vector<SmallVector<uint64_t>> addresses;
};
struct OriginalStructure {
    func::FuncOp function;
    Region body;
    std::vector<PhysicalOperation> operations;
    std::vector<Cell> cells;
    std::vector<mlir::Operation*> originalSites;
    std::vector<Value> storageRoots;
    std::vector<PhysicalAddressRelation> physicalAddresses;
    std::unique_ptr<SyncTileDescriptorState> descriptors;
    OriginalProgramVersion version = OriginalProgramVersion::fresh();
};
// A realizable ORIGINAL IR gap. `before` is a real operation in `block`;
// region exits use the original terminator, not a fabricated post-terminator gap.
// No mutation, selected command offset, or guard-availability claim is involved.
struct OriginalInsertionPoint {
    Block* block = nullptr;
    mlir::Operation* before = nullptr;
};
inline std::optional<OriginalInsertionPoint> resolveOriginalCut(
    const OriginalStructure& original, const OriginalCut& cut)
{
    auto before = [](mlir::Operation* op) -> std::optional<OriginalInsertionPoint> {
        if (!op || !op->getBlock()) {
            return {};
        }
        return OriginalInsertionPoint{op->getBlock(), op};
    };
    auto regionBoundary = [&](mlir::Region& region) -> std::optional<OriginalInsertionPoint> {
        if (!region.hasOneBlock() || region.front().empty()) {
            return {};
        }
        auto& block = region.front();
        auto* anchor = cut.side == OriginalCut::Before ? &block.front() : &block.back();
        if (cut.side == OriginalCut::After && !anchor->hasTrait<OpTrait::IsTerminator>()) {
            return {};
        }
        return before(anchor);
    };
    if (cut.side != OriginalCut::Before && cut.side != OriginalCut::After) {
        return {};
    }
    if (cut.kind == OriginalCut::Kind::Payload) {
        if (cut.operation >= original.operations.size() || cut.owner != NoControlId || cut.child != NoControlId) {
            return {};
        }
        const auto& phase = original.operations[cut.operation];
        if (!phase.instruction || !(cut.side == OriginalCut::Before ? phase.beforeExecutable : phase.afterExecutable)) {
            return {};
        }
        auto* op = phase.instruction->elementOp;
        if (!op || op->hasTrait<OpTrait::IsTerminator>()) {
            return {};
        }
        return before(cut.side == OriginalCut::Before ? op : op->getNextNode());
    }
    if (cut.operation != NoControlId) {
        return {};
    }
    if (cut.kind == OriginalCut::Kind::Scope && cut.child == NoControlId && cut.owner == NoControlId) {
        if (!original.function) {
            return {};
        }
        auto function = original.function;
        return regionBoundary(function.getBody());
    }
    if (cut.owner >= original.originalSites.size()) {
        return {};
    }
    auto* op = original.originalSites[cut.owner];
    if (!op || !isa<scf::IfOp, scf::ForOp, scf::WhileOp>(op)) {
        return {};
    }
    if (cut.kind == OriginalCut::Kind::Scope && cut.child == NoControlId) {
        return before(cut.side == OriginalCut::Before ? op : op->getNextNode());
    }
    if (cut.kind == OriginalCut::Kind::Child && cut.child < op->getNumRegions()) {
        return regionBoundary(op->getRegion(cut.child));
    }
    return {};
}
// Requires verified original IR. Failure leaves the caller's result unchanged.
LogicalResult importOriginalStructure(func::FuncOp function, const SyncInput& input, OriginalStructure& result);
} // namespace mlir::pto::frontiersynch
#endif
