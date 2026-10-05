// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ExplicitAnalysis.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "PTO/Transforms/FrontierSynch/Recognition.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
void boundary(ExplicitAnalysis& result)
{
    std::unordered_map<uint32_t, std::size_t> cells;
    for (const auto& occurrence : result.occurrences) {
        std::unordered_map<uint32_t, CellAccess> modes;
        for (const auto& access : occurrence.accesses) {
            auto& mode = modes[access.atom];
            mode.read |= access.read;
            mode.write |= access.write;
        }
        for (const auto& [atom, mode] : modes) {
            auto [position, inserted] = cells.emplace(atom, result.storageBoundary.size());
            if (inserted) {
                ExplicitCellBoundary cell;
                cell.atom = atom;
                result.storageBoundary.push_back(std::move(cell));
            }
            auto& cell = result.storageBoundary[position->second];
            if (mode.write) {
                if (!cell.firstWriter) {
                    cell.firstWriter = occurrence.payload;
                }
                cell.lastWriter = occurrence.payload;
                std::unordered_map<uint32_t, uint32_t>().swap(cell.lastReaders);
            } else if (mode.read) {
                if (!cell.firstWriter) {
                    cell.firstReaders.emplace(occurrence.pipe, occurrence.payload);
                }
                cell.lastReaders[occurrence.pipe] = occurrence.payload;
            }
        }
    }
}
} // namespace
ExplicitAnalysis analyzeExplicit(Block& block, const PhaseIndex& index, const SyncInput& input)
{
    ExplicitAnalysis result;
    auto recognized = recognizeExplicit(block, index, input.accesses());
    auto sequence = index.explicitSequence(block);
    if (recognized.state != RecognitionState::Applicable || failed(sequence)) {
        result.error = "explicit route requires resolved control and exact enumerated physical effects";
        return result;
    }
    if (sequence->size() > std::numeric_limits<uint32_t>::max() ||
        input.accesses().cells().size() > std::numeric_limits<uint32_t>::max()) {
        result.error = "explicit occurrence or cell identity overflow";
        return result;
    }
    result.phases = *sequence;
    HardwareProtectionBuilder protection;
    for (const auto* phase : result.phases) {
        ExplicitEffects occurrence;
        occurrence.payload = static_cast<uint32_t>(result.occurrences.size());
        occurrence.pipe = static_cast<uint32_t>(phase->kPipeValue);
        SmallVector<uint32_t> accumulatorAtoms;
        for (auto id : input.accesses().effectsFor(phase)) {
            const auto& effect = input.accesses().effects()[id];
            for (auto cell : effect.cells) {
                const auto atom = static_cast<uint32_t>(cell);
                occurrence.accesses.push_back({atom, effect.mode == SyncAccessMode::Read,
                                               effect.mode == SyncAccessMode::Write});
                if (input.accesses().cells()[cell].space == AddressSpace::ACC) {
                    accumulatorAtoms.push_back(atom);
                }
            }
        }
        protection.observe(phase->elementOp, occurrence, accumulatorAtoms);
        result.occurrences.push_back(std::move(occurrence));
    }
    result.scan = scanStorageLifetimes(result.occurrences);
    if (!result.scan.error.empty()) {
        result.error = result.scan.error;
        return result;
    }
    result.reduction = reduceExplicitDemands(result.occurrences, result.scan.generators);
    result.error = result.reduction.error;
    if (result.error.empty()) {
        boundary(result);
    }
    return result;
}
FailureOr<std::unique_ptr<PreparedLogicalPlan>> prepareExplicitInsertion(
    func::FuncOp function, const ExplicitAnalysis& analysis)
{
    if (!analysis.error.empty() || function.isDeclaration() || !llvm::hasSingleElement(function.getBody())) {
        return failure();
    }
    auto plan = std::make_unique<PreparedLogicalPlan>(0);
    plan->completeInvocation = !analysis.phases.empty();
    if (analysis.reduction.retained.empty()) {
        return plan;
    }
    auto* first = &function.getBody().front().front();
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToEnd(&plan->addPreparation(first));
    auto enabled = builder.create<arith::ConstantIntOp>(first->getLoc(), 1, 1);
    auto identity = builder.create<arith::ConstantIndexOp>(first->getLoc(), 0);
    int64_t record = 0;
    for (const auto& edge : analysis.reduction.retained) {
        const auto sourcePipe = analysis.occurrences[edge.source].pipe;
        const auto targetPipe = analysis.occurrences[edge.target].pipe;
        auto* source = analysis.phases[edge.source]->elementOp;
        auto* target = analysis.phases[edge.target]->elementOp;
        if (source->getBlock() != &function.getBody().front() || target->getBlock() != source->getBlock() ||
            !source->getNextNode()) {
            return failure();
        }
        if (sourcePipe == targetPipe) {
            if (analysis.reduction.localRanks[edge.target] != analysis.reduction.localRanks[edge.source] + 1) {
                function.emitError("explicit minimum demand violates adjacent same-pipe insertion contract");
                return failure();
            }
            plan->endpoints.push_back({target, LogicalCommandKind::Barrier, sourcePipe, targetPipe,
                                       record, enabled, {}});
        } else {
            plan->endpoints.push_back({source->getNextNode(), LogicalCommandKind::Set, sourcePipe, targetPipe,
                                       record, enabled, identity});
            plan->endpoints.push_back({target, LogicalCommandKind::Wait, sourcePipe, targetPipe,
                                       record, enabled, identity});
        }
        ++record;
    }
    return plan;
}
} // namespace mlir::pto::frontiersynch
