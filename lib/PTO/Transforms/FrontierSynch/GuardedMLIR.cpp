// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/GuardedAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace mlir::pto::frontiersynch {
namespace {
class OccurrenceCollector {
public:
    explicit OccurrenceCollector(const PhaseIndex& index) : index(index) {}
    LogicalResult collect(Region& region, Predicate guard)
    {
        if (region.empty()) {
            return success();
        }
        if (!region.hasOneBlock()) {
            return region.getParentOp()->emitError("guarded demand analysis requires single-block regions");
        }
        for (Operation& operation : region.front()) {
            if (failed(collectOperation(operation, guard))) { return failure(); }
        }
        return success();
    }
    LogicalResult collectOperation(Operation& operation, Predicate guard)
    {
        if (auto branch = dyn_cast<scf::IfOp>(operation)) {
            auto condition = predicates.atom(branch.getCondition());
            if (failed(condition) || !index.phasesFor(&operation).empty()) {
                return operation.emitError("invalid shared phase or condition at scf.if");
            }
            if (failed(collect(branch.getThenRegion(), predicates.conjunction(guard, *condition))) ||
                failed(collect(
                    branch.getElseRegion(), predicates.conjunction(guard, predicates.negate(*condition))))) {
                return failure();
            }
            return success();
        }
        if (operation.getNumRegions() != 0) {
            return operation.emitError("guarded demand analysis supports loop-free scf.if/else only");
        }
        auto phases = index.phasesFor(&operation);
        if (phases.size() > 1) {
            return operation.emitError("guarded demand analysis requires single-phase operation anchors");
        }
        llvm::append_range(sequence, phases);
        for (std::size_t count = 0; count < phases.size(); ++count) {
            occurrences.push_back(guard);
        }
        return success();
    }
    PredicateArena predicates;
    SmallVector<const CompoundInstanceElement*> sequence;
    SmallVector<Predicate> occurrences;

private:
    const PhaseIndex& index;
};
} // namespace
LogicalResult GuardedDemandAnalysis::build(func::FuncOp source, const SyncInput& input, const StorageAnalysis& storage)
{
    *this = GuardedDemandAnalysis();
    PhaseIndex index;
    if (failed(index.build(source, input))) {
        return failure();
    }
    return build(source, index, input.instructions(), storage);
}
LogicalResult GuardedDemandAnalysis::build(
    Operation* scope, const PhaseIndex& index, ArrayRef<const CompoundInstanceElement*> phases,
    const StorageAnalysis& storage)
{
    *this = GuardedDemandAnalysis();
    if (!scope) { return failure(); }
    OccurrenceCollector collector(index);
    Predicate guard = 1;
    SmallVector<Region*> enclosing;
    for (Region* region = isa<func::FuncOp>(scope) ? nullptr : scope->getParentRegion();
         region && !isa<func::FuncOp>(region->getParentOp());
         region = region->getParentRegion()) { enclosing.push_back(region); }
    for (Region* region : llvm::reverse(enclosing)) {
        auto branch = dyn_cast<scf::IfOp>(region->getParentOp());
        if (!branch) { return failure(); }
        auto condition = collector.predicates.atom(branch.getCondition());
        if (failed(condition)) { return failure(); }
        guard = collector.predicates.conjunction(guard, region == &branch.getThenRegion() ? *condition :
            collector.predicates.negate(*condition));
    }
    auto collected = isa<func::FuncOp>(scope) ? collector.collect(scope->getRegion(0), guard) :
                                               collector.collectOperation(*scope, guard);
    if (failed(collected)) { return failure(); }
    if (ArrayRef<const CompoundInstanceElement*>(collector.sequence) != phases) {
        return failure();
    }
    return build(collector.sequence, collector.occurrences, storage.footprints(), storage.aliases(),
                 std::move(collector.predicates));
}
} // namespace mlir::pto::frontiersynch
