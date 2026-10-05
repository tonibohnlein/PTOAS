// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Serialize the analysis certificate; allocate IDs within each directed event domain.
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "PTO/IR/PTO.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include <limits>
namespace mlir::pto::frontiersynch {
DictionaryAttr encodeCyclicAllocation(const PeriodicAllocation& allocation, int64_t planId, MLIRContext* context)
{
    Builder builder(context);
    SmallVector<Attribute> directions;
    for (const auto& direction : allocation.directions) {
        SmallVector<int64_t> records;
        for (const auto& handoff : direction.handoffs) {
            records.push_back(handoff.record);
        }
        int64_t budget = -1; // No representable finite uniform capacity certificate.
        if (direction.uniformBudget && *direction.uniformBudget <= INT64_MAX) {
            budget = static_cast<int64_t>(*direction.uniformBudget);
        }
        directions.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("source", builder.getI64IntegerAttr(direction.sourcePipe)),
            builder.getNamedAttr("target", builder.getI64IntegerAttr(direction.targetPipe)),
            builder.getNamedAttr("budget", builder.getI64IntegerAttr(budget)),
            builder.getNamedAttr("records", builder.getDenseI64ArrayAttr(records))}));
    }
    return builder.getDictionaryAttr({builder.getNamedAttr("version", builder.getI64IntegerAttr(1)),
        builder.getNamedAttr("plan", builder.getI64IntegerAttr(planId)),
        builder.getNamedAttr("directions", builder.getArrayAttr(directions))});
}
namespace {
std::optional<int64_t> number(DictionaryAttr dictionary, StringRef name)
{
    auto attr = dictionary.getAs<IntegerAttr>(name);
    if (!attr || !attr.getType().isInteger(64)) {
        return std::nullopt;
    }
    return attr.getInt();
}
bool concretePipe(int64_t value)
{
    return value >= 0 && value <= static_cast<int64_t>(PIPE::PIPE_FIX) &&
        value != static_cast<int64_t>(PIPE::PIPE_ALL);
}
LogicalResult validateIds(func::FuncOp function, ArrayRef<int64_t> ids)
{
    llvm::DenseSet<int64_t> seen;
    for (auto id : ids) {
        if (id < 0 || id > UINT32_MAX || !symbolizeEVENT(static_cast<uint32_t>(id))) {
            return function.emitError("eligible event ID is outside the native ID range: ") << id;
        }
        // Static-Tensor programming uses 0..5; 6/7 are reserved for the
        // system/framework even though both values exist in the native enum.
        constexpr int64_t staticEventIdCount = 6;
        if (id >= staticEventIdCount) {
            return function.emitError("event ID is reserved for system/framework use; available IDs are 0..5: ") << id;
        }
        if (!seen.insert(id).second) {
            return function.emitError("duplicate eligible event ID: ") << id;
        }
    }
    return success();
}
struct Direction {
    int64_t source = 0;
    int64_t target = 0;
    int64_t budget = 0;
    DenseI64ArrayAttr records;
};
FailureOr<Direction> decodeDirection(func::FuncOp function, Attribute attr)
{
    auto dictionary = dyn_cast<DictionaryAttr>(attr);
    if (!dictionary) {
        return function.emitError("malformed cyclic allocation direction"), failure();
    }
    auto source = number(dictionary, "source"), target = number(dictionary, "target");
    auto budget = number(dictionary, "budget");
    auto records = dictionary.getAs<DenseI64ArrayAttr>("records");
    if (!source || !target || !budget || !records || records.empty() ||
        !concretePipe(*source) || !concretePipe(*target) || *source == *target) {
        return function.emitError("malformed cyclic allocation direction"), failure();
    }
    if (*budget <= 0) {
        return function.emitError("no finite positive uniform event-ID budget certified for direction ")
            << *source << " -> " << *target, failure();
    }
    return Direction{*source, *target, *budget, records};
}
} // namespace
FailureOr<PhysicalAllocationPlan> decodeCyclicAllocation(func::FuncOp function, ArrayRef<int64_t> eligibleIds)
{
    if (failed(validateIds(function, eligibleIds))) {
        return failure();
    }
    auto certificate = function->getAttrOfType<DictionaryAttr>(CyclicAllocationAttr);
    if (!certificate) {
        return function.emitError("physical allocation requires a cyclic allocation certificate"), failure();
    }
    auto version = number(certificate, "version"), plan = number(certificate, "plan");
    auto directions = certificate.getAs<ArrayAttr>("directions");
    if (!version || *version != 1 || !plan || *plan < 0 || !directions) {
        return function.emitError("malformed cyclic allocation certificate"), failure();
    }
    PhysicalAllocationPlan result;
    result.planId = *plan;
    llvm::DenseSet<uint64_t> directionKeys;
    llvm::DenseSet<int64_t> recordKeys;
    for (auto attr : directions) {
        auto direction = decodeDirection(function, attr);
        if (failed(direction)) {
            return failure();
        }
        const auto& d = *direction;
        const uint64_t key = (static_cast<uint64_t>(d.source) << 32) | static_cast<uint64_t>(d.target);
        if (!directionKeys.insert(key).second) {
            return function.emitError("duplicate cyclic allocation direction"), failure();
        }
        const auto budget = static_cast<uint64_t>(d.budget);
        if (budget > eligibleIds.size()) {
            return function.emitError("directed event-ID allocation does not fit: direction ")
                << d.source << " -> " << d.target << " needs " << budget << ", only "
                << eligibleIds.size() << " eligible IDs are available per direction; no repair applied", failure();
        }
        // The hardware event is identified by (source, destination, numeric ID).
        // Equal numbers in different directions are distinct events, so budgets
        // are checked independently; they must not be summed against this list.
        auto ids = eligibleIds.take_front(budget);
        for (auto [phase, record] : llvm::enumerate(d.records.asArrayRef())) {
            if (record < 0 || !recordKeys.insert(record).second) {
                return function.emitError("invalid or duplicate cyclic allocation record"), failure();
            }
            PhysicalRecordAllocation item{record, static_cast<uint32_t>(d.source), static_cast<uint32_t>(d.target),
                                           static_cast<uint64_t>(d.records.size()), phase, {}};
            item.ids.append(ids.begin(), ids.end());
            result.records.push_back(std::move(item));
        }
    }
    return result;
}
} // namespace mlir::pto::frontiersynch
