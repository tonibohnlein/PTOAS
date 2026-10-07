// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Serialization consumes a typed proof; it never reconstructs endpoint reuse.
#include "PTO/Transforms/FrontierSynch/GeneralArithmeticAllocation.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
DictionaryAttr executedCertificate(const ArithmeticHandoffAllocation& proof, ArrayRef<uint32_t> pipes,
    const std::map<std::pair<std::size_t, std::size_t>, int64_t>& records, int64_t plan, MLIRContext* context)
{
    Builder b(context); SmallVector<Attribute> families;
    std::set<int64_t> seen;
    uint64_t total = 0;
    for (const auto& family : proof.families) {
        auto record = records.find({family.sourceSite, family.targetSite});
        if (family.sourceSite >= pipes.size() || family.targetSite >= pipes.size() ||
            !family.width || family.width > 6 || total > INT64_MAX - family.width ||
            record == records.end() || record->second < 0 ||
            !seen.insert(record->second).second) { return {}; }
        total += family.width;
        families.push_back(b.getDictionaryAttr({
            b.getNamedAttr("record", b.getI64IntegerAttr(record->second)),
            b.getNamedAttr("source", b.getI64IntegerAttr(pipes[family.sourceSite])),
            b.getNamedAttr("target", b.getI64IntegerAttr(pipes[family.targetSite])),
            b.getNamedAttr("width", b.getI64IntegerAttr(family.width))}));
    }
    return b.getDictionaryAttr({b.getNamedAttr("version", b.getI64IntegerAttr(2)),
        b.getNamedAttr("plan", b.getI64IntegerAttr(plan)), b.getNamedAttr("scope", b.getStringAttr("function")),
        b.getNamedAttr("strategy", b.getStringAttr("executed-family-counters")),
        b.getNamedAttr("families", b.getArrayAttr(families)),
        b.getNamedAttr("total_budget", b.getI64IntegerAttr(total))});
}
} // namespace
DictionaryAttr encodeGeneralArithmeticAllocationCertificate(
    const ArithmeticHandoffAllocation& proof, ArrayRef<uint32_t> pipes,
    const std::map<std::pair<std::size_t, std::size_t>, int64_t>& records, int64_t plan, MLIRContext* context)
{
    if (!context || !proof.error.empty()) { return {}; }
    if (llvm::any_of(proof.families, [](const auto& family) { return family.width > 1; })) {
        return executedCertificate(proof, pipes, records, plan, context);
    }
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> groups;
    std::set<uint32_t> used;
    for (const auto& family : proof.families) {
        if (family.width != 1 || family.sourceSite >= pipes.size() || family.targetSite >= pipes.size()) { return {}; }
        auto record = records.find({family.sourceSite, family.targetSite});
        if (record == records.end() || record->second < 0 || static_cast<uint64_t>(record->second) > UINT32_MAX) {
            return {};
        }
        auto id = static_cast<uint32_t>(record->second);
        if (!used.insert(id).second) { return {}; }
        groups[{pipes[family.sourceSite], pipes[family.targetSite]}].push_back(id);
    }
    PeriodicAllocation allocation;
    for (const auto& [direction, ids] : groups) {
        DirectedAllocation group;
        group.sourcePipe = direction.first; group.targetPipe = direction.second;
        group.uniformBudget = ids.size();
        for (auto id : ids) {
            DirectedHandoff handoff; handoff.record = id;
            group.handoffs.push_back(handoff);
        }
        allocation.directions.push_back(std::move(group));
    }
    NamedAttrList attributes(encodeCyclicAllocation(allocation, plan, context));
    attributes.set("strategy", StringAttr::get(context, "dedicated-families"));
    return attributes.getDictionary(context);
}
DictionaryAttr generalArithmeticAllocationCertificate(
    const GeneralArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes,
    const std::map<std::pair<std::size_t, std::size_t>, int64_t>& records, int64_t plan, MLIRContext* context)
{
    auto proof = buildArithmeticHandoffAllocation(analysis, pipes, 6);
    return encodeGeneralArithmeticAllocationCertificate(proof, pipes, records, plan, context);
}
} // namespace mlir::pto::frontiersynch
