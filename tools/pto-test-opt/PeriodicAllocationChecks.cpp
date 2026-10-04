// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Bounded finite-prefix queries for independent causal-lifetime test oracles.
#include "PTO/Transforms/FrontierSynch/PeriodicAllocation.h"
#include "llvm/Support/JSON.h"
namespace fs = mlir::pto::frontiersynch;
llvm::json::Object dumpPeriodicAllocation(const fs::PeriodicAllocation& result);
namespace {
llvm::json::Value value(const fs::AllocationNumber& result)
{
    return result.value ? llvm::json::Value(*result.value) : llvm::json::Value(nullptr);
}
bool integers(const llvm::json::Object& input, llvm::StringRef key, std::vector<uint64_t>& result)
{
    const auto* array = input.getArray(key);
    if (!array) {
        return true;
    }
    if (array->size() > 128) {
        return false;
    }
    for (const auto& item : *array) {
        auto number = item.getAsUINT64();
        if (!number) {
            return false;
        }
        result.push_back(*number);
    }
    return true;
}
bool offsets(const llvm::json::Object& input, const fs::PeriodicAllocation& allocation,
             llvm::json::Object& output)
{
    const auto* queries = input.getArray("offset_queries");
    if (!queries) {
        return true;
    }
    if (queries->size() > 128) {
        return false;
    }
    llvm::json::Array answers;
    for (const auto& item : *queries) {
        const auto* tuple = item.getAsArray();
        if (!tuple || tuple->size() != 4) {
            return false;
        }
        uint64_t values[4] = {};
        for (unsigned i = 0; i < 4; ++i) {
            auto number = (*tuple)[i].getAsUINT64();
            if (!number) {
                return false;
            }
            values[i] = *number;
        }
        if (values[0] >= allocation.directions.size() || values[1] > UINT32_MAX) {
            return false;
        }
        const auto answer = allocation.directions[values[0]].localOffset(values[1], values[2], values[3]);
        answers.push_back(llvm::json::Object{{"error", static_cast<unsigned>(answer.error)}, {"value", value(answer)}});
    }
    output["offset_answers"] = std::move(answers);
    return true;
}
bool contractChecks()
{
    const std::vector<fs::PeriodicPayload> pipes{{0}, {0}, {1}, {1}};
    const std::vector<fs::PeriodicRecord> records{{0, 2, 0}, {1, 3, 0}};
    const auto original = fs::analyzePeriodicDemands(pipes, records);
    if (!original.error.empty() || original.retained.size() != 2) {
        return false;
    }
    auto duplicateSource = original, duplicateTarget = original, reversed = original, wrap = original;
    duplicateSource.generators[1].source = 0;
    duplicateTarget.generators[1].target = 2;
    reversed.generators[0].target = 3;
    reversed.generators[1].target = 2;
    wrap.generators[1].displacement = 2;
    for (const auto* invalid : {&duplicateSource, &duplicateTarget, &reversed, &wrap}) {
        const auto rejected = fs::buildPeriodicAllocation(*invalid);
        if (rejected.error.empty() || !rejected.directions.empty()) {
            return false;
        }
    }
    // Exact threshold index of the four-type graph with readiness 0->2,1->3
    // and return 2->1 at q-1 periods, q=(UINT64_MAX-1)/2. Its scaled distances
    // are known in closed form; the graph builder's conservative overflow check
    // can reject unused cyclic relaxations, so isolate the allocation arithmetic.
    auto wide = original;
    const uint64_t q = UINT64_MAX / 2;
    wide.generators.push_back({2, 1, q - 1});
    wide.retained.push_back(2);
    wide.frontiers[0].distances = {UINT64_MAX, 1, UINT64_MAX - 2, 0, 1, 1, 0, 0};
    wide.frontiers[1].distances = {UINT64_MAX, UINT64_MAX, UINT64_MAX - 2, UINT64_MAX - 2,
                                  UINT64_MAX, 1, UINT64_MAX - 2, 0};
    const auto allocation = fs::buildPeriodicAllocation(wide);
    if (!allocation.error.empty() || allocation.directions.size() != 2) {
        return false;
    }
    const auto& forward = allocation.directions[0];
    // The phase-one, s=0 candidate multiplies to UINT64_MAX+1 before
    // subtracting r=1; it must not invalidate the smaller representable result.
    return forward.handoffs.size() == 2 && forward.handoffs[0].firstReuse == UINT64_MAX - 2 &&
        forward.handoffs[1].firstReuse == UINT64_MAX - 1 && forward.uniformBudget == UINT64_MAX - 1;
}
} // namespace
bool appendAllocationChecks(const llvm::json::Object& input, const fs::PeriodicAnalysis& analysis,
                            llvm::json::Object& output)
{
    if (!input.getBoolean("allocation").value_or(false)) {
        return true;
    }
    const auto allocation = fs::buildPeriodicAllocation(analysis);
    auto result = dumpPeriodicAllocation(allocation);
    if (input.getBoolean("allocation_contracts").value_or(false)) {
        result["contract_checks"] = contractChecks();
    }
    std::vector<uint64_t> prefixes, capacities;
    if (!integers(input, "allocation_prefixes", prefixes) || !integers(input, "capacities", capacities)) {
        return false;
    }
    llvm::json::Array checks;
    if (allocation.error.empty()) {
        for (const auto& direction : allocation.directions) {
            llvm::json::Array finite, uniform;
            for (auto capacity : capacities) {
                const auto sufficient = direction.capacitySuffices(capacity);
                uniform.push_back(llvm::json::Object{{"error", static_cast<unsigned>(sufficient.error)},
                                                    {"sufficient", sufficient.sufficient}});
            }
            for (auto prefix : prefixes) {
                const auto count = direction.handoffCount(prefix), budget = direction.finiteBudget(prefix);
                llvm::json::Array fits;
                for (auto capacity : capacities) {
                    const auto sufficient = direction.capacitySuffices(capacity, prefix);
                    fits.push_back(llvm::json::Object{{"error", static_cast<unsigned>(sufficient.error)},
                                                    {"sufficient", sufficient.sufficient}});
                }
                finite.push_back(llvm::json::Object{{"prefix", prefix}, {"handoffs", value(count)},
                    {"count_error", static_cast<unsigned>(count.error)}, {"budget", value(budget)},
                    {"budget_error", static_cast<unsigned>(budget.error)}, {"fits", std::move(fits)}});
            }
            checks.push_back(llvm::json::Object{{"finite", std::move(finite)}, {"uniform", std::move(uniform)}});
        }
        if (!offsets(input, allocation, result)) {
            return false;
        }
    }
    result["checks"] = std::move(checks);
    output["allocation"] = std::move(result);
    return true;
}
