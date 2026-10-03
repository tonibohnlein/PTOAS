// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Invocation-local, opt-in cost accounting. Nested scopes charge exclusive time;
// attempts separately report inclusive latency and must not be added to stages.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COSTLEDGER_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COSTLEDGER_H
#include <array>
#include <chrono>
#include <cstdint>
namespace mlir::pto::frontiersynch {
enum class CostStage {
    Effects,
    Recognition,
    Backend,
    Merge,
    Selectors,
    Insertion,
    Allocation,
    Lowering,
    Diagnostics,
    Count
};
struct StageCost {
    int64_t nanoseconds = 0;
    int64_t invocations = 0;
};
class CostScope;
class CostLedger {
public:
    explicit CostLedger(bool enabled = false) : enabled(enabled) {}
    CostLedger(const CostLedger&) = delete;
    CostLedger& operator=(const CostLedger&) = delete;
    bool active() const { return enabled; }
    const StageCost& stage(CostStage value) const { return stages[static_cast<unsigned>(value)]; }

private:
    friend class CostScope;
    bool enabled = false;
    CostScope* current = nullptr;
    std::array<StageCost, static_cast<unsigned>(CostStage::Count)> stages{};
};
class CostScope {
public:
    CostScope(CostLedger& ledger, CostStage stage) : ledger(ledger), kind(stage)
    {
        if (ledger.active()) {
            parent = ledger.current;
            ledger.current = this;
            start = Clock::now();
            ++ledger.stages[static_cast<unsigned>(kind)].invocations;
        }
    }
    ~CostScope()
    {
        if (ledger.active()) {
            auto elapsed = nanoseconds();
            ledger.stages[static_cast<unsigned>(kind)].nanoseconds += elapsed - children;
            ledger.current = parent;
            if (parent) {
                parent->children += elapsed;
            }
        }
    }
    CostScope(const CostScope&) = delete;
    CostScope& operator=(const CostScope&) = delete;
    int64_t nanoseconds() const
    {
        return ledger.active() ? std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count() : 0;
    }

private:
    using Clock = std::chrono::steady_clock;
    CostLedger& ledger;
    CostStage kind;
    CostScope* parent = nullptr;
    Clock::time_point start{};
    int64_t children = 0;
};
} // namespace mlir::pto::frontiersynch
#endif
