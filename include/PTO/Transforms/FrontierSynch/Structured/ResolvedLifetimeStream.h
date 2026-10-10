// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_RESOLVEDLIFETIMESTREAM_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_RESOLVEDLIFETIMESTREAM_H
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <string>
#include <vector>
namespace mlir::pto::frontiersynch {
struct ResolvedLifetimeStep {
    std::string error;
    uint64_t rank = 0;
    std::vector<uint64_t> start, completion, retained;
};
struct ResolvedLifetimeCost {
    uint64_t occurrences = 0, candidates = 0, retained = 0, native = 0, rowEntries = 0;
};
// Executes an already constructed exact forward generator stream. Candidates
// name source occurrence IDs in strictly decreasing reference order. Native
// prerequisites seed the start row and are never emitted as demands. The caller
// certifies the generating set, fixed pipe assignment and uniform span bound.
// ID = iteration*sites + site; all IDs and ranks fit the declared invocation.
// Absent sites may be skipped entirely. Optional explicit absent calls cost
// O(1) each and change no native row; they return no profile or demands.
//
// Beyond candidate construction/sorting, n present calls and g candidates use
// O(n+g+nk+k*retained+k*native) word operations. With no extra native edges this
// is the resolved streaming theorem bound. State uses O((m(R+1)+k)k) words;
// output profiles belong to the caller. No expression history or T rows remain.
class ResolvedLifetimeStream {
public:
    ResolvedLifetimeStream(llvm::ArrayRef<uint32_t> pipes, uint32_t span, uint64_t maximumIterations);
    const std::string& error() const { return constructionError; }
    const ResolvedLifetimeCost& cost() const { return costs; }
    std::size_t historySlots() const { return history.size(); }
    std::size_t columns() const { return counters.size(); }
    ResolvedLifetimeStep advance(uint64_t occurrence, bool present,
        llvm::ArrayRef<uint64_t> candidates = {}, llvm::ArrayRef<uint64_t> native = {});
private:
    struct Slot {
        uint64_t stamp = UINT64_MAX, rank = 0;
        std::vector<uint64_t> completion;
    };
    std::string constructionError;
    uint32_t span = 0;
    uint64_t limit = 0, last = 0;
    bool started = false;
    std::vector<uint32_t> siteColumns;
    std::vector<uint64_t> counters;
    std::vector<std::vector<uint64_t>> starts, completions;
    std::vector<Slot> history;
    ResolvedLifetimeCost costs;
};
} // namespace mlir::pto::frontiersynch
#endif
