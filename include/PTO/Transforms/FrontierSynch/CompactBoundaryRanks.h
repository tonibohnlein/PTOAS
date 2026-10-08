// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTBOUNDARYRANKS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTBOUNDARYRANKS_H
#include "PTO/Transforms/FrontierSynch/BoundaryExcess.h"
#include "PTO/Transforms/FrontierSynch/PeriodicExcessCertificate.h"
namespace mlir::pto::frontiersynch {
class CompactBoundaryRanks {
public:
    const PeriodicExcessSnapshot& upperSnapshot() const { return upperGraph; }
    const PeriodicExcessSnapshot& lowerSnapshot() const { return lowerGraph; }
    uint64_t trips() const { return tripCount; }
    llvm::ArrayRef<RegionalEvent> ports() const { return events; }
    llvm::ArrayRef<uint32_t> pipeLabels() const { return labels; }
    llvm::ArrayRef<uint64_t> counts() const { return sizes; }
    const std::vector<bool>& present() const { return presence; }
    const BoundaryExcessSelectors& upper() const { return upperSelectors; }
    const BoundaryExcessSelectors& lower() const { return lowerSelectors; }
    uint64_t thresholdQueries() const { return queries; }
    uint64_t frontierEntries() const { return entries; }
private:
    CompactBoundaryRanks() = default;
    PeriodicExcessSnapshot upperGraph, lowerGraph;
    uint64_t tripCount = 0, queries = 0, entries = 0;
    std::vector<RegionalEvent> events;
    std::vector<uint32_t> labels;
    std::vector<uint64_t> sizes;
    std::vector<bool> presence;
    BoundaryExcessSelectors upperSelectors, lowerSelectors;
    friend std::shared_ptr<const CompactBoundaryRanks> captureCompactBoundaryRanks(
        PeriodicExcessSnapshot, PeriodicExcessSnapshot, uint64_t, llvm::ArrayRef<RegionalEvent>, std::string&);
};
// Evaluate two immutable numerical graphs on ONE qualified fixed-body domain at
// the supplied numerical trip count. This instantiates the domain's exact trip
// circuit; it does not prove a particular runtime parameter valuation has that
// count. Both graphs must be exact uniformly in that domain, as required by their
// private snapshot binding. The caller's lower/input/upper sandwich is unchanged.
// Ports use constant ordinal roots in the shared arena and no outer visits.
// An absent port has lambda=0 and etaExcluded=n on every pipe. Present ports use
// reflexive event reachability, including Completion ports and Start ports.
// Counts h*T must fit uint64; etaExcluded avoids an overflowing n+1 sentinel.
// After existing snapshot indexing, time O(m*k + g + P*(m+k)), space O(P*k+m+k).
// Work is independent of T and binary displacement magnitudes. Frontier checks
// and threshold queries are charged separately from rectangle-union costs.
// No input mutation, guard replay, endpoint placement or hardware claim is made.
std::shared_ptr<const CompactBoundaryRanks> captureCompactBoundaryRanks(
    PeriodicExcessSnapshot upper, PeriodicExcessSnapshot lower, uint64_t trips,
    llvm::ArrayRef<RegionalEvent> ports, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
