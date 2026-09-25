// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALCOVERINGQUERIES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALCOVERINGQUERIES_H
#include "PTO/Transforms/FrontierSynch/CoveringBoundary.h"
#include "PTO/Transforms/FrontierSynch/OriginalValueQueries.h"
#include <memory>

namespace mlir::pto::frontiersynch {
class OriginalLifetimes;
struct CoveringControlQualification {
    std::size_t owner = NoControlId;
    OriginalValueQualification qualification;
};
struct OriginalCoveringBoundary : CoveringBoundary {
    // The new endpoint guard is true. A counted-loop proof may separately name
    // original typed completion prerequisites; they remain obligations, not
    // acquired credit or a guard circularly enabling its own computation.
    OriginalValueQualification endpointQualification;
    std::vector<CoveringControlQualification> controlQualifications;
};
// I.3 only. Independent of exact frontiers, request grouping, descriptors,
// subscriptions and packet binding. Owns one shared, immutable query index.
class OriginalCoveringQueries {
public:
    OriginalCoveringQueries(const OriginalStructure&, const OriginalLifetimes&, const OriginalValueQueries&);
    ~OriginalCoveringQueries();
    OriginalCoveringQueries(const OriginalCoveringQueries&) = delete;
    OriginalCoveringQueries& operator=(const OriginalCoveringQueries&) = delete;
    OriginalCoveringBoundary source(const OriginalInterval&) const;
    OriginalCoveringBoundary target(const OriginalInterval&) const;
    const CoveringBoundaryStats& stats() const;
private:
    OriginalCoveringBoundary query(const OriginalInterval&, CoveringDirection) const;
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace mlir::pto::frontiersynch
#endif
