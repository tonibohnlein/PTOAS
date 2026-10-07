// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Numerical universal queries over a certified affine boundary sequence.
#ifndef PTO_FRONTIERSYNCH_VARYINGBOUNDARYQUERY_H
#define PTO_FRONTIERSYNCH_VARYINGBOUNDARYQUERY_H
#include "PTO/Transforms/FrontierSynch/RotatingBoundary.h"
#include <map>
namespace mlir::pto::frontiersynch {
struct VaryingBoundaryPort {
    BoundaryOccurrence occurrence;
    PeriodicEventKind kind;
};
using VaryingBoundaryMatrix = std::vector<std::vector<uint8_t>>;
// Borrows the immutable certificate and transfer matrices owned by the regional
// state. Startup[k] crosses k->k+1. Suffix[p] crosses phase p->p+1 relative to
// startup; powers[p][b] crosses 2^b complete suffix periods from phase p.
class VaryingBoundaryQuery {
public:
    VaryingBoundaryQuery(const AffineRotatingVisits& visits, const std::vector<VaryingBoundaryPort>& ports,
        const std::vector<VaryingBoundaryMatrix>& startup, const std::vector<VaryingBoundaryMatrix>& suffix,
        const std::vector<std::vector<VaryingBoundaryMatrix>>& powers);
    // Short sourceVisit is exact. Long sourceVisit represents every visit in
    // its certified suffix residue class. Invalid or nonconstant attachment
    // relations return unavailable rather than extrapolating a sample.
    std::optional<bool> query(BoundaryOccurrence source, PeriodicEventKind sourceKind,
        BoundaryOccurrence target, PeriodicEventKind targetKind, uint64_t sourceVisit, uint64_t gap);
    const std::string& error() const { return constructionError; }
    uint64_t queryCount = 0, thresholdQueries = 0, matrixApplications = 0;
private:
    const AffineRotatingVisits& visits;
    const std::vector<VaryingBoundaryPort>& ports;
    const std::vector<VaryingBoundaryMatrix>& startup;
    const std::vector<VaryingBoundaryMatrix>& suffix;
    const std::vector<std::vector<VaryingBoundaryMatrix>>& powers;
    std::string constructionError;
    using Key = std::tuple<uint32_t, uint64_t, bool, unsigned, uint32_t, uint64_t, bool, unsigned, uint64_t, uint64_t>;
    std::map<Key, std::optional<bool>> cache;
    bool valid(VaryingBoundaryPort port) const;
    std::optional<uint64_t> length(uint64_t visit) const;
    std::optional<bool> inside(VaryingBoundaryPort source, VaryingBoundaryPort target, uint64_t visit);
    bool advance(std::vector<uint8_t>& row, const VaryingBoundaryMatrix& matrix);
    std::optional<bool> compute(VaryingBoundaryPort source, VaryingBoundaryPort target,
                               uint64_t sourceVisit, uint64_t gap);
};
} // namespace mlir::pto::frontiersynch
#endif
