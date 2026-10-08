// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic storage generators without expanding banks or loop iterations.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ROTATINGBOUNDARY_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ROTATINGBOUNDARY_H
#include "PTO/Transforms/FrontierSynch/RotatingExtraction.h"
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
// Explicit boundary cells, not an enumeration of every byte or runtime visit.
struct RotatingBoundaryCell {
    uint32_t family, atom;
    uint64_t slot;
};
struct BoundaryOccurrence {
    uint32_t type = 0;
    uint64_t coordinate = 0; // Head index, or positive distance from the end.
    bool tail = false;
    uint64_t at(uint64_t length) const { return tail ? length - coordinate : coordinate; }
};
struct BoundaryCellSelectors {
    std::optional<BoundaryOccurrence> firstWriter, lastWriter;
    std::map<uint32_t, BoundaryOccurrence> firstReaders, lastReaders;
};
struct RotatingBoundaryType {
    uint64_t representative = 0;
    std::vector<BoundaryCellSelectors> cells;
    std::map<uint32_t, BoundaryOccurrence> firstPayloads, lastPayloads;
    std::vector<BoundaryOccurrence> relationshipPorts;
    NumericalChainInterface index;
    std::map<std::tuple<uint32_t, uint64_t, PeriodicEventKind>, uint32_t> eventIds;
};
struct BoundaryDemand {
    BoundaryOccurrence source, target;
};
struct RotatingBoundaryCertificate {
    std::string error;
    uint64_t refresh = 1, period = 1, cutoff = 1;
    PeriodicAnalysis quotient;
    StorageProtectionPolicy storageProtection;
    std::vector<RotatingFragment> fragments;
    std::vector<RotatingBoundaryCell> cells;
    // Uniform conflict pairs across visits; these are not child prerequisites.
    std::vector<std::pair<uint32_t, uint32_t>> uniformCrossings;
    // Long types indexed by length modulo period. Construction charges this
    // explicit expansion; it is not polynomial in the encoded period.
    std::vector<RotatingBoundaryType> types;
    RotatingBoundaryType select(uint64_t length) const;
    std::optional<std::vector<BoundaryDemand>> crossings(
        const RotatingBoundaryType& left, const RotatingBoundaryType& right) const;
};
// Same fixed physical mapping on every visit. The cells must account for all
// accessed family/atom/slot identities; this check is made before publishing.
// Preserve unconditional pipe protection. Scoped protected writer fragments
// require a corresponding cross-visit protection adapter.
RotatingBoundaryCertificate buildRotatingBoundaryCertificate(
    llvm::ArrayRef<PeriodicPayload> payloads, llvm::ArrayRef<RotatingFragment> fragments,
    llvm::ArrayRef<RotatingBoundaryCell> cells, llvm::ArrayRef<PeriodicRecord> prerequisites = {},
    uint64_t maximumTypes = 256, StorageProtectionPolicy protection = {});
// Preserve the child's analyzed native and demand relations. The caller proves
// uniformCrossings from shared effects and supplies fragments whose protection
// groups describe the enclosing visit scope, independently of the child index.
RotatingBoundaryCertificate buildRotatingBoundaryCertificate(
    PeriodicAnalysis quotient, llvm::ArrayRef<RotatingFragment> fragments,
    llvm::ArrayRef<RotatingBoundaryCell> cells,
    llvm::ArrayRef<std::pair<uint32_t, uint32_t>> uniformCrossings,
    uint64_t maximumTypes = 256, StorageProtectionPolicy protection = {});
struct AffineRotatingVisits {
    std::string error;
    uint64_t slope = 0, intercept = 0, startup = 0, period = 1;
    RotatingBoundaryCertificate child;
    // One transition ending at each startup visit and one per suffix phase.
    // Entry zero has no predecessor. The startup/suffix seam is separate from
    // repeating suffix transitions; its preceding visit may still be short.
    std::vector<std::vector<BoundaryDemand>> startupCrossings, suffixCrossings;
    std::vector<BoundaryDemand> seam;
    const std::vector<BoundaryDemand>& crossingInto(uint64_t visit) const;
};
// K(t)=slope*t+intercept, slope>0. Native/storage crossings only; arbitrary
// cross-visit prerequisites need their own supplied certificate. Child F*
// remains its quotient records, filtered by each visit's actual length.
AffineRotatingVisits analyzeAffineRotatingVisits(
    RotatingBoundaryCertificate child, uint64_t slope, uint64_t intercept, uint64_t maximumStartup = 256);
} // namespace mlir::pto::frontiersynch
#endif
