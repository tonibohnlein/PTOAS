// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Preserve finite endpoint families before logical IR is materialized.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ENDPOINTFAMILIES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ENDPOINTFAMILIES_H
#include "PTO/Transforms/FrontierSynch/NumericTemplateEndpoints.h"
namespace mlir::pto::frontiersynch {
struct EndpointFamilyMember {
    uint32_t record = 0;
    uint32_t source = 0;
    uint32_t target = 0;
    SmallVector<TemplateCoordinate> sourceCoordinates;
    SmallVector<TemplateCoordinate> targetCoordinates;
};
struct EndpointFamily {
    uint32_t id = 0; // Minimum original record ID; stable within the plan.
    uint32_t sourcePipe = 0;
    uint32_t targetPipe = 0;
    uint64_t displacement = 0;
    bool local = false;
    TemplateEndpointCut sourceCut;
    TemplateEndpointCut targetCut;
    std::vector<EndpointFamilyMember> members; // Original-record order, not dynamic order.
    std::size_t sourceOrder = 0; // Order among pieces at sourceCut.
    std::size_t targetOrder = 0; // Order among pieces at targetCut.
};
struct EndpointFamilies {
    std::string error;
    std::vector<EndpointFamily> families;
};
// Pure, non-mutating preparation. Coordinates and cuts borrow the original IR;
// serialize their identities before mutation or destruction. Membership is exact:
// every original record occurs once and both endpoint maps select that member.
// Potentially conflicting family orders are split conservatively, never reordered.
EndpointFamilies buildEndpointFamilies(const NumericTemplateEndpoints& plan);
} // namespace mlir::pto::frontiersynch
#endif
