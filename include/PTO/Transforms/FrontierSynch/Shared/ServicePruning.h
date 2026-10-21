// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Finite source-pipe projection of union/GEN/class-wide-KILL transfers.
// No origin identities, distances, readiness, or executable cuts are inferred.
// Fixed-candidate delivery certification; no effect recovery or allocation.
#ifndef PTO_FRONTIERSYNCH_SERVICEPRUNING_H
#define PTO_FRONTIERSYNCH_SERVICEPRUNING_H
#include "PTO/Transforms/FrontierSynch/Shared/SourcePipeServices.h"
namespace mlir::pto::frontiersynch {
using ServiceMask = std::vector<std::vector<uint32_t>>;
struct DeliveryCertificate {
    bool accepted = false;
    std::string error;
    uint64_t retained = 0, transmissions = 0, inspected = 0, violations = 0;
};
struct ServiceProposal {
    ServiceMask kept;
    std::string error;
    uint64_t nodes = 0, tested = 0, classTests = 0, updates = 0, cleared = 0, deleted = 0;
};
// Inputs must have passed the source-pipe solver. Invalid masks refuse.
// Protected paths are supplied separately; an empty access set proves none.
DeliveryCertificate certifyServiceDelivery(
    const std::vector<SourcePipeNode>& graph, const std::vector<SourcePipeSite>& sites, const ServiceMask& baseline,
    const ServiceMask& candidate, const ServiceMask& protectedPaths);
ServiceProposal proposeLocalServices(
    const std::vector<SourcePipeNode>& graph, const std::vector<SourcePipeSite>& sites, const ServiceMask& baseline,
    const ServiceMask& protectedPaths);
} // namespace mlir::pto::frontiersynch
#endif
