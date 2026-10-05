// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Partition each endpoint side independently while retaining exact handoff identities.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ENDPOINTPIECES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ENDPOINTPIECES_H
#include "PTO/Transforms/FrontierSynch/NumericTemplateEndpoints.h"
namespace mlir::pto::frontiersynch {
struct EndpointPieceMember {
    uint32_t record = 0;
    SmallVector<TemplateCoordinate> coordinates;
};
struct EndpointPiece {
    uint32_t namespaceId = 0; // Minimum record of the pipe-pair/displacement class.
    uint32_t sourcePipe = 0;
    uint32_t targetPipe = 0;
    uint64_t displacement = 0;
    EndpointKind kind = EndpointKind::Set;
    TemplateEndpointCut cut;
    std::size_t order = 0;
    std::vector<EndpointPieceMember> members;
};
struct EndpointPieces {
    std::string error;
    std::vector<EndpointPiece> pieces;
};
// Consumes the prevalidated exact endpoint plan. Each recipe occurs in exactly
// one piece. Every piece has unique own-side coordinates, and order preserves
// all potentially coexecuted commands. Matching uses original record identities;
// source and destination may consequently use different piece partitions.
EndpointPieces buildEndpointPieces(const NumericTemplateEndpoints& plan);
} // namespace mlir::pto::frontiersynch
#endif
