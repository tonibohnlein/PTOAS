// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERPROTOCOLINTERNAL_H
#define PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERPROTOCOLINTERNAL_H

#include "PTO/Transforms/CanonicalSync/SyncCoverProtocol.h"

namespace mlir {
namespace pto {
namespace sync_cover_protocol_detail {

struct ResolvedChannel {
  const SyncCoverEventChannel *description = nullptr;
  SyncCoverTimelinePosition setPosition = 0;
  SyncCoverTimelinePosition waitPosition = 0;
  SyncCoverGuard setGuard;
  SyncCoverGuard waitGuard;
};

struct ResolvedProtocol {
  std::vector<ResolvedChannel> channels;
  std::vector<std::size_t> reachablePhases;
  std::size_t initialPhase = 0;
  std::vector<std::size_t> nextPhase;
  std::vector<SyncCoverGuard> guardByPhase;
  std::size_t phasePreperiod = 0;
  std::size_t phasePeriod = 0;
  std::size_t verificationHorizon = 1;
};

bool consumeWork(SyncCoverCoverageWorkBudget *budget, std::size_t amount = 1);

bool graphFitsProtocolLimits(const SyncCoverGraph &graph,
                             SyncCoverProtocolLimits limits);

bool targetFitsProtocolLimits(const SyncCoverProtocolTargetContract &target,
                              SyncCoverProtocolLimits limits);

SyncCoverProtocolError
validateProtocolTargetContract(const SyncCoverProtocolTargetContract &target,
                               SyncCoverProtocolLimits limits,
                               SyncCoverCoverageWorkBudget *workBudget);

SyncCoverProtocolError resolveProtocol(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverEventProtocol &protocol, SyncCoverProtocolLimits limits,
    ResolvedProtocol &resolved, std::optional<std::size_t> &invalidIndex,
    SyncCoverCoverageWorkBudget *workBudget, bool validateGraph = true,
    bool validateTarget = true);

SyncCoverProtocolVerificationResult verifyProtocolAssumingValidGraph(
    const SyncCoverGraph &graph, const SyncCoverProtocolTargetContract &target,
    const SyncCoverEventProtocol &protocol, SyncCoverProtocolLimits limits,
    SyncCoverCoverageWorkBudget *workBudget, bool validateTarget = true);

SyncCoverProtocolError verifyResolvedProtocolAutomaton(
    const SyncCoverEventProtocol &protocol, const ResolvedProtocol &resolved,
    SyncCoverProtocolLimits limits, SyncCoverProtocolStatistics &statistics,
    SyncCoverCoverageWorkBudget *workBudget);

std::optional<SyncCoverGuard>
effectivePointGuard(const SyncCoverGraph &graph, const SyncCoverCutPoint &point,
                    SyncCoverCoverageWorkBudget *workBudget = nullptr);

} // namespace sync_cover_protocol_detail
} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_SYNCCOVERPROTOCOLINTERNAL_H
