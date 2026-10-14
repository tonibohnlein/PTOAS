// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared early-publication policy. Cleanup is terminal, never a regional drain.
#ifndef PTO_FRONTIERSYNCH_GUARDEDPUBLICATION_H
#define PTO_FRONTIERSYNCH_GUARDEDPUBLICATION_H
#include "PTO/Transforms/FrontierSynch/Shared/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include <functional>
#include <optional>
namespace mlir::pto::frontiersynch {
using EndpointExpressionEmitter =
    std::function<FailureOr<Value>(RegionExpressions::Id, Operation*)>;
enum class EndpointFailureStage {
    None, ConsumerGuard, SourceGuard, PublicationContract, CleanupGuard, MatchingIdentity, CleanupIdentity
};
struct EndpointEmissionFailure {
    EndpointFailureStage stage = EndpointFailureStage::None;
    RegionExpressions::Id expression = RegionExpressions::invalid;
    Operation* cut = nullptr;
    std::string diagnostic;
};
const char* endpointFailureStageName(EndpointFailureStage stage);
struct GuardedPublication {
    Value sourceGuard;
    RegionExpressions::Id publicationGuard = RegionExpressions::invalid;
    Operation* cleanupCut = nullptr;
    Value cleanupGuard;
    std::shared_ptr<const EndpointCutChoices> targetChoices;
};
// An exact regional selector may supply a single source occurrence from a
// repeated cut. existence describes that occurrence in invocation coordinates;
// atCut additionally selects its actual original visit. The shared qualifier
// verifies one fixed coordinate per enclosing loop before accepting this form.
struct SelectedPublication {
    RegionExpressions::Id existence, atCut;
};
// The caller supplies an exact activity predicate requiring both occurrences,
// and a matching identity for its consumer WAIT. First try the ordinary SET
// guard. If unavailable, publish whenever the source cut executes and consume
// under source-presence AND NOT activity in the function's terminal epilogue.
// This lowering certifies one publication per invocation from original branch
// structure or a supplied exact singleton selector. Multiple publications and
// unavailable histories are rejected, with no payload replay or private counters.
// The emitter stages detached code and must preserve valid state on failure.
FailureOr<GuardedPublication> prepareGuardedPublication(func::FuncOp function,
    RegionExpressions& arena, TemplateEndpointCut source, TemplateEndpointCut consumer,
    RegionExpressions::Id sourceGuard, RegionExpressions::Id activity,
    const EndpointExpressionEmitter& emit, std::string& error, Value consumerPredicate,
    std::optional<SelectedPublication> selectedSource = {}, EndpointEmissionFailure* failure = nullptr);
// Pure protocol planning: availability must be established without generated
// SSA. The recipe borrows this arena and unchanged original cuts. It retains
// the exact chosen predicates, not emitted Values or necessary-domain evidence.
// Matching identities remain the caller's separate obligation.
struct GuardedPublicationRecipe {
    const RegionExpressions* owner = nullptr;
    TemplateEndpointCut source, consumer;
    RegionExpressions::Id sourceGuard = RegionExpressions::invalid;
    RegionExpressions::Id publicationGuard = RegionExpressions::invalid;
    Operation* cleanupCut = nullptr;
    RegionExpressions::Id cleanupGuard = RegionExpressions::invalid;
};
using EndpointExpressionAvailability =
    std::function<LogicalResult(RegionExpressions::Id, Operation*)>;
FailureOr<std::shared_ptr<const GuardedPublicationRecipe>> preparePublicationRecipe(func::FuncOp function,
    RegionExpressions& arena, TemplateEndpointCut source, TemplateEndpointCut consumer,
    RegionExpressions::Id sourceGuard, RegionExpressions::Id activity, RegionExpressions::Id consumerGuard,
    const EndpointExpressionAvailability& available, std::string& error,
    std::optional<SelectedPublication> selectedSource = {});
// Append the alternative WAIT using the same record and matching tuple as the
// ordinary pair. The final tuple component is the logical occurrence identity;
// preceding components follow the normal memberCoordinates convention.
LogicalResult appendPublicationCleanup(const GuardedPublication& publication,
    uint32_t record, uint32_t sourcePipe, uint32_t targetPipe,
    ArrayRef<RegionExpressions::Id> identity, RegionExpressions& arena,
    const EndpointExpressionEmitter& emit, PreparedLogicalPlan& plan, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
