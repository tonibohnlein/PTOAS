// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_SELECTEDPLAN_H
#define PTO_TRANSFORMS_OAHS_SELECTEDPLAN_H

#include "PTO/Transforms/OAHS/CausalFrontier.h"
#include "PTO/Transforms/OAHS/StorageFrontiers.h"

namespace mlir::pto::oahs {

enum class SelectedFailure {
    None, InvalidInput, UnsupportedContract, UnqualifiedControl,
    MissingParticipation, EventResource, SelectedUpdate, LoopInvariant, FinalValidation
};
enum class EndpointPurpose { Fixed, Completion, ConsumptionAcknowledgment, LocalFence, Retirement };
enum class RequirementStage { Known, Overlap };

// IDs survive insertion into an earlier word. cut/word positions are reconstructed
// from this ledger, not cached in an event-generation certificate.
struct SelectedEndpoint {
    std::size_t id = NoAnalysisId;
    Cut cut = NoAnalysisId;
    Command command;
    EndpointPurpose purpose = EndpointPurpose::Completion;
    // Index into decisions, or into channels for a qualified cyclic result.
    std::size_t request = NoAnalysisId;
    std::size_t acknowledges = NoAnalysisId;
};
struct SelectedSource {
    Pipe pipe = Pipe::S;
    std::size_t origin = NoAnalysisId;
    Cut cut = NoAnalysisId;
    uint64_t version = 0;
    FrontierState snapshot;
};
struct SelectedDecision {
    Cut consumer = NoAnalysisId, publication = NoAnalysisId;
    RequirementStage stage = RequirementStage::Overlap;
    Pipe source = Pipe::S, observer = Pipe::S;
    std::vector<FrontierRequirement> required;
    std::vector<std::size_t> endpoints;
    bool commonCut = false, enlargedPrefix = false;
};
struct SelectedUpdate {
    uint64_t version = 0, siteEvaluations = 0;
    std::size_t finalizedQueries = 0;
    std::vector<Cut> changedCuts;
};
struct SelectedChannel {
    unsigned cell = 0, key = 0;
    Pipe source = Pipe::S, observer = Pipe::S;
    std::vector<Cut> publications, acquisitions;
};
struct SelectedWork {
    uint64_t frontierVisits = 0, selectedUpdates = 0, replaySiteEvaluations = 0;
    uint64_t forwardSiteEvaluations = 0;
    uint64_t keyQueries = 0, invariantSiteEvaluations = 0;
    // Total portable construction includes model/control/storage preparation and
    // final validation. It excludes native import/emission and test references.
    uint64_t elapsedMicroseconds = 0, preparationMicroseconds = 0;
    std::size_t sourceHandles = 0, acknowledgments = 0, commonCutTransfers = 0;
    std::size_t recurringChannels = 0;
};
struct SelectedPlan {
    bool success = false;
    SelectedFailure failure = SelectedFailure::None;
    std::string reason;
    Cut cut = NoAnalysisId;
    // Populated only after final validation. A partial diagnostic ledger is not
    // an emitted synchronization program.
    Commands commands;
    std::vector<SelectedEndpoint> ledger;
    std::vector<SelectedSource> sources;
    std::vector<SelectedDecision> decisions;
    std::vector<SelectedChannel> channels;
    std::vector<SelectedUpdate> updates;
    FrontierCheck certificate;
    SelectedWork work;
};

// F1--F8 service, not a new pass mode. The live handoff driver remains gated.
// Fixed words are preserved in order. Unsupported typed effects are refused,
// never erased or sent to the historical constructor on failure.
SelectedPlan constructSelectedPlan(const Program&, const Commands& fixed = {});

} // namespace mlir::pto::oahs
#endif
