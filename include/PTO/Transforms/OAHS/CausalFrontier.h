// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_OAHS_CAUSALFRONTIER_H
#define PTO_TRANSFORMS_OAHS_CAUSALFRONTIER_H

#include "PTO/Transforms/OAHS/Analysis.h"
#include <memory>

namespace mlir::pto::oahs {

using FrontierBits = std::vector<uint64_t>;
bool frontierContains(const FrontierBits&, std::size_t);

struct FrontierBinding {
    Cut cut = NoAnalysisId;
    std::size_t command = NoAnalysisId;
    bool operator==(const FrontierBinding& b) const { return cut == b.cut && command == b.command; }
    bool operator<(const FrontierBinding& b) const { return cut < b.cut || (cut == b.cut && command < b.command); }
};

struct FrontierEvent {
    // May balance: 1 = empty, 2 = full, 3 = either. No implicit region reset.
    uint8_t occupancy = 1;
    // Static alternatives for the CURRENT live publication, not visit numbers
    // or a generation-correspondence certificate. Empty paths add no binding.
    std::vector<FrontierBinding> publishers;
    bool operator==(const FrontierEvent& b) const { return occupancy == b.occupancy && publishers == b.publishers; }
};

struct FrontierFacts {
    // A[p], T[p], S[e], D[e], in that order. A is the next launch gate;
    // T aggregates earlier finishes, but does not serialize their completion.
    // Only must-full keys have an S port. Other rows/columns for S are zero.
    std::vector<FrontierBits> reach;
    // Class order: (cell * PipeCount + engine) * 2 + mode, R=0, W=1.
    // nullopt is absence; a present empty bitset is unresolved history.
    std::vector<std::optional<FrontierBits>> history;
    std::vector<FrontierEvent> events;
    bool operator==(const FrontierFacts& b) const
    {
        return reach == b.reach && history == b.history && events == b.events;
    }
};

namespace detail {
struct CausalFrontierModel;
struct CausalFrontierState;
} // namespace detail

// Immutable and bound to one immutable original program. Default construction
// means unreachable, not fresh entry. Copies preserve the snapshot; a future
// constructor must still invalidate checkpoints after earlier endpoint edits.
class FrontierState {
public:
    FrontierState() = default;
    bool reachable() const { return bool(data); }
    const FrontierFacts* facts() const;
    bool operator==(const FrontierState&) const;
    bool operator!=(const FrontierState& b) const { return !(*this == b); }

private:
    std::shared_ptr<const detail::CausalFrontierState> data;
    friend class CausalFrontier;
};

struct FrontierRequirement {
    unsigned cell = 0;
    Pipe source = Pipe::S;
    bool sourceWrite = false;
    std::size_t consumer = NoAnalysisId;
    // RMW consumers can require both WAR and WAW from separate classes.
    bool consumerRead = false, consumerWrite = false;
};

enum class FrontierFailure {
    None,
    InvalidInput,
    UnsupportedContract,
    Payload,
    PublicationNotEmpty,
    AcquisitionNotFull,
    ConsumptionNotEstablished,
    UnconsumedAtExit
};

struct FrontierStep {
    bool applied = false;
    // Rejection is atomic: this is exactly the input snapshot on failure.
    FrontierState state;
    FrontierFailure failure = FrontierFailure::None;
    std::string reason;
    std::vector<FrontierRequirement> residuals;
};

// Ordinary issue-ordered asynchronous core only. This increment deliberately
// does not reinterpret synchronous lanes, ALL/retirement, phase/resource or
// visibility contracts. It does not replace the current native checker yet.
class CausalFrontier {
public:
    explicit CausalFrontier(Program);
    bool complete() const;
    const std::string& reason() const;
    const std::vector<EventIdentity>& keys() const;
    FrontierState initial() const;
    FrontierStep join(const FrontierState&, const FrontierState&) const;
    FrontierStep issue(const FrontierState&, std::size_t operation) const;
    FrontierStep command(const FrontierState&, const Command&, FrontierBinding) const;
    FrontierStep exit(const FrontierState&) const;

private:
    std::shared_ptr<const detail::CausalFrontierModel> model;
    FrontierStep checkState(const FrontierState&) const;
    FrontierStep extend(
        const FrontierState&, Pipe, const Command*, std::size_t operation, std::size_t key, FrontierBinding) const;
};

struct FrontierCut {
    FrontierState incoming, beforeIssue, outgoing;
};
struct FrontierCheck {
    // A supported check returns its FIRST refusal. Unlike AnalysisResult, this
    // is not an all-residual diagnostic traversal of an invalid protocol.
    bool complete = false, accepted = false;
    FrontierFailure failure = FrontierFailure::None;
    std::string reason;
    Cut cut = NoAnalysisId;
    std::size_t command = NoAnalysisId;
    std::vector<FrontierRequirement> residuals;
    std::vector<EventIdentity> keys;
    // Exported only after acceptance and convergence. Never stale partial
    // invariants from an earlier successful visit to a now-rejected site.
    std::vector<FrontierCut> cuts;
    uint64_t siteEvaluations = 0;
};

// Reuses original control and command validation, with a finite must worklist.
// Failed endpoints grant no provisional credit; no hidden plan repair occurs.
FrontierCheck checkCausalFrontier(const Program&, const Commands&);

} // namespace mlir::pto::oahs
#endif
