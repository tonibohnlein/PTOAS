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
#include <algorithm>
#include <memory>
#include <utility>

namespace mlir::pto::oahs {

namespace selected { class Constructor; }

using FrontierBits = std::vector<uint64_t>;
// Defined here because it is the innermost query of the frontier primitives.
// A measured profile attributed 21% of a large-cell construction to the call
// overhead of this three-line test alone. The value is unchanged.
inline bool frontierContains(const FrontierBits& b, std::size_t i)
{
    return i / 64 < b.size() && (b[i / 64] & (uint64_t(1) << (i % 64)));
}

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

// The must-history of the access classes, ordered
// (cell * PipeCount + engine) * 2 + mode, R=0, W=1, followed by sparse
// per-cell native access-order partitions. All rows use identical causal
// transport; partitions are obligations, not another completion state. Absence means no
// represented history; a PRESENT EMPTY bitset means unresolved history, so the
// two are distinct and `find` returning null is the only expression of absence.
// Only present classes are stored: a program with many storage cells otherwise
// carries a dense array of mostly absent classes through every state copy, which
// measurement showed to dominate both time and peak memory. `size` still reports
// the dense class extent, so callers may enumerate every class.
class FrontierHistory {
public:
    void reset(std::size_t classes)
    {
        extent = classes;
        entries.clear();
    }
    std::size_t size() const { return extent; }
    const FrontierBits* find(std::size_t index) const
    {
        const auto at = locate(index);
        return at != entries.end() && at->first == index ? &at->second : nullptr;
    }
    FrontierBits* find(std::size_t index)
    {
        const auto at = locate(index);
        return at != entries.end() && at->first == index ? &at->second : nullptr;
    }
    void assign(std::size_t index, FrontierBits value)
    {
        const auto at = locate(index);
        if (at != entries.end() && at->first == index) {
            at->second = std::move(value);
            return;
        }
        entries.insert(at, {index, std::move(value)});
    }
    using Entry = std::pair<std::size_t, FrontierBits>;
    std::vector<Entry>& present() { return entries; }
    const std::vector<Entry>& present() const { return entries; }
    bool operator==(const FrontierHistory& b) const { return extent == b.extent && entries == b.entries; }

private:
    std::vector<Entry>::const_iterator locate(std::size_t index) const
    {
        return std::lower_bound(entries.begin(), entries.end(), index,
            [](const Entry& entry, std::size_t key) { return entry.first < key; });
    }
    std::vector<Entry>::iterator locate(std::size_t index)
    {
        return std::lower_bound(entries.begin(), entries.end(), index,
            [](const Entry& entry, std::size_t key) { return entry.first < key; });
    }
    // Sorted by class index, so the representation of a given history is unique
    // and equality is a plain comparison.
    std::size_t extent = 0;
    std::vector<Entry> entries;
};

struct FrontierFacts {
    // A[p], T[p], S[e], D[e], in that order. A is the next launch gate;
    // T aggregates earlier finishes, but does not serialize their completion.
    // Only must-full keys have an S port. Other rows/columns for S are zero.
    std::vector<FrontierBits> reach;
    FrontierHistory history;
    std::vector<FrontierEvent> events;
    // Terminal retirement is not a reusable completion or event-reset receipt.
    bool terminalRetired = false, mayBeRetired = false;
    bool operator==(const FrontierFacts& b) const
    {
        return reach == b.reach && history == b.history && events == b.events &&
               terminalRetired == b.terminalRetired && mayBeRetired == b.mayBeRetired;
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
    UnconsumedAtExit,
    MissingRetirement
};

struct FrontierStep {
    bool applied = false;
    // Rejection is atomic: this is exactly the input snapshot on failure.
    FrontierState state;
    FrontierFailure failure = FrontierFailure::None;
    std::string reason;
    std::vector<FrontierRequirement> residuals;
};

// Ordinary prefix core, including the shared synchronous-payload contract and
// terminal-only retirement. Resource, phase, visibility and implicit transfers
// still require their own adapters. This does not select a native pass driver.
class CausalFrontier {
public:
    explicit CausalFrontier(Program);
    bool complete() const;
    const std::string& reason() const;
    const std::vector<EventIdentity>& keys() const;
    FrontierState initial() const;
    FrontierStep join(const FrontierState&, const FrontierState&) const;
    // A query never advances payload or installs prospective credit.
    FrontierStep inspect(const FrontierState&, std::size_t operation) const;
    FrontierStep issue(const FrontierState&, std::size_t operation) const;
    // Conservative loop hypothesis: add possible earlier body classes, retaining
    // only their source-prefix successor. The constructor must separately check
    // the selected body's inductive closure from its actual incoming interface.
    FrontierStep assumePreviousAccesses(const FrontierState&, const std::vector<std::size_t>&) const;
    FrontierStep command(const FrontierState&, const Command&, FrontierBinding) const;
    FrontierStep exit(const FrontierState&) const;

private:
    friend class selected::Constructor;
    // Construction-only transfer of actual effects. Adds no required-conflict
    // edges and grants no missing completion. Final acceptance always uses issue().
    FrontierStep pendingIssue(const FrontierState&, std::size_t operation) const;
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
