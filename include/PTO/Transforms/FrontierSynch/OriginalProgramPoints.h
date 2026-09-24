// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALPROGRAMPOINTS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALPROGRAMPOINTS_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir::pto::frontiersynch {
inline constexpr std::size_t NoControlId = std::numeric_limits<std::size_t>::max();

// An identity for one immutable original-program snapshot. Keeping the token
// alive prevents address reuse from making an old query match a new import.
// Rebuild the analyses with a fresh version after changing original control,
// effects, aliases, selectors or qualification. Selected sync is not this version.
struct OriginalProgramVersion {
    std::shared_ptr<const unsigned char> identity;
    std::uint64_t revision = 0;
    static OriginalProgramVersion fresh() { return {std::make_shared<const unsigned char>(0), 0}; }
    explicit operator bool() const { return bool(identity); }
    bool operator==(const OriginalProgramVersion& other) const
    {
        const std::owner_less<std::shared_ptr<const unsigned char>> less;
        return !less(identity, other.identity) && !less(other.identity, identity) && revision == other.revision;
    }
    bool operator!=(const OriginalProgramVersion& other) const { return !(*this == other); }
    bool operator<(const OriginalProgramVersion& other) const
    {
        const std::owner_less<std::shared_ptr<const unsigned char>> less;
        if (less(identity, other.identity)) {
            return true;
        }
        if (less(other.identity, identity)) {
            return false;
        }
        return revision < other.revision;
    }
};

struct Region {
    enum Kind { Sequence, Choice, For, While, Operation } kind = Sequence;
    std::vector<Region> children;
    std::size_t operation = 0;
    bool zeroTripPossible = false;
    std::size_t originalOwner = NoControlId;
    bool qualifiedCounted = false;
};

// Static ORIGINAL insertion positions, not positions in a selected command word.
// The first two fields retain SourceMilestone's payload aggregate spelling.
// Scope: before/after the complete structured operation (or the function body).
// Child: entry/before-terminator of the numbered original region. In particular
// a loop's external entry is different from its repeatedly visited body entry.
struct OriginalCut {
    std::size_t operation = NoControlId;
    enum Side { Before, After } side = After;
    enum class Kind { Payload, Scope, Child } kind = Kind::Payload;
    std::size_t owner = NoControlId, child = NoControlId;
    static OriginalCut scope(std::size_t owner, Side side)
    {
        return {NoControlId, side, Kind::Scope, owner, NoControlId};
    }
    static OriginalCut childBoundary(std::size_t owner, std::size_t child, Side side)
    {
        return {NoControlId, side, Kind::Child, owner, child};
    }
    auto key() const { return std::make_tuple(kind, operation, side, owner, child); }
    bool operator==(const OriginalCut& other) const { return key() == other.key(); }
    bool operator!=(const OriginalCut& other) const { return !(*this == other); }
    bool operator<(const OriginalCut& other) const { return key() < other.key(); }
};
using SourceMilestone = OriginalCut;

// A selector is separate from the occurrence relation. A physical-relation ID
// filters translated effect witnesses; it is NOT a bank correspondence proof.
struct OriginalAccessSelector {
    std::size_t cell = 0;
    bool read = true, write = true;
    std::optional<unsigned> engine;
    std::size_t physicalRelation = NoControlId;
    std::size_t qualification = NoControlId;
    std::vector<std::size_t> predicateDependencies;
    auto key() const
    {
        return std::tie(cell, read, write, engine, physicalRelation, qualification, predicateDependencies);
    }
    bool operator==(const OriginalAccessSelector& other) const { return key() == other.key(); }
    bool operator<(const OriginalAccessSelector& other) const { return key() < other.key(); }
};

// This is an INTERPRETATION, not a positive D1/D2/D4 certificate. FirstReach
// stops at the first represented visit of stop. AfterBackedge stops at the first
// such visit after crossing at least one backedge of the named original loop.
// It does not mean an iteration distance, the previous bank use, or finite-loop
// exactness. Unqualified preserves the legacy conservative recurrence outcome.
struct OriginalOccurrenceContext {
    enum class StopVisit { Unqualified, FirstReach, AfterBackedge } stopVisit = StopVisit::Unqualified;
    std::size_t source = NoControlId, target = NoControlId;
    std::size_t backedgeOwner = NoControlId;
    std::size_t incomingInterface = NoControlId;
    std::size_t qualification = NoControlId;
    auto key() const { return std::tie(stopVisit, source, target, backedgeOwner, incomingInterface, qualification); }
    bool operator==(const OriginalOccurrenceContext& other) const { return key() == other.key(); }
    bool operator<(const OriginalOccurrenceContext& other) const { return key() < other.key(); }
};

struct OriginalIntervalRequest {
    OriginalProgramVersion version;
    OriginalAccessSelector selector;
    OriginalOccurrenceContext occurrence;
    OriginalCut start, stop;
    // Independent of the stopping cut: the named stopping payload's access is
    // included iff true. A control boundary has no access, so requires false.
    // Start-after always excludes the access preceding that start visit.
    bool includeStoppingAccess = false;
    // The declared continuation participates in owner selection. nullopt asks
    // for the least owner of the named uses/cuts; NoControlId explicitly means
    // the function horizon. A child exit is not an owner-level next overwrite.
    std::optional<std::size_t> continuationOwner;
    auto key() const
    {
        return std::tie(version, selector, occurrence, start, stop, includeStoppingAccess, continuationOwner);
    }
    bool operator==(const OriginalIntervalRequest& other) const { return key() == other.key(); }
    bool operator<(const OriginalIntervalRequest& other) const { return key() < other.key(); }
};
struct OriginalInterval {
    OriginalIntervalRequest query;
    // Derived from source, target, BOTH cuts and the declared continuation.
    std::size_t owner = NoControlId;
    bool operator==(const OriginalInterval& other) const { return owner == other.owner && query == other.query; }
    bool operator!=(const OriginalInterval& other) const { return !(*this == other); }
    bool operator<(const OriginalInterval& other) const
    {
        return std::tie(owner, query) < std::tie(other.owner, other.query);
    }
};
struct OriginalIntervalResult {
    bool valid = false;
    OriginalInterval interval;
    std::string reason;
};
// These are represented original-control cases, never fresh storage/event state.
struct OriginalContinuationCases {
    bool incoming = false, childEntry = false, bypass = false, backedge = false;
    bool reachedStop = false, reachedOwnerExit = false;
};
} // namespace mlir::pto::frontiersynch
#endif
