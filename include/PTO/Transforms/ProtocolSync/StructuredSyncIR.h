// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- StructuredSyncIR.h - Immutable physical schedule --------*- C++ -*-===//
//
// StructuredSyncIR preserves lexical control structure, physical phases,
// accesses, logical slots, and program points. Construction APIs are private;
// consumers receive an immutable schedule after freeze().
//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_PROTOCOLSYNC_STRUCTUREDSYNCIR_H
#define PTO_TRANSFORMS_PROTOCOLSYNC_STRUCTUREDSYNCIR_H

#include "PTO/Transforms/ProtocolSync/SyncSemantics.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace mlir::pto::protocol_sync {

using SyncRegionId = std::uint32_t;
using SyncSummaryId = std::uint32_t;
using SyncPhaseId = std::uint32_t;
using SyncAccessId = std::uint32_t;
using SyncProgramPointId = std::uint32_t;

inline constexpr std::uint32_t kInvalidSyncId = std::numeric_limits<std::uint32_t>::max();

enum class SyncRegionKind : std::uint8_t {
    Function,
    Sequence,
    Choice,
    Alternative,
    Loop,
    PhysicalSection,
};

enum class SyncCardinality : std::uint8_t {
    ExactlyOnce,
    ZeroOrOne,
    ZeroOrMore,
    OneOrMore,
};

enum class SyncProgramPointKind : std::uint8_t {
    RegionEntry,
    RegionExit,
    PhaseBefore,
    PhaseAfter,
};

struct SyncControlAtom {
    SyncRegionId choice = kInvalidSyncId;
    unsigned arm = 0;
};

struct SyncIterationDomain {
    llvm::SmallVector<SyncRegionId, 2> loops;
};

struct SyncRegionElement {
    enum class Kind : std::uint8_t { Phase, ChildRegion };

    Kind kind = Kind::Phase;
    unsigned order = 0;
    SyncPhaseId phase = kInvalidSyncId;
    SyncRegionId child = kInvalidSyncId;
};

struct SyncRegion {
    SyncRegionId id = kInvalidSyncId;
    SyncRegionId parent = kInvalidSyncId;
    SyncRegionKind kind = SyncRegionKind::Sequence;
    SyncCardinality cardinality = SyncCardinality::ExactlyOnce;
    Operation* operation = nullptr;
    unsigned arm = 0;
    llvm::SmallVector<SyncControlAtom, 2> guard;
    SyncIterationDomain iterationDomain;
    SyncProgramPointId entry = kInvalidSyncId;
    SyncProgramPointId exit = kInvalidSyncId;
    llvm::SmallVector<SyncRegionElement, 16> elements;
};

struct SyncPhase {
    SyncPhaseId id = kInvalidSyncId;
    SyncSummaryId summary = kInvalidSyncId;
    SyncRegionId region = kInvalidSyncId;
    Operation* operation = nullptr;
    std::optional<unsigned> macroPhase;
    SyncPhysicalCore core = SyncPhysicalCore::Unknown;
    PIPE pipe = PIPE::PIPE_UNASSIGNED;
    SyncCompletionKind completion = SyncCompletionKind::PhaseEnd;
    llvm::SmallVector<SyncControlAtom, 2> guard;
    SyncIterationDomain iterationDomain;
    SyncProgramPointId before = kInvalidSyncId;
    SyncProgramPointId after = kInvalidSyncId;
    llvm::SmallVector<SyncAccessId, 4> accesses;
};

struct SyncAccess {
    SyncAccessId id = kInvalidSyncId;
    SyncPhaseId phase = kInvalidSyncId;
    Value value;
    SyncStorageProvenance storage;
    SyncAccessMode mode = SyncAccessMode::ReadWrite;
    std::optional<SyncSlotExpression> slot;
    SyncVisibilityClass visibility = SyncVisibilityClass::Unknown;
};

struct SyncProgramPoint {
    SyncProgramPointId id = kInvalidSyncId;
    SyncProgramPointKind kind = SyncProgramPointKind::PhaseBefore;
    SyncRegionId region = kInvalidSyncId;
    SyncPhaseId phase = kInvalidSyncId;
};

struct SyncFailure {
    SyncFailureReason reason = SyncFailureReason::None;
    Operation* operation = nullptr;
    std::string detail;
};

class StructuredSyncIR {
public:
    explicit StructuredSyncIR(func::FuncOp function) : function(function) {}

    func::FuncOp getFunction() const { return function; }
    bool isFrozen() const { return frozen; }
    llvm::ArrayRef<SyncRegion> getRegions() const { return regions; }
    llvm::ArrayRef<SyncOpSummary> getSummaries() const { return summaries; }
    llvm::ArrayRef<SyncPhase> getPhases() const { return phases; }
    llvm::ArrayRef<SyncAccess> getAccesses() const { return accesses; }
    llvm::ArrayRef<SyncProgramPoint> getProgramPoints() const { return points; }
    llvm::ArrayRef<SyncFailure> getFailures() const { return failures; }
    const SyncRegion* findRegion(SyncRegionId id) const;
    const SyncPhase* findPhase(SyncPhaseId id) const;
    const SyncAccess* findAccess(SyncAccessId id) const;
    LogicalResult freeze();

private:
    friend class StructuredSyncIRBuilder;
    friend class StructuredSyncIRConstruction;
    func::FuncOp function;
    bool frozen = false;
    llvm::SmallVector<SyncRegion, 16> regions;
    llvm::SmallVector<SyncOpSummary, 32> summaries;
    llvm::SmallVector<SyncPhase, 32> phases;
    llvm::SmallVector<SyncAccess, 64> accesses;
    llvm::SmallVector<SyncProgramPoint, 64> points;
    llvm::SmallVector<SyncFailure, 4> failures;
};

class StructuredSyncIRBuilder {
public:
    StructuredSyncIRBuilder(const SyncSemanticContext& context, ProtocolSyncStatistics* statistics = nullptr)
        : context(context), statistics(statistics)
    {}

    LogicalResult build(func::FuncOp function, StructuredSyncIR& schedule) const;

private:
    const SyncSemanticContext& context;
    ProtocolSyncStatistics* statistics;
};

void printStructuredSyncIR(const StructuredSyncIR& schedule, llvm::raw_ostream& output);
llvm::StringRef stringifySyncRegionKind(SyncRegionKind kind);
llvm::StringRef stringifySyncCardinality(SyncCardinality cardinality);

} // namespace mlir::pto::protocol_sync

#endif // PTO_TRANSFORMS_PROTOCOLSYNC_STRUCTUREDSYNCIR_H
