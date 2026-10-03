// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// A semantic one-stream result, not a new program representation. Borrowed
// original MLIR and phase records obey SyncInput's unchanged-source lifetime.
#ifndef PTO_FRONTIERSYNCH_SINGLESTREAMLOOP_H
#define PTO_FRONTIERSYNCH_SINGLESTREAMLOOP_H
#include "PTO/Transforms/FrontierSynch/PeriodicDemandAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <memory>
namespace mlir { class IRMapping; namespace func { class FuncOp; } }
namespace mlir::pto { class SyncInput; struct SyncParticipation; }
namespace mlir::pto::frontiersynch {
class TraceDemandAnalysis;
class CostLedger;
struct SelectedAnalysis;
struct DirectEmissionResult;
struct SingleStreamPending;
enum class SingleStreamProtocol {
    OneGenerationBoundary, RepeatedReadyRelease, SequentialBoundaries, ExclusiveArmBoundaries
};
// A symbolic bound endpoint, not a concrete trip expansion or physical cell.
// Present iff (activation is absent or equals its arm outcome) && n>0.
// Ordinal = coefficient*n + offset in the original loop frame.
struct SingleStreamBoundPort {
    std::size_t site = 0;
    int64_t coefficient = 0, offset = 0;
    bool requiresPositiveTrips = true;
    // Original immutable i1 Value and enclosing If, shared by both ports.
    Value activation;
    Operation* participation = nullptr;
    Region* armRegion = nullptr;
    bool activationOutcome = true;
    SmallVector<scf::ForOp> frameOwners;
};
// Closed guard for a requested first/last event pair in its ORIGINAL frame:
// n>0 and coefficient*n+constant>=0. Unreachable queries have reachable=false.
struct SingleStreamPortQuery {
    bool reachable = false;
    Value bound, activation;
    int64_t coefficient = 0, constant = 0;
    Region* armRegion = nullptr;
    bool activationOutcome = true;
};
struct SingleStreamLoop {
    SingleStreamProtocol protocol = SingleStreamProtocol::OneGenerationBoundary;
    bool repeatedPair() const { return protocol == SingleStreamProtocol::RepeatedReadyRelease; }
    bool boundaryAlternatives() const
    {
        return protocol == SingleStreamProtocol::SequentialBoundaries ||
               protocol == SingleStreamProtocol::ExclusiveArmBoundaries;
    }
    bool regionOnly = false;
    bool hasPrefix = false;
    std::shared_ptr<const SingleStreamLoop> loopChild;
    SingleStreamBoundPort firstPort() const
    { return {first(), 0, 0, true, activation, participation, armRegion, activationOutcome, frames}; }
    SingleStreamBoundPort lastPort() const
    { return {last(), 1, -1, true, activation, participation, armRegion, activationOutcome, frames}; }
    Value activation;
    Operation* participation = nullptr;
    Region* armRegion = nullptr;
    bool activationOutcome = true;
    int64_t activationArgument = -1, participationSource = -1;
    DictionaryAttr activationBinding;
    // Original before-operation cut, admitted before any pending IR mutation.
    Operation* publicationCut = nullptr;
    int64_t publicationCutSource = -1;
    Operation* originalBarrier = nullptr;
    int64_t originalBarrierSource = -1;
    scf::ForOp loop;
    // Original outer-to-inner rectangular occurrence frame. Empty denotes the
    // legacy single-frame artifact; coordinates are never flattened/products.
    SmallVector<scf::ForOp> frames;
    SmallVector<int64_t> frameSources;
    // Each child keeps its original bound/frame and its own quotient index.
    Value bound;
    SmallVector<std::shared_ptr<const SingleStreamLoop>> children;
    scf::IfOp exitGuard;
    std::size_t entry = 0, exit = 0;
    SmallVector<std::size_t> bodies;
    std::size_t first() const { return bodies.front(); }
    std::size_t firstCompute() const { return bodies[1]; }
    std::size_t last() const { return bodies.back(); }
    ArrayAttr chainWitnesses;
    ArrayAttr crossingMinimum, portClosure;
    std::size_t portClosureUpdates = 0, portClosureMaximumPieces = 0;
    SmallVector<int64_t> selectorSources;
    int64_t loopSource = -1, guardSource = -1;
    int64_t boundArgument = -1;
    IntegerAttr boundConstant;
    uint64_t banks = 0;
    ArrayAttr selectorWitnesses;
    RotatingFootprintAnalysis storage;
    PeriodicDemandReduction reduction;
    static FailureOr<std::shared_ptr<const SingleStreamLoop>> build(
        func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
        CostLedger& costs, std::string& reason, std::shared_ptr<const SingleStreamLoop> child = {});
    const PeriodicDemandReduction& bodyIndex() const
    { return loopChild ? loopChild->bodyIndex() : reduction; }
    static FailureOr<std::shared_ptr<const SingleStreamLoop>> buildBodyRegion(
        func::FuncOp function, scf::ForOp owner, ArrayRef<std::size_t> sites,
        const SyncInput& input, const TraceDemandAnalysis& trace, CostLedger& costs, std::string& reason);
    // Requested finite port queries use actual occurrence identity. No absent
    // port or duplicate first/last alias is introduced as an intermediate event.
    FailureOr<SingleStreamPortQuery> portQuery(bool sourceLast, PeriodicEventKind sourceKind,
        bool targetLast, PeriodicEventKind targetKind) const;
    static FailureOr<std::shared_ptr<const SingleStreamLoop>> composeSequentialRegions(
        func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
        ArrayRef<std::shared_ptr<const SingleStreamLoop>> children, CostLedger& costs, std::string& reason);
    static FailureOr<std::shared_ptr<const SingleStreamLoop>> composeExclusiveRegions(
        func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
        ArrayRef<std::shared_ptr<const SingleStreamLoop>> children, CostLedger& costs, std::string& reason);
    // Requested nested port guard, closed over original bounds and activation.
    // Affine rows are a finite union, not an opaque numerical callback.
    FailureOr<ArrayAttr> framePortQuery(bool sourceLast, PeriodicEventKind sourceKind,
        bool targetLast, PeriodicEventKind targetKind) const;
    static FailureOr<std::shared_ptr<const SingleStreamLoop>> buildRepeatedPair(
        func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
        CostLedger& costs, std::string& reason);
    static FailureOr<std::shared_ptr<const SingleStreamLoop>> buildRepeatedRegion(
        func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
        CostLedger& costs, std::string& reason);
    static FailureOr<std::shared_ptr<const SingleStreamLoop>> composeRepeatedRegion(
        func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace,
        std::shared_ptr<const SingleStreamLoop> child, CostLedger& costs, std::string& reason);
    DictionaryAttr evidence(MLIRContext* context) const;
    DictionaryAttr frameEvidence(MLIRContext* context) const;
    SingleStreamLoop(const SingleStreamLoop&) = delete;
    SingleStreamLoop& operator=(const SingleStreamLoop&) = delete;
    SingleStreamLoop(SingleStreamLoop&&) = delete;
    SingleStreamLoop& operator=(SingleStreamLoop&&) = delete;
private:
    SingleStreamLoop() = default;
    static FailureOr<DictionaryAttr> crossingWitness(const SyncInput& input,
        const CompoundInstanceElement* source, const CompoundInstanceElement* consumer,
        std::size_t sourceSite, std::size_t consumerSite, int64_t distance, MLIRContext* context);
    static bool prefixPure(Operation* begin, Operation* end);
    static LogicalResult entryCut(const SyncInput& input, func::FuncOp function, Operation* entry,
        scf::ForOp loop, const SyncParticipation& participation, std::string& reason);
    LogicalResult qualifySequentialSource(func::FuncOp function, const SyncInput& input,
        const TraceDemandAnalysis& trace, std::string& reason);
    LogicalResult qualifyExclusiveSource(func::FuncOp function, const SyncInput& input,
        const TraceDemandAnalysis& trace, std::string& reason);
    LogicalResult buildSequentialClosure(MLIRContext* context, std::string& reason);
    LogicalResult buildBody(func::FuncOp function, const SyncInput& input,
        const TraceDemandAnalysis& trace, CostLedger& costs, std::string& reason);
};
LogicalResult emitSingleStreamLoop(const SingleStreamLoop& plan, const SyncInput& input,
                                  const TraceDemandAnalysis& trace,
                                  IRMapping& mapping, DirectEmissionResult& result);
LogicalResult emitRepeatedPairLoop(const SingleStreamLoop& plan, const TraceDemandAnalysis& trace,
                                   IRMapping& mapping, DirectEmissionResult& result);
LogicalResult assignSingleStreamPhysical(const SyncInput& input, const TraceDemandAnalysis& trace,
                                        DirectEmissionResult& result, CostLedger& costs);
} // namespace mlir::pto::frontiersynch
#endif
