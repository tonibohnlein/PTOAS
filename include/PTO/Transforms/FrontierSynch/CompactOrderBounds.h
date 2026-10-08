// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owned numerical order snapshots on one fixed original loop invocation.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTORDERBOUNDS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_COMPACTORDERBOUNDS_H
#include "PTO/Transforms/FrontierSynch/CompactWriterReaderInput.h"
#include "PTO/Transforms/FrontierSynch/CompactLowerFacts.h"
#include "PTO/Transforms/FrontierSynch/PeriodicExcessCertificate.h"
namespace mlir::pto::frontiersynch {
// Only the factory can establish this fixed, all-sites-present occurrence frame.
// The arena/context are owned. Original loop, phases and SyncInput are borrowed
// unchanged, as in OrderContext. One invocation has no enclosing visit tuple;
// enclosing loops require a future composition adapter with outer coordinates.
class CompactFixedBodyContext {
public:
    const OrderContext& context() const { return owner; }
    llvm::ArrayRef<PeriodicPayload> payloads() const { return word; }
    RegionExpressions::Id trips() const { return count; }
private:
    CompactFixedBodyContext(OrderContext context, std::vector<PeriodicPayload> payloads,
                            RegionExpressions::Id trips);
    OrderContext owner;
    std::vector<PeriodicPayload> word;
    RegionExpressions::Id count;
    friend std::shared_ptr<const CompactFixedBodyContext> captureBalancedCompactFixedBodyContext(
        scf::ForOp, const SyncInput&, const PhaseIndex&, std::shared_ptr<RegionExpressions>,
        RegionExpressions::Id, std::string&);
    friend std::shared_ptr<const CompactFixedBodyContext> captureCompactFixedBodyContext(
        scf::ForOp, const SyncInput&, const PhaseIndex&, std::shared_ptr<RegionExpressions>,
        RegionExpressions::Id, std::string&);
};
// Producer supplies the exact nonnegative trip-count circuit for this loop, in
// this arena, uniformly for all admitted parameters. This is an explicit domain
// binding contract, not normalization or a proof inferred from sampled trips.
// The factory validates the unchanged shared fixed body and constructs presence
// and reference callbacks itself. Guarded/nested/macro bodies need another
// adapter; failure leaves every supplied mathematical analysis untouched.
std::shared_ptr<const CompactFixedBodyContext> captureCompactFixedBodyContext(
    scf::ForOp loop, const SyncInput& input, const PhaseIndex& index,
    std::shared_ptr<RegionExpressions> arena, RegionExpressions::Id exactTrips, std::string& error);

// Balanced branches retain the same all-slots-present word on every visit.
// The structural recognizer establishes this bijection to executed alternatives.
// Multi-alternative slots have null concrete anchors/cuts: these views describe
// abstract slot order only; distributed endpoint preparation is separate.
// As above, exactTrips is a supplied uniform domain binding. All numerical
// upper/lower/native facts must hold for EVERY alternative in the same slot.
std::shared_ptr<const CompactFixedBodyContext> captureBalancedCompactFixedBodyContext(
    scf::ForOp loop, const SyncInput& input, const PhaseIndex& index,
    std::shared_ptr<RegionExpressions> arena, RegionExpressions::Id exactTrips, std::string& error);

struct CompactOrderBounds {
    // Deep snapshots survive mutation/destruction of caller-owned result structs.
    std::shared_ptr<const CompactWriterReaderAnalysis> mathematical;
    std::shared_ptr<const CompactLowerFacts> lowerFacts;
    std::shared_ptr<const CompactFixedBodyContext> domain;
    PeriodicExcessSnapshot upperGraph, lowerGraph;
    BoundingRegionalResult bounds;
    PeriodicExcessCertificate certificate;
    // Unavailable export/certificate never erases mathematical records. No
    // endpoint/storage summary is manufactured by this query-only adapter.
    std::string exportError;
};
// The producer certifies upper covers and lower facts are contained in the
// ORIGINAL shared model on domain's namespace/native context. The upper must
// be the completed M3 result (including actual local-barrier strengthening).
// Empty successful lower facts give the common native graph. Finite trips only
// instantiate the excess count; equivalence promotion uses uniform frontiers.
// Every RegionalOrderView captures the same immutable snapshot used by its
// certificate. Index construction costs are the existing periodic analyzer's;
// each exported symbolic event query uses O(1) threshold and circuit operations.
// No trip expansion, sampled promotion, payload mutation or placement is done.
CompactOrderBounds buildCompactOrderBounds(std::shared_ptr<const CompactFixedBodyContext> domain,
    const CompactWriterReaderAnalysis& upper, const CompactLowerFacts& lower, uint64_t certificateTrips);
} // namespace mlir::pto::frontiersynch
#endif
