// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Invocation-owned regional attempts over original MLIR, not a program IR.
// The source and shared input stay unchanged while this request cache is alive.
#ifndef PTO_FRONTIERSYNCH_REGIONAL_REQUESTS_H
#define PTO_FRONTIERSYNCH_REGIONAL_REQUESTS_H
#include "DirectEmissionInternal.h"
#include "llvm/ADT/SmallPtrSet.h"
namespace mlir::pto::frontiersynch {
enum class RegionalStatus { Ready, NotApplicable, UnmetObligation };
enum class RegionalMode { Modeled, MinimumExact };
// Static selector/predicate/local-placement qualification. Physical preflight
// and actual staged emission still establish ExecutableEndpoints separately.
enum class RegionalPreparation { Mathematical, SelectorMatching };
enum class RegionalRepresentation { NativeSummaries, ArithmeticRelations, DifferenceRelations, PresburgerRelations };
struct RegionalContext {
    Operation* root = nullptr;
    // Original control operands qualify every region invocation; branch arms
    // are analyzed under their entry guard, never flattened into parent traces.
    std::shared_ptr<const SmallVector<Value>> controlValues;
    Operation* scope = nullptr;
    SmallVector<Region*> entryRegions;
private:
    friend class RegionalRequests;
    RegionalContext() = default;
};
using RegionalContextHandle = std::shared_ptr<const RegionalContext>;
struct RegionalRequestResult {
    SelectedAnalysisHandle analysis;
    std::string outcome, obligation;
    SmallVector<SelectedAnalysisHandle> children;
};
class RegionalRequests {
public:
    RegionalRequests(
        func::FuncOp function, const SyncInput& input, const TraceDemandAnalysis& trace, CostLedger& costs);
    RegionalContextHandle context() const { return admitted; }
    RegionalContextHandle contextFor(Operation* owner);
    std::size_t signatureMembershipQueries() const { return membershipQueries; }
    RegionalRequestResult request(Operation* owner, RegionalContextHandle context, RegionalMode mode,
                                  const AnalysisNeeds& needs,
                                  RegionalRepresentation representation = RegionalRepresentation::NativeSummaries,
                                  RegionalPreparation preparation = RegionalPreparation::Mathematical,
                                  RegionalRoutePolicy policy = RegionalRoutePolicy::Economical);
    ArrayRef<AnalysisAttempt> attempts() const { return history; }
    std::size_t cacheHits() const { return hits; }
    std::size_t regionCount() const { return visitedRegions.size(); }

private:
    FailureOr<StructuredInputHandle> arithmeticInput(Operation* owner, std::string& reason);
    FailureOr<SignedInputs> arithmeticPrimitives(Operation* owner, ArithmeticClass arithmetic, std::string& reason);
    struct Signature {
        SmallVector<std::size_t> sites;
        SmallVector<Operation*> children;
        llvm::SmallPtrSet<Operation*, 8> childMembership, regionChildren;
        bool control = false, loopFree = true;
    };
    struct Candidate {
        std::shared_ptr<SelectedAnalysis> analysis;
        RegionalStatus status = RegionalStatus::NotApplicable;
        std::string obligation;
    };
    using Key = std::tuple<Operation*, const RegionalContext*, RegionalMode, RegionalRepresentation,
                           RegionalPreparation, RegionalRoutePolicy,
                           SelectedReduction, bool, bool, std::uint32_t>;
    Candidate explicitRegion(Operation* owner, const Signature& signature);
    Candidate generalCountedRegion(Operation* owner, const Signature& signature);
    Candidate generalComposeRegion(Operation* owner, const Signature& signature,
                                   RegionalMode mode, const AnalysisNeeds& needs,
                                   RegionalPreparation preparation, RegionalRoutePolicy policy);
    Candidate countedRegion(Operation* owner, const Signature& signature);
    Candidate upperFixedBodyRegion(Operation* owner, const Signature& signature, const AnalysisNeeds& needs);
    Candidate periodicRegion(Operation* owner, const Signature& signature);
    Candidate boundaryLoopRegion(Operation* owner, const Signature& signature, const AnalysisNeeds& needs);
    Candidate stationaryRegion(Operation* owner, const Signature& signature);
    FailureOr<SmallVector<const CompoundInstanceElement*>> fixedLoopPhases(
        scf::ForOp loop, const Signature& signature, std::string& reason);
    Candidate arithmeticRegion(Operation* owner, const Signature& signature, ArithmeticClass arithmetic);
    Candidate guardedRegion(Operation* owner, const Signature& signature);
    LogicalResult qualifyMatching(Operation* owner, SelectedAnalysis& selected, std::string& reason);
    std::shared_ptr<const PhaseIndex> placements;
    Candidate composeRegion(
        Operation* owner, const Signature& signature, RegionalMode mode, const AnalysisNeeds& needs,
        RegionalPreparation preparation, RegionalRoutePolicy policy);
    std::unique_ptr<EndpointRelationImporter> importer;
    SignedSpaceHandle arithmeticSpace;
    std::map<Operation*, std::pair<StructuredInputHandle, std::string>> imported;
    std::map<std::pair<Operation*, ArithmeticClass>, std::pair<std::optional<SignedInputs>, std::string>> primitives;
    DenseMap<Operation*, RegionalContextHandle> regionalContexts;
    void record(Operation* owner, StringRef route, StringRef outcome, StringRef reason);
    func::FuncOp function;
    const SyncInput& input;
    const TraceDemandAnalysis& trace;
    CostLedger& costs;
    RegionalContextHandle admitted;
    DenseMap<Operation*, Signature> signatures;
    std::map<Key, RegionalRequestResult> cache;
    // Context owners retained: raw identity keys never dangle or get recycled.
    SmallVector<RegionalContextHandle> contexts;
    SmallVector<AnalysisAttempt> history;
    SmallVector<std::chrono::steady_clock::time_point> attemptStarts;
    llvm::SmallPtrSet<Region*, 16> visitedRegions;
    std::size_t membershipQueries = 0;
    std::size_t hits = 0;
};
} // namespace mlir::pto::frontiersynch
#endif
