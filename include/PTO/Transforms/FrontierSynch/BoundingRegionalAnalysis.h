// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Mathematical order bounds on one unchanged shared input. Endpoint availability
// is independent of these bounds; this interface owns no insertion capability.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDINGREGIONALANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_BOUNDINGREGIONALANALYSIS_H
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
namespace mlir::pto {
class SyncInput;
namespace frontiersynch {
enum class InputOrderGuarantee { InputOrderEquivalent, InputOrderCovering };
enum class ReductionQuality { Covers, Partial, Generators };
class RegionalOrderContext;
class RequirementProvenance;
using OrderContext = std::shared_ptr<const RegionalOrderContext>;

// The arena and canonical callbacks are owned; SyncInput and original IR are
// borrowed and must remain alive and unchanged. The producer binds its fixed
// native prerequisites to this identity too. A context is not a graph snapshot:
// several selected graphs may share it. Graph proofs also bind their snapshot.
class RegionalOrderContext {
public:
    const std::shared_ptr<RegionExpressions>& expressions() const { return canonical.expressions; }
    const SyncStorageEffects* accessModel() const { return canonical.accessModel; }
    GMAliasPolicy gmAliasPolicy() const { return canonical.gmAliasPolicy; }
    const SyncInput& input() const { return *source; }
    const RegionalAnalysis& domain() const { return canonical; }
private:
    RegionalOrderContext(const SyncInput& input, RegionalAnalysis domain);
    const SyncInput* source;
    RegionalAnalysis canonical;
    friend FailureOr<OrderContext> captureRegionalOrderContext(
        const SyncInput&, const RegionalAnalysis&, std::string&);
};

// Requires successfully built input. Checks arena, shared modeled accesses,
// alias policy and occurrence metadata, never effect precision or endpoints.
// Presence/reference callbacks are captured once, not compared semantically.
// A domain-changing adapter must capture a new context for its mapped domain.
FailureOr<OrderContext> captureRegionalOrderContext(
    const SyncInput& input, const RegionalAnalysis& domain, std::string& error);

struct RegionalOrderQueries {
    // Exact closure of the selected graph on this context, including endpoint
    // presence, reflexivity and unchanged native prerequisites. Unavailable
    // individual queries return nullopt. All answers belong to the context arena.
    std::function<std::optional<RegionExpressions::Id>(RegionalEvent, RegionalEvent)> reachability;
    // Optional certified index for THIS selected graph. No index is inherited
    // from the canonical domain; explicitly reusing one requires equal order.
    std::shared_ptr<const RegionalNumericalInterface> numerical;
};

// Factory creation ensures every view derives its occurrence callbacks from
// the canonical context. Numerical structure validation costs O(kP) for k chains
// and P ports; graph/query certification remains the producer's obligation.
class RegionalOrderView {
public:
    const OrderContext& context() const { return owner; }
    const RegionalAnalysis& regional() const { return *analysis; }
private:
    RegionalOrderView(OrderContext context, std::shared_ptr<const RegionalAnalysis> regional);
    OrderContext owner;
    std::shared_ptr<const RegionalAnalysis> analysis;
    friend FailureOr<RegionalOrderView> makeRegionalOrderView(
        OrderContext, RegionalOrderQueries, std::string&);
};
FailureOr<RegionalOrderView> makeRegionalOrderView(
    OrderContext context, RegionalOrderQueries queries, std::string& error);

struct BoundingRegionalResult {
    OrderContext context;
    // Preserve supplied record ownership even when query/endpoint exports are
    // unavailable. Original typed record definitions remain producer-owned.
    std::shared_ptr<const RequirementProvenance> provenance;
    // Missing means unavailable query export, never an empty graph. Mathematical
    // records/provenance remain with their producer even when both are missing.
    std::optional<RegionalOrderView> lower, upper;
    // Producer certificate: Hlower <= Hinput <= Hupper, all relative to the
    // unchanged shared modeled effects. Equivalent additionally means
    // Hupper == Hinput. This never classifies addresses or governs admission.
    InputOrderGuarantee guarantee = InputOrderGuarantee::InputOrderCovering;
    // Reduction quality describes the selected mathematical generators only.
    ReductionQuality reduction = ReductionQuality::Generators;
};
// Structural validation only: view/provenance context identity and enum values. Does not prove
// closure containment or require exports. Insertion returns separate actual
// placement bounds when a consumer-adjacent local barrier strengthens order.
LogicalResult validateBoundingRegionalResult(const BoundingRegionalResult& result, std::string& error);
} // namespace frontiersynch
} // namespace mlir::pto
#endif
