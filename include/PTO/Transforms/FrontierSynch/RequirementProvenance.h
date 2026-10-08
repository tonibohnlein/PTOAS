// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Guarded ownership before global identity merging or demand reduction.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REQUIREMENTPROVENANCE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REQUIREMENTPROVENANCE_H
#include "PTO/Transforms/FrontierSynch/BoundingRegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/LifetimeScan.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace mlir::pto::frontiersynch {
using RequirementGroupId = uint64_t;
struct RequirementRecordKey {
    uint64_t producer = 0;
    uint64_t originalRecord = 0;
    bool operator==(const RequirementRecordKey& other) const
    {
        return producer == other.producer && originalRecord == other.originalRecord;
    }
};
enum class RequirementScope { Internal, Boundary, FixedPrerequisite };
struct RequirementGroup {
    RequirementGroupId id = 0;
    RequirementScope scope = RequirementScope::Internal;
};
struct RequirementMembership {
    RequirementGroupId group = 0;
    RequirementRecordKey record;
    RegionExpressions::Id guard = RegionExpressions::invalid;
};
struct RequirementExactRecord {
    RequirementRecordKey record;
    RegionExpressions::Id guard = RegionExpressions::invalid;
};
class RequirementProvenance;
using RequirementSnapshot = std::shared_ptr<const RequirementProvenance>;
class RequirementReplacementEvidence;
using RequirementReplacement = std::shared_ptr<const RequirementReplacementEvidence>;

// Keys name immutable records in their original producer namespace, not endpoint
// IDs or positions in a reduced list. The caller retains the corresponding
// producer descriptions for the snapshot lifetime. Groups identify ALL original
// model requirements in their named scope, including supplied prerequisites.
// Initial memberships are conservative generators covering their groups; this
// helper validates representation only, not that mathematical coverage premise.
RequirementSnapshot createRequirementProvenance(OrderContext context,
    std::vector<RequirementGroup> groups, std::vector<RequirementMembership> memberships, std::string& error);

// Capture original scan witnesses, never reduction.retained. atomGroups maps
// each scan atom to an Internal group; several atoms may share one group.
// prerequisiteGroup must be distinct and is always FixedPrerequisite. Each
// original generator needs a witness; shared endpoints keep all contributions.
// The producer namespace and original scan generator index form the record key.
RequirementSnapshot captureStorageRequirementProvenance(OrderContext context, uint64_t producer,
    const StorageScanResult& scan, llvm::ArrayRef<RequirementGroupId> atomGroups,
    RequirementGroupId prerequisiteGroup, std::string& error);

class RequirementProvenance {
public:
    const OrderContext& context() const { return owner; }
    llvm::ArrayRef<RequirementGroup> groups() const { return *groupDefinitions; }
    llvm::ArrayRef<RequirementMembership> originalMemberships() const { return *original; }
    llvm::ArrayRef<RequirementMembership> memberships() const { return selected; }
    // Exact replacement facts to add to any separately certified lower graph.
    // These facts insert no commands. Empty means no new lower facts supplied.
    llvm::ArrayRef<RequirementMembership> lowerFacts() const { return definite; }
private:
    RequirementProvenance() = default;
    OrderContext owner;
    std::shared_ptr<const std::vector<RequirementGroup>> groupDefinitions;
    std::shared_ptr<const std::vector<RequirementMembership>> original;
    std::vector<RequirementMembership> selected, definite;
    friend RequirementSnapshot createRequirementProvenance(OrderContext,
        std::vector<RequirementGroup>, std::vector<RequirementMembership>, std::string&);
    friend RequirementSnapshot applyRequirementReplacement(
        RequirementSnapshot, RequirementReplacement, std::string&);
};

// TRUSTED mathematical-producer boundary, not an equivalence proof checker.
// The producer establishes, on every admitted execution satisfying domain,
// TC(B union replacement) = TC(B union D_group), where D_group comprises all
// ORIGINAL model requirements of the entire named group. Equality with a
// strengthened upper graph is insufficient. Native order, occurrence identities,
// presence, alias assumptions and surrounding context remain unchanged.
// Records/guards refer to the context's common occurrence and expression frame;
// this factory neither derives these facts from IDs nor checks arbitrary closure.
// Concrete proof-producing adapters are a separate integration obligation.
RequirementReplacement bindTrustedRequirementReplacement(RequirementSnapshot snapshot,
    RequirementGroupId group, RegionExpressions::Id domain,
    std::vector<RequirementExactRecord> replacement, std::string& error);

class RequirementReplacementEvidence {
private:
    RequirementReplacementEvidence() = default;
    RequirementSnapshot snapshot;
    OrderContext context;
    RequirementGroupId group = 0;
    RegionExpressions::Id domain = RegionExpressions::invalid;
    std::vector<RequirementExactRecord> replacement;
    friend RequirementReplacement bindTrustedRequirementReplacement(RequirementSnapshot,
        RequirementGroupId, RegionExpressions::Id, std::vector<RequirementExactRecord>, std::string&);
    friend RequirementSnapshot applyRequirementReplacement(
        RequirementSnapshot, RequirementReplacement, std::string&);
};

// Requires the identical immutable snapshot used by the trusted producer.
// For that group only, alpha becomes alpha AND NOT domain; replacement guards
// become guard AND domain. Other ownership (including shared records) survives.
// FixedPrerequisite groups cannot be replaced. Original ownership is retained.
// No global deduplication/reduction, query rebuilding, endpoint placement or IR
// mutation occurs here. A consumer must rebuild selected graph interfaces.
// Work/added DAG size is linear in memberships plus replacement records, beyond
// validation's ordered group-directory lookups. Failure publishes no snapshot.
RequirementSnapshot applyRequirementReplacement(
    RequirementSnapshot snapshot, RequirementReplacement evidence, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
