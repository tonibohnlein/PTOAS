// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Transaction-local state; no mutable plan or certificate escapes its owner.
#ifndef PTO_FRONTIERSYNCH_PREFIXPREPARATION_H
#define PTO_FRONTIERSYNCH_PREFIXPREPARATION_H
#include "PTO/Transforms/FrontierSynch/Shared/PrefixServices.h"
#include "PTO/Transforms/FrontierSynch/Shared/PrefixInput.h"
#include "PTO/Transforms/FrontierSynch/Shared/PrefixRecipe.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhysicalAllocation.h"
#include "PTO/Transforms/FrontierSynch/Shared/PhaseIndex.h"
#include "PTO/IR/PTOSyncCapabilities.h"
#include "CompletionInlineChild.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch::prefix {
namespace fs = mlir::pto::frontiersynch;
struct Command {
    Operation* cut = nullptr;
    bool barrier = false, publish = false;
    uint32_t source = 0, target = 0;
    std::shared_ptr<const fs::PhysicalEndpointMap> mapping;
    int64_t physicalId = -1;
};
class Preparation {
public:
    explicit Preparation(func::FuncOp function, pto::GMAliasPolicy policy, const fs::PrefixOptions& options)
        : function(function), input(policy), pruning(options.prune), aliasPolicy(policy),
          childRegion(options.childRegion), childProvider(options.childProvider), pool(options.eligibleIds)
    {}
    LogicalResult prepare();
    LogicalResult emit();
    fs::PrefixResult result;
    fs::PrefixResult takeResult()
    {
        result.work = std::move(work);
        return std::move(result);
    }

private:
    func::FuncOp function;
    pto::SyncInput input;
    fs::PhaseIndex index;
    SmallVector<Command> commands;
    std::string architecture;
    pto::SyncPhysicalCore core = pto::SyncPhysicalCore::Unknown;
    bool pruning = false;
    pto::GMAliasPolicy aliasPolicy;
    std::optional<std::size_t> childRegion;
    std::function<std::shared_ptr<const fs::CompletionInlineChild>(func::FuncOp)> childProvider;
    std::shared_ptr<const fs::CompletionInlineChild> child;
    std::optional<fs::CompletionInlineChild::PrefixBoundary> boundary;
    std::unique_ptr<fs::PreparedLogicalPlan> childPlan;
    fs::ServiceMask entryMasks;
    std::vector<uint32_t> entryPipes;
    LogicalResult prepareChild(fs::PrefixInput&, fs::ServiceMask&);
    LogicalResult appendCommands(Operation*, uint32_t, const std::vector<uint32_t>&,
                                 const pto::CompoundInstanceElement* phase = nullptr);
    bool insideChild(Operation*) const;
    uint64_t nativeDischarges = 0, selectedSelf = 0, selectedCross = 0;
    double recoverySeconds = 0, indexSeconds = 0, mapsSeconds = 0;
    double nativeSeconds = 0, physicalMapSeconds = 0;
    llvm::json::Object work;
    SmallVector<int64_t> pool;
    using RoleKey = std::tuple<uint32_t, uint32_t, bool>;
    std::map<RoleKey, fs::PhysicalRecordAllocation> assignments;
    bool refine(const fs::PrefixInput&, const fs::SourcePipeServices&, fs::ServiceMask&);
    LogicalResult reserveRoles(const fs::PrefixInput&, const fs::ServiceMask&);
    LogicalResult constructCommands(const fs::PrefixInput&, const fs::ServiceMask&);
    LogicalResult qualify();
    bool verifyProtocol() const;
    bool protectPaths(const fs::PrefixInput& description, const fs::ServiceMask& baseline, fs::ServiceMask& paths);
    LogicalResult refuse(fs::PrefixRefusal reason, llvm::StringRef detail)
    {
        result.refusal = reason;
        result.reason = detail.str();
        function.emitError("current-cut: ") << fs::prefixRefusalName(reason) << ": " << detail;
        return failure();
    }
};
} // namespace mlir::pto::frontiersynch::prefix
#endif
