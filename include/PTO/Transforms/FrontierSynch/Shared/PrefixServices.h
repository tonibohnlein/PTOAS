// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Standalone conservative prefix route. Requires module/symbol-table ownership.
#ifndef PTO_FRONTIERSYNCH_PREFIXSERVICES_H
#define PTO_FRONTIERSYNCH_PREFIXSERVICES_H
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include "PTO/Transforms/FrontierSynch/ResultGuarantees.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include <functional>
#include <optional>
namespace mlir::pto::frontiersynch {
class CompletionInlineChild;
enum class PrefixRefusal { None, Admission, Visibility, Readiness, CutOrDirection, Resources, Certificate, Emission };
llvm::StringRef prefixRefusalName(PrefixRefusal reason);
struct PrefixOptions {
    bool prune = false;
    std::optional<std::size_t> childRegion;
    // A trusted caller may supply an already qualified provider in the staged
    // original frame. Mutually exclusive with a session region request.
    std::function<std::shared_ptr<const CompletionInlineChild>(func::FuncOp)> childProvider;
    SmallVector<int64_t> eligibleIds;
};
struct PrefixResult {
    PrefixRefusal refusal = PrefixRefusal::None;
    std::string reason;
    // Inspection only; never used as input to analysis or preparation.
    llvm::json::Object work;
    ResultGuarantees guarantees; // No reduction/minimum-demand guarantee is supplied.
    bool accepted() const { return refusal == PrefixRefusal::None; }
};
// No original mutation on refusal. Entry roles must be empty and exclusively
// owned; manual/open protocols are refused. Optional whole-loop children require
// separately qualified boundaries, disjoint palettes and causal entry wrappers. Return ALL
// completes payload work separately from notification matching/reuse.
PrefixResult insertConservativePrefix(func::FuncOp function, GMAliasPolicy policy, const PrefixOptions& options);
// Module-owned transaction, including rollback if a later function refuses.
LogicalResult insertModuleConservativePrefix(ModuleOp module, GMAliasPolicy policy, const PrefixOptions& options);
llvm::json::Object inspectPrefixContracts(func::FuncOp function, GMAliasPolicy policy);
} // namespace mlir::pto::frontiersynch
#endif
