// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Thin experimental entry points delegate to the typed library service.
#include "PTO/Transforms/FrontierSynch/Shared/PrefixServices.h"
#include "PTO/Transforms/FrontierSynch/Shared/EventIdPool.h"
using namespace mlir;
namespace fs = pto::frontiersynch;
LogicalResult emitCurrentCutServices(func::FuncOp function, pto::GMAliasPolicy policy, bool pruning)
{
    fs::PrefixOptions options;
    options.prune = pruning;
    options.eligibleIds.assign(fs::defaultEventIds().begin(), fs::defaultEventIds().end());
    // Synthetic pools remain confined to this test adapter.
    if (auto eligible = function->getAttrOfType<DenseI64ArrayAttr>("test.current_cut_ids")) {
        options.eligibleIds.assign(eligible.asArrayRef().begin(), eligible.asArrayRef().end());
        if (llvm::any_of(options.eligibleIds, [](int64_t id) { return id < 0 || id >= 6; })) {
            return function.emitError("current-cut: invalid six-ID eligible pool");
        }
    }
    auto result = fs::insertConservativePrefix(function, policy, options);
    if (!result.accepted()) {
        return failure();
    }
    llvm::errs() << "current-cut-work " << llvm::json::Value(std::move(result.work)) << "\n";
    return success();
}
llvm::json::Object inspectCurrentCutContracts(func::FuncOp function, pto::GMAliasPolicy policy)
{
    return fs::inspectPrefixContracts(function, policy);
}
