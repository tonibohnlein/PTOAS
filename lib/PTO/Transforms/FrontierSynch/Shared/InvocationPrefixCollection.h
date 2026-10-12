// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_INVOCATIONPREFIXCOLLECTION_H
#define PTO_FRONTIERSYNCH_INVOCATIONPREFIXCOLLECTION_H
#include "InvocationPrefix.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include <set>
namespace mlir::pto::frontiersynch {
struct MathematicalResult;
struct ExplicitAnalysis;
// One collector owns the complete original invocation. A regional contributor
// cannot materialize it or silently mix direct commands in selected directions.
class InvocationPrefixCollection {
public:
    explicit InvocationPrefixCollection(func::FuncOp original) : original(original) {}
    func::FuncOp function() const { return original; }
    InvocationPrefixRecipe& recipe() { return pending; }
    LogicalResult source(const CompoundInstanceElement* phase,
        std::function<LogicalResult(OpBuilder&, const IRMapping&, llvm::MutableArrayRef<Value>,
                                    Value, std::string&)> snapshot, std::string& error);
    LogicalResult consumer(Operation* original, PrefixDirection direction,
                           PrefixIncomingSelector selector, std::string& error);
    // Bounds and completeness are certified by the root coordinator after all
    // selected child and crossing records have contributed successfully.
private:
    func::FuncOp original;
    InvocationPrefixRecipe pending;
    std::map<const CompoundInstanceElement*, std::size_t> sources;
    std::map<std::pair<Operation*, uint32_t>, const CompoundInstanceElement*> sourceCuts;
    std::map<std::pair<Operation*, PrefixDirection>, std::size_t> consumers;
    std::set<PrefixDirection> directions;
};
LogicalResult appendExplicitPrefixRealization(const ExplicitAnalysis& retained,
                                             InvocationPrefixCollection& collection, std::string& error);
LogicalResult collectInvocationPrefix(std::shared_ptr<const MathematicalResult> retained,
                                      InvocationPrefixCollection& collection, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
