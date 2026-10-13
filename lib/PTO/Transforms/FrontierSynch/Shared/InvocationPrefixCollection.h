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
#include "PrefixOccurrence.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include <set>
namespace mlir::pto::frontiersynch {
class RegionExpressions;
struct MathematicalResult;
struct ExplicitAnalysis;
struct RegionalAnalysis;
struct SequenceAnalysisState;
// One collector owns the complete original invocation. A regional contributor
// cannot materialize it or silently mix direct commands in selected directions.
class InvocationPrefixCollection {
public:
    explicit InvocationPrefixCollection(func::FuncOp original, std::shared_ptr<const SyncInput> input = {})
        : original(original), input(std::move(input)) {}
    LogicalResult bindInput(std::shared_ptr<const SyncInput> owner, std::string& error);
    func::FuncOp function() const { return original; }
    InvocationPrefixRecipe& recipe() { return pending; }
    LogicalResult source(const CompoundInstanceElement* phase,
        std::function<LogicalResult(OpBuilder&, const IRMapping&, llvm::MutableArrayRef<Value>,
                                    Value, std::string&)> snapshot, std::string& error);
    LogicalResult consumer(Operation* original, PrefixDirection direction,
                           PrefixIncomingSelector selector, std::string& error);
    // Snapshot an unconditional direct-body Boolean at an invocation-invariant
    // visit ordinal of a unit-step loop. The caller proves availability at reads.
    FailureOr<std::pair<std::size_t, std::size_t>> snapshotPredicate(
        Value original, scf::ForOp loop, std::shared_ptr<RegionExpressions> arena,
        RegionExpressions::Id ordinal, std::string& error);
    // Bounds and completeness are certified by the root coordinator after all
    // selected child and crossing records have contributed successfully.
private:
    func::FuncOp original;
    std::shared_ptr<const SyncInput> input;
    InvocationPrefixRecipe pending;
    std::map<const CompoundInstanceElement*, std::size_t> sources;
    std::map<std::pair<Operation*, uint32_t>, const CompoundInstanceElement*> sourceCuts;
    std::map<std::pair<Operation*, PrefixDirection>, std::size_t> consumers;
    std::set<PrefixDirection> directions;
    std::map<std::tuple<const void*, const RegionExpressions*, RegionExpressions::Id>,
             std::pair<std::size_t, std::size_t>> predicateSnapshots;
};
// The callback owns producer-local evidence and proper child owners, never its
// own MathematicalResult. The source owner keeps borrowed phase records alive.
struct RegionalPrefixContribution {
    std::shared_ptr<const SyncInput> input;
    std::shared_ptr<RegionExpressions> expressions;
    func::FuncOp function;
    const SyncStorageEffects* accessModel = nullptr;
    GMAliasPolicy aliasPolicy = GMAliasPolicy::MayAlias;
    std::function<LogicalResult(InvocationPrefixCollection&, std::string&)> collect;
    // The directory accepts only exact exported occurrence identities. Missing
    // provenance is unavailable, never permission to infer a primitive leaf.
    std::function<PrefixOccurrencePtr(RegionalEvent, std::string&)> occurrence;
    std::function<bool(const RegionalAnalysis&)> matchesOccurrenceFrame;
};
std::shared_ptr<const RegionalPrefixContribution> sequencePrefixContribution(
    const RegionalAnalysis& region, std::shared_ptr<SequenceAnalysisState> state);
LogicalResult collectRegionalInvocationPrefix(const RegionalAnalysis& region,
    InvocationPrefixCollection& collection, std::string& error);
LogicalResult appendExplicitPrefixRealization(const ExplicitAnalysis& retained,
                                             InvocationPrefixCollection& collection, std::string& error);
LogicalResult collectInvocationPrefix(std::shared_ptr<const MathematicalResult> retained,
                                      InvocationPrefixCollection& collection, std::string& error);
} // namespace mlir::pto::frontiersynch
#endif
