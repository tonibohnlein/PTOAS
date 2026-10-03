// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Owned mathematical relations; original MLIR/shared schema remain the source.
#ifndef PTO_FRONTIERSYNCH_GENERAL_QUERIES_H
#define PTO_FRONTIERSYNCH_GENERAL_QUERIES_H
#include "DirectEmissionInternal.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
namespace mlir::pto::frontiersynch {
AnalysisContract generalAnalysisContract();
// Exact composition of Event->Event relations aligned to the supplied original
// schema. Literal intermediate event identities avoid unrelated disjunct joins;
// nonliteral identities retain ordinary Presburger composition semantics.
presburger::PresburgerRelation composeGeneralEventRelations(
    presburger::PresburgerRelation first, const presburger::PresburgerRelation& second,
    SymbolicSchemaHandle schema);
struct GeneralEndpoint {
    PipelineType source, target;
    bool outgoing = false;
    presburger::PWMAFunction function;
    GeneralEndpoint(PipelineType source, PipelineType target, bool outgoing, presburger::PWMAFunction function)
        : source(source), target(target), outgoing(outgoing), function(std::move(function)) {}
};
struct GeneralEndpointPlan { SmallVector<GeneralEndpoint, 0> endpoints; unsigned arithmeticWidth = 0; };
struct GeneralQueries {
    SymbolicSchemaHandle schema;
    // Uniform original-parameter relations, never parameter-bound artifacts.
    presburger::PresburgerRelation context, present, native, minimum, reachability;
    SymbolicAnalysisHandle premises;
    GeneralQueries(SymbolicSchemaHandle schema, presburger::PresburgerRelation context,
        presburger::PresburgerRelation present, presburger::PresburgerRelation native,
        presburger::PresburgerRelation minimum, presburger::PresburgerRelation reachability)
        : schema(std::move(schema)), context(std::move(context)), present(std::move(present)),
          native(std::move(native)), minimum(std::move(minimum)), reachability(std::move(reachability)) {}
};
LogicalResult buildGeneralCounted(SelectedAnalysis& selected, const SyncInput& input, CostLedger& costs,
                                 std::string& reason);
FailureOr<std::shared_ptr<const GeneralQueries>> generalQueries(const SelectedAnalysis& selected, std::string& reason);
LogicalResult prepareGeneralEndpoints(SelectedAnalysis& selected, std::string& reason);
LogicalResult emitGeneralEndpoints(IRMapping& mapping, const SelectedAnalysis& selected, DirectEmissionResult& result);
FailureOr<SelectedAnalysisHandle> composeGeneralRegions(func::FuncOp function, StructuredInputHandle input,
    ArrayRef<SelectedAnalysisHandle> children, CostLedger& costs, std::string& reason);
} // namespace mlir::pto::frontiersynch
#endif
