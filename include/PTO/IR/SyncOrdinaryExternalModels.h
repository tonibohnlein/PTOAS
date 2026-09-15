// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_IR_SYNCORDINARYEXTERNALMODELS_H
#define PTO_IR_SYNCORDINARYEXTERNALMODELS_H
#include "PTO/IR/PTO.h"
#include "PTO/IR/PTOTypeUtils.h"
#include "mlir/IR/DialectRegistry.h"
namespace mlir::pto {
namespace sync_semantic_models {
// These are explicit IR-owned interface implementations, not planner admission
// tests. They only declare audited ordinary variants; effect import and target
// validation remain required. Register through a dialect extension before the
// pass manager starts parallel execution; never mutate interfaces per function.
template <typename Op>
struct VectorArithmetic : SinglePhaseSyncOpInterface::ExternalModel<VectorArithmetic<Op>, Op> {
  bool hasCompleteSinglePhaseSyncEffects(Operation *operation) const {
    return getTargetArch(operation) == PTOArch::A3 &&
           cast<Op>(operation).getPipe() == PIPE::PIPE_V && operation->getNumResults() == 0;
  }
};
struct OrdinaryStore : SinglePhaseSyncOpInterface::ExternalModel<OrdinaryStore, TStoreOp> {
  bool hasCompleteSinglePhaseSyncEffects(Operation *operation) const {
    auto store = cast<TStoreOp>(operation);
    return getTargetArch(operation) == PTOArch::A3 && store.getPipe() == PIPE::PIPE_MTE3 && !store.getFp() &&
           !store.getPreQuantScalar() && !store.getResult() &&
           store.getStPhase() == STPhase::Unspecified &&
           store.getAtomicType() == AtomicType::AtomicNone &&
           store.getReluPreMode() == ReluPreMode::NoRelu;
  }
};
template <typename Op, typename Model>
inline void attach(MLIRContext &context) {
  if (!OperationName(Op::getOperationName(), &context).hasInterface<SinglePhaseSyncOpInterface>())
    Op::template attachInterface<Model>(context);
}
} // namespace sync_semantic_models
inline void registerSyncOrdinaryExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, PTODialect *) {
    using namespace sync_semantic_models;
    attach<TAbsOp, VectorArithmetic<TAbsOp>>(*context);
    attach<TMulOp, VectorArithmetic<TMulOp>>(*context);
    attach<TSubOp, VectorArithmetic<TSubOp>>(*context);
    attach<TStoreOp, OrdinaryStore>(*context);
  });
}
} // namespace mlir::pto
#endif
