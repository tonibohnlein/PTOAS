// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared state for numeric-template control specialization and exact effects.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATEINTERNAL_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_NUMERICTEMPLATEINTERNAL_H
#include "PTO/Transforms/FrontierSynch/NumericTemplate.h"
namespace mlir::pto::frontiersynch::detail {
struct TemplateBuilder {
    NumericTemplate& output;
    const PhaseIndex& index;
    const SyncInput& input;
    DenseMap<Value, int64_t> coordinates;
    SmallVector<TemplateCoordinate> path;
    uint64_t visits = 0;
    uint64_t phases = 0;
    TemplateGeometryConstant geometryConstant;
    TemplateControlConstant controlConstant;
    MLIRContext* context() const { return output.outer.getContext(); }
    AffineExpr scalar(Value value, SmallVectorImpl<Value>* invariants = nullptr,
                      bool control = false) const;
    std::optional<int64_t> integer(Value value) const;
    std::optional<bool> guard(Value value, unsigned depth = 0) const;
    bool charge(uint64_t count, uint64_t& total, uint64_t limit, Operation* anchor);
    bool block(Block& body, bool emit, unsigned depth);
    bool loop(scf::ForOp loop, bool emit, unsigned depth);
    bool payload(const CompoundInstanceElement* phase);
};
bool prepareTemplateEffects(TemplateBuilder& builder);
} // namespace mlir::pto::frontiersynch::detail
#endif
