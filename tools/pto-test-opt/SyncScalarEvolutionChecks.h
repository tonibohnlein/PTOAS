// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Sample normalized scalar expressions independently at selected loop indices.
#ifndef PTO_TEST_SYNC_SCALAR_EVOLUTION_CHECKS_H
#define PTO_TEST_SYNC_SCALAR_EVOLUTION_CHECKS_H
#include "../../lib/PTO/Transforms/InsertSync/SyncScalarEvolution.h"
#include "mlir/IR/AffineMap.h"
namespace {
bool checkScalarSamples(mlir::Operation* op, mlir::DenseI64ArrayAttr samples)
{
    using namespace mlir;
    auto loop = op->getParentOfType<scf::ForOp>();
    if (!loop || op->getNumResults() != 1 || samples.size() % 2 != 0) {
        return false;
    }
    pto::detail::ScalarEvolution scalars(op->getContext());
    SmallVector<Value> symbols;
    auto expression = scalars.value(op->getResult(0), [&](Value v) {
        auto it = llvm::find(symbols, v);
        if (it == symbols.end()) {
            symbols.push_back(v);
            it = symbols.end() - 1;
        }
        return getAffineSymbolExpr(it - symbols.begin(), op->getContext());
    });
    // Every remaining symbol must be this loop's IV, not an opaque update or
    // iter_arg. The expected values exercise the closed form, including resets.
    if (!expression || llvm::any_of(symbols, [&](Value v) { return v != loop.getInductionVar(); })) {
        return false;
    }
    auto map = AffineMap::get(0, symbols.size(), expression);
    auto type = IntegerType::get(op->getContext(), 64);
    for (int64_t i = 0; i < samples.size(); i += 2) {
        SmallVector<Attribute> operands(symbols.size(), IntegerAttr::get(type, samples[i]));
        SmallVector<Attribute> values;
        if (failed(map.constantFold(operands, values)) || values.size() != 1 ||
            cast<IntegerAttr>(values[0]).getInt() != samples[i + 1]) {
            return false;
        }
    }
    return true;
}
bool checkOpaqueScalar(mlir::Operation* op)
{
    using namespace mlir;
    if (op->getNumResults() != 1) {
        return false;
    }
    pto::detail::ScalarEvolution scalars(op->getContext());
    SmallVector<Value> symbols;
    auto expression = scalars.value(op->getResult(0), [&](Value v) {
        symbols.push_back(v);
        return getAffineSymbolExpr(symbols.size() - 1, op->getContext());
    });
    auto symbol = dyn_cast<AffineSymbolExpr>(expression);
    return symbol && symbol.getPosition() < symbols.size() && symbols[symbol.getPosition()] == op->getResult(0);
}
bool checkScalarEvolution(mlir::func::FuncOp function)
{
    bool valid = true;
    unsigned count = 0;
    function.walk([&](mlir::Operation* op) {
        if (auto samples = op->getAttrOfType<mlir::DenseI64ArrayAttr>("test.scalar_samples")) {
            valid &= checkScalarSamples(op, samples);
            ++count;
        }
        if (op->hasAttr("test.scalar_opaque")) {
            valid &= checkOpaqueScalar(op);
            ++count;
        }
    });
    if (count) {
        llvm::outs() << "scalar-evolution " << function.getSymName() << ": " << count
                     << (valid ? " passed\n" : " FAILED\n");
    }
    return valid;
}
}
#endif
