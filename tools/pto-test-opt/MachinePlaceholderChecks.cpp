// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Sparse machine recipe names must not allocate by the configured limit.
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticProgramInternal.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <climits>
using namespace mlir;
namespace fs = pto::frontiersynch;
namespace {
bool contains(const fs::IntegerSystem& system, ArrayRef<fs::BoundInteger> point)
{
    if (system.isKnownEmpty() || system.dimensions() != point.size()) { return false; }
    for (const auto& row : system.constraints()) {
        fs::BoundInteger value(0);
        for (unsigned i = 0; i < point.size(); ++i) { value += row.coefficients[i] * point[i]; }
        if (value > row.bound) { return false; }
    }
    for (const auto& row : system.congruences()) {
        fs::BoundInteger value(0);
        for (unsigned i = 0; i < point.size(); ++i) { value += row.coefficients[i] * point[i]; }
        if ((value - row.residue) % row.modulus != 0) { return false; }
    }
    return true;
}
bool check(MLIRContext* context, func::FuncOp function, const fs::PhaseIndex& index, unsigned dimensions)
{
    fs::ArithmeticProgram output;
    output.context = {function, function};
    const fs::ArithmeticLimits limits{8, dimensions, 1, 4096};
    fs::detail::ProgramBuilder builder{output, limits, context, index, DenseMap<Value, unsigned>(), {}};
    Value first, second;
    function.walk([&](Operation* op) {
        if (op->hasAttr("test.first")) { first = op->getResult(0); }
        if (op->hasAttr("test.second")) { second = op->getResult(0); }
    });
    fs::ArithmeticSite site;
    if (!first || !second || !builder.prepareValue(first, site)) { return false; }
    const auto a = builder.value(first, site, 0);
    // Add a true parameter after creating the first sparse recipe identity.
    const auto parameter = function.getArgument(1);
    if (!builder.prepareValue(parameter, site) || !builder.prepareValue(second, site)) { return false; }
    const auto b = builder.value(second, site, 0);
    if (!a || !b || a == b || builder.machineBinding(a) != 0 || builder.machineBinding(b) != 1 ||
        output.parameters.size() != 2 || builder.machineBinding(getAffineSymbolExpr(0, context))) { return false; }
    auto relation = builder.relation(fs::PrimitiveKind::Context, 0);
    builder.emit(relation, {b - a - 1 + builder.value(parameter, site, 0)});
    if (!builder.constructionReady() || !output.machinePrimitiveConversions) { return false; }
    for (int64_t x : {-10LL, -9LL, -4LL, -3LL, -2LL, -1LL, 0LL, 1LL, 2LL, 3LL, 8LL, 9LL,
                      static_cast<long long>(INT32_MIN), static_cast<long long>(INT32_MAX)}) {
        for (int64_t other : {-1, 0, 1}) {
            const bool expected = (x + 1) / 3 - x / 3 - 1 + other >= 0;
            bool actual = false;
            for (const auto& piece : relation.pieces) {
                if (!piece.machineSystem) { return false; }
                actual |= contains(*piece.machineSystem, {fs::BoundInteger(x), fs::BoundInteger(other)});
            }
            if (actual != expected) { return false; }
        }
    }
    return true;
}
} // namespace
bool runMachinePlaceholderChecks(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>(R"mlir(
module { func.func @sparse(%raw: i32, %other: index) {
  %x = arith.index_cast %raw : i32 to index
  %one = arith.constant 1 : index
  %three = arith.constant 3 : index
  %a = arith.divsi %x, %three {test.first} : index
  %next = arith.addi %x, %one : index
  %b = arith.divsi %next, %three {test.second} : index
  return
}})mlir", context);
    if (!module || failed(verify(*module))) { return false; }
    const auto function = module->lookupSymbol<func::FuncOp>("sparse");
    fs::PhaseIndex index;
    if (failed(index.build(function, ArrayRef<const pto::CompoundInstanceElement*>{}))) { return false; }
    if (!check(context, function, index, 8) || !check(context, function, index, UINT_MAX)) { return false; }
    llvm::outs() << "machine placeholders: sparse names, late parameters, two recipes "
                    "and unlimited dimension profile passed\n";
    return true;
}
