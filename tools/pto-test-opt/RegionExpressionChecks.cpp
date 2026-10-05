// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check expression algebra, total detached emission and original SSA availability.
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
std::string render(func::FuncOp function)
{
    std::string result;
    llvm::raw_string_ostream stream(result);
    function.print(stream);
    return result;
}
bool checkAlgebra(Value index, Value predicate)
{
    fs::RegionExpressions expressions;
    auto x = expressions.input(index), p = expressions.input(predicate);
    auto zero = expressions.constant(0), one = expressions.constant(1);
    auto maximum = expressions.constant(UINT64_MAX);
    auto sum = expressions.add(x, one);
    const auto before = expressions.size();
    bool valid = sum == expressions.add(one, x) && before == expressions.size();
    valid &= expressions.add(maximum, one) == zero && expressions.sub(zero, one) == maximum;
    valid &= expressions.div(maximum, one) == maximum && expressions.rem(maximum, one) == zero;
    valid &= expressions.lt(maximum, zero) == expressions.boolean(false);
    valid &= expressions.le(zero, maximum) == expressions.boolean(true);
    valid &= expressions.slt(maximum, zero) == expressions.boolean(true);
    valid &= expressions.sle(zero, maximum) == expressions.boolean(false);
    valid &= expressions.land(p, expressions.boolean(true)) == p;
    valid &= expressions.lor(p, expressions.boolean(false)) == p;
    valid &= expressions.lnot(expressions.lnot(p)) == p;
    valid &= expressions.land(p, expressions.lnot(p)) == expressions.boolean(false);
    valid &= expressions.lor(p, expressions.lnot(p)) == expressions.boolean(true);
    valid &= expressions.select(p, sum, sum) == sum;
    auto q = expressions.lt(x, expressions.constant(5));
    auto r = expressions.lt(x, expressions.constant(6));
    auto both = expressions.land(p, q), either = expressions.lor(p, q);
    const auto beforeAbsorption = expressions.size();
    valid &= expressions.lor(p, both) == p && expressions.lor(both, p) == p;
    valid &= expressions.land(p, either) == p && expressions.land(either, p) == p;
    valid &= expressions.lor(q, both) == q && expressions.land(q, either) == q;
    valid &= expressions.land(p, both) == both && expressions.land(both, p) == both;
    valid &= expressions.lor(p, either) == either && expressions.lor(either, p) == either;
    valid &= expressions.size() == beforeAbsorption;
    valid &= expressions.implies(expressions.land(p, q), p);
    valid &= expressions.implies(expressions.land(p, expressions.lnot(q)), expressions.lnot(q));
    valid &= expressions.implies(expressions.land(expressions.land(p, q), r), expressions.land(q, p));
    valid &= !expressions.implies(p, q) && !expressions.implies(expressions.lor(p, q), p);
    valid &= !expressions.implies(q, r); // Arithmetic correlation is deliberately not inferred.
    valid &= expressions.implies(expressions.boolean(false), q);
    valid &= expressions.implies(p, expressions.boolean(true));
    auto masked = expressions.select(p, q, expressions.boolean(false));
    valid &= expressions.implies(masked, p) && expressions.implies(masked, q);
    auto maskedFalse = expressions.select(p, expressions.boolean(true), q);
    valid &= expressions.implies(expressions.lnot(maskedFalse), expressions.lnot(p));
    valid &= expressions.implies(expressions.lnot(maskedFalse), expressions.lnot(q));
    // A retained cover and an active intermediate event cannot coexist when
    // that intermediate supplies an alternative native-chain path.
    auto endpoints = expressions.land(p, q);
    auto alternative = expressions.land(endpoints, r);
    auto cover = expressions.land(endpoints, expressions.lnot(alternative));
    valid &= expressions.implies(cover, expressions.lnot(r));
    valid &= !expressions.implies(endpoints, expressions.lnot(r));
    valid &= !expressions.implies(expressions.lor(p, q), p);
    fs::RegionExpressions bad;
    valid &= bad.div(bad.input(index), bad.constant(0)) == fs::RegionExpressions::invalid;
    fs::RegionExpressions dynamic;
    valid &= dynamic.rem(dynamic.constant(3), dynamic.input(index)) == fs::RegionExpressions::invalid;
    fs::RegionExpressions typed;
    valid &= typed.add(typed.input(index), typed.input(predicate)) == fs::RegionExpressions::invalid;
    return valid && expressions.error().empty();
}
bool checkImplicationTruthTables(Value index, Value predicate)
{
    fs::RegionExpressions expressions;
    const auto p = expressions.input(predicate), x = expressions.input(index);
    const auto q = expressions.lt(x, expressions.constant(5));
    const auto r = expressions.lt(x, expressions.constant(6));
    using Formula = std::pair<fs::RegionExpressions::Id, unsigned>;
    // Eight valuations of three abstract atoms. Arithmetic correlations are
    // deliberately omitted, matching the proof engine's conservative contract.
    std::vector<Formula> formulas{{p, 0xaa}, {q, 0xcc}, {r, 0xf0},
        {expressions.boolean(false), 0}, {expressions.boolean(true), 255}};
    for (uint32_t i = 0; i < 3; ++i) {
        auto [a, av] = formulas[i];
        formulas.push_back({expressions.lnot(a), (~av) & 255});
        for (uint32_t j = 0; j < 3; ++j) {
            auto [b, bv] = formulas[j];
            formulas.push_back({expressions.land(a, b), av & bv});
            formulas.push_back({expressions.lor(a, b), av | bv});
            formulas.push_back({expressions.select(a, b, expressions.boolean(false)), av & bv});
            formulas.push_back({expressions.select(a, expressions.boolean(true), b), av | bv});
            formulas.push_back({expressions.select(a, expressions.boolean(false), b), (~av) & bv & 255});
            formulas.push_back({expressions.select(a, b, expressions.boolean(true)), (av & bv) | ((~av) & 255)});
            formulas.push_back({expressions.select(a, b, r), (av & bv) | ((~av) & 0xf0)});
            auto covered = expressions.land(expressions.land(a, b),
                expressions.lnot(expressions.land(expressions.land(a, b), r)));
            formulas.push_back({covered, av & bv & (~0xf0) & 255});
        }
    }
    for (auto [premise, truth] : formulas) {
        for (auto [consequence, implied] : formulas) {
            if (expressions.implies(premise, consequence) && (truth & ~implied)) { return false; }
        }
    }
    return expressions.constructionError().empty();
}
bool checkEmission(func::FuncOp function, ArrayRef<Operation*> cuts)
{
    fs::RegionExpressions expressions;
    auto x = expressions.input(function.getArgument(0));
    auto divisor = expressions.constant(3);
    auto result = expressions.add(expressions.div(x, divisor), expressions.rem(x, divisor));
    fs::PreparedLogicalPlan plan(0);
    OpBuilder builder(function.getContext());
    auto& first = plan.addPreparation(cuts[0]);
    builder.setInsertionPointToEnd(&first);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> memo;
    auto firstValue = expressions.emit(result, builder, cuts[0], memo);
    const auto count = first.getOperations().size();
    auto repeated = expressions.emit(result, builder, cuts[0], memo);
    if (failed(firstValue) || failed(repeated) || *firstValue != *repeated || first.getOperations().size() != count) {
        return false;
    }
    for (auto& operation : first) {
        if (failed(verify(&operation))) { return false; }
    }
    auto& second = plan.addPreparation(cuts[1]);
    builder.setInsertionPointToEnd(&second);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> secondMemo;
    auto secondValue = expressions.emit(result, builder, cuts[1], secondMemo);
    return succeeded(secondValue) && *firstValue != *secondValue && !second.empty();
}
bool checkPlacementRetry(func::FuncOp function, ArrayRef<Operation*> cuts)
{
    fs::RegionExpressions expressions;
    auto late = expressions.input(cuts[0]->getResult(0));
    auto root = expressions.add(late, expressions.constant(7));
    fs::PreparedLogicalPlan plan(0);
    auto& unavailable = plan.addPreparation(cuts[0]);
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToEnd(&unavailable);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> earlyMemo;
    if (succeeded(expressions.emit(root, builder, cuts[0], earlyMemo)) || !unavailable.empty() ||
        !earlyMemo.empty() || !expressions.constructionError().empty() || expressions.lastEmissionError().empty()) {
        return false;
    }
    // Exact queries and new expressions remain usable after a placement failure.
    auto predicate = expressions.lt(late, root);
    if (!expressions.implies(predicate, predicate)) { return false; }
    auto& available = plan.addPreparation(cuts[1]);
    builder.setInsertionPointToEnd(&available);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> lateMemo;
    return succeeded(expressions.emit(root, builder, cuts[1], lateMemo)) && !available.empty() &&
        expressions.error().empty() && expressions.lastEmissionError().empty();
}
bool rejectedWithoutCode(func::FuncOp function, Operation* cut, Value unavailable)
{
    fs::RegionExpressions expressions;
    auto input = expressions.input(unavailable);
    auto root = expressions.add(input, expressions.constant(7));
    fs::PreparedLogicalPlan plan(0);
    auto& block = plan.addPreparation(cut);
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToEnd(&block);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> memo;
    return failed(expressions.emit(root, builder, cut, memo)) && block.empty() && memo.empty();
}
} // namespace
LogicalResult runRegionExpressionChecks(func::FuncOp function)
{
    SmallVector<Operation*> cuts;
    Value hidden;
    function.walk([&](Operation* operation) {
        if (operation->hasAttr("test.cut")) { cuts.push_back(operation); }
        if (operation->hasAttr("test.hidden") && operation->getNumResults()) { hidden = operation->getResult(0); }
    });
    if (function.getNumArguments() != 2 || cuts.size() != 2 || !cuts[0]->getNumResults() || !hidden) {
        return function.emitError("regional expression fixture requires two arguments, cuts and a hidden result");
    }
    const auto before = render(function);
    if (!checkAlgebra(function.getArgument(0), function.getArgument(1)) || !checkEmission(function, cuts) ||
        !checkImplicationTruthTables(function.getArgument(0), function.getArgument(1)) ||
        !checkPlacementRetry(function, cuts) ||
        !rejectedWithoutCode(function, cuts[0], cuts[0]->getResult(0)) ||
        !rejectedWithoutCode(function, cuts[1], hidden) || render(function) != before) {
        return function.emitError("regional expression checks failed");
    }
    llvm::outs() << "regional expression checks passed\n";
    return success();
}
