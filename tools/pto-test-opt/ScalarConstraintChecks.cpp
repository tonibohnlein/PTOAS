// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check path isolation and machine-semantic range proofs independently of
// recognizer selection, including the literal prefill wrapping counterexample.
#include "../../lib/PTO/Transforms/InsertSync/SyncScalarEvolution.h"
#include "../../lib/PTO/Transforms/FrontierSynch/ArithmeticProgramInternal.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>
using namespace mlir;
namespace {
struct Case {
    const char* name;
    std::string body;
    int64_t lower, upper;
    unsigned bits = 64;
    enum Expression { Any, X, XPlusOne, Opaque, Scaled } expression = Any;
};
std::string probe(StringRef value = "%x")
{
    return "%probe = arith.addi " + value.str() + ", %zero {test.probe} : index\n";
}
bool check(MLIRContext* context, const Case& test)
{
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, " << test.bits
        << " : i32>>} { func.func @case(%x: index, %a: i32, %b: i32) {\n"
        << "%zero = arith.constant 0 : index\n%one = arith.constant 1 : index\n"
        << "%three = arith.constant 3 : index\n%four = arith.constant 4 : index\n"
        << "%hundred = arith.constant 100 : index\n"
        << "%min = arith.constant " << (test.bits == 64 ? INT64_MIN : INT32_MIN) << " : index\n"
        << "%max = arith.constant " << (test.bits == 64 ? INT64_MAX : INT32_MAX) << " : index\n"
        << test.body << "\nreturn\n}}";
    auto module = parseSourceString<ModuleOp>(text, context);
    if (!module || failed(verify(*module))) {
        return false;
    }
    Operation* anchor = nullptr;
    module->walk([&](Operation* operation) {
        if (operation->hasAttr("test.probe")) {
            anchor = operation;
        }
    });
    if (!anchor) {
        return false;
    }
    const auto before = [&]() {
        std::string source;
        llvm::raw_string_ostream stream(source);
        module->print(stream);
        return source;
    }();
    auto function = module->lookupSymbol<func::FuncOp>("case");
    pto::detail::ScalarEvolution evolution(context, anchor);
    auto input = anchor->getOperand(0);
    auto expectedRange = std::make_pair(test.lower, test.upper);
    auto symbol = [&](Value value) { return getAffineSymbolExpr(value == function.getArgument(0) ? 0 : 1, context); };
    // Exercise both query orders and memoized repeats: range-only queries must
    // never contaminate the expression cache with their placeholder symbols.
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        pto::detail::ScalarEvolution expressionFirst(context, anchor);
        auto firstExpression = expressionFirst.value(input, symbol);
        auto firstRange = expressionFirst.signedRange(input);
        auto range = evolution.signedRange(input);
        auto expression = evolution.value(input, symbol);
        if (firstExpression != expression || firstRange != range) {
            return false;
        }
        AffineExpr expected;
        auto x = getAffineSymbolExpr(0, context), opaque = getAffineSymbolExpr(1, context);
        switch (test.expression) {
            case Case::Any:
                break;
            case Case::X:
                expected = x;
                break;
            case Case::XPlusOne:
                expected = x + 1;
                break;
            case Case::Opaque:
                expected = opaque;
                break;
            case Case::Scaled:
                expected = opaque * 4;
                break;
        }
        if (range != expectedRange || (expected && expression != expected)) {
            llvm::errs() << test.name << ": unexpected range/expression ";
            if (range) {
                llvm::errs() << range->first << "," << range->second;
            }
            if (expression) {
                llvm::errs() << " " << expression;
            }
            llvm::errs() << "\n";
            return false;
        }
    }
    std::string after;
    llvm::raw_string_ostream stream(after);
    module->print(stream);
    return before == after;
}
bool interleavedContexts(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>(
        R"(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i32>>} {
  func.func @contexts(%x: index) {
    %zero = arith.constant 0 : index
    %condition = arith.cmpi sgt, %x, %zero : index
    scf.if %condition {
      %then = arith.addi %x, %zero {test.probe} : index
    } else {
      %else = arith.addi %x, %zero {test.probe} : index
    }
    %root = arith.addi %x, %zero {test.probe} : index
    return
  }
})",
        context);
    if (!module || failed(verify(*module))) {
        return false;
    }
    SmallVector<Operation*> anchors;
    module->walk([&](Operation* operation) {
        if (operation->hasAttr("test.probe")) {
            anchors.push_back(operation);
        }
    });
    if (anchors.size() != 3) {
        return false;
    }
    pto::detail::ScalarEvolution thenContext(context, anchors[0]), elseContext(context, anchors[1]),
        rootContext(context, anchors[2]);
    auto x = anchors[0]->getOperand(0);
    auto thenRange = std::make_pair(int64_t{1}, int64_t{INT32_MAX});
    auto elseRange = std::make_pair(int64_t{INT32_MIN}, int64_t{0});
    auto rootRange = std::make_pair(int64_t{INT32_MIN}, int64_t{INT32_MAX});
    return thenContext.signedRange(x) == thenRange && rootContext.signedRange(x) == rootRange &&
           elseContext.signedRange(x) == elseRange && thenContext.signedRange(x) == thenRange &&
           elseContext.signedRange(x) == elseRange && rootContext.signedRange(x) == rootRange;
}
bool controlProofContext(MLIRContext* context)
{
    auto module = parseSourceString<ModuleOp>(
        R"(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i32>>} {
  func.func @control() {
    %lower = arith.constant -4 : index
    %upper = arith.constant 8 : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %four = arith.constant 4 : index
    scf.for %i = %lower to %upper step %one {
      %quotient = arith.divui %i, %four : index
      %nonnegative = arith.cmpi sge, %i, %zero : index
      %positive = arith.cmpi sgt, %quotient, %zero : index
      scf.if %nonnegative {
        scf.if %positive {
          %probe = arith.addi %quotient, %zero {test.probe} : index
        }
      }
    }
    return
  }
})",
        context);
    if (!module || failed(verify(*module))) {
        return false;
    }
    auto function = module->lookupSymbol<func::FuncOp>("control");
    Operation* anchor = nullptr;
    scf::ForOp loop;
    module->walk([&](Operation* operation) {
        if (operation->hasAttr("test.probe")) {
            anchor = operation;
        }
        if (auto found = dyn_cast<scf::ForOp>(operation)) {
            loop = found;
        }
    });
    if (!anchor || !loop) {
        return false;
    }
    namespace fs = pto::frontiersynch;
    fs::PhaseIndex index;
    if (failed(index.build(function, ArrayRef<const pto::CompoundInstanceElement*>{}))) {
        return false;
    }
    pto::CompoundInstanceElement phase(0, {}, {}, pto::PipelineType::PIPE_S, anchor->getName());
    phase.elementOp = anchor;
    fs::ArithmeticSite site{&phase, {loop}, {}};
    fs::ArithmeticProgram output;
    output.context.function = function;
    output.context.root = function;
    fs::ArithmeticLimits limits;
    fs::detail::ProgramBuilder builder{output, limits, context, index, DenseMap<Value, unsigned>(), {}};
    auto quotient = anchor->getOperand(0);
    auto comparison = cast<scf::IfOp>(anchor->getParentOp()).getCondition();
    auto expected = getAffineDimExpr(0, context).floorDiv(4);
    // Geometry is exact on executed nonnegative visits. The presence formula
    // itself must not borrow that assumption: unsigned division differs on
    // negative visits, and its loop-dependent result cannot become a parameter.
    return builder.prepareValue(quotient, site) && builder.value(quotient, site, 0) == expected &&
           !builder.value(quotient, site, 0, fs::detail::ProgramBuilder::ScalarProofContext::Definition) &&
           !builder.prepareGuard(comparison, site) && output.parameters.empty();
}
} // namespace
bool runScalarConstraintChecks(MLIRContext* context)
{
    std::vector<Case> cases;
    for (unsigned bits : {32U, 64U}) {
        int64_t low = bits == 64 ? INT64_MIN : INT32_MIN;
        int64_t high = bits == 64 ? INT64_MAX : INT32_MAX;
        cases.push_back(
            {"then", "%g = arith.cmpi sgt, %x, %zero : index\nscf.if %g {" + probe() + "}", 1, high, bits, Case::X});
        cases.push_back(
            {"else", "%g = arith.cmpi sgt, %x, %zero : index\nscf.if %g {} else {" + probe() + "}", low, 0, bits});
        cases.push_back(
            {"swapped", "%g = arith.cmpi slt, %zero, %x : index\nscf.if %g {" + probe() + "}", 1, high, bits});
        cases.push_back(
            {"sibling", "%g = arith.cmpi sgt, %x, %zero : index\nscf.if %g {} else {" + probe() + "}", low, 0, bits});
        cases.push_back(
            {"post-merge", "%g = arith.cmpi sgt, %x, %zero : index\nscf.if %g {}\n" + probe(), low, high, bits});
        cases.push_back(
            {"unsigned-low", "%g = arith.cmpi ult, %x, %hundred : index\nscf.if %g {" + probe() + "}", 0, 99, bits});
        cases.push_back(
            {"unsigned-cross-sign", "%g = arith.cmpi uge, %x, %one : index\nscf.if %g {" + probe() + "}", low, high,
             bits});
        cases.push_back(
            {"unsigned-negative-half", "%g = arith.cmpi uge, %x, %min : index\nscf.if %g {" + probe() + "}", low, -1,
             bits});
        cases.push_back(
            {"unsigned-positive-half", "%g = arith.cmpi ult, %x, %min : index\nscf.if %g {" + probe() + "}", 0, high,
             bits});
        cases.push_back(
            {"signed-empty", "%g = arith.cmpi slt, %x, %min : index\nscf.if %g {" + probe() + "}", low, high, bits});
        cases.push_back(
            {"unsigned-empty",
             "%g = arith.cmpi ugt, %x, %max : index\n%h = arith.cmpi ult, %x, %zero : index\nscf.if %h {" + probe() +
                 "}",
             low, high, bits});
        std::string comparisons = "%g = arith.cmpi sge, %x, %zero : index\n%h = arith.cmpi slt, %x, %hundred : index\n";
        cases.push_back(
            {"and-true", comparisons + "%both = arith.andi %g, %h : i1\nscf.if %both {" + probe() + "}", 0, 99, bits});
        cases.push_back(
            {"and-false", comparisons + "%both = arith.andi %g, %h : i1\nscf.if %both {} else {" + probe() + "}", low,
             high, bits});
        cases.push_back(
            {"or-true", comparisons + "%either = arith.ori %g, %h : i1\nscf.if %either {" + probe() + "}", low, high,
             bits});
        cases.push_back(
            {"or-false",
             "%g = arith.cmpi slt, %x, %zero : index\n%h = arith.cmpi sge, %x, %hundred : index\n%either = arith.ori "
             "%g, %h : i1\nscf.if %either {} else {" +
                 probe() + "}",
             0, 99, bits});
        cases.push_back(
            {"contradictory",
             comparisons + "scf.if %g {scf.if %h {} else {\n%j = arith.cmpi slt, %x, %zero : index\nscf.if %j {" +
                 probe() + "}}}",
             low, high, bits});
        cases.push_back(
            {"result-does-not-prove-operation",
             "%y = arith.addi %x, %one : index\n%g = arith.cmpi sgt, %y, %zero : index\nscf.if %g {" + probe("%y") +
                 "}",
             1, high, bits, Case::Opaque});
        cases.push_back(
            {"bounded-operand-proves-operation",
             comparisons + "%both = arith.andi %g, %h : i1\n%y = arith.addi %x, %one : index\nscf.if %both {" +
                 probe("%y") + "}",
             1, 100, bits, Case::XPlusOne});
        cases.push_back(
            {"nested-loop",
             comparisons +
                 "%both = arith.andi %g, %h : i1\nscf.if %both {scf.for %i = %zero to %four step %one {\n%y = "
                 "arith.addi %x, %one : index\n" +
                 probe("%y") + "}}",
             1, 100, bits, Case::XPlusOne});
    }
    // Loaded i32 endpoints in a 64-bit index layout: only an enclosing
    // positivity guard can turn unsigned chunk division into affine division.
    std::string chunks =
        "%start = arith.index_cast %a : i32 to index\n%end = arith.index_cast %b : i32 to index\n%length = arith.subi "
        "%end, %start : index\n%sum = arith.addi %length, %three : index\n%count = arith.divui %sum, %four : "
        "index\n%positive = arith.cmpi sgt, %length, %zero : index\n";
    std::string loop =
        "scf.for %i = %zero to %count step %one {\n%scaled = arith.muli %i, %four : index\n" + probe("%scaled") + "}\n";
    cases.push_back({"guarded-chunks", chunks + "scf.if %positive {" + loop + "}", 0, 4294967292LL, 64, Case::Scaled});
    cases.push_back(
        {"literal-prefill-sibling", chunks + "scf.if %positive {}\n" + loop, INT64_MIN, INT64_MAX, 64, Case::Opaque});
    for (const auto& test : cases) {
        if (!check(context, test)) {
            return false;
        }
    }
    if (!interleavedContexts(context) || !controlProofContext(context)) {
        return false;
    }
    // Independent bit-vector witness: start=0, end=-5, iv=2^61 is an
    // executed visit under literal unsigned ceil-div, and 4*iv wraps signed.
    APInt length(64, static_cast<uint64_t>(-5LL));
    APInt count = (length + 3).udiv(APInt(64, 4));
    APInt ordinal(64, uint64_t{1} << 61);
    APInt scaled = ordinal * 4;
    if (!ordinal.ult(count) || scaled.getSExtValue() != INT64_MIN ||
        (length - scaled).getSExtValue() != INT64_MAX - 4) {
        return false;
    }
    llvm::outs() << "scalar constraints: " << cases.size()
                 << " occurrence-local checks and literal wrapping witness passed\n";
    return true;
}
