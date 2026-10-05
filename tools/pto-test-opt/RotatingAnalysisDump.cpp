// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/RotatingAnalysis.h"
#include "SyncLogicalInsertionChecks.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
llvm::json::Object dumpPeriodicAnalysis(const fs::PeriodicAnalysis& result);
LogicalResult runRotatingAnalysisChecks(func::FuncOp function, const pto::SyncInput& input)
{
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) {
        return failure();
    }
    scf::ForOp loop;
    for (auto candidate : function.getOps<scf::ForOp>()) {
        if (loop) {
            return failure();
        }
        loop = candidate;
    }
    if (!loop) {
        return failure();
    }
    auto recognition = fs::recognizeRotating(loop, index, input, input.accesses());
    auto analysis = fs::analyzeRotating(loop, index, input, recognition);
    llvm::json::Object result;
    result["function"] = function.getSymName();
    result["error"] = analysis.error;
    result["periodic"] = dumpPeriodicAnalysis(analysis.periodic);
    result["refresh"] = analysis.extraction.refreshBound;
    llvm::json::Array fragments;
    for (const auto& a : analysis.fragments) {
        fragments.push_back(llvm::json::Object{{"payload", a.payload}, {"family", a.family},
            {"atom", a.atom}, {"slots", a.slots}, {"stride", a.stride}, {"offset", a.offset},
            {"read", a.read}, {"write", a.write}, {"protection_group", a.protectionGroup}});
    }
    result["fragments"] = std::move(fragments);
    auto program = fs::recognizeProgram(function, input);
    if (failed(program)) {
        return failure();
    }
    auto render = [&]() {
        std::string text;
        llvm::raw_string_ostream stream(text);
        function.print(stream);
        return text;
    };
    const auto before = render();
    auto prepared = fs::prepareRotatingInsertion(function, input, *program);
    result["prepared"] = succeeded(prepared);
    result["unchanged_preparation"] = before == render();
    if (succeeded(prepared)) {
        // Interpreter metadata only: no numerical template analysis/unrolling.
        // Reuse the actual-IR interpreter with this body's original phase map.
        fs::NumericTemplate traceInput;
        traceInput.outer = loop;
        APInt lower, step;
        if (!matchPattern(loop.getLowerBound(), m_ConstantInt(&lower)) ||
            !matchPattern(loop.getStep(), m_ConstantInt(&step))) {
            return failure();
        }
        traceInput.lower = lower.getSExtValue();
        traceInput.step = step.getSExtValue();
        for (auto* phase : analysis.phases) {
            fs::TemplatePayload payload;
            payload.phase = phase;
            traceInput.payloads.push_back(std::move(payload));
        }
        if (failed(fs::insertLogicalSynchronization(function, **prepared)) || failed(verify(function))) {
            return failure();
        }
        result["trace"] = traceLogicalInsertion(function, traceInput);
    }
    llvm::outs() << llvm::json::Value(std::move(result)) << "\n";
    return success();
}
