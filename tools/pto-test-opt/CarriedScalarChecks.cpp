// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check the shared recurrence proof against independently stepped bank state.
#include "PTO/Transforms/FrontierSynch/PhaseIndex.h"
#include "../../lib/PTO/Transforms/FrontierSynch/PhaseNormalization.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
struct Case {
    int64_t seed, stride, modulus, lower, step;
    bool supported = true, escaping = false;
    unsigned width = 64;
};
bool checkCase(MLIRContext* context, const Case& test)
{
    std::string text;
    llvm::raw_string_ostream out(text);
    out << "module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, " << test.width
        << " : i32>>} { func.func @case(%n: index)" << (test.escaping ? " -> index" : "") << " {\n"
        << "%zero = arith.constant 0 : index\n"
        << "%lower = arith.constant " << test.lower << " : index\n"
        << "%step = arith.constant " << test.step << " : index\n"
        << "%seed = arith.constant " << test.seed << " : index\n"
        << "%stride = arith.constant " << test.stride << " : index\n"
        << "%modulus = arith.constant " << test.modulus << " : index\n"
        << "%last = scf.for %i = %lower to %n step %step iter_args(%bank = %seed) -> index {\n"
        << "%present = arith.cmpi eq, %bank, %zero : index\nscf.if %present {}\n"
        << "%sum = arith.addi %bank, %stride : index\n"
        << "%next = arith.remui %sum, %modulus : index\nscf.yield %next : index\n}\n"
        << (test.escaping ? "return %last : index" : "return") << "\n}}";
    auto module = parseSourceString<ModuleOp>(text, context);
    if (!module) {
        return false;
    }
    auto function = module->lookupSymbol<func::FuncOp>("case");
    scf::ForOp loop;
    function.walk([&](scf::ForOp found) { loop = found; });
    fs::PhaseIndex index;
    if (failed(index.build(function, ArrayRef<const pto::CompoundInstanceElement*>{}))) {
        return false;
    }
    auto argument = loop.getRegionIterArgs().front();
    auto* proof = index.carriedOrdinal(argument);
    if (static_cast<bool>(proof) != test.supported || !index.hasRelevantCarriedState(loop) ||
        index.hasUnprovedCarriedState(loop) != (!test.supported || test.escaping)) {
        return false;
    }
    if (!proof) {
        return true;
    }
    fs::RegionExpressions arena;
    fs::PhaseNormalization normalizer(loop, index, arena);
    if (!normalizer.periodic(argument, proof->period())) {
        return false;
    }
    if (proof->period() > 1 && normalizer.periodic(argument, proof->period() - 1)) {
        return false;
    }
    uint64_t expected = test.seed;
    for (uint64_t ordinal = 0; ordinal < 16; ++ordinal) {
        auto actual = normalizer.atPhase(argument, ordinal, proof->period());
        if (!actual || arena.constantValue(*actual) != expected) {
            return false;
        }
        expected = (expected + static_cast<uint64_t>(test.stride)) % test.modulus;
    }
    // A huge ordinal must not overflow the unreduced affine product. This
    // independent reference uses 256 bits; production uses bounded 128 bits.
    auto expectedWide = (APInt(256, test.seed) + APInt(256, UINT64_MAX) * APInt(256, test.stride))
                            .urem(APInt(256, test.modulus))
                            .getZExtValue();
    auto actualWide = normalizer.atPhase(argument, UINT64_MAX, proof->period());
    return actualWide && arena.constantValue(*actualWide) == expectedWide;
}
} // namespace
bool runCarriedScalarChecks(MLIRContext* context)
{
    const std::array<Case, 10> cases{
        {{1, 1, 2, 0, 1},
         {0, 2, 6, 3, 2},
         {4, 0, 7, 1, 3},
         {1, 4000000000000000000LL, 4000000000000000001LL, 0, 1},
         {1, 1, 2, 0, 1, true, true},
         {2, 1, 2, 0, 1, false},
         {1, -1, 2, 0, 1, false},
         {1, INT64_MAX, 2, 0, 1, false},
         {1, 1, 2, 0, 1, true, false, 32},
         {1, INT32_MAX, 2, 0, 1, false, false, 32}}};
    for (const auto& test : cases) {
        if (!checkCase(context, test)) {
            llvm::errs() << "carried scalar proof check failed\n";
            return false;
        }
    }
    llvm::outs() << "carried scalar proofs: ordinal mapping, periods, wide evaluation and refusal passed\n";
    return true;
}
