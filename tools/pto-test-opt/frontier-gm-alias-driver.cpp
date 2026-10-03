// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared GM records and finite concrete address valuations, not native certificates.
#include "PTO/Transforms/FrontierSynch/StorageAnalysis.h"
#include "PTO/Transforms/FrontierSynch/DemandAnalysis.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <array>

namespace {
using namespace mlir;
using namespace mlir::pto;
bool checkIntervals(const SyncInput& input, Value first, Value second)
{
    constexpr std::array<uint64_t, 4> offsets{0, 4, 8, 16};
    constexpr uint64_t size = 8;
    bool mayAlias = input.memory().gmPolicy() == GMAliasPolicy::MayAlias;
    for (uint64_t a : offsets) {
        for (uint64_t b : offsets) {
            // These bounded nonwrapping supplied records have a common origin.
            // Half-open finite-set intersection is independent of end arithmetic.
            bool overlap = false;
            for (uint64_t byte = a; byte < a + size; ++byte) {
                overlap |= byte >= b && byte < b + size;
            }
            BaseMemInfo left(first, first, AddressSpace::GM, {a}, size);
            BaseMemInfo same(second, first, AddressSpace::GM, {b}, size);
            if (input.memory().MemAlias(&left, &same) != overlap) { return false; }
            BaseMemInfo distinct(second, second, AddressSpace::GM, {b}, size);
            if (input.memory().MemAlias(&left, &distinct) != mayAlias) { return false; }
            BaseMemInfo other(second, second, AddressSpace::VEC, {b}, size);
            if (input.memory().MemAlias(&left, &other)) { return false; }
        }
    }
    // Two independent pointer arguments can have equal concrete base addresses.
    // Their root-relative offsets alone cannot rule out this valid valuation.
    BaseMemInfo unknownA(first, first, AddressSpace::GM, {0}, 0);
    BaseMemInfo unknownB(second, second, AddressSpace::GM, {0}, 0);
    BaseMemInfo sameUnknown(second, first, AddressSpace::GM, {0}, 0);
    BaseMemInfo missing(first, first, AddressSpace::GM, {}, 0);
    return input.memory().MemAlias(&unknownA, &unknownB) == mayAlias &&
        input.memory().MemAlias(&unknownA, &sameUnknown) && input.memory().MemAlias(&missing, &unknownB);
}
} // namespace
int runGMAliasChecks(llvm::StringRef path, mlir::MLIRContext& context)
{
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(path, &context);
    if (!module) { return 1; }
    auto function = module->lookupSymbol<mlir::func::FuncOp>("gm_unknown_arguments");
    if (!function || function.getNumArguments() != 2) { return 1; }
    for (auto policy : {GMAliasPolicy::MayAlias, GMAliasPolicy::MayNotAlias}) {
        SyncInput input(policy);
        bool mayAlias = policy == GMAliasPolicy::MayAlias;
        if (failed(input.build(function)) || input.instructions().size() != 2 ||
            !checkIntervals(input, function.getArgument(0), function.getArgument(1))) { return 1; }
        frontiersynch::StorageAnalysis storage(input);
        if (storage.footprints().size() != 4 || storage.aliases().size() != unsigned(mayAlias)) { return 1; }
        if (mayAlias) {
            auto alias = storage.aliases().front();
            auto* left = storage.footprints()[alias.first].memory;
            auto* right = storage.footprints()[alias.second].memory;
            if (left->scope != AddressSpace::GM || right->scope != AddressSpace::GM ||
                left->rootBuffer == right->rootBuffer || !input.memory().MemAlias(left, right)) { return 1; }
        }
        frontiersynch::LifetimeAnalysis lifetimes;
        if (failed(lifetimes.build(input.instructions(), storage.footprints(), storage.aliases())) ||
            lifetimes.generators().size() != unsigned(mayAlias)) { return 1; }
        if (mayAlias) {
            const auto& demand = lifetimes.generators().front();
            if (demand.source != 0 || demand.consumer != 1 || demand.witnesses.size() != 1 ||
                demand.witnesses.front().hazard != frontiersynch::Hazard::RAW) { return 1; }
        }
    }
    if (MemoryDependentAnalyzer().gmPolicy() != GMAliasPolicy::MayNotAlias) { return 1; }
    llvm::outs() << "verified shared GM alias policies, interval boundaries and original RAW incidence\n";
    return 0;
}
