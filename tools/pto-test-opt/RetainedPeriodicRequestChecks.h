// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_RETAINED_PERIODIC_REQUEST_CHECKS_H
#define PTO_FRONTIER_RETAINED_PERIODIC_REQUEST_CHECKS_H
#include "PTO/Transforms/FrontierSynch/Structured/RotatingAnalysis.h"
#include "PTO/Transforms/FrontierSynch/Structured/NumericTemplateAnalysis.h"
#include "../../lib/PTO/Transforms/FrontierSynch/WholeInvocationIdentity.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
namespace mlir::pto::test {
namespace retainedPeriodicRequestChecks {
namespace fs = frontiersynch;
inline bool reject(unsigned line)
{
    llvm::errs() << "cross-mode periodic occurrence equivalence failed at line " << line << "\n";
    return false;
}
struct View {
    const fs::PeriodicAnalysis* periodic = nullptr;
    scf::ForOp loop;
    SmallVector<Operation*> sites;
};
inline View view(const fs::MathematicalResult& root)
{
    const auto* math = &root;
    while (math->identityComposition) { math = math->identityComposition->child.get(); }
    View result;
    if (math->rotatingDemands) {
        result.periodic = &math->rotatingDemands->periodic;
        result.loop = math->rotatingDemands->loop;
        for (const auto* phase : math->rotatingDemands->phases) {
            if (!phase) { return {}; }
            result.sites.push_back(phase->elementOp);
        }
    } else if (math->numericalDemands) {
        const auto& numerical = *math->numericalDemands;
        result.periodic = &numerical.analysis;
        result.loop = numerical.form.outer;
        for (const auto& payload : numerical.form.payloads) {
            // This control compares unexpanded original occurrences only.
            if (!payload.phase || !payload.coordinates.empty()) { return {}; }
            result.sites.push_back(payload.phase->elementOp);
        }
    }
    return result;
}
inline bool equivalent(const fs::MathematicalResult& original, const fs::MathematicalResult& other)
{
    if (original.input != other.input || original.region != other.region || other.compactDemands) {
        return reject(__LINE__);
    }
    const auto a = view(original), b = view(other);
    if (!a.periodic || !b.periodic || !a.loop || a.loop != b.loop || a.sites.empty() ||
        !a.periodic->error.empty() || !b.periodic->error.empty() || a.sites.size() != b.sites.size() ||
        a.sites.size() != a.periodic->payloads.size() || b.sites.size() != b.periodic->payloads.size()) {
        return reject(__LINE__);
    }
    SmallVector<uint32_t> types;
    for (auto* site : a.sites) {
        if (!site || llvm::count(a.sites, site) != 1 || llvm::count(b.sites, site) != 1) { return reject(__LINE__); }
        types.push_back(std::distance(b.sites.begin(), llvm::find(b.sites, site)));
    }
    // A finite prefix must describe the same ordered original word, rather
    // than matching numeric type IDs from unrelated representations.
    for (uint32_t i = 0; i < types.size(); ++i) {
        if (types[i] != i || a.periodic->payloads[i].pipe != b.periodic->payloads[types[i]].pipe) {
            return reject(__LINE__);
        }
    }
    for (uint32_t source = 0; source < types.size(); ++source) {
        for (uint32_t target = 0; target < types.size(); ++target) {
            for (auto sk : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
                for (auto tk : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
                    const fs::PeriodicEvent as{source, sk}, at{target, tk};
                    const fs::PeriodicEvent bs{types[source], sk}, bt{types[target], tk};
                    const auto left = a.periodic->eventThreshold(as, at);
                    const auto right = b.periodic->eventThreshold(bs, bt);
                    if (left.error != fs::PeriodicQueryError::None || right.error != left.error ||
                        right.displacement != left.displacement) { return reject(__LINE__); }
                    // Exact thresholds cover all distances. Also exercise
                    // presence filtering: empty/partial words and absent visits,
                    // reverse queries, diagonals, and strict versus reflexive.
                    for (uint64_t prefix : {uint64_t{0}, uint64_t{1}, uint64_t{3}, uint64_t{8}}) {
                        for (uint64_t from = 0; from < 5; ++from) {
                            for (uint64_t to = 0; to < 5; ++to) {
                                for (bool strict : {false, true}) {
                                    auto x = a.periodic->eventPrecedes(as, from, at, to, prefix, strict);
                                    auto y = b.periodic->eventPrecedes(bs, from, bt, to, prefix, strict);
                                    const bool present = from * types.size() + source < prefix &&
                                        to * types.size() + target < prefix;
                                    const auto expectedError = present ? fs::PeriodicQueryError::None :
                                        fs::PeriodicQueryError::InvalidInput;
                                    if (x.error != expectedError || y.error != expectedError ||
                                        x.value != y.value || (!present && x.value)) { return reject(__LINE__); }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}
} // namespace retainedPeriodicRequestChecks
} // namespace mlir::pto::test
#endif
