// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_INTEGERIMPLICATIONCACHE_H
#define PTO_FRONTIERSYNCH_INTEGERIMPLICATIONCACHE_H
#include "PTO/Transforms/FrontierSynch/Arithmetic/IntegerRelations.h"
#include <memory>
namespace mlir::pto::frontiersynch {
// Internal numeric IDs name immutable, append-only entries in systems. The
// vector object outlives this cache; its elements may move when it grows.
class IntegerImplicationCache {
public:
    struct Cost {
        uint64_t hits = 0, rationalProofs = 0, witnessRefutations = 0, exactChecks = 0;
        uint64_t latticeBuilds = 0, latticeQueries = 0, latticeHits = 0;
        uint64_t latticeProofs = 0, latticeEmptyProofs = 0;
    };
    explicit IntegerImplicationCache(const std::vector<IntegerSystem>& systems);
    ~IntegerImplicationCache();
    bool isSubsetOf(std::size_t source, std::size_t target);
    // True is a sound inclusion proof. False means unproved, not noninclusion.
    // This skips exact search and is suitable only for optional union pruning;
    // the exact isSubsetOf query remains available with an independent cache.
    bool hasInclusionCertificate(std::size_t source, std::size_t target);
    const Cost& cost() const;

private:
    struct State;
    std::unique_ptr<State> state;
};
} // namespace mlir::pto::frontiersynch
#endif
