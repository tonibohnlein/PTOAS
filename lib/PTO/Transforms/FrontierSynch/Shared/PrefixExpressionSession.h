// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Attempt-owned sharing for prefix expressions at one unchanged execution cut.
#ifndef PTO_FRONTIERSYNCH_PREFIX_EXPRESSION_SESSION_H
#define PTO_FRONTIERSYNCH_PREFIX_EXPRESSION_SESSION_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionExpressions.h"
#include "mlir/IR/IRMapping.h"
#include <map>
#include <tuple>
namespace mlir::pto::frontiersynch {
class PrefixExpressionSession {
public:
    using Environment = std::vector<std::pair<RegionExpressions::Id, const void*>>;
    class Scope {
    public:
        explicit Scope(PrefixExpressionSession& owner) : owner(owner) {}
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() { owner.entries.clear(); owner.active = false; }
    private:
        PrefixExpressionSession& owner;
    };
    std::unique_ptr<Scope> begin()
    {
        if (active) { return {}; }
        entries.clear(); cut = nullptr; block = nullptr; mapping = nullptr;
        active = true;
        return std::make_unique<Scope>(*this);
    }
    // Callbacks append in cut order. Retain only the current cut, rather than
    // keeping compiler-side copies of every emitted expression in the function.
    // Different history bindings never share a memo. Direct emission uses an
    // empty environment. This cache is inactive for standalone recipe calls.
    RegionExpressions::CutEmission* bindings(const RegionExpressions* arena,
        Operation* at, Block* into, const IRMapping& mapped, Environment environment)
    {
        if (!active) { return nullptr; }
        if (cut != at || block != into || mapping != &mapped) {
            entries.clear(); cut = at; block = into; mapping = &mapped;
        }
        llvm::sort(environment);
        return &entries[{arena, std::move(environment)}];
    }
private:
    bool active = false;
    Operation* cut = nullptr;
    Block* block = nullptr;
    const IRMapping* mapping = nullptr;
    using Key = std::pair<const RegionExpressions*, Environment>;
    std::map<Key, RegionExpressions::CutEmission> entries;
};
} // namespace mlir::pto::frontiersynch
#endif
