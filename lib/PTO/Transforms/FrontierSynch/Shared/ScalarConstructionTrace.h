// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIER_SCALARCONSTRUCTIONTRACE_H
#define PTO_FRONTIER_SCALARCONSTRUCTIONTRACE_H
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <optional>
#include <vector>
namespace mlir::pto::frontiersynch::detail {
// Region-free arithmetic identity, matching OperationEquivalence with ignored
// locations. Raw attributes and properties remain separate; result identities
// are deliberately absent from equivalence. Commutative operands form a multiset.
struct ScalarInstruction {
    Operation* cut = nullptr;
    StringAttr name;
    DictionaryAttr attributes;
    Attribute properties;
    SmallVector<Type> types;
    SmallVector<uint64_t> operands, results;
    bool commutative = false;
};
inline bool sameScalarInstruction(const ScalarInstruction& a, const ScalarInstruction& b)
{
    return a.name == b.name && a.attributes == b.attributes && a.properties == b.properties &&
        a.types == b.types && a.operands == b.operands;
}
inline std::size_t scalarInstructionHash(const ScalarInstruction& instruction)
{
    return static_cast<std::size_t>(llvm::hash_combine(instruction.name, instruction.attributes,
        instruction.properties, llvm::hash_combine_range(instruction.types.begin(), instruction.types.end()),
        llvm::hash_combine_range(instruction.operands.begin(), instruction.operands.end())));
}
struct ScalarDeduplication {
    llvm::DenseMap<uint64_t, uint64_t> replacements;
    SmallVector<std::size_t> redundant;
    uint64_t canonical(uint64_t id) const
    {
        for (auto found = replacements.find(id); found != replacements.end(); found = replacements.find(id)) {
            id = found->second;
        }
        return id;
    }
};
// Preparation is topological (the insertion preflight checks this). Therefore
// replacing a duplicate result cannot change an already visited instruction.
inline ScalarDeduplication deduplicateScalarInstructions(ArrayRef<ScalarInstruction> source)
{
    ScalarDeduplication result;
    std::map<std::pair<Operation*, std::size_t>, SmallVector<std::size_t>> buckets;
    std::vector<ScalarInstruction> canonical;
    for (const auto& instruction : source) {
        canonical.push_back(instruction);
        auto& current = canonical.back();
        for (auto& operand : current.operands) { operand = result.canonical(operand); }
        if (current.commutative) { llvm::sort(current.operands); }
        auto& candidates = buckets[{current.cut, scalarInstructionHash(current)}];
        auto found = llvm::find_if(candidates, [&](std::size_t candidate) {
            return sameScalarInstruction(canonical[candidate], current);
        });
        if (found == candidates.end()) { candidates.push_back(canonical.size() - 1); continue; }
        const auto& earlier = canonical[*found];
        for (auto [old, shared] : llvm::zip(current.results, earlier.results)) {
            result.replacements[old] = shared;
        }
        result.redundant.push_back(canonical.size() - 1);
    }
    return result;
}
template<class Identify>
std::optional<ScalarInstruction> describeScalarInstruction(Operation& operation, Operation* cut, Identify identify)
{
    if (operation.getName().getDialectNamespace() != "arith" || operation.getNumRegions() ||
        operation.getNumSuccessors() || !isMemoryEffectFree(&operation)) { return std::nullopt; }
    ScalarInstruction result;
    result.cut = cut; result.name = operation.getName().getIdentifier();
    result.attributes = operation.getRawDictionaryAttrs(); result.properties = operation.getPropertiesAsAttribute();
    llvm::append_range(result.types, operation.getResultTypes());
    for (auto operand : operation.getOperands()) { result.operands.push_back(identify(operand)); }
    for (auto value : operation.getResults()) { result.results.push_back(identify(value)); }
    result.commutative = operation.hasTrait<OpTrait::IsCommutative>();
    return result;
}
} // namespace mlir::pto::frontiersynch::detail
#endif
