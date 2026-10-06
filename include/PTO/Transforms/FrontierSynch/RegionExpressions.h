// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared, typed expression DAG for exact regional queries and endpoint recipes.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REGIONEXPRESSIONS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REGIONEXPRESSIONS_H
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
namespace mlir::pto::frontiersynch {
class RegionExpressions {
public:
    using Id = uint32_t;
    static constexpr Id invalid = std::numeric_limits<Id>::max();
    Id constant(uint64_t value);
    Id boolean(bool value);
    Id input(Value value);
    // Simultaneous, typed DAG substitution. Bindings are immutable; the memo
    // belongs to this context and one arena. Replacement expressions are not
    // themselves substituted, so coordinate shifts never recursively expand.
    class Substitution {
    public:
        explicit Substitution(llvm::ArrayRef<std::pair<Id, Id>> bindings)
            : bindings(bindings.begin(), bindings.end()) {}
    private:
        friend class RegionExpressions;
        std::vector<std::pair<Id, Id>> bindings;
        llvm::DenseMap<Id, Id> memo;
        const RegionExpressions* owner = nullptr;
    };
    Id substitute(Id expression, Substitution& context);
    Id add(Id a, Id b);
    Id sub(Id a, Id b);
    Id div(Id a, Id b);
    Id rem(Id a, Id b);
    Id lt(Id a, Id b);
    Id le(Id a, Id b);
    Id eq(Id a, Id b);
    Id slt(Id a, Id b); // Signed index ordering for source loop bounds.
    Id sle(Id a, Id b);
    Id land(Id a, Id b);
    Id lor(Id a, Id b);
    Id lnot(Id a);
    Id select(Id condition, Id yes, Id no);
    const std::string& error() const { return constructionMessage.empty() ? emissionMessage : constructionMessage; }
    const std::string& constructionError() const { return constructionMessage; }
    const std::string& lastEmissionError() const { return emissionMessage; }
    std::size_t size() const { return nodes.size(); }
    std::optional<uint64_t> constantValue(Id expression) const;
    bool isBoolean(Id expression) const;
    // Sound constant specialization under an immutable Boolean premise.
    // Unknown means unsupported, not nonconstant. Uses O(G^2) Boolean proof
    // work in the worst case; no valuation search or arithmetic solver.
    std::optional<uint64_t> constantUnder(Id premise, Id expression);
    // Sound sufficient implication by forced Boolean facts, without arithmetic
    // reasoning, Boolean search or SAT. A conjunction of A distinct obligations costs O(A*G)
    // for G DAG nodes; one obligation costs O(G). Does not grow the DAG.
    bool implies(Id premise, Id consequence) const;

    // Arithmetic has unsigned 64-bit modular semantics. The producer establishes
    // a 64-bit index layout and guards against wrap where natural-number results
    // are required. Div/rem require a positive constant denominator, so every
    // expression is total even when its endpoint presence guard is false.
    // Inputs must already dominate the original cut; no source operation is
    // cloned or speculated. Non-i1 integers <=64 bits are zero-extended to index.
    // The builder must target detached preparation. Memo belongs to this DAG and
    // this cut only and remains valid while that preparation block is retained.
    // Invalid roots/inputs fail before any new operation is emitted. Placement
    // failures report lastEmissionError without invalidating the DAG or its
    // queries; that diagnostic is cleared at the next emission attempt.
    FailureOr<Value> emit(Id expression, OpBuilder& builder, Operation* cut,
                         llvm::DenseMap<Id, Value>& memo);

    struct CutEmission {
        llvm::DenseMap<Id, Value> values;
        llvm::DenseMap<Value, Value> inputs;
        llvm::DenseMap<Id, Id> cofactors;
    };
    // Context is private to one original cut. Enclosing branch decisions are
    // substituted before availability checks. Missing scalar inputs may be
    // replayed only through deterministic arith/index operations that are
    // memory-effect-free, region-free and speculatable. Pure alone does not
    // establish deterministic duplication (for example, LLVM freeze).
    // Payloads are registered explicitly and may never be replayed.
    void forbidRecomputation(Operation* operation) { forbiddenRecomputation.insert(operation); }
    FailureOr<Value> emitContextual(Id expression, OpBuilder& builder, Operation* cut, CutEmission& context);

private:
    enum class Kind { Constant, Input, Add, Sub, Div, Rem, Lt, Le, Eq, SLt, SLe, And, Or, Not, Select };
    struct Node {
        Kind kind = Kind::Constant;
        bool boolean = false;
        Id a = invalid, b = invalid, c = invalid;
        uint64_t literal = 0;
        Value value;
        bool operator==(const Node& other) const;
    };
    struct Hash { std::size_t operator()(const Node& node) const; };
    enum class Truth : uint8_t { Unknown, False, True };
    Truth evaluateBoolean(const Node& node, ArrayRef<Truth> values) const;
    bool refutesNegation(Id premise, Id consequence) const;
    Id intern(Node node);
    Id reject(const char* message);
    bool valid(Id id) const { return id < nodes.size(); }
    Id binary(Kind kind, Id a, Id b);
    Id absorbBoolean(Kind kind, Id a, Id b) const;
    std::optional<uint64_t> fold(Kind kind, uint64_t a, uint64_t b) const;
    bool prepareEmission(Id expression, OpBuilder& builder, Operation* cut,
                         const llvm::DenseMap<Id, Value>& memo, SmallVectorImpl<Id>& order);
    Value emitNode(const Node& node, OpBuilder& builder, Location location,
                   const llvm::DenseMap<Id, Value>& memo) const;
    Id cofactorAtCut(Id expression, Operation* cut, llvm::DenseMap<Id, Id>& memo);
    llvm::DenseSet<Operation*> forbiddenRecomputation;
    std::vector<Node> nodes;
    std::unordered_map<Node, Id, Hash> interned;
    std::string constructionMessage;
    std::string emissionMessage;
    Operation* dominanceRoot = nullptr;
    std::unique_ptr<DominanceInfo> dominance;
};
} // namespace mlir::pto::frontiersynch
#endif
