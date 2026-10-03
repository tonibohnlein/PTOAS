// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_SIGNEDINTERNAL_H
#define PTO_FRONTIERSYNCH_SIGNEDINTERNAL_H
#include "PTO/Transforms/FrontierSynch/SignedDemandAnalysis.h"
#include <map>
namespace mlir::pto::frontiersynch::signed_detail {
using BigInt = llvm::DynamicAPInt;
using Direction = std::pair<std::int64_t, std::int64_t>;
// Signed axis encoding +(index+1)/-(index+1); zero is the constant coordinate.
struct Poly {
    unsigned dimensions = 0;
    bool empty = false;
    std::map<Direction, BigInt> bounds;
    explicit Poly(unsigned dimensions = 0) : dimensions(dimensions) {}
    bool add(ArrayRef<std::int64_t> terms, BigInt constant);
    bool add(Direction direction, const BigInt& constant);
    void eliminate(unsigned axis);
    void project(ArrayRef<unsigned> retained);
    bool feasible() const;
    bool implies(const Poly& other) const;
    bool contains(ArrayRef<BigInt> point) const;
};
struct Layout {
    unsigned domain = 0, range = 0, parameters = 0;
    unsigned free() const { return domain + range + parameters; }
};
SignedResult<unsigned> active(const SignedSpaceHandle& space, SymbolicTuple role, const SignedTag& tag);
SignedResult<Layout> layout(
    const SignedSpaceHandle& space, SymbolicTuple domain, SymbolicTuple range, const SignedPiece& piece);
SignedResult<Poly> decode(
    const SignedSpaceHandle& space, SymbolicTuple domain, SymbolicTuple range, const SignedPiece& piece);
SignedPiece encode(
    const Poly& poly, const Layout& layout, SignedTag domain, SignedTag range, ArrayRef<BigInt> residues);
Poly remap(const Poly& source, ArrayRef<unsigned> mapping, unsigned dimensions);
void conjoin(Poly& target, const Poly& source);
bool signature(const SignedPiece& a, const SignedPiece& b);
void append(SmallVectorImpl<SignedPiece>& output, SignedPiece piece, const Layout& shape);
struct Access {
    static SignedRelationHandle make(
        SignedSpaceHandle space, SymbolicTuple domain, SymbolicTuple range, SmallVector<SignedPiece, 0> pieces);
};
SignedPoint point(const SymbolicEvent& event);
SignedResult<SignedRelationHandle> identity(SignedRelationHandle set);
SignedResult<SignedRelationHandle> lift(
    SignedRelationHandle relation, PeriodicEventKind source, PeriodicEventKind target);
SignedResult<SignedRelationHandle> filterPipes(SignedRelationHandle relation, PipelineType source, PipelineType target);
} // namespace mlir::pto::frontiersynch::signed_detail
#endif
