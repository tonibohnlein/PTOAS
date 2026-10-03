// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_SIGNEDDEMANDANALYSIS_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_SIGNEDDEMANDANALYSIS_H
#include "PTO/Transforms/FrontierSynch/SymbolicDemandAnalysis.h"
namespace mlir::pto::frontiersynch {
namespace signed_detail {
struct Access;
}
enum class SignedStatus {
    Success,
    InvalidInput,
    InvalidBinding,
    InvalidEvent,
    AbsentEndpoint,
    InadmissibleContext,
    NonFunctional,
    UnboundedOutput,
    RepresentationLimit
};
llvm::StringRef signedDiagnostic(SignedStatus status);
template <typename T>
struct SignedResult {
    SignedStatus status = SignedStatus::Success;
    T value{};
    bool succeeded() const { return status == SignedStatus::Success; }
};
class SignedSpace;
using SignedSpaceHandle = std::shared_ptr<const SignedSpace>;
class SignedSpace {
public:
    static SignedResult<SignedSpaceHandle> create(SymbolicSchemaHandle schema, const llvm::DynamicAPInt& period);
    SymbolicSchemaHandle schema() const { return owner; }
    const llvm::DynamicAPInt& period() const { return modulus; }

private:
    SignedSpace() = default;
    SymbolicSchemaHandle owner;
    llvm::DynamicAPInt modulus = llvm::DynamicAPInt(1);
};
// Finite tags are never numeric solver axes. Active coordinate meaning is taken
// from this exact schema site; source/range copies remain distinct.
struct SignedTag {
    std::optional<std::size_t> site;
    std::optional<PeriodicEventKind> kind;
    bool operator==(const SignedTag& other) const { return site == other.site && kind == other.kind; }
};
enum class SignedAxisRole { Domain, Range, Parameter, Local };
struct SignedAxis {
    SignedAxisRole role = SignedAxisRole::Domain;
    unsigned index = 0;
    bool operator==(const SignedAxis& other) const { return role == other.role && index == other.index; }
};
struct SignedTerm {
    SignedAxis axis;
    int sign = 1;
};
struct SignedAtom {
    // Sum of zero, one or two +/-1 terms <= bound. Equality is two atoms.
    SmallVector<SignedTerm, 2> terms;
    llvm::DynamicAPInt bound = llvm::DynamicAPInt(0);
};
struct SignedPiece {
    SignedTag domain, range;
    // ALL active domain axes, active range axes, then ordered SSA parameters.
    // Each original coordinate x is period*q+residue. Atoms constrain q.
    SmallVector<llvm::DynamicAPInt> residues;
    unsigned locals = 0;
    SmallVector<SignedAtom> atoms;
};
struct SignedPoint {
    SignedTag tag;
    // Original mathematical coordinates, NOT quotient coordinates.
    SmallVector<llvm::DynamicAPInt> coordinates;
};
class SignedRelation;
using SignedRelationHandle = std::shared_ptr<const SignedRelation>;
class SignedRelation {
public:
    // Supplied explicit residue split, not recognition of arbitrary Presburger
    // equivalents. Owns normalized pieces; private integer locals are eliminated.
    static SignedResult<SignedRelationHandle> import(
        SignedSpaceHandle space, SymbolicTuple domain, SymbolicTuple range, ArrayRef<SignedPiece> pieces);
    SignedSpaceHandle space() const { return owner; }
    SymbolicTuple domain() const { return source; }
    SymbolicTuple range() const { return target; }
    ArrayRef<SignedPiece> pieces() const { return data; }
    bool empty() const { return data.empty(); }
    SignedResult<SignedRelationHandle> unite(const SignedRelationHandle& other) const;
    SignedResult<SignedRelationHandle> intersect(const SignedRelationHandle& other) const;
    SignedResult<SignedRelationHandle> subtract(const SignedRelationHandle& other) const;
    SignedResult<SignedRelationHandle> compose(const SignedRelationHandle& other) const;
    SignedResult<SignedRelationHandle> inverse() const;
    SignedResult<SignedRelationHandle> domainSet() const;
    SignedResult<SignedRelationHandle> rangeSet() const;
    SignedResult<SignedRelationHandle> restrictDomain(const SignedRelationHandle& set) const;
    SignedResult<SignedRelationHandle> restrictRange(const SignedRelationHandle& set) const;
    SignedResult<SignedRelationHandle> restrictContext(const SignedRelationHandle& set) const;
    // Parameters also use original mathematical values in canonical SSA order.
    SignedResult<bool> contains(
        const SignedPoint& source, const SignedPoint& target, ArrayRef<llvm::DynamicAPInt> parameters) const;
    // Uniform, schema-annotated interchange; introduces fresh q locals and
    // x=P*q+r equations, never a call to general Presburger normalization.
    FailureOr<SymbolicPrimitive> toSymbolic() const;

private:
    friend struct signed_detail::Access;
    SignedRelation() = default;
    SignedSpaceHandle owner;
    SymbolicTuple source = SymbolicTuple::Unit, target = SymbolicTuple::Unit;
    SmallVector<SignedPiece, 0> data;
};
struct SignedInputs {
    SignedRelationHandle context, present, reference, reads, writes, extras;
    SignedRelationHandle generators;
};
struct SignedExpressionTerm {
    // Input coordinate or parameter quotient; absent axis means constant.
    std::optional<SignedAxis> axis;
    int sign = 1;
    llvm::DynamicAPInt constant = llvm::DynamicAPInt(0);
};
struct SignedOutputCoordinate {
    SmallVector<SignedExpressionTerm> lower, upper;
};
struct SignedSelectorPiece {
    SignedRelationHandle guard; // Unit -> input Event; includes parameter residues.
    SignedTag output;
    SmallVector<llvm::DynamicAPInt> residues;
    SmallVector<SignedOutputCoordinate> coordinates;
};
class SignedSelector;
using SignedSelectorHandle = std::shared_ptr<const SignedSelector>;
class SignedSelector {
public:
    // Exact functionality is checked, including overlapping output tags/residues.
    static SignedResult<SignedSelectorHandle> build(SignedRelationHandle oriented);
    SignedSpaceHandle space() const { return relation->space(); }
    ArrayRef<SignedSelectorPiece> pieces() const { return data; }
    SignedResult<std::optional<SymbolicEvent>> evaluate(
        const SymbolicEvent& input, ArrayRef<llvm::DynamicAPInt> parameters) const;

private:
    SignedSelector() = default;
    SignedRelationHandle relation;
    SmallVector<SignedSelectorPiece, 0> data;
};
class SignedDemandAnalysis;
using SignedAnalysisHandle = std::shared_ptr<const SignedDemandAnalysis>;
class SignedDemandAnalysis {
public:
    // Same supplied exactness, strict reference/native-chain and finite-execution
    // per valuation premises as M5. P, total primitive dimension and pipe count
    // are fixed for the polynomial class. No numeric P/trip enumeration occurs.
    static SignedResult<SignedAnalysisHandle> build(SignedSpaceHandle space, const SignedInputs& inputs);
    SignedSpaceHandle space() const { return owner; }
    SignedRelationHandle context() const { return admitted; }
    SignedRelationHandle presence() const { return present; }
    SignedRelationHandle identity() const { return id; }
    SignedRelationHandle native() const { return n; }
    SignedRelationHandle generators() const { return g; }
    SignedRelationHandle reachability() const { return r; }
    SignedRelationHandle strictReachability() const { return h; }
    SignedRelationHandle minimum() const { return f; }
    SignedResult<SignedSelectorHandle> outgoing(PipelineType source, PipelineType target) const;
    SignedResult<SignedSelectorHandle> incoming(PipelineType source, PipelineType target) const;

private:
    SignedDemandAnalysis() = default;
    SignedSpaceHandle owner;
    SignedRelationHandle admitted, present, id, n, g, r, h, f;
    struct PipePair {
        PipelineType source = PipelineType::PIPE_UNASSIGNED, target = PipelineType::PIPE_UNASSIGNED;
        SignedSelectorHandle incoming, outgoing;
    };
    SmallVector<PipePair> selectors;
};
class BoundSignedAnalysis;
using BoundSignedHandle = std::shared_ptr<const BoundSignedAnalysis>;
class BoundSignedAnalysis {
public:
    static SignedResult<BoundSignedHandle> bind(
        SignedAnalysisHandle analysis, ArrayRef<ImmutableParameterBinding> bindings);
    SignedAnalysisHandle analysis() const { return owner; }
    std::shared_ptr<const SymbolicContext> boundContext() const { return context; }
    SignedResult<bool> contains(const SymbolicEvent& event) const;
    SignedResult<bool> reaches(const SymbolicEvent& source, const SymbolicEvent& target) const;
    SignedResult<bool> minimumDemand(const SymbolicEvent& source, const SymbolicEvent& target) const;
    SignedResult<std::optional<SymbolicEvent>> incoming(const SymbolicEvent& target, PipelineType source) const;
    SignedResult<std::optional<SymbolicEvent>> outgoing(const SymbolicEvent& source, PipelineType target) const;

private:
    BoundSignedAnalysis() = default;
    SignedResult<bool> pair(
        SignedRelationHandle relation, const SymbolicEvent& source, const SymbolicEvent& target) const;
    SignedAnalysisHandle owner;
    std::shared_ptr<const SymbolicContext> context;
};
// All handles/snapshots own arithmetic data. Borrowed MLIR schema identities and
// their contexts must outlive them. Restricted solving uses no MLIR solver; later
// consumers of interchange retain their own backend capability requirements.
} // namespace mlir::pto::frontiersynch
#endif
