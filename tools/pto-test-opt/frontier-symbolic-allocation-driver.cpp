// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite occurrence/slot-owner oracle for the uniform symbolic
// reuse certificate. Relations are supplied directly, without source admission.
#include "../../lib/PTO/Transforms/FrontierSynch/ExplicitPhysicalEmission.h"
#include "../../lib/PTO/Transforms/FrontierSynch/GeneralPhysicalEmission.h"
#include "PTO/Transforms/FrontierSynch/SignedDemandAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
namespace {
using namespace mlir;
namespace fs = pto::frontiersynch;
using Int = llvm::DynamicAPInt;
using Role = fs::SignedAxisRole;
using Kind = fs::PeriodicEventKind;
using Relation = fs::SignedRelationHandle;
using Pipe = pto::PipelineType;
struct Relations {
    Relation demand, native, readiness;
};
fs::SignedAtom atom(std::initializer_list<fs::SignedTerm> terms, int64_t bound)
{
    return {SmallVector<fs::SignedTerm, 2>(terms), Int(bound)};
}
fs::SignedTerm term(Role role, int sign)
{
    return {{role, 0}, sign};
}
fs::SignedPiece piece(std::size_t source, Kind sourceKind, std::size_t target, Kind targetKind,
                      std::optional<unsigned> fixed)
{
    fs::SignedPiece result;
    result.domain = {source, sourceKind};
    result.range = {target, targetKind};
    result.residues = {Int(0), Int(0), Int(0)};
    result.atoms = {
        atom({term(Role::Parameter, -1)}, 0),
        atom({term(Role::Domain, -1)}, 0),
        atom({term(Role::Range, -1)}, 0),
        atom({term(Role::Domain, 1), term(Role::Parameter, -1)}, -1),
        atom({term(Role::Range, 1), term(Role::Parameter, -1)}, -1)};
    if (fixed) {
        result.atoms.push_back(atom({term(Role::Parameter, 1)}, *fixed));
        result.atoms.push_back(atom({term(Role::Parameter, -1)}, -static_cast<int64_t>(*fixed)));
    }
    return result;
}
FailureOr<Relation> import(fs::SignedSpaceHandle space, ArrayRef<fs::SignedPiece> pieces)
{
    auto result = fs::SignedRelation::import(space, fs::SymbolicTuple::Event, fs::SymbolicTuple::Event, pieces);
    if (!result.succeeded()) { return failure(); }
    return result.value;
}
FailureOr<Relations> relations(fs::SignedSpaceHandle space, unsigned distance,
                              std::optional<unsigned> fixed, bool missing, bool reversed)
{
    auto handoff = piece(0, Kind::Completion, 1, Kind::Start, fixed);
    handoff.atoms.push_back(atom({term(Role::Domain, 1), term(Role::Range, -1)}, 0));
    handoff.atoms.push_back(atom({term(Role::Range, 1), term(Role::Domain, -1)}, 0));
    SmallVector<fs::SignedPiece> order;
    for (std::size_t site : {0, 1}) {
        for (Kind kind : {Kind::Completion, Kind::Start}) {
            auto chain = piece(site, kind, site, kind, fixed);
            auto first = term(site == 1 && reversed ? Role::Range : Role::Domain, 1);
            auto second = term(site == 1 && reversed ? Role::Domain : Role::Range, -1);
            chain.atoms.push_back(atom({first, second}, 0));
            order.push_back(chain);
        }
    }
    SmallVector<fs::SignedPiece> ready;
    if (!missing) {
        auto availability = piece(1, Kind::Completion, 0, Kind::Start, fixed);
        availability.atoms.push_back(atom({term(Role::Domain, 1), term(Role::Range, -1)},
            -static_cast<int64_t>(distance)));
        ready.push_back(availability);
    }
    auto demand = import(space, {handoff});
    auto native = import(space, order);
    auto readiness = import(space, ready);
    if (failed(demand) || failed(native) || failed(readiness)) { return failure(); }
    return Relations{*demand, *native, *readiness};
}
FailureOr<bool> query(Relation relation, std::size_t source, Kind sourceKind, int64_t i,
                      std::size_t target, Kind targetKind, int64_t j, unsigned n)
{
    fs::SignedPoint first{{source, sourceKind}, {Int(i)}};
    fs::SignedPoint second{{target, targetKind}, {Int(j)}};
    auto answer = relation->contains(first, second, {Int(n)});
    if (!answer.succeeded()) { return failure(); }
    return answer.value;
}
struct GeneralRelations {
    presburger::PresburgerRelation demand, native, readiness;
};
FailureOr<GeneralRelations> exportGeneral(const Relations& inputs)
{
    auto demand = inputs.demand->toSymbolic(), native = inputs.native->toSymbolic();
    auto readiness = inputs.readiness->toSymbolic();
    if (failed(demand) || failed(native) || failed(readiness)) { return failure(); }
    return GeneralRelations{demand->relation(), native->relation(), readiness->relation()};
}
FailureOr<bool> query(const presburger::PresburgerRelation& relation,
                      std::size_t source, Kind sourceKind, int64_t i,
                      std::size_t target, Kind targetKind, int64_t j, unsigned n)
{
    // This fixture has one coordinate per site, one parameter, and literal
    // site/kind axes. Check exact interchange points, including integer locals.
    return relation.containsPoint(SmallVector<Int>{Int(source), Int(i), Int(static_cast<int64_t>(sourceKind)),
        Int(target), Int(j), Int(static_cast<int64_t>(targetKind)), Int(n)});
}
// Change only site 1's coordinate from ordinal i to y=2*i+1. Multiplying
// complete constraint rows clears the substitution denominator; retained
// integer locals preserve the odd-coordinate lattice exactly.
void warpRows(presburger::IntegerRelation& piece, unsigned sourceScale, unsigned targetScale, bool equalities)
{
    unsigned rows = equalities ? piece.getNumEqualities() : piece.getNumInequalities();
    unsigned targetAxis = piece.getNumDomainVars() + 1;
    unsigned constant = piece.getNumCols() - 1;
    unsigned factor = sourceScale * targetScale;
    auto coefficient = [&piece, equalities](unsigned row, unsigned column) -> Int& {
        return equalities ? piece.atEq(row, column) : piece.atIneq(row, column);
    };
    for (unsigned row = 0; row < rows; ++row) {
        Int source = coefficient(row, 1), target = coefficient(row, targetAxis);
        for (unsigned column = 0; column < piece.getNumCols(); ++column) {
            coefficient(row, column) *= Int(factor);
        }
        coefficient(row, 1) = source * Int(factor / sourceScale);
        coefficient(row, targetAxis) = target * Int(factor / targetScale);
        if (sourceScale == 2) { coefficient(row, constant) -= source * Int(factor / sourceScale); }
        if (targetScale == 2) { coefficient(row, constant) -= target * Int(factor / targetScale); }
    }
}
struct AffineRelation { presburger::PresburgerRelation value; };
struct AffineRelations { AffineRelation demand, native, readiness; };
FailureOr<AffineRelation> warp(const presburger::PresburgerRelation& input)
{
    auto result = presburger::PresburgerRelation::getEmpty(input.getSpace());
    for (auto piece : input.getAllDisjuncts()) {
        auto source = piece.getConstantBound64(presburger::BoundType::EQ, 0);
        auto target = piece.getConstantBound64(presburger::BoundType::EQ, piece.getNumDomainVars());
        if (piece.getNumDomainVars() != 3 || piece.getNumRangeVars() != 3 ||
            !source || !target || *source < 0 || *source > 1 || *target < 0 || *target > 1) { return failure(); }
        unsigned sourceScale = *source == 1 ? 2 : 1, targetScale = *target == 1 ? 2 : 1;
        warpRows(piece, sourceScale, targetScale, true);
        warpRows(piece, sourceScale, targetScale, false);
        result.unionInPlace(piece);
    }
    return AffineRelation{result};
}
FailureOr<AffineRelations> affineRelations(const Relations& input)
{
    auto general = exportGeneral(input);
    if (failed(general)) { return failure(); }
    auto demand = warp(general->demand), native = warp(general->native), readiness = warp(general->readiness);
    if (failed(demand) || failed(native) || failed(readiness)) { return failure(); }
    return AffineRelations{*demand, *native, *readiness};
}
FailureOr<bool> query(const AffineRelation& relation,
                      std::size_t source, Kind sourceKind, int64_t i,
                      std::size_t target, Kind targetKind, int64_t j, unsigned n)
{
    int64_t sourceCoordinate = source == 1 ? 2 * i + 1 : i;
    int64_t targetCoordinate = target == 1 ? 2 * j + 1 : j;
    return query(relation.value, source, sourceKind, sourceCoordinate, target, targetKind, targetCoordinate, n);
}
template <typename Inputs>
bool checkQueries(const Inputs& inputs, unsigned n, unsigned distance, bool missing, bool reversed)
{
    int64_t bound = static_cast<int64_t>(n);
    for (int64_t i = -1; i <= bound; ++i) {
        for (int64_t j = -1; j <= bound; ++j) {
            bool present = i >= 0 && j >= 0 && i < bound && j < bound;
            auto pair = query(inputs.demand, 0, Kind::Completion, i, 1, Kind::Start, j, n);
            auto ready = query(inputs.readiness, 1, Kind::Completion, i, 0, Kind::Start, j, n);
            if (failed(pair) || failed(ready) || *pair != (present && i == j) ||
                *ready != (present && !missing && j - i >= static_cast<int64_t>(distance))) { return false; }
            for (std::size_t site : {0, 1}) {
                for (Kind kind : {Kind::Completion, Kind::Start}) {
                    auto native = query(inputs.native, site, kind, i, site, kind, j, n);
                    bool expected = present && (site == 1 && reversed ? j <= i : i <= j);
                    if (failed(native) || *native != expected) { return false; }
                }
            }
        }
    }
    return true;
}
// Enumerate actual handoffs and reuse an owned slot. Each reused slot must have
// its previous consumer completed before the new producer issues. This oracle
// does not calculate strict-order differences, successors, or relation powers.
template <typename Inputs>
FailureOr<bool> finiteSlots(const Inputs& inputs, unsigned n, std::size_t capacity)
{
    SmallVector<int64_t> consumers;
    for (unsigned producer = 0; producer < n; ++producer) {
        std::optional<int64_t> consumer;
        for (unsigned target = 0; target < n; ++target) {
            auto paired = query(inputs.demand, 0, Kind::Completion, producer, 1, Kind::Start, target, n);
            if (failed(paired)) { return failure(); }
            if (*paired) {
                if (consumer) { return failure(); }
                consumer = target;
            }
        }
        if (!consumer) { return failure(); }
        consumers.push_back(*consumer);
    }
    for (std::size_t before = 0; before < consumers.size(); ++before) {
        for (std::size_t after = before + 1; after < consumers.size(); ++after) {
            auto ordered = query(inputs.native, 1, Kind::Start, consumers[before],
                1, Kind::Start, consumers[after], n);
            if (failed(ordered)) { return failure(); }
            if (!*ordered) { return false; }
        }
    }
    if (consumers.empty()) { return true; }
    if (capacity == 0) { return false; }
    SmallVector<std::optional<int64_t>> owners(capacity);
    std::size_t slot = 0;
    for (std::size_t producer = 0; producer < consumers.size(); ++producer) {
        if (owners[slot]) {
            auto released = query(inputs.readiness, 1, Kind::Completion, *owners[slot],
                0, Kind::Start, producer, n);
            if (failed(released)) { return failure(); }
            if (!*released) { return false; }
        }
        owners[slot] = consumers[producer];
        slot = slot + 1 == capacity ? 0 : slot + 1;
    }
    return true;
}
bool checkCase(fs::SignedSpaceHandle space, unsigned distance, std::optional<unsigned> fixed,
               bool missing, bool reversed, std::size_t capacity)
{
    auto input = relations(space, distance, fixed, missing, reversed);
    if (failed(input)) { return false; }
    bool expected = true;
    unsigned first = fixed.value_or(0);
    unsigned last = fixed.value_or(8);
    for (unsigned n = first; n <= last; ++n) {
        if (!checkQueries(*input, n, distance, missing, reversed)) { return false; }
        auto finite = finiteSlots(*input, n, capacity);
        if (failed(finite)) { return false; }
        expected = expected && *finite;
    }
    // For this explicit family every possible counterexample occurs by n=8:
    // capacities <=3 and readiness distance <=3. Larger n introduces no new
    // relative position or slot-owner condition; this is not a general bound.
    std::string reason;
    bool certified = succeeded(fs::certifySymbolicReuse(input->demand, input->native, input->readiness,
        Pipe::PIPE_MTE2, Pipe::PIPE_V, capacity, reason));
    if (certified != expected || (!certified &&
        (reason.empty() || llvm::StringRef(reason).starts_with("unsupported:")))) {
        llvm::errs() << "symbolic allocation oracle mismatch: distance=" << distance << " capacity=" << capacity
                     << " fixed=" << (fixed ? static_cast<int64_t>(*fixed) : -1)
                     << " missing=" << missing << " reversed=" << reversed << " reason=" << reason << "\n";
        return false;
    }
    auto general = exportGeneral(*input);
    if (failed(general)) { return false; }
    bool generalExpected = true;
    for (unsigned n = first; n <= last; ++n) {
        if (!checkQueries(*general, n, distance, missing, reversed)) { return false; }
        auto finite = finiteSlots(*general, n, capacity);
        if (failed(finite)) { return false; }
        generalExpected = generalExpected && *finite;
    }
    reason.clear();
    bool generalCertified = succeeded(fs::certifyGeneralReuse(space->schema(), general->demand,
        general->native, general->readiness, Pipe::PIPE_MTE2, Pipe::PIPE_V, capacity, reason));
    if (generalExpected != expected || generalCertified != generalExpected || (!generalCertified &&
        (reason.empty() || llvm::StringRef(reason).starts_with("unsupported:")))) {
        llvm::errs() << "general exported allocation oracle mismatch: distance=" << distance
                     << " capacity=" << capacity << " reason=" << reason << "\n";
        return false;
    }
    return true;
}
bool checkAffineCase(fs::SignedSpaceHandle space, std::optional<unsigned> fixed,
                     bool missing, bool reversed, std::size_t capacity)
{
    auto input = relations(space, 2, fixed, missing, reversed);
    if (failed(input)) { return false; }
    auto affine = affineRelations(*input);
    if (failed(affine)) { return false; }
    bool expected = true;
    unsigned first = fixed.value_or(0), last = fixed.value_or(8);
    for (unsigned n = first; n <= last; ++n) {
        if (!checkQueries(*affine, n, 2, missing, reversed)) { return false; }
        auto finite = finiteSlots(*affine, n, capacity);
        if (failed(finite)) { return false; }
        expected = expected && *finite;
    }
    unsigned witnessLength = fixed.value_or(4);
    if (witnessLength != 0) {
        // Direct original-coordinate points distinguish y=2*i+1 from an
        // ordinal identity map and reject the intervening even coordinate.
        if (!affine->demand.value.containsPoint(SmallVector<Int>{Int(0), Int(0), Int(1),
                Int(1), Int(1), Int(0), Int(witnessLength)}) ||
            affine->demand.value.containsPoint(SmallVector<Int>{Int(0), Int(0), Int(1),
                Int(1), Int(0), Int(0), Int(witnessLength)})) { return false; }
        if (witnessLength > 2 &&
            !affine->demand.value.containsPoint(SmallVector<Int>{Int(0), Int(2), Int(1),
                Int(1), Int(5), Int(0), Int(witnessLength)})) { return false; }
    }
    std::string reason;
    bool certified = succeeded(fs::certifyGeneralReuse(space->schema(), affine->demand.value,
        affine->native.value, affine->readiness.value, Pipe::PIPE_MTE2, Pipe::PIPE_V, capacity, reason));
    if (certified != expected || (!certified &&
        (reason.empty() || llvm::StringRef(reason).starts_with("unsupported:")))) {
        llvm::errs() << "general affine allocation oracle mismatch: capacity=" << capacity
                     << " missing=" << missing << " reversed=" << reversed << " reason=" << reason << "\n";
        return false;
    }
    return true;
}
} // namespace
int runSymbolicAllocationChecks(mlir::MLIRContext& context)
{
    auto module = parseSourceString<ModuleOp>(
        "module { func.func @schema(%source: index, %target: index, %n: index) { return } }", &context);
    if (!module) { return 1; }
    auto function = *module->getOps<func::FuncOp>().begin();
    SmallVector<std::unique_ptr<pto::CompoundInstanceElement>> phases;
    SmallVector<fs::SymbolicSite> sites;
    for (std::size_t site : {0, 1}) {
        phases.push_back(std::make_unique<pto::CompoundInstanceElement>(site,
            SmallVector<const pto::BaseMemInfo*>{}, SmallVector<const pto::BaseMemInfo*>{},
            site == 0 ? Pipe::PIPE_MTE2 : Pipe::PIPE_V, OperationName("test.phase", &context)));
        sites.push_back({phases.back().get(), {function.getArgument(site)}});
    }
    auto schema = fs::SymbolicSchema::create(sites, {function.getArgument(2)}, {}, 0);
    if (failed(schema)) { return 1; }
    auto space = fs::SignedSpace::create(*schema, Int(1));
    if (!space.succeeded()) { return 1; }
    for (unsigned distance : {1, 2, 3}) {
        for (std::size_t capacity : {0, 1, 2, 3}) {
            if (!checkCase(space.value, distance, std::nullopt, false, false, capacity)) { return 1; }
        }
    }
    for (unsigned n : {0, 1, 2, 3}) {
        for (std::size_t capacity : {0, 1, 2, 3}) {
            if (!checkCase(space.value, 2, n, false, false, capacity)) { return 1; }
        }
    }
    for (std::size_t capacity : {1, 2, 3}) {
        if (!checkCase(space.value, 2, std::nullopt, true, false, capacity) ||
            !checkCase(space.value, 2, std::nullopt, false, true, capacity)) { return 1; }
    }
    for (std::size_t capacity : {0, 1, 2, 3}) {
        if (!checkAffineCase(space.value, std::nullopt, false, false, capacity)) { return 1; }
    }
    if (!checkAffineCase(space.value, 0, false, false, 0) ||
        !checkAffineCase(space.value, 1, false, false, 1) ||
        !checkAffineCase(space.value, std::nullopt, true, false, 2) ||
        !checkAffineCase(space.value, std::nullopt, false, true, 2)) { return 1; }
    llvm::outs() << "symbolic allocation: 34 Signed, 34 exact-export General, and 8 non-DB affine "
                    "independent point/slot oracles passed\n";
    return 0;
}
