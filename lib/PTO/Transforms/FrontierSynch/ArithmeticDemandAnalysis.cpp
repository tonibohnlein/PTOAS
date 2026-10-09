// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include "llvm/ADT/DenseSet.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include <algorithm>
#include <numeric>
#include <tuple>
#include <type_traits>
namespace mlir::pto::frontiersynch {
bool ArithmeticEventKey::operator<(const ArithmeticEventKey& other) const
{
    return std::tie(site, event, residues) < std::tie(other.site, other.event, other.residues);
}
bool ArithmeticEventKey::operator==(const ArithmeticEventKey& other) const
{
    return site == other.site && event == other.event && residues == other.residues;
}
bool ArithmeticRelationKey::operator<(const ArithmeticRelationKey& other) const
{
    return std::tie(source, target, parameterResidues) <
           std::tie(other.source, other.target, other.parameterResidues);
}
namespace {
template<class System>
struct ImportedPiece {
    const PrimitiveRelation* schema;
    std::vector<uint64_t> residues; // Original dimension order, then canonical parameter order.
    System system;
};
template<class System>
void append(std::vector<System>& pieces, System piece)
{
    if (piece.isEmpty()) { return; }
    for (const auto& existing : pieces) {
        if (piece.isSubsetOf(existing)) { return; }
    }
    pieces.erase(std::remove_if(pieces.begin(), pieces.end(),
                               [&](const System& old) { return old.isSubsetOf(piece); }), pieces.end());
    pieces.push_back(std::move(piece));
}
template<class System>
void append(TypedArithmeticRelation<System>& relation, const ArithmeticRelationKey& key, System piece)
{
    if (!piece.isEmpty()) { append(relation[key], std::move(piece)); }
}
template<class System>
void unite(TypedArithmeticRelation<System>& target, const TypedArithmeticRelation<System>& source)
{
    for (const auto& [key, pieces] : source) {
        for (const auto& piece : pieces) { append(target, key, piece); }
    }
}
unsigned dimensions(const ArithmeticRelationKey& key)
{
    return key.source.residues.size() + key.target.residues.size() + key.parameterResidues.size();
}
struct DifferencePolicy {
    using System = DifferenceBoundSystem;
    static bool accepts(ArithmeticClass kind) { return kind == ArithmeticClass::Differences; }
    static FailureOr<System> create(unsigned dimensions, ArrayRef<LinearRow> rows)
    {
        std::vector<DifferenceBoundConstraint> constraints;
        for (const auto& row : rows) {
            if (row.coefficients.size() != dimensions) { return failure(); }
            unsigned positive = 0, negative = 0;
            for (unsigned i = 0; i < dimensions; ++i) {
                auto coefficient = row.coefficients[i];
                if (!coefficient) { continue; }
                if ((coefficient != 1 && coefficient != -1) ||
                    (coefficient == 1 ? positive != 0 : negative != 0)) { return failure(); }
                (coefficient == 1 ? positive : negative) = i + 1;
            }
            BoundInteger bound(row.constant);
            constraints.push_back({negative, positive, bound});
            if (row.equality) { constraints.push_back({positive, negative, -bound}); }
        }
        return System::create(dimensions, constraints);
    }
    static FailureOr<std::vector<System>> project(const System& system, ArrayRef<unsigned> keep)
    {
        auto projected = system.project(keep);
        if (failed(projected)) { return failure(); }
        return std::vector<System>{std::move(*projected)};
    }
    static FailureOr<std::vector<System>> subtract(unsigned dimensions, ArrayRef<System> a, ArrayRef<System> b)
    {
        return subtractDifferenceBoundUnions(dimensions, a, b);
    }
};
struct IntegerPolicy {
    using System = IntegerSystem;
    static bool accepts(ArithmeticClass kind)
    {
        return kind == ArithmeticClass::Differences || kind == ArithmeticClass::Octagons ||
               kind == ArithmeticClass::BoundedCoefficients;
    }
    static FailureOr<System> create(unsigned dimensions, ArrayRef<LinearRow> rows)
    {
        std::vector<IntegerConstraint> constraints;
        for (const auto& row : rows) {
            if (row.coefficients.size() != dimensions) { return failure(); }
            IntegerConstraint forward;
            forward.bound = BoundInteger(row.constant);
            for (auto coefficient : row.coefficients) {
                forward.coefficients.push_back(-BoundInteger(coefficient));
            }
            constraints.push_back(forward);
            if (row.equality) {
                for (auto& coefficient : forward.coefficients) { coefficient = -coefficient; }
                forward.bound = -forward.bound;
                constraints.push_back(std::move(forward));
            }
        }
        return System::create(dimensions, constraints, {});
    }
    static FailureOr<std::vector<System>> project(const System& system, ArrayRef<unsigned> keep)
    {
        return system.project(keep);
    }
    static FailureOr<std::vector<System>> subtract(unsigned dimensions, ArrayRef<System> a, ArrayRef<System> b)
    {
        return subtractIntegerUnions(dimensions, a, b);
    }
};
template<class Policy>
class Analysis {
    using System = typename Policy::System;
    using Relation = TypedArithmeticRelation<System>;
    using Piece = ImportedPiece<System>;
public:
    explicit Analysis(const ArithmeticProgram& program, const StructuredProtection* facts)
        : program(program), protection(facts)
    {
        result.period = program.primitives.period;
        result.parameterCount = program.primitives.parameters.size();
        result.pipeCount = program.primitives.pipeCount;
    }
    TypedArithmeticDemandAnalysis<System> result;
    void run(bool generatorsOnly = false)
    {
        if (program.extraction.state != RecognitionState::Applicable ||
            program.recognition.state != RecognitionState::Applicable ||
            !Policy::accepts(program.recognition.arithmeticClass) || !result.period) {
            fail("exact supported arithmetic primitives are required"); return;
        }
        if (!import() || !buildRelations() || !buildConflicts()) { return; }
        if (!generatorsOnly) { complete(); }
    }
    std::vector<TypedArithmeticOccurrenceDomain<System>> occurrenceDomains() const
    {
        std::vector<TypedArithmeticOccurrenceDomain<System>> domains;
        for (const auto& piece : pieces) {
            if (piece.schema->kind != PrimitiveKind::Occurrences || !piece.schema->sourceSite) { continue; }
            domains.push_back({*piece.schema->sourceSite,
                {piece.residues.begin(), piece.residues.begin() + piece.schema->sourceDimensions},
                parameterResidues(piece), piece.system});
        }
        return domains;
    }
    void complete()
    {
        if (!result.error.empty()) { return; }
        Relation nativeCompletions;
        for (const auto& [key, pieces] : result.nativeOrder) {
            if (key.source.event == ArithmeticEvent::Completion && key.target.event == ArithmeticEvent::Start) {
                nativeCompletions.emplace(key, pieces);
            }
        }
        if (!nativeCompletions.empty()) {
            // The core native closure is supplied. Any added C->I edges obey
            // the same chain-shortcut bound as storage generators.
            auto nativeStep = compose(nativeCompletions, result.nativeOrder);
            for (unsigned round = 0; round < result.pipeCount && result.error.empty(); ++round) {
                auto next = compose(result.nativeOrder, nativeStep);
                unite(result.nativeOrder, next);
            }
        }
        auto step = compose(result.generators, result.nativeOrder);
        auto reach = result.nativeOrder;
        for (unsigned round = 0; round < result.pipeCount && result.error.empty(); ++round) {
            auto next = compose(reach, step);
            unite(reach, next);
        }
        if (!result.error.empty()) { return; }
        result.requiredOrder = subtract(reach, identity);
        // Only generator endpoint keys can contribute to this subtraction.
        // Required-order exports remain complete for later regional queries.
        auto covered = compose(result.requiredOrder, result.requiredOrder, &result.generators);
        unite(covered, result.nativeOrder);
        result.minimumDemands = subtract(result.generators, covered);
        if (!result.error.empty()) { return; }
        // A same-pipe demand is adjacent precisely when no strictly intermediate
        // payload exists on that pipe. Remove identities explicitly even if the
        // producer supplied the reflexive native closure.
        auto strictPipeOrder = subtract(samePipeOrder, identity);
        auto nonadjacent = compose(strictPipeOrder, strictPipeOrder);
        Relation localIntervals;
        for (const auto& [key, pieces] : nonadjacent) {
            auto endpoints = key;
            endpoints.source.event = ArithmeticEvent::Completion;
            endpoints.target.event = ArithmeticEvent::Start;
            for (const auto& piece : pieces) { append(localIntervals, endpoints, piece); }
        }
        auto badLocal = intersect(result.minimumDemands, localIntervals);
        if (!result.error.empty()) { return; }
        result.adjacentLocalDemands = badLocal.empty();
        result.exactMinimum = true;
        for (const auto& [key, pieces] : result.minimumDemands) {
            (void)key;
            result.cost.outputPieces += pieces.size();
        }
    }
private:
    const ArithmeticProgram& program;
    const StructuredProtection* protection;
    std::vector<Piece> pieces;
    Relation identity, order, samePipeOrder;
    void fail(const char* message) { if (result.error.empty()) { result.error = message; } }
    std::vector<uint64_t> parameterResidues(const Piece& piece) const
    {
        return {piece.residues.end() - result.parameterCount, piece.residues.end()};
    }
    bool canonicalOrder(const PrimitiveRelation& relation, std::vector<unsigned>& keep)
    {
        const bool access = relation.kind == PrimitiveKind::Reads || relation.kind == PrimitiveKind::Writes;
        if (relation.dimensions != relation.sourceDimensions + relation.targetDimensions + (access ? 1 : 0) ||
            relation.coordinates.size() != relation.dimensions + result.parameterCount) {
            fail("arithmetic primitive contains unsupported auxiliary coordinates"); return false;
        }
        for (unsigned i = 0; i < relation.dimensions; ++i) {
            const auto expected = access && i + 1 == relation.dimensions ?
                                  CoordinateKind::Storage : CoordinateKind::Occurrence;
            if (relation.coordinates[i].kind != expected) {
                fail("arithmetic primitive endpoint/storage coordinate order is invalid"); return false;
            }
            keep.push_back(i);
        }
        for (const auto& name : program.primitives.parameters) {
            auto found = relation.coordinates.end();
            for (auto it = relation.coordinates.begin() + relation.dimensions;
                 it != relation.coordinates.end(); ++it) {
                if (it->kind == CoordinateKind::Parameter && it->name == name) { found = it; break; }
            }
            if (found == relation.coordinates.end()) {
                fail("arithmetic primitive does not preserve the common parameter tuple"); return false;
            }
            keep.push_back(found - relation.coordinates.begin());
        }
        return true;
    }
    bool import()
    {
        std::vector<Piece> raw;
        DenseMap<Value, DenseSet<unsigned>> writtenSpaces;
        if (program.finiteExpansion) {
            for (const auto& schema : program.primitives.relations) {
                const bool writer = schema.kind == PrimitiveKind::Writes && !schema.pieces.empty();
                if (writer && schema.storageSpace) {
                    writtenSpaces[schema.storageBase].insert(static_cast<unsigned>(*schema.storageSpace));
                }
            }
        }
        for (const auto& normalized : program.recognition.pieces) {
            if (normalized.empty) { continue; }
            if (normalized.relation >= program.primitives.relations.size()) {
                fail("invalid normalized arithmetic relation"); return false;
            }
            const auto& schema = program.primitives.relations[normalized.relation];
            if (normalized.piece >= schema.pieces.size()) {
                fail("invalid normalized arithmetic piece"); return false;
            }
            std::vector<unsigned> keep;
            if (!canonicalOrder(schema, keep)) { return false; }
            const auto liftedDimensions = schema.pieces[normalized.piece].system.getNumInputs();
            auto primitive = Policy::create(liftedDimensions, normalized.rows);
            if (failed(primitive)) { fail("arithmetic primitive construction failed"); return false; }
            if (program.finiteExpansion && schema.kind == PrimitiveKind::Reads) {
                const bool valid = schema.storageSpace && schema.sourceSite &&
                    *schema.sourceSite < program.sites.size() && !schema.targetDimensions;
                if (!valid) { fail("finite read primitive has an invalid storage/endpoint schema"); return false; }
                auto found = writtenSpaces.find(schema.storageBase);
                const bool readOnly = found == writtenSpaces.end() ||
                    !found->second.count(static_cast<unsigned>(*schema.storageSpace));
                // Such a read contributes no internal conflict. Keep its full
                // original primitive in program for future storage exports,
                // but avoid expensive quotient projection in demands-only import.
                if (readOnly) { continue; }
            }
            std::vector<uint64_t> residues;
            for (auto column : keep) { residues.push_back(schema.pieces[normalized.piece].residues[column]); }
            if (liftedDimensions == keep.size()) {
                // Parameter canonicalization is a permutation. Keep the cheap
                // conjunction path when there are no existential quotients.
                std::vector<unsigned> permutation(keep.size());
                for (unsigned i = 0; i < keep.size(); ++i) { permutation[keep[i]] = i; }
                auto canonical = primitive->remap(keep.size(), permutation);
                if (failed(canonical)) { fail("arithmetic parameter reorder failed"); return false; }
                ++result.cost.primitivePieces;
                if (!canonical->isEmpty()) { raw.push_back({&schema, std::move(residues), std::move(*canonical)}); }
            } else {
                // Exact projection can split and introduce congruences; retain
                // every resulting conjunction with the original endpoint tags.
                auto projected = Policy::project(*primitive, keep);
                if (failed(projected)) { fail("arithmetic quotient projection failed"); return false; }
                ++result.cost.projections;
                for (auto& canonical : *projected) {
                    ++result.cost.primitivePieces;
                    if (!canonical.isEmpty()) { raw.push_back({&schema, residues, std::move(canonical)}); }
                }
            }
        }
        // Restrict every primitive to the same admitted parameter context.
        // Distinct residue cases are incompatible executions, never independent
        // existential choices at either endpoint of a join.
        std::vector<const Piece*> contexts;
        for (const auto& piece : raw) {
            if (piece.schema->kind == PrimitiveKind::Context) { contexts.push_back(&piece); }
        }
        for (const auto& piece : raw) {
            if (piece.schema->kind == PrimitiveKind::Context) { continue; }
            for (const auto* context : contexts) {
                if (parameterResidues(*context) != parameterResidues(piece)) { continue; }
                std::vector<unsigned> map(result.parameterCount);
                std::iota(map.begin(), map.end(), piece.schema->dimensions);
                auto expanded = context->system.remap(piece.system.dimensions(), map);
                if (failed(expanded)) { fail("parameter-context remapping failed"); return false; }
                auto restricted = piece.system.intersect(*expanded);
                ++result.cost.pieceJoins;
                if (failed(restricted)) { fail("parameter-context intersection failed"); return false; }
                if (!restricted->isEmpty()) {
                    pieces.push_back({piece.schema, piece.residues, std::move(*restricted)});
                }
            }
        }
        return true;
    }
    ArithmeticRelationKey endpoints(const Piece& piece) const
    {
        const auto& schema = *piece.schema;
        auto begin = piece.residues.begin();
        return {{*schema.sourceSite, schema.sourceEvent, {begin, begin + schema.sourceDimensions}},
                {*schema.targetSite, schema.targetEvent,
                 {begin + schema.sourceDimensions, begin + schema.sourceDimensions + schema.targetDimensions}},
                parameterResidues(piece)};
    }
    bool buildRelations()
    {
        for (const auto& piece : pieces) {
            const auto& schema = *piece.schema;
            if (schema.kind == PrimitiveKind::Occurrences) {
                if (!schema.sourceSite || schema.targetSite || *schema.sourceSite >= program.sites.size()) {
                    fail("invalid occurrence primitive endpoint"); return false;
                }
                const auto d = schema.sourceDimensions;
                std::vector<unsigned> map(d + result.parameterCount);
                std::iota(map.begin(), map.begin() + d, 0);
                std::iota(map.begin() + d, map.end(), 2 * d);
                auto lifted = piece.system.remap(2 * d + result.parameterCount, map);
                SmallVector<LinearRow> equalities;
                for (unsigned i = 0; i < d; ++i) {
                    LinearRow row;
                    row.coefficients.resize(2 * d + result.parameterCount, 0);
                    row.coefficients[i] = 1;
                    row.coefficients[d + i] = -1;
                    row.equality = true;
                    equalities.push_back(std::move(row));
                }
                auto diagonal = Policy::create(2 * d + result.parameterCount, equalities);
                if (failed(lifted) || failed(diagonal)) { fail("event identity construction failed"); return false; }
                auto value = lifted->intersect(*diagonal);
                ++result.cost.pieceJoins;
                if (failed(value)) { fail("event identity restriction failed"); return false; }
                for (auto event : {ArithmeticEvent::Start, ArithmeticEvent::Completion}) {
                    ArithmeticEventKey endpoint{*schema.sourceSite, event,
                                                {piece.residues.begin(), piece.residues.begin() + d}};
                    append(identity, {endpoint, endpoint, parameterResidues(piece)}, *value);
                }
                continue;
            }
            if (schema.kind != PrimitiveKind::Order && schema.kind != PrimitiveKind::Native &&
                schema.kind != PrimitiveKind::Prerequisites) { continue; }
            if (!schema.sourceSite || !schema.targetSite || *schema.sourceSite >= program.sites.size() ||
                *schema.targetSite >= program.sites.size()) {
                fail("arithmetic relation is missing valid endpoint identities"); return false;
            }
            auto key = endpoints(piece);
            if (schema.kind == PrimitiveKind::Order) {
                if (schema.sourceEvent != ArithmeticEvent::Payload || schema.targetEvent != ArithmeticEvent::Payload) {
                    fail("reference-order primitive requires payload endpoints"); return false;
                }
                append(order, key, piece.system);
            } else if (schema.kind == PrimitiveKind::Native) {
                if (schema.sourceEvent == ArithmeticEvent::Payload || schema.targetEvent == ArithmeticEvent::Payload) {
                    fail("native primitive requires start/completion endpoints"); return false;
                }
                append(result.nativeOrder, key, piece.system);
                if (schema.sourceEvent == ArithmeticEvent::Start && schema.targetEvent == ArithmeticEvent::Start) {
                    append(samePipeOrder, key, piece.system);
                }
            } else {
                if (schema.sourceEvent != ArithmeticEvent::Completion || schema.targetEvent != ArithmeticEvent::Start) {
                    fail("prerequisite primitive requires completion-to-start endpoints"); return false;
                }
                append(result.generators, key, piece.system);
            }
        }
        unite(result.nativeOrder, identity);
        return true;
    }
    std::optional<System> protectionDomain(const Piece& a, const Piece& b, unsigned dimensions)
    {
        if (!protection || a.schema->storageSpace != AddressSpace::ACC || b.schema->storageSpace != AddressSpace::ACC ||
            !a.schema->sourceSite || !b.schema->sourceSite || *a.schema->sourceSite >= program.sites.size() ||
            *b.schema->sourceSite >= program.sites.size()) { return std::nullopt; }
        const auto& left = program.sites[*a.schema->sourceSite];
        const auto& right = program.sites[*b.schema->sourceSite];
        auto x = protection->facts.find(left.phase), y = protection->facts.find(right.phase);
        if (!left.phase || !right.phase || x == protection->facts.end() || y == protection->facts.end() ||
            x->second.scope != y->second.scope || !hardwareProtectsConflict(
                static_cast<uint32_t>(left.phase->kPipeValue), x->second.group,
                static_cast<uint32_t>(right.phase->kPipeValue), y->second.group)) { return std::nullopt; }
        SmallVector<LinearRow> rows;
        // A scoped chain can be used only within the same visit of its scope
        // and every enclosing represented loop. The remaining inner indices
        // may differ. Equal residues reduce original-coordinate equality to
        // quotient-coordinate equality, in both supported arithmetic domains.
        if (auto* scope = x->second.scope) {
            for (auto fixed : left.fixedCoordinates) {
                auto loop = fixed.loop;
                const bool outsideScope = loop.getOperation() != scope && !loop->isProperAncestor(scope);
                if (outsideScope) { continue; }
                auto match = llvm::find_if(right.fixedCoordinates, [&](const FixedLoopCoordinate& other) {
                    return other.loop == loop;
                });
                const bool differentVisit = match == right.fixedCoordinates.end() ||
                    match->induction != fixed.induction;
                if (differentVisit) {
                    return std::nullopt;
                }
            }
            for (unsigned i = 0; i < left.loops.size(); ++i) {
                auto loop = left.loops[i];
                if (loop.getOperation() != scope && !loop->isProperAncestor(scope)) { continue; }
                auto match = llvm::find(right.loops, loop);
                if (match == right.loops.end()) { return std::nullopt; }
                const auto j = static_cast<unsigned>(match - right.loops.begin());
                if (a.residues[i] != b.residues[j]) { return std::nullopt; }
                LinearRow row;
                row.coefficients.assign(dimensions, 0);
                row.coefficients[i] = 1;
                row.coefficients[a.schema->sourceDimensions + j] = -1;
                row.constant = 0;
                row.equality = true;
                rows.push_back(std::move(row));
            }
        }
        auto domain = Policy::create(dimensions, rows);
        if (failed(domain)) { fail("hardware protection scope projection failed"); return std::nullopt; }
        return std::move(*domain);
    }
    bool scalarPair(uint32_t source, uint32_t target) const
    {
        if (source >= program.sites.size() || target >= program.sites.size()) { return false; }
        const auto* a = program.sites[source].phase;
        const auto* b = program.sites[target].phase;
        return a && b && ptoStorageProtection().protectsScalar(static_cast<uint32_t>(a->kPipeValue),
                                                              static_cast<uint32_t>(b->kPipeValue));
    }
    bool buildConflicts()
    {
        Relation conflicts;
        std::vector<const Piece*> accesses;
        for (const auto& piece : pieces) {
            if (piece.schema->kind == PrimitiveKind::Reads || piece.schema->kind == PrimitiveKind::Writes) {
                accesses.push_back(&piece);
            }
        }
        for (const auto* first : accesses) {
            const auto& a = *first;
            for (const auto* second : accesses) {
                const auto& b = *second;
                if ((a.schema->kind == PrimitiveKind::Reads && b.schema->kind == PrimitiveKind::Reads) ||
                    a.schema->storageSpace != b.schema->storageSpace ||
                    a.schema->storageBase != b.schema->storageBase ||
                    parameterResidues(a) != parameterResidues(b)) { continue; }
                if (!a.schema->sourceSite || !b.schema->sourceSite || !a.schema->storageSpace ||
                    a.schema->targetDimensions || b.schema->targetDimensions ||
                    *a.schema->sourceSite >= program.sites.size() || *b.schema->sourceSite >= program.sites.size()) {
                    fail("access primitive has unsupported storage/endpoint schema"); return false;
                }
                if (scalarPair(*a.schema->sourceSite, *b.schema->sourceSite)) { continue; }
                const auto x = a.schema->sourceDimensions, y = b.schema->sourceDimensions;
                if (a.residues[x] != b.residues[y]) { continue; }
                const unsigned joinedDimensions = x + y + 1 + result.parameterCount;
                std::vector<unsigned> am(x + 1 + result.parameterCount), bm(y + 1 + result.parameterCount);
                std::iota(am.begin(), am.begin() + x, 0);
                std::iota(bm.begin(), bm.begin() + y, x);
                am[x] = bm[y] = x + y;
                std::iota(am.begin() + x + 1, am.end(), x + y + 1);
                std::iota(bm.begin() + y + 1, bm.end(), x + y + 1);
                auto left = a.system.remap(joinedDimensions, am), right = b.system.remap(joinedDimensions, bm);
                if (failed(left) || failed(right)) { fail("physical-byte access join failed"); return false; }
                auto joined = left->intersect(*right);
                ++result.cost.pieceJoins;
                if (failed(joined)) { fail("physical-byte intersection failed"); return false; }
                std::vector<unsigned> keep(x + y + result.parameterCount);
                std::iota(keep.begin(), keep.begin() + x + y, 0);
                std::iota(keep.begin() + x + y, keep.end(), x + y + 1);
                auto projected = Policy::project(*joined, keep);
                ++result.cost.projections;
                if (failed(projected)) { fail("physical-byte witness projection failed"); return false; }
                auto protectedRegion = protectionDomain(a, b, keep.size());
                if (!result.error.empty()) { return false; }
                if (protectedRegion) {
                    const std::vector<System> protectedPieces{std::move(*protectedRegion)};
                    projected = Policy::subtract(keep.size(), *projected, protectedPieces);
                    ++result.cost.differences;
                    if (failed(projected)) { fail("hardware protection subtraction failed"); return false; }
                }
                ArithmeticRelationKey key{
                    {*a.schema->sourceSite, ArithmeticEvent::Payload, {a.residues.begin(), a.residues.begin() + x}},
                    {*b.schema->sourceSite, ArithmeticEvent::Payload, {b.residues.begin(), b.residues.begin() + y}},
                    parameterResidues(a)};
                for (auto& value : *projected) { append(conflicts, key, std::move(value)); }
            }
        }
        auto forward = intersect(conflicts, order);
        for (const auto& [key, values] : order) {
            if (scalarPair(key.source.site, key.target.site)) { continue; }
            const auto pair = std::minmax(key.source.site, key.target.site);
            if (llvm::any_of(program.uniformConflicts, [&](const auto& conflict) {
                    return conflict.first == pair.first && conflict.second == pair.second;
                })) {
                for (const auto& value : values) { append(forward, key, value); }
            }
        }
        for (const auto& [key, values] : forward) {
            auto endpoints = key;
            endpoints.source.event = ArithmeticEvent::Completion;
            endpoints.target.event = ArithmeticEvent::Start;
            for (const auto& value : values) { append(result.generators, endpoints, value); }
        }
        return result.error.empty();
    }
    Relation intersect(const Relation& a, const Relation& b)
    {
        Relation output;
        for (const auto& [key, left] : a) {
            auto right = b.find(key);
            if (right == b.end()) { continue; }
            for (const auto& x : left) {
                for (const auto& y : right->second) {
                    auto value = x.intersect(y);
                    ++result.cost.pieceJoins;
                    if (failed(value)) { fail("arithmetic relation intersection failed"); return {}; }
                    append(output, key, std::move(*value));
                }
            }
        }
        return output;
    }
    Relation compose(const Relation& a, const Relation& b, const Relation* outputKeys = nullptr)
    {
        ++result.cost.relationCompositions;
        Relation output;
        using Match = std::pair<ArithmeticEventKey, std::vector<uint64_t>>;
        using Entry = typename Relation::value_type;
        std::map<Match, std::vector<const Entry*>> bySource;
        for (const auto& entry : b) { bySource[{entry.first.source, entry.first.parameterResidues}].push_back(&entry); }
        for (const auto& [ak, av] : a) {
            auto found = bySource.find({ak.target, ak.parameterResidues});
            if (found == bySource.end()) { continue; }
            for (const auto* entry : found->second) {
                const auto& [bk, bv] = *entry;
                ArithmeticRelationKey key{ak.source, bk.target, ak.parameterResidues};
                if (outputKeys && !outputKeys->count(key)) { continue; }
                const unsigned x = ak.source.residues.size(), z = ak.target.residues.size();
                const unsigned y = bk.target.residues.size(), p = result.parameterCount;
                const unsigned joinedDimensions = x + z + y + p;
                std::vector<unsigned> am(x + z + p), bm(z + y + p), keep(x + y + p);
                std::iota(am.begin(), am.begin() + x + z, 0);
                std::iota(am.begin() + x + z, am.end(), x + z + y);
                std::iota(bm.begin(), bm.begin() + z + y, x);
                std::iota(bm.begin() + z + y, bm.end(), x + z + y);
                std::iota(keep.begin(), keep.begin() + x, 0);
                std::iota(keep.begin() + x, keep.end(), x + z);
                for (const auto& left : av) {
                    auto expandedLeft = left.remap(joinedDimensions, am);
                    if (failed(expandedLeft)) { fail("left composition remapping failed"); return {}; }
                    for (const auto& right : bv) {
                        auto expandedRight = right.remap(joinedDimensions, bm);
                        if (failed(expandedRight)) { fail("right composition remapping failed"); return {}; }
                        auto joined = expandedLeft->intersect(*expandedRight);
                        ++result.cost.pieceJoins;
                        if (failed(joined)) { fail("arithmetic composition intersection failed"); return {}; }
                        auto value = Policy::project(*joined, keep);
                        ++result.cost.projections;
                        if (failed(value)) { fail("arithmetic composition projection failed"); return {}; }
                        for (auto& piece : *value) { append(output, key, std::move(piece)); }
                    }
                }
            }
        }
        return output;
    }
    Relation subtract(const Relation& a, const Relation& b)
    {
        ++result.cost.differences;
        Relation output;
        for (const auto& [key, pieces] : a) {
            auto found = b.find(key);
            if (found == b.end()) { output.emplace(key, pieces); continue; }
            auto difference = Policy::subtract(dimensions(key), pieces, found->second);
            if (failed(difference)) { fail("exact arithmetic relation difference failed"); return {}; }
            for (auto& piece : *difference) { append(output, key, std::move(piece)); }
        }
        return output;
    }
};
template<class Policy>
auto analyze(const ArithmeticProgram& program, const StructuredProtection* protection = nullptr)
{
    Analysis<Policy> analysis(program, protection);
    analysis.run();
    if (!analysis.result.error.empty()) {
        analysis.result.generators.clear();
        analysis.result.nativeOrder.clear();
        analysis.result.requiredOrder.clear();
        analysis.result.minimumDemands.clear();
        analysis.result.exactMinimum = false;
        analysis.result.adjacentLocalDemands = false;
    }
    return std::move(analysis.result);
}
} // namespace
struct GeneralArithmeticGeneratorStage::State {
    std::unique_ptr<Analysis<IntegerPolicy>> engine;
    const ArithmeticProgram* program = nullptr;
    GeneralArithmeticDemandAnalysis unavailable;
    std::optional<GeneralArithmeticDemandAnalysis> generators;
    std::shared_ptr<GeneralArithmeticDemandAnalysis> reduced;
    std::vector<ArithmeticOccurrenceDomain> domains;
    State() { unavailable.error = "arithmetic generator stage is unavailable or already consumed"; }
};
GeneralArithmeticGeneratorStage::GeneralArithmeticGeneratorStage() : state(std::make_unique<State>()) {}
GeneralArithmeticGeneratorStage::~GeneralArithmeticGeneratorStage() = default;
GeneralArithmeticGeneratorStage::GeneralArithmeticGeneratorStage(GeneralArithmeticGeneratorStage&&) noexcept = default;
GeneralArithmeticGeneratorStage& GeneralArithmeticGeneratorStage::operator=(
    GeneralArithmeticGeneratorStage&&) noexcept = default;
const GeneralArithmeticDemandAnalysis& GeneralArithmeticGeneratorStage::analysis() const
{
    if (!state) {
        static const auto missing = [] {
            GeneralArithmeticDemandAnalysis result;
            result.error = "arithmetic generator stage is already consumed";
            return result;
        }();
        return missing;
    }
    if (state->generators) { return *state->generators; }
    return state->engine ? state->engine->result : state->unavailable;
}
bool GeneralArithmeticGeneratorStage::belongsTo(const ArithmeticProgram& program) const
{
    return state && state->program == &program;
}
const std::vector<ArithmeticOccurrenceDomain>& GeneralArithmeticGeneratorStage::occurrences() const
{
    static const std::vector<ArithmeticOccurrenceDomain> empty;
    return state ? state->domains : empty;
}
GeneralArithmeticGeneratorStage analyzeGeneralArithmeticGenerators(
    const ArithmeticProgram& program, const StructuredProtection* protection)
{
    GeneralArithmeticGeneratorStage stage;
    stage.state->program = &program;
    stage.state->engine = std::make_unique<Analysis<IntegerPolicy>>(program, protection);
    stage.state->engine->run(true);
    if (stage.state->engine->result.error.empty()) {
        stage.state->domains = stage.state->engine->occurrenceDomains();
    } else {
        stage.state->engine->result.generators.clear();
        stage.state->engine->result.nativeOrder.clear();
    }
    stage.state->generators = std::move(stage.state->engine->result);
    return stage;
}
GeneralArithmeticDemandAnalysis completeGeneralArithmeticDemands(GeneralArithmeticGeneratorStage&& stage)
{
    auto state = std::move(stage.state);
    if (state && state->reduced) { return *state->reduced; }
    if (!state || !state->engine) {
        GeneralArithmeticDemandAnalysis result;
        result.error = "arithmetic generator stage is unavailable or already consumed";
        return result;
    }
    if (state->generators) { state->engine->result = std::move(*state->generators); }
    state->engine->complete();
    auto result = std::move(state->engine->result);
    if (!result.error.empty()) {
        result.generators.clear(); result.nativeOrder.clear();
        result.requiredOrder.clear(); result.minimumDemands.clear();
        result.exactMinimum = false; result.adjacentLocalDemands = false;
    }
    return result;
}
std::shared_ptr<GeneralArithmeticDemandAnalysis> reduceGeneralArithmeticDemands(
    GeneralArithmeticGeneratorStage& stage)
{
    if (!stage.state) { return {}; }
    auto& state = *stage.state;
    if (state.reduced) { return state.reduced; }
    // Transfer only the engine to the consuming adapter. Preserve the original
    // generator snapshot and domains for future periodic export requests.
    GeneralArithmeticGeneratorStage completion;
    completion.state->engine = std::move(state.engine);
    if (completion.state->engine) { completion.state->engine->result = *state.generators; }
    state.reduced = std::make_shared<GeneralArithmeticDemandAnalysis>(
        completeGeneralArithmeticDemands(std::move(completion)));
    return state.reduced;
}
struct DifferenceArithmeticGeneratorStage::State {
    std::unique_ptr<Analysis<DifferencePolicy>> engine;
    const ArithmeticProgram* program = nullptr;
    ArithmeticDemandAnalysis unavailable;
    std::optional<ArithmeticDemandAnalysis> generators;
    std::shared_ptr<ArithmeticDemandAnalysis> reduced;
    std::vector<DifferenceArithmeticOccurrenceDomain> domains;
    State() { unavailable.error = "arithmetic generator stage is unavailable or already consumed"; }
};
DifferenceArithmeticGeneratorStage::DifferenceArithmeticGeneratorStage() : state(std::make_unique<State>()) {}
DifferenceArithmeticGeneratorStage::~DifferenceArithmeticGeneratorStage() = default;
DifferenceArithmeticGeneratorStage::DifferenceArithmeticGeneratorStage(
    DifferenceArithmeticGeneratorStage&&) noexcept = default;
DifferenceArithmeticGeneratorStage& DifferenceArithmeticGeneratorStage::operator=(
    DifferenceArithmeticGeneratorStage&&) noexcept = default;
const ArithmeticDemandAnalysis& DifferenceArithmeticGeneratorStage::analysis() const
{
    if (!state) {
        static const auto missing = [] {
            ArithmeticDemandAnalysis result;
            result.error = "arithmetic generator stage is already consumed";
            return result;
        }();
        return missing;
    }
    if (state->generators) { return *state->generators; }
    return state->engine ? state->engine->result : state->unavailable;
}
bool DifferenceArithmeticGeneratorStage::belongsTo(const ArithmeticProgram& program) const
{
    return state && state->program == &program;
}
const std::vector<DifferenceArithmeticOccurrenceDomain>& DifferenceArithmeticGeneratorStage::occurrences() const
{
    static const std::vector<DifferenceArithmeticOccurrenceDomain> empty;
    return state ? state->domains : empty;
}
DifferenceArithmeticGeneratorStage analyzeDifferenceArithmeticGenerators(
    const ArithmeticProgram& program, const StructuredProtection* protection)
{
    DifferenceArithmeticGeneratorStage stage;
    stage.state->program = &program;
    stage.state->engine = std::make_unique<Analysis<DifferencePolicy>>(program, protection);
    stage.state->engine->run(true);
    if (stage.state->engine->result.error.empty()) {
        stage.state->domains = stage.state->engine->occurrenceDomains();
    } else {
        stage.state->engine->result.generators.clear();
        stage.state->engine->result.nativeOrder.clear();
    }
    stage.state->generators = std::move(stage.state->engine->result);
    return stage;
}
ArithmeticDemandAnalysis completeDifferenceArithmeticDemands(DifferenceArithmeticGeneratorStage&& stage)
{
    auto state = std::move(stage.state);
    if (state && state->reduced) { return *state->reduced; }
    if (!state || !state->engine) {
        ArithmeticDemandAnalysis result;
        result.error = "arithmetic generator stage is unavailable or already consumed";
        return result;
    }
    if (state->generators) { state->engine->result = std::move(*state->generators); }
    state->engine->complete();
    auto result = std::move(state->engine->result);
    if (!result.error.empty()) {
        result.generators.clear(); result.nativeOrder.clear();
        result.requiredOrder.clear(); result.minimumDemands.clear();
        result.exactMinimum = false; result.adjacentLocalDemands = false;
    }
    return result;
}
std::shared_ptr<ArithmeticDemandAnalysis> reduceDifferenceArithmeticDemands(
    DifferenceArithmeticGeneratorStage& stage)
{
    if (!stage.state) { return {}; }
    auto& state = *stage.state;
    if (state.reduced) { return state.reduced; }
    // Transfer only the engine to the consuming adapter. Preserve the original
    // generator snapshot and domains for future periodic export requests.
    DifferenceArithmeticGeneratorStage completion;
    completion.state->engine = std::move(state.engine);
    if (completion.state->engine) { completion.state->engine->result = *state.generators; }
    state.reduced = std::make_shared<ArithmeticDemandAnalysis>(
        completeDifferenceArithmeticDemands(std::move(completion)));
    return state.reduced;
}
ArithmeticDemandAnalysis analyzeArithmeticDemandsWithProtection(const ArithmeticProgram& program,
                                                               const StructuredProtection& protection)
{
    return analyze<DifferencePolicy>(program, &protection);
}
GeneralArithmeticDemandAnalysis analyzeGeneralArithmeticDemandsWithProtection(const ArithmeticProgram& program,
                                                                             const StructuredProtection& protection)
{
    return analyze<IntegerPolicy>(program, &protection);
}
ArithmeticDemandAnalysis analyzeArithmeticDemands(const ArithmeticProgram& program)
{
    return analyze<DifferencePolicy>(program);
}
GeneralArithmeticDemandAnalysis analyzeGeneralArithmeticDemands(const ArithmeticProgram& program)
{
    return analyze<IntegerPolicy>(program);
}
} // namespace mlir::pto::frontiersynch
