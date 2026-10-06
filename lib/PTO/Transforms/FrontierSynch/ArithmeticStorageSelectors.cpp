// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticStorageSelectors.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>
#include <limits>
namespace mlir::pto::frontiersynch {
namespace {
using Piece = ArithmeticIntegerPiece;
using Systems = std::vector<IntegerSystem>;
class Builder {
public:
    Builder(const ArithmeticProgram& program, llvm::ArrayRef<uint32_t> pipes) : program(program), pipes(pipes)
    {
        result.period = program.primitives.period;
        result.parameterCount = program.primitives.parameters.size();
    }
    ArithmeticStorageSelectors result;
    LogicalResult run();
    LogicalResult import();
    std::vector<Piece> pieces;
private:
    const ArithmeticProgram& program;
    llvm::ArrayRef<uint32_t> pipes;
    LogicalResult support();
    FailureOr<std::vector<Piece>> importPiece(const NormalizedPiece& normalized);
    FailureOr<Systems> blockers(const Piece& candidate, const Piece& other, bool earlier, bool equal);
    FailureOr<Systems> select(const Piece& candidate, ArithmeticBoundaryKind kind);
    LogicalResult append(const Piece& candidate, ArithmeticBoundaryKind kind, llvm::ArrayRef<IntegerSystem> selected);
    llvm::ArrayRef<uint64_t> parameters(const Piece& piece) const
    {
        return llvm::ArrayRef<uint64_t>(piece.residues).take_back(result.parameterCount);
    }
    bool sameStorage(const Piece& a, const Piece& b) const
    {
        return a.schema->storageSpace == b.schema->storageSpace && a.schema->storageBase == b.schema->storageBase;
    }
};
bool storage(const Piece& piece)
{
    return piece.schema->kind == PrimitiveKind::Reads || piece.schema->kind == PrimitiveKind::Writes;
}
bool first(ArithmeticBoundaryKind kind)
{
    return kind == ArithmeticBoundaryKind::FirstWriter || kind == ArithmeticBoundaryKind::FirstReaderBeforeWrite ||
           kind == ArithmeticBoundaryKind::FirstPayload || kind == ArithmeticBoundaryKind::FirstSite;
}
bool native(ArithmeticBoundaryKind kind)
{
    return kind == ArithmeticBoundaryKind::FirstPayload || kind == ArithmeticBoundaryKind::LastPayload ||
           kind == ArithmeticBoundaryKind::FirstSite || kind == ArithmeticBoundaryKind::LastSite;
}
bool reader(ArithmeticBoundaryKind kind)
{
    return kind == ArithmeticBoundaryKind::FirstReaderBeforeWrite ||
           kind == ArithmeticBoundaryKind::LastReaderAfterWrite;
}
FailureOr<IntegerSystem> rows(unsigned dimensions, llvm::ArrayRef<LinearRow> input)
{
    std::vector<IntegerConstraint> constraints;
    for (const auto& row : input) {
        if (row.coefficients.size() != dimensions) { return failure(); }
        IntegerConstraint constraint;
        constraint.bound = BoundInteger(row.constant);
        for (int64_t coefficient : row.coefficients) { constraint.coefficients.push_back(-BoundInteger(coefficient)); }
        constraints.push_back(constraint);
        if (row.equality) {
            constraint.bound = -constraint.bound;
            for (auto& coefficient : constraint.coefficients) { coefficient = -coefficient; }
            constraints.push_back(std::move(constraint));
        }
    }
    return IntegerSystem::create(dimensions, constraints);
}
FailureOr<std::vector<Piece>> Builder::importPiece(const NormalizedPiece& normalized)
{
    if (normalized.relation >= program.primitives.relations.size()) { return failure(); }
    const auto& schema = program.primitives.relations[normalized.relation];
    if (normalized.piece >= schema.pieces.size() ||
        schema.coordinates.size() > std::numeric_limits<unsigned>::max() ||
        schema.coordinates.size() != static_cast<std::size_t>(schema.dimensions) + result.parameterCount ||
        schema.sourceDimensions > schema.dimensions ||
        schema.targetDimensions > schema.dimensions - schema.sourceDimensions) { return failure(); }
    const auto& raw = schema.pieces[normalized.piece];
    if (raw.residues.size() != schema.coordinates.size()) { return failure(); }
    std::vector<unsigned> permutation(schema.coordinates.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    for (unsigned p = 0; p < result.parameterCount; ++p) {
        unsigned matches = 0;
        for (unsigned i = schema.dimensions; i < schema.coordinates.size(); ++i) {
            if (schema.coordinates[i].kind == CoordinateKind::Parameter &&
                schema.coordinates[i].name == program.primitives.parameters[p]) {
                permutation[i] = schema.dimensions + p;
                ++matches;
            }
        }
        if (matches != 1) { return failure(); }
    }
    const auto liftedDimensions = raw.system.getNumInputs();
    auto system = rows(liftedDimensions, normalized.rows);
    if (failed(system)) { return failure(); }
    std::vector<uint64_t> residues(raw.residues.size());
    for (unsigned i = 0; i < raw.residues.size(); ++i) {
        if (raw.residues[i] >= result.period) { return failure(); }
        residues[permutation[i]] = raw.residues[i];
    }
    std::vector<Piece> imported;
    if (liftedDimensions == schema.coordinates.size()) {
        system = system->remap(schema.coordinates.size(), permutation);
        if (failed(system)) { return failure(); }
        imported.push_back({&schema, std::move(residues), std::move(*system)});
    } else {
        std::vector<unsigned> keep(schema.coordinates.size());
        for (unsigned i = 0; i < keep.size(); ++i) { keep[permutation[i]] = i; }
        auto projected = system->project(keep);
        if (failed(projected)) { return failure(); }
        for (auto& selected : *projected) {
            imported.push_back({&schema, residues, std::move(selected)});
        }
    }
    return imported;
}
LogicalResult Builder::import()
{
    std::vector<Piece> raw;
    for (const auto& normalized : program.recognition.pieces) {
        if (normalized.empty) { continue; }
        auto imported = importPiece(normalized);
        if (failed(imported)) { return failure(); }
        for (auto& entry : *imported) {
            auto* piece = &entry;
            const auto& schema = *piece->schema;
            if (schema.kind == PrimitiveKind::Context) {
                if (schema.dimensions) { return failure(); }
            } else {
                if (!schema.sourceSite || *schema.sourceSite >= program.sites.size() ||
                    schema.dimensions - schema.sourceDimensions - schema.targetDimensions !=
                        (storage(*piece) ? 1U : 0U)) {
                    return failure();
                }
                if (schema.kind == PrimitiveKind::Order || schema.kind == PrimitiveKind::Native ||
                    schema.kind == PrimitiveKind::Prerequisites) {
                    if (!schema.targetSite || *schema.targetSite >= program.sites.size()) { return failure(); }
                } else if (schema.targetSite || schema.targetDimensions) { return failure(); }
                if (storage(*piece) && !schema.storageSpace) { return failure(); }
            }
            raw.push_back(std::move(*piece));
        }
    }
    for (const auto& piece : raw) {
        if (piece.schema->kind == PrimitiveKind::Context) { continue; }
        for (const auto& context : raw) {
            if (context.schema->kind != PrimitiveKind::Context || parameters(piece) != parameters(context)) {
                continue;
            }
            std::vector<unsigned> map(result.parameterCount);
            std::iota(map.begin(), map.end(), piece.schema->dimensions);
            auto lifted = context.system.remap(piece.system.dimensions(), map);
            if (failed(lifted)) { return failure(); }
            auto restricted = piece.system.intersect(*lifted);
            if (failed(restricted)) { return failure(); }
            if (!restricted->isEmpty()) { pieces.push_back({piece.schema, piece.residues, std::move(*restricted)}); }
        }
    }
    return success();
}
// Combined columns: candidate coordinates, other coordinates, shared byte (if
// present), shared parameters. The order join never gives independent copies
// of the region-entry parameters or the queried physical byte.
FailureOr<Systems> Builder::blockers(const Piece& candidate, const Piece& other, bool earlier, bool equal)
{
    Systems resultPieces;
    const auto& a = *candidate.schema;
    const auto& b = *other.schema;
    const unsigned da = a.sourceDimensions, db = b.sourceDimensions;
    const unsigned tail = (storage(candidate) ? 1 : 0) + result.parameterCount;
    if (parameters(candidate) != parameters(other) || storage(candidate) != storage(other)) { return resultPieces; }
    if (storage(candidate) && (!sameStorage(candidate, other) || candidate.residues[da] != other.residues[db])) {
        return resultPieces;
    }
    const unsigned dimensions = da + db + tail;
    std::vector<unsigned> am(da + tail), bm(db + tail), keep;
    std::iota(am.begin(), am.begin() + da, 0);
    std::iota(am.begin() + da, am.end(), da + db);
    std::iota(bm.begin(), bm.end(), da);
    keep = am;
    auto left = candidate.system.remap(dimensions, am), right = other.system.remap(dimensions, bm);
    if (failed(left) || failed(right)) { return failure(); }
    auto both = left->intersect(*right);
    if (failed(both)) { return failure(); }
    if (both->isEmpty()) { return resultPieces; }
    auto add = [&](const IntegerSystem& order) -> LogicalResult {
        auto joined = both->intersect(order);
        if (failed(joined)) { return failure(); }
        auto projected = joined->project(keep);
        if (failed(projected)) { return failure(); }
        for (auto& piece : *projected) {
            if (!piece.isEmpty()) { resultPieces.push_back(std::move(piece)); }
        }
        return success();
    };
    if (equal && a.sourceSite == b.sourceSite && da == db &&
        llvm::ArrayRef<uint64_t>(candidate.residues).take_front(da) ==
            llvm::ArrayRef<uint64_t>(other.residues).take_front(db)) {
        SmallVector<LinearRow> diagonal;
        for (unsigned i = 0; i < da; ++i) {
            LinearRow row;
            row.coefficients.resize(dimensions, 0);
            row.coefficients[i] = 1;
            row.coefficients[da + i] = -1;
            row.equality = true;
            diagonal.push_back(std::move(row));
        }
        auto identity = rows(dimensions, diagonal);
        if (failed(identity) || failed(add(*identity))) { return failure(); }
    }
    for (const auto& order : pieces) {
        const auto& schema = *order.schema;
        const auto& source = earlier ? other : candidate;
        const auto& target = earlier ? candidate : other;
        if (schema.kind != PrimitiveKind::Order || schema.sourceSite != source.schema->sourceSite ||
            schema.targetSite != target.schema->sourceSite || parameters(order) != parameters(candidate) ||
            schema.sourceDimensions != source.schema->sourceDimensions ||
            schema.targetDimensions != target.schema->sourceDimensions) { continue; }
        const unsigned ds = schema.sourceDimensions, dt = schema.targetDimensions;
        if (llvm::ArrayRef<uint64_t>(order.residues).take_front(ds) !=
                llvm::ArrayRef<uint64_t>(source.residues).take_front(ds) ||
            llvm::ArrayRef<uint64_t>(order.residues).slice(ds, dt) !=
                llvm::ArrayRef<uint64_t>(target.residues).take_front(dt)) { continue; }
        std::vector<unsigned> map(ds + dt + result.parameterCount);
        std::iota(map.begin(), map.begin() + ds, earlier ? da : 0);
        std::iota(map.begin() + ds, map.begin() + ds + dt, earlier ? 0 : da);
        std::iota(map.begin() + ds + dt, map.end(), da + db + (storage(candidate) ? 1 : 0));
        auto lifted = order.system.remap(dimensions, map);
        if (failed(lifted) || failed(add(*lifted))) { return failure(); }
    }
    return resultPieces;
}
// IntegerSystem::operator== proves semantic equality using projection. These
// systems are already normalized; remove identical stored rows without another
// arithmetic query before constructing the complement arrangement.
bool sameStructure(const IntegerSystem& a, const IntegerSystem& b)
{
    if (a.dimensions() != b.dimensions() || a.constraints().size() != b.constraints().size() ||
        a.congruences().size() != b.congruences().size()) { return false; }
    for (auto [left, right] : llvm::zip(a.constraints(), b.constraints())) {
        if (left.coefficients != right.coefficients || left.bound != right.bound) { return false; }
    }
    for (auto [left, right] : llvm::zip(a.congruences(), b.congruences())) {
        if (left.coefficients != right.coefficients || left.residue != right.residue ||
            left.modulus != right.modulus) { return false; }
    }
    return true;
}
void appendUnique(Systems& systems, IntegerSystem system)
{
    if (!llvm::any_of(systems, [&](const IntegerSystem& present) { return sameStructure(present, system); })) {
        systems.push_back(std::move(system));
    }
}
FailureOr<Systems> Builder::select(const Piece& candidate, ArithmeticBoundaryKind kind)
{
    Systems forbidden;
    const uint32_t pipe = pipes[*candidate.schema->sourceSite];
    for (const auto& other : pieces) {
        bool relevant = other.schema->kind == candidate.schema->kind &&
                        ((!native(kind) && !reader(kind)) || pipes[*other.schema->sourceSite] == pipe);
        if ((kind == ArithmeticBoundaryKind::FirstSite || kind == ArithmeticBoundaryKind::LastSite) &&
            other.schema->sourceSite != candidate.schema->sourceSite) { relevant = false; }
        const bool write = reader(kind) && other.schema->kind == PrimitiveKind::Writes;
        if (!relevant && !write) { continue; }
        auto excluded = blockers(candidate, other, first(kind), write);
        if (failed(excluded)) { return failure(); }
        for (auto& system : *excluded) { appendUnique(forbidden, std::move(system)); }
    }
    return subtractIntegerUnions(candidate.system.dimensions(), {candidate.system}, forbidden);
}
LogicalResult Builder::append(const Piece& candidate, ArithmeticBoundaryKind kind,
                              llvm::ArrayRef<IntegerSystem> selected)
{
    const auto& schema = *candidate.schema;
    std::optional<uint32_t> pipe = native(kind) || reader(kind) ? std::optional<uint32_t>(pipes[*schema.sourceSite]) :
                                                               std::nullopt;
    const bool perSite = kind == ArithmeticBoundaryKind::FirstSite || kind == ArithmeticBoundaryKind::LastSite;
    const std::optional<std::size_t> site = perSite ? schema.sourceSite : std::nullopt;
    ArithmeticBoundarySelector* target = nullptr;
    for (auto& boundary : result.boundaries) {
        if (boundary.kind == kind && boundary.storageSpace == schema.storageSpace &&
            boundary.storageBase == schema.storageBase && boundary.pipe == pipe && boundary.site == site) {
            target = &boundary;
            break;
        }
    }
    if (!target) {
        result.boundaries.push_back({kind, schema.storageSpace, schema.storageBase, pipe, {}, site});
        target = &result.boundaries.back();
        target->selector.inputDimensions = storage(candidate) ? 1 : 0;
        target->selector.parameterCount = result.parameterCount;
        target->selector.outputPipe = pipe.value_or(0);
    }
    const unsigned depth = schema.sourceDimensions, inputs = target->selector.inputDimensions + result.parameterCount;
    std::vector<unsigned> map(candidate.system.dimensions());
    std::iota(map.begin(), map.begin() + depth, inputs);
    std::iota(map.begin() + depth, map.end(), 0);
    for (const auto& system : selected) {
        auto ordered = system.remap(system.dimensions(), map);
        if (failed(ordered)) { return failure(); }
        auto witnesses = buildIntegerTupleWitnesses(*ordered, inputs,
            llvm::ArrayRef<uint64_t>(candidate.residues).take_front(depth));
        if (failed(witnesses)) { return failure(); }
        for (auto& witness : *witnesses) {
            std::vector<uint64_t> inputResidues;
            if (storage(candidate)) { inputResidues.push_back(candidate.residues[depth]); }
            target->selector.pieces.push_back({std::move(witness.domain), std::move(inputResidues),
                parameters(candidate).vec(), *schema.sourceSite, std::move(witness.outputs)});
        }
    }
    return success();
}
LogicalResult Builder::support()
{
    for (const auto& piece : pieces) {
        if (!storage(piece)) { continue; }
        std::vector<unsigned> keep(1 + result.parameterCount);
        std::iota(keep.begin(), keep.end(), piece.schema->sourceDimensions);
        auto projected = piece.system.project(keep);
        if (failed(projected)) { return failure(); }
        for (auto& domain : *projected) {
            if (domain.isEmpty()) { continue; }
            result.support.push_back({*piece.schema->storageSpace, piece.schema->storageBase,
                piece.residues[piece.schema->sourceDimensions], parameters(piece).vec(), std::move(domain)});
        }
    }
    return success();
}
LogicalResult Builder::run()
{
    if (program.extraction.state != RecognitionState::Applicable ||
        program.recognition.state != RecognitionState::Applicable || !result.period ||
        pipes.size() != program.sites.size()) {
        return failure();
    }
    if (failed(import()) || failed(support())) { return failure(); }
    for (const auto& candidate : pieces) {
        SmallVector<ArithmeticBoundaryKind> kinds;
        switch (candidate.schema->kind) {
        case PrimitiveKind::Occurrences:
            kinds = {ArithmeticBoundaryKind::FirstPayload, ArithmeticBoundaryKind::LastPayload,
                     ArithmeticBoundaryKind::FirstSite, ArithmeticBoundaryKind::LastSite}; break;
        case PrimitiveKind::Writes:
            kinds = {ArithmeticBoundaryKind::FirstWriter, ArithmeticBoundaryKind::LastWriter}; break;
        case PrimitiveKind::Reads:
            kinds = {ArithmeticBoundaryKind::FirstReaderBeforeWrite, ArithmeticBoundaryKind::LastReaderAfterWrite};
            break;
        default: continue;
        }
        for (auto kind : kinds) {
            auto selected = select(candidate, kind);
            if (failed(selected) || failed(append(candidate, kind, *selected))) { return failure(); }
        }
    }
    return success();
}
} // namespace
FailureOr<std::vector<ArithmeticIntegerPiece>> importArithmeticIntegerPieces(const ArithmeticProgram& program)
{
    if (program.extraction.state != RecognitionState::Applicable ||
        program.recognition.state != RecognitionState::Applicable || !program.primitives.period) { return failure(); }
    Builder builder(program, {});
    if (failed(builder.import())) { return failure(); }
    return std::move(builder.pieces);
}
ArithmeticStorageSelectors buildArithmeticStorageSelectors(
    const ArithmeticProgram& program, llvm::ArrayRef<uint32_t> sitePipes)
{
    Builder builder(program, sitePipes);
    if (failed(builder.run())) {
        builder.result.boundaries.clear();
        builder.result.support.clear();
        builder.result.error = "arithmetic storage boundary construction requires complete valid primitives "
                               "and exact projections";
    }
    return std::move(builder.result);
}
} // namespace mlir::pto::frontiersynch
