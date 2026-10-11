// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "ArithmeticImportedPrimitives.h"
#include "IntegerProjectionCache.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <iterator>
#include <map>
#include <tuple>
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
    Builder(const ArithmeticProgram& program, llvm::ArrayRef<uint32_t> pipes,
            llvm::ArrayRef<Piece> imported) : Builder(program, pipes)
    {
        pieces = imported;
    }
    ArithmeticStorageSelectors result;
    LogicalResult run();
    LogicalResult import(bool occurrencesOnly = false);
    IntegerProjectionCache numericalQueries;
    std::vector<Piece> ownedPieces;
    llvm::ArrayRef<Piece> pieces;
private:
    void indexOrders();
    void indexCandidates();
    std::vector<std::size_t> candidates(const Piece& candidate, ArithmeticBoundaryKind kind) const;
    const ArithmeticProgram& program;
    llvm::ArrayRef<uint32_t> pipes;
    using OrderKey = std::tuple<std::size_t, std::size_t, unsigned, unsigned, std::vector<uint64_t>>;
    std::map<OrderKey, std::vector<std::size_t>> orderIndex;
    using CandidateKey = std::pair<PrimitiveKind, std::vector<uint64_t>>;
    std::map<CandidateKey, std::vector<std::size_t>> candidateIndex;
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
    const auto liftedDimensions = normalized.machineSystem ? normalized.machineSystem->dimensions() :
        raw.system.getNumInputs();
    auto system = normalized.machineSystem ? FailureOr<IntegerSystem>(*normalized.machineSystem) :
        rows(liftedDimensions, normalized.rows);
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
LogicalResult Builder::import(bool occurrencesOnly)
{
    std::vector<Piece> raw;
    for (const auto& normalized : program.recognition.pieces) {
        if (normalized.empty) { continue; }
        if (normalized.relation >= program.primitives.relations.size()) { return failure(); }
        const auto kind = program.primitives.relations[normalized.relation].kind;
        if (occurrencesOnly && kind != PrimitiveKind::Occurrences && kind != PrimitiveKind::Context) { continue; }
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
    // Context pieces are immutable now. Preserve their original order while
    // avoiding an all-raw-piece scan for every noncontext primitive.
    std::map<std::vector<uint64_t>, std::vector<std::size_t>> contexts;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (raw[i].schema->kind == PrimitiveKind::Context) { contexts[parameters(raw[i]).vec()].push_back(i); }
    }
    for (const auto& piece : raw) {
        if (piece.schema->kind == PrimitiveKind::Context) { continue; }
        auto matching = contexts.find(parameters(piece).vec());
        if (matching == contexts.end()) { continue; }
        for (auto index : matching->second) {
            const auto& context = raw[index];
            if (context.schema->kind != PrimitiveKind::Context || parameters(piece) != parameters(context)) {
                continue;
            }
            std::vector<unsigned> map(result.parameterCount);
            std::iota(map.begin(), map.end(), piece.schema->dimensions);
            auto lifted = context.system.remap(piece.system.dimensions(), map);
            if (failed(lifted)) { return failure(); }
            auto restricted = piece.system.intersect(*lifted);
            if (failed(restricted)) { return failure(); }
            if (!restricted->isEmpty()) {
                ownedPieces.push_back({piece.schema, piece.residues, std::move(*restricted)});
            }
        }
    }
    pieces = ownedPieces;
    return success();
}
void Builder::indexOrders()
{
    // Every indexed piece remains immutable throughout synchronous construction.
    // Keep each matching order list in its original iteration order.
    orderIndex.clear();
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        const auto& order = pieces[i];
        const auto& schema = *order.schema;
        if (schema.kind != PrimitiveKind::Order) { continue; }
        orderIndex[{*schema.sourceSite, *schema.targetSite, schema.sourceDimensions,
                    schema.targetDimensions, order.residues}].push_back(i);
    }
}
// Only occurrence and storage pieces can block an endpoint. Parameters are
// shared entry bindings, so pieces with different parameter residues cannot
// meet. Keep the original ordering, including writes from every pipe for readers.
void Builder::indexCandidates()
{
    candidateIndex.clear();
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        const auto& piece = pieces[i];
        if (piece.schema->kind != PrimitiveKind::Occurrences && !storage(piece)) { continue; }
        candidateIndex[{piece.schema->kind, parameters(piece).vec()}].push_back(i);
    }
}
std::vector<std::size_t> Builder::candidates(const Piece& candidate, ArithmeticBoundaryKind kind) const
{
    const auto residues = parameters(candidate).vec();
    auto matching = candidateIndex.find({candidate.schema->kind, residues});
    std::vector<std::size_t> result;
    if (matching != candidateIndex.end()) { result = matching->second; }
    if (!reader(kind)) { return result; }
    auto writes = candidateIndex.find({PrimitiveKind::Writes, residues});
    if (writes == candidateIndex.end()) { return result; }
    std::vector<std::size_t> merged;
    std::merge(result.begin(), result.end(), writes->second.begin(), writes->second.end(),
               std::back_inserter(merged));
    return merged;
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
    auto left = numericalQueries.remap(candidate.system, dimensions, am);
    auto right = numericalQueries.remap(other.system, dimensions, bm);
    if (failed(left) || failed(right)) { return failure(); }
    auto both = numericalQueries.intersect(**left, **right);
    if (failed(both)) { return failure(); }
    if (numericalQueries.isEmpty(**both)) { return resultPieces; }
    auto add = [&](const IntegerSystem& order) -> LogicalResult {
        auto joined = numericalQueries.intersect(**both, order);
        if (failed(joined)) { return failure(); }
        auto projected = numericalQueries.project(**joined, keep);
        if (failed(projected)) { return failure(); }
        for (auto& piece : *projected) {
            if (!numericalQueries.isEmpty(piece)) { resultPieces.push_back(piece); }
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
    const auto& source = earlier ? other : candidate;
    const auto& target = earlier ? candidate : other;
    const unsigned sourceDepth = source.schema->sourceDimensions;
    const unsigned targetDepth = target.schema->sourceDimensions;
    std::vector<uint64_t> orderResidues(source.residues.begin(), source.residues.begin() + sourceDepth);
    orderResidues.insert(orderResidues.end(), target.residues.begin(), target.residues.begin() + targetDepth);
    const auto parameterResidues = parameters(candidate);
    orderResidues.insert(orderResidues.end(), parameterResidues.begin(), parameterResidues.end());
    auto matching = orderIndex.find({*source.schema->sourceSite, *target.schema->sourceSite,
                                    sourceDepth, targetDepth, std::move(orderResidues)});
    if (matching == orderIndex.end()) { return resultPieces; }
    for (auto index : matching->second) {
        const auto& order = pieces[index];
        const auto& schema = *order.schema;
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
        auto lifted = numericalQueries.remap(order.system, dimensions, map);
        if (failed(lifted) || failed(add(**lifted))) { return failure(); }
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
    for (auto index : candidates(candidate, kind)) {
        const auto& other = pieces[index];
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
        auto witnesses = numericalQueries.witnesses(*ordered, inputs,
            llvm::ArrayRef<uint64_t>(candidate.residues).take_front(depth));
        if (failed(witnesses)) { return failure(); }
        for (const auto& witness : *witnesses) {
            std::vector<uint64_t> inputResidues;
            if (storage(candidate)) { inputResidues.push_back(candidate.residues[depth]); }
            target->selector.pieces.push_back({witness.domain, std::move(inputResidues),
                parameters(candidate).vec(), *schema.sourceSite, witness.outputs});
        }
    }
    return success();
}
LogicalResult Builder::support()
{
    // A touched byte either has a writer, or has a first reader on some
    // pipe without a writer. These already constructed selector domains
    // therefore cover exactly the access support, including empty executions.
    for (const auto& boundary : result.boundaries) {
        if (boundary.kind != ArithmeticBoundaryKind::FirstWriter &&
            boundary.kind != ArithmeticBoundaryKind::FirstReaderBeforeWrite) { continue; }
        if (!boundary.storageSpace || boundary.selector.inputDimensions != 1) { return failure(); }
        for (const auto& piece : boundary.selector.pieces) {
            if (piece.inputResidues.size() != 1 || piece.parameterResidues.size() != result.parameterCount ||
                piece.domain.dimensions() != static_cast<std::size_t>(result.parameterCount) + 1) {
                return failure();
            }
            result.support.push_back({*boundary.storageSpace, boundary.storageBase,
                piece.inputResidues.front(), piece.parameterResidues, piece.domain});
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
    indexOrders();
    indexCandidates();
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
    return support();
}
} // namespace
FailureOr<ImportedArithmeticPrimitives> importOwnedArithmeticIntegerPieces(
    const ArithmeticProgram& program, bool occurrencesOnly)
{
    if (program.extraction.state != RecognitionState::Applicable ||
        program.recognition.state != RecognitionState::Applicable || !program.primitives.period) { return failure(); }
    Builder builder(program, {});
    if (failed(builder.import(occurrencesOnly))) { return failure(); }
    return ImportedArithmeticPrimitives(program, occurrencesOnly, std::move(builder.ownedPieces));
}
FailureOr<std::vector<ArithmeticIntegerPiece>> importArithmeticIntegerPieces(
    const ArithmeticProgram& program, bool occurrencesOnly)
{
    auto imported = importOwnedArithmeticIntegerPieces(program, occurrencesOnly);
    if (failed(imported)) { return failure(); }
    return std::move(imported->values);
}
ArithmeticStorageSelectors buildArithmeticStorageSelectorsFromImported(
    const ArithmeticProgram& program, const ImportedArithmeticPrimitives& imported,
    llvm::ArrayRef<uint32_t> sitePipes)
{
    if (!imported.isCompleteFor(program)) {
        ArithmeticStorageSelectors result;
        result.error = "arithmetic storage selectors require a complete import of the unchanged source program";
        return result;
    }
    Builder builder(program, sitePipes, imported.pieces());
    if (failed(builder.run())) {
        builder.result.boundaries.clear();
        builder.result.support.clear();
        builder.result.error = "arithmetic storage boundary construction requires complete valid primitives "
                               "and exact projections";
    }
    builder.result.cost = builder.numericalQueries.cost();
    return std::move(builder.result);
}
ArithmeticStorageSelectors buildArithmeticStorageSelectors(
    const ArithmeticProgram& program, llvm::ArrayRef<uint32_t> sitePipes)
{
    auto imported = importOwnedArithmeticIntegerPieces(program);
    if (failed(imported)) {
        ArithmeticStorageSelectors result;
        result.error = "arithmetic storage boundary construction requires complete valid primitives "
                       "and exact projections";
        return result;
    }
    return buildArithmeticStorageSelectorsFromImported(program, *imported, sitePipes);
}
} // namespace mlir::pto::frontiersynch
