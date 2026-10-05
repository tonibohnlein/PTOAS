// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticDemandAnalysis.h"
#include <algorithm>
#include <numeric>
#include <tuple>
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
using System = DifferenceBoundSystem;
using Union = std::vector<System>;
struct ImportedPiece {
    const PrimitiveRelation* schema;
    std::vector<uint64_t> residues; // Original dimension order, then canonical parameter order.
    System system;
};
void append(Union& pieces, System piece)
{
    if (piece.isEmpty()) { return; }
    for (const auto& existing : pieces) {
        if (piece.isSubsetOf(existing)) { return; }
    }
    pieces.erase(std::remove_if(pieces.begin(), pieces.end(),
                               [&](const System& old) { return old.isSubsetOf(piece); }), pieces.end());
    pieces.push_back(std::move(piece));
}
void append(ArithmeticRelation& relation, const ArithmeticRelationKey& key, System piece)
{
    if (!piece.isEmpty()) { append(relation[key], std::move(piece)); }
}
void unite(ArithmeticRelation& target, const ArithmeticRelation& source)
{
    for (const auto& [key, pieces] : source) {
        for (const auto& piece : pieces) { append(target, key, piece); }
    }
}
unsigned dimensions(const ArithmeticRelationKey& key)
{
    return key.source.residues.size() + key.target.residues.size() + key.parameterResidues.size();
}
class Analysis {
public:
    explicit Analysis(const ArithmeticProgram& program) : program(program)
    {
        result.period = program.primitives.period;
        result.parameterCount = program.primitives.parameters.size();
        result.pipeCount = program.primitives.pipeCount;
    }
    ArithmeticDemandAnalysis result;
    void run()
    {
        if (program.extraction.state != RecognitionState::Applicable ||
            program.recognition.state != RecognitionState::Applicable ||
            program.recognition.arithmeticClass != ArithmeticClass::Differences || !result.period) {
            fail("exact difference-bound primitives are required"); return;
        }
        if (!import() || !buildRelations() || !buildConflicts()) { return; }
        auto step = compose(result.generators, result.nativeOrder);
        auto reach = result.nativeOrder;
        for (unsigned round = 0; round < result.pipeCount && result.error.empty(); ++round) {
            auto next = compose(reach, step);
            unite(reach, next);
        }
        if (!result.error.empty()) { return; }
        result.requiredOrder = subtract(reach, identity);
        auto covered = compose(result.requiredOrder, result.requiredOrder);
        unite(covered, result.nativeOrder);
        result.minimumDemands = subtract(result.generators, covered);
        if (!result.error.empty()) { return; }
        // A same-pipe demand is adjacent precisely when no strictly intermediate
        // payload exists on that pipe. Remove identities explicitly even if the
        // producer supplied the reflexive native closure.
        auto strictPipeOrder = subtract(samePipeOrder, identity);
        auto nonadjacent = compose(strictPipeOrder, strictPipeOrder);
        ArithmeticRelation localIntervals;
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
    std::vector<ImportedPiece> pieces;
    ArithmeticRelation identity, order, samePipeOrder;
    void fail(const char* message) { if (result.error.empty()) { result.error = message; } }
    std::vector<uint64_t> parameterResidues(const ImportedPiece& piece) const
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
        std::vector<ImportedPiece> raw;
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
            std::vector<DifferenceBoundConstraint> constraints;
            for (const auto& row : normalized.rows) {
                if (row.coefficients.size() != keep.size()) {
                    fail("normalized arithmetic row has inconsistent dimensions"); return false;
                }
                unsigned positive = 0, negative = 0;
                for (unsigned i = 0; i < row.coefficients.size(); ++i) {
                    auto coefficient = row.coefficients[i];
                    if (!coefficient) { continue; }
                    if ((coefficient != 1 && coefficient != -1) ||
                        (coefficient == 1 ? positive != 0 : negative != 0)) {
                        fail("arithmetic row is outside the difference-bound fragment"); return false;
                    }
                    (coefficient == 1 ? positive : negative) = i + 1;
                }
                BoundInteger bound(row.constant);
                constraints.push_back({negative, positive, bound});
                if (row.equality) { constraints.push_back({positive, negative, -bound}); }
            }
            auto dbm = System::create(keep.size(), constraints);
            if (failed(dbm)) { fail("difference-bound primitive construction failed"); return false; }
            auto canonical = dbm->project(keep);
            if (failed(canonical)) { fail("difference-bound parameter reorder failed"); return false; }
            ++result.cost.primitivePieces;
            if (canonical->isEmpty()) { continue; }
            std::vector<uint64_t> residues;
            for (auto column : keep) { residues.push_back(schema.pieces[normalized.piece].residues[column]); }
            raw.push_back({&schema, std::move(residues), std::move(*canonical)});
        }
        // Restrict every primitive to the same admitted parameter context.
        // Distinct residue cases are incompatible executions, never independent
        // existential choices at either endpoint of a join.
        std::vector<const ImportedPiece*> contexts;
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
    ArithmeticRelationKey endpoints(const ImportedPiece& piece) const
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
                std::vector<DifferenceBoundConstraint> equalities;
                for (unsigned i = 0; i < d; ++i) {
                    equalities.push_back({i + 1, d + i + 1, BoundInteger(0)});
                    equalities.push_back({d + i + 1, i + 1, BoundInteger(0)});
                }
                auto diagonal = System::create(2 * d + result.parameterCount, equalities);
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
    bool buildConflicts()
    {
        ArithmeticRelation conflicts;
        std::vector<const ImportedPiece*> accesses;
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
                auto projected = joined->project(keep);
                ++result.cost.projections;
                if (failed(projected)) { fail("physical-byte witness projection failed"); return false; }
                ArithmeticRelationKey key{
                    {*a.schema->sourceSite, ArithmeticEvent::Payload, {a.residues.begin(), a.residues.begin() + x}},
                    {*b.schema->sourceSite, ArithmeticEvent::Payload, {b.residues.begin(), b.residues.begin() + y}},
                    parameterResidues(a)};
                append(conflicts, key, std::move(*projected));
            }
        }
        auto forward = intersect(conflicts, order);
        for (const auto& [key, values] : forward) {
            auto endpoints = key;
            endpoints.source.event = ArithmeticEvent::Completion;
            endpoints.target.event = ArithmeticEvent::Start;
            for (const auto& value : values) { append(result.generators, endpoints, value); }
        }
        return result.error.empty();
    }
    ArithmeticRelation intersect(const ArithmeticRelation& a, const ArithmeticRelation& b)
    {
        ArithmeticRelation output;
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
    ArithmeticRelation compose(const ArithmeticRelation& a, const ArithmeticRelation& b)
    {
        ++result.cost.relationCompositions;
        ArithmeticRelation output;
        using Match = std::pair<ArithmeticEventKey, std::vector<uint64_t>>;
        using Entry = ArithmeticRelation::value_type;
        std::map<Match, std::vector<const Entry*>> bySource;
        for (const auto& entry : b) { bySource[{entry.first.source, entry.first.parameterResidues}].push_back(&entry); }
        for (const auto& [ak, av] : a) {
            auto found = bySource.find({ak.target, ak.parameterResidues});
            if (found == bySource.end()) { continue; }
            for (const auto* entry : found->second) {
                const auto& [bk, bv] = *entry;
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
                ArithmeticRelationKey key{ak.source, bk.target, ak.parameterResidues};
                for (const auto& left : av) {
                    auto expandedLeft = left.remap(joinedDimensions, am);
                    if (failed(expandedLeft)) { fail("left composition remapping failed"); return {}; }
                    for (const auto& right : bv) {
                        auto expandedRight = right.remap(joinedDimensions, bm);
                        if (failed(expandedRight)) { fail("right composition remapping failed"); return {}; }
                        auto joined = expandedLeft->intersect(*expandedRight);
                        ++result.cost.pieceJoins;
                        if (failed(joined)) { fail("arithmetic composition intersection failed"); return {}; }
                        auto value = joined->project(keep);
                        ++result.cost.projections;
                        if (failed(value)) { fail("arithmetic composition projection failed"); return {}; }
                        append(output, key, std::move(*value));
                    }
                }
            }
        }
        return output;
    }
    ArithmeticRelation subtract(const ArithmeticRelation& a, const ArithmeticRelation& b)
    {
        ++result.cost.differences;
        ArithmeticRelation output;
        for (const auto& [key, pieces] : a) {
            auto found = b.find(key);
            if (found == b.end()) { output.emplace(key, pieces); continue; }
            auto difference = subtractDifferenceBoundUnions(dimensions(key), pieces, found->second);
            if (failed(difference)) { fail("exact arithmetic relation difference failed"); return {}; }
            for (auto& piece : *difference) { append(output, key, std::move(piece)); }
        }
        return output;
    }
};
} // namespace
ArithmeticDemandAnalysis analyzeArithmeticDemands(const ArithmeticProgram& program)
{
    Analysis analysis(program);
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
} // namespace mlir::pto::frontiersynch
