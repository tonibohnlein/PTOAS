// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent occurrence/byte oracle for symbolic region boundary selectors.
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticStorageSelectors.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"
#include <array>
#include <set>
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticByteCover.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/IntegerProjectionCache.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticImportedPrimitives.h"
namespace fs = mlir::pto::frontiersynch;
using namespace mlir;
namespace {
using Kind = fs::PrimitiveKind;
using Boundary = fs::ArithmeticBoundaryKind;
using Integer = fs::BoundInteger;
bool sameImportedPiece(const fs::ArithmeticIntegerPiece& a, const fs::ArithmeticIntegerPiece& b)
{
    if (a.schema != b.schema || a.residues != b.residues || a.system.dimensions() != b.system.dimensions() ||
        a.system.constraints().size() != b.system.constraints().size() ||
        a.system.congruences().size() != b.system.congruences().size()) { return false; }
    for (std::size_t i = 0; i < a.system.constraints().size(); ++i) {
        const auto& x = a.system.constraints()[i];
        const auto& y = b.system.constraints()[i];
        if (x.coefficients != y.coefficients || x.bound != y.bound) { return false; }
    }
    for (std::size_t i = 0; i < a.system.congruences().size(); ++i) {
        const auto& x = a.system.congruences()[i];
        const auto& y = b.system.congruences()[i];
        if (x.coefficients != y.coefficients || x.residue != y.residue || x.modulus != y.modulus) { return false; }
    }
    return true;
}
FailureOr<std::vector<fs::ArithmeticStorageSelectors>> selectorAdapters(
    const fs::ArithmeticProgram& program, llvm::ArrayRef<uint32_t> pipes)
{
    auto imported = fs::importOwnedArithmeticIntegerPieces(program);
    auto incomplete = fs::importOwnedArithmeticIntegerPieces(program, fs::ArithmeticPrimitiveRequest::Occurrences);
    auto requested = fs::importOwnedArithmeticIntegerPieces(program, fs::ArithmeticPrimitiveRequest::Storage);
    if (failed(imported) || failed(incomplete) || failed(requested)) { return failure(); }
    const auto otherProgram = program;
    if (fs::buildArithmeticStorageSelectorsFromImported(otherProgram, *imported, pipes).error.empty() ||
        fs::buildArithmeticStorageSelectorsFromImported(program, *incomplete, pipes).error.empty()) {
        return failure();
    }
    std::vector<fs::ArithmeticIntegerPiece> before(imported->pieces().begin(), imported->pieces().end());
    const auto importCost = imported->constructionCost();
    if (!importCost.remapHits || !importCost.intersectionHits || !importCost.emptinessHits) { return failure(); }
    auto moved = std::move(*imported);
    if (moved.constructionCost().remapRequests != importCost.remapRequests) { return failure(); }
    if (fs::buildArithmeticStorageSelectorsFromImported(program, *imported, pipes).error.empty()) { return failure(); }
    std::vector<fs::ArithmeticStorageSelectors> results;
    results.push_back(fs::buildArithmeticStorageSelectors(program, pipes));
    results.push_back(fs::buildArithmeticStorageSelectorsFromImported(program, *requested, pipes));
    results.push_back(fs::buildArithmeticStorageSelectorsFromImported(program, moved, pipes));
    results.push_back(fs::buildArithmeticStorageSelectorsFromImported(program, moved, pipes));
    if (before.size() != moved.pieces().size()) { return failure(); }
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (!sameImportedPiece(before[i], moved.pieces()[i])) { return failure(); }
    }
    // All results leave the import owner's lifetime here. The independent
    // execution oracle below must still evaluate their owned mathematical data.
    return results;
}
struct Occurrence {
    std::size_t site;
    int64_t i;
    uint32_t pipe;
    std::set<int64_t> reads, writes;
};
class Fixture {
    MLIRContext context;
public:
    fs::ArithmeticProgram program;
    std::vector<uint32_t> pipes{0, 1, 0};
    explicit Fixture(unsigned unusedRelations = 0)
    {
        program.primitives.period = 1;
        program.primitives.pipeCount = 2;
        program.primitives.parameters = {"n", "guard"};
        program.sites.resize(3);
        for (auto kind : {Kind::Native, Kind::Prerequisites}) {
            fs::PrimitiveRelation empty;
            empty.kind = kind;
            empty.coordinates = {{"n", fs::CoordinateKind::Parameter}, {"guard", fs::CoordinateKind::Parameter}};
            program.primitives.relations.push_back(std::move(empty));
        }
        add(Kind::Context, {}, {});
        for (unsigned a = 0; a < 3; ++a) {
            add(Kind::Occurrences, a, {});
            if (a != 1) { add(Kind::Reads, a, {}); }
            if (a != 0) { add(Kind::Writes, a, {}); }
            for (unsigned b = 0; b < 3; ++b) { add(Kind::Order, a, b); }
        }
        // Reduction owns these valid native/prerequisite relations. Storage
        // export has no consumer for their extra existential coordinate.
        const auto order = llvm::find_if(program.primitives.relations, [](const auto& relation) {
            return relation.kind == Kind::Order;
        });
        const auto sourceOrder = *order;
        for (unsigned i = 0; i < unusedRelations; ++i) {
            auto relation = sourceOrder;
            relation.kind = i % 2 ? Kind::Native : Kind::Prerequisites;
            auto& piece = relation.pieces.front();
            SmallVector<AffineExpr> rows(piece.system.getConstraints());
            SmallVector<bool> equalities(piece.system.getEqFlags());
            rows.push_back(getAffineSymbolExpr(2, &context));
            equalities.push_back(true);
            piece.system = IntegerSet::get(relation.dimensions, 3, rows, equalities);
            program.primitives.relations.push_back(std::move(relation));
        }
        program.recognition = fs::recognizeArithmetic(program.primitives, {2, 5, 1, 1});
    }
private:
    void add(Kind kind, std::optional<std::size_t> source, std::optional<std::size_t> target)
    {
        fs::PrimitiveRelation relation;
        relation.kind = kind;
        relation.sourceSite = source;
        relation.targetSite = target;
        relation.sourceDimensions = source ? 1 : 0;
        relation.targetDimensions = target ? 1 : 0;
        const bool access = kind == Kind::Reads || kind == Kind::Writes;
        relation.dimensions = relation.sourceDimensions + relation.targetDimensions + (access ? 1 : 0);
        for (unsigned d = 0; d < relation.dimensions; ++d) {
            relation.coordinates.push_back({"d" + std::to_string(d), access && d + 1 == relation.dimensions ?
                fs::CoordinateKind::Storage : fs::CoordinateKind::Occurrence});
        }
        // Reverse the symbol declaration on access relations. The import must
        // recover the one shared canonical parameter tuple before any join.
        relation.coordinates.push_back({access ? "guard" : "n", fs::CoordinateKind::Parameter});
        relation.coordinates.push_back({access ? "n" : "guard", fs::CoordinateKind::Parameter});
        if (access) { relation.storageSpace = pto::AddressSpace::VEC; }
        relation.pieces.push_back({{}, SmallVector<uint64_t>(relation.coordinates.size(), 0)});
        fs::NormalizedPiece normalized;
        normalized.relation = program.primitives.relations.size();
        const unsigned n = relation.dimensions + (access ? 1 : 0), g = relation.dimensions + (access ? 0 : 1);
        auto row = [&](std::initializer_list<std::pair<unsigned, int64_t>> coefficients, int64_t constant,
                       bool equal = false) {
            fs::LinearRow value;
            value.coefficients.resize(relation.coordinates.size(), 0);
            for (auto [column, weight] : coefficients) { value.coefficients[column] = weight; }
            value.constant = constant;
            value.equality = equal;
            normalized.rows.push_back(std::move(value));
        };
        row({{n, 1}}, 0); row({{g, 1}}, 0); row({{g, -1}}, 1);
        if (source) {
            row({{0, 1}}, 0); row({{n, 1}, {0, -1}}, -1);
            if (*source == 2) { row({{g, 1}}, -1, true); }
        }
        if (target) {
            row({{1, 1}}, 0); row({{n, 1}, {1, -1}}, -1);
            if (*target == 2) { row({{g, 1}}, -1, true); }
            row({{1, 1}, {0, -1}}, *source < *target ? 0 : -1);
        }
        if (access) {
            // Site0 reads [i,i+1], site1 writes i, site2 RMWs byte0.
            if (*source == 0) { row({{1, 1}, {0, -1}}, 0); row({{0, 1}, {1, -1}}, 1); }
            else if (*source == 1) { row({{1, 1}, {0, -1}}, 0, true); }
            else { row({{1, 1}}, 0, true); }
        }
        SmallVector<AffineExpr> expressions;
        SmallVector<bool> equalities;
        for (const auto& constraint : normalized.rows) {
            AffineExpr expression = getAffineConstantExpr(constraint.constant, &context);
            for (unsigned column = 0; column < constraint.coefficients.size(); ++column) {
                AffineExpr variable = column < relation.dimensions ? getAffineDimExpr(column, &context) :
                    getAffineSymbolExpr(column - relation.dimensions, &context);
                expression = expression + variable * constraint.coefficients[column];
            }
            expressions.push_back(expression);
            equalities.push_back(constraint.equality);
        }
        relation.pieces[0].system = IntegerSet::get(relation.dimensions, 2, expressions, equalities);
        program.primitives.relations.push_back(std::move(relation));
    }
};
std::vector<Occurrence> execution(int64_t n, int64_t guard)
{
    std::vector<Occurrence> result;
    for (int64_t i = 0; i < n; ++i) {
        result.push_back({0, i, 0, {i, i + 1}, {}});
        result.push_back({1, i, 1, {}, {i}});
        if (guard) { result.push_back({2, i, 0, {0}, {0}}); }
    }
    return result;
}
bool supportMatches(const fs::ArithmeticStorageSelectors& result, const std::vector<Occurrence>& execution,
                    int64_t byte, llvm::ArrayRef<Integer> parameters)
{
    const bool touched = llvm::any_of(execution, [&](const Occurrence& occurrence) {
        return occurrence.reads.count(byte) || occurrence.writes.count(byte);
    });
    for (auto space : {pto::AddressSpace::VEC, pto::AddressSpace::GM}) {
        fs::GeneralArithmeticEndpointSelector membership;
        membership.inputDimensions = 1;
        membership.parameterCount = parameters.size();
        for (const auto& piece : result.support) {
            if (piece.space != space || piece.base) { continue; }
            membership.pieces.push_back({piece.domain, {piece.byteResidue}, piece.parameterResidues, 0, {}});
        }
        auto selected = fs::evaluateGeneralArithmeticSelector(membership, result.period, {Integer(byte)}, parameters);
        if (failed(selected) || selected->has_value() != (space == pto::AddressSpace::VEC && touched)) { return false; }
    }
    return true;
}
std::optional<fs::ArithmeticSelectedEndpoint> expected(
    const std::vector<Occurrence>& occurrences, const fs::ArithmeticBoundarySelector& boundary, int64_t byte)
{
    bool writeSeen = false;
    const bool reverse = boundary.kind == Boundary::LastWriter || boundary.kind == Boundary::LastReaderAfterWrite ||
                         boundary.kind == Boundary::LastPayload || boundary.kind == Boundary::LastSite;
    for (std::size_t k = 0; k < occurrences.size(); ++k) {
        const auto& occurrence = occurrences[reverse ? occurrences.size() - 1 - k : k];
        const bool writes = occurrence.writes.count(byte);
        writeSeen |= writes;
        if (boundary.pipe && *boundary.pipe != occurrence.pipe) { continue; }
        if (boundary.site && *boundary.site != occurrence.site) { continue; }
        bool match = false;
        switch (boundary.kind) {
        case Boundary::FirstWriter: case Boundary::LastWriter: match = writes; break;
        case Boundary::FirstReaderBeforeWrite: case Boundary::LastReaderAfterWrite:
            match = occurrence.reads.count(byte) && !writeSeen; break;
        default: match = true; break;
        }
        if (match) { return fs::ArithmeticSelectedEndpoint{occurrence.site, {Integer(occurrence.i)}}; }
    }
    return std::nullopt;
}
bool residueCheck(bool splitContextsAndOrders = false)
{
    MLIRContext context;
    fs::ArithmeticProgram program;
    program.primitives.period = 2;
    program.primitives.pipeCount = 1;
    program.primitives.parameters = {"n"};
    program.sites.resize(1);
    for (auto kind : {Kind::Context, Kind::Occurrences, Kind::Reads, Kind::Writes,
                      Kind::Order, Kind::Native, Kind::Prerequisites}) {
        fs::PrimitiveRelation relation;
        relation.kind = kind;
        const bool access = kind == Kind::Reads || kind == Kind::Writes;
        const bool order = kind == Kind::Order;
        const bool empty = kind == Kind::Native || kind == Kind::Prerequisites;
        if (kind != Kind::Context && !empty) { relation.sourceSite = 0; relation.sourceDimensions = 1; }
        if (order) { relation.targetSite = 0; relation.targetDimensions = 1; }
        relation.dimensions = relation.sourceDimensions + relation.targetDimensions + (access ? 1 : 0);
        for (unsigned d = 0; d < relation.dimensions; ++d) {
            relation.coordinates.push_back({"d" + std::to_string(d), access && d == 1 ?
                fs::CoordinateKind::Storage : fs::CoordinateKind::Occurrence});
        }
        relation.coordinates.push_back({"n", fs::CoordinateKind::Parameter});
        if (access) { relation.storageSpace = pto::AddressSpace::VEC; }
        for (unsigned mask = 0; !empty && mask < (1U << relation.coordinates.size()); ++mask) {
            SmallVector<AffineExpr> variables;
            SmallVector<uint64_t> residues;
            for (unsigned c = 0; c < relation.coordinates.size(); ++c) {
                const unsigned residue = (mask >> c) & 1U;
                auto variable = c < relation.dimensions ? getAffineDimExpr(c, &context) :
                                                        getAffineSymbolExpr(0, &context);
                variables.push_back(2 * variable + residue);
                residues.push_back(residue);
            }
            SmallVector<AffineExpr> constraints{variables.back()};
            SmallVector<bool> equalities{false};
            if (relation.sourceSite) {
                constraints.append({variables[0], variables.back() - variables[0] - 1});
                equalities.append({false, false});
            }
            if (order) {
                constraints.append({variables[1], variables.back() - variables[1] - 1,
                                    variables[1] - variables[0] - 1});
                equalities.append({false, false, false});
            }
            if (access) { constraints.push_back(variables[1] - residues[0]); equalities.push_back(true); }
            if (splitContextsAndOrders && (kind == Kind::Context || kind == Kind::Order)) {
                // Two disjoint alternatives share exactly the same residue key.
                // Both must survive indexed context and order enumeration.
                for (bool low : {true, false}) {
                    auto selectedRows = constraints;
                    auto selectedEqualities = equalities;
                    selectedRows.push_back(low ? 3 - variables.back() : variables.back() - 4);
                    selectedEqualities.push_back(false);
                    relation.pieces.push_back({IntegerSet::get(relation.dimensions, 1,
                        selectedRows, selectedEqualities), residues});
                }
            } else {
                relation.pieces.push_back({IntegerSet::get(relation.dimensions, 1, constraints, equalities), residues});
            }
        }
        program.primitives.relations.push_back(std::move(relation));
    }
    program.recognition = fs::recognizeArithmetic(program.primitives, {1, 3, 2, 2});
    auto adapters = selectorAdapters(program, {0});
    if (failed(adapters)) { return false; }
    for (const auto& result : *adapters) {
        if (!result.error.empty()) { llvm::errs() << result.error << "\n"; return false; }
        for (int64_t n = 0; n <= 5; ++n) {
            std::vector<Occurrence> occurrences;
            for (int64_t i = 0; i < n; ++i) { occurrences.push_back({0, i, 0, {i % 2}, {i % 2}}); }
            for (int64_t byte = -1; byte <= 2; ++byte) {
                if (!supportMatches(result, occurrences, byte, {Integer(n)})) { return false; }
            }
            for (const auto& boundary : result.boundaries) {
                for (int64_t byte = -1; byte <= 2; ++byte) {
                    SmallVector<Integer> input;
                    if (boundary.storageSpace) { input.push_back(Integer(byte)); }
                    auto selected = fs::evaluateGeneralArithmeticSelector(boundary.selector, 2, input, {Integer(n)});
                    if (failed(selected)) { return false; }
                    auto wanted = expected(occurrences, boundary, byte);
                    if (selected->has_value() != wanted.has_value() ||
                        (wanted && ((*selected)->site != wanted->site ||
                                    (*selected)->coordinates != wanted->coordinates))) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}
bool quotientImportCheck()
{
    MLIRContext context;
    fs::ArithmeticProgram program;
    program.sites.resize(1);
    program.primitives.period = 1;
    program.primitives.pipeCount = 1;
    for (auto kind : {Kind::Context, Kind::Occurrences, Kind::Order, Kind::Native,
                      Kind::Reads, Kind::Writes, Kind::Prerequisites}) {
        fs::PrimitiveRelation relation;
        relation.kind = kind;
        if (kind == Kind::Context) {
            relation.pieces.push_back({IntegerSet::get(0, 0,
                {getAffineConstantExpr(0, &context)}, {false}), {}});
        } else if (kind == Kind::Writes) {
            relation.sourceSite = 0;
            relation.sourceDimensions = 1;
            relation.dimensions = 2;
            relation.storageSpace = pto::AddressSpace::VEC;
            relation.coordinates = {{"i", fs::CoordinateKind::Occurrence},
                                    {"byte", fs::CoordinateKind::Storage}};
            auto i = getAffineDimExpr(0, &context), byte = getAffineDimExpr(1, &context);
            auto q = getAffineSymbolExpr(0, &context);
            SmallVector<AffineExpr> rows{i, 95 - i, i - 32 * q, 32 * q + 31 - i,
                                        byte - 4 * q, 4 * q + 3 - byte};
            relation.pieces.push_back({IntegerSet::get(2, 1, rows,
                SmallVector<bool>(rows.size(), false)), {0, 0}});
        }
        program.primitives.relations.push_back(std::move(relation));
    }
    program.recognition = fs::recognizeArithmetic(program.primitives, {1, 3, 1, 32});
    if (program.recognition.state != fs::RecognitionState::Applicable ||
        program.recognition.observedDimensions != 3 ||
        program.recognition.arithmeticClass != fs::ArithmeticClass::BoundedCoefficients) { return false; }
    // Locals count against the class dimension bound before elimination.
    if (fs::recognizeArithmetic(program.primitives, {1, 2, 1, 32}).state ==
        fs::RecognitionState::Applicable) { return false; }
    auto imported = fs::importArithmeticIntegerPieces(program);
    if (failed(imported) || imported->empty()) { return false; }
    for (int64_t i = -1; i <= 97; ++i) {
        for (int64_t byte = -1; byte <= 13; ++byte) {
            bool present = false;
            for (const auto& piece : *imported) {
                if (piece.schema->kind != Kind::Writes) { continue; }
                if (piece.system.dimensions() != 2 || piece.residues.size() != 2) { return false; }
                auto dot = [&](const std::vector<Integer>& coefficients) {
                    return coefficients[0] * Integer(i) + coefficients[1] * Integer(byte);
                };
                bool accepts = true;
                for (const auto& constraint : piece.system.constraints()) {
                    accepts &= dot(constraint.coefficients) <= constraint.bound;
                }
                for (const auto& constraint : piece.system.congruences()) {
                    accepts &= (dot(constraint.coefficients) - constraint.residue) % constraint.modulus == 0;
                }
                present |= accepts;
            }
            const bool wanted = i >= 0 && i < 96 && byte >= 4 * (i / 32) && byte < 4 * (i / 32) + 4;
            if (present != wanted) { return false; }
        }
    }
    return true;
}
bool ownedProjectionIdentityChecks()
{
    using Row = fs::IntegerConstraint;
    Integer large(1);
    for (unsigned i = 0; i < 80; ++i) { large *= Integer(2); }
    auto source = fs::IntegerSystem::create(2,
        {Row{{large, Integer(1)}, large}, Row{{Integer(-1), Integer(0)}, Integer(0)},
         Row{{Integer(1), Integer(0)}, Integer(1)}, Row{{Integer(0), Integer(-1)}, Integer(0)},
         Row{{Integer(0), Integer(1)}, Integer(2)}},
        {{{Integer(1), Integer(1)}, Integer(1), Integer(2)}});
    if (failed(source)) { return false; }
    fs::IntegerProjectionCache cache;
    auto mapped = cache.remap(*source, 2, {0, 1});
    if (failed(mapped)) { return false; }
    auto joined = cache.intersect(**mapped, **mapped);
    auto projected = cache.project(*source, {0, 1});
    if (failed(joined) || failed(projected) || projected->empty()) { return false; }
    const std::array<const fs::IntegerSystem*, 3> owned{*mapped, *joined, &projected->front()};
    for (const auto* value : owned) { if (cache.isEmpty(*value)) { return false; } }
    const auto before = cache.cost();
    for (unsigned repeat = 0; repeat < 64; ++repeat) {
        for (const auto* value : owned) { if (cache.isEmpty(*value)) { return false; } }
    }
    if (cache.cost().ownedIdentityHits != before.ownedIdentityHits + 192 ||
        cache.cost().structuralHashes != before.structuralHashes ||
        cache.cost().structuralComparisons != before.structuralComparisons) { return false; }
    // Caller-owned values can change at an unchanged address. A new numerical
    // context must be interned even when an old owned output has equal rows.
    auto empty = fs::IntegerSystem::create(2,
        {Row{{Integer(1), Integer(0)}, Integer(-1)}, Row{{Integer(-1), Integer(0)}, Integer(0)}});
    if (failed(empty)) { return false; }
    auto external = *source;
    const auto externalHashes = cache.cost().structuralHashes;
    if (cache.isEmpty(external)) { return false; }
    external = *empty;
    if (!cache.isEmpty(external) || cache.cost().structuralHashes != externalHashes + 2) { return false; }
    // Other cache owners must not borrow this cache's registered addresses.
    fs::IntegerProjectionCache independent;
    if (independent.isEmpty(**mapped) || independent.cost().ownedIdentityHits != 0 ||
        independent.cost().structuralHashes != 1) { return false; }
    for (int64_t bound = 2; bound < 130; ++bound) {
        auto value = fs::IntegerSystem::create(2,
            {Row{{Integer(1), Integer(0)}, Integer(bound)}, Row{{Integer(0), Integer(1)}, Integer(bound)}});
        if (failed(value) || failed(cache.remap(*value, 2, {1, 0})) ||
            failed(cache.intersect(*value, *source)) || failed(cache.project(*value, {1}))) { return false; }
    }
    const auto hashesAfterGrowth = cache.cost().structuralHashes;
    for (const auto* value : owned) { if (cache.isEmpty(*value)) { return false; } }
    if (cache.cost().structuralHashes != hashesAfterGrowth) { return false; }
    cache.clear(); // No previously returned pointer or view is used afterward.
    if (cache.cost().identityRequests || cache.cost().ownedIdentityHits || cache.cost().structuralHashes ||
        cache.cost().structuralComparisons || cache.cost().projectionBuilds || cache.cost().emptinessBuilds) {
        return false;
    }
    auto rebuilt = cache.remap(*empty, 2, {0, 1});
    if (failed(rebuilt) || !cache.isEmpty(**rebuilt) || cache.cost().remapBuilds != 1) { return false; }
    return true;
}
bool projectionCacheChecks()
{
    if (!ownedProjectionIdentityChecks()) { return false; }
    using Row = fs::IntegerConstraint;
    auto system = fs::IntegerSystem::create(2,
        {Row{{Integer(1), Integer(-1)}, Integer(1)}, Row{{Integer(-1), Integer(1)}, Integer(-1)},
         Row{{Integer(0), Integer(-1)}, Integer(0)}, Row{{Integer(0), Integer(1)}, Integer(3)}},
        {{{Integer(1), Integer(0)}, Integer(0), Integer(2)}});
    if (failed(system)) { return false; }
    fs::IntegerProjectionCache cache;
    auto first = cache.project(*system, {0});
    auto repeated = cache.project(*system, {0});
    if (failed(first) || failed(repeated) || first->empty() || first->data() != repeated->data() ||
        cache.cost().projectionBuilds != 1 || cache.cost().projectionHits != 1) { return false; }
    if (cache.isEmpty(*system) || cache.isEmpty(*system) || cache.cost().emptinessBuilds != 1 ||
        cache.cost().emptinessHits != 1) { return false; }
    auto contains = [](llvm::ArrayRef<fs::IntegerSystem> unionPieces, llvm::ArrayRef<Integer> point) {
        std::vector<Row> binding;
        for (unsigned i = 0; i < point.size(); ++i) {
            std::vector<Integer> positive(point.size(), Integer(0)), negative(point.size(), Integer(0));
            positive[i] = 1; negative[i] = -1;
            binding.push_back({std::move(positive), point[i]});
            binding.push_back({std::move(negative), -point[i]});
        }
        auto fixed = fs::IntegerSystem::create(point.size(), binding);
        if (failed(fixed)) { return false; }
        for (const auto& piece : unionPieces) {
            auto intersection = piece.intersect(*fixed);
            if (succeeded(intersection) && !intersection->isEmpty()) { return true; }
        }
        return false;
    };
    auto mapped = cache.remap(*system, 2, {1, 0});
    auto mappedRetry = cache.remap(*system, 2, {1, 0});
    auto unchanged = cache.remap(*system, 2, {0, 1});
    auto limit = fs::IntegerSystem::create(2, {Row{{Integer(1), Integer(0)}, Integer(1)}});
    if (failed(mapped) || failed(mappedRetry) || failed(unchanged) || failed(limit) ||
        *mapped != *mappedRetry || cache.cost().remapBuilds != 2 || cache.cost().remapHits != 1) {
        return false;
    }
    auto limited = cache.intersect(**mapped, *limit);
    auto limitedRetry = cache.intersect(**mapped, *limit);
    if (failed(limited) || failed(limitedRetry) || *limited != *limitedRetry ||
        cache.cost().intersectionBuilds != 1 || cache.cost().intersectionHits != 1) { return false; }
    for (int64_t x = -1; x <= 5; ++x) {
        for (int64_t y = -1; y <= 5; ++y) {
            const bool expected = y >= 0 && y <= 3 && x == y + 1 && x % 2 == 0;
            if (contains(llvm::ArrayRef<fs::IntegerSystem>(*mapped, 1), {Integer(y), Integer(x)}) != expected ||
                contains(llvm::ArrayRef<fs::IntegerSystem>(*unchanged, 1), {Integer(x), Integer(y)}) != expected ||
                contains(llvm::ArrayRef<fs::IntegerSystem>(*limited, 1), {Integer(y), Integer(x)}) !=
                    (expected && y <= 1)) { return false; }
        }
    }
    auto witnesses = cache.witnesses(*system, 1, {0});
    auto witnessesRetry = cache.witnesses(*system, 1, {0});
    auto shiftedResidue = cache.witnesses(*system, 1, {1});
    if (failed(witnesses) || failed(witnessesRetry) || failed(shiftedResidue) || witnesses->empty() ||
        witnesses->data() != witnessesRetry->data() || shiftedResidue->empty() ||
        cache.cost().witnessBuilds != 2 || cache.cost().witnessHits != 1) { return false; }
    for (const auto& witness : *witnesses) {
        if (witness.outputs.size() != 1 || witness.outputs.front().residue != 0) { return false; }
    }
    for (const auto& witness : *shiftedResidue) {
        if (witness.outputs.size() != 1 || witness.outputs.front().residue != 1) { return false; }
    }
    auto second = cache.project(*system, {1});
    auto reversed = cache.project(*system, {1, 0});
    if (failed(second) || failed(reversed)) { return false; }
    for (int64_t x = -1; x <= 5; ++x) {
        if (contains(*first, {Integer(x)}) != (x == 2 || x == 4) ||
            contains(*second, {Integer(x)}) != (x == 1 || x == 3)) { return false; }
        for (int64_t y = -1; y <= 5; ++y) {
            const bool expected = y >= 0 && y <= 3 && x == y + 1 && x % 2 == 0;
            if (contains(*reversed, {Integer(y), Integer(x)}) != expected) { return false; }
        }
    }
    // Different bound/residue contexts must not share cached answers. Growing
    // the pool also must not invalidate an earlier map-owned projection view.
    for (int64_t bound = 0; bound <= 32; ++bound) {
        for (int64_t residue = 0; residue < 2; ++residue) {
            auto changed = fs::IntegerSystem::create(1,
                {Row{{Integer(-1)}, Integer(0)}, Row{{Integer(1)}, Integer(bound)}},
                {{{Integer(1)}, Integer(residue), Integer(2)}});
            if (failed(changed)) { return false; }
            auto projected = cache.project(*changed, {0});
            if (failed(projected)) { return false; }
            for (int64_t point = -1; point <= 33; ++point) {
                const bool expected = point >= 0 && point <= bound && point % 2 == residue;
                if (contains(*projected, {Integer(point)}) != expected) { return false; }
            }
        }
    }
    if (!contains(*first, {Integer(2)}) || contains(*first, {Integer(3)})) { return false; }
    if (!contains(llvm::ArrayRef<fs::IntegerSystem>(*mapped, 1), {Integer(3), Integer(4)}) ||
        !contains(llvm::ArrayRef<fs::IntegerSystem>(*limited, 1), {Integer(1), Integer(2)})) { return false; }
    if (witnesses->empty() || witnesses->front().outputs.front().residue != 0) { return false; }
    const auto witnessBuilds = cache.cost().witnessBuilds, witnessHits = cache.cost().witnessHits;
    if (succeeded(cache.witnesses(*system, 2, {0})) || succeeded(cache.witnesses(*system, 2, {0})) ||
        cache.cost().witnessBuilds != witnessBuilds + 2 || cache.cost().witnessHits != witnessHits) { return false; }
    const auto remapBuilds = cache.cost().remapBuilds, remapHits = cache.cost().remapHits;
    if (succeeded(cache.remap(*system, 2, {0})) || succeeded(cache.remap(*system, 2, {0})) ||
        cache.cost().remapBuilds != remapBuilds + 2 || cache.cost().remapHits != remapHits) { return false; }
    auto wrongDimensions = fs::IntegerSystem::create(1, {});
    if (failed(wrongDimensions)) { return false; }
    const auto intersectionBuilds = cache.cost().intersectionBuilds;
    const auto intersectionHits = cache.cost().intersectionHits;
    if (succeeded(cache.intersect(*system, *wrongDimensions)) ||
        succeeded(cache.intersect(*system, *wrongDimensions)) ||
        cache.cost().intersectionBuilds != intersectionBuilds + 2 ||
        cache.cost().intersectionHits != intersectionHits) { return false; }

    const auto builds = cache.cost().projectionBuilds, hits = cache.cost().projectionHits;
    if (succeeded(cache.project(*system, {2})) || succeeded(cache.project(*system, {2})) ||
        succeeded(cache.project(*system, {0, 0})) || cache.cost().projectionBuilds != builds + 3 ||
        cache.cost().projectionHits != hits) { return false; }
    auto empty = fs::IntegerSystem::create(0, {Row{{}, Integer(-1)}});
    auto nonempty = fs::IntegerSystem::create(0, {});
    if (failed(empty) || failed(nonempty)) { return false; }
    auto emptyProjection = cache.project(*empty, {}), trueProjection = cache.project(*nonempty, {});
    auto emptyRetry = cache.project(*empty, {});
    return succeeded(emptyProjection) && emptyProjection->empty() && succeeded(trueProjection) &&
        trueProjection->size() == 1 && succeeded(emptyRetry) && emptyRetry->empty();
}
bool rationalCoverChecks()
{
    using Row = fs::IntegerConstraint;
    auto coupled = fs::IntegerSystem::create(3,
        {Row{{Integer(-1), Integer(0), Integer(0)}, Integer(0)},
         Row{{Integer(1), Integer(0), Integer(-1)}, Integer(0)},
         Row{{Integer(0), Integer(0), Integer(1)}, Integer(3)},
         Row{{Integer(-4), Integer(1), Integer(0)}, Integer(1)},
         Row{{Integer(4), Integer(-1), Integer(0)}, Integer(-1)}},
        {{{Integer(0), Integer(1), Integer(0)}, Integer(1), Integer(2)}});
    if (failed(coupled)) { return false; }
    uint64_t requests = 0;
    auto cover = fs::rationalCoordinateCover(*coupled, 1, requests);
    if (!cover || *cover != std::make_pair(Integer(1), Integer(13)) || requests != 2) { return false; }
    // Direct integer inequalities supply the expected accesses independently.
    for (int64_t i = -1; i <= 4; ++i) {
        for (int64_t parameter = -1; parameter <= 4; ++parameter) {
            for (int64_t byte = -2; byte <= 16; ++byte) {
                const bool accessed = i >= 0 && i <= parameter && parameter <= 3 && byte == 4 * i + 1;
                if (accessed && (Integer(byte) < cover->first || Integer(byte) > cover->second)) { return false; }
            }
        }
    }
    auto negative = fs::IntegerSystem::create(2,
        {Row{{Integer(1), Integer(1)}, Integer(-3)}, Row{{Integer(1), Integer(-1)}, Integer(0)},
         Row{{Integer(-1), Integer(0)}, Integer(4)}});
    auto half = fs::IntegerSystem::create(2,
        {Row{{Integer(1), Integer(1)}, Integer(1)}, Row{{Integer(-1), Integer(-1)}, Integer(-1)},
         Row{{Integer(1), Integer(-1)}, Integer(0)}, Row{{Integer(-1), Integer(1)}, Integer(0)}});
    auto empty = fs::IntegerSystem::create(1, {Row{{Integer(0)}, Integer(-1)}});
    auto free = fs::IntegerSystem::create(1, {});
    if (failed(negative) || failed(half) || failed(empty) || failed(free)) { return false; }
    auto signedCover = fs::rationalCoordinateCover(*negative, 0, requests);
    auto emptyIntegerCover = fs::rationalCoordinateCover(*half, 0, requests);
    if (!signedCover || *signedCover != std::make_pair(Integer(-4), Integer(-2)) ||
        !emptyIntegerCover || *emptyIntegerCover != std::make_pair(Integer(1), Integer(0)) ||
        fs::rationalCoordinateCover(*empty, 0, requests) || fs::rationalCoordinateCover(*free, 0, requests) ||
        fs::rationalCoordinateCover(*negative, 2, requests)) { return false; }
    Integer huge(1);
    for (unsigned bit = 0; bit < 100; ++bit) { huge *= 2; }
    auto wide = fs::IntegerSystem::create(1, {Row{{huge}, -huge * 3}, Row{{-huge}, huge * 7}});
    if (failed(wide)) { return false; }
    auto wideCover = fs::rationalCoordinateCover(*wide, 0, requests);
    return wideCover && *wideCover == std::make_pair(Integer(-7), Integer(-3));
}
bool selectiveImportChecks()
{
    for (unsigned count : {4U, 16U, 64U}) {
        Fixture fixture(count);
        auto& program = fixture.program;
        if (program.recognition.state != fs::RecognitionState::Applicable) { return false; }
        auto all = fs::importOwnedArithmeticIntegerPieces(program);
        auto needed = fs::importOwnedArithmeticIntegerPieces(program, fs::ArithmeticPrimitiveRequest::Storage);
        if (failed(all) || failed(needed) || all->constructionCost().projectionRequests != count ||
            needed->constructionCost().projectionRequests != 0) { return false; }
        unsigned ignored = 0;
        std::size_t at = 0;
        for (const auto& piece : all->pieces()) {
            if (piece.schema->kind == Kind::Native || piece.schema->kind == Kind::Prerequisites) {
                ++ignored; continue;
            }
            if (at == needed->pieces().size() || !sameImportedPiece(piece, needed->pieces()[at++])) { return false; }
        }
        if (ignored != count || at != needed->pieces().size()) { return false; }
        auto audit = fs::importArithmeticIntegerPieces(program);
        if (failed(audit) || audit->size() != all->pieces().size()) { return false; }
    }
    return true;
}
bool check()
{
    Fixture fixture(4);
    auto adapters = selectorAdapters(fixture.program, fixture.pipes);
    if (failed(adapters)) { return false; }
    for (const auto& result : *adapters) {
        if (!result.error.empty()) { llvm::errs() << result.error << "\n"; return false; }
        if (result.boundaries.empty() || result.support.empty() || !result.cost.projectionHits ||
            !result.cost.emptinessHits || !result.cost.remapHits || !result.cost.intersectionHits ||
            !result.cost.witnessHits || !result.importCost.remapHits ||
            !result.importCost.intersectionHits || !result.importCost.emptinessHits ||
            result.cost.witnessRequests != result.cost.witnessHits + result.cost.witnessBuilds ||
            result.cost.remapRequests != result.cost.remapHits + result.cost.remapBuilds ||
            result.cost.intersectionRequests != result.cost.intersectionHits + result.cost.intersectionBuilds ||
            result.cost.projectionRequests != result.cost.projectionHits + result.cost.projectionBuilds) {
            return false;
        }
        for (int64_t n : {0, 1, 2, 4, 1000000000}) {
            for (int64_t guard = 0; guard <= 1; ++guard) {
                // The large instance checks constant-size selectors without unfolding.
                const auto occurrences = n < 5 ? execution(n, guard) : std::vector<Occurrence>{};
                if (n < 5) {
                    for (int64_t byte = -1; byte <= 5; ++byte) {
                        if (!supportMatches(result, occurrences, byte, {Integer(n), Integer(guard)})) { return false; }
                    }
                }
                for (const auto& boundary : result.boundaries) {
                    for (int64_t byte = -1; byte <= 5; ++byte) {
                        SmallVector<Integer> input;
                        if (boundary.storageSpace) { input.push_back(Integer(byte)); }
                        auto selected = fs::evaluateGeneralArithmeticSelector(boundary.selector, result.period,
                                                                               input, {Integer(n), Integer(guard)});
                        if (failed(selected)) { return false; }
                        if (n >= 5) { continue; }
                        const auto wanted = expected(occurrences, boundary, byte);
                        if (selected->has_value() != wanted.has_value() ||
                            (wanted && ((*selected)->site != wanted->site ||
                                        (*selected)->coordinates != wanted->coordinates))) {
                            llvm::errs() << "boundary mismatch kind=" << static_cast<unsigned>(boundary.kind)
                                         << " n=" << n << " guard=" << guard << " byte=" << byte << "\n";
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}
bool absentOrderChecks()
{
    MLIRContext context;
    for (unsigned sites : {4U, 16U, 64U}) {
        for (bool accesses : {false, true}) {
            fs::ArithmeticProgram program;
            program.primitives.period = 1;
            program.primitives.pipeCount = 1;
            program.primitives.parameters = {"choice"};
            program.sites.resize(sites);
            auto parameter = getAffineSymbolExpr(0, &context);
            auto append = [&](Kind kind, std::optional<unsigned> site) {
                fs::PrimitiveRelation relation;
                relation.kind = kind;
                relation.sourceSite = site;
                const bool access = kind == Kind::Reads || kind == Kind::Writes;
                if (access) {
                    relation.dimensions = 1;
                    relation.storageSpace = pto::AddressSpace::VEC;
                    relation.coordinates.push_back({"byte", fs::CoordinateKind::Storage});
                }
                relation.coordinates.push_back({"choice", fs::CoordinateKind::Parameter});
                if (kind == Kind::Context || site) {
                    SmallVector<AffineExpr> constraints;
                    SmallVector<bool> equalities;
                    if (site) {
                        constraints.push_back(parameter - *site);
                        equalities.push_back(true);
                    } else {
                        constraints = {parameter, sites - 1 - parameter};
                        equalities = {false, false};
                    }
                    if (access) {
                        constraints.push_back(getAffineDimExpr(0, &context));
                        equalities.push_back(true);
                    }
                    relation.pieces.push_back({IntegerSet::get(relation.dimensions, 1,
                        constraints, equalities), SmallVector<uint64_t>(relation.coordinates.size(), 0)});
                }
                program.primitives.relations.push_back(std::move(relation));
            };
            for (auto kind : {Kind::Context, Kind::Order, Kind::Native, Kind::Prerequisites}) {
                append(kind, {});
            }
            if (!accesses) { append(Kind::Reads, {}); append(Kind::Writes, {}); }
            // Exactly one scalar occurrence is present for each admissible
            // entry choice, so the strict reference-order relation is empty.
            for (unsigned site = 0; site < sites; ++site) {
                append(Kind::Occurrences, site);
                if (accesses) { append(Kind::Reads, site); append(Kind::Writes, site); }
            }
            program.recognition = fs::recognizeArithmetic(program.primitives, {1, 2, 1, 1});
            auto result = fs::buildArithmeticStorageSelectors(program, std::vector<uint32_t>(sites, 0));
            if (!result.error.empty() || !result.cost.emptyOrderSkips ||
                result.cost.intersectionRequests != (accesses ? 4 * sites : 0) ||
                result.cost.projectionRequests != (accesses ? 2 * sites : 0)) {
                llvm::errs() << "absent-order construction sites=" << sites << " accesses=" << accesses
                    << " error=" << result.error << " skips=" << result.cost.emptyOrderSkips
                    << " intersections=" << result.cost.intersectionRequests
                    << " projections=" << result.cost.projectionRequests << "\n";
                return false;
            }
            std::set<std::pair<Boundary, std::optional<std::size_t>>> expectedBoundaries{
                {Boundary::FirstPayload, {}}, {Boundary::LastPayload, {}}};
            for (unsigned site = 0; site < sites; ++site) {
                expectedBoundaries.emplace(Boundary::FirstSite, site);
                expectedBoundaries.emplace(Boundary::LastSite, site);
            }
            if (accesses) {
                for (auto kind : {Boundary::FirstWriter, Boundary::LastWriter,
                                  Boundary::FirstReaderBeforeWrite, Boundary::LastReaderAfterWrite}) {
                    expectedBoundaries.emplace(kind, std::nullopt);
                }
            }
            if (result.cost.emptyOrderSkips > result.cost.blockerRequests) { return false; }
            for (const auto& boundary : result.boundaries) {
                if (!expectedBoundaries.erase({boundary.kind, boundary.site})) { return false; }
            }
            if (!expectedBoundaries.empty()) { return false; }
            for (int64_t choice = -1; choice <= sites; ++choice) {
                std::vector<Occurrence> occurrences;
                if (choice >= 0 && choice < sites) {
                    occurrences.push_back({static_cast<std::size_t>(choice), 0, 0,
                        accesses ? std::set<int64_t>{0} : std::set<int64_t>{},
                        accesses ? std::set<int64_t>{0} : std::set<int64_t>{}});
                }
                for (int64_t byte : {-1, 0, 1}) {
                    if (!supportMatches(result, occurrences, byte, {Integer(choice)})) { return false; }
                }
            }
            for (int64_t choice = -1; choice <= sites; ++choice) {
                for (const auto& boundary : result.boundaries) {
                    for (int64_t byte : {-1, 0, 1}) {
                        SmallVector<Integer> input;
                        if (boundary.storageSpace) { input.push_back(Integer(byte)); }
                        auto actual = fs::evaluateGeneralArithmeticSelector(
                            boundary.selector, 1, input, {Integer(choice)});
                        const bool reader = boundary.kind == Boundary::FirstReaderBeforeWrite ||
                            boundary.kind == Boundary::LastReaderAfterWrite;
                        const bool present = choice >= 0 && choice < sites && !reader &&
                            (!boundary.site || *boundary.site == static_cast<std::size_t>(choice)) &&
                            (!boundary.storageSpace || byte == 0);
                        if (failed(actual) || actual->has_value() != present ||
                            (present && ((*actual)->site != static_cast<std::size_t>(choice) ||
                                         !(*actual)->coordinates.empty()))) { return false; }
                    }
                }
            }
        }
    }
    return true;
}
} // namespace
int runArithmeticStorageSelectorChecks()
{
    for (const auto& test : std::vector<std::pair<const char*, bool(*)()>>{
             {"projection cache", projectionCacheChecks}, {"rational cover", rationalCoverChecks},
             {"occurrence oracle", check}, {"selective import", selectiveImportChecks},
             {"absent order", absentOrderChecks},
             {"residue", [] { return residueCheck(); }},
             {"split residues", [] { return residueCheck(true); }}, {"quotient import", quotientImportCheck}}) {
        if (!test.second()) {
            llvm::errs() << "arithmetic storage selector checks failed: " << test.first << "\n";
            return 1;
        }
    }
    llvm::outs() << "arithmetic storage boundary selectors match independent byte lifetimes\n";
    return 0;
}
