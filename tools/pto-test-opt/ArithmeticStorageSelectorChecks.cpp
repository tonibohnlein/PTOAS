// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent occurrence/byte oracle for symbolic region boundary selectors.
#include "PTO/Transforms/FrontierSynch/ArithmeticStorageSelectors.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"
#include <set>
namespace fs = mlir::pto::frontiersynch;
using namespace mlir;
namespace {
using Kind = fs::PrimitiveKind;
using Boundary = fs::ArithmeticBoundaryKind;
using Integer = fs::BoundInteger;
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
    Fixture()
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
        program.recognition = fs::recognizeArithmetic(program.primitives, {2, 4, 1, 1});
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
bool residueCheck()
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
            relation.pieces.push_back({IntegerSet::get(relation.dimensions, 1, constraints, equalities), residues});
        }
        program.primitives.relations.push_back(std::move(relation));
    }
    program.recognition = fs::recognizeArithmetic(program.primitives, {1, 3, 2, 2});
    auto result = fs::buildArithmeticStorageSelectors(program, {0});
    if (!result.error.empty()) { llvm::errs() << result.error << "\n"; return false; }
    for (int64_t n = 0; n <= 5; ++n) {
        std::vector<Occurrence> occurrences;
        for (int64_t i = 0; i < n; ++i) { occurrences.push_back({0, i, 0, {i % 2}, {i % 2}}); }
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
bool check()
{
    Fixture fixture;
    const auto result = fs::buildArithmeticStorageSelectors(fixture.program, fixture.pipes);
    if (!result.error.empty()) { llvm::errs() << result.error << "\n"; return false; }
    if (result.boundaries.empty() || result.support.empty()) { return false; }
    for (int64_t n : {0, 1, 2, 4, 1000000000}) {
        for (int64_t guard = 0; guard <= 1; ++guard) {
            // The large instance checks constant-size selectors without unfolding.
            const auto occurrences = n < 5 ? execution(n, guard) : std::vector<Occurrence>{};
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
    return true;
}
} // namespace
int runArithmeticStorageSelectorChecks()
{
    if (!check() || !residueCheck() || !quotientImportCheck()) {
        llvm::errs() << "arithmetic storage selector checks failed\n";
        return 1;
    }
    llvm::outs() << "arithmetic storage boundary selectors match independent byte lifetimes\n";
    return 0;
}
