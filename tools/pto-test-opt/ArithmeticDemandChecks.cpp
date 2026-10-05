// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite physical-conflict DAGs for the symbolic demand backend.
#include "PTO/Transforms/FrontierSynch/ArithmeticSelectors.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>
#include <tuple>
namespace fs = mlir::pto::frontiersynch;
using namespace mlir;
namespace {
struct Occurrence {
    unsigned site;
    int64_t iteration;
    uint32_t pipe;
    std::set<int64_t> reads, writes;
};
std::vector<uint32_t> pipes(unsigned scene)
{
    return scene == 0 ? std::vector<uint32_t>{0, 1, 2} :
           scene == 1 ? std::vector<uint32_t>{0, 1} : std::vector<uint32_t>{0, 0, 0};
}
std::vector<Occurrence> execution(unsigned scene, int64_t trips, int64_t guard)
{
    std::vector<Occurrence> result;
    for (int64_t i = 0; i < trips; ++i) {
        if (scene == 0) {
            result.push_back({0, i, 0, {}, {0}});
            result.push_back({1, i, 1, guard ? std::set<int64_t>{} : std::set<int64_t>{0},
                             guard ? std::set<int64_t>{1} : std::set<int64_t>{}});
            result.push_back({2, i, 2, {1}, {}});
        } else if (scene == 1) {
            result.push_back({0, i, 0, {}, {i % 2}});
            if (guard) { result.push_back({1, i, 1, {i % 2}, {}}); }
        } else {
            result.push_back({0, i, 0, {}, {0}});
            result.push_back({1, i, 0, {}, {1}});
            result.push_back({2, i, 0, {}, {0}});
        }
    }
    return result;
}
bool hazard(const Occurrence& a, const Occurrence& b)
{
    for (auto cell : a.writes) {
        if (b.reads.count(cell) || b.writes.count(cell)) { return true; }
    }
    for (auto cell : a.reads) {
        if (b.writes.count(cell)) { return true; }
    }
    return false;
}
using Matrix = std::vector<std::vector<bool>>;
Matrix closure(const std::vector<Occurrence>& occurrences, bool conflicts)
{
    const auto count = 2 * occurrences.size();
    Matrix result(count, std::vector<bool>(count, false));
    for (std::size_t a = 0; a < occurrences.size(); ++a) {
        result[2 * a][2 * a] = result[2 * a + 1][2 * a + 1] = true;
        result[2 * a][2 * a + 1] = true;
        for (std::size_t b = a + 1; b < occurrences.size(); ++b) {
            if (occurrences[a].pipe == occurrences[b].pipe) {
                result[2 * a][2 * b] = result[2 * a + 1][2 * b + 1] = true;
            }
            if (conflicts && hazard(occurrences[a], occurrences[b])) { result[2 * a + 1][2 * b] = true; }
        }
    }
    for (std::size_t via = 0; via < count; ++via) {
        for (std::size_t a = 0; a < count; ++a) {
            for (std::size_t b = 0; b < count; ++b) {
                result[a][b] = result[a][b] || (result[a][via] && result[via][b]);
            }
        }
    }
    return result;
}
Matrix covers(const std::vector<Occurrence>& occurrences, const Matrix& reach)
{
    const auto count = reach.size();
    Matrix result(count, std::vector<bool>(count, false));
    for (std::size_t a = 0; a < occurrences.size(); ++a) {
        for (std::size_t b = a + 1; b < occurrences.size(); ++b) {
            if (!hazard(occurrences[a], occurrences[b])) { continue; }
            const auto source = 2 * a + 1, target = 2 * b;
            bool intermediate = false;
            for (std::size_t via = 0; via < count; ++via) {
                intermediate |= via != source && via != target && reach[source][via] && reach[via][target];
            }
            result[source][target] = !intermediate;
        }
    }
    return result;
}
class Bundle {
public:
    Bundle(MLIRContext& context, unsigned scene, uint64_t period) : context(context)
    {
        program.primitives.period = period;
        const auto labels = pipes(scene);
        program.primitives.pipeCount = std::set<uint32_t>(labels.begin(), labels.end()).size();
        program.primitives.parameters = {"trips", "guard"};
        program.sites.resize(labels.size());
        // Explicit empty roles distinguish known absence from missing premises.
        for (auto kind : {fs::PrimitiveKind::Context, fs::PrimitiveKind::Occurrences, fs::PrimitiveKind::Order,
                          fs::PrimitiveKind::Native, fs::PrimitiveKind::Reads, fs::PrimitiveKind::Writes,
                          fs::PrimitiveKind::Prerequisites}) {
            schema(kind, -1, -1, fs::ArithmeticEvent::Payload, fs::ArithmeticEvent::Payload);
        }
        for (int64_t n = 0; n <= 3; ++n) {
            for (int64_t g = 0; g <= 1; ++g) { addExecution(scene, n, g); }
        }
        program.recognition = fs::recognizeArithmetic(program.primitives, {3, 4, period, 1});
    }
    fs::ArithmeticProgram program;
private:
    MLIRContext& context;
    using Tag = std::tuple<fs::PrimitiveKind, int, int, fs::ArithmeticEvent, fs::ArithmeticEvent>;
    std::map<Tag, std::size_t> schemas;
    std::size_t schema(fs::PrimitiveKind kind, int source, int target, fs::ArithmeticEvent a, fs::ArithmeticEvent b)
    {
        auto [entry, inserted] = schemas.emplace(Tag{kind, source, target, a, b}, program.primitives.relations.size());
        if (!inserted) { return entry->second; }
        fs::PrimitiveRelation relation;
        relation.kind = kind;
        if (source >= 0) { relation.sourceSite = source; relation.sourceDimensions = 1; }
        if (target >= 0) { relation.targetSite = target; relation.targetDimensions = 1; }
        relation.sourceEvent = a; relation.targetEvent = b;
        const bool access = kind == fs::PrimitiveKind::Reads || kind == fs::PrimitiveKind::Writes;
        relation.dimensions = relation.sourceDimensions + relation.targetDimensions + (access ? 1 : 0);
        for (unsigned i = 0; i < relation.dimensions; ++i) {
            relation.coordinates.push_back({"d" + std::to_string(i),
                access && i + 1 == relation.dimensions ? fs::CoordinateKind::Storage : fs::CoordinateKind::Occurrence});
        }
        for (const auto& name : program.primitives.parameters) {
            relation.coordinates.push_back({name, fs::CoordinateKind::Parameter});
        }
        if (access) { relation.storageSpace = mlir::pto::AddressSpace::VEC; }
        program.primitives.relations.push_back(std::move(relation));
        return entry->second;
    }
    void point(fs::PrimitiveKind kind, int source, int target, fs::ArithmeticEvent a, fs::ArithmeticEvent b,
               SmallVector<int64_t> values, int64_t n, int64_t g)
    {
        auto id = schema(kind, source, target, a, b);
        auto& relation = program.primitives.relations[id];
        values.push_back(n); values.push_back(g);
        SmallVector<AffineExpr> rows;
        SmallVector<uint64_t> residues;
        const auto period = static_cast<int64_t>(program.primitives.period);
        for (unsigned i = 0; i < values.size(); ++i) {
            auto variable = i < relation.dimensions ? getAffineDimExpr(i, &context) :
                                                     getAffineSymbolExpr(i - relation.dimensions, &context);
            rows.push_back(variable - values[i] / period);
            residues.push_back(values[i] % period);
        }
        relation.pieces.push_back({IntegerSet::get(relation.dimensions, 2, rows,
                                                  SmallVector<bool>(rows.size(), true)), residues});
    }
    void addExecution(unsigned scene, int64_t n, int64_t g)
    {
        using Kind = fs::PrimitiveKind;
        using Event = fs::ArithmeticEvent;
        point(Kind::Context, -1, -1, Event::Payload, Event::Payload, {}, n, g);
        auto occurrences = execution(scene, n, g);
        auto native = closure(occurrences, false);
        for (const auto& occurrence : occurrences) {
            point(Kind::Occurrences, occurrence.site, -1, Event::Payload, Event::Payload,
                  {occurrence.iteration}, n, g);
            for (auto cell : occurrence.reads) {
                point(Kind::Reads, occurrence.site, -1, Event::Payload, Event::Payload,
                      {occurrence.iteration, cell}, n, g);
            }
            for (auto cell : occurrence.writes) {
                point(Kind::Writes, occurrence.site, -1, Event::Payload, Event::Payload,
                      {occurrence.iteration, cell}, n, g);
            }
        }
        // The supplied native primitive here is the strict closure; reflexive
        // identity is recovered independently from occurrence primitives.
        for (std::size_t a = 0; a < occurrences.size(); ++a) {
            for (std::size_t b = a; b < occurrences.size(); ++b) {
                const auto& source = occurrences[a]; const auto& target = occurrences[b];
                if (a < b) {
                    point(Kind::Order, source.site, target.site, Event::Payload, Event::Payload,
                          {source.iteration, target.iteration}, n, g);
                }
                for (unsigned ea = 0; ea < 2; ++ea) {
                    for (unsigned eb = 0; eb < 2; ++eb) {
                        if (a == b && ea == eb) { continue; }
                        if (native[2 * a + ea][2 * b + eb]) {
                            point(Kind::Native, source.site, target.site,
                                  ea ? Event::Completion : Event::Start, eb ? Event::Completion : Event::Start,
                                  {source.iteration, target.iteration}, n, g);
                        }
                    }
                }
            }
        }
    }
};
bool containsPoint(const fs::DifferenceBoundSystem& piece, llvm::ArrayRef<fs::BoundInteger> coordinates)
{
    if (piece.isEmpty()) { return false; }
    for (const auto& row : piece.constraints()) {
        if (coordinates[row.lhs] - coordinates[row.rhs] > row.bound) { return false; }
    }
    return true;
}
bool containsPoint(const fs::IntegerSystem& piece, llvm::ArrayRef<fs::BoundInteger> coordinates)
{
    auto dot = [&](llvm::ArrayRef<fs::BoundInteger> coefficients) {
        fs::BoundInteger sum(0);
        for (auto [i, coefficient] : llvm::enumerate(coefficients)) { sum += coefficient * coordinates[i + 1]; }
        return sum;
    };
    for (const auto& row : piece.constraints()) {
        if (dot(row.coefficients) > row.bound) { return false; }
    }
    for (const auto& row : piece.congruences()) {
        if (mod(dot(row.coefficients) - row.residue, row.modulus) != fs::BoundInteger(0)) { return false; }
    }
    return true;
}
template<class System>
bool member(const fs::TypedArithmeticRelation<System>& relation,
            uint64_t period, unsigned source, int64_t x, unsigned ea,
            unsigned target, int64_t y, unsigned eb, int64_t n, int64_t g)
{
    const auto p = static_cast<int64_t>(period);
    fs::ArithmeticRelationKey key{{source, ea ? fs::ArithmeticEvent::Completion : fs::ArithmeticEvent::Start,
                                   {static_cast<uint64_t>(x % p)}},
                                  {target, eb ? fs::ArithmeticEvent::Completion : fs::ArithmeticEvent::Start,
                                   {static_cast<uint64_t>(y % p)}},
                                  {static_cast<uint64_t>(n % p), static_cast<uint64_t>(g % p)}};
    auto found = relation.find(key);
    if (found == relation.end()) { return false; }
    const std::vector<fs::BoundInteger> coordinates{
        fs::BoundInteger(0), fs::BoundInteger(x / p), fs::BoundInteger(y / p),
        fs::BoundInteger(n / p), fs::BoundInteger(g / p)};
    return llvm::any_of(found->second, [&](const auto& piece) { return containsPoint(piece, coordinates); });
}
template<class System>
bool checkCase(const fs::TypedArithmeticDemandAnalysis<System>& analysis,
               unsigned scene, int64_t n, int64_t g,
               bool& adjacent)
{
    auto occurrences = execution(scene, n, g);
    auto native = closure(occurrences, false), reach = closure(occurrences, true);
    auto minimum = covers(occurrences, reach);
    std::map<std::pair<unsigned, int64_t>, std::size_t> positions;
    for (auto [i, occurrence] : llvm::enumerate(occurrences)) {
        positions[{occurrence.site, occurrence.iteration}] = i;
    }
    for (std::size_t a = 0; a < occurrences.size(); ++a) {
        for (std::size_t b = a + 1; b < occurrences.size(); ++b) {
            if (!minimum[2 * a + 1][2 * b] || occurrences[a].pipe != occurrences[b].pipe) { continue; }
            for (auto between = a + 1; between < b; ++between) {
                adjacent &= occurrences[between].pipe != occurrences[a].pipe;
            }
        }
    }
    const auto labels = pipes(scene);
    // Query absent sites and one coordinate beyond each finite domain as well.
    for (unsigned a = 0; a < labels.size(); ++a) {
        for (unsigned b = 0; b < labels.size(); ++b) {
            for (int64_t x = 0; x <= n; ++x) {
                for (int64_t y = 0; y <= n; ++y) {
                    auto source = positions.find({a, x}), target = positions.find({b, y});
                    const bool present = source != positions.end() && target != positions.end();
                    for (unsigned ea = 0; ea < 2; ++ea) {
                        for (unsigned eb = 0; eb < 2; ++eb) {
                            const auto i = present ? 2 * source->second + ea : 0;
                            const auto j = present ? 2 * target->second + eb : 0;
                            if (member(analysis.nativeOrder, analysis.period, a, x, ea, b, y, eb, n, g) !=
                                    (present && native[i][j]) ||
                                member(analysis.requiredOrder, analysis.period, a, x, ea, b, y, eb, n, g) !=
                                    (present && i != j && reach[i][j]) ||
                                member(analysis.minimumDemands, analysis.period, a, x, ea, b, y, eb, n, g) !=
                                    (present && minimum[i][j])) { return false; }
                        }
                    }
                }
            }
        }
    }
    return true;
}
template<class Analyze>
int runChecks(Analyze analyze, llvm::StringRef name)
{
    MLIRContext context;
    context.disableMultithreading();
    unsigned checked = 0;
    for (unsigned scene = 0; scene < 3; ++scene) {
        for (uint64_t period : {1, 2}) {
            Bundle bundle(context, scene, period);
            auto analysis = analyze(bundle.program);
            if (!analysis.error.empty() || !analysis.exactMinimum) {
                llvm::errs() << "arithmetic demand construction failed: " << analysis.error << "\n";
                return 1;
            }
            bool adjacent = true;
            for (int64_t n = 0; n <= 3; ++n) {
                for (int64_t g = 0; g <= 1; ++g) {
                    if (!checkCase(analysis, scene, n, g, adjacent)) {
                        llvm::errs() << "arithmetic demand oracle differs: scene=" << scene << " period=" << period
                                     << " trips=" << n << " guard=" << g << "\n";
                        return 1;
                    }
                    ++checked;
                }
            }
            if (analysis.adjacentLocalDemands != adjacent) {
                llvm::errs() << "arithmetic local-adjacency certificate differs\n";
                return 1;
            }
        }
    }
    llvm::outs() << name << " physical-conflict oracle passed " << checked << " parameter valuations\n";
    return 0;
}
} // namespace
int runArithmeticDemandChecks()
{
    return runChecks(fs::analyzeArithmeticDemands, "arithmetic demand");
}
int runGeneralArithmeticDemandChecks()
{
    return runChecks(fs::analyzeGeneralArithmeticDemands, "general arithmetic demand");
}
