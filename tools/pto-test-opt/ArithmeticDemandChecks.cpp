// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent finite physical-conflict DAGs for the symbolic demand backend.
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticSelectors.h"
#include "PTO/Transforms/FrontierSynch/Shared/HardwareProtection.h"
#include "PTO/Transforms/InsertSync/SyncCommon.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/IntegerImplicationCache.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/NumericUnionMergeCache.h"
#include "../../lib/PTO/Transforms/FrontierSynch/Arithmetic/NumericUnionDifferenceCache.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <algorithm>
#include <set>
#include <tuple>
namespace fs = mlir::pto::frontiersynch;
using namespace mlir;
namespace {
bool checkNumericalUnionMergeCache()
{
    fs::NumericUnionMergeCache cache;
    std::vector<std::set<int>> unions{{}, {1}, {1, 2}, {3}};
    unsigned builds = 0;
    auto merge = [&](std::size_t left, std::size_t right) {
        ++builds;
        auto merged = unions[left];
        merged.insert(unions[right].begin(), unions[right].end());
        auto found = std::find(unions.begin(), unions.end(), merged);
        if (found != unions.end()) { return static_cast<std::size_t>(found - unions.begin()); }
        unions.push_back(std::move(merged)); return unions.size() - 1;
    };
    std::map<std::pair<unsigned, unsigned>, std::size_t> tagged;
    // Compare the cached result with an independent pointwise union, including
    // reversed operands, empty unions, and a subsumed operand. Endpoint tags
    // deliberately vary while each numerical operand pair stays unchanged.
    for (const auto& operands : std::vector<std::pair<std::size_t, std::size_t>>{
            {1, 3}, {3, 1}, {0, 2}, {2, 1}, {2, 2}}) {
        const auto left = operands.first, right = operands.second;
        const auto before = builds;
        for (unsigned tag = 0; tag < 3; ++tag) {
            const auto id = cache.getOrBuild(left, right, [&] { return merge(left, right); });
            tagged[{tag, static_cast<unsigned>(left + right)}] = id;
            for (int point = -1; point <= 4; ++point) {
                const bool expected = unions[left].count(point) || unions[right].count(point);
                if (static_cast<bool>(unions[id].count(point)) != expected) {
                    llvm::errs() << "cached numerical union changes its integer points\n"; return false;
                }
            }
        }
        if (builds != before + 1) {
            llvm::errs() << "numerical union merge rebuilt under distinct endpoint tags\n"; return false;
        }
    }
    if (cache.cost().attempts != 15 || cache.cost().hits != 10 || cache.cost().builds != 5) {
        llvm::errs() << "numerical union merge diagnostics differ\n"; return false;
    }
    // Growing the external pool cannot invalidate a cached numerical identity.
    const auto oldCapacity = unions.capacity();
    while (unions.size() <= oldCapacity) { unions.push_back({-2}); }
    const auto before = builds;
    const auto again = cache.getOrBuild(1, 3, [&] { return merge(1, 3); });
    if (builds != before || unions[again] != std::set<int>({1, 3})) {
        llvm::errs() << "numerical union merge failed after pool growth\n"; return false;
    }
    cache.clear();
    if (cache.cost().attempts || cache.cost().hits || cache.cost().builds) { return false; }
    const auto refreshed = cache.getOrBuild(1, 3, [&] { return merge(1, 3); });
    if (builds != before + 1 || unions[refreshed] != std::set<int>({1, 3})) {
        llvm::errs() << "numerical union merge escaped its reducer context\n"; return false;
    }
    llvm::outs() << "numerical union merge cache: tags, reverse, empty, subsumption, growth, reset\n";
    return true;
}
bool checkNumericalUnionDifferenceCache()
{
    fs::NumericUnionDifferenceCache cache;
    std::vector<std::set<int>> unions{{}, {1}, {1, 2}, {3}};
    unsigned builds = 0;
    auto difference = [&](std::size_t left, std::size_t right) -> FailureOr<std::size_t> {
        ++builds;
        std::set<int> result;
        for (auto point : unions[left]) { if (!unions[right].count(point)) { result.insert(point); } }
        auto found = std::find(unions.begin(), unions.end(), result);
        if (found != unions.end()) { return static_cast<std::size_t>(found - unions.begin()); }
        unions.push_back(std::move(result)); return unions.size() - 1;
    };
    for (auto operands : std::vector<std::pair<std::size_t, std::size_t>>{
            {2, 1}, {1, 2}, {2, 2}, {0, 2}, {2, 0}}) {
        const auto left = operands.first;
        const auto right = operands.second;
        const auto before = builds;
        for (unsigned endpointTag = 0; endpointTag < 3; ++endpointTag) {
            auto id = cache.getOrBuild(left, right, 1, [&] { return difference(left, right); });
            if (failed(id)) { return false; }
            for (int point = -1; point <= 4; ++point) {
                if (static_cast<bool>(unions[*id].count(point)) !=
                    (unions[left].count(point) && !unions[right].count(point))) { return false; }
            }
        }
        if (builds != before + 1) { return false; }
    }
    if (cache.cost().attempts != 15 || cache.cost().hits != 10 || cache.cost().builds != 5) { return false; }
    // Distinct schemas and ordered operand pairs never share an answer.
    auto schema = cache.getOrBuild(2, 1, 2, [&] { return difference(2, 1); });
    if (failed(schema) || builds != 6) { return false; }
    const auto capacity = unions.capacity();
    while (unions.size() <= capacity) { unions.push_back({-2}); }
    auto afterGrowth = cache.getOrBuild(2, 1, 1, [&] { return difference(2, 1); });
    if (failed(afterGrowth) || unions[*afterGrowth] != std::set<int>({2}) || builds != 6) { return false; }
    unsigned retries = 0;
    auto failOnce = [&]() -> FailureOr<std::size_t> {
        if (!retries++) { return failure(); }
        return std::size_t(0); // Successful empty union must be cached.
    };
    if (succeeded(cache.getOrBuild(3, 3, 1, failOnce)) ||
        failed(cache.getOrBuild(3, 3, 1, failOnce)) ||
        failed(cache.getOrBuild(3, 3, 1, failOnce)) || retries != 2) { return false; }
    cache.clear();
    if (cache.cost().attempts || cache.cost().hits || cache.cost().builds ||
        failed(cache.getOrBuild(2, 1, 1, [&] { return difference(2, 1); })) || builds != 7) { return false; }
    llvm::outs() << "numerical union difference cache: ordered, tags, empty, failure, schema, growth, reset\n";
    return true;
}
bool checkLatticeImplicationCertificates()
{
    using B = fs::BoundInteger;
    std::vector<fs::IntegerSystem> systems;
    auto add = [&](unsigned dimensions, std::vector<fs::IntegerConstraint> rows,
                   std::vector<fs::IntegerCongruence> congruences = {}) {
        auto value = fs::IntegerSystem::create(dimensions, rows, congruences);
        if (failed(value)) { return std::size_t(-1); }
        systems.push_back(std::move(*value)); return systems.size() - 1;
    };
    auto even = add(1, {{{B(-1)}, B(0)}, {{B(1)}, B(1)}}, {{{B(1)}, B(0), B(2)}});
    auto zero = add(1, {{{B(1)}, B(0)}});
    auto odd = add(1, {{{B(-1)}, B(0)}, {{B(1)}, B(2)}}, {{{B(1)}, B(1), B(2)}});
    auto one = add(1, {{{B(1)}, B(1)}});
    auto negative = add(1, {{{B(-1)}, B(5)}, {{B(1)}, B(-2)}}, {{{B(1)}, B(-1), B(2)}});
    auto minusThree = add(1, {{{B(1)}, B(-3)}});
    auto minusCoefficient = add(1, {{{B(-1)}, B(4)}, {{B(1)}, B(3)}}, {{{B(-1)}, B(1), B(3)}});
    auto two = add(1, {{{B(1)}, B(2)}});
    auto coupled = add(2, {{{B(-1), B(0)}, B(0)}, {{B(1), B(0)}, B(2)},
        {{B(0), B(-1)}, B(0)}, {{B(0), B(1)}, B(2)}, {{B(1), B(1)}, B(2)}},
        {{{B(1), B(0)}, B(0), B(2)}, {{B(0), B(1)}, B(1), B(2)}});
    auto coupledTarget = add(2, {{{B(1), B(1)}, B(1)}});
    B wide(1); for (unsigned i = 0; i < 70; ++i) { wide *= B(2); }
    auto wideSource = add(1, {{{B(-1)}, B(0)}, {{B(1)}, wide}}, {{{B(1)}, wide - 1, wide}});
    auto wideTarget = add(1, {{{B(1)}, wide - 1}});
    auto extra = add(2, {{{B(-1), B(0)}, B(0)}, {{B(1), B(0)}, B(1)},
        {{B(0), B(-1)}, B(0)}, {{B(0), B(1)}, B(2)}},
        {{{B(1), B(0)}, B(0), B(2)}, {{B(1), B(1)}, B(0), B(3)}});
    auto xZero = add(2, {{{B(1), B(0)}, B(0)}});
    auto empty = add(1, {{{B(-1)}, B(-1)}, {{B(1)}, B(1)}}, {{{B(1)}, B(0), B(2)}});
    auto unbounded = add(1, {}, {{{B(1)}, B(0), B(2)}});
    auto nonUnary = add(2, {{{B(-1), B(0)}, B(0)}, {{B(1), B(0)}, B(1)},
        {{B(0), B(-1)}, B(0)}, {{B(0), B(1)}, B(0)}}, {{{B(1), B(1)}, B(0), B(2)}});
    auto newCongruence = add(1, {}, {{{B(1)}, B(0), B(3)}});
    auto unsupported = add(1, {{{B(-1)}, B(0)}, {{B(1)}, B(2)}}, {{{B(2)}, B(0), B(5)}});
    fs::IntegerImplicationCache cache(systems);
    // The designated positives require the stronger lattice proof, while
    // unhandled congruences keep the original exact fallback available.
    for (auto [a, b, expected, requireLattice] :
        std::vector<std::tuple<std::size_t, std::size_t, bool, bool>>{
            {even, zero, true, true}, {odd, one, true, true}, {negative, minusThree, true, true},
            {minusCoefficient, two, true, true}, {coupled, coupledTarget, true, true},
            {wideSource, wideTarget, true, true}, {extra, xZero, true, true}, {empty, zero, true, true},
            {unbounded, zero, false, false}, {nonUnary, xZero, true, false},
            {even, newCongruence, true, false}, {unsupported, zero, true, false}, {odd, zero, false, false}}) {
        if (a == std::size_t(-1) || b == std::size_t(-1)) { return false; }
        const auto before = cache.cost().latticeProofs;
        if (cache.isSubsetOf(a, b) != expected || systems[a].isSubsetOf(systems[b]) != expected ||
            (requireLattice && cache.cost().latticeProofs == before)) {
            llvm::errs() << "lattice implication differs or bypassed its designated proof\n"; return false;
        }
    }
    // An already queried original direction is shared across different target
    // bounds. Its transformed objective includes the source-dependent shift.
    auto negativeTarget = add(1, {{{B(1)}, B(-1)}});
    // Isolate objective reuse from the verified counterexample samples left
    // by the fallback cases above; those may settle a negative query sooner.
    fs::IntegerImplicationCache objectiveCache(systems);
    if (!objectiveCache.isSubsetOf(even, zero)) { return false; }
    const auto beforeHits = objectiveCache.cost().latticeHits;
    if (objectiveCache.isSubsetOf(even, negativeTarget) || objectiveCache.cost().latticeHits == beforeHits ||
        !cache.cost().latticeEmptyProofs || !cache.cost().exactChecks) {
        llvm::errs() << "lattice objective caching/empty/exact paths were not exercised\n"; return false;
    }
    const auto capacity = systems.capacity();
    while (systems.size() <= capacity) { add(1, {{{B(1)}, B(7)}}); }
    if (!cache.isSubsetOf(even, zero) || !cache.isSubsetOf(wideSource, wideTarget)) {
        llvm::errs() << "lattice proof cache failed after numerical pool growth\n"; return false;
    }
    llvm::outs() << "lattice implication: signs, offsets, coupled gcd, wide coefficients, fallback, growth\n";
    return true;
}
bool checkOptionalInclusionCertificates()
{
    using B = fs::BoundInteger;
    auto source = fs::IntegerSystem::create(2,
        {{{B(-1), B(0)}, B(0)}, {{B(1), B(0)}, B(1)},
         {{B(0), B(-1)}, B(0)}, {{B(0), B(1)}, B(0)}},
        {{{B(1), B(1)}, B(0), B(2)}});
    auto target = fs::IntegerSystem::create(2, {{{B(1), B(0)}, B(0)}});
    if (failed(source) || failed(target)) { return false; }
    std::vector<fs::IntegerSystem> systems{*source, *target};
    fs::IntegerImplicationCache cache(systems);
    // x+y even, y=0 and 0<=x<=1 imply x=0, but the implemented
    // sufficient proofs intentionally ignore this coupled modular fact.
    if (cache.hasInclusionCertificate(0, 1) || cache.hasInclusionCertificate(0, 1) ||
        cache.cost().exactChecks || cache.cost().witnessRefutations ||
        !cache.isSubsetOf(0, 1) || !cache.cost().exactChecks || !cache.hasInclusionCertificate(0, 1) ||
        !cache.hasInclusionCertificate(0, 0)) {
        llvm::errs() << "unproved union inclusion blocked the independent exact query\n"; return false;
    }
    llvm::outs() << "optional inclusion certificate preserves unproved pieces and exact queries\n";
    return true;
}
bool checkImplicationCertificates()
{
    using B = fs::BoundInteger;
    std::vector<fs::IntegerSystem> systems;
    auto add = [&](unsigned dimensions, std::vector<fs::IntegerConstraint> rows,
                   std::vector<fs::IntegerCongruence> congruences = {}) {
        auto system = fs::IntegerSystem::create(dimensions, rows, congruences);
        if (failed(system)) { return std::size_t(-1); }
        systems.push_back(std::move(*system)); return systems.size() - 1;
    };
    // 2*x+y<=1, x>=0, y>=0 has rational max x=1/2,
    // but integer max x=0. The target's coefficient row is absent.
    auto fractional = add(2, {{{B(2), B(1)}, B(1)}, {{B(-1), B(0)}, B(0)}, {{B(0), B(-1)}, B(0)}});
    auto zero = add(2, {{{B(1), B(0)}, B(0)}});
    auto negative = add(2, {{{B(1), B(0)}, B(-1)}});
    auto unbounded = add(2, {{{B(-1), B(0)}, B(0)}});
    auto even = add(1, {{{B(-1)}, B(0)}, {{B(1)}, B(1)}}, {{{B(1)}, B(0), B(2)}});
    auto scalarZero = add(1, {{{B(1)}, B(0)}});
    // No sufficient LP certificate infers this new target congruence.
    auto multipleOfThree = add(1, {}, {{{B(1)}, B(0), B(3)}});
    auto odd = add(1, {}, {{{B(1)}, B(1), B(2)}});
    auto empty = add(1, {{{B(0)}, B(-1)}});
    B wide(1); for (unsigned i = 0; i < 70; ++i) { wide *= B(2); }
    auto wideSource = add(1, {{{B(-1)}, -wide}, {{B(1)}, wide + 1}});
    auto wideTarget = add(1, {{{B(1)}, wide}});
    fs::IntegerImplicationCache cache(systems);
    const std::vector<std::tuple<std::size_t, std::size_t, bool>> cases{
        {fractional, zero, true}, {fractional, negative, false}, {unbounded, zero, false},
        {even, scalarZero, true}, {even, multipleOfThree, true}, {even, odd, false}, {empty, scalarZero, true},
        {wideSource, wideTarget, false}};
    for (auto [a, b, expected] : cases) {
        if (cache.isSubsetOf(a, b) != expected || cache.isSubsetOf(a, b) != systems[a].isSubsetOf(systems[b])) {
            llvm::errs() << "integer implication certificate differs\n"; return false;
        }
    }
    // The cache owns numerical IDs, not pointers into vector storage. Grow
    // the vector after real queries, then exercise both cached and new IDs.
    const auto oldCapacity = systems.capacity();
    std::size_t appended = 0;
    while (systems.size() <= oldCapacity) {
        appended = add(2, {{{B(1), B(0)}, B(1)}});
        if (appended == std::size_t(-1)) { return false; }
    }
    if (systems.capacity() <= oldCapacity || !cache.isSubsetOf(fractional, zero) ||
        !cache.isSubsetOf(fractional, appended) || cache.isSubsetOf(appended, zero) ||
        cache.isSubsetOf(appended, zero) != systems[appended].isSubsetOf(systems[zero])) {
        llvm::errs() << "integer implication cache failed after storage growth\n"; return false;
    }
    if (!cache.cost().hits || !cache.cost().rationalProofs || !cache.cost().witnessRefutations ||
        !cache.cost().exactChecks) {
        llvm::errs() << "integer implication certificate path was not exercised\n"; return false;
    }
    llvm::outs() << "integer implication certificates: rational, integer witness, exact fallback, wide bounds\n";
    return true;
}
struct Occurrence {
    unsigned site;
    int64_t iteration;
    uint32_t pipe;
    std::set<int64_t> reads, writes;
};
std::vector<uint32_t> pipes(unsigned scene)
{
    if (scene == 3) { return {0, 1, 2, 0}; }
    if (scene == 4 || scene == 5) { return {0, 1, 1, 2, 0}; }
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
        } else if (scene == 3) {
            // Completion(0) -> Start(3) needs three handoffs on three pipes;
            // same-pipe native start/start and completion/completion edges
            // cannot bypass this final-round completion prerequisite.
            result.push_back({0, i, 0, {}, {0}});
            result.push_back({1, i, 1, {0}, {1}});
            result.push_back({2, i, 2, {1}, {2}});
            result.push_back({3, i, 0, {2}, {}});
        } else if (scene == 4 || scene == 5) {
            // Scene4: the short path holds only at n=1, while the longer
            // path holds at every nonempty trip count. Scene5 reverses the
            // domains, exercising both directions of frontier subsumption.
            const bool special = trips == 1 && guard == 0;
            result.push_back({0, i, 0, {}, {0}});
            if (scene == 5 || special) { result.push_back({1, i, 1, {0}, {1}}); }
            if (scene == 4 || special) {
                result.push_back({2, i, 1, {0}, {2}});
                result.push_back({3, i, 2, {2}, {3}});
            }
            result.push_back({4, i, 0, {1, 3}, {}});
        } else if (scene == 6) {
            for (unsigned site = 0; site < 3; ++site) { result.push_back({site, i, 0, {}, {0}}); }
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
        if (scene >= 3 && period == 1) { broadenFirstVisit(scene); }
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
    void broadenFirstVisit(unsigned scene)
    {
        // Every first-visit tuple for trips=1 also occurs at trips=2/3.
        // Append its exact interval hull after the individual points, forcing
        // subsumption to replace an earlier numerical piece without changing
        // the concrete relation checked by the independent graph oracle.
        for (auto& relation : program.primitives.relations) {
            auto conditional = [&](auto site) {
                return site && ((scene == 4 && *site == 1) ||
                    (scene == 5 && (*site == 2 || *site == 3)));
            };
            // Conditional path domains must stay at n=1. Broaden only the
            // original first-visit tuples present for every n=1,2,3.
            if (conditional(relation.sourceSite) || conditional(relation.targetSite)) { continue; }
            const auto original = relation.pieces.size();
            for (std::size_t i = 0; i < original; ++i) {
                const auto& piece = relation.pieces[i];
                auto constraints = piece.system.getConstraints();
                const auto nRow = relation.dimensions;
                auto n = getAffineSymbolExpr(0, &context);
                if (constraints[nRow] != n - 1) { continue; }
                SmallVector<AffineExpr> rows(constraints.begin(), constraints.end());
                SmallVector<bool> equalities(rows.size(), true);
                equalities[nRow] = false;
                rows.push_back(3 - n); equalities.push_back(false);
                relation.pieces.push_back({IntegerSet::get(relation.dimensions, 2, rows, equalities),
                    piece.residues});
            }
        }
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
bool checkProtectedPhysicalJoinReuse()
{
    MLIRContext context;
    context.disableMultithreading();
    Bundle bundle(context, 6, 1);
    std::vector<std::unique_ptr<mlir::pto::CompoundInstanceElement>> phases;
    fs::StructuredProtection protection;
    for (unsigned site = 0; site < 3; ++site) {
        phases.push_back(std::make_unique<mlir::pto::CompoundInstanceElement>(site,
            SmallVector<const mlir::pto::BaseMemInfo*>{}, SmallVector<const mlir::pto::BaseMemInfo*>{},
            mlir::pto::PipelineType::PIPE_M, OperationName("pto.test", &context)));
        bundle.program.sites[site].phase = phases.back().get();
        protection.facts[phases.back().get()] = {site < 2 ? uint64_t(7) : uint64_t(8), nullptr};
    }
    for (auto& relation : bundle.program.primitives.relations) {
        if (relation.kind == fs::PrimitiveKind::Reads || relation.kind == fs::PrimitiveKind::Writes) {
            relation.storageSpace = mlir::pto::AddressSpace::ACC;
        }
    }
    auto analysis = fs::analyzeGeneralArithmeticDemandsWithProtection(bundle.program, protection);
    if (!analysis.error.empty() || !analysis.exactMinimum || !analysis.cost.physicalJoinCacheHits) {
        llvm::errs() << "protected numerical access join did not construct and reuse\n"; return false;
    }
    // All three writes have identical numerical access relations. Only 0->1
    // shares a hardware protection group. Protection must be applied after
    // numerical join reuse, preserving the other two completion/start hazards.
    for (unsigned a = 0; a < 3; ++a) {
        for (unsigned b = a + 1; b < 3; ++b) {
            const bool expected = !(a == 0 && b == 1);
            if (member(analysis.generators, 1, a, 0, 1, b, 0, 0, 1, 0) != expected) {
                llvm::errs() << "numerical access join reused endpoint-specific protection\n"; return false;
            }
        }
    }
    // Independent event DAG at one visit: native issue/issue, completion/
    // completion and each instruction's issue/completion; add only the two
    // nonprotected write hazards, then compute reachability and covers.
    auto occurrences = execution(6, 1, 0);
    auto reach = closure(occurrences, false);
    reach[1][4] = reach[3][4] = true;
    for (unsigned via = 0; via < 6; ++via) {
        for (unsigned a = 0; a < 6; ++a) {
            for (unsigned b = 0; b < 6; ++b) { reach[a][b] = reach[a][b] || (reach[a][via] && reach[via][b]); }
        }
    }
    for (unsigned a = 0; a < 3; ++a) {
        for (unsigned b = 0; b < 3; ++b) {
            for (unsigned ea = 0; ea < 2; ++ea) {
                for (unsigned eb = 0; eb < 2; ++eb) {
                    const unsigned source = 2 * a + ea, target = 2 * b + eb;
                    bool cover = a < b && ea == 1 && eb == 0 && !(a == 0 && b == 1);
                    for (unsigned via = 0; via < 6; ++via) {
                        if (via != source && via != target && reach[source][via] && reach[via][target]) {
                            cover = false;
                        }
                    }
                    if (member(analysis.requiredOrder, 1, a, 0, ea, b, 0, eb, 1, 0) !=
                            (source != target && reach[source][target]) ||
                        member(analysis.minimumDemands, 1, a, 0, ea, b, 0, eb, 1, 0) != cover) {
                        llvm::errs() << "protected numerical access join differs from independent event DAG\n";
                        return false;
                    }
                }
            }
        }
    }
    llvm::outs() << "numerical access join reuse preserves distinct hardware protection groups\n";
    return true;
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
bool checkLocalAdjacencyQueries()
{
    MLIRContext context;
    context.disableMultithreading();
    // Isolate empty execution, cross-pipe-only demands and a same-pipe demand
    // with an intervening same-pipe payload. Every expected event/cover is
    // still derived by the independent finite physical DAG above.
    for (const auto& item : std::vector<std::tuple<unsigned, int64_t, bool>>{
            {2, 0, true}, {0, 1, true}, {2, 1, false}}) {
        const auto scene = std::get<0>(item);
        const auto trips = std::get<1>(item);
        const auto expectedAdjacent = std::get<2>(item);
        Bundle bundle(context, scene, 1);
        for (auto& relation : bundle.program.primitives.relations) {
            relation.pieces.erase(std::remove_if(relation.pieces.begin(), relation.pieces.end(),
                [&](const auto& piece) {
                    const auto rows = piece.system.getConstraints();
                    return rows[relation.dimensions] != getAffineSymbolExpr(0, &context) - trips ||
                           rows[relation.dimensions + 1] != getAffineSymbolExpr(1, &context);
                }), relation.pieces.end());
        }
        bundle.program.recognition = fs::recognizeArithmetic(bundle.program.primitives, {3, 4, 1, 1});
        auto analysis = fs::analyzeGeneralArithmeticDemands(bundle.program);
        bool oracleAdjacent = true;
        if (!analysis.error.empty() || !analysis.exactMinimum ||
            !checkCase(analysis, scene, trips, 0, oracleAdjacent) ||
            oracleAdjacent != expectedAdjacent || analysis.adjacentLocalDemands != expectedAdjacent ||
            (trips == 0 && !analysis.minimumDemands.empty()) ||
            (trips != 0 && analysis.minimumDemands.empty())) {
            llvm::errs() << "local-adjacency boolean query differs from independent DAG\n"; return false;
        }
    }
    llvm::outs() << "local-adjacency queries: empty, cross-pipe-only, nonadjacent same-pipe\n";
    return true;
}
template<class Analyze>
int runChecks(Analyze analyze, llvm::StringRef name)
{
    MLIRContext context;
    context.disableMultithreading();
    unsigned checked = 0;
    for (unsigned scene = 0; scene < 6; ++scene) {
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
    if (!checkNumericalUnionMergeCache() || !checkNumericalUnionDifferenceCache() || !checkImplicationCertificates() ||
        !checkLatticeImplicationCertificates() || !checkOptionalInclusionCertificates() ||
        !checkProtectedPhysicalJoinReuse() || !checkLocalAdjacencyQueries()) { return 1; }
    bool reused = false, differenceReused = false, physicalReused = false, emptyPhysicalReused = false;
    bool importReused = false;
    auto analyze = [&](const fs::ArithmeticProgram& program) {
        auto result = fs::analyzeGeneralArithmeticDemands(program);
        reused |= result.cost.unionMergeCacheHits > 0;
        differenceReused |= result.cost.unionDifferenceCacheHits > 0;
        physicalReused |= result.cost.physicalJoinCacheHits > 0;
        emptyPhysicalReused |= result.cost.physicalJoinEmptyHits > 0;
        const auto& cost = result.cost;
        importReused |= cost.importQueryHits > 0;
        if (cost.importQueryRequests != cost.importQueryBuilds + cost.importQueryHits) {
            result.error = "arithmetic import counters disagree with actual constructions";
        }
        if (cost.unionDifferenceAttempts != cost.unionDifferenceBuilds + cost.unionDifferenceCacheHits) {
            result.error = "numerical union difference counters disagree with actual constructions";
        }
        if (cost.physicalJoinAttempts != cost.physicalJoinBuilds + cost.physicalJoinCacheHits ||
            cost.physicalJoinRemaps != 2 * cost.physicalJoinBuilds ||
            cost.physicalJoinEmptyHits > cost.physicalJoinCacheHits) {
            result.error = "physical access join counters disagree with actual constructions";
        }
        return result;
    };
    const auto status = runChecks(analyze, "general arithmetic demand");
    if (!status && !importReused) {
        llvm::errs() << "independent physical DAG oracle did not exercise arithmetic import reuse\n"; return 1;
    }
    if (!status && !reused) {
        llvm::errs() << "general arithmetic oracle did not exercise numerical union reuse\n"; return 1;
    }
    if (!status && !differenceReused) {
        llvm::errs() << "independent physical DAG oracle did not exercise numerical difference reuse\n"; return 1;
    }
    if (!status && !physicalReused) {
        llvm::errs() << "independent physical DAG oracle did not exercise numerical access join reuse\n"; return 1;
    }
    if (!status && !emptyPhysicalReused) {
        llvm::errs() << "independent physical DAG oracle did not reuse an empty access join\n"; return 1;
    }
    return status;
}
