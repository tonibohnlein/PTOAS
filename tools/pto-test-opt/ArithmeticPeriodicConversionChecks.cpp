// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticPeriodicConversion.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "PTO/Transforms/FrontierSynch/GuardedPeriodicInsertion.h"
#include "PTO/Transforms/FrontierSynch/HardwareProtection.h"
#include "SyncLogicalInsertionChecks.h"
#include "mlir/IR/Verifier.h"
#include <set>
#include <vector>
namespace fs = mlir::pto::frontiersynch;
namespace {
using B = fs::BoundInteger;
using Graph = std::vector<std::vector<bool>>;
void close(Graph& g)
{
    for (std::size_t z = 0; z < g.size(); ++z) {
        for (std::size_t a = 0; a < g.size(); ++a) {
            if (!g[a][z]) { continue; }
            for (std::size_t b = 0; b < g.size(); ++b) { g[a][b] = g[a][b] || g[z][b]; }
        }
    }
}
fs::IntegerSystem relation(unsigned dimensions, std::vector<fs::IntegerConstraint> constraints)
{
    return *fs::IntegerSystem::create(dimensions, constraints);
}
bool numerical(unsigned pattern, uint64_t& comparisons)
{
    fs::ArithmeticPeriodicInput input;
    input.expressions = std::make_shared<fs::RegionExpressions>();
    auto& e = *input.expressions;
    constexpr unsigned m = 3;
    for (unsigned s = 0; s < m; ++s) { input.payloads.push_back({s % 2, e.boolean(true)}); }
    struct Interval { unsigned a, b, lo, hi; };
    std::vector<Interval> intervals;
    for (unsigned a = 0; a < m; ++a) {
        for (unsigned b = 0; b < m; ++b) {
            if ((a + 2 * b + pattern) % 3 == 0) { continue; }
            const unsigned lo = (a + pattern) % 3, hi = lo + (b + pattern) % 3;
            intervals.push_back({a, b, lo, hi});
            auto system = relation(2, {{{B(1), B(-1)}, B(-int64_t(lo))},
                                       {{B(-1), B(1)}, B(hi)}});
            input.pieces.push_back({a, b, system, {}, {}, {}, false, input.pieces.size()});
        }
    }
    auto converted = fs::convertArithmeticPeriodicIntervals(input);
    if (converted.status != fs::ArithmeticPeriodicStatus::Applicable || !converted.guarded) { return false; }
    for (unsigned n = 0; n <= 5 * m; ++n) {
        Graph full(2 * n, std::vector<bool>(2 * n)), reduced = full;
        auto native = [&](Graph& g) {
            for (unsigned a = 0; a < n; ++a) {
                g[2*a][2*a+1] = true;
                for (unsigned b = a + 1; b < n; ++b) {
                    if (input.payloads[a%m].pipe == input.payloads[b%m].pipe) {
                        g[2*a][2*b] = g[2*a+1][2*b+1] = true;
                    }
                }
            }
        };
        native(full); native(reduced);
        for (unsigned a = 0; a < n; ++a) {
            for (unsigned b = a + 1; b < n; ++b) {
                for (const auto& v : intervals) {
                    const auto d = b/m-a/m;
                    if (a%m == v.a && b%m == v.b && v.lo <= d && d <= v.hi) {
                        full[2*a+1][2*b] = true;
                    }
                }
            }
        }
        for (std::size_t r = 0; r < converted.generators.size(); ++r) {
            if (e.constantValue(converted.guarded->retained[r]) != 1) { continue; }
            auto edge = converted.generators[r];
            auto distance = e.constantValue(edge.displacement);
            if (!distance) { return false; }
            for (unsigned a = edge.source; a < n; a += m) {
                auto b = (a/m + *distance)*m + edge.target;
                if (b < n) { reduced[2*a+1][2*b] = true; }
            }
        }
        close(full); close(reduced); ++comparisons;
        if (full != reduced) { return false; }
    }
    return true;
}
bool parameterBounds()
{
    for (int64_t theta = -3; theta < 13; ++theta) {
        fs::ArithmeticPeriodicInput input;
        input.expressions = std::make_shared<fs::RegionExpressions>();
        auto& e = *input.expressions;
        input.payloads = {{0, e.boolean(true)}, {1, e.boolean(true)}};
        input.parameters = {e.constant(static_cast<uint64_t>(theta))};
        auto system = relation(3, {{{B(2), B(-2), B(1)}, B(-1)},
                                   {{B(-3), B(3), B(-1)}, B(9)}});
        input.pieces.push_back({0, 1, system, {0}, {}, {}, false, 0});
        auto result = fs::convertArithmeticPeriodicIntervals(input);
        if (result.status != fs::ArithmeticPeriodicStatus::Applicable || !result.guarded ||
            result.generators.size() != 1) { return false; }
        const int64_t lower = std::max<int64_t>(0, theta < -1 ? (theta + 1) / 2 : (theta + 2) / 2);
        const int64_t upper = (theta + 9) / 3;
        const auto& edge = result.generators.front();
        if (e.constantValue(edge.active) != uint64_t(lower <= upper) ||
            (lower <= upper && e.constantValue(edge.displacement) != uint64_t(lower))) { return false; }
    }
    return true;
}
bool symbolicBounds()
{
    mlir::MLIRContext context;
    mlir::Block symbols;
    for (uint64_t period : {1U, 2U}) {
        fs::ArithmeticPeriodicInput input;
        input.expressions = std::make_shared<fs::RegionExpressions>();
        auto& e = *input.expressions;
        auto value = symbols.addArgument(mlir::IndexType::get(&context), mlir::UnknownLoc::get(&context));
        auto parameter = e.input(value);
        input.parameters = {parameter}; input.parameterPeriod = period;
        input.payloads = {{0, e.boolean(true)}, {1, e.boolean(true)}};
        auto system = relation(3, {{{B(2),B(-2),B(1)},B(-1)}, {{B(-3),B(3),B(-1)},B(9)},
                                  {{B(0),B(0),B(1)},B(30)}, {{B(0),B(0),B(-1)},B(10)}});
        input.pieces.push_back({0, 1, system, {period-1}, {}, {}, false, 0});
        auto result = fs::convertArithmeticPeriodicIntervals(input);
        if (!result.guarded || result.generators.size() != 1) { return false; }
        for (int64_t raw = -6; raw <= 25; ++raw) {
            const auto p = static_cast<int64_t>(period);
            const auto q = raw / p - (raw % p < 0 ? 1 : 0);
            const auto residue = raw - q*p;
            const auto lower = std::max<int64_t>(0, q < -1 ? (q+1)/2 : (q+2)/2);
            const auto upper = (q+9)/3;
            fs::RegionExpressions::Substitution bind({{parameter, e.constant(static_cast<uint64_t>(raw))}});
            const auto& edge = result.generators.front();
            const auto active = e.constantValue(e.substitute(edge.active, bind));
            const bool expected = residue == p-1 && lower <= upper;
            if (active != uint64_t(expected)) { return false; }
            if (expected && e.constantValue(e.substitute(edge.displacement, bind)) != uint64_t(lower)) {
                return false;
            }
        }
    }
    return true;
}
bool rejectedAndEmpty()
{
    fs::ArithmeticPeriodicInput input;
    input.expressions = std::make_shared<fs::RegionExpressions>();
    input.payloads = {{0, input.expressions->boolean(true)}, {1, input.expressions->boolean(true)}};
    input.pieces.push_back({0, 1, relation(2, {{{B(1),B(0)},B(3)}}), {}, {}, {}, false, 0});
    if (fs::convertArithmeticPeriodicIntervals(input).status != fs::ArithmeticPeriodicStatus::NotDistanceIntervals) {
        return false;
    }
    input.pieces[0].sourceDomains = {relation(1, {{{B(1)}, B(3)}})};
    if (fs::convertArithmeticPeriodicIntervals(input).status != fs::ArithmeticPeriodicStatus::Applicable) {
        return false;
    }
    input.pieces[0].relation = relation(2, {{{B(0),B(0)},B(-1)}});
    auto empty = fs::convertArithmeticPeriodicIntervals(input);
    if (empty.status != fs::ArithmeticPeriodicStatus::Applicable || !empty.guarded) { return false; }
    for (auto retained : empty.guarded->retained) {
        if (input.expressions->constantValue(retained) != 0) { return false; }
    }
    const B huge = B(INT64_MAX) * B(INT64_MAX);
    input.pieces[0].relation = relation(2, {{{B(1),B(-1)},-huge}});
    const auto oldSize = input.expressions->size();
    auto wide = fs::convertArithmeticPeriodicIntervals(input);
    return wide.status == fs::ArithmeticPeriodicStatus::Applicable && !wide.exportError.empty() &&
        !wide.guarded && wide.generators.empty() && input.expressions->size() == oldSize &&
        input.expressions->constructionError().empty();
}
}
int runArithmeticPeriodicConversionChecks()
{
    uint64_t comparisons = 0;
    for (unsigned pattern = 0; pattern < 12; ++pattern) {
        if (!numerical(pattern, comparisons)) {
            llvm::errs() << "arithmetic periodic interval closure failed pattern " << pattern << "\n"; return 1;
        }
    }
    if (!parameterBounds() || !symbolicBounds() || !rejectedAndEmpty()) {
        llvm::errs() << "arithmetic periodic cutoff/empty checks failed\n"; return 1;
    }
    llvm::outs() << "arithmetic periodic: " << comparisons << " finite-prefix closure comparisons passed\n";
    return 0;
}

int runArithmeticPeriodicInputChecks(mlir::func::FuncOp function, const mlir::pto::SyncInput& input)
{
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return 1; }
    auto program = fs::recognizeArithmeticProgram(function, index, input, input.accesses(), {8,8,2,4096});
    auto protection = fs::structuredProtection(input.accesses());
    auto stage = fs::analyzeGeneralArithmeticGenerators(program, &protection);
    auto converted = fs::convertArithmeticPeriodicProgram(program, stage);
    if (!converted.conversion.guarded) {
        llvm::errs() << "periodic importer: " << stage.analysis().error << " "
                     << converted.conversion.diagnostic << " " << converted.conversion.exportError << "\n";
        return 1;
    }
    const auto generatorCost = stage.analysis().cost.primitivePieces;
    auto resumed = fs::completeGeneralArithmeticDemands(std::move(stage));
    auto ordinary = fs::analyzeGeneralArithmeticDemandsWithProtection(program, protection);
    if (!resumed.error.empty() || !ordinary.error.empty() || !resumed.exactMinimum ||
        resumed.cost.primitivePieces != generatorCost ||
        !std::equal(resumed.minimumDemands.begin(), resumed.minimumDemands.end(),
                    ordinary.minimumDemands.begin(), ordinary.minimumDemands.end(),
                    [](const auto& a, const auto& b) {
                        return !(a.first < b.first) && !(b.first < a.first) && a.second == b.second;
                    })) {
        llvm::errs() << "arithmetic generator continuation differs\n"; return 1;
    }
    std::vector<uint64_t> residues;
    for (const auto& site : converted.sites) { residues.push_back(site.residue); }
    auto& result = converted.conversion;
    fs::GuardedPeriodicEndpointInput endpoints{converted.loop, result.expressions, converted.phases,
        result.payloads, result.generators, &*result.guarded, converted.period, residues};
    std::string error;
    auto prepared = fs::prepareGuardedPeriodicEndpoints(function, endpoints, error);
    if (failed(prepared) || failed(fs::insertLogicalSynchronization(function, **prepared)) ||
        failed(verify(function))) {
        llvm::errs() << "arithmetic periodic endpoint preparation: " << error << "\n"; return 1;
    }
    auto trace = traceStructuredLogicalInsertion(function, input.instructions());
    if (trace.getString("error").value_or("missing") != "") {
        llvm::errs() << llvm::json::Value(std::move(trace)) << "\n"; return 1;
    }
    std::set<std::pair<int64_t,int64_t>> pending;
    uint64_t published = 0;
    auto* events = trace.getArray("events");
    if (!events) { return 1; }
    for (auto& value : *events) {
        auto* event = value.getAsObject();
        if (!event) { return 1; }
        const auto kind = event->getString("kind");
        if (kind != "set" && kind != "wait") { continue; }
        auto record = event->getInteger("record"), ordinal = event->getInteger("source_ordinal");
        auto gap = event->getInteger("gap");
        if (!record || !ordinal || !gap || *record < 0 || uint64_t(*record) >= result.generators.size()) { return 1; }
        const auto& edge = result.generators[*record];
        const auto distance = result.expressions->constantValue(edge.displacement);
        const auto position = *gap - (kind == "set" ? 1 : 0);
        const auto type = kind == "set" ? edge.source : edge.target;
        if (!distance || position < 0 || uint64_t(position) % result.payloads.size() != type ||
            uint64_t(position) / result.payloads.size() != uint64_t(*ordinal) + (kind == "set" ? 0 : *distance)) {
            return 1;
        }
        const auto key = std::make_pair(*record,*ordinal);
        if (kind == "set") {
            ++published;
            if (!pending.insert(key).second) { return 1; }
        }
        else if (!pending.erase(key)) { return 1; }
    }
    // The fixture alternates a load and a store on one reused tile: all
    // adjacent payload pairs conflict across pipes. Check coverage as well
    // as pairing, so an empty or partially omitted plan cannot pass.
    const auto payloadCount = trace.getInteger("payloads");
    if (!payloadCount || !pending.empty() ||
        published != (*payloadCount ? uint64_t(*payloadCount - 1) : 0)) { return 1; }
    llvm::outs() << "arithmetic periodic importer/continuation/paired endpoints passed\n";
    return 0;
}
