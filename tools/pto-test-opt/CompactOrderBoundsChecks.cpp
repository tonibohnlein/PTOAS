// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Algebraic graph bounds on an actual borrowed fixed occurrence frame. Synthetic
// lower/upper pairs exercise certification; they do not assert extra input facts.
#include "PTO/Transforms/FrontierSynch/CompactOrderBounds.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
namespace {
using Expr = fs::RegionExpressions::Id;
using Kind = fs::PeriodicEventKind;
bool evaluate(fs::RegionExpressions& arena, Expr expression, Expr count, uint64_t trips, bool expected)
{
    const std::pair<Expr, Expr> binding{count, arena.constant(trips)};
    fs::RegionExpressions::Substitution substitution(binding);
    return arena.constantValue(arena.substitute(expression, substitution)) == uint64_t(expected);
}
bool allQueries(const fs::RegionalOrderView& view, const fs::PeriodicAnalysis& graph, Expr count)
{
    auto& arena = *view.context()->expressions();
    const auto sites = static_cast<uint32_t>(graph.payloads.size());
    for (uint64_t trips = 0; trips <= 4; ++trips) {
        for (uint32_t source = 0; source < sites; ++source) {
            for (uint32_t target = 0; target < sites; ++target) {
                for (auto aKind : {Kind::Start, Kind::Completion}) {
                    for (auto bKind : {Kind::Start, Kind::Completion}) {
                        for (uint64_t a = 0; a <= trips; ++a) {
                            for (uint64_t b = 0; b <= trips; ++b) {
                                auto expected = graph.eventPrecedes(
                                    {source, aKind}, a, {target, bKind}, b, sites * trips);
                                auto actual = fs::regionalReachability(view.regional(),
                                    {source, arena.constant(a), aKind}, {target, arena.constant(b), bKind});
                                const bool present = a < trips && b < trips;
                                if (!actual || (present && expected.error != fs::PeriodicQueryError::None) ||
                                    !evaluate(arena, *actual, count, trips, present && expected.value)) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    fs::RegionalEvent valid{0, arena.constant(0), Kind::Start};
    auto invalid = valid;
    invalid.visits.push_back(arena.constant(0));
    if (fs::regionalReachability(view.regional(), invalid, valid)) { return false; }
    invalid = valid; invalid.type = sites;
    if (fs::regionalReachability(view.regional(), invalid, valid)) { return false; }
    invalid = valid; invalid.ordinal = arena.boolean(true);
    return !fs::regionalReachability(view.regional(), invalid, valid);
}
bool check(scf::ForOp loop, const pto::SyncInput& input, const fs::PhaseIndex& index)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    const auto count = arena->input(loop.getUpperBound());
    const auto trips = arena->select(arena->slt(arena->constant(0), count), count, arena->constant(0));
    std::string error;
    auto domain = fs::captureCompactFixedBodyContext(loop, input, index, arena, trips, error);
    if (!domain || !error.empty() || domain->payloads().empty()) { return false; }
    const auto payloads = domain->payloads();
    auto upper = fs::analyzeCompactWriterReader(payloads, {}, {}, {{0, 0, {true, 3, 3}}});
    if (!upper.error.empty() || upper.upper.generators.empty()) { return false; }
    auto nativeOnly = fs::buildCompactOrderBounds(domain, upper, {}, 0);
    if (!nativeOnly.exportError.empty() || !nativeOnly.certificate.error.empty() ||
        nativeOnly.bounds.guarantee != fs::InputOrderGuarantee::InputOrderCovering ||
        nativeOnly.certificate.excess != 0 || nativeOnly.certificate.equalFrontiers ||
        !nativeOnly.bounds.upper || !nativeOnly.bounds.lower ||
        nativeOnly.certificate.upper != nativeOnly.upperGraph ||
        nativeOnly.certificate.lower != nativeOnly.lowerGraph ||
        !allQueries(*nativeOnly.bounds.upper, nativeOnly.upperGraph->analysis(), count) ||
        !allQueries(*nativeOnly.bounds.lower, nativeOnly.lowerGraph->analysis(), count)) { return false; }
    fs::CompactLowerFacts lower;
    lower.records = upper.upper.generators;
    auto equal = fs::buildCompactOrderBounds(domain, upper, lower, 4);
    if (!equal.exportError.empty() || !equal.certificate.error.empty() || !equal.certificate.equalFrontiers ||
        !equal.certificate.exactOriginalCovers ||
        equal.bounds.guarantee != fs::InputOrderGuarantee::InputOrderEquivalent ||
        failed(fs::validateBoundingRegionalResult(equal.bounds, error))) { return false; }
    auto foreign = fs::captureCompactFixedBodyContext(loop, input, index, arena, trips, error);
    auto foreignGraph = fs::bindPeriodicExcessGraph(foreign, equal.lowerGraph->analysis(),
                                                   fs::ReductionQuality::Covers, error);
    if (!foreignGraph || fs::certifyPeriodicExcess(equal.upperGraph, foreignGraph, 4).error.empty()) { return false; }
    // Deep snapshots and exported callbacks outlive all mutable source/result structs.
    auto view = *equal.bounds.upper;
    auto snapshot = equal.upperGraph;
    upper.upper.generators.clear(); upper.upper.frontiers.clear(); lower.records.clear(); equal = {};
    if (!allQueries(view, snapshot->analysis(), count) || snapshot->analysis().generators.empty()) { return false; }
    auto unavailable = fs::buildCompactOrderBounds({}, *nativeOnly.mathematical, {}, 4);
    if (unavailable.exportError.empty() || !unavailable.mathematical ||
        unavailable.mathematical->upper.generators.empty() || unavailable.bounds.upper || unavailable.bounds.lower) {
        return false;
    }
    fs::CompactLowerFacts malformed;
    malformed.error = "test lower proof unavailable";
    auto partial = fs::buildCompactOrderBounds(domain, *nativeOnly.mathematical, malformed, 4);
    if (partial.exportError.empty() || !partial.bounds.upper || partial.bounds.lower ||
        partial.mathematical->upper.generators.empty()) { return false; }
    auto onlyNative = fs::analyzeCompactWriterReader(payloads, {}, {});
    fs::CompactLowerFacts tooStrong;
    tooStrong.records.push_back({0, 0, 1});
    auto contradicted = fs::buildCompactOrderBounds(domain, onlyNative, tooStrong, 0);
    if (contradicted.exportError.empty() || contradicted.certificate.error.empty() ||
        !contradicted.bounds.upper || contradicted.bounds.lower || !contradicted.lowerGraph ||
        contradicted.lowerFacts->records.size() != 1 ||
        contradicted.bounds.guarantee != fs::InputOrderGuarantee::InputOrderCovering) { return false; }
    if (payloads.size() > 1) {
        auto nativeUpper = fs::analyzeCompactWriterReader(payloads, {}, {}, {}, {{0, 1, 0}});
        auto native = fs::buildCompactOrderBounds(domain, nativeUpper, {}, 4);
        if (!native.exportError.empty() || !native.certificate.error.empty() ||
            native.lowerGraph->analysis().nativePrerequisites.size() != 1 || !native.certificate.equalFrontiers ||
            !allQueries(*native.bounds.lower, native.lowerGraph->analysis(), count)) { return false; }
    }
    return true;
}
} // namespace
int runCompactOrderBoundsChecks(func::FuncOp function, const pto::SyncInput& input)
{
    fs::PhaseIndex index;
    if (failed(index.build(function, input))) { return 1; }
    scf::ForOp loop;
    for (auto candidate : function.getOps<scf::ForOp>()) { if (loop) { return 1; } loop = candidate; }
    if (!loop) { return 1; }
    if (function->hasAttr("test.compact_nested")) {
        auto arena = std::make_shared<fs::RegionExpressions>();
        std::string error;
        if (fs::captureCompactFixedBodyContext(loop, input, index, arena, arena->constant(4), error) || error.empty()) {
            return 1;
        }
    } else if (!check(loop, input, index)) {
        llvm::errs() << "compact order bounds failed: " << function.getSymName() << "\n";
        return 1;
    }
    llvm::outs() << "compact order bounds passed: " << function.getSymName() << "\n";
    return 0;
}
