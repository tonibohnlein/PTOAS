// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent explicit CI closure oracle for the regional loss ceiling.
#include "PTO/Transforms/FrontierSynch/Shared/CompletionRegion.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "PTO/Transforms/FrontierSynch/Shared/CompletionLoss.h"
#include "PTO/Transforms/FrontierSynch/Structured/PeriodicExcessCertificate.h"
#include "llvm/Support/raw_ostream.h"
#include <random>
namespace fs = mlir::pto::frontiersynch;
namespace {
using Graph = std::vector<std::vector<bool>>;
void close(Graph& graph)
{
    for (std::size_t k = 0; k < graph.size(); ++k) {
        for (std::size_t i = 0; i < graph.size(); ++i) {
            for (std::size_t j = 0; j < graph.size(); ++j) {
                graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
            }
        }
    }
}
bool randomCase(std::mt19937& random)
{
    const unsigned groups = 1 + random() % 8;
    std::vector<fs::CompletionLossAtom> atoms;
    std::vector<fs::CompletionLossVisit> visits;
    std::vector<std::vector<std::size_t>> next(groups);
    std::vector<unsigned> owner;
    for (unsigned g = 0; g < groups; ++g) {
        const unsigned n = random() % 4;
        atoms.push_back({g, (g % 2) ? std::optional<uint64_t>(n) : std::nullopt, bool(random() % 2)});
        visits.push_back({g, n});
        if (g + 1 < groups) { next[g].push_back(g + 1); }
        for (unsigned i = 0; i < n; ++i) { owner.push_back(g); }
    }
    Graph lower(2 * owner.size(), std::vector<bool>(2 * owner.size(), false));
    for (unsigned i = 0; i < owner.size(); ++i) {
        lower[2*i][2*i+1] = true;
        for (unsigned j = i + 1; j < owner.size(); ++j) {
            if (random() % 5 == 0) { lower[2*i+1][2*j] = true; }
        }
    }
    close(lower);
    auto upper = lower;
    for (unsigned i = 0; i < owner.size(); ++i) {
        for (unsigned j = i + 1; j < owner.size(); ++j) {
            if (owner[i] != owner[j] || !atoms[owner[i]].exactRealizedOrder) { upper[2*i+1][2*j] = true; }
        }
    }
    close(upper);
    uint64_t actual = 0;
    for (unsigned i = 0; i < owner.size(); ++i) {
        for (unsigned j = 0; j < owner.size(); ++j) { actual += upper[2*i+1][2*j] && !lower[2*i+1][2*j]; }
    }
    fs::CompletionLossCertificate certificate(atoms, next, 0, groups - 1);
    auto value = certificate.evaluate(visits);
    return value.error.empty() && value.payloads == owner.size() && value.bound.uge(actual);
}
}
bool runCompletionLossChecks()
{
    std::mt19937 random(5489);
    for (unsigned i = 0; i < 4000; ++i) {
        if (!randomCase(random)) { llvm::errs() << "regional loss oracle failed\n"; return false; }
    }
    // A final-false condition visit is distinct from a body visit. Zero body
    // trips still execute node0 once. Repeated visits reuse the same atom.
    fs::CompletionLossCertificate loop({{0, 1, true}, {1, 2, true}, {2, 0, true}}, {{1,2},{0},{}}, 0, 2);
    auto zero = loop.evaluate({{0,1},{2,0}});
    auto twice = loop.evaluate({{0,1},{1,2},{0,1},{1,2},{0,1},{2,0}});
    if (!zero.error.empty() || zero.bound != 0 || !twice.error.empty() || twice.bound != 19 ||
        loop.evaluate({{0,1},{1,2},{2,0}}).error.empty() ||
        loop.evaluate({{0,2},{2,0}}).error.empty()) { return false; }
    fs::CompletionLossCertificate malformed({{9, 0, true}}, {{}}, 0, 0);
    if (malformed.valid() || malformed.evaluate({{0,0}}).error.empty()) { return false; }
    fs::CompletionLossValue overflow;
    const auto maximum = llvm::APInt::getMaxValue(256);
    if (!overflow.append(maximum, llvm::APInt(256,0)) ||
        overflow.append(llvm::APInt(256,2), llvm::APInt(256,0)) ||
        overflow.payloads != maximum || !overflow.bound.isZero()) { return false; }
    fs::CompletionLossValue unavailable;
    if (unavailable.appendPeriodic(fs::PeriodicExcessCertificate{}, true) || unavailable.error.empty()) {
        return false;
    }
    llvm::outs() << "regional loss: 4000 explicit CI closures, final-false/zero/repeated visits and overflow passed\n";
    return true;
}

mlir::LogicalResult checkEmittedCompletionLoss(mlir::func::FuncOp function,
    const fs::CompletionRegion& region)
{
    using namespace mlir;
    const auto certificate = region.lossCertificate();
    auto encoded = region.lossReport();
    const auto atoms = encoded.getAs<ArrayAttr>("atoms");
    uint64_t cursor = encoded.getAs<IntegerAttr>("entry").getInt();
    const uint64_t exit = encoded.getAs<IntegerAttr>("exit").getInt();
    std::vector<fs::CompletionLossVisit> visits;
    while (true) {
        const auto& atom = certificate->partition()[cursor];
        if (!atom.payloads || visits.size() > atoms.size()) { return failure(); }
        visits.push_back({cursor, *atom.payloads});
        if (cursor == exit) { break; }
        auto next = cast<DictionaryAttr>(atoms[cursor]).getAs<DenseI64ArrayAttr>("successors");
        if (next.size() != 1) { return failure(); }
        cursor = next[0];
    }
    const auto bound = certificate->evaluate(visits);
    if (!bound.error.empty()) { llvm::errs() << bound.error << "\n"; return failure(); }
    std::vector<int64_t> cells;
    std::vector<std::size_t> barriers;
    bool supported = true;
    function.walk<WalkOrder::PreOrder>([&](Operation* operation) {
        if (auto branch = dyn_cast<scf::IfOp>(operation)) {
            auto constant = branch.getCondition().getDefiningOp<arith::ConstantOp>();
            auto value = constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
            supported &= value && value.getValue().isOne() && branch.getElseRegion().empty();
        }
        if (auto cell = operation->getAttrOfType<IntegerAttr>("test.loss_cell")) { cells.push_back(cell.getInt()); }
        if (auto barrier = dyn_cast<pto::BarrierOp>(operation)) {
            supported &= barrier.getPipe().getPipe() == pto::PIPE::PIPE_V ||
                barrier.getPipe().getPipe() == pto::PIPE::PIPE_ALL;
            barriers.push_back(cells.size());
        }
    });
    if (!supported) { return failure(); }
    // These two fixtures consist of same-pipe writes/RMWs with independently
    // supplied disjoint cell labels. Native I/I and C/C chains never imply C/I.
    const auto n = cells.size();
    Graph lower(2*n, std::vector<bool>(2*n, false));
    for (std::size_t i = 0; i < n; ++i) {
        lower[2*i][2*i+1] = true;
        for (std::size_t j = i+1; j < n; ++j) {
            lower[2*i][2*j] = lower[2*i+1][2*j+1] = true;
        }
    }
    auto emitted = lower;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i+1; j < n; ++j) {
            if (cells[i] == cells[j]) { lower[2*i+1][2*j] = true; }
            for (auto gap : barriers) {
                if (i < gap && gap <= j) { emitted[2*i+1][2*j] = true; }
            }
        }
    }
    close(lower); close(emitted);
    uint64_t excess = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (lower[2*i+1][2*j] && !emitted[2*i+1][2*j]) { return failure(); }
            excess += emitted[2*i+1][2*j] && !lower[2*i+1][2*j];
        }
    }
    const bool exact = function->getAttrOfType<StringAttr>("test.loss_internal").getValue() == "zero";
    if (bound.payloads != n || bound.bound.ult(excess) || (exact ? excess != 0 : excess == 0)) {
        llvm::errs() << "loss mismatch n=" << n << " actual=" << excess << "\n";
        function.print(llvm::errs()); encoded.print(llvm::errs());
        return failure();
    }
    llvm::outs() << "emitted loss: " << function.getSymName() << " actual=" << excess
                 << " bound=" << bound.bound.getZExtValue() << "\n";
    return success();
}
