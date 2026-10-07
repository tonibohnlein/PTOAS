// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// The cutoff proves boundary-class constancy; matrix powers compose those classes.
#include "VaryingBoundaryQuery.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include <numeric>
namespace mlir::pto::frontiersynch {
namespace {
bool count(uint64_t& value)
{
    if (value == UINT64_MAX) { return false; }
    ++value;
    return true;
}
bool matrixShape(const VaryingBoundaryMatrix& matrix, std::size_t size)
{
    return matrix.size() == size && llvm::all_of(matrix, [size](const auto& row) { return row.size() == size; });
}
bool validTypes(const AffineRotatingVisits& visits)
{
    const auto& child = visits.child;
    if (!visits.error.empty() || !child.error.empty() || !child.quotient.error.empty() || !visits.slope ||
        !visits.period || !child.period || child.types.size() != child.period || !child.refresh ||
        child.refresh > (UINT64_MAX - 1) / 2 || child.cutoff < 2 * child.refresh + 1) { return false; }
    const auto delta = child.cutoff > visits.intercept ? child.cutoff - visits.intercept : 0;
    if (visits.startup != delta / visits.slope + uint64_t(delta % visits.slope != 0) ||
        visits.period != child.period / std::gcd(visits.slope, child.period)) { return false; }
    for (uint64_t residue = 0; residue < child.period; ++residue) {
        const auto current = child.cutoff % child.period;
        const auto shift = residue >= current ? residue - current : child.period - (current - residue);
        if (shift > UINT64_MAX - child.cutoff || child.types[residue].representative != child.cutoff + shift) {
            return false;
        }
    }
    return true;
}
} // namespace
VaryingBoundaryQuery::VaryingBoundaryQuery(const AffineRotatingVisits& visits,
    const std::vector<VaryingBoundaryPort>& ports, const std::vector<VaryingBoundaryMatrix>& startup,
    const std::vector<VaryingBoundaryMatrix>& suffix, const std::vector<std::vector<VaryingBoundaryMatrix>>& powers)
    : visits(visits), ports(ports), startup(startup), suffix(suffix), powers(powers)
{
    if (!validTypes(visits) || startup.size() != visits.startup || suffix.size() != visits.period ||
        powers.size() != visits.period || !llvm::all_of(ports, [&](auto port) { return valid(port); })) {
        constructionError = "unavailable certified varying boundary domain or port form";
        return;
    }
    auto shaped = [&](const auto& matrix) { return matrixShape(matrix, ports.size()); };
    if (!llvm::all_of(startup, shaped) || !llvm::all_of(suffix, shaped) ||
        !llvm::all_of(powers, [&](const auto& phase) { return phase.size() == 64 && llvm::all_of(phase, shaped); })) {
        constructionError = "invalid varying boundary transfer dimensions";
    }
}
bool VaryingBoundaryQuery::valid(VaryingBoundaryPort port) const
{
    const auto& occurrence = port.occurrence;
    return occurrence.type < visits.child.quotient.payloads.size() &&
        (port.kind == PeriodicEventKind::Start || port.kind == PeriodicEventKind::Completion) &&
        (occurrence.tail ? occurrence.coordinate > 0 && occurrence.coordinate <= visits.child.refresh :
                           occurrence.coordinate < visits.child.refresh);
}
std::optional<uint64_t> VaryingBoundaryQuery::length(uint64_t visit) const
{
    const auto wide = llvm::APInt(128, visits.slope) * llvm::APInt(128, visit) + llvm::APInt(128, visits.intercept);
    if (visit < visits.startup) {
        return wide.getActiveBits() > 64 ? std::nullopt : std::optional<uint64_t>(wide.getZExtValue());
    }
    const auto residue = wide.urem(llvm::APInt(128, visits.child.period)).getZExtValue();
    return visits.child.types[residue].representative;
}
std::optional<bool> VaryingBoundaryQuery::inside(
    VaryingBoundaryPort source, VaryingBoundaryPort target, uint64_t visit)
{
    if (!valid(source) || !valid(target) || !count(thresholdQueries)) { return std::nullopt; }
    const auto n = length(visit);
    if (!n) { return std::nullopt; }
    auto present = [n](BoundaryOccurrence occurrence) {
        return occurrence.tail ? occurrence.coordinate <= *n : occurrence.coordinate < *n;
    };
    if (!present(source.occurrence) || !present(target.occurrence)) { return false; }
    const auto threshold = visits.child.quotient.eventThreshold(
        {source.occurrence.type, source.kind}, {target.occurrence.type, target.kind});
    if (threshold.error != PeriodicQueryError::None) { return std::nullopt; }
    if (!threshold.displacement) { return false; }
    const auto a = source.occurrence.at(*n), b = target.occurrence.at(*n);
    const bool reached = a <= b && b - a >= *threshold.displacement;
    if (visit >= visits.startup && source.occurrence.tail != target.occurrence.tail) {
        // Every port, including a port collected from a short startup type, is
        // head<R or tail<=R. Adding arbitrary-site head0/tail1 preserves this.
        // With C=2R+B+1, head->tail differences exceed every finite threshold;
        // tail->head differences are negative. Check the needed inequality at
        // the least long length of this residue rather than assuming it. A
        // false head->tail or true tail->head relation could change later and
        // is therefore unavailable, never a numerical representative proof.
        if ((!source.occurrence.tail && !reached) || (source.occurrence.tail && reached)) { return std::nullopt; }
    }
    // Head/head and tail/tail differences are independent of length. The two
    // other cases above are monotone away from their threshold for every longer
    // length in this residue. This establishes the entire suffix class.
    return reached;
}
bool VaryingBoundaryQuery::advance(std::vector<uint8_t>& row, const VaryingBoundaryMatrix& matrix)
{
    if (!count(matrixApplications)) { return false; }
    std::vector<uint8_t> next(ports.size());
    for (std::size_t i = 0; i < ports.size(); ++i) {
        if (!row[i]) { continue; }
        for (std::size_t j = 0; j < ports.size(); ++j) { next[j] |= matrix[i][j]; }
    }
    row = std::move(next);
    return true;
}
std::optional<bool> VaryingBoundaryQuery::compute(
    VaryingBoundaryPort source, VaryingBoundaryPort target, uint64_t sourceVisit, uint64_t gap)
{
    if (gap > UINT64_MAX - sourceVisit) { return std::nullopt; }
    if (!gap) { return inside(source, target, sourceVisit); }
    const auto targetVisit = sourceVisit + gap;
    std::vector<uint8_t> row(ports.size());
    for (std::size_t i = 0; i < ports.size(); ++i) {
        const auto reaches = inside(source, ports[i], sourceVisit);
        if (!reaches) { return std::nullopt; }
        row[i] = *reaches;
    }
    auto begin = sourceVisit;
    while (begin < visits.startup && begin < targetVisit) {
        if (!advance(row, startup[begin])) { return std::nullopt; }
        ++begin;
    }
    if (begin < targetVisit) {
        const auto phase = (begin - visits.startup) % visits.period;
        const auto remaining = targetVisit - begin, tail = remaining % visits.period;
        auto cycles = remaining / visits.period;
        for (unsigned bit = 0; cycles; ++bit, cycles >>= 1) {
            if ((cycles & 1) && !advance(row, powers[phase][bit])) { return std::nullopt; }
        }
        for (uint64_t j = 0; j < tail; ++j) {
            if (!advance(row, suffix[(phase + j) % visits.period])) { return std::nullopt; }
        }
    }
    for (std::size_t i = 0; i < ports.size(); ++i) {
        if (!row[i]) { continue; }
        const auto reaches = inside(ports[i], target, targetVisit);
        if (!reaches) { return std::nullopt; }
        if (*reaches) { return true; }
    }
    return false;
}
std::optional<bool> VaryingBoundaryQuery::query(BoundaryOccurrence source, PeriodicEventKind sourceKind,
    BoundaryOccurrence target, PeriodicEventKind targetKind, uint64_t sourceVisit, uint64_t gap)
{
    if (!constructionError.empty() || !count(queryCount) || !valid({source, sourceKind}) ||
        !valid({target, targetKind})) { return std::nullopt; }
    if (sourceVisit >= visits.startup) {
        const auto residue = (sourceVisit - visits.startup) % visits.period;
        if (residue > UINT64_MAX - visits.startup) { return std::nullopt; }
        sourceVisit = visits.startup + residue;
    }
    const Key key{source.type, source.coordinate, source.tail, unsigned(sourceKind), target.type,
        target.coordinate, target.tail, unsigned(targetKind), sourceVisit, gap};
    if (auto found = cache.find(key); found != cache.end()) { return found->second; }
    auto result = compute({source, sourceKind}, {target, targetKind}, sourceVisit, gap);
    cache.emplace(key, result);
    return result;
}
} // namespace mlir::pto::frontiersynch
