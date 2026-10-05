// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Closed integer DBMs preserve exact projection; unions remain explicit unions.
#include "PTO/Transforms/FrontierSynch/DifferenceBoundRelations.h"
#include <algorithm>
#include <limits>
#include <map>
namespace mlir::pto::frontiersynch {
namespace {
bool representableDimensions(unsigned dimensions)
{
    if (dimensions == std::numeric_limits<unsigned>::max()) { return false; }
    const auto side = std::size_t(dimensions) + 1;
    return side <= std::vector<std::optional<BoundInteger>>().max_size() / side;
}
} // namespace
std::size_t DifferenceBoundSystem::position(unsigned lhs, unsigned rhs) const
{
    return std::size_t(lhs) * (std::size_t(dimensionCount) + 1) + rhs;
}
void DifferenceBoundSystem::tighten(unsigned lhs, unsigned rhs, const BoundInteger& value)
{
    auto& current = bounds[position(lhs, rhs)];
    if (!current || value < *current) { current = value; }
}
void DifferenceBoundSystem::close()
{
    for (unsigned via = 0; via <= dimensionCount; ++via) {
        for (unsigned lhs = 0; lhs <= dimensionCount; ++lhs) {
            const auto left = bounds[position(lhs, via)];
            if (!left) { continue; }
            for (unsigned rhs = 0; rhs <= dimensionCount; ++rhs) {
                const auto right = bounds[position(via, rhs)];
                if (right) { tighten(lhs, rhs, *left + *right); }
            }
        }
        for (unsigned coordinate = 0; coordinate <= dimensionCount; ++coordinate) {
            if (*bounds[position(coordinate, coordinate)] < 0) {
                empty = true;
                return;
            }
        }
    }
}
FailureOr<DifferenceBoundSystem> DifferenceBoundSystem::create(
    unsigned dimensions, llvm::ArrayRef<DifferenceBoundConstraint> constraints)
{
    // Reject an unrepresentable augmented dimension before either incrementing
    // unsigned coordinates or multiplying the allocation size.
    if (!representableDimensions(dimensions)) { return failure(); }
    const auto side = std::size_t(dimensions) + 1;
    for (const auto& atom : constraints) {
        if (atom.lhs > dimensions || atom.rhs > dimensions) { return failure(); }
    }
    DifferenceBoundSystem result;
    result.dimensionCount = dimensions;
    result.bounds.assign(side * side, std::nullopt);
    for (unsigned i = 0; i <= dimensions; ++i) { result.tighten(i, i, BoundInteger(0)); }
    for (const auto& atom : constraints) { result.tighten(atom.lhs, atom.rhs, atom.bound); }
    result.close();
    return result;
}
std::optional<BoundInteger> DifferenceBoundSystem::bound(unsigned lhs, unsigned rhs) const
{
    if (empty || lhs > dimensionCount || rhs > dimensionCount) { return std::nullopt; }
    return bounds[position(lhs, rhs)];
}
std::vector<DifferenceBoundConstraint> DifferenceBoundSystem::constraints() const
{
    if (empty) { return {{0, 0, BoundInteger(-1)}}; }
    std::vector<DifferenceBoundConstraint> result;
    for (unsigned lhs = 0; lhs <= dimensionCount; ++lhs) {
        for (unsigned rhs = 0; rhs <= dimensionCount; ++rhs) {
            const auto& value = bounds[position(lhs, rhs)];
            if (lhs != rhs && value) { result.push_back({lhs, rhs, *value}); }
        }
    }
    return result;
}
FailureOr<DifferenceBoundSystem> DifferenceBoundSystem::intersect(const DifferenceBoundSystem& other) const
{
    if (dimensionCount != other.dimensionCount) { return failure(); }
    if (empty) { return *this; }
    if (other.empty) { return other; }
    auto result = *this;
    for (std::size_t i = 0; i < bounds.size(); ++i) {
        const auto& value = other.bounds[i];
        if (value && (!result.bounds[i] || *value < *result.bounds[i])) { result.bounds[i] = value; }
    }
    result.close();
    return result;
}
FailureOr<DifferenceBoundSystem> DifferenceBoundSystem::remap(
    unsigned newDimensions, llvm::ArrayRef<unsigned> oldToNew) const
{
    if (oldToNew.size() != dimensionCount ||
        std::any_of(oldToNew.begin(), oldToNew.end(),
                    [&](unsigned coordinate) { return coordinate >= newDimensions; })) {
        return failure();
    }
    auto atoms = constraints();
    for (auto& atom : atoms) {
        if (atom.lhs) { atom.lhs = oldToNew[atom.lhs - 1] + 1; }
        if (atom.rhs) { atom.rhs = oldToNew[atom.rhs - 1] + 1; }
    }
    return create(newDimensions, atoms);
}
FailureOr<DifferenceBoundSystem> DifferenceBoundSystem::project(llvm::ArrayRef<unsigned> keep) const
{
    if (keep.size() > dimensionCount) { return failure(); }
    std::vector<bool> selected(dimensionCount, false);
    for (auto coordinate : keep) {
        if (coordinate >= dimensionCount || selected[coordinate]) { return failure(); }
        selected[coordinate] = true;
    }
    auto projected = create(static_cast<unsigned>(keep.size()), {});
    if (failed(projected)) { return failure(); }
    projected->empty = empty;
    if (empty) { return projected; }
    // Every bound induced by eliminated variables is already in the closure.
    // Integer bounds give an integer extension exactly when this submatrix holds.
    for (unsigned lhs = 0; lhs <= keep.size(); ++lhs) {
        const auto originalLhs = lhs ? keep[lhs - 1] + 1 : 0;
        for (unsigned rhs = 0; rhs <= keep.size(); ++rhs) {
            const auto originalRhs = rhs ? keep[rhs - 1] + 1 : 0;
            projected->bounds[projected->position(lhs, rhs)] = bounds[position(originalLhs, originalRhs)];
        }
    }
    return projected;
}
bool DifferenceBoundSystem::isSubsetOf(const DifferenceBoundSystem& other) const
{
    if (dimensionCount != other.dimensionCount) { return false; }
    if (empty) { return true; }
    if (other.empty) { return false; }
    for (std::size_t i = 0; i < bounds.size(); ++i) {
        if (other.bounds[i] && (!bounds[i] || *bounds[i] > *other.bounds[i])) { return false; }
    }
    return true;
}
bool DifferenceBoundSystem::operator==(const DifferenceBoundSystem& other) const
{
    return dimensionCount == other.dimensionCount && empty == other.empty &&
           (empty || bounds == other.bounds);
}
namespace {
struct Direction {
    unsigned lhs = 0, rhs = 0;
    std::vector<BoundInteger> thresholds;
};
std::vector<Direction> directions(llvm::ArrayRef<DifferenceBoundSystem> lhs,
                                  llvm::ArrayRef<DifferenceBoundSystem> rhs)
{
    std::map<std::pair<unsigned, unsigned>, std::vector<BoundInteger>> values;
    for (auto pieces : {lhs, rhs}) {
        for (const auto& piece : pieces) {
            if (piece.isEmpty()) { continue; }
            for (const auto& atom : piece.constraints()) { values[{atom.lhs, atom.rhs}].push_back(atom.bound); }
        }
    }
    std::vector<Direction> result;
    for (auto& entry : values) {
        auto& thresholds = entry.second;
        std::sort(thresholds.begin(), thresholds.end());
        thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
        result.push_back({entry.first.first, entry.first.second, std::move(thresholds)});
    }
    return result;
}
bool contained(const DifferenceBoundSystem& region, llvm::ArrayRef<DifferenceBoundSystem> pieces)
{
    return std::any_of(pieces.begin(), pieces.end(),
                       [&](const auto& piece) { return region.isSubsetOf(piece); });
}
FailureOr<bool> canContribute(const DifferenceBoundSystem& region,
                             llvm::ArrayRef<DifferenceBoundSystem> lhs,
                             llvm::ArrayRef<DifferenceBoundSystem> rhs)
{
    if (contained(region, rhs)) { return false; }
    for (const auto& piece : lhs) {
        auto overlap = region.intersect(piece);
        if (failed(overlap)) { return failure(); }
        if (!overlap->isEmpty()) { return true; }
    }
    return false;
}
FailureOr<DifferenceBoundSystem> interval(unsigned dimensions, const Direction& direction, std::size_t choice)
{
    std::vector<DifferenceBoundConstraint> atoms;
    if (choice < direction.thresholds.size()) {
        atoms.push_back({direction.lhs, direction.rhs, direction.thresholds[choice]});
    }
    if (choice) {
        // d > c over integers is -d <= -c-1. No machine-width negation.
        atoms.push_back({direction.rhs, direction.lhs, -direction.thresholds[choice - 1] - 1});
    }
    return DifferenceBoundSystem::create(dimensions, atoms);
}
FailureOr<std::vector<DifferenceBoundSystem>> arrange(
    unsigned dimensions, llvm::ArrayRef<DifferenceBoundSystem> lhs,
    llvm::ArrayRef<DifferenceBoundSystem> rhs, llvm::ArrayRef<Direction> axes)
{
    auto universe = DifferenceBoundSystem::create(dimensions, {});
    if (failed(universe)) { return failure(); }
    struct Frame { DifferenceBoundSystem region; std::size_t axis = 0, next = 0; };
    std::vector<Frame> stack{{*universe, 0, 0}};
    std::vector<DifferenceBoundSystem> output;
    // The stack enumerates a single arrangement, retaining only one partial
    // cell per depth. It never repeatedly expands a growing union of pieces.
    while (!stack.empty()) {
        auto& current = stack.back();
        if (current.axis == axes.size()) {
            if (contained(current.region, lhs) && !contained(current.region, rhs)) {
                output.push_back(std::move(current.region));
            }
            stack.pop_back();
            continue;
        }
        const auto& axis = axes[current.axis];
        if (current.next > axis.thresholds.size()) { stack.pop_back(); continue; }
        auto slice = interval(dimensions, axis, current.next++);
        if (failed(slice)) { return failure(); }
        auto next = current.region.intersect(*slice);
        if (failed(next)) { return failure(); }
        if (next->isEmpty()) { continue; }
        // Exact pruning does not split a union: a partial arrangement cell
        // contributes nothing if already covered or disjoint from the input.
        auto useful = canContribute(*next, lhs, rhs);
        if (failed(useful)) { return failure(); }
        if (*useful) { stack.push_back({std::move(*next), current.axis + 1, 0}); }
    }
    return output;
}
} // namespace
FailureOr<std::vector<DifferenceBoundSystem>> subtractDifferenceBoundUnions(
    unsigned dimensions, llvm::ArrayRef<DifferenceBoundSystem> lhs,
    llvm::ArrayRef<DifferenceBoundSystem> rhs)
{
    if (!representableDimensions(dimensions)) { return failure(); }
    for (auto pieces : {lhs, rhs}) {
        if (std::any_of(pieces.begin(), pieces.end(),
                        [&](const auto& piece) { return piece.dimensions() != dimensions; })) {
            return failure();
        }
    }
    if (std::all_of(lhs.begin(), lhs.end(), [](const auto& piece) { return piece.isEmpty(); })) {
        return std::vector<DifferenceBoundSystem>{};
    }
    if (std::all_of(rhs.begin(), rhs.end(), [](const auto& piece) { return piece.isEmpty(); })) {
        return std::vector<DifferenceBoundSystem>(lhs.begin(), lhs.end());
    }
    if (std::all_of(lhs.begin(), lhs.end(), [&](const auto& piece) { return contained(piece, rhs); })) {
        return std::vector<DifferenceBoundSystem>{};
    }
    const auto axes = directions(lhs, rhs);
    return arrange(dimensions, lhs, rhs, axes);
}
} // namespace mlir::pto::frontiersynch
