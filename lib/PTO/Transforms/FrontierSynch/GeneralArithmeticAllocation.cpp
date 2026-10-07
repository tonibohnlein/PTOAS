// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Certify every earlier/later pair; never infer reuse from sampled invocations.
#include "PTO/Transforms/FrontierSynch/GeneralArithmeticAllocation.h"
#include "PTO/Transforms/FrontierSynch/PhysicalAllocation.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <set>
namespace mlir::pto::frontiersynch {
namespace {
using Pair = std::pair<std::size_t, std::size_t>;
using Entry = std::pair<const ArithmeticRelationKey*, const IntegerSystem*>;
bool validKey(const ArithmeticRelationKey& key, const GeneralArithmeticDemandAnalysis& analysis,
              ArrayRef<uint32_t> pipes)
{
    const auto s = key.source.residues.size(), t = key.target.residues.size();
    const auto maximum = std::numeric_limits<unsigned>::max();
    auto residues = [&](ArrayRef<uint64_t> values) {
        return llvm::all_of(values, [&](uint64_t value) { return value < analysis.period; });
    };
    return key.source.site < pipes.size() && key.target.site < pipes.size() &&
           key.source.event == ArithmeticEvent::Completion && key.target.event == ArithmeticEvent::Start &&
           s <= maximum && t <= maximum - s && s + t <= (maximum - analysis.parameterCount) / 2 &&
           key.parameterResidues.size() == analysis.parameterCount && residues(key.source.residues) &&
           residues(key.target.residues) && residues(key.parameterResidues);
}
IntegerConstraint difference(unsigned dimensions, unsigned a, unsigned b, BoundInteger bound)
{
    IntegerConstraint result{std::vector<BoundInteger>(dimensions, BoundInteger(0)), std::move(bound)};
    result.coefficients[a] = BoundInteger(1);
    result.coefficients[b] = BoundInteger(-1);
    return result;
}
bool covered(const IntegerSystem& domain, ArrayRef<unsigned> query,
             const std::vector<IntegerSystem>* required)
{
    if (domain.isEmpty()) { return true; }
    if (!required) { return false; }
    auto projected = domain.project(query);
    if (failed(projected)) { return false; }
    auto missing = subtractIntegerUnions(static_cast<unsigned>(query.size()), *projected, *required);
    return succeeded(missing) && missing->empty();
}
bool orderedReuse(const GeneralArithmeticDemandAnalysis& analysis, const Entry& first, const Entry& second)
{
    const auto& a = *first.first;
    const auto& b = *second.first;
    const unsigned s = a.source.residues.size(), t = a.target.residues.size(), tuple = s + t;
    if (b.source.residues.size() != s || b.target.residues.size() != t) { return false; }
    if (a.parameterResidues != b.parameterResidues) { return true; }
    const unsigned dimensions = 2 * tuple + analysis.parameterCount;
    SmallVector<unsigned> left, right, query;
    for (unsigned i = 0; i < tuple; ++i) { left.push_back(i); right.push_back(tuple + i); }
    for (unsigned i = 0; i < t; ++i) { query.push_back(s + i); }
    for (unsigned i = 0; i < s; ++i) { query.push_back(tuple + i); }
    for (unsigned i = 0; i < analysis.parameterCount; ++i) {
        left.push_back(2 * tuple + i); right.push_back(2 * tuple + i); query.push_back(2 * tuple + i);
    }
    auto x = first.second->remap(dimensions, left), y = second.second->remap(dimensions, right);
    if (failed(x) || failed(y)) { return false; }
    auto pair = x->intersect(*y);
    if (failed(pair)) { return false; }
    auto from = a.target, to = b.source;
    from.event = ArithmeticEvent::Completion;
    to.event = ArithmeticEvent::Start;
    auto found = analysis.requiredOrder.find({from, to, a.parameterResidues});
    const auto* required = found == analysis.requiredOrder.end() ? nullptr : &found->second;
    SmallVector<IntegerConstraint> prefix;
    for (unsigned i = 0; i < s; ++i) {
        // period*q_a+r_a < period*q_b+r_b, after equal earlier coordinates.
        const auto limit = BoundInteger(static_cast<int64_t>(b.source.residues[i])) -
                           BoundInteger(static_cast<int64_t>(a.source.residues[i])) - BoundInteger(1);
        auto atoms = prefix;
        atoms.push_back(difference(dimensions, i, tuple + i,
            floorDiv(limit, BoundInteger(static_cast<int64_t>(analysis.period)))));
        auto order = IntegerSystem::create(dimensions, atoms);
        if (failed(order)) { return false; }
        auto candidate = pair->intersect(*order);
        if (failed(candidate) || !covered(*candidate, query, required)) { return false; }
        // Different canonical residues cannot denote equal original coordinates.
        if (a.source.residues[i] != b.source.residues[i]) { break; }
        prefix.push_back(difference(dimensions, i, tuple + i, BoundInteger(0)));
        prefix.push_back(difference(dimensions, tuple + i, i, BoundInteger(0)));
    }
    return true;
}
} // namespace
DictionaryAttr generalArithmeticAllocationCertificate(
    const GeneralArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes,
    const std::map<Pair, int64_t>& records, int64_t plan, MLIRContext* context)
{
    if (!context || !analysis.error.empty() || !analysis.exactMinimum || !analysis.period ||
        analysis.period > INT64_MAX) { return {}; }
    std::map<Pair, std::vector<Entry>> families;
    for (const auto& [key, pieces] : analysis.minimumDemands) {
        if (!validKey(key, analysis, pipes)) { return {}; }
        const auto dimensions = key.source.residues.size() + key.target.residues.size() + analysis.parameterCount;
        for (const auto& piece : pieces) {
            if (piece.dimensions() != dimensions) { return {}; }
            if (pipes[key.source.site] != pipes[key.target.site] && !piece.isEmpty()) {
                families[{key.source.site, key.target.site}].push_back({&key, &piece});
            }
        }
    }
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> groups;
    std::set<uint32_t> used;
    for (const auto& [family, entries] : families) {
        auto record = records.find(family);
        if (record == records.end() || record->second < 0 || static_cast<uint64_t>(record->second) > UINT32_MAX) {
            return {};
        }
        const auto id = static_cast<uint32_t>(record->second);
        if (!used.insert(id).second) { return {}; }
        for (const auto& a : entries) {
            for (const auto& b : entries) {
                if (!orderedReuse(analysis, a, b)) { return {}; }
            }
        }
        groups[{pipes[family.first], pipes[family.second]}].push_back(id);
    }
    PeriodicAllocation allocation;
    for (const auto& [direction, ids] : groups) {
        DirectedAllocation group;
        group.sourcePipe = direction.first; group.targetPipe = direction.second;
        group.uniformBudget = ids.size();
        for (auto id : ids) {
            DirectedHandoff handoff; handoff.record = id;
            group.handoffs.push_back(handoff);
        }
        allocation.directions.push_back(std::move(group));
    }
    NamedAttrList attributes(encodeCyclicAllocation(allocation, plan, context));
    attributes.set("strategy", StringAttr::get(context, "dedicated-families"));
    return attributes.getDictionary(context);
}
} // namespace mlir::pto::frontiersynch
