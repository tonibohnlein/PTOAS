// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticHandoffAllocation.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <numeric>
namespace mlir::pto::frontiersynch {
namespace {
using Relation = GeneralArithmeticRelation;
using Systems = std::vector<IntegerSystem>;
using Key = ArithmeticRelationKey;
using Pair = std::pair<std::size_t, std::size_t>;
void append(Relation& output, const Key& key, Systems pieces)
{
    for (auto& piece : pieces) {
        if (!piece.isEmpty()) { output[key].push_back(std::move(piece)); }
    }
}
unsigned dimensions(const Key& key)
{
    return key.source.residues.size() + key.target.residues.size() + key.parameterResidues.size();
}
class Proof {
public:
    Proof(uint64_t period, unsigned parameters, ArithmeticAnalysisCost& cost)
        : period(period), parameters(parameters), cost(cost) {}
    FailureOr<Relation> compose(const Relation& a, const Relation& b);
    FailureOr<Relation> subtract(const Relation& a, const Relation& b);
    FailureOr<Relation> endpoints(const Relation& a, bool target);
    FailureOr<Relation> order(const Relation& support);
    FailureOr<GeneralArithmeticEndpointSelector> extremum(const Relation& support, bool first);
    FailureOr<Relation> inverse(const Relation& demands);
    FailureOr<bool> orderedConsumers(const Relation& demands, const Relation& inverse,
                                     const Relation& successor, const Relation& targets);
private:
    uint64_t period;
    unsigned parameters;
    ArithmeticAnalysisCost& cost;
};
FailureOr<Relation> Proof::compose(const Relation& a, const Relation& b)
{
    ++cost.relationCompositions;
    Relation output;
    for (const auto& [ak, av] : a) {
        for (const auto& [bk, bv] : b) {
            if (!(ak.target == bk.source) || ak.parameterResidues != bk.parameterResidues) { continue; }
            const unsigned x = ak.source.residues.size(), z = ak.target.residues.size();
            const unsigned y = bk.target.residues.size(), total = x + z + y + parameters;
            std::vector<unsigned> am(x + z + parameters), bm(z + y + parameters), keep(x + y + parameters);
            std::iota(am.begin(), am.begin() + x + z, 0);
            std::iota(am.begin() + x + z, am.end(), x + z + y);
            std::iota(bm.begin(), bm.begin() + z + y, x);
            std::iota(bm.begin() + z + y, bm.end(), x + z + y);
            std::iota(keep.begin(), keep.begin() + x, 0);
            std::iota(keep.begin() + x, keep.end(), x + z);
            for (const auto& left : av) {
                auto expanded = left.remap(total, am);
                if (failed(expanded)) { return failure(); }
                for (const auto& right : bv) {
                    auto other = right.remap(total, bm);
                    if (failed(other)) { return failure(); }
                    auto joined = expanded->intersect(*other); ++cost.pieceJoins;
                    if (failed(joined)) { return failure(); }
                    auto projected = joined->project(keep); ++cost.projections;
                    if (failed(projected)) { return failure(); }
                    append(output, {ak.source, bk.target, ak.parameterResidues}, std::move(*projected));
                }
            }
        }
    }
    return output;
}
FailureOr<Relation> Proof::subtract(const Relation& a, const Relation& b)
{
    Relation result;
    for (const auto& [key, pieces] : a) {
        auto found = b.find(key);
        if (found == b.end()) { append(result, key, pieces); continue; }
        auto remaining = subtractIntegerUnions(dimensions(key), pieces, found->second); ++cost.differences;
        if (failed(remaining)) { return failure(); }
        append(result, key, std::move(*remaining));
    }
    return result;
}
// Unary supports use an empty dummy target. All real endpoints are normalized
// to Start here; event kinds are restored only in the consumption reuse query.
FailureOr<Relation> Proof::endpoints(const Relation& a, bool target)
{
    Relation result;
    for (const auto& [key, pieces] : a) {
        const unsigned s = key.source.residues.size(), t = key.target.residues.size();
        auto endpoint = target ? key.target : key.source;
        endpoint.event = ArithmeticEvent::Start;
        std::vector<unsigned> keep(endpoint.residues.size() + parameters);
        std::iota(keep.begin(), keep.begin() + endpoint.residues.size(), target ? s : 0);
        std::iota(keep.begin() + endpoint.residues.size(), keep.end(), s + t);
        for (const auto& piece : pieces) {
            auto projected = piece.project(keep); ++cost.projections;
            if (failed(projected)) { return failure(); }
            append(result, {endpoint, {}, key.parameterResidues}, std::move(*projected));
        }
    }
    return result;
}
IntegerConstraint difference(unsigned count, unsigned a, unsigned b, BoundInteger bound)
{
    IntegerConstraint row{std::vector<BoundInteger>(count, BoundInteger(0)), std::move(bound)};
    row.coefficients[a] = BoundInteger(1); row.coefficients[b] = BoundInteger(-1);
    return row;
}
FailureOr<Relation> Proof::order(const Relation& support)
{
    Relation result;
    for (const auto& [a, av] : support) {
        for (const auto& [b, bv] : support) {
            if (a.parameterResidues != b.parameterResidues) { continue; }
            const unsigned n = a.source.residues.size(), total = 2 * n + parameters;
            if (a.source.site != b.source.site || b.source.residues.size() != n) { return failure(); }
            std::vector<unsigned> am(n + parameters), bm(n + parameters);
            std::iota(am.begin(), am.begin() + n, 0); std::iota(bm.begin(), bm.begin() + n, n);
            std::iota(am.begin() + n, am.end(), 2 * n); std::iota(bm.begin() + n, bm.end(), 2 * n);
            SmallVector<IntegerConstraint> prefix;
            for (unsigned i = 0; i < n; ++i) {
                auto rows = prefix;
                auto bound = BoundInteger(b.source.residues[i]) - BoundInteger(a.source.residues[i]) - 1;
                rows.push_back(difference(total, i, n + i, floorDiv(bound, BoundInteger(period))));
                auto lex = IntegerSystem::create(total, rows);
                if (failed(lex)) { return failure(); }
                for (const auto& left : av) {
                    auto x = left.remap(total, am);
                    if (failed(x)) { return failure(); }
                    for (const auto& right : bv) {
                        auto y = right.remap(total, bm);
                        if (failed(y)) { return failure(); }
                        auto pair = x->intersect(*y); ++cost.pieceJoins;
                        if (failed(pair)) { return failure(); }
                        auto ordered = pair->intersect(*lex);
                        if (failed(ordered)) { return failure(); }
                        append(result, {a.source, b.source, a.parameterResidues}, {*ordered});
                    }
                }
                if (a.source.residues[i] != b.source.residues[i]) { break; }
                prefix.push_back(difference(total, i, n + i, BoundInteger(0)));
                prefix.push_back(difference(total, n + i, i, BoundInteger(0)));
            }
        }
    }
    return result;
}
FailureOr<GeneralArithmeticEndpointSelector> Proof::extremum(const Relation& support, bool first)
{
    auto ordered = order(support);
    if (failed(ordered)) { return failure(); }
    auto dominated = endpoints(*ordered, first);
    if (failed(dominated)) { return failure(); }
    auto selected = subtract(support, *dominated);
    if (failed(selected)) { return failure(); }
    GeneralArithmeticEndpointSelector result;
    result.parameterCount = parameters;
    for (const auto& [key, pieces] : *selected) {
        const unsigned n = key.source.residues.size();
        std::vector<unsigned> map(n + parameters);
        std::iota(map.begin(), map.begin() + n, parameters);
        std::iota(map.begin() + n, map.end(), 0);
        for (const auto& piece : pieces) {
            auto reordered = piece.remap(n + parameters, map);
            if (failed(reordered)) { return failure(); }
            auto witnesses = buildIntegerTupleWitnesses(*reordered, parameters, key.source.residues);
            if (failed(witnesses)) { return failure(); }
            for (auto& witness : *witnesses) {
                result.pieces.push_back({std::move(witness.domain), {}, key.parameterResidues,
                                         key.source.site, std::move(witness.outputs)});
            }
        }
    }
    return result;
}
FailureOr<Relation> Proof::inverse(const Relation& demands)
{
    Relation result;
    for (const auto& [key, pieces] : demands) {
        const unsigned s = key.source.residues.size(), t = key.target.residues.size();
        std::vector<unsigned> map(s + t + parameters);
        std::iota(map.begin(), map.begin() + s, t); std::iota(map.begin() + s, map.begin() + s + t, 0);
        std::iota(map.begin() + s + t, map.end(), s + t);
        auto source = key.target, target = key.source;
        source.event = ArithmeticEvent::Completion; target.event = ArithmeticEvent::Start;
        for (const auto& piece : pieces) {
            auto value = piece.remap(map.size(), map);
            if (failed(value)) { return failure(); }
            append(result, {source, target, key.parameterResidues}, {*value});
        }
    }
    return result;
}
FailureOr<bool> Proof::orderedConsumers(const Relation& demands, const Relation& inverse,
                                        const Relation& successor, const Relation& targets)
{
    Relation forward;
    for (const auto& [key, pieces] : demands) {
        auto from = key.source, to = key.target;
        from.event = ArithmeticEvent::Start; to.event = ArithmeticEvent::Completion;
        append(forward, {from, to, key.parameterResidues}, pieces);
    }
    auto next = compose(inverse, successor);
    if (failed(next)) { return failure(); }
    auto consumers = compose(*next, forward), targetOrder = order(targets);
    if (failed(consumers) || failed(targetOrder)) { return failure(); }
    Relation completionOrder;
    for (const auto& [key, pieces] : *targetOrder) {
        auto from = key.source, to = key.target;
        from.event = ArithmeticEvent::Completion; to.event = ArithmeticEvent::Completion;
        append(completionOrder, {from, to, key.parameterResidues}, pieces);
    }
    auto missing = subtract(*consumers, completionOrder);
    if (failed(missing)) { return failure(); }
    return missing->empty();
}
bool valid(const Key& key, const GeneralArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes)
{
    // Every intermediate join has at most three endpoint tuples plus context.
    const auto limit = (std::numeric_limits<unsigned>::max() - analysis.parameterCount) / 3;
    auto residues = [&](ArrayRef<uint64_t> values) {
        return llvm::all_of(values, [&](uint64_t value) { return value < analysis.period; });
    };
    return key.source.site < pipes.size() && key.target.site < pipes.size() &&
        key.source.event == ArithmeticEvent::Completion && key.target.event == ArithmeticEvent::Start &&
        key.source.residues.size() <= limit && key.target.residues.size() <= limit &&
        key.parameterResidues.size() == analysis.parameterCount && residues(key.source.residues) &&
        residues(key.target.residues) && residues(key.parameterResidues);
}
FailureOr<ArithmeticHandoffFamily> family(Proof& proof, const Pair& sites, const Relation& demands,
                                        const Relation& required, unsigned maxWidth)
{
    auto source = proof.endpoints(demands, false), target = proof.endpoints(demands, true);
    if (failed(source) || failed(target)) { return failure(); }
    auto order = proof.order(*source), inverse = proof.inverse(demands);
    if (failed(order) || failed(inverse)) { return failure(); }
    // Preserve the inexpensive one-ID proof before constructing successors.
    // Consecutive reuse implies all later reuse by transitivity of H, so this
    // all-pairs test has the same obligation for the certified strict order.
    auto allReuse = proof.compose(*inverse, *order);
    if (failed(allReuse)) { return failure(); }
    auto uncovered = proof.subtract(*allReuse, required);
    if (failed(uncovered)) { return failure(); }
    unsigned width = uncovered->empty() ? 1 : 0;
    if (!width && maxWidth > 1) {
        auto square = proof.compose(*order, *order);
        if (failed(square)) { return failure(); }
        auto successor = proof.subtract(*order, *square);
        if (failed(successor)) { return failure(); }
        // Separate SET/WAIT counters denote the same handoff rank only if
        // matching preserves order. F* already supplies unique partners.
        auto monotone = proof.orderedConsumers(demands, *inverse, *successor, *target);
        if (failed(monotone) || !*monotone) { return failure(); }
        Relation power = *successor;
        for (unsigned candidate = 2; candidate <= maxWidth; ++candidate) {
            auto next = proof.compose(power, *successor);
            if (failed(next)) { return failure(); }
            power = std::move(*next);
            auto reuse = proof.compose(*inverse, power);
            if (failed(reuse)) { return failure(); }
            auto missing = proof.subtract(*reuse, required);
            if (failed(missing)) { return failure(); }
            if (missing->empty()) { width = candidate; break; }
        }
    }
    if (!width) { return failure(); }
    auto first = proof.extremum(*source, true), last = proof.extremum(*target, false);
    if (failed(first) || failed(last)) { return failure(); }
    return ArithmeticHandoffFamily{sites.first, sites.second, width, std::move(*first), std::move(*last)};
}
} // namespace
ArithmeticHandoffAllocation buildArithmeticHandoffAllocation(
    const GeneralArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes, unsigned maxWidth)
{
    ArithmeticHandoffAllocation result;
    result.period = analysis.period; result.parameterCount = analysis.parameterCount;
    if (!analysis.error.empty() || !analysis.exactMinimum || !analysis.period ||
        analysis.period > INT64_MAX || !maxWidth || maxWidth > 6) {
        result.error = "arithmetic allocation requires exact demands and a width in 1..6"; return result;
    }
    std::map<Pair, Relation> families;
    for (const auto& [key, pieces] : analysis.minimumDemands) {
        if (!valid(key, analysis, pipes)) {
            result.error = "arithmetic allocation endpoint schema is invalid"; return result;
        }
        for (const auto& piece : pieces) {
            if (piece.dimensions() != dimensions(key)) {
                result.error = "arithmetic allocation relation dimensions disagree"; return result;
            }
        }
        if (pipes[key.source.site] != pipes[key.target.site]) {
            append(families[{key.source.site, key.target.site}], key, pieces);
        }
    }
    Proof proof(analysis.period, analysis.parameterCount, result.cost);
    for (const auto& [sites, demands] : families) {
        if (demands.empty()) { continue; }
        auto built = family(proof, sites, demands, analysis.requiredOrder, maxWidth);
        if (failed(built)) {
            result.error = "no certified uniform arithmetic handoff width within the supplied bound";
            result.families.clear(); return result;
        }
        result.families.push_back(std::move(*built));
    }
    return result;
}
namespace {
FailureOr<GeneralArithmeticRelation> convert(const ArithmeticRelation& relation)
{
    GeneralArithmeticRelation result;
    for (const auto& [key, pieces] : relation) {
        for (const auto& piece : pieces) {
            SmallVector<IntegerConstraint> rows;
            if (piece.isEmpty()) { continue; }
            for (const auto& atom : piece.constraints()) {
                IntegerConstraint row{std::vector<BoundInteger>(piece.dimensions(), BoundInteger(0)), atom.bound};
                if (atom.lhs > piece.dimensions() || atom.rhs > piece.dimensions()) { return failure(); }
                if (atom.lhs) { row.coefficients[atom.lhs - 1] += 1; }
                if (atom.rhs) { row.coefficients[atom.rhs - 1] -= 1; }
                rows.push_back(std::move(row));
            }
            auto system = IntegerSystem::create(piece.dimensions(), rows);
            if (failed(system)) { return failure(); }
            append(result, key, {*system});
        }
    }
    return result;
}
} // namespace
ArithmeticHandoffAllocation buildArithmeticHandoffAllocation(
    const ArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes, unsigned maxWidth)
{
    GeneralArithmeticDemandAnalysis converted;
    converted.period = analysis.period; converted.parameterCount = analysis.parameterCount;
    converted.exactMinimum = analysis.exactMinimum; converted.error = analysis.error;
    auto demands = convert(analysis.minimumDemands), required = convert(analysis.requiredOrder);
    if (failed(demands) || failed(required)) {
        ArithmeticHandoffAllocation result; result.error = "completed arithmetic DBM conversion failed"; return result;
    }
    converted.minimumDemands = std::move(*demands); converted.requiredOrder = std::move(*required);
    return buildArithmeticHandoffAllocation(converted, pipes, maxWidth);
}
} // namespace mlir::pto::frontiersynch
