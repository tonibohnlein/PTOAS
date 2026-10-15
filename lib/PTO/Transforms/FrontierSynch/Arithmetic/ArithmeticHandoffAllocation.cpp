// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/Arithmetic/ArithmeticHandoffAllocation.h"
#include "ArithmeticHandoffProof.h"
#include "PTO/Transforms/FrontierSynch/Shared/AllocationProfile.h"
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
struct CompositionFrame {
    unsigned total;
    std::vector<unsigned> left, right, keep;
    CompositionFrame(const Key& a, const Key& b, unsigned parameters)
    {
        const unsigned x = a.source.residues.size(), z = a.target.residues.size();
        const unsigned y = b.target.residues.size();
        total = x + z + y + parameters;
        left.resize(x + z + parameters); right.resize(z + y + parameters); keep.resize(x + y + parameters);
        std::iota(left.begin(), left.begin() + x + z, 0);
        std::iota(left.begin() + x + z, left.end(), x + z + y);
        std::iota(right.begin(), right.begin() + z + y, x);
        std::iota(right.begin() + z + y, right.end(), x + z + y);
        std::iota(keep.begin(), keep.begin() + x, 0);
        std::iota(keep.begin() + x, keep.end(), x + z);
    }
};
class Proof {
public:
    Proof(uint64_t period, unsigned parameters, ArithmeticAnalysisCost& cost, AllocationProfile* profile)
        : period(period), parameters(parameters), cost(cost), profile(profile) {}
    FailureOr<Relation> compose(const Relation& a, const Relation& b);
    FailureOr<bool> compositionCovered(const Relation& a, const Relation& b, const Relation& required);
    uint64_t containmentQueries = 0;
    FailureOr<Relation> subtract(const Relation& a, const Relation& b);
    FailureOr<Relation> endpoints(const Relation& a, bool target);
    FailureOr<Relation> order(const Relation& support, bool sourceAlreadyPresent = false);
    FailureOr<bool> nonnegativeCoordinate(const Relation& support);
    FailureOr<Relation> coordinateDistance(const Relation& order, unsigned distance);
    FailureOr<GeneralArithmeticEndpointSelector> extremum(const Relation& support, bool first);
    FailureOr<Relation> inverse(const Relation& demands);
    FailureOr<bool> orderedConsumers(const Relation& demands, const Relation& inverse,
                                     const Relation& successor, const Relation& targets);
private:
    uint64_t period;
    unsigned parameters;
    ArithmeticAnalysisCost& cost;
    AllocationProfile* profile;
    FailureOr<Systems> project(const IntegerSystem& system, ArrayRef<unsigned> keep);
};
FailureOr<Systems> Proof::project(const IntegerSystem& system, ArrayRef<unsigned> keep)
{
    AllocationProfile::Scope timing(profile, AllocationProfile::ArithmeticProjection);
    ++cost.projections;
    return system.project(keep);
}
FailureOr<Relation> Proof::compose(const Relation& a, const Relation& b)
{
    AllocationProfile::Scope timing(profile, AllocationProfile::ArithmeticComposition);
    ++cost.relationCompositions;
    Relation output;
    for (const auto& [ak, av] : a) {
        for (const auto& [bk, bv] : b) {
            if (!(ak.target == bk.source) || ak.parameterResidues != bk.parameterResidues) { continue; }
            const CompositionFrame frame(ak, bk, parameters);
            for (const auto& left : av) {
                auto expanded = left.remap(frame.total, frame.left);
                if (failed(expanded)) { return failure(); }
                for (const auto& right : bv) {
                    auto other = right.remap(frame.total, frame.right);
                    if (failed(other)) { return failure(); }
                    auto joined = expanded->intersect(*other); ++cost.pieceJoins;
                    if (failed(joined)) { return failure(); }
                    auto projected = project(*joined, frame.keep);
                    if (failed(projected)) { return failure(); }
                    append(output, {ak.source, bk.target, ak.parameterResidues}, std::move(*projected));
                }
            }
        }
    }
    return output;
}
FailureOr<bool> Proof::compositionCovered(const Relation& a, const Relation& b, const Relation& required)
{
    AllocationProfile::Scope timing(profile, AllocationProfile::ArithmeticContainment);
    ++containmentQueries;
    for (const auto& [ak, av] : a) {
        for (const auto& [bk, bv] : b) {
            if (!(ak.target == bk.source) || ak.parameterResidues != bk.parameterResidues) { continue; }
            const CompositionFrame frame(ak, bk, parameters);
            Systems cover;
            auto found = required.find({ak.source, bk.target, ak.parameterResidues});
            if (found != required.end()) {
                for (const auto& piece : found->second) {
                    auto lifted = piece.remap(frame.total, frame.keep);
                    if (failed(lifted)) { return failure(); }
                    cover.push_back(std::move(*lifted));
                }
            }
            for (const auto& left : av) {
                auto expanded = left.remap(frame.total, frame.left);
                if (failed(expanded)) { return failure(); }
                for (const auto& right : bv) {
                    auto other = right.remap(frame.total, frame.right);
                    if (failed(other)) { return failure(); }
                    auto joined = expanded->intersect(*other); ++cost.pieceJoins;
                    if (failed(joined)) { return failure(); }
                    // Projection(J) is covered iff J is covered by the entire
                    // lifted required union. Retain shared parameter columns;
                    // no intermediate witness or projected relation is needed.
                    AllocationProfile::Scope differenceTiming(profile, AllocationProfile::ArithmeticDifference);
                    ++cost.differences;
                    auto missing = subtractIntegerUnions(frame.total, {*joined}, cover);
                    if (failed(missing)) { return failure(); }
                    if (!missing->empty()) { return false; }
                }
            }
        }
    }
    return true;
}
FailureOr<Relation> Proof::subtract(const Relation& a, const Relation& b)
{
    AllocationProfile::Scope timing(profile, AllocationProfile::ArithmeticDifference);
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
    AllocationProfile::Scope timing(profile, AllocationProfile::ArithmeticProjection);
    Relation result;
    for (const auto& [key, pieces] : a) {
        const unsigned s = key.source.residues.size(), t = key.target.residues.size();
        auto endpoint = target ? key.target : key.source;
        endpoint.event = ArithmeticEvent::Start;
        std::vector<unsigned> keep(endpoint.residues.size() + parameters);
        std::iota(keep.begin(), keep.begin() + endpoint.residues.size(), target ? s : 0);
        std::iota(keep.begin() + endpoint.residues.size(), keep.end(), s + t);
        for (const auto& piece : pieces) {
            auto projected = project(piece, keep);
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
FailureOr<Relation> Proof::order(const Relation& support, bool sourceAlreadyPresent)
{
    AllocationProfile::Scope timing(profile, AllocationProfile::ArithmeticOrder);
    Relation result;
    for (const auto& [a, av] : support) {
        for (const auto& [b, bv] : support) {
            if (a.parameterResidues != b.parameterResidues) { continue; }
            const unsigned n = a.source.residues.size(), total = 2 * n + parameters;
            if (a.source.site != b.source.site || b.source.residues.size() != n) { return failure(); }
            // In inverse(F*) ; order, inverse(F*) already establishes the
            // intermediate source's exact support, including parameters. The
            // coordinate proof only consumes order through that composition.
            // Keep the later-source support; avoid multiplying the earlier
            // source's projected pieces into the join a second time.
            SmallVector<IntegerSystem, 1> universal;
            if (sourceAlreadyPresent) {
                auto domain = IntegerSystem::create(n + parameters, {});
                if (failed(domain)) { return failure(); }
                universal.push_back(std::move(*domain));
            }
            const ArrayRef<IntegerSystem> leftPieces = sourceAlreadyPresent ?
                ArrayRef<IntegerSystem>(universal) : ArrayRef<IntegerSystem>(av);
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
                for (const auto& left : leftPieces) {
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
    AllocationProfile::Scope timing(profile, AllocationProfile::ArithmeticExtrema);
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
FailureOr<bool> Proof::nonnegativeCoordinate(const Relation& support)
{
    for (const auto& [key, pieces] : support) {
        if (key.source.residues.size() != 1) { return false; }
        // Original IV = period*q + residue, 0 <= residue < period, so IV < 0
        // iff q <= -1. Parameters remain unconstrained except by the support.
        IntegerConstraint row{std::vector<BoundInteger>(1 + parameters, BoundInteger(0)), BoundInteger(-1)};
        row.coefficients[0] = BoundInteger(1);
        auto negative = IntegerSystem::create(1 + parameters, {row});
        if (failed(negative)) { return failure(); }
        for (const auto& piece : pieces) {
            auto overlap = piece.intersect(*negative);
            if (failed(overlap)) { return failure(); }
            if (!overlap->isEmpty()) { return false; }
        }
    }
    return true;
}
FailureOr<Relation> Proof::coordinateDistance(const Relation& order, unsigned distance)
{
    Relation result;
    for (const auto& [key, pieces] : order) {
        if (key.source.residues.size() != 1 || key.target.residues.size() != 1) { return failure(); }
        auto bound = floorDiv(BoundInteger(key.target.residues[0]) -
            BoundInteger(key.source.residues[0]) - BoundInteger(distance), BoundInteger(period));
        auto separated = IntegerSystem::create(2 + parameters, {difference(2 + parameters, 0, 1, bound)});
        if (failed(separated)) { return failure(); }
        for (const auto& piece : pieces) {
            auto restricted = piece.intersect(*separated);
            if (failed(restricted)) { return failure(); }
            append(result, key, {*restricted});
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
// Diagnostic state describes the last attempted exact operation, not a new
// admission condition. A failed sufficient reuse proof is not a capacity bound.
struct FamilyProgress {
    const char* stage = "supports";
    const char* outcome = "construction-unavailable";
};
FailureOr<ArithmeticHandoffFamily> family(Proof& proof, const Pair& sites, const Relation& demands,
                                        const Relation& required, unsigned maxWidth, bool coordinateMapping,
                                        FamilyProgress& progress)
{
    auto source = proof.endpoints(demands, false), target = proof.endpoints(demands, true);
    if (failed(source) || failed(target)) { return failure(); }
    progress.stage = "source-order-and-inverse";
    auto order = proof.order(*source, coordinateMapping), inverse = proof.inverse(demands);
    if (failed(order) || failed(inverse)) { return failure(); }
    // Preserve the inexpensive one-ID proof before constructing successors.
    // Consecutive reuse implies all later reuse by transitivity of H, so this
    // all-pairs test has the same obligation for the certified strict order.
    unsigned width = 0;
    if (coordinateMapping) {
        progress.stage = "one-id-containment";
        auto covered = proof.compositionCovered(*inverse, *order, required);
        if (failed(covered)) { return failure(); }
        width = *covered ? 1 : 0;
    } else {
        progress.stage = "one-id-composition";
        auto allReuse = proof.compose(*inverse, *order);
        if (failed(allReuse)) { return failure(); }
        progress.stage = "one-id-difference";
        auto uncovered = proof.subtract(*allReuse, required);
        if (failed(uncovered)) { return failure(); }
        width = uncovered->empty() ? 1 : 0;
    }
    if (!width && maxWidth > 1 && coordinateMapping) {
        progress.stage = "coordinate-nonnegative";
        auto nonnegative = proof.nonnegativeCoordinate(*source);
        if (failed(nonnegative)) { return failure(); }
        if (!*nonnegative) { progress.outcome = "coordinate-premise-not-proved"; return failure(); }
        for (unsigned candidate = 2; candidate <= maxWidth; ++candidate) {
            progress.stage = "coordinate-distance";
            auto separated = proof.coordinateDistance(*order, candidate);
            if (failed(separated)) { return failure(); }
            progress.stage = "coordinate-containment";
            auto covered = proof.compositionCovered(*inverse, *separated, required);
            if (failed(covered)) { return failure(); }
            if (*covered) { width = candidate; break; }
        }
    } else if (!width && maxWidth > 1) {
        progress.stage = "successor-square";
        auto square = proof.compose(*order, *order);
        if (failed(square)) { return failure(); }
        progress.stage = "successor-difference";
        auto successor = proof.subtract(*order, *square);
        if (failed(successor)) { return failure(); }
        // Separate SET/WAIT counters denote the same handoff rank only if
        // matching preserves order. F* already supplies unique partners.
        progress.stage = "consumer-order";
        auto monotone = proof.orderedConsumers(demands, *inverse, *successor, *target);
        if (failed(monotone)) { return failure(); }
        if (!*monotone) { progress.outcome = "consumer-order-not-proved"; return failure(); }
        Relation power = *successor;
        for (unsigned candidate = 2; candidate <= maxWidth; ++candidate) {
            progress.stage = "successor-power";
            auto next = proof.compose(power, *successor);
            if (failed(next)) { return failure(); }
            power = std::move(*next);
            progress.stage = "executed-rank-composition";
            auto reuse = proof.compose(*inverse, power);
            if (failed(reuse)) { return failure(); }
            progress.stage = "executed-rank-difference";
            auto missing = proof.subtract(*reuse, required);
            if (failed(missing)) { return failure(); }
            if (missing->empty()) { width = candidate; break; }
        }
    }
    if (!width) { progress.outcome = "reuse-uncovered-within-bound"; return failure(); }
    progress.stage = "lifetime-extrema";
    auto first = proof.extremum(*source, true), last = proof.extremum(*target, false);
    if (failed(first) || failed(last)) { return failure(); }
    return ArithmeticHandoffFamily{sites.first, sites.second, width, std::move(*first), std::move(*last)};
}
ArithmeticHandoffAllocation buildAllocation(
    const GeneralArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes, unsigned maxWidth,
    bool coordinateMapping, AllocationProfile* profile)
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
    Proof proof(analysis.period, analysis.parameterCount, result.cost, profile);
    auto recordWork = [&](std::optional<Pair> failedFamily) {
        if (!profile || !profile->enabled()) { return; }
        std::vector<std::pair<std::string, uint64_t>> work{
            {"families", families.size()}, {"proved-families", result.families.size()},
            {"piece-joins", result.cost.pieceJoins}, {"projections", result.cost.projections},
            {"compositions", result.cost.relationCompositions}, {"differences", result.cost.differences},
            {"containment-queries", proof.containmentQueries}};
        if (failedFamily) {
            work.emplace_back("failed-source", failedFamily->first);
            work.emplace_back("failed-target", failedFamily->second);
        }
        profile->recordWork("arithmetic-handoff-construction", std::move(work));
    };
    for (const auto& [sites, demands] : families) {
        if (demands.empty()) { continue; }
        FamilyProgress progress;
        auto built = family(proof, sites, demands, analysis.requiredOrder, maxWidth, coordinateMapping, progress);
        if (failed(built)) {
            recordWork(sites);
            result.error = "no certified uniform arithmetic handoff width within the supplied bound: family " +
                std::to_string(sites.first) + " -> " + std::to_string(sites.second) + ", stage " + progress.stage +
                ", outcome " + progress.outcome;
            if (profile && profile->enabled()) {
                const auto method =
                    std::string("arithmetic-handoff-refusal:") + progress.stage + ":" + progress.outcome;
                profile->recordWork(method,
                    {{"source", sites.first}, {"target", sites.second}, {"max-width", maxWidth},
                     {"coordinate-policy", coordinateMapping}});
            }
            result.families.clear(); return result;
        }
        result.families.push_back(std::move(*built));
    }
    recordWork(std::nullopt);
    return result;
}
} // namespace
FailureOr<GeneralArithmeticRelation> detail::materializeHandoffComposition(const GeneralArithmeticRelation& left,
    const GeneralArithmeticRelation& right, unsigned parameters, ArithmeticAnalysisCost& cost)
{
    return Proof(1, parameters, cost, nullptr).compose(left, right);
}
FailureOr<bool> detail::handoffCompositionCovered(const GeneralArithmeticRelation& left,
    const GeneralArithmeticRelation& right, const GeneralArithmeticRelation& required,
    unsigned parameters, ArithmeticAnalysisCost& cost, AllocationProfile* profile)
{
    return Proof(1, parameters, cost, profile).compositionCovered(left, right, required);
}
ArithmeticHandoffAllocation buildArithmeticHandoffAllocation(
    const GeneralArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes, unsigned maxWidth,
    AllocationProfile* profile)
{
    return buildAllocation(analysis, pipes, maxWidth, false, profile);
}
ArithmeticCoordinateHandoffAllocation buildArithmeticCoordinateHandoffAllocation(
    const GeneralArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes, unsigned maxWidth,
    AllocationProfile* profile)
{
    auto proof = buildAllocation(analysis, pipes, maxWidth, true, profile);
    ArithmeticCoordinateHandoffAllocation result;
    result.cost = std::move(proof.cost); result.error = std::move(proof.error);
    for (auto& family : proof.families) {
        auto coordinate = family.width > 1 ? std::optional<unsigned>(0) : std::nullopt;
        result.families.push_back({std::move(family), coordinate});
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
    const ArithmeticDemandAnalysis& analysis, ArrayRef<uint32_t> pipes, unsigned maxWidth, AllocationProfile* profile)
{
    GeneralArithmeticDemandAnalysis converted;
    converted.period = analysis.period; converted.parameterCount = analysis.parameterCount;
    converted.exactMinimum = analysis.exactMinimum; converted.error = analysis.error;
    auto demands = convert(analysis.minimumDemands), required = convert(analysis.requiredOrder);
    if (failed(demands) || failed(required)) {
        ArithmeticHandoffAllocation result; result.error = "completed arithmetic DBM conversion failed"; return result;
    }
    converted.minimumDemands = std::move(*demands); converted.requiredOrder = std::move(*required);
    return buildArithmeticHandoffAllocation(converted, pipes, maxWidth, profile);
}
} // namespace mlir::pto::frontiersynch
