// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Partition one counted loop at finitely many comparison thresholds. Each
// interval has an immutable skeleton and uses the ordinary periodic exporter.
// Their original ordinals/cuts survive; sequence composition supplies all
// cross-interval edges. No payloads or runtime iterations are cloned.
#include "SequenceAnalysisInternal.h"
#include "BoundarySlices.h"
#include "RecognitionInternal.h"
#include "ArithmeticRows.h"
#include "PTO/Transforms/FrontierSynch/GuardedRotatingRegional.h"
#include "../InsertSync/SyncScalarEvolution.h"
namespace mlir::pto::frontiersynch {
namespace {
struct CutBounds {
    std::optional<uint64_t> fromZero, toEnd;
};
std::optional<uint64_t> minimumBound(std::optional<uint64_t> a, std::optional<uint64_t> b)
{
    if (!a) { return b; }
    if (!b) { return a; }
    return std::min(*a, *b);
}
std::optional<uint64_t> maximumBound(std::optional<uint64_t> a, std::optional<uint64_t> b)
{
    return a && b ? std::optional<uint64_t>(std::max(*a, *b)) : std::nullopt;
}
struct Split {
    Expr ordinal;
    bool prefixThen;
    std::optional<Expr> equalityEnd;
    CutBounds ordinalBounds, endBounds;
    Expr predicate(Expr begin, RegionExpressions& dag) const
    {
        auto value = equalityEnd ? dag.land(dag.le(ordinal, begin), dag.lt(begin, *equalityEnd)) :
                                   dag.lt(begin, ordinal);
        return prefixThen ? value : dag.lnot(value);
    }
};
std::optional<Split> unitOffsetCondition(Value condition, scf::ForOp loop,
    const PhaseIndex& index, RegionExpressions& dag, Expr trips)
{
    auto compare = condition.getDefiningOp<arith::CmpIOp>();
    if (!compare || !compare.getLhs().getType().isIndex()) { return std::nullopt; }
    Value varying = compare.getLhs(), bound = compare.getRhs();
    auto predicate = compare.getPredicate();
    SmallVector<Operation*> recipe;
    if (!detail::entryExpression(bound, loop, index, recipe)) { return std::nullopt; }
    SmallVector<Value> symbols;
    mlir::pto::detail::ScalarEvolution evolution(loop.getContext(), loop);
    auto expression = evolution.value(varying, [&](Value value) {
        auto found = llvm::find(symbols, value);
        auto position = found - symbols.begin();
        if (found == symbols.end()) { symbols.push_back(value); }
        return getAffineSymbolExpr(position, loop.getContext());
    });
    LinearRow row;
    if (detail::collectRow(expression, 0, symbols.size(), row) || row.constant < 0) { return std::nullopt; }
    bool foundIV = false;
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        if (symbols[i] == loop.getInductionVar() && row.coefficients[i] == 1) { foundIV = true; }
        else if (row.coefficients[i]) { return std::nullopt; }
    }
    if (!foundIV) { return std::nullopt; }
    bool signedCompare = false, inclusive = false, prefixThen = true, equality = false;
    switch (predicate) {
    case arith::CmpIPredicate::slt: signedCompare = true; break;
    case arith::CmpIPredicate::ult: break;
    case arith::CmpIPredicate::sle: signedCompare = true; inclusive = true; break;
    case arith::CmpIPredicate::ule: inclusive = true; break;
    case arith::CmpIPredicate::sge: signedCompare = true; prefixThen = false; break;
    case arith::CmpIPredicate::uge: prefixThen = false; break;
    case arith::CmpIPredicate::sgt: signedCompare = true; inclusive = true; prefixThen = false; break;
    case arith::CmpIPredicate::ugt: inclusive = true; prefixThen = false; break;
    case arith::CmpIPredicate::eq: equality = true; break;
    case arith::CmpIPredicate::ne: equality = true; prefixThen = false; break;
    default: return std::nullopt;
    }
    // The varying expression is proved iv+c without machine wrap. Clip before
    // adding one, so inclusive comparisons remain total even at UINT64_MAX.
    auto zero = dag.constant(0), offset = dag.constant(row.constant), threshold = dag.input(bound);
    auto below = signedCompare ? dag.slt(threshold, offset) : dag.lt(threshold, offset);
    auto delta = dag.sub(threshold, offset);
    auto clipped = dag.minimum(delta, trips);
    auto next = dag.select(dag.lt(clipped, trips), dag.add(clipped, dag.constant(1)), trips);
    auto first = dag.select(below, zero, clipped);
    auto end = dag.select(below, zero, next);
    CutBounds firstBounds, endBounds;
    if (bound == loop.getUpperBound()) {
        // On the executed domain, comparing iv+c with the loop bound cuts at
        // max(trips-c,0). The equality interval ends at max(trips-c+1,0),
        // capped by trips. This also covers nonpositive source bounds, whose
        // domain is empty, and avoids opaque min/select identities at 0/T.
        auto subtract = [&](uint64_t amount) {
            if (!amount) { return trips; }
            auto distance = dag.constant(amount);
            return dag.select(dag.lt(trips, distance), zero, dag.sub(trips, distance));
        };
        first = subtract(row.constant);
        end = row.constant ? subtract(row.constant - 1) : trips;
        firstBounds.toEnd = static_cast<uint64_t>(row.constant);
        endBounds.toEnd = static_cast<uint64_t>(row.constant ? row.constant - 1 : 0);
    } else if (auto value = dag.constantValue(first)) {
        // first is already clipped to [0,trips]. A literal increment cannot
        // overflow unless first is UINT64_MAX; that endpoint already equals T.
        end = dag.select(below, zero,
            *value == UINT64_MAX ? first : dag.minimum(dag.constant(*value + 1), trips));
    }
    if (auto value = dag.constantValue(threshold)) {
        const bool negative = signedCompare && (*value > static_cast<uint64_t>(INT64_MAX));
        auto cap = negative || *value < static_cast<uint64_t>(row.constant) ? 0 : *value - row.constant;
        firstBounds.fromZero = cap;
        endBounds.fromZero = negative || *value < static_cast<uint64_t>(row.constant) ? 0 :
            (cap == UINT64_MAX ? cap : cap + 1);
    }
    if (equality) { return Split{first, prefixThen, end, firstBounds, endBounds}; }
    return Split{inclusive ? end : first, prefixThen, std::nullopt,
                 inclusive ? endBounds : firstBounds, {}};
}
// Inputs to integer circuits are signed index bits. Narrow integer SSA inputs
// have unsigned arena extension, so restore their signed value explicitly.
Expr signedEntry(Value value, RegionExpressions& dag)
{
    auto result = dag.input(value);
    auto type = dyn_cast<IntegerType>(value.getType());
    if (!type || type.getWidth() == 64 || type.getWidth() == 1) { return result; }
    const unsigned width = type.getWidth();
    return dag.select(dag.lt(result, dag.constant(uint64_t{1} << (width - 1))), result,
                      dag.sub(result, dag.constant(uint64_t{1} << width)));
}

// Clip floor(numerator/divisor) BEFORE interpreting its possibly out-of-domain
// low 64 bits as an ordinal. The wide integer circuit proves all intermediates
// fit; its division is total even in an unselected branch.
Expr clippedQuotient(const IntegerAffine& numerator, const BoundInteger& divisor,
    ArrayRef<Expr> inputs, Expr trips, RegionExpressions& dag)
{
    SmallVector<uint64_t> residues(inputs.size(), 0);
    auto low = IntegerSystem::create(inputs.size(),
        {IntegerConstraint{numerator.coefficients, -numerator.constant}});
    auto highInputs = llvm::to_vector(inputs);
    highInputs.push_back(trips);
    auto highCoefficients = numerator.coefficients;
    for (auto& coefficient : highCoefficients) { coefficient = -coefficient; }
    highCoefficients.push_back(divisor);
    auto high = IntegerSystem::create(highInputs.size(),
        {IntegerConstraint{std::move(highCoefficients), numerator.constant}});
    // These systems contain one well-formed inequality and cannot fail to
    // normalize. Still propagate a failed representation through the caller.
    if (failed(low) || failed(high)) { return RegionExpressions::invalid; }
    auto below = dag.integerPredicate(*low, inputs, 1, residues);
    auto above = dag.integerPredicate(*high, highInputs, 1, SmallVector<uint64_t>(highInputs.size(), 0));
    auto quotient = dag.integerWitness(numerator, divisor, inputs, 1, residues, 0);
    return dag.select(below, dag.constant(0), dag.select(above, trips, quotient));
}

std::optional<Split> affineCondition(Value condition, scf::ForOp loop,
    const PhaseIndex& index, RegionExpressions& dag, Expr trips)
{
    auto compare = condition.getDefiningOp<arith::CmpIOp>();
    if (!compare || !compare.getLhs().getType().isIndex()) { return std::nullopt; }
    bool inclusive = false, prefixThen = true, equality = false;
    switch (compare.getPredicate()) {
    case arith::CmpIPredicate::slt: break;
    case arith::CmpIPredicate::sle: inclusive = true; break;
    case arith::CmpIPredicate::sge: prefixThen = false; break;
    case arith::CmpIPredicate::sgt: inclusive = true; prefixThen = false; break;
    case arith::CmpIPredicate::eq: equality = true; break;
    case arith::CmpIPredicate::ne: equality = true; prefixThen = false; break;
    default: return std::nullopt; // Unsigned order keeps the existing iv+c proof.
    }
    SmallVector<Value> symbols;
    mlir::pto::detail::ScalarEvolution evolution(loop.getContext(), loop);
    auto symbol = [&](Value value) {
        auto found = llvm::find(symbols, value);
        auto position = found - symbols.begin();
        if (found == symbols.end()) { symbols.push_back(value); }
        return getAffineSymbolExpr(position, loop.getContext());
    };
    auto left = evolution.value(compare.getLhs(), symbol);
    auto right = evolution.value(compare.getRhs(), symbol);
    LinearRow lhs, rhs;
    if (detail::collectRow(left, 0, symbols.size(), lhs) ||
        detail::collectRow(right, 0, symbols.size(), rhs)) { return std::nullopt; }
    IntegerAffine difference;
    difference.constant = BoundInteger(lhs.constant) - BoundInteger(rhs.constant);
    BoundInteger slope(0);
    SmallVector<Expr> inputs;
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        auto coefficient = BoundInteger(lhs.coefficients[i]) - BoundInteger(rhs.coefficients[i]);
        if (coefficient == 0) { continue; }
        if (symbols[i] == loop.getInductionVar()) { slope += coefficient; continue; }
        SmallVector<Operation*> recipe;
        auto type = dyn_cast<IntegerType>(symbols[i].getType());
        if ((!symbols[i].getType().isIndex() && (!type || type.getWidth() > 64 || type.getWidth() == 1)) ||
            !detail::entryExpression(symbols[i], loop, index, recipe)) { return std::nullopt; }
        inputs.push_back(signedEntry(symbols[i], dag));
        difference.coefficients.push_back(coefficient);
    }
    if (slope == 0) { return std::nullopt; }
    // Normalize to an increasing affine function. Reversing a signed strict
    // inequality exchanges prefix/suffix AND the inclusive threshold.
    if (slope < 0) {
        slope = -slope;
        difference.constant = -difference.constant;
        for (auto& coefficient : difference.coefficients) { coefficient = -coefficient; }
        if (!equality) { inclusive = !inclusive; prefixThen = !prefixThen; }
    }
    // Check the circuit's fixed-width emission contract before interning:
    // an unsupported speculative route must not poison the shared arena.
    BoundInteger coefficientSum = slope;
    for (const auto& coefficient : difference.coefficients) { coefficientSum += llvm::abs(coefficient); }
    const BoundInteger inputMagnitude = BoundInteger(INT64_MAX) + 1;
    const BoundInteger circuitLimit = inputMagnitude * inputMagnitude;
    if (llvm::abs(difference.constant) + slope + 2 + coefficientSum * inputMagnitude >= circuitLimit) {
        return std::nullopt;
    }
    // For integral d(i)=a*i+c, first d>=0 is ceil(-c/a), and
    // first d>=1 is ceil((1-c)/a). Their interval is exactly d==0;
    // a nondivisible root therefore creates an empty equality slice.
    IntegerAffine numerator;
    numerator.constant = -difference.constant + slope - 1;
    for (const auto& coefficient : difference.coefficients) { numerator.coefficients.push_back(-coefficient); }
    auto first = clippedQuotient(numerator, slope, inputs, trips, dag);
    if (inclusive || equality) { numerator.constant += 1; }
    auto end = inclusive || equality ? clippedQuotient(numerator, slope, inputs, trips, dag) : first;
    if (!dag.constructionError().empty() || first == RegionExpressions::invalid || end == RegionExpressions::invalid) {
        return std::nullopt;
    }
    return equality ? Split{first, prefixThen, end} : Split{end, prefixThen, std::nullopt};
}

std::optional<Split> splitCondition(Value condition, scf::ForOp loop,
    const PhaseIndex& index, RegionExpressions& dag, Expr trips)
{
    if (auto simple = unitOffsetCondition(condition, loop, index, dag, trips)) { return simple; }
    return affineCondition(condition, loop, index, dag, trips);
}

} // namespace
std::optional<Expr> boundaryGuard(Value value, PhaseNormalization& normalizer,
    const DenseMap<Value, Expr>& bindings, RegionExpressions& arena)
{
    DenseMap<Value, std::optional<Expr>> memo;
    std::function<std::optional<Expr>(Value)> evaluate = [&](Value current) -> std::optional<Expr> {
        if (auto found = bindings.find(current); found != bindings.end()) { return found->second; }
        if (auto found = memo.find(current); found != memo.end()) { return found->second; }
        auto compute = [&]() -> std::optional<Expr> {
            auto* op = current.getDefiningOp();
            if (op && isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
                auto a = evaluate(op->getOperand(0));
                if (a && arena.constantValue(*a) == 0 && isa<arith::AndIOp>(op)) { return *a; }
                if (a && arena.constantValue(*a) == 1 && isa<arith::OrIOp>(op)) { return *a; }
                auto b = evaluate(op->getOperand(1));
                if (b && arena.constantValue(*b) == 0 && isa<arith::AndIOp>(op)) { return *b; }
                if (b && arena.constantValue(*b) == 1 && isa<arith::OrIOp>(op)) { return *b; }
                if (!a || !b) { return std::nullopt; }
                if (isa<arith::AndIOp>(op)) { return arena.land(*a, *b); }
                if (isa<arith::OrIOp>(op)) { return arena.lor(*a, *b); }
                return arena.lnot(arena.eq(*a, *b));
            }
            return normalizer.independent(current) ? std::optional<Expr>(arena.input(current)) : std::nullopt;
        };
        auto result = compute();
        memo[current] = result;
        return result;
    };
    return evaluate(value);
}
std::optional<std::vector<BoundarySlice>> collectBoundarySlices(scf::ForOp loop,
    const PhaseIndex& index, RegionExpressions& arena, Expr trips,
    const DenseMap<Value, Expr>& inherited, std::string& error)
{
    if (sequenceInteger(loop.getLowerBound()) != 0 || sequenceInteger(loop.getStep()) != 1 ||
        loop.getNumRegionIterArgs()) {
        error = "boundary slicing requires a zero-based unit-step result-free loop";
        return std::nullopt;
    }
    PhaseNormalization normalizer(loop, index, arena);
    SmallVector<std::pair<Value, Split>> predicates;
    SmallVector<Expr> cuts;
    std::map<Expr, CutBounds> bounds;
    auto addCut = [&](Expr cut, CutBounds known) {
        auto& prior = bounds[cut];
        prior.fromZero = minimumBound(prior.fromZero, known.fromZero);
        prior.toEnd = minimumBound(prior.toEnd, known.toEnd);
        cuts.push_back(cut);
    };
    DenseSet<Value> seen;
    std::function<bool(Value)> collect = [&](Value value) {
        if (!seen.insert(value).second || inherited.count(value) || normalizer.independent(value)) { return true; }
        // A parent interval can already fix a compound predicate, e.g. false
        // AND an inner first-iteration test. Its unused atom does not change
        // the child's skeleton and must not introduce artificial child slices.
        if (auto fixed = boundaryGuard(value, normalizer, inherited, arena)) {
            if (arena.constantValue(*fixed)) { return true; }
        }
        auto* op = value.getDefiningOp();
        if (op && isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
            return collect(op->getOperand(0)) && collect(op->getOperand(1));
        }
        auto split = splitCondition(value, loop, index, arena, trips);
        if (!split) { return false; }
        predicates.emplace_back(value, *split);
        addCut(split->ordinal, split->ordinalBounds);
        if (split->equalityEnd) { addCut(*split->equalityEnd, split->endBounds); }
        return true;
    };
    bool supported = true;
    loop.getBody()->walk([&](scf::IfOp branch) { supported &= collect(branch.getCondition()); });
    if (!supported) { error = "outer control has no finite affine boundary partition"; return std::nullopt; }
    // Endpoints are already present in every partition. Clipped thresholds
    // equal to either endpoint contribute no additional interval.
    cuts.erase(std::remove_if(cuts.begin(), cuts.end(), [&](Expr cut) {
        return cut == arena.constant(0) || cut == trips;
    }), cuts.end());
    llvm::sort(cuts);
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    SmallVector<CutBounds> orderedBounds;
    for (auto cut : cuts) { orderedBounds.push_back(bounds[cut]); }
    for (std::size_t i = 1; i < cuts.size(); ++i) {
        for (std::size_t j = i; j; --j) {
            auto a = cuts[j - 1], b = cuts[j], ordered = arena.le(a, b);
            cuts[j - 1] = arena.select(ordered, a, b);
            cuts[j] = arena.select(ordered, b, a);
            auto left = orderedBounds[j - 1], right = orderedBounds[j];
            // min(a,b) inherits either upper bound from zero but needs both
            // lower bounds from T. max(a,b) has the dual certificates.
            orderedBounds[j - 1] = {minimumBound(left.fromZero, right.fromZero),
                                    maximumBound(left.toEnd, right.toEnd)};
            orderedBounds[j] = {maximumBound(left.fromZero, right.fromZero),
                                minimumBound(left.toEnd, right.toEnd)};
        }
    }
    cuts.insert(cuts.begin(), arena.constant(0));
    cuts.push_back(trips);
    orderedBounds.insert(orderedBounds.begin(), CutBounds{uint64_t{0}, std::nullopt});
    orderedBounds.push_back({std::nullopt, uint64_t{0}});
    std::vector<BoundarySlice> output;
    for (std::size_t i = 0; i + 1 < cuts.size(); ++i) {
        if (cuts[i] == cuts[i + 1]) { continue; }
        BoundarySlice slice{{cuts[i], cuts[i + 1]}, inherited};
        // Every cut lies in [0,T]. Hence end-begin <= end and <= T-begin.
        slice.maximumLength = minimumBound(orderedBounds[i + 1].fromZero, orderedBounds[i].toEnd);
        auto nonempty = arena.lt(slice.interval.begin, slice.interval.end);
        if (arena.constantValue(nonempty) == 0) { continue; }
        for (const auto& [value, split] : predicates) {
            auto guard = split.predicate(cuts[i], arena);
            // No payload or endpoint belongs to an empty slice. Constants
            // proved under its nonempty premise therefore preserve the exact
            // slice graph, and avoid analyzing mutually impossible arms.
            if (arena.implies(nonempty, guard)) { guard = arena.boolean(true); }
            else if (arena.implies(nonempty, arena.lnot(guard))) { guard = arena.boolean(false); }
            slice.bindings[value] = guard;
        }
        output.push_back(std::move(slice));
    }
    return output;
}
bool SequenceAnalysisState::boundaryLoop(scf::ForOp loop)
{
    if (sequenceInteger(loop.getLowerBound()) != 0 || sequenceInteger(loop.getStep()) != 1 ||
        loop.getNumRegionIterArgs()) { return false; }
    auto upper = expressions.input(loop.getUpperBound());
    auto trips = expressions.select(expressions.slt(c(0), upper), upper, c(0));
    SmallVector<std::pair<Value, Split>> predicates;
    SmallVector<Expr> cuts;
    bool supported = true;
    loop.getBody()->walk([&](scf::IfOp branch) {
        SmallVector<Operation*> recipe;
        if (detail::entryExpression(branch.getCondition(), loop, index, recipe)) { return; }
        auto split = splitCondition(branch.getCondition(), loop, index, expressions, trips);
        if (!split) { supported = false; return; }
        predicates.emplace_back(branch.getCondition(), *split);
        cuts.push_back(split->ordinal);
        if (split->equalityEnd) { cuts.push_back(*split->equalityEnd); }
    });
    if (!supported || cuts.empty()) { return false; }
    llvm::sort(cuts);
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    const bool singleCut = cuts.size() == 1 &&
        llvm::none_of(predicates, [](const auto& predicate) { return predicate.second.equalityEnd.has_value(); });
    // A fixed insertion network sorts cut VALUES, not parameter valuations.
    // There are O(h^2) select gates and h+1 slices for h distinct cut expressions.
    for (std::size_t i = 1; i < cuts.size(); ++i) {
        for (std::size_t j = i; j; --j) {
            auto left = cuts[j - 1], right = cuts[j];
            auto ordered = expressions.le(left, right);
            cuts[j - 1] = expressions.select(ordered, left, right);
            cuts[j] = expressions.select(ordered, right, left);
        }
    }
    cuts.insert(cuts.begin(), c(0));
    cuts.push_back(trips);
    std::vector<Child> slices;
    for (std::size_t part = 0; part + 1 < cuts.size(); ++part) {
        PeriodicSlice interval{cuts[part], cuts[part + 1]};
        if (interval.begin == interval.end) { continue; }
        DenseMap<Value, bool> choices;
        DenseMap<Value, Expr> bindings;
        SmallVector<Value> sliceGuards;
        for (const auto& [condition, split] : predicates) {
            auto guard = split.predicate(interval.begin, expressions);
            if (singleCut) { choices[condition] = part ? !split.prefixThen : split.prefixThen; }
            else if (auto known = expressions.constantValue(guard)) { choices[condition] = *known != 0; }
            else { bindings[condition] = guard; sliceGuards.push_back(condition); }
        }
        auto recognized = detail::recognizeRotatingSlice(loop, index, *input, choices, sliceGuards);
        if (recognized.result.state != RecognitionState::Applicable) {
            repeatedAttempt += "; boundary slice recognition";
            for (const auto& diagnostic : recognized.result.diagnostics) {
                repeatedAttempt += " / " + recognitionName(diagnostic.issue).str();
            }
            return false;
        }
        auto analyzed = analyzeGuardedRotating(loop, *input, recognized, arena, bindings);
        if (!analyzed.error.empty()) { repeatedAttempt += "; boundary slice: " + analyzed.error; return false; }
        std::string exportError;
        auto regional = guardedRotatingRegionalResult(function, *input, analyzed, exportError, interval);
        if (failed(regional)) { repeatedAttempt += "; boundary export: " + exportError; return false; }
        Child child;
        child.loop = loop;
        child.regional = std::move(*regional);
        child.anchors = child.regional.anchors;
        slices.push_back(std::move(child));
    }
    for (auto& slice : slices) { children.push_back(std::move(slice)); }
    return true;
}
} // namespace mlir::pto::frontiersynch
