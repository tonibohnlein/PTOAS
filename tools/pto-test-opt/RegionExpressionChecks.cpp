// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check expression algebra, total detached emission and original SSA availability.
#include "PTO/Transforms/FrontierSynch/LogicalInsertion.h"
#include "PTO/Transforms/FrontierSynch/RegionExpressions.h"
#include "../../lib/PTO/Transforms/FrontierSynch/SequenceAnalysisInternal.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
using namespace mlir;
namespace fs = mlir::pto::frontiersynch;
bool runNestedRegionalChecks(func::FuncOp function);
bool runRepeatedRegionChecks(func::FuncOp function);
bool runRepeatedStorageChecks(MLIRContext* context);
bool runRepeatedPhaseChecks(func::FuncOp function);
namespace {
std::string render(func::FuncOp function)
{
    std::string result;
    llvm::raw_string_ostream stream(result);
    function.print(stream);
    return result;
}
bool checkAlgebra(Value index, Value predicate)
{
    fs::RegionExpressions expressions;
    auto x = expressions.input(index), p = expressions.input(predicate);
    auto zero = expressions.constant(0), one = expressions.constant(1);
    auto maximum = expressions.constant(UINT64_MAX);
    auto sum = expressions.add(x, one);
    const auto before = expressions.size();
    bool valid = sum == expressions.add(one, x) && before == expressions.size();
    valid &= expressions.add(maximum, one) == zero && expressions.sub(zero, one) == maximum;
    valid &= expressions.div(maximum, one) == maximum && expressions.rem(maximum, one) == zero;
    valid &= expressions.lt(maximum, zero) == expressions.boolean(false);
    valid &= expressions.le(zero, maximum) == expressions.boolean(true);
    valid &= expressions.le(zero, x) == expressions.boolean(true);
    valid &= expressions.le(x, maximum) == expressions.boolean(true);
    valid &= expressions.lt(x, zero) == expressions.boolean(false);
    valid &= expressions.lt(maximum, x) == expressions.boolean(false);
    valid &= !expressions.constantValue(expressions.slt(x, zero));
    valid &= !expressions.constantValue(expressions.sle(zero, x));
    auto prior = expressions.sub(x, one);
    valid &= expressions.le(one, x) == expressions.lt(zero, x);
    valid &= expressions.lt(prior, x) == expressions.le(one, x);
    valid &= expressions.le(prior, x) == expressions.le(one, x);
    valid &= expressions.eq(prior, zero) == expressions.eq(x, one);
    valid &= expressions.eq(zero, prior) == expressions.eq(x, one);
    valid &= expressions.lt(expressions.sub(x, maximum), x) == expressions.le(maximum, x);
    valid &= expressions.slt(maximum, zero) == expressions.boolean(true);
    valid &= expressions.sle(zero, maximum) == expressions.boolean(false);
    valid &= expressions.land(p, expressions.boolean(true)) == p;
    valid &= expressions.lor(p, expressions.boolean(false)) == p;
    valid &= expressions.lnot(expressions.lnot(p)) == p;
    valid &= expressions.land(p, expressions.lnot(p)) == expressions.boolean(false);
    valid &= expressions.lor(p, expressions.lnot(p)) == expressions.boolean(true);
    valid &= expressions.select(p, sum, sum) == sum;
    valid &= expressions.select(p, expressions.boolean(true), expressions.boolean(false)) == p;
    valid &= expressions.select(p, expressions.boolean(false), expressions.boolean(true)) == expressions.lnot(p);
    auto q = expressions.lt(x, expressions.constant(5));
    auto r = expressions.lt(x, expressions.constant(6));
    auto both = expressions.land(p, q), either = expressions.lor(p, q);
    const auto beforeAbsorption = expressions.size();
    valid &= expressions.lor(p, both) == p && expressions.lor(both, p) == p;
    valid &= expressions.land(p, either) == p && expressions.land(either, p) == p;
    valid &= expressions.lor(q, both) == q && expressions.land(q, either) == q;
    valid &= expressions.land(p, both) == both && expressions.land(both, p) == both;
    valid &= expressions.lor(p, either) == either && expressions.lor(either, p) == either;
    valid &= expressions.size() == beforeAbsorption;
    valid &= expressions.implies(expressions.land(p, q), p);
    valid &= expressions.implies(expressions.land(p, expressions.lnot(q)), expressions.lnot(q));
    valid &= expressions.implies(expressions.land(expressions.land(p, q), r), expressions.land(q, p));
    valid &= !expressions.implies(p, q) && !expressions.implies(expressions.lor(p, q), p);
    valid &= expressions.implies(q, r); // Exact unsigned bounds on the same operand.
    valid &= expressions.land(q, r) == q && expressions.lor(q, r) == r;
    auto lower5 = expressions.lt(expressions.constant(5), x);
    auto lower6 = expressions.lt(expressions.constant(6), x);
    valid &= expressions.land(lower5, lower6) == lower6 && expressions.lor(lower5, lower6) == lower5;
    valid &= expressions.land(q, expressions.le(x, expressions.constant(5))) == q;
    valid &= expressions.div(expressions.minimum(x, expressions.constant(3)), expressions.constant(4)) == zero;
    valid &= expressions.div(expressions.minimum(expressions.constant(3), x), expressions.constant(4)) == zero;
    valid &= !expressions.constantValue(expressions.div(expressions.minimum(x, expressions.constant(4)),
                                                        expressions.constant(4)));
    valid &= expressions.implies(expressions.boolean(false), q);
    valid &= expressions.implies(p, expressions.boolean(true));
    auto masked = expressions.select(p, q, expressions.boolean(false));
    valid &= expressions.implies(masked, p) && expressions.implies(masked, q);
    auto maskedFalse = expressions.select(p, expressions.boolean(true), q);
    valid &= expressions.implies(expressions.lnot(maskedFalse), expressions.lnot(p));
    valid &= expressions.implies(expressions.lnot(maskedFalse), expressions.lnot(q));
    // A retained cover and an active intermediate event cannot coexist when
    // that intermediate supplies an alternative native-chain path.
    auto endpoints = expressions.land(p, q);
    auto alternative = expressions.land(endpoints, r);
    auto cover = expressions.land(endpoints, expressions.lnot(alternative));
    valid &= expressions.implies(cover, expressions.lnot(r));
    valid &= !expressions.implies(endpoints, expressions.lnot(r));
    valid &= !expressions.implies(expressions.lor(p, q), p);
    auto choice = expressions.select(p, expressions.constant(2), expressions.constant(5));
    valid &= expressions.constantUnder(p, choice) == 2;
    valid &= expressions.constantUnder(expressions.lnot(p), choice) == 5;
    valid &= !expressions.constantUnder(expressions.boolean(true), choice);
    auto nested = expressions.add(choice, expressions.select(q, expressions.constant(7), expressions.constant(9)));
    valid &= expressions.constantUnder(expressions.land(p, q), nested) == 9;
    valid &= expressions.constantUnder(expressions.land(expressions.lnot(p), expressions.lnot(q)), nested) == 14;
    valid &= !expressions.constantUnder(p, expressions.add(choice, x));
    // Unrelated prior queries must not be specialized or gain negation nodes.
    // The requested root is deliberately newer than this unrelated arena DAG.
    for (uint64_t i = 1000; i < 2024; ++i) {
        (void)expressions.lt(x, expressions.constant(i));
    }
    auto sparseChoice = expressions.select(p, expressions.constant(41), expressions.constant(73));
    auto sharedRoot = expressions.add(sparseChoice, sparseChoice);
    const auto beforeSparseProof = expressions.size();
    const auto sparseValue = expressions.constantUnder(p, sharedRoot);
    if (sparseValue != 82 || expressions.size() != beforeSparseProof) {
        llvm::errs() << "sparse constant proof: value=" << sparseValue.value_or(UINT64_MAX)
                     << " nodes=" << expressions.size() << " before=" << beforeSparseProof << "\n";
    }
    valid &= sparseValue == 82;
    valid &= expressions.size() == beforeSparseProof;
    // A premise may be newer than the root and need not occur in its DAG.
    auto newerPremise = expressions.land(p, expressions.lt(x, expressions.constant(9000)));
    valid &= expressions.constantUnder(newerPremise, sharedRoot) == 82;
    valid &= expressions.constantUnder(expressions.lnot(p), sharedRoot) == 146;
    valid &= !expressions.constantUnder(expressions.boolean(true), sharedRoot);
    fs::RegionExpressions bad;
    valid &= bad.div(bad.input(index), bad.constant(0)) == fs::RegionExpressions::invalid;
    fs::RegionExpressions dynamic;
    valid &= dynamic.rem(dynamic.constant(3), dynamic.input(index)) == fs::RegionExpressions::invalid;
    fs::RegionExpressions dynamicMinimum;
    auto dynamicIndex = dynamicMinimum.input(index);
    auto capped = dynamicMinimum.minimum(dynamicIndex, dynamicMinimum.constant(3));
    valid &= dynamicMinimum.div(capped, dynamicIndex) == fs::RegionExpressions::invalid;
    fs::RegionExpressions typed;
    valid &= typed.add(typed.input(index), typed.input(predicate)) == fs::RegionExpressions::invalid;
    return valid && expressions.error().empty();
}
bool checkDistanceAlgebra(Value index, Value predicate)
{
    fs::RegionExpressions e;
    const auto p = e.input(predicate), q = e.lt(e.input(index), e.constant(5));
    // Independent truth tables cover equal finite alternatives, reversed
    // alternatives, constant clamps and general (unfactored) choices.
    for (uint64_t a = 0; a <= 3; ++a) { for (uint64_t b = 0; b <= 3; ++b) {
        for (uint64_t c = 0; c <= 3; ++c) { for (uint64_t d = 0; d <= 3; ++d) {
            auto left = e.select(p, e.constant(a), e.constant(b));
            auto right = e.select(q, e.constant(c), e.constant(d));
            auto minimum = e.minimum(left, right);
            auto sum = e.boundedMinPlus(e.constant(3), left, right, 3);
            auto relaxation = e.boundedMinPlus(e.constant(1), left, right, 3);
            for (unsigned bits = 0; bits < 4; ++bits) {
                fs::RegionExpressions::Substitution valuation({
                    {p, e.boolean(bits & 1)}, {q, e.boolean(bits & 2)}});
                auto x = bits & 1 ? a : b, y = bits & 2 ? c : d;
                if (e.constantValue(e.substitute(minimum, valuation)) != std::min(x, y) ||
                    e.constantValue(e.substitute(sum, valuation)) != std::min(x + y, uint64_t(3)) ||
                    e.constantValue(e.substitute(relaxation, valuation)) != std::min(x + y, uint64_t(1))) {
                    return false;
                }
            }
        } }
    } }
    return e.error().empty();
}
bool checkThresholdAlgebra(Value index)
{
    fs::RegionExpressions e;
    const auto x = e.input(index);
    const uint64_t values[] = {0, 1, 2, 3, UINT64_MAX - 1, UINT64_MAX};
    struct Test { fs::RegionExpressions::Id id; uint64_t bound; unsigned kind; };
    SmallVector<Test> tests;
    for (uint64_t bound : values) {
        auto c = e.constant(bound);
        tests.push_back({e.lt(x, c), bound, 0}); tests.push_back({e.le(x, c), bound, 1});
        tests.push_back({e.lt(c, x), bound, 2}); tests.push_back({e.le(c, x), bound, 3});
    }
    auto truth = [](const Test& test, uint64_t value) {
        switch (test.kind) {
            case 0: return value < test.bound;
            case 1: return value <= test.bound;
            case 2: return test.bound < value;
            default: return test.bound <= value;
        }
    };
    for (const auto& a : tests) { for (const auto& b : tests) {
        auto both = e.land(a.id, b.id), either = e.lor(a.id, b.id);
        for (uint64_t value : values) {
            fs::RegionExpressions::Substitution valuation({{x, e.constant(value)}});
            if (e.constantValue(e.substitute(both, valuation)) != uint64_t(truth(a, value) && truth(b, value)) ||
                e.constantValue(e.substitute(either, valuation)) != uint64_t(truth(a, value) || truth(b, value))) {
                return false;
            }
        }
    } }
    return e.error().empty();
}
bool checkGuardedMinimum(Value index, Value predicate)
{
    fs::RegionExpressions e;
    const auto p = e.input(predicate), q = e.lt(e.input(index), e.constant(5));
    for (uint64_t cap : {uint64_t(0), uint64_t(1), uint64_t(3), UINT64_MAX}) {
        std::set<uint64_t> values{0, 1, cap};
        if (cap) { values.insert(cap - 1); }
        if (cap < UINT64_MAX) { values.insert(cap + 1); }
        for (auto a : values) { for (auto b : values) {
            const auto left = e.select(p, e.constant(a), e.constant(cap));
            const auto right = e.select(q, e.constant(b), e.constant(cap));
            const auto answer = e.minimum(left, right);
            // Check both guarded distances directly for all activity patterns,
            // including neither active, finite==infinity and out-of-bound arms
            // where the specialization must not apply.
            for (unsigned mask = 0; mask < 4; ++mask) {
                const bool activeLeft = mask & 1, activeRight = mask & 2;
                fs::RegionExpressions::Substitution bindings({{p, e.boolean(activeLeft)},
                                                              {q, e.boolean(activeRight)}});
                const auto expected = std::min(activeLeft ? a : cap, activeRight ? b : cap);
                if (e.constantValue(e.substitute(answer, bindings)) != expected) {
                    llvm::errs() << "guarded minimum mismatch: cap=" << cap << " a=" << a
                        << " b=" << b << " mask=" << mask << "\n";
                    return false;
                }
            }
            if ((a < b && b < cap && answer != e.select(p, e.constant(a), right)) ||
                (b < a && a < cap && answer != e.select(q, e.constant(b), left))) {
                llvm::errs() << "guarded minimum not factored: cap=" << cap << " a=" << a << " b=" << b << "\n";
                return false;
            }
        } }
    }
    return e.error().empty();
}
bool checkBoundedZeroEdges(Value index, Value predicate)
{
    fs::RegionExpressions e;
    const auto p = e.input(predicate), q = e.lt(e.input(index), e.constant(5));
    // All four q/r valuations are possible (index 0,1,6,7). Ordered
    // thresholds such as x<5 and x<7 are correlated and cannot model two
    // independent atoms in this truth-table oracle.
    const auto r = e.eq(e.rem(e.input(index), e.constant(2)), e.constant(0));
    for (uint64_t cap : {uint64_t(0), uint64_t(1), uint64_t(3), UINT64_MAX / 2}) {
        std::set<uint64_t> values{0, cap};
        if (cap) { values.insert(1); }
        for (auto current : values) { for (auto a : values) { for (auto b : values) {
            const auto old = e.select(r, e.constant(current), e.constant(cap));
            const auto left = e.select(p, e.constant(a), e.constant(cap));
            const auto right = e.select(q, e.constant(b), e.constant(cap));
            const auto answer = e.boundedMinPlus(old, left, right, cap);
            for (unsigned mask = 0; mask < 8; ++mask) {
                const bool pv = mask & 1, qv = mask & 2, rv = mask & 4;
                fs::RegionExpressions::Substitution bindings({{p, e.boolean(pv)},
                    {q, e.boolean(qv)}, {r, e.boolean(rv)}});
                const auto expected = std::min(rv ? current : cap, (pv ? a : cap) + (qv ? b : cap));
                if (e.constantValue(e.substitute(answer, bindings)) != expected) {
                    llvm::errs() << "bounded zero-edge mismatch: cap=" << cap << " current=" << current
                        << " a=" << a << " b=" << b << " mask=" << mask << "\n";
                    return false;
                }
            }
        } } }
    }
    return e.error().empty();
}
bool checkTransaction(Value index)
{
    fs::RegionExpressions expressions;
    auto x = expressions.input(index), one = expressions.constant(1);
    auto prefix = expressions.add(x, one);
    const auto before = expressions.size();
    {
        fs::RegionExpressions::Transaction outer(expressions);
        auto two = expressions.constant(2);
        auto suffix = expressions.add(x, two);
        {
            fs::RegionExpressions::Transaction nested(expressions);
            expressions.add(suffix, two);
            nested.commit();
        }
        if (expressions.div(x, expressions.constant(0)) != fs::RegionExpressions::invalid ||
            expressions.constructionError().empty()) { return false; }
    }
    if (expressions.size() != before || !expressions.error().empty() ||
        expressions.add(x, one) != prefix || expressions.size() != before) { return false; }
    // A removed intern-table entry must be rebuilt with a currently valid ID.
    auto two = expressions.constant(2);
    if (two != before || expressions.size() != before + 1) { return false; }
    auto suffix = expressions.add(x, two);
    {
        fs::RegionExpressions::Transaction committed(expressions);
        expressions.add(suffix, one);
        committed.commit();
    }
    const auto retained = expressions.size();
    expressions.div(x, expressions.constant(0));
    const auto originalError = expressions.constructionError();
    const auto rejectedSize = expressions.size();
    {
        fs::RegionExpressions::Transaction rejected(expressions);
        expressions.constant(123);
    }
    return retained > before + 1 && expressions.size() == rejectedSize &&
        expressions.constructionError() == originalError && !originalError.empty();
}
bool checkPortProvenance(func::FuncOp function)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    fs::SequenceAnalysisState state(function, arena);
    const auto zero = e.constant(0), one = e.constant(1);
    const auto p = e.input(function.getArgument(1));
    const auto q = e.lt(e.input(function.getArgument(0)), e.constant(5));
    unsigned rejectedCoordinates = 0;
    for (unsigned i = 0; i < 2; ++i) {
        fs::Child child;
        child.regional.expressions = arena;
        child.regional.presence = [arena, &rejectedCoordinates](fs::RegionalEvent event)
            -> std::optional<fs::Expr> {
            auto ordinal = arena->constantValue(event.ordinal);
            if (!ordinal) { ++rejectedCoordinates; return std::nullopt; }
            return arena->boolean(*ordinal < 2);
        };
        child.regional.reachability = [arena, present = child.regional.presence]
            (fs::RegionalEvent a, fs::RegionalEvent b) -> std::optional<fs::Expr> {
            auto pa = present(a), pb = present(b);
            if (!pa || !pb) { return std::nullopt; }
            auto before = arena->lt(a.ordinal, b.ordinal);
            auto same = arena->land(arena->eq(a.ordinal, b.ordinal),
                arena->boolean(a.kind == fs::PeriodicEventKind::Start ||
                               b.kind == fs::PeriodicEventKind::Completion));
            return arena->land(arena->land(*pa, *pb), arena->lor(before, same));
        };
        state.children.push_back(std::move(child));
    }
    auto a = state.port(0, 0, zero), b = state.port(0, 0, one);
    auto c = state.port(1, 0, zero), d = state.port(1, 0, one);
    state.crossings = {{a, c, p}, {b, c, q}};
    state.foldCrossingEndpoints(true);
    if (state.crossings.size() != 1 || state.portChoices.size() != 1) { return false; }
    auto source = state.crossings.front().source;
    state.crossings = {{a, c, p}, {a, d, q}};
    state.foldCrossingEndpoints(false);
    if (state.crossings.size() != 1 || state.portChoices.size() != 2) { return false; }
    auto target = state.crossings.front().target;
    // The backend rejects every synthetic coordinate. Both folds must instead
    // query the four original occurrences and preserve their selected presence.
    auto first = state.eventReachability(2 * source + 1, 2 * b);
    auto second = state.eventReachability(2 * c + 1, 2 * target);
    if (!first || !second || rejectedCoordinates || !state.error.empty()) { return false; }
    state.incoming[1].push_back({2 * a + 1, 2 * target, e.boolean(true)});
    auto across = state.eventReachability(2 * a + 1, 2 * d);
    if (!across || e.constantValue(*across) != 1 || rejectedCoordinates) { return false; }
    for (unsigned valuation = 0; valuation < 4; ++valuation) {
        bool pv = valuation & 1, qv = valuation & 2;
        fs::RegionExpressions::Substitution bindings({{p, e.boolean(pv)}, {q, e.boolean(qv)}});
        if (e.constantValue(e.substitute(*first, bindings)) != !qv ||
            e.constantValue(e.substitute(*second, bindings)) != (qv && !pv)) { return false; }
    }
    const auto size = e.size();
    if (state.eventReachability(2 * source + 1, 2 * b) != first || e.size() != size) { return false; }
    // Imported extremum lists contain alternatives for one actual event.
    // Normalize once, share the chosen identity across neighboring cells, and
    // keep the original-coordinate query contract even for conditional maps.
    std::vector<fs::Selected> alternatives{{a, p}, {b, e.lnot(p)}, {a, p}};
    auto originalAlternatives = alternatives;
    state.normalizeSelectorAlternatives(alternatives);
    if (alternatives.size() != 1 || e.constantValue(alternatives.front().present) != 1) { return false; }
    auto chosen = state.eventReachability(2 * alternatives.front().port + 1, 2 * b);
    if (!chosen || rejectedCoordinates) { return false; }
    for (bool enabled : {false, true}) {
        fs::RegionExpressions::Substitution bindings({{p, e.boolean(enabled)}});
        if (e.constantValue(e.substitute(*chosen, bindings)) != uint64_t(enabled)) { return false; }
    }
    const auto normalizedSize = e.size(), normalizedPorts = state.ports.size();
    state.normalizeSelectorAlternatives(originalAlternatives);
    return originalAlternatives.size() == 1 && originalAlternatives.front().port == alternatives.front().port &&
        e.size() == normalizedSize && state.ports.size() == normalizedPorts &&
        !rejectedCoordinates && e.error().empty();
}
bool checkRegionalNativeOrder(func::FuncOp function)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    const auto zero = e.constant(0), n = e.input(function.getArgument(0));
    const auto p = e.input(function.getArgument(1)), last = e.sub(n, e.constant(1));
    pto::CompoundInstanceElement phase(0, {}, {}, pto::PipelineType::PIPE_MTE1, function->getName());
    fs::RegionalAnalysis region;
    region.expressions = arena;
    region.capabilities.exactQueries = true;
    for (unsigned type = 0; type < 2; ++type) {
        region.anchors.push_back({&phase, {}, {}, {}});
        region.occurrenceLoops.push_back({});
    }
    region.presence = [arena, n, p](fs::RegionalEvent event) {
        return arena->land(arena->lt(event.ordinal, n), event.type ? arena->boolean(true) : p);
    };
    unsigned generalQueries = 0;
    region.reachability = [arena, &generalQueries](fs::RegionalEvent a, fs::RegionalEvent b)
        -> std::optional<fs::Expr> {
        ++generalQueries;
        // This body has no C->I requirements. All other same-pipe event pairs
        // must be answered by native order, without querying this backend.
        if (a.kind != fs::PeriodicEventKind::Completion || b.kind != fs::PeriodicEventKind::Start) {
            return std::nullopt;
        }
        return arena->boolean(false);
    };
    for (uint32_t sourceType : {0U, 1U}) { for (uint32_t targetType : {0U, 1U}) {
        for (bool sourceLast : {false, true}) { for (bool targetLast : {false, true}) {
            for (auto sourceKind : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
                for (auto targetKind : {fs::PeriodicEventKind::Start, fs::PeriodicEventKind::Completion}) {
                    fs::RegionalEvent a{sourceType, sourceLast ? last : zero, sourceKind};
                    fs::RegionalEvent b{targetType, targetLast ? last : zero, targetKind};
                    auto answer = fs::regionalReachability(region, a, b);
                    if (!answer) { return false; }
                    if (sourceType == targetType && !sourceLast && targetLast &&
                        !(sourceKind == fs::PeriodicEventKind::Completion &&
                          targetKind == fs::PeriodicEventKind::Start) &&
                        *answer != e.land(region.presence(a).value(), region.presence(b).value())) { return false; }
                    for (uint64_t trips = 0; trips < 5; ++trips) { for (bool active : {false, true}) {
                        const uint64_t sa = sourceLast ? trips - 1 : 0, tb = targetLast ? trips - 1 : 0;
                        const bool present = trips && (sourceType || active) && (targetType || active);
                        const bool order = sa < tb || (sa == tb && sourceType <= targetType);
                        const bool expected = present && order &&
                            !(sourceKind == fs::PeriodicEventKind::Completion &&
                          targetKind == fs::PeriodicEventKind::Start);
                        fs::RegionExpressions::Substitution bindings({{n, e.constant(trips)}, {p, e.boolean(active)}});
                        if (e.constantValue(e.substitute(*answer, bindings)) != expected) { return false; }
                    } }
                }
            }
        } }
    } }
    return generalQueries && e.error().empty();
}
bool checkCrossChildNative(func::FuncOp function)
{
    auto arena = std::make_shared<fs::RegionExpressions>();
    auto& e = *arena;
    fs::SequenceAnalysisState state(function, arena);
    const auto zero = e.constant(0), p = e.input(function.getArgument(1));
    const auto q = e.lt(e.input(function.getArgument(0)), e.constant(5));
    pto::CompoundInstanceElement phase(0, {}, {}, pto::PipelineType::PIPE_MTE1, function->getName());
    unsigned queried = 0;
    for (auto presence : {p, q}) {
        fs::Child child;
        child.regional.expressions = arena;
        child.anchors.push_back({&phase, {}, {}, {}});
        child.regional.anchors = child.anchors;
        child.regional.occurrenceLoops.push_back({});
        child.regional.presence = [presence](fs::RegionalEvent) { return presence; };
        child.regional.reachability = [arena, presence, &queried](fs::RegionalEvent a, fs::RegionalEvent b) {
            ++queried;
            return arena->land(presence, arena->boolean(a.kind == fs::PeriodicEventKind::Start ||
                                                       b.kind == fs::PeriodicEventKind::Completion));
        };
        state.children.push_back(std::move(child));
    }
    const auto a = state.port(0, 0, zero), b = state.port(1, 0, zero);
    SmallVector<fs::Expr> answers;
    for (auto [x, y] : {std::pair<unsigned, unsigned>{0, 0}, {0, 1}, {1, 1}}) {
        auto answer = state.eventReachability(2 * a + x, 2 * b + y);
        if (!answer) { return false; }
        answers.push_back(*answer);
    }
    if (queried) { return false; }
    auto absentCompletionRequirement = state.eventReachability(2 * a + 1, 2 * b);
    auto backward = state.eventReachability(2 * b, 2 * a + 1);
    if (!absentCompletionRequirement || e.constantValue(*absentCompletionRequirement) != 0 ||
        !backward || e.constantValue(*backward) != 0) { return false; }
    state.incoming[1].push_back({2 * a + 1, 2 * b, e.boolean(true)});
    state.reachabilityCache.clear();
    auto completionRequirement = state.eventReachability(2 * a + 1, 2 * b);
    if (!completionRequirement || !queried) { return false; }
    answers.push_back(*completionRequirement);
    for (unsigned valuation = 0; valuation < 4; ++valuation) {
        const bool pv = valuation & 1, qv = valuation & 2;
        fs::RegionExpressions::Substitution bindings({{p, e.boolean(pv)}, {q, e.boolean(qv)}});
        for (auto answer : answers) {
            if (e.constantValue(e.substitute(answer, bindings)) != (pv && qv)) { return false; }
        }
    }
    return e.error().empty();
}
bool checkIncompatibleCrossings(func::FuncOp function)
{
    for (bool exclusive : {false, true}) {
        auto arena = std::make_shared<fs::RegionExpressions>();
        auto& e = *arena;
        fs::SequenceAnalysisState state(function, arena);
        const auto zero = e.constant(0), p = e.input(function.getArgument(1));
        const auto q = e.lt(e.input(function.getArgument(0)), e.constant(5));
        const auto other = exclusive ? e.lnot(p) : q;
        std::vector<pto::CompoundInstanceElement> phases;
        for (auto pipe : {pto::PipelineType::PIPE_MTE1, pto::PipelineType::PIPE_MTE2,
                          pto::PipelineType::PIPE_M, pto::PipelineType::PIPE_V}) {
            phases.emplace_back(phases.size(), SmallVector<const pto::BaseMemInfo*>{},
                SmallVector<const pto::BaseMemInfo*>{}, pipe, function->getName());
        }
        unsigned queried = 0;
        for (unsigned childId = 0; childId < 2; ++childId) {
            fs::Child child;
            child.regional.expressions = arena;
            for (unsigned type = 0; type < 2; ++type) {
                child.anchors.push_back({&phases[2 * childId + type], {}, {}, {}});
                child.regional.anchors.push_back(child.anchors.back());
                child.regional.occurrenceLoops.push_back({});
            }
            child.regional.presence = [arena](fs::RegionalEvent) { return arena->boolean(true); };
            child.regional.reachability = [arena, &queried](fs::RegionalEvent a, fs::RegionalEvent b)
                -> std::optional<fs::Expr> {
                ++queried;
                return arena->boolean(a.type < b.type || (a.type == b.type &&
                    (a.kind == fs::PeriodicEventKind::Start || b.kind == fs::PeriodicEventKind::Completion)));
            };
            state.children.push_back(std::move(child));
        }
        state.nativeFirst.resize(2); state.nativeLast.resize(2);
        const auto a = state.port(0, 0, zero), b = state.port(0, 1, zero);
        const auto c = state.port(1, 0, zero), d = state.port(1, 1, zero);
        state.crossings = {{a, d, p}, {b, c, other}};
        if (!state.closure() || state.crossings.size() != 2 || (exclusive && queried)) { return false; }
        if (!exclusive && !queried) { return false; }
        // a->d has an alternative exactly when b->c is active. Disjoint
        // guards must avoid backend queries; overlapping guards must still
        // delete the redundant demand under their shared valuation.
        for (unsigned valuation = 0; valuation < 4; ++valuation) {
            const bool pv = valuation & 1, qv = valuation & 2;
            fs::RegionExpressions::Substitution bindings({{p, e.boolean(pv)}, {q, e.boolean(qv)}});
            const bool ov = exclusive ? !pv : qv;
            if (e.constantValue(e.substitute(state.crossings[0].guard, bindings)) != (pv && !ov) ||
                e.constantValue(e.substitute(state.crossings[1].guard, bindings)) != ov) { return false; }
        }
    }
    return true;
}
bool checkConsolidatedCandidateExclusion(func::FuncOp function)
{
    for (bool nativeMiddle : {false, true}) {
        auto arena = std::make_shared<fs::RegionExpressions>();
        auto& e = *arena;
        fs::SequenceAnalysisState state(function, arena);
        const auto zero = e.constant(0), x = e.input(function.getArgument(0));
        SmallVector<fs::Expr> guards{e.input(function.getArgument(1))};
        for (uint64_t divisor : {uint64_t(1), uint64_t(2), uint64_t(4)}) {
            guards.push_back(e.eq(e.rem(e.div(x, e.constant(divisor)), e.constant(2)), e.constant(1)));
        }
        pto::CompoundInstanceElement source(0, {}, {}, pto::PipelineType::PIPE_MTE1, function->getName());
        pto::CompoundInstanceElement target(1, {}, {}, pto::PipelineType::PIPE_V, function->getName());
        for (auto* phase : {&source, &target}) {
            fs::Child child;
            child.regional.expressions = arena;
            for (unsigned type = 0; type < 2; ++type) {
                child.anchors.push_back({phase, {}, {}, {}});
                child.regional.anchors.push_back(child.anchors.back());
                child.regional.occurrenceLoops.push_back({});
            }
            child.regional.presence = [arena](fs::RegionalEvent) { return arena->boolean(true); };
            child.regional.reachability = [arena](fs::RegionalEvent a, fs::RegionalEvent b) {
                const bool same = a.type == b.type && (a.kind == fs::PeriodicEventKind::Start ||
                                                     b.kind == fs::PeriodicEventKind::Completion);
                const bool forward = a.type < b.type && !(a.kind == fs::PeriodicEventKind::Completion &&
                                                          b.kind == fs::PeriodicEventKind::Start);
                return arena->boolean(same || forward);
            };
            state.children.push_back(std::move(child));
        }
        state.nativeFirst.resize(2); state.nativeLast.resize(2);
        const auto a = state.port(0, 0, zero), b = state.port(0, 1, zero);
        const auto c = state.port(1, 0, zero), d = state.port(1, 1, zero);
        const std::array<std::pair<uint32_t, uint32_t>, 4> endpoints{{{a,c}, {b,c}, {a,d}, {b,d}}};
        for (unsigned i = 0; i < endpoints.size(); ++i) {
            auto [s, t] = endpoints[i];
            if (nativeMiddle && i == 1) { state.nativeValueCrossings.push_back({s, t, guards[i]}); }
            else { state.crossings.push_back({s, t, guards[i]}); }
        }
        if (!state.closure()) { return false; }
        // Independent occurrence graph: two native chains and four guarded
        // crossings. Removing each candidate and recomputing closure decides
        // whether it is indispensable, without using consolidation rules.
        for (unsigned mask = 0; mask < 16; ++mask) {
            SmallVector<std::pair<fs::Expr, fs::Expr>> bindings;
            for (unsigned i = 0; i < guards.size(); ++i) {
                bindings.push_back({guards[i], e.boolean(mask & (1U << i))});
            }
            fs::RegionExpressions::Substitution valuation(bindings);
            for (const auto& edge : state.crossings) {
                std::array<std::array<bool, 8>, 8> graph{};
                for (unsigned i = 0; i < 4; ++i) { graph[2*i][2*i+1] = true; }
                for (unsigned start : {0U, 4U}) {
                    graph[start][start+2] = graph[start+1][start+3] = true;
                }
                bool active = false;
                for (unsigned i = 0; i < endpoints.size(); ++i) {
                    const auto [s, t] = endpoints[i];
                    if (s == edge.source && t == edge.target) { active = mask & (1U << i); }
                    else if (mask & (1U << i)) { graph[2*s+1][2*t] = true; }
                }
                for (unsigned k = 0; k < 8; ++k) { for (unsigned i = 0; i < 8; ++i) {
                    for (unsigned j = 0; j < 8; ++j) { graph[i][j] |= graph[i][k] && graph[k][j]; }
                } }
                const bool expected = active && !graph[2*edge.source+1][2*edge.target];
                if (e.constantValue(e.substitute(edge.guard, valuation)) != expected) {
                    llvm::errs() << "consolidated crossing mismatch: native=" << nativeMiddle
                        << " mask=" << mask << " source=" << edge.source << " target=" << edge.target << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}
bool checkSubstitution(Value index, Value predicate)
{
    fs::RegionExpressions a;
    auto x = a.input(index), p = a.input(predicate), one = a.constant(1);
    auto shifted = a.add(x, one);
    fs::RegionExpressions::Substitution shift({{x, shifted}});
    auto twice = a.substitute(shifted, shift);
    if (twice != a.add(shifted, one)) { return false; }
    auto size = a.size();
    if (a.substitute(shifted, shift) != twice || a.size() != size) { return false; }
    fs::RegionExpressions::Substitution bind({{x, a.constant(7)}, {p, a.boolean(false)}});
    auto choice = a.select(p, x, a.add(x, a.constant(3)));
    if (a.constantValue(a.substitute(choice, bind)) != 10) { return false; }
    // Swapped nodes are simultaneous, not recursively rewritten.
    fs::RegionExpressions::Substitution swap({{x, shifted}, {shifted, x}});
    if (a.substitute(x, swap) != shifted || a.substitute(shifted, swap) != x) { return false; }
    fs::RegionExpressions wrong;
    auto y = wrong.input(index);
    if (wrong.substitute(y, bind) != fs::RegionExpressions::invalid) { return false; }
    fs::RegionExpressions typed;
    fs::RegionExpressions::Substitution invalid({{typed.input(index), typed.input(predicate)}});
    if (typed.substitute(typed.input(index), invalid) != fs::RegionExpressions::invalid) { return false; }
    fs::RegionExpressions duplicate;
    auto v = duplicate.input(index);
    fs::RegionExpressions::Substitution conflict({{v, duplicate.constant(1)}, {v, duplicate.constant(2)}});
    return duplicate.substitute(v, conflict) == fs::RegionExpressions::invalid && a.error().empty();
}
bool checkImport(Value index, Value predicate)
{
    using Arena = fs::RegionExpressions;
    Arena source, target;
    const auto x = source.input(index), p = source.input(predicate), three = source.constant(3);
    const auto sum = source.add(x, three), less = source.lt(x, three);
    SmallVector<Arena::Id> roots{x, p, three, source.boolean(true), sum, source.sub(x, three),
        source.div(x, three), source.rem(x, three), less, source.le(x, three), source.eq(x, three),
        source.slt(x, three), source.sle(x, three), source.land(p, less), source.lor(p, less),
        source.lnot(p), source.select(p, x, sum)};
    auto system = fs::IntegerSystem::create(1, {{{fs::BoundInteger(1)}, fs::BoundInteger(2)}});
    if (failed(system)) { return false; }
    const fs::IntegerAffine numerator{{fs::BoundInteger(3)}, fs::BoundInteger(-2)};
    roots.push_back(source.integerPredicate(*system, {x}, 2, {1}));
    roots.push_back(source.integerFloor(numerator, fs::BoundInteger(2), {x}, 1, {0}));
    roots.push_back(source.integerWitness(numerator, fs::BoundInteger(2), {x}, 2, {1}, 1));
    std::string diagnostic;
    auto mapped = source.integerMappedPredicate(*system, {x},
        {getAffineDimExpr(0, index.getContext()) + 1}, 1, {0}, diagnostic);
    if (!mapped) { return false; }
    roots.push_back(*mapped);
    target.constant(999); // Numeric IDs deliberately disagree with the source.
    const auto before = target.size();
    {
        Arena::Transaction transaction(target);
        uint64_t work = UINT64_MAX - 1;
        auto imported = target.import(source, roots, &work);
        const bool complete = succeeded(imported) && imported->size() == roots.size() && work == UINT64_MAX;
        if (!complete) { return false; }
        const auto count = target.size();
        auto retry = target.import(source, roots);
        const bool reused = succeeded(retry) && *retry == *imported && target.size() == count;
        if (!reused) { return false; }
        for (int64_t value : {-9, -1, 0, 2, 3, 8}) {
            for (bool flag : {false, true}) {
                Arena::Substitution a({{x, source.constant(static_cast<uint64_t>(value))}, {p, source.boolean(flag)}});
                Arena::Substitution b({{(*imported)[0], target.constant(static_cast<uint64_t>(value))},
                                       {(*imported)[1], target.boolean(flag)}});
                for (std::size_t i = 0; i < roots.size(); ++i) {
                    const auto expected = source.constantValue(source.substitute(roots[i], a));
                    const auto actual = target.constantValue(target.substitute((*imported)[i], b));
                    const bool sameType = target.isBoolean((*imported)[i]) == source.isBoolean(roots[i]);
                    if (!expected || actual != expected || !sameType) {
                        return false;
                    }
                }
            }
        }
    }
    const bool rolledBack = target.size() == before;
    if (!rolledBack) { return false; }
    target.constant(12345); // Reuse rolled-back target IDs with different meanings.
    auto renewed = target.import(source, roots);
    const bool renewedInput = succeeded(renewed) && (*renewed)[0] == target.input(index);
    if (!renewedInput) { return false; }
    Arena other;
    const auto otherRoot = other.constant(77);
    if (otherRoot != x) { return false; }
    auto different = target.import(other, {otherRoot});
    const bool distinct = succeeded(different) && target.constantValue(different->front()) == 77;
    if (!distinct) { return false; }
    const auto stable = target.size();
    auto invalid = target.import(source, {Arena::invalid});
    other.add(other.boolean(true), other.constant(1));
    auto poisoned = target.import(other, {otherRoot});
    const bool refused = failed(invalid) && failed(poisoned) && target.size() == stable && target.error().empty();
    if (!refused) { return false; }
    auto same = target.import(target, *renewed);
    return succeeded(same) && *same == *renewed;
}
bool checkImportOwners(Value index)
{
    using Arena = fs::RegionExpressions;
    std::weak_ptr<Block> weak;
    {
        Arena target;
        Arena::Id imported;
        {
            Arena source;
            auto owner = std::make_shared<Block>(); weak = owner;
            auto input = owner->addArgument(index.getType(), UnknownLoc::get(index.getContext()));
            source.retainInputOwner(owner);
            auto result = target.import(source, {source.add(source.input(input), source.constant(3))});
            if (failed(result)) { return false; }
            imported = result->front();
            Arena mirror;
            for (unsigned attempt = 0; attempt < 12; ++attempt) {
                Arena::Transaction rollback(target);
                auto copied = mirror.import(target, {imported});
                if (failed(copied)) { return false; }
                auto returned = target.import(mirror, *copied);
                const bool boundedOwners = succeeded(returned) && owner.use_count() == 4;
                if (!boundedOwners) { return false; }
            }
            // Aliased pointers with one control block still retain one owner.
            source.retainInputOwner(std::shared_ptr<void>(owner, input.getAsOpaquePointer()));
            auto retried = target.import(source, {source.input(input)});
            const bool oneImportedOwner = succeeded(retried) && owner.use_count() == 4;
            if (!oneImportedOwner) { return false; }
        }
        if (weak.expired()) { return false; }
        auto inputs = target.referencedInputs(imported);
        const bool oneInput = inputs.size() == 1;
        if (!oneInput) { return false; }
        Arena::Substitution binding({{inputs[0].first, target.constant(7)}});
        const bool preserved = target.constantValue(target.substitute(imported, binding)) == 10;
        if (!preserved) { return false; }
    }
    return weak.expired();
}
bool checkImportReplay(Operation* cut)
{
    using Arena = fs::RegionExpressions;
    Arena target, unrestricted;
    Arena::Id imported, control;
    {
        Arena source;
        const auto root = source.input(cut->getResult(0));
        auto allowed = unrestricted.import(source, {root});
        source.forbidRecomputation(cut);
        auto forbidden = target.import(source, {root});
        const bool importedBoth = succeeded(allowed) && succeeded(forbidden);
        if (!importedBoth) { return false; }
        imported = forbidden->front(); control = allowed->front();
    }
    fs::PreparedLogicalPlan plan(0);
    auto& blockedCode = plan.addPreparation(cut);
    OpBuilder builder(cut->getContext()); builder.setInsertionPointToEnd(&blockedCode);
    Arena::CutEmission blockedContext;
    auto blocked = target.emitContextual(imported, builder, cut, blockedContext);
    const bool rejectedWithoutCode = failed(blocked) && blockedCode.empty();
    if (!rejectedWithoutCode) { return false; }
    auto& allowedCode = plan.addPreparation(cut);
    builder.setInsertionPointToEnd(&allowedCode);
    Arena::CutEmission allowedContext;
    auto allowed = unrestricted.emitContextual(control, builder, cut, allowedContext);
    return succeeded(allowed) && !allowedCode.empty();
}
bool checkIntegerAdapters(Value index, Value predicate)
{
    fs::RegionExpressions expressions;
    const auto x = expressions.input(index), p = expressions.input(predicate);
    auto system = fs::IntegerSystem::create(1, {{{fs::BoundInteger(1)}, fs::BoundInteger(2)}},
        {{{fs::BoundInteger(1)}, fs::BoundInteger(1), fs::BoundInteger(3)}});
    if (failed(system)) { return false; }
    fs::IntegerAffine numerator{{fs::BoundInteger(3)}, fs::BoundInteger(-2)};
    for (int64_t period = 1; period <= 3; ++period) {
        for (int64_t residue = 0; residue < period; ++residue) {
            const auto condition = expressions.integerPredicate(*system, {x}, period, {static_cast<uint64_t>(residue)});
            const auto witness = expressions.integerWitness(numerator, fs::BoundInteger(2), {x}, period,
                {static_cast<uint64_t>(residue)}, 0);
            const auto size = expressions.size();
            if (condition != expressions.integerPredicate(*system, {x}, period, {static_cast<uint64_t>(residue)}) ||
                expressions.size() != size) { return false; }
            for (int64_t value = -9; value <= 9; ++value) {
                // C++ division truncates, so implement floor independently.
                const int64_t quotient = value / period - (value % period < 0);
                const int64_t remainder = value - quotient * period;
                const bool expected = remainder == residue && quotient <= 2 && (quotient % 3 + 3) % 3 == 1;
                const int64_t n = 3 * quotient - 2;
                const int64_t output = (n / 2 - (n % 2 < 0)) * period;
                fs::RegionExpressions::Substitution bind({{x, expressions.constant(static_cast<uint64_t>(value))}});
                if (expressions.constantValue(expressions.substitute(condition, bind)) !=
                        static_cast<uint64_t>(expected) ||
                    expressions.constantValue(expressions.substitute(witness, bind)) != static_cast<uint64_t>(output)) {
                    return false;
                }
            }
        }
    }
    auto trueSystem = fs::IntegerSystem::create(1, {});
    if (failed(trueSystem)) { return false; }
    auto boolAsInteger = expressions.integerWitness({{fs::BoundInteger(1)}, fs::BoundInteger(0)},
        fs::BoundInteger(1), {p}, 1, {0}, 0);
    fs::RegionExpressions::Substitution trueBinding({{p, expressions.boolean(true)}});
    if (expressions.constantValue(expressions.substitute(boolAsInteger, trueBinding)) != 1) { return false; }
    // Coefficients may fit i128 while their products with signed64 inputs do not.
    fs::BoundInteger large(1);
    for (unsigned i = 0; i < 70; ++i) { large *= 2; }
    fs::RegionExpressions rejected;
    if (rejected.integerWitness({{large}, fs::BoundInteger(0)}, fs::BoundInteger(1),
        {rejected.input(index)}, 1, {0}, 0) != fs::RegionExpressions::invalid) { return false; }
    fs::RegionExpressions badDivisor;
    return badDivisor.integerWitness({{fs::BoundInteger(1)}, fs::BoundInteger(0)}, fs::BoundInteger(0),
        {badDivisor.input(index)}, 1, {0}, 0) == fs::RegionExpressions::invalid && expressions.error().empty();
}
bool checkPartialIntegerAdapters(Value index, Value predicate)
{
    fs::RegionExpressions expressions;
    const auto x = expressions.input(index), unused = expressions.input(predicate);
    auto pair = fs::IntegerSystem::create(2, {{{fs::BoundInteger(1), fs::BoundInteger(4)}, fs::BoundInteger(11)}});
    auto single = fs::IntegerSystem::create(1, {{{fs::BoundInteger(1)}, fs::BoundInteger(2)}});
    if (failed(pair) || failed(single)) { return false; }
    const auto expected = expressions.integerPredicate(*single, {x}, 1, {0});
    for (uint64_t byte = 0; byte < 4; ++byte) {
        if (expressions.integerPredicate(*pair, {expressions.constant(byte), x}, 1, {0,0}) != expected ||
            expressions.integerWitness({{fs::BoundInteger(1), fs::BoundInteger(4)}, fs::BoundInteger(0)},
                fs::BoundInteger(4), {expressions.constant(byte), x}, 1, {0,0}, 0) != x) { return false; }
    }
    auto sparse = fs::IntegerSystem::create(2, {{{fs::BoundInteger(0), fs::BoundInteger(1)}, fs::BoundInteger(2)}});
    if (failed(sparse) || expressions.integerPredicate(*sparse, {unused, x}, 1, {0,0}) != expected) { return false; }
    auto congruence = fs::IntegerSystem::create(2, {},
        {{{fs::BoundInteger(2), fs::BoundInteger(3)}, fs::BoundInteger(1), fs::BoundInteger(5)}});
    auto reduced = fs::IntegerSystem::create(1, {},
        {{{fs::BoundInteger(3)}, fs::BoundInteger(2), fs::BoundInteger(5)}});
    if (failed(congruence) || failed(reduced)) { return false; }
    const auto negative = expressions.constant(static_cast<uint64_t>(-5));
    if (expressions.integerPredicate(*congruence, {negative,x}, 2, {1,1}) !=
        expressions.integerPredicate(*reduced, {x}, 2, {1})) { return false; }
    if (expressions.integerPredicate(*congruence, {negative,x}, 2, {0,1}) != expressions.boolean(false)) {
        return false;
    }
    auto residueOnly = fs::IntegerSystem::create(1, {});
    if (failed(residueOnly) || expressions.constantValue(expressions.integerPredicate(*residueOnly, {x}, 2, {1}))) {
        return false;
    }
    auto interval = fs::IntegerSystem::create(1, {
        {{fs::BoundInteger(1)}, fs::BoundInteger(2)},
        {{fs::BoundInteger(-1)}, fs::BoundInteger(3)}});
    if (failed(interval) || expressions.integerPredicate(*interval, {x}, 1, {0}) !=
        expressions.land(expressions.sle(x, expressions.constant(2)),
                         expressions.sle(expressions.constant(static_cast<uint64_t>(-3)), x))) { return false; }
    // Mathematical bounds beyond signed64 fold without wrapping the bound.
    for (int sign : {-1, 1}) {
        auto all = fs::IntegerSystem::create(1, {{{fs::BoundInteger(sign)},
            fs::BoundInteger(INT64_MAX) + fs::BoundInteger(1)}});
        auto none = fs::IntegerSystem::create(1, {{{fs::BoundInteger(sign)},
            fs::BoundInteger(INT64_MIN) - fs::BoundInteger(1)}});
        if (failed(all) || failed(none) ||
            expressions.integerPredicate(*all, {x}, 1, {0}) != expressions.boolean(true) ||
            expressions.integerPredicate(*none, {x}, 1, {0}) != expressions.boolean(false)) { return false; }
    }
    auto sameCoordinate = fs::IntegerSystem::create(2,
        {{{fs::BoundInteger(1), fs::BoundInteger(-1)}, fs::BoundInteger(0)}});
    auto strictCoordinate = fs::IntegerSystem::create(2,
        {{{fs::BoundInteger(1), fs::BoundInteger(-1)}, fs::BoundInteger(-1)}});
    if (failed(sameCoordinate) || failed(strictCoordinate) ||
        expressions.integerPredicate(*sameCoordinate, {x,x}, 1, {0,0}) != expressions.boolean(true) ||
        expressions.integerPredicate(*strictCoordinate, {x,x}, 1, {0,0}) != expressions.boolean(false) ||
        expressions.integerPredicate(*sameCoordinate, {x,x}, 2, {0,1}) != expressions.boolean(false)) { return false; }
    // Partial substitution through an existing DAG must run the same adapter.
    const auto dynamic = expressions.integerPredicate(*pair, {x,unused}, 1, {0,0});
    fs::RegionExpressions::Substitution binding({{x, expressions.constant(3)}});
    return expressions.substitute(dynamic, binding) == expressions.integerPredicate(*single, {unused}, 1, {0}) &&
        expressions.error().empty();
}
bool evaluateIntegerCode(Block& code, Value input, int64_t coordinate, Value guard, Value witness,
                         std::optional<bool> mappedExpected = {})
{
    llvm::DenseMap<Value, APInt> values;
    values[input] = APInt(64, static_cast<uint64_t>(coordinate));
    for (auto& operation : code) {
        SmallVector<APInt> operands;
        for (Value value : operation.getOperands()) {
            auto found = values.find(value);
            if (found == values.end()) { return false; }
            operands.push_back(found->second);
        }
        APInt result;
        if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
            result = cast<IntegerAttr>(constant.getValue()).getValue();
        } else if (isa<arith::IndexCastOp>(operation)) {
            Type type = operation.getResult(0).getType();
            result = operands[0].sextOrTrunc(type.isIndex() ? 64 : cast<IntegerType>(type).getWidth());
        } else if (auto extend = dyn_cast<arith::ExtUIOp>(operation)) {
            result = operands[0].zext(cast<IntegerType>(extend.getType()).getWidth());
        } else if (isa<arith::FloorDivSIOp>(operation)) {
            result = operands[0].sdiv(operands[1]);
            if (operands[0].isNegative() && !operands[0].srem(operands[1]).isZero()) { --result; }
        } else if (isa<arith::CeilDivSIOp>(operation)) {
            result = operands[0].sdiv(operands[1]);
            const bool roundUp = !operands[0].isNegative() && !operands[0].srem(operands[1]).isZero();
            if (roundUp) { ++result; }
        } else if (isa<arith::MulIOp>(operation)) {
            result = operands[0] * operands[1];
        } else if (isa<arith::AddIOp>(operation)) {
            result = operands[0] + operands[1];
        } else if (isa<arith::SubIOp>(operation)) {
            result = operands[0] - operands[1];
        } else if (isa<arith::AndIOp>(operation)) {
            result = operands[0] & operands[1];
        } else if (auto comparison = dyn_cast<arith::CmpIOp>(operation)) {
            bool truth;
            if (comparison.getPredicate() == arith::CmpIPredicate::eq) { truth = operands[0] == operands[1]; }
            else if (comparison.getPredicate() == arith::CmpIPredicate::sle) { truth = operands[0].sle(operands[1]); }
            else { return false; }
            result = APInt(1, truth);
        } else { return false; }
        values[operation.getResult(0)] = result;
    }
    if (mappedExpected) {
        return values.lookup(guard).getZExtValue() == *mappedExpected;
    }
    const int64_t q = coordinate / 2 - (coordinate % 2 < 0);
    const bool expectedGuard = coordinate - q * 2 == 1 && 2 * q <= 7;
    const int64_t numerator = 3 * q - 2;
    const int64_t expectedWitness = (numerator / 2 - (numerator % 2 < 0)) * 2 + 1;
    return values.lookup(guard).getZExtValue() == expectedGuard &&
        values.lookup(witness).getSExtValue() == expectedWitness;
}
bool checkMappedIntegerAdapters(func::FuncOp function, ArrayRef<Operation*> cuts)
{
    fs::RegionExpressions arena;
    const auto x = arena.input(function.getArgument(0));
    auto* context = function.getContext();
    const auto a = getAffineDimExpr(0, context), b = getAffineDimExpr(1, context);
    SmallVector<AffineExpr> maps{a - 1, a + 1, a * 2 - b,
        a - (b.floorDiv(3) * 5 + 7), a - b % 5, a.ceilDiv(3)};
    auto relation = fs::IntegerSystem::create(1, {{{fs::BoundInteger(1)}, fs::BoundInteger(3)},
        {{fs::BoundInteger(-1)}, fs::BoundInteger(5)}});
    auto domain = fs::IntegerSystem::create(1, {});
    const bool systemsReady = succeeded(relation) && succeeded(domain);
    if (!systemsReady) { return false; }
    std::string diagnostic;
    fs::RegionExpressions::Id previous = fs::RegionExpressions::invalid;
    for (auto [caseID, map] : llvm::enumerate(maps)) {
        auto query = arena.integerMappedPredicate(*relation, {x, x}, {map}, 2, {1}, diagnostic);
        const bool queryReady = query && diagnostic.empty() && *query != previous;
        if (!queryReady) { return false; }
        previous = *query;
        fs::PreparedLogicalPlan plan(0);
        auto& code = plan.addPreparation(cuts[0]);
        OpBuilder builder(context); builder.setInsertionPointToEnd(&code);
        fs::RegionExpressions::CutEmission emission;
        auto emitted = arena.emitContextual(*query, builder, cuts[0], emission);
        if (failed(emitted)) { return false; }
        for (auto& operation : code) { if (failed(verify(&operation))) { return false; } }
        for (int64_t sample : {INT64_MIN, INT64_MAX, int64_t(-11), int64_t(-9), int64_t(-3), int64_t(-1),
                               int64_t(0), int64_t(1), int64_t(2), int64_t(9), int64_t(11)}) {
            const fs::BoundInteger value(sample), divisor(3);
            fs::BoundInteger mapped(0);
            switch (caseID) {
            case 0: mapped = value - fs::BoundInteger(1); break;
            case 1: mapped = value + fs::BoundInteger(1); break;
            case 2: mapped = value; break;
            case 3: mapped = value - (floorDiv(value, divisor) * fs::BoundInteger(5) + fs::BoundInteger(7)); break;
            case 4: mapped = floorDiv(value, fs::BoundInteger(5)) * fs::BoundInteger(5); break;
            case 5: mapped = -floorDiv(-value, divisor); break;
            default: return false;
            }
            const auto quotient = floorDiv(mapped, fs::BoundInteger(2));
            const bool expected = mapped - quotient * fs::BoundInteger(2) == 1 && quotient >= -5 && quotient <= 3;
            const auto fixed = arena.constant(static_cast<uint64_t>(sample));
            fs::RegionExpressions::Substitution substitution({{x, fixed}});
            const auto folded = arena.substitute(*query, substitution);
            auto direct = arena.integerMappedPredicate(*relation, {fixed, fixed}, {map}, 2, {1}, diagnostic);
            const bool agrees = direct && arena.constantValue(folded) == expected &&
                arena.constantValue(*direct) == expected &&
                evaluateIntegerCode(code, function.getArgument(0), sample, *emitted, *emitted, expected);
            if (!agrees) {
                return false;
            }
        }
        const auto shifted = arena.add(x, arena.constant(1));
        fs::RegionExpressions::Substitution rebind({{x, shifted}}), fix({{x, arena.constant(5)}});
        const auto nested = arena.substitute(arena.substitute(*query, rebind), fix);
        auto directSix = arena.integerMappedPredicate(*relation, {arena.constant(6), arena.constant(6)},
            {map}, 2, {1}, diagnostic);
        const bool rebound = directSix && arena.constantValue(nested) == arena.constantValue(*directSix);
        if (!rebound) { return false; }
        for (auto predicate : {*query, arena.lnot(*query)}) {
            diagnostic.clear();
            auto lowered = arena.integerRelation(predicate, {x}, *domain, diagnostic);
            const bool refusedLowering = failed(lowered) && StringRef(diagnostic).contains("mapped");
            if (!refusedLowering) { return false; }
        }
    }
    auto scalar = fs::IntegerSystem::create(0, {});
    if (failed(scalar)) { return false; }
    auto scalarQuery = arena.integerMappedPredicate(*scalar, {}, {}, 1, {}, diagnostic);
    const bool scalarReady = scalarQuery && arena.constantValue(*scalarQuery) == 1;
    if (!scalarReady) { return false; }
    const auto before = arena.size();
    const auto huge = a * INT64_MAX + b * INT64_MAX;
    auto refused = arena.integerMappedPredicate(*relation, {x, x}, {huge}, 1, {0}, diagnostic);
    auto amplified = fs::IntegerSystem::create(2,
        {{{fs::BoundInteger(2), fs::BoundInteger(1)}, fs::BoundInteger(0)}});
    if (failed(amplified)) { return false; }
    auto coefficientOverflow = arena.integerMappedPredicate(*amplified, {x, x},
        {a * INT64_MAX, b}, 1, {0, 0}, diagnostic);
    auto invalidDimension = arena.integerMappedPredicate(*relation, {x}, {b}, 1, {0}, diagnostic);
    auto invalidSymbol = arena.integerMappedPredicate(*relation, {x}, {getAffineSymbolExpr(0, context)},
        1, {0}, diagnostic);
    return !refused && !coefficientOverflow && !invalidDimension && !invalidSymbol &&
        arena.size() == before && arena.constructionError().empty();
}
bool checkIntegerEmission(func::FuncOp function, ArrayRef<Operation*> cuts, Value hidden)
{
    fs::RegionExpressions expressions;
    auto x = expressions.input(function.getArgument(0));
    auto relation = fs::IntegerSystem::create(1, {{{fs::BoundInteger(2)}, fs::BoundInteger(7)}});
    if (failed(relation)) { return false; }
    auto query = expressions.integerPredicate(*relation, {x}, 2, {1});
    auto witness = expressions.integerWitness({{fs::BoundInteger(3)}, fs::BoundInteger(-2)},
        fs::BoundInteger(2), {x}, 2, {1}, 1);
    fs::PreparedLogicalPlan plan(0);
    auto& code = plan.addPreparation(cuts[0]);
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToEnd(&code);
    fs::RegionExpressions::CutEmission context;
    auto emittedQuery = expressions.emitContextual(query, builder, cuts[0], context);
    auto emittedWitness = expressions.emitContextual(witness, builder, cuts[0], context);
    if (failed(emittedQuery) || failed(emittedWitness)) { return false; }
    for (int64_t value = -9; value <= 9; ++value) {
        if (!evaluateIntegerCode(code, function.getArgument(0), value, *emittedQuery, *emittedWitness)) {
            return false;
        }
    }
    bool hasWide = false;
    for (auto& operation : code) {
        if (failed(verify(&operation))) { return false; }
        for (auto type : operation.getResultTypes()) { hasWide |= type.isInteger(128); }
    }
    auto unavailable = expressions.integerPredicate(*relation, {expressions.input(hidden)}, 2, {1});
    auto& absent = plan.addPreparation(cuts[1]);
    builder.setInsertionPointToEnd(&absent);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> memo;
    return hasWide && failed(expressions.emit(unavailable, builder, cuts[1], memo)) && absent.empty() && memo.empty();
}
bool checkConstantBoundImplications(Value index)
{
    fs::RegionExpressions e;
    const auto x = e.input(index), zero = e.constant(0), one = e.constant(1), maximum = e.constant(UINT64_MAX);
    const auto plus = e.add(x, one), minus = e.sub(x, one);
    const auto four = e.constant(4);
    const auto singleton = e.land(e.le(four, x), e.le(x, four));
    if (!e.implies(singleton, e.eq(x, four)) ||
        !e.implies(e.land(singleton, e.lnot(e.eq(x, four))), e.boolean(false)) ||
        !e.implies(e.land(e.lt(x, four), e.le(four, x)), e.boolean(false)) ||
        !e.implies(e.lt(four, x), e.lt(e.constant(3), x)) ||
        e.implies(e.lnot(e.eq(x, four)), e.lt(x, four)) ||
        // Actual counterexamples: MAX+1=0 and 0-1=MAX in this machine layout.
        e.implies(e.eq(x, maximum), e.le(one, plus)) ||
        e.implies(e.eq(x, zero), e.lt(minus, maximum)) ||
        e.implies(e.slt(x, zero), e.lt(x, maximum))) { return false; }
    const std::vector<uint64_t> bounds{0, 1, 4, UINT64_MAX - 1, UINT64_MAX};
    std::set<uint64_t> samples;
    for (auto bound : bounds) {
        for (uint64_t delta = 0; delta <= 2; ++delta) {
            samples.insert(bound + delta);
            samples.insert(bound - delta);
        }
    }
    // Boundary values include both sides of each comparison and the shifted
    // boundaries of the modular operands; the oracle evaluates unsigned bits.
    if (samples.size() >= 64) { return false; }
    const auto mask = (uint64_t(1) << samples.size()) - 1;
    std::vector<std::pair<fs::RegionExpressions::Id, uint64_t>> formulas;
    for (unsigned root = 0; root < 3; ++root) {
        const auto operand = root == 0 ? x : root == 1 ? plus : minus;
        for (auto bound : bounds) {
            const auto c = e.constant(bound);
            for (unsigned comparison = 0; comparison < 5; ++comparison) {
                const auto formula = comparison == 0 ? e.eq(operand, c) :
                    comparison == 1 ? e.lt(operand, c) : comparison == 2 ? e.lt(c, operand) :
                    comparison == 3 ? e.le(operand, c) : e.le(c, operand);
                uint64_t truth = 0;
                unsigned bit = 0;
                for (auto sample : samples) {
                    const auto value = root == 0 ? sample : root == 1 ? sample + 1 : sample - 1;
                    const bool yes = comparison == 0 ? value == bound : comparison == 1 ? value < bound :
                        comparison == 2 ? bound < value : comparison == 3 ? value <= bound : bound <= value;
                    if (yes) { truth |= uint64_t(1) << bit; }
                    ++bit;
                }
                formulas.push_back({formula, truth});
                formulas.push_back({e.lnot(formula), (~truth) & mask});
            }
        }
    }
    for (const auto& premise : formulas) {
        for (const auto& consequence : formulas) {
            if (e.implies(premise.first, consequence.first) && (premise.second & ~consequence.second)) {
                return false;
            }
        }
    }
    return e.constructionError().empty();
}
bool checkImplicationTruthTables(Value index, Value predicate)
{
    fs::RegionExpressions expressions;
    const auto p = expressions.input(predicate), x = expressions.input(index);
    const auto q = expressions.eq(x, expressions.constant(5));
    const auto r = expressions.eq(x, expressions.constant(6));
    using Formula = std::pair<fs::RegionExpressions::Id, unsigned>;
    // p is independent; q=(x==5) and r=(x==6) cannot both hold. Keep the
    // existing eight-bit encoding and ignore its two impossible valuations.
    std::vector<Formula> formulas{{p, 0xaa}, {q, 0xcc}, {r, 0xf0},
        {expressions.boolean(false), 0}, {expressions.boolean(true), 255}};
    for (uint32_t i = 0; i < 3; ++i) {
        auto [a, av] = formulas[i];
        formulas.push_back({expressions.lnot(a), (~av) & 255});
        for (uint32_t j = 0; j < 3; ++j) {
            auto [b, bv] = formulas[j];
            formulas.push_back({expressions.land(a, b), av & bv});
            formulas.push_back({expressions.lor(a, b), av | bv});
            formulas.push_back({expressions.select(a, b, expressions.boolean(false)), av & bv});
            formulas.push_back({expressions.select(a, expressions.boolean(true), b), av | bv});
            formulas.push_back({expressions.select(a, expressions.boolean(false), b), (~av) & bv & 255});
            formulas.push_back({expressions.select(a, b, expressions.boolean(true)), (av & bv) | ((~av) & 255)});
            formulas.push_back({expressions.select(a, b, r), (av & bv) | ((~av) & 0xf0)});
            auto covered = expressions.land(expressions.land(a, b),
                expressions.lnot(expressions.land(expressions.land(a, b), r)));
            formulas.push_back({covered, av & bv & (~0xf0) & 255});
        }
    }
    for (auto [premise, truth] : formulas) {
        for (auto [consequence, implied] : formulas) {
            if (expressions.implies(premise, consequence) && (truth & ~implied & 0x3f)) { return false; }
        }
    }
    return expressions.constructionError().empty();
}
bool checkEmission(func::FuncOp function, ArrayRef<Operation*> cuts)
{
    fs::RegionExpressions expressions;
    auto x = expressions.input(function.getArgument(0));
    auto divisor = expressions.constant(3);
    auto result = expressions.add(expressions.div(x, divisor), expressions.rem(x, divisor));
    fs::PreparedLogicalPlan plan(0);
    OpBuilder builder(function.getContext());
    auto& first = plan.addPreparation(cuts[0]);
    builder.setInsertionPointToEnd(&first);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> memo;
    auto firstValue = expressions.emit(result, builder, cuts[0], memo);
    const auto count = first.getOperations().size();
    auto repeated = expressions.emit(result, builder, cuts[0], memo);
    if (failed(firstValue) || failed(repeated) || *firstValue != *repeated || first.getOperations().size() != count) {
        return false;
    }
    for (auto& operation : first) {
        if (failed(verify(&operation))) { return false; }
    }
    auto& second = plan.addPreparation(cuts[1]);
    builder.setInsertionPointToEnd(&second);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> secondMemo;
    auto secondValue = expressions.emit(result, builder, cuts[1], secondMemo);
    return succeeded(secondValue) && *firstValue != *secondValue && !second.empty();
}
bool checkPlacementRetry(func::FuncOp function, ArrayRef<Operation*> cuts)
{
    fs::RegionExpressions expressions;
    auto late = expressions.input(cuts[0]->getResult(0));
    auto root = expressions.add(late, expressions.constant(7));
    fs::PreparedLogicalPlan plan(0);
    auto& unavailable = plan.addPreparation(cuts[0]);
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToEnd(&unavailable);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> earlyMemo;
    if (succeeded(expressions.emit(root, builder, cuts[0], earlyMemo)) || !unavailable.empty() ||
        !earlyMemo.empty() || !expressions.constructionError().empty() || expressions.lastEmissionError().empty()) {
        return false;
    }
    const auto failedPlacement = expressions.lastEmissionError();
    {
        fs::RegionExpressions::Transaction transaction(expressions);
        fs::PreparedLogicalPlan speculative(0);
        auto& available = speculative.addPreparation(cuts[1]);
        builder.setInsertionPointToEnd(&available);
        llvm::DenseMap<fs::RegionExpressions::Id, Value> memo;
        if (failed(expressions.emit(root, builder, cuts[1], memo)) ||
            !expressions.lastEmissionError().empty()) { return false; }
    }
    if (expressions.lastEmissionError() != failedPlacement) { return false; }
    // Exact queries and new expressions remain usable after a placement failure.
    auto predicate = expressions.lt(late, root);
    if (!expressions.implies(predicate, predicate)) { return false; }
    auto& available = plan.addPreparation(cuts[1]);
    builder.setInsertionPointToEnd(&available);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> lateMemo;
    return succeeded(expressions.emit(root, builder, cuts[1], lateMemo)) && !available.empty() &&
        expressions.error().empty() && expressions.lastEmissionError().empty();
}
bool rejectedWithoutCode(func::FuncOp function, Operation* cut, Value unavailable)
{
    fs::RegionExpressions expressions;
    auto input = expressions.input(unavailable);
    auto root = expressions.add(input, expressions.constant(7));
    fs::PreparedLogicalPlan plan(0);
    auto& block = plan.addPreparation(cut);
    OpBuilder builder(function.getContext());
    builder.setInsertionPointToEnd(&block);
    llvm::DenseMap<fs::RegionExpressions::Id, Value> memo;
    return failed(expressions.emit(root, builder, cut, memo)) && block.empty() && memo.empty();
}
} // namespace
LogicalResult runRegionExpressionChecks(func::FuncOp function)
{
    SmallVector<Operation*> cuts;
    Value hidden;
    function.walk([&](Operation* operation) {
        if (operation->hasAttr("test.cut")) { cuts.push_back(operation); }
        if (operation->hasAttr("test.hidden") && operation->getNumResults()) { hidden = operation->getResult(0); }
    });
    if (function.getNumArguments() != 2 || cuts.size() != 2 || !cuts[0]->getNumResults() || !hidden) {
        return function.emitError("regional expression fixture requires two arguments, cuts and a hidden result");
    }
    const auto before = render(function);
    if (!checkAlgebra(function.getArgument(0), function.getArgument(1))) {
        return function.emitError("regional expression algebra checks failed");
    }
    auto checked = [](StringRef name, bool result) {
        if (!result) { llvm::errs() << "regional expression subcheck failed: " << name << "\n"; }
        return result;
    };
    if (!checked("checkEmission", checkEmission(function, cuts)) ||
        !checked("checkTransaction", checkTransaction(function.getArgument(0))) ||
        !checked("checkPortProvenance", checkPortProvenance(function)) ||
        !checked("checkIncompatibleCrossings", checkIncompatibleCrossings(function)) ||
        !checked("checkRegionalNativeOrder", checkRegionalNativeOrder(function)) ||
        !checked("checkCrossChildNative", checkCrossChildNative(function)) ||
        !checked("checkConsolidatedCandidateExclusion", checkConsolidatedCandidateExclusion(function)) ||
        !checked("checkDistanceAlgebra", checkDistanceAlgebra(function.getArgument(0), function.getArgument(1))) ||
        !checked("checkGuardedMinimum", checkGuardedMinimum(function.getArgument(0), function.getArgument(1))) ||
        !checked("checkBoundedZeroEdges", checkBoundedZeroEdges(function.getArgument(0), function.getArgument(1))) ||
        !checked("checkThresholdAlgebra", checkThresholdAlgebra(function.getArgument(0))) ||
        !checked("checkConstantBoundImplications", checkConstantBoundImplications(function.getArgument(0))) ||
        !checked("checkImplicationTruthTables",
                 checkImplicationTruthTables(function.getArgument(0), function.getArgument(1))) ||
        !checked("checkIntegerAdapters", checkIntegerAdapters(function.getArgument(0), function.getArgument(1))) ||
        !checked("checkPartialIntegerAdapters",
                 checkPartialIntegerAdapters(function.getArgument(0), function.getArgument(1))) ||
        !checked("checkMappedIntegerAdapters", checkMappedIntegerAdapters(function, cuts)) ||
        !checked("checkIntegerEmission", checkIntegerEmission(function, cuts, hidden)) ||
        !checked("checkPlacementRetry", checkPlacementRetry(function, cuts)) ||
        !checked("checkSubstitution", checkSubstitution(function.getArgument(0), function.getArgument(1))) ||
        !checked("checkImport", checkImport(function.getArgument(0), function.getArgument(1))) ||
        !checked("checkImportOwners", checkImportOwners(function.getArgument(0))) ||
        !checked("checkImportReplay", checkImportReplay(cuts[0])) ||
        !checked("runNestedRegionalChecks", runNestedRegionalChecks(function)) ||
        !checked("runRepeatedRegionChecks", runRepeatedRegionChecks(function)) ||
        !checked("runRepeatedStorageChecks", runRepeatedStorageChecks(function.getContext())) ||
        !checked("runRepeatedPhaseChecks", runRepeatedPhaseChecks(function)) ||
        !checked("rejectedWithoutCode", rejectedWithoutCode(function, cuts[0], cuts[0]->getResult(0))) ||
        !checked("rejectedWithoutCode", rejectedWithoutCode(function, cuts[1], hidden)) || render(function) != before) {
        return function.emitError("regional expression checks failed");
    }
    llvm::outs() << "regional expression checks passed\n";
    return success();
}
