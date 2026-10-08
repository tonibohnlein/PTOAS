// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/FrontierSynch/ArithmeticPeriodicConversion.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
namespace mlir::pto::frontiersynch {
namespace {
using Id = RegionExpressions::Id;
using Range = std::pair<BoundInteger, BoundInteger>;
void charge(uint64_t& count)
{
    if (count != UINT64_MAX) { ++count; }
}
BoundInteger unsignedInteger(uint64_t value)
{
    return BoundInteger(static_cast<int64_t>(value >> 1)) * BoundInteger(2) +
           BoundInteger(static_cast<int64_t>(value & 1));
}
BoundInteger unsignedLimit()
{
    return BoundInteger(INT64_MAX) * BoundInteger(2) + BoundInteger(1);
}
uint64_t bits(const BoundInteger& value)
{
    std::string text;
    llvm::raw_string_ostream stream(text); stream << value;
    StringRef digits(text);
    const bool negative = digits.consume_front("-");
    APInt number(64, digits, 10);
    return (negative ? -number : number).getZExtValue();
}
BoundInteger evaluate(const ArithmeticDistanceBound& bound, ArrayRef<BoundInteger> parameters)
{
    auto value = bound.numerator.constant;
    for (std::size_t i = 0; i < parameters.size(); ++i) { value += bound.numerator.coefficients[i] * parameters[i]; }
    return bound.ceiling ? -floorDiv(-value, bound.denominator) : floorDiv(value, bound.denominator);
}
bool commonAtom(const IntegerConstraint& atom, bool source, const std::vector<IntegerSystem>& domains,
                ArithmeticPeriodicConversionCost& cost)
{
    if (domains.empty() || atom.coefficients[source ? 1 : 0] != 0) { return false; }
    std::vector<BoundInteger> coefficients{atom.coefficients[source ? 0 : 1]};
    coefficients.insert(coefficients.end(), atom.coefficients.begin() + 2, atom.coefficients.end());
    for (const auto& domain : domains) {
        bool covered = false;
        for (const auto& premise : domain.constraints()) {
            charge(cost.domainComparisons);
            if (premise.coefficients == coefficients && premise.bound <= atom.bound) { covered = true; break; }
        }
        if (!covered) { return false; }
    }
    return true;
}
bool commonCongruence(const IntegerCongruence& atom, bool source, const std::vector<IntegerSystem>& domains,
                      ArithmeticPeriodicConversionCost& cost)
{
    if (domains.empty() || atom.coefficients[source ? 1 : 0] != 0) { return false; }
    std::vector<BoundInteger> coefficients{atom.coefficients[source ? 0 : 1]};
    coefficients.insert(coefficients.end(), atom.coefficients.begin() + 2, atom.coefficients.end());
    for (const auto& domain : domains) {
        bool covered = false;
        for (const auto& premise : domain.congruences()) {
            charge(cost.domainComparisons);
            if (premise.coefficients == coefficients && premise.modulus == atom.modulus &&
                premise.residue == atom.residue) { covered = true; break; }
        }
        if (!covered) { return false; }
    }
    return true;
}
std::optional<ArithmeticDistanceInterval> normalize(const ArithmeticIntervalPiece& piece, unsigned parameters,
                                                   ArithmeticPeriodicConversionCost& cost)
{
    ArithmeticDistanceInterval interval;
    interval.source = piece.source; interval.target = piece.target;
    interval.native = piece.native; interval.originalPiece = piece.originalPiece;
    interval.parameterResidues = piece.parameterResidues;
    std::vector<IntegerConstraint> guards;
    std::vector<IntegerCongruence> congruences;
    for (const auto& atom : piece.relation.constraints()) {
        charge(cost.constraints);
        const auto& a = atom.coefficients[0]; const auto& b = atom.coefficients[1];
        if (a == 0 && b == 0) {
            guards.push_back({{atom.coefficients.begin() + 2, atom.coefficients.end()}, atom.bound}); continue;
        }
        if (a != -b) {
            if (commonAtom(atom, true, piece.sourceDomains, cost) ||
                commonAtom(atom, false, piece.targetDomains, cost)) { continue; }
            return {};
        }
        ArithmeticDistanceBound bound;
        bound.denominator = llvm::abs(b);
        bound.ceiling = b < 0;
        bound.numerator.constant = bound.ceiling ? -atom.bound : atom.bound;
        for (auto coefficient : llvm::ArrayRef<BoundInteger>(atom.coefficients).drop_front(2)) {
            bound.numerator.coefficients.push_back(bound.ceiling ? coefficient : -coefficient);
        }
        (bound.ceiling ? interval.lower : interval.upper).push_back(std::move(bound));
    }
    for (const auto& atom : piece.relation.congruences()) {
        charge(cost.constraints);
        if (atom.coefficients[0] != 0 || atom.coefficients[1] != 0) {
            if (commonCongruence(atom, true, piece.sourceDomains, cost) ||
                commonCongruence(atom, false, piece.targetDomains, cost)) { continue; }
            return {};
        }
        congruences.push_back({{atom.coefficients.begin() + 2, atom.coefficients.end()}, atom.residue, atom.modulus});
    }
    auto guard = IntegerSystem::create(parameters, guards, congruences);
    if (failed(guard)) { return {}; }
    interval.guard = std::move(*guard);
    return interval;
}
std::vector<Range> parameterRanges(const ArithmeticPeriodicInput& input, const ArithmeticDistanceInterval& interval)
{
    std::vector<Range> result;
    const BoundInteger period(static_cast<int64_t>(input.parameterPeriod));
    for (auto parameter : input.parameters) {
        if (auto value = input.expressions->constantValue(parameter)) {
            BoundInteger quotient = floorDiv(BoundInteger(APInt(64, *value).getSExtValue()), period);
            result.push_back({quotient, quotient});
        } else {
            const bool boolean = input.expressions->isBoolean(parameter);
            result.push_back({floorDiv(BoundInteger(boolean ? 0 : INT64_MIN), period),
                              floorDiv(BoundInteger(boolean ? 1 : INT64_MAX), period)});
        }
    }
    for (const auto& atom : interval.guard.constraints()) {
        std::optional<std::size_t> variable;
        bool unary = true;
        for (std::size_t i = 0; i < atom.coefficients.size(); ++i) {
            if (atom.coefficients[i] == 0) { continue; }
            if (variable) { unary = false; break; }
            variable = i;
        }
        if (!unary || !variable) { continue; }
        const auto& coefficient = atom.coefficients[*variable];
        auto& range = result[*variable];
        if (coefficient > 0) { range.second = std::min(range.second, floorDiv(atom.bound, coefficient)); }
        else { range.first = std::max(range.first, -floorDiv(atom.bound, -coefficient)); }
    }
    return result;
}
Range boundRange(const ArithmeticDistanceBound& bound, const std::vector<Range>& parameters)
{
    std::vector<BoundInteger> low, high;
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const bool positive = bound.numerator.coefficients[i] >= 0;
        low.push_back(positive ? parameters[i].first : parameters[i].second);
        high.push_back(positive ? parameters[i].second : parameters[i].first);
    }
    return {evaluate(bound, low), evaluate(bound, high)};
}
std::optional<Id> emitBound(const ArithmeticPeriodicInput& input, const ArithmeticDistanceInterval& interval,
                           const ArithmeticDistanceBound& bound, const Range& range)
{
    auto& e = *input.expressions;
    if (range.first == range.second) {
        if (range.first < BoundInteger(INT64_MIN) || range.first > unsignedLimit()) { return {}; }
        return e.constant(bits(range.first));
    }
    // Mixed signed values use signed comparison before converting a lower
    // bound into a nonnegative distance. Positive uint64 constants need no such
    // comparison; variable bounds must fit signed i64 under the exact guard.
    if (range.first < BoundInteger(INT64_MIN) || range.second > BoundInteger(INT64_MAX)) { return {}; }
    auto numerator = bound.numerator;
    if (bound.ceiling) { numerator.constant += bound.denominator - BoundInteger(1); }
    auto value = e.integerFloor(numerator, bound.denominator, input.parameters,
                                input.parameterPeriod, interval.parameterResidues);
    return value == RegionExpressions::invalid ? std::nullopt : std::optional<Id>(value);
}
bool emitInterval(const ArithmeticPeriodicInput& input, const ArithmeticDistanceInterval& interval,
                  GuardedPeriodicRecord& record)
{
    auto& e = *input.expressions;
    const uint64_t d0 = interval.source < interval.target ? 0 : 1;
    record = {interval.source, interval.target, e.constant(d0), e.boolean(false), d0};
    auto ranges = parameterRanges(input, interval);
    if (llvm::any_of(ranges, [](const auto& range) { return range.first > range.second; })) { return true; }
    for (const auto& bound : interval.upper) {
        if (boundRange(bound, ranges).second < BoundInteger(static_cast<int64_t>(d0))) { return true; }
    }
    record.active = e.integerPredicate(interval.guard, input.parameters, input.parameterPeriod,
                                       interval.parameterResidues);
    if (e.constantValue(record.active) == 0) { return true; }
    for (const auto& bound : interval.lower) {
        const auto range = boundRange(bound, ranges);
        if (range.second <= BoundInteger(static_cast<int64_t>(d0))) { continue; }
        auto value = emitBound(input, interval, bound, range);
        if (!value || range.second > unsignedLimit()) { return false; }
        auto nonnegative = range.first < 0 ? e.select(e.slt(*value, e.constant(0)), e.constant(0), *value) : *value;
        record.displacement = e.select(e.lt(record.displacement, nonnegative), nonnegative, record.displacement);
        record.maxDisplacement = std::max(record.maxDisplacement, bits(range.second));
    }
    for (const auto& bound : interval.upper) {
        const auto range = boundRange(bound, ranges);
        if (range.first >= BoundInteger(0) && range.first >= unsignedInteger(record.maxDisplacement)) { continue; }
        auto value = emitBound(input, interval, bound, range);
        if (!value) { return false; }
        auto nonnegative = range.first < 0 ? e.lnot(e.slt(*value, e.constant(0))) : e.boolean(true);
        record.active = e.land(record.active, e.land(nonnegative, e.le(record.displacement, *value)));
    }
    return e.constructionError().empty();
}
bool validInput(const ArithmeticPeriodicInput& input)
{
    if (!input.expressions || !input.expressions->constructionError().empty() || !input.parameterPeriod ||
        input.parameterPeriod > INT64_MAX || input.parameters.size() > UINT32_MAX - 2 ||
        input.payloads.size() > UINT32_MAX) { return false; }
    for (const auto& payload : input.payloads) {
        if (!input.expressions->isBoolean(payload.presence)) { return false; }
    }
    for (auto parameter : input.parameters) { if (parameter >= input.expressions->size()) { return false; } }
    for (const auto& piece : input.pieces) {
        if (piece.source >= input.payloads.size() || piece.target >= input.payloads.size() ||
            piece.relation.dimensions() != input.parameters.size() + 2 ||
            piece.parameterResidues.size() != input.parameters.size()) { return false; }
        for (auto residue : piece.parameterResidues) { if (residue >= input.parameterPeriod) { return false; } }
        for (const auto* domains : {&piece.sourceDomains, &piece.targetDomains}) {
            for (const auto& domain : *domains) {
                if (domain.dimensions() != input.parameters.size() + 1) { return false; }
            }
        }
    }
    return true;
}
} // namespace
ArithmeticPeriodicConversion convertArithmeticPeriodicIntervals(const ArithmeticPeriodicInput& input)
{
    ArithmeticPeriodicConversion out;
    out.expressions = input.expressions; out.payloads = input.payloads;
    if (!validInput(input)) { out.diagnostic = "invalid arithmetic periodic generator schema"; return out; }
    for (const auto& piece : input.pieces) {
        charge(out.cost.pieces);
        if (piece.relation.isKnownEmpty()) { continue; }
        auto interval = normalize(piece, input.parameters.size(), out.cost);
        if (!interval) {
            out.status = ArithmeticPeriodicStatus::NotDistanceIntervals;
            out.diagnostic = "generator has an unfactored absolute cutoff, occurrence congruence or varying coordinate";
            out.intervals.clear(); return out;
        }
        out.intervals.push_back(std::move(*interval));
    }
    if (out.cost.pieces == UINT64_MAX || out.cost.constraints == UINT64_MAX ||
        out.cost.domainComparisons == UINT64_MAX) {
        out.diagnostic = "arithmetic periodic normalization cost overflow";
        out.intervals.clear();
        return out;
    }
    out.status = ArithmeticPeriodicStatus::Applicable;
    auto& e = *input.expressions;
    RegionExpressions::Transaction transaction(e);
    const auto begin = e.size();
    for (const auto& interval : out.intervals) {
        GuardedPeriodicRecord record;
        if (!emitInterval(input, interval, record)) {
            out.exportError = "distance interval has no checked machine bound export";
            if (!e.constructionError().empty()) { out.exportError += ": " + e.constructionError(); }
            out.generators.clear(); out.nativePrerequisites.clear();
            out.generatorPieces.clear(); out.nativePieces.clear();
            return out;
        }
        (interval.native ? out.nativePrerequisites : out.generators).push_back(record);
        (interval.native ? out.nativePieces : out.generatorPieces).push_back(interval.originalPiece);
    }
    bool numerical = true;
    std::vector<PeriodicPayload> payloads;
    std::vector<PeriodicRecord> records, native;
    for (const auto& payload : out.payloads) {
        numerical &= e.constantValue(payload.presence) == 1; payloads.push_back({payload.pipe});
    }
    for (const auto* source : {&out.generators, &out.nativePrerequisites}) {
        for (const auto& record : *source) {
            auto active = e.constantValue(record.active), distance = e.constantValue(record.displacement);
            if (!active || (*active && !distance)) { numerical = false; continue; }
            if (*active) {
                (source == &out.generators ? records : native).push_back({record.source, record.target, *distance});
            }
        }
    }
    if (numerical) {
        auto analysis = analyzePeriodicDemands(payloads, records, native);
        if (analysis.error.empty()) { out.numerical = std::move(analysis); }
    }
    out.guarded = analyzeGuardedPeriodicQuotient(input.expressions, out.payloads, out.generators,
                                                out.nativePrerequisites);
    if (!out.guarded->error.empty()) {
        out.exportError = out.guarded->error;
        out.guarded.reset();
        if (!out.numerical) {
            out.generators.clear(); out.nativePrerequisites.clear();
            out.generatorPieces.clear(); out.nativePieces.clear();
            return out;
        }
    }
    out.cost.expressionNodes = e.size() - begin;
    transaction.commit(); return out;
}
} // namespace mlir::pto::frontiersynch
