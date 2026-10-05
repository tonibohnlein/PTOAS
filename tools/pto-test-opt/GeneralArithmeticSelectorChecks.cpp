// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Independent integer-point and complete-tuple checks for exact general selectors.
#include "PTO/Transforms/FrontierSynch/GeneralArithmeticSelectors.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
namespace fs = mlir::pto::frontiersynch;
using mlir::failed;
using mlir::succeeded;
namespace {
using Integer = fs::BoundInteger;
using System = fs::IntegerSystem;
fs::IntegerConstraint row(std::initializer_list<int64_t> coefficients, int64_t bound)
{
    fs::IntegerConstraint result;
    for (auto coefficient : coefficients) { result.coefficients.emplace_back(coefficient); }
    result.bound = Integer(bound);
    return result;
}
Integer dot(llvm::ArrayRef<Integer> coefficients, llvm::ArrayRef<Integer> point)
{
    Integer result(0);
    for (auto [coefficient, value] : llvm::zip(coefficients, point)) { result += coefficient * value; }
    return result;
}
bool contains(const System& system, llvm::ArrayRef<Integer> point)
{
    if (system.dimensions() != point.size()) { return false; }
    for (const auto& constraint : system.constraints()) {
        if (dot(constraint.coefficients, point) > constraint.bound) { return false; }
    }
    for (const auto& congruence : system.congruences()) {
        if (mod(dot(congruence.coefficients, point) - congruence.residue, congruence.modulus) != Integer(0)) {
            return false;
        }
    }
    return true;
}
bool contains(llvm::ArrayRef<System> systems, llvm::ArrayRef<Integer> point)
{
    return llvm::any_of(systems, [&](const auto& system) { return contains(system, point); });
}
bool witnesses(const System& system)
{
    auto projected = system.project({0});
    auto selected = system.eliminateWithWitness(1);
    if (failed(projected) || failed(selected)) { return false; }
    for (int64_t x = -5; x <= 5; ++x) {
        bool expected = false, found = false;
        for (int64_t y = -16; y <= 16; ++y) { expected |= contains(system, {Integer(x), Integer(y)}); }
        if (contains(*projected, {Integer(x)}) != expected) { return false; }
        for (const auto& branch : *selected) {
            if (!contains(branch.domain, {Integer(x)})) { continue; }
            if (branch.denominator <= Integer(0) || branch.numerator.coefficients.size() != 1) { return false; }
            const auto numerator = dot(branch.numerator.coefficients, {Integer(x)}) + branch.numerator.constant;
            if (mod(numerator, branch.denominator) != Integer(0) ||
                !contains(system, {Integer(x), numerator / branch.denominator})) { return false; }
            found = true;
        }
        if (found != expected) { return false; }
    }
    return true;
}
bool primitiveChecks()
{
    auto parity = System::create(2, {row({-1, 2}, 0), row({1, -2}, 0)});
    auto upper = System::create(2, {row({-1, 1}, 1)});
    auto lower = System::create(2, {row({2, -1}, 0)});
    auto free = System::create(2, {});
    auto modular = System::create(2, {}, {{{Integer(-1), Integer(1)}, Integer(1), Integer(3)}});
    auto impossible = System::create(2, {row({1, 1}, 1), row({-1, -1}, -1),
                                          row({1, -1}, 0), row({-1, 1}, 0)});
    if (failed(parity) || failed(upper) || failed(lower) || failed(free) || failed(modular) || failed(impossible)) {
        return false;
    }
    if (!witnesses(*parity) || !witnesses(*upper) || !witnesses(*lower) || !witnesses(*free) ||
        !witnesses(*modular) || !impossible->isEmpty()) { return false; }
    auto projected = impossible->project({0});
    auto falsehood = System::create(0, {row({}, -1)});
    System truth;
    return succeeded(projected) && projected->empty() && succeeded(falsehood) && falsehood->isEmpty() &&
           !truth.isEmpty() && failed(parity->project({0, 0})) && failed(parity->eliminateWithWitness(2)) &&
           failed(System::create(1, {row({1, 0}, 0)})) &&
           failed(System::create(1, {}, {{{Integer(1)}, Integer(0), Integer(0)}}));
}
bool modularSubtraction()
{
    auto interval = System::create(1, {row({1}, 3), row({-1}, 3)});
    auto even = System::create(1, {}, {{{Integer(1)}, Integer(0), Integer(2)}});
    auto high = System::create(1, {row({-1}, -2)});
    if (failed(interval) || failed(even) || failed(high)) { return false; }
    auto difference = fs::subtractIntegerUnions(1, {*interval, *interval}, {*even, *high, *even});
    if (failed(difference)) { return false; }
    for (int64_t x = -5; x <= 5; ++x) {
        const bool expected = x >= -3 && x <= 3 && x % 2 != 0 && x < 2;
        if (contains(*difference, {Integer(x)}) != expected) { return false; }
    }
    Integer large(1);
    for (unsigned i = 0; i < 100; ++i) { large *= Integer(2); }
    auto scaled = System::create(2, {{{-large, large}, Integer(0)}, {{large, -large}, Integer(0)}});
    auto wideModulus = System::create(1, {}, {{{Integer(1)}, Integer(3), large}});
    auto contradiction = System::create(1, {}, {{{large}, Integer(1), large}});
    return succeeded(scaled) && witnesses(*scaled) && succeeded(wideModulus) &&
           wideModulus->congruences().size() == 1 && wideModulus->congruences().front().modulus == large &&
           contains(*wideModulus, {large + Integer(3)}) && !contains(*wideModulus, {large + Integer(2)}) &&
           succeeded(contradiction) && contradiction->isEmpty();
}
fs::GeneralArithmeticDemandAnalysis certified(uint64_t period)
{
    fs::GeneralArithmeticDemandAnalysis result;
    result.period = period;
    result.parameterCount = 1;
    result.pipeCount = 2;
    result.exactMinimum = true;
    return result;
}
fs::ArithmeticRelationKey key(unsigned target, uint64_t period)
{
    return {{0, fs::ArithmeticEvent::Completion, {period - 1}},
            {target, fs::ArithmeticEvent::Start, {0, period - 1}}, {period - 1}};
}
bool rationalTuples(uint64_t period)
{
    // [x,u,v,p]: u=(x+p)/2, v=(u+p)/3. The second coordinate requires
    // backsubstitution, with both divisibility constraints retained jointly.
    auto first = System::create(4, {row({-1, 2, 0, -1}, 0), row({1, -2, 0, 1}, 0),
        row({0, -1, 3, -1}, 0), row({0, 1, -3, 1}, 0), row({1, 0, 0, -1}, 12), row({-1, 0, 0, 1}, 0)});
    auto second = System::create(4, {row({-1, 2, 0, -1}, 0), row({1, -2, 0, 1}, 0),
        row({0, -1, 3, -1}, 0), row({0, 1, -3, 1}, 0), row({1, 0, 0, -1}, 24), row({-1, 0, 0, 1}, -13)});
    if (failed(first) || failed(second)) { return false; }
    auto analysis = certified(period);
    analysis.minimumDemands[key(1, period)] = {*first, *first};
    analysis.minimumDemands[key(2, period)] = {*second};
    const auto selectors = fs::buildGeneralArithmeticSelectors(analysis, {0, 1, 1});
    if (!selectors.error.empty() || selectors.forward.size() != 1 || selectors.inverse.size() != 2) { return false; }
    const auto scale = static_cast<int64_t>(period), residue = scale - 1;
    for (int64_t p = -4; p <= 4; ++p) {
        for (int64_t x = -8; x <= 29; ++x) {
            auto value = fs::evaluateGeneralArithmeticSelector(selectors.forward.front(), period,
                {Integer(scale * x + residue)}, {Integer(scale * p + residue)});
            const auto u = (x + p) / 2, v = (u + p) / 3;
            const bool expected = x >= p && x <= p + 24 && (x + p) % 2 == 0 && (u + p) % 3 == 0;
            if (failed(value) || value->has_value() != expected) { return false; }
            if (!expected) { continue; }
            const auto tag = x <= p + 12 ? 1U : 2U;
            const std::vector<Integer> tuple{Integer(scale * u), Integer(scale * v + residue)};
            if ((**value).site != tag || (**value).coordinates != tuple) { return false; }
            auto inverse = fs::evaluateGeneralArithmeticSelector(selectors.inverse[tag - 1], period,
                                                                 tuple, {Integer(scale * p + residue)});
            if (failed(inverse) || !inverse->has_value() || (**inverse).site != 0 ||
                (**inverse).coordinates != std::vector<Integer>{Integer(scale * x + residue)}) { return false; }
        }
    }
    if (period == 2) {
        auto wrong = fs::evaluateGeneralArithmeticSelector(selectors.forward.front(), 2, {Integer(0)}, {Integer(1)});
        if (failed(wrong) || wrong->has_value()) { return false; }
    }
    return true;
}
bool reflectedSizeAndValidation()
{
    std::optional<std::size_t> count;
    for (int64_t last : {7, 1000000007}) {
        auto relation = System::create(3, {row({1, 1, 0}, last), row({-1, -1, 0}, -last),
                                          row({1, 0, -1}, 3), row({-1, 0, 1}, 0)});
        if (failed(relation)) { return false; }
        auto analysis = certified(1);
        fs::ArithmeticRelationKey relationKey{{0, fs::ArithmeticEvent::Completion, {0}},
                                              {1, fs::ArithmeticEvent::Start, {0}}, {0}};
        analysis.minimumDemands[relationKey] = {*relation};
        const auto selectors = fs::buildGeneralArithmeticSelectors(analysis, {0, 1});
        if (!selectors.error.empty() || selectors.forward.size() != 1) { return false; }
        const auto size = selectors.forward.front().pieces.size();
        if (count && *count != size) { return false; }
        count = size;
        for (int64_t x = -4; x <= 4; ++x) {
            auto value = fs::evaluateGeneralArithmeticSelector(selectors.forward.front(), 1,
                                                               {Integer(x)}, {Integer(-2)});
            const bool expected = x >= -2 && x <= 1;
            if (failed(value) || value->has_value() != expected ||
                (expected && (**value).coordinates != std::vector<Integer>{Integer(last - x)})) { return false; }
        }
        analysis.exactMinimum = false;
        if (fs::buildGeneralArithmeticSelectors(analysis, {0, 1}).error.empty()) { return false; }
    }
    return true;
}
bool scalarEndpoint()
{
    auto relation = System::create(2, {row({1, -1}, 0), row({-1, 1}, 0)});
    if (failed(relation)) { return false; }
    auto analysis = certified(1);
    fs::ArithmeticRelationKey relationKey{{0, fs::ArithmeticEvent::Completion, {0}},
                                          {1, fs::ArithmeticEvent::Start, {}}, {0}};
    analysis.minimumDemands[relationKey] = {*relation};
    auto selectors = fs::buildGeneralArithmeticSelectors(analysis, {0, 1});
    if (!selectors.error.empty() || selectors.forward.size() != 1 || selectors.inverse.size() != 1) { return false; }
    auto forward = fs::evaluateGeneralArithmeticSelector(selectors.forward.front(), 1, {Integer(3)}, {Integer(3)});
    auto inverse = fs::evaluateGeneralArithmeticSelector(selectors.inverse.front(), 1, {}, {Integer(3)});
    auto absent = fs::evaluateGeneralArithmeticSelector(selectors.forward.front(), 1, {Integer(4)}, {Integer(3)});
    return succeeded(forward) && forward->has_value() && (**forward).site == 1 && (**forward).coordinates.empty() &&
           succeeded(inverse) && inverse->has_value() && (**inverse).site == 0 &&
           (**inverse).coordinates == std::vector<Integer>{Integer(3)} &&
           succeeded(absent) && !absent->has_value();
}
} // namespace
int runGeneralArithmeticSelectorChecks()
{
    if (!primitiveChecks() || !modularSubtraction()) {
        llvm::errs() << "general integer relation primitive checks failed\n";
        return 1;
    }
    if (!rationalTuples(1) || !rationalTuples(2) || !reflectedSizeAndValidation() || !scalarEndpoint()) {
        llvm::errs() << "general arithmetic selector tuple checks failed\n";
        return 1;
    }
    llvm::outs() << "exact integer primitives and general arithmetic selector tuples passed\n";
    return 0;
}
