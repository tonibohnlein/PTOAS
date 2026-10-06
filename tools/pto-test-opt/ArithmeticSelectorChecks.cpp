// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Small independent integer-point checks for DBM primitives and selectors.
#include "PTO/Transforms/FrontierSynch/ArithmeticSelectors.h"
#include "PTO/Transforms/FrontierSynch/CompactAllocation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
namespace fs = mlir::pto::frontiersynch;
using mlir::failed;
using mlir::succeeded;
namespace {
using Integer = fs::BoundInteger;
using System = fs::DifferenceBoundSystem;
fs::DifferenceBoundConstraint atom(unsigned lhs, unsigned rhs, int64_t bound)
{
    return {lhs, rhs, Integer(bound)};
}
bool contains(const System& system, llvm::ArrayRef<Integer> point)
{
    if (system.isEmpty() || system.dimensions() != point.size()) { return false; }
    for (const auto& row : system.constraints()) {
        const auto lhs = row.lhs ? point[row.lhs - 1] : Integer(0);
        const auto rhs = row.rhs ? point[row.rhs - 1] : Integer(0);
        if (lhs - rhs > row.bound) { return false; }
    }
    return true;
}
bool contains(llvm::ArrayRef<System> systems, llvm::ArrayRef<Integer> point)
{
    return llvm::any_of(systems, [&](const auto& system) { return contains(system, point); });
}
bool closureAndProjection()
{
    const auto input = System::create(2, {atom(1, 2, 3), atom(2, 0, 5), atom(0, 1, 2)});
    if (failed(input) || input->bound(1, 0) != Integer(8)) { return false; }
    auto projected = input->project({0});
    auto reversed = input->project({1, 0});
    if (failed(projected) || failed(reversed)) { return false; }
    for (int64_t x = -5; x <= 10; ++x) {
        bool witness = false;
        for (int64_t y = -10; y <= 10; ++y) {
            const bool original = x - y <= 3 && y <= 5 && x >= -2;
            witness |= original;
            if (contains(*input, {Integer(x), Integer(y)}) != original ||
                contains(*reversed, {Integer(y), Integer(x)}) != original) { return false; }
        }
        if (contains(*projected, {Integer(x)}) != witness) { return false; }
    }
    const auto strict = System::create(2, {atom(1, 2, -1)});
    const auto unbounded = System::create(2, {});
    if (failed(strict) || failed(unbounded)) { return false; }
    auto identified = strict->remap(1, {0, 0});
    auto extended = input->remap(3, {2, 0});
    auto freeProjection = unbounded->project({1});
    if (failed(identified) || !identified->isEmpty() || failed(extended) || failed(freeProjection) ||
        freeProjection->bound(0, 1) || freeProjection->bound(1, 0) ||
        !contains(*extended, {Integer(5), Integer(1000000000), Integer(8)})) { return false; }
    return failed(input->project({0, 0})) && failed(input->project({2})) &&
           failed(input->remap(1, {0})) && failed(input->remap(1, {0, 1})) &&
           failed(input->intersect(*projected)) && failed(System::create(1, {atom(2, 0, 0)}));
}
bool largeConstantsAndEmpty()
{
    const int64_t maximum = std::numeric_limits<int64_t>::max();
    const int64_t minimum = std::numeric_limits<int64_t>::min();
    for (auto value : {minimum, maximum}) {
        auto large = System::create(2, {atom(1, 2, value), atom(2, 0, value)});
        if (failed(large) || large->isEmpty() || large->bound(1, 0) != Integer(value) * Integer(2)) {
            return false;
        }
    }
    const auto universe = System::create(1, {});
    const auto upper = System::create(1, {atom(1, 0, maximum)});
    const auto lower = System::create(1, {{0, 1, -Integer(minimum)}});
    const auto truth = System::create(0, {});
    const auto falsehood = System::create(0, {atom(0, 0, -1)});
    if (failed(universe) || failed(upper) || failed(lower) || failed(truth) || failed(falsehood)) { return false; }
    auto above = fs::subtractDifferenceBoundUnions(1, {*universe}, {*upper});
    auto below = fs::subtractDifferenceBoundUnions(1, {*universe}, {*lower});
    auto empty = fs::subtractDifferenceBoundUnions(0, {*truth}, {*truth});
    auto nonempty = fs::subtractDifferenceBoundUnions(0, {*truth}, {*falsehood});
    auto emptyProjection = falsehood->project({});
    return succeeded(above) && succeeded(below) && succeeded(empty) && succeeded(nonempty) &&
        succeeded(emptyProjection) && emptyProjection->isEmpty() && empty->empty() &&
        contains(*nonempty, {}) && contains(*above, {Integer(maximum) + Integer(1)}) &&
        !contains(*above, {Integer(maximum)}) && contains(*below, {Integer(minimum) - Integer(1)}) &&
        !contains(*below, {Integer(minimum)});
}
bool subtraction()
{
    auto first = System::create(2, {atom(1, 0, 2), atom(0, 1, 2), atom(2, 0, 2), atom(0, 2, 2)});
    auto second = System::create(2, {atom(1, 0, 3), atom(0, 1, -1), atom(2, 0, 1), atom(0, 2, 1)});
    auto cut = System::create(2, {atom(1, 2, 0), atom(2, 1, 0)});
    auto half = System::create(2, {atom(1, 0, -1)});
    if (failed(first) || failed(second) || failed(cut) || failed(half)) { return false; }
    auto result = fs::subtractDifferenceBoundUnions(2, {*first, *second}, {*cut, *half, *cut});
    if (failed(result)) { return false; }
    for (int64_t x = -4; x <= 4; ++x) {
        for (int64_t y = -4; y <= 4; ++y) {
            const bool box = (x >= -2 && x <= 2 && y >= -2 && y <= 2) ||
                             (x >= 1 && x <= 3 && y >= -1 && y <= 1);
            if (contains(*result, {Integer(x), Integer(y)}) != (box && x != y && x >= 0)) { return false; }
        }
    }
    return true;
}
fs::ArithmeticRelationKey key(std::size_t target, uint64_t period = 2)
{
    return {{0, fs::ArithmeticEvent::Completion, {period - 1}},
            {target, fs::ArithmeticEvent::Start, {0}}, {period - 1}};
}
fs::ArithmeticDemandAnalysis certified()
{
    fs::ArithmeticDemandAnalysis result;
    result.period = 2;
    result.parameterCount = 1;
    result.pipeCount = 2;
    result.exactMinimum = true;
    return result;
}
bool selected(const fs::ArithmeticEndpointSelector& selector, int64_t x, int64_t p,
              std::optional<std::pair<std::size_t, int64_t>> expected)
{
    auto actual = fs::evaluateArithmeticSelector(selector, 2, {Integer(x)}, {Integer(p)});
    if (failed(actual) || actual->has_value() != expected.has_value()) { return false; }
    return !expected || ((**actual).site == expected->first && (**actual).coordinates.size() == 1 &&
                         (**actual).coordinates.front() == Integer(expected->second));
}
bool signedResiduesAndTags()
{
    // Quotient variables [source, target, parameter]. Each admitted invocation
    // has finite coordinates even for negative/unbounded symbolic parameters.
    auto first = System::create(3, {atom(2, 1, 1), atom(1, 2, -1), atom(1, 3, 2), atom(3, 1, 0)});
    auto second = System::create(3, {atom(2, 1, 2), atom(1, 2, -2), atom(1, 3, 4), atom(3, 1, -3)});
    if (failed(first) || failed(second)) { return false; }
    auto analysis = certified();
    analysis.minimumDemands[key(1)] = {*first, *first}; // Overlapping identical outputs: first match only.
    analysis.minimumDemands[key(2)] = {*second};
    auto selectors = fs::buildArithmeticSelectors(analysis, {0, 1, 1});
    if (!selectors.error.empty() || selectors.forward.size() != 1 || selectors.inverse.size() != 2 ||
        selectors.forward.front().pieces.size() != 2) { return false; }
    for (int64_t p = -7; p <= 7; ++p) {
        for (int64_t x = -9; x <= 15; ++x) {
            std::optional<std::pair<std::size_t, int64_t>> expected;
            if (p % 2 != 0 && x % 2 != 0) {
                const auto delta = (x - p) / 2;
                if (delta >= 0 && delta <= 2) { expected = {{1, x + 1}}; }
                if (delta >= 3 && delta <= 4) { expected = {{2, x + 3}}; }
            }
            if (!selected(selectors.forward.front(), x, p, expected)) { return false; }
            if (expected && !selected(selectors.inverse[expected->first - 1], expected->second, p, {{0, x}})) {
                return false;
            }
        }
    }
    Integer large(1);
    for (unsigned i = 0; i < 100; ++i) { large *= Integer(2); }
    auto value = fs::evaluateArithmeticSelector(selectors.forward.front(), 2,
                                                {large + Integer(1)}, {large + Integer(1)});
    return succeeded(value) && value->has_value() && (**value).coordinates.front() == large + Integer(2);
}
bool selectorValidation()
{
    auto relation = System::create(3, {atom(2, 1, 1), atom(1, 2, -1)});
    auto unbounded = System::create(3, {});
    if (failed(relation) || failed(unbounded)) { return false; }
    auto analysis = certified();
    analysis.minimumDemands[key(1)] = {*relation};
    analysis.exactMinimum = false;
    if (fs::buildArithmeticSelectors(analysis, {0, 1}).error.empty()) { return false; }
    analysis.exactMinimum = true;
    if (fs::buildArithmeticSelectors(analysis, {0}).error.empty()) { return false; }
    analysis.minimumDemands[key(1)] = {*unbounded};
    if (fs::buildArithmeticSelectors(analysis, {0, 1}).error.empty()) { return false; }
    analysis.minimumDemands.clear();
    auto invalid = key(1);
    invalid.parameterResidues = {2};
    analysis.minimumDemands[invalid] = {*relation};
    return !fs::buildArithmeticSelectors(analysis, {0, 1}).error.empty();
}
bool selectorLargeConstants()
{
    Integer large(1);
    for (unsigned i = 0; i < 100; ++i) { large *= Integer(2); }
    auto relation = System::create(3, {{2, 1, large}, {1, 2, -large}, atom(1, 3, 0), atom(3, 1, 0)});
    if (failed(relation)) { return false; }
    auto analysis = certified();
    analysis.period = 1;
    analysis.minimumDemands[key(1, 1)] = {*relation};
    auto selectors = fs::buildArithmeticSelectors(analysis, {0, 1});
    if (!selectors.error.empty()) { return false; }
    auto value = fs::evaluateArithmeticSelector(selectors.forward.front(), 1, {Integer(0)}, {Integer(0)});
    if (failed(value) || !value->has_value() || (**value).coordinates.front() != large) { return false; }
    relation = System::create(3, {atom(2, 1, 1), atom(1, 2, -1), atom(1, 3, 0), atom(3, 1, 0)});
    if (failed(relation)) { return false; }
    analysis.period = std::numeric_limits<uint64_t>::max();
    analysis.minimumDemands.clear();
    analysis.minimumDemands[key(1, analysis.period)] = {*relation};
    selectors = fs::buildArithmeticSelectors(analysis, {0, 1});
    if (!selectors.error.empty()) { return false; }
    value = fs::evaluateArithmeticSelector(selectors.forward.front(), analysis.period, {Integer(-1)}, {Integer(-1)});
    return succeeded(value) && value->has_value() && (**value).coordinates.front() == Integer(0);
}
bool exactCoalescing()
{
    for (int64_t gap : {1, 2}) {
        auto a = System::create(3, {atom(2, 1, 1), atom(1, 2, -1), atom(0, 1, 2),
            atom(1, 0, 0), atom(0, 3, 1), atom(3, 0, 1)});
        auto b = System::create(3, {atom(2, 1, 1), atom(1, 2, -1), atom(0, 1, -gap),
            atom(1, 0, 3), atom(0, 3, 1), atom(3, 0, 1)});
        if (failed(a) || failed(b)) { return false; }
        auto analysis = certified();
        analysis.period = 1;
        analysis.minimumDemands[key(1, 1)] = {*a, *b};
        auto selectors = fs::buildArithmeticSelectors(analysis, {0, 1});
        if (!selectors.error.empty() || selectors.forward.size() != 1 ||
            selectors.forward.front().pieces.size() != static_cast<std::size_t>(gap)) { return false; }
        for (int64_t x = -4; x <= 5; ++x) {
            for (int64_t p = -3; p <= 3; ++p) {
                bool present = p >= -1 && p <= 1 && ((x >= -2 && x <= 0) || (x >= gap && x <= 3));
                auto value = fs::evaluateArithmeticSelector(selectors.forward.front(), 1, {Integer(x)}, {Integer(p)});
                if (failed(value) || value->has_value() != present ||
                    (present && (**value).coordinates != std::vector<Integer>{Integer(x + 1)})) { return false; }
                auto inverse = fs::evaluateArithmeticSelector(selectors.inverse.front(), 1,
                    {Integer(x + 1)}, {Integer(p)});
                if (failed(inverse) || inverse->has_value() != present ||
                    (present && (**inverse).coordinates != std::vector<Integer>{Integer(x)})) { return false; }
            }
        }
    }
    return true;
}
bool uniformAllocationQueries()
{
    mlir::MLIRContext context;
    auto analysis = certified();
    analysis.period = 1;
    // One handoff a_i -> b_i, with 0<=i<N. Native pipe chains do not
    // themselves order b_i completion before the next a start.
    auto handoff = System::create(3, {atom(2, 1, 0), atom(1, 2, 0), atom(0, 1, 0), atom(1, 3, -1)});
    if (failed(handoff)) { return false; }
    analysis.minimumDemands[key(1, 1)] = {*handoff};
    const std::map<std::pair<std::size_t, std::size_t>, int64_t> records{{{0, 1}, 0}};
    if (fs::arithmeticAllocationCertificate(analysis, {0, 1}, records, 0, &context)) { return false; }
    fs::ArithmeticRelationKey reuse{{1, fs::ArithmeticEvent::Completion, {0}},
                                   {0, fs::ArithmeticEvent::Start, {0}}, {0}};
    for (int64_t gap : {1, 2}) {
        auto order = System::create(3, {atom(1, 2, -gap), atom(0, 1, 0), atom(2, 3, -1)});
        if (failed(order)) { return false; }
        analysis.requiredOrder[reuse] = {*order};
        auto proof = fs::arithmeticAllocationCertificate(analysis, {0, 1}, records, 0, &context);
        if (static_cast<bool>(proof) != (gap == 1)) { return false; }
    }
    return true;
}
bool zeroCoordinates()
{
    auto truth = System::create(0, {});
    if (failed(truth)) { return false; }
    auto analysis = certified();
    analysis.period = 1;
    analysis.parameterCount = 0;
    fs::ArithmeticRelationKey relation{{0, fs::ArithmeticEvent::Completion, {}},
                                       {1, fs::ArithmeticEvent::Start, {}}, {}};
    analysis.minimumDemands[relation] = {*truth};
    auto selectors = fs::buildArithmeticSelectors(analysis, {0, 1});
    if (!selectors.error.empty() || selectors.forward.size() != 1) { return false; }
    auto value = fs::evaluateArithmeticSelector(selectors.forward.front(), 1, {}, {});
    return succeeded(value) && value->has_value() && (**value).site == 1 && (**value).coordinates.empty();
}
} // namespace
int runArithmeticSelectorChecks()
{
    if (!closureAndProjection() || !largeConstantsAndEmpty() || !subtraction()) {
        llvm::errs() << "exact difference-bound primitive checks failed\n";
        return 1;
    }
    if (!signedResiduesAndTags() || !selectorValidation() || !selectorLargeConstants() ||
        !exactCoalescing() || !uniformAllocationQueries() || !zeroCoordinates()) {
        llvm::errs() << "arithmetic endpoint selector checks failed\n";
        return 1;
    }
    llvm::outs() << "difference-bound primitives and arithmetic endpoint selectors passed\n";
    return 0;
}
