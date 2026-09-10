// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "PTO/Transforms/InsertSync/LogicalSyncCompactForms.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
using namespace mlir::pto::logical_sync::compact;
namespace {
uint64_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
__int128 evaluate(const LinearForm& form, const std::vector<int64_t>& values) {
    __int128 result = form.constant;
    for (size_t i = 0; i < values.size(); ++i) result += __int128(form.coefficients[i]) * values[i];
    return result;
}
bool accepts(__int128 expression, Bound bound) {
    if (bound.comparison == Comparison::Equal) return expression == bound.value;
    if (bound.comparison == Comparison::AtLeast) return expression >= bound.value;
    return expression <= bound.value;
}
}
int main() {
    const int64_t lo = std::numeric_limits<int64_t>::min(), hi = std::numeric_limits<int64_t>::max();
    const std::vector<int64_t> extremes{lo, lo + 1, -3, -1, 0, 1, 3, hi - 1, hi};
    for (int64_t a : extremes) for (int64_t b : extremes) {
        auto sum = checkedAdd(a, b), difference = checkedSubtract(a, b);
        __int128 exactSum = __int128(a) + b, exactDifference = __int128(a) - b;
        require(bool(sum) == (exactSum >= lo && exactSum <= hi), "addition representability");
        require(!sum || __int128(*sum) == exactSum, "addition value");
        require(bool(difference) == (exactDifference >= lo && exactDifference <= hi), "subtraction representability");
        require(!difference || __int128(*difference) == exactDifference, "subtraction value");
    }
    std::mt19937_64 random(0x0a45c011);
    for (unsigned trial = 0; trial < 4000; ++trial) {
        LinearForm expression{{int64_t(random() % 9) - 4, int64_t(random() % 9) - 4}, int64_t(random() % 31) - 15};
        if (expression.coefficients == std::vector<int64_t>{0, 0}) expression.coefficients[0] = 1;
        LinearForm row = expression;
        const bool opposite = random() & 1;
        if (opposite) for (auto& c : row.coefficients) c = -c;
        row.constant = int64_t(random() % 41) - 20;
        for (bool equality : {false, true}) {
            auto bound = matchBound(row, equality, expression);
            require(bool(bound), "matching affine vector must be recognized");
            for (int64_t x = -4; x <= 4; ++x) for (int64_t y = -3; y <= 3; ++y) {
                auto value = evaluate(row, {x, y});
                require(accepts(evaluate(expression, {x, y}), *bound) == (equality ? value == 0 : value >= 0),
                        "direct comparison changes a predicate");
            }
        }
    }
    for (int64_t c : extremes) for (int64_t m : {int64_t(2), int64_t(3), int64_t(16), hi})
        for (int64_t sign : {int64_t(-1), int64_t(1)}) {
            auto residue = matchResidue({sign, 0, -m, c}, 2);
            require(bool(residue) && residue->coordinate == 0 && residue->modulus == m, "congruence recognized");
            require(residue->value >= 0 && residue->value < m, "canonical residue range");
            __int128 equation = __int128(sign) * residue->value + c;
            require(equation % m == 0, "congruence with extreme constant");
            for (int64_t x = -10; x <= 10; ++x) {
                __int128 original = __int128(sign) * x + c;
                __int128 delta = __int128(x) - residue->value;
                require((original % m == 0) == (delta % m == 0), "residue changes integer membership");
            }
        }
    require(!matchResidue({1, 1, -2, 0}, 2), "coupled public coordinates must not be guessed");
    require(!matchResidue({1, -2, 1, 0}, 1), "two witnesses must not be guessed");
    require(!matchResidue({1, lo, 0}, 1), "unrepresentable positive modulus");
    require(!matchResidue({2, -3, 0}, 1), "nonunit scalar coefficient");
    require(!matchBound(LinearForm{{1}, lo}, false, LinearForm{{1}, hi}), "overflowing comparison bound");
    require(!matchBound(LinearForm{{1, 1}, 0}, false, LinearForm{{1, 0}, 0}), "unrelated coefficient vector");
    require(!matchBound(LinearForm{{0}, 0}, true, LinearForm{{0}, 0}), "constant form is not an SSA test");
    auto minCoefficient = matchBound(LinearForm{{lo}, 0}, false, LinearForm{{lo}, 0});
    require(bool(minCoefficient) && minCoefficient->value == 0, "INT64_MIN vector must not overflow during sign test");
    for (unsigned capacity = 0; capacity <= 12; ++capacity) {
        for (unsigned mask = 0; mask < (1u << capacity); ++mask) {
            unsigned visits = 0;
            auto keys = unusedFirstKeys(capacity, [&](unsigned k) { ++visits; return mask & (1u << k); });
            require(visits == capacity, "one occupancy query per key");
            require(keys.size() == capacity, "retain entire available key pool");
            auto fitting = eventKeyTrialOrder(capacity, capacity, [&](unsigned k) { return mask & (1u << k); });
            require(fitting == keys, "fitting domain must use unused-first order");
            unsigned unnecessaryLookups = 0;
            auto scarce = eventKeyTrialOrder(capacity, size_t(capacity) + 1, [&](unsigned) {
                ++unnecessaryLookups; return true;
            });
            require(unnecessaryLookups == 0, "scarce-domain order needs no occupancy lookups");
            for (unsigned k = 0; k < capacity; ++k)
                require(scarce[k] == k, "scarce domain must retain the previous assignment search");
            bool sawOccupied = false;
            std::vector<unsigned> seen;
            for (unsigned k : keys) {
                const bool occupied = mask & (1u << k);
                require(!sawOccupied || occupied, "occupied key tried before an unused key");
                sawOccupied |= occupied;
                seen.push_back(k);
            }
            std::sort(seen.begin(), seen.end());
            for (unsigned k = 0; k < capacity; ++k) require(seen[k] == k, "key missing or duplicated");
        }
    }
    std::cout << checks << " compact-form/key-order assertions passed\n";
}
