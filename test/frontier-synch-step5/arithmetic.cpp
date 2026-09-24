// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/OriginalValueArithmetic.h"
#include <cstdlib>
#include <cstdint>
#include <iostream>
static void check(bool condition, int line = __builtin_LINE())
{
    if (!condition) {
        std::cerr << "step-5 semantic assertion failed at line " << line << "\n";
        std::exit(1);
    }
}
using namespace mlir::pto::frontiersynch::value_arithmetic;

static int64_t decode(uint64_t bits, unsigned width, bool unsign)
{
    return !unsign && (bits & (uint64_t(1) << (width - 1))) ? int64_t(bits) - (int64_t(1) << width) : int64_t(bits);
}
int main()
{
    uint64_t checks = 0;
    // Independent widened-integer oracle, not another call to checked().
    for (unsigned width = 2; width <= 8; ++width) {
        for (bool unsign : {false, true}) {
            Integer type{width, unsign};
            const int64_t minimum = unsign ? 0 : -(int64_t(1) << (width - 1));
            const int64_t maximum = unsign ? type.mask() : (int64_t(1) << (width - 1)) - 1;
            for (uint64_t a = 0; a <= type.mask(); ++a) {
                for (uint64_t b = 0; b <= type.mask(); ++b) {
                    const auto x = decode(a, width, unsign), y = decode(b, width, unsign);
                    for (auto op :
                         {Binary::Add, Binary::Subtract, Binary::Multiply, Binary::Minimum, Binary::Maximum,
                          Binary::Remainder}) {
                        auto answer = checked(op, type, a, b);
                        bool admitted = true;
                        int64_t expected = 0;
                        switch (op) {
                            case Binary::Add:
                                expected = x + y;
                                break;
                            case Binary::Subtract:
                                expected = x - y;
                                break;
                            case Binary::Multiply:
                                expected = x * y;
                                break;
                            case Binary::Minimum:
                                expected = x < y ? x : y;
                                break;
                            case Binary::Maximum:
                                expected = x > y ? x : y;
                                break;
                            case Binary::Remainder:
                                admitted = y != 0 && !(!unsign && x == minimum && y == -1);
                                expected = admitted ? x % y : 0;
                                break;
                        }
                        admitted &= expected >= minimum && expected <= maximum;
                        check(bool(answer) == admitted);
                        if (answer) {
                            check(*answer == (uint64_t(expected) & type.mask()));
                        }
                        ++checks;
                    }
                    const auto extrema = interval(type, a, b);
                    check(extrema && extrema->nonempty == (x < y));
                    if (x < y) {
                        check(extrema->first == a && extrema->last == (uint64_t(y - 1) & type.mask()));
                    }
                    ++checks;
                }
            }
        }
    }
    // Enumerate complete finite loops only in the independent test oracle.
    // The production qualifier uses the closed forms regardless of trip count.
    for (bool unsign : {false, true}) {
        Integer type{5, unsign};
        const int64_t maximum = unsign ? 31 : 15;
        for (uint64_t lower = 0; lower < 32; ++lower) {
            for (uint64_t upper = 0; upper < 32; ++upper) {
                for (uint64_t step = 1; step <= uint64_t(maximum); ++step) {
                    const auto lb = decode(lower, 5, unsign), ub = decode(upper, 5, unsign);
                    int64_t exit = lb;
                    unsigned trips = 0;
                    while (exit < ub) {
                        ++trips;
                        exit += step;
                    }
                    const bool safe = exit <= maximum;
                    check(countedLoop(type, lower, upper, step) == safe);
                    ++checks;
                    if (!safe) {
                        continue;
                    }
                    for (unsigned visit = 0; visit < trips; ++visit) {
                        const auto iv = uint64_t(lb + visit * int64_t(step)) & 31;
                        for (unsigned distance = 0; distance <= trips + 1; ++distance) {
                            auto previous = hasPrevious(type, lower, upper, step, iv, distance);
                            auto next = hasNext(type, lower, upper, step, iv, distance);
                            check(previous && *previous == (visit >= distance));
                            check(next && *next == (visit + distance < trips));
                            checks += 2;
                        }
                    }
                }
            }
        }
    }
    Integer s64{64, false}, u64{64, true};
    const auto maximum = std::numeric_limits<uint64_t>::max();
    check(!checked(Binary::Add, s64, s64.maximum(), 1));
    check(!checked(Binary::Subtract, s64, s64.minimum(), 1));
    check(!checked(Binary::Multiply, s64, s64.minimum(), maximum));
    check(checked(Binary::Multiply, s64, s64.minimum(), 1) == s64.minimum());
    check(!checked(Binary::Add, u64, maximum, 1));
    check(!checked(Binary::Multiply, u64, maximum, 2));
    check(checked(Binary::Multiply, u64, maximum, 1) == maximum);
    check(countedLoop(s64, s64.minimum(), s64.maximum(), 1));
    check(!countedLoop(s64, s64.maximum() - 1, s64.maximum(), 2));
    check(!interval(u64, 0, 0)->nonempty);
    check(!interval(s64, s64.minimum(), s64.minimum())->nonempty);
    check(interval(u64, 0, 1)->last == 0);
    check(hasNext(s64, 0, s64.maximum(), 1, s64.maximum() - 1, 1) == false);
    check(hasPrevious(u64, 0, maximum, 1, maximum - 1, maximum - 1) == true);
    check(hasNext(u64, 0, maximum, 1, maximum - 1, 1) == false);
    check(!checked(Binary::Add, {0, false}, 0, 0));
    check(!checked(Binary::Add, {65, false}, 0, 0));
    std::cout << "PASS: " << checks << " exhaustive arithmetic/loop checks plus 64-bit boundaries\n";
}
