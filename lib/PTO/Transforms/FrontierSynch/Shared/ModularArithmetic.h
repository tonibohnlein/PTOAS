// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Shared exact modular inversion for certified coprime operands.
#ifndef PTO_FRONTIERSYNCH_MODULAR_ARITHMETIC_H
#define PTO_FRONTIERSYNCH_MODULAR_ARITHMETIC_H
#include "llvm/ADT/APInt.h"
#include <cstdint>
#include <numeric>
#include <optional>
namespace mlir::pto::frontiersynch::detail {
// Precondition: a nonzero modulus and coprime value; callers divide by gcd.
inline uint64_t modularInverse(uint64_t value, uint64_t modulus)
{
    uint64_t oldRemainder = modulus, remainder = value;
    // 256-bit signed intermediates contain products of a 64-bit quotient and
    // the extended-Euclidean coefficients, without host signed overflow.
    llvm::APInt oldCoefficient(256, 0), coefficient(256, 1);
    while (remainder != 0) {
        const auto quotient = oldRemainder / remainder;
        const auto nextRemainder = oldRemainder % remainder;
        const auto nextCoefficient = oldCoefficient - llvm::APInt(256, quotient) * coefficient;
        oldRemainder = remainder;
        remainder = nextRemainder;
        oldCoefficient = coefficient;
        coefficient = nextCoefficient;
    }
    const llvm::APInt bound(256, modulus);
    auto result = oldCoefficient.srem(bound);
    if (result.isNegative()) {
        result += bound;
    }
    return result.getZExtValue();
}
// Preprocess a*n+b == lane (mod width) once, then answer each lane in
// constant arithmetic work. Zero width and incompatible lanes are unavailable.
class ModuloLaneCongruence {
public:
    ModuloLaneCongruence(uint64_t stride, uint64_t phase, uint64_t width)
        : width(width), phase(width ? phase % width : 0), divisor(std::gcd(stride, width)),
          modulus(width && divisor ? width / divisor : 0)
    {
        if (modulus > 1) { inverse = modularInverse(stride / divisor, modulus); }
    }
    uint64_t period() const { return modulus; }
    std::optional<uint64_t> residue(uint64_t lane) const
    {
        if (!modulus || lane >= width) { return std::nullopt; }
        const auto delta = lane >= phase ? lane - phase : width - (phase - lane);
        if (delta % divisor) { return std::nullopt; }
        if (modulus == 1) { return 0; }
        return (llvm::APInt(128, delta / divisor) * llvm::APInt(128, inverse))
            .urem(llvm::APInt(128, modulus)).getZExtValue();
    }
private:
    uint64_t width, phase, divisor, modulus, inverse = 0;
};
} // namespace mlir::pto::frontiersynch::detail
#endif
