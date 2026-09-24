// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_FRONTIERSYNCH_WRITECOVERAGE_H
#define PTO_FRONTIERSYNCH_WRITECOVERAGE_H
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::pto::frontiersynch {
namespace write_coverage_detail {
inline std::optional<uint64_t> add(uint64_t a, uint64_t b)
{
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return {};
    }
    return a + b;
}
inline std::optional<uint64_t> multiply(uint64_t a, uint64_t b)
{
    if (b && a > std::numeric_limits<uint64_t>::max() / b) {
        return {};
    }
    return a * b;
}
} // namespace write_coverage_detail

// A lower bound on bytes written, not a may-footprint or an allocation bound.
// Blocks are simultaneous portions of ONE physical write, never alternative
// selector values. No row, element, guard valuation or dynamic visit expansion.
struct StridedWriteRange {
    uint64_t begin = 0, blocks = 0, blockBytes = 0, strideBytes = 0;

    static std::optional<StridedWriteRange> get(
        uint64_t begin, uint64_t blocks, uint64_t blockBytes, uint64_t strideBytes)
    {
        if (!blocks || !blockBytes) {
            return StridedWriteRange{begin, 0, 0, 0};
        }
        if (strideBytes < blockBytes) {
            return {};
        }
        using namespace write_coverage_detail;
        const auto distance = multiply(blocks - 1, strideBytes);
        const auto last = distance ? add(begin, *distance) : std::nullopt;
        if (!last || !add(*last, blockBytes)) {
            return {};
        }
        return StridedWriteRange{begin, blocks, blockBytes, strideBytes};
    }

    // Nonempty half-open cells only. Recheck the shape so even an invalid
    // aggregate cannot manufacture a proof by wrapping arithmetic.
    bool covers(uint64_t cellBegin, uint64_t cellBytes) const
    {
        if (!blocks || !blockBytes || !cellBytes || cellBegin < begin || !get(begin, blocks, blockBytes, strideBytes)) {
            return false;
        }
        const uint64_t offset = cellBegin - begin;
        if (strideBytes == blockBytes) {
            // get() proves this product fits: its end is begin + blocks * blockBytes.
            const uint64_t span = blocks * blockBytes;
            return offset < span && cellBytes <= span - offset;
        }
        const uint64_t block = offset / strideBytes, inBlock = offset % strideBytes;
        return block < blocks && inBlock < blockBytes && cellBytes <= blockBytes - inBlock;
    }
};

// Retained with each translated write incidence. Qualified can still cover no
// complete current cell (an empty/partial valid rectangle or a coarse cell).
struct WriteCoverage {
    enum class Status {
        UnknownSemantics,
        NotWholeInstruction,
        NonUniqueLocation,
        UnknownDescriptor,
        UnsupportedLayout,
        UnqualifiedView,
        UnknownAddress,
        GeometryMismatch,
        Qualified
    } status = Status::UnknownSemantics;
    StridedWriteRange bytes;

    bool covers(uint64_t begin, uint64_t size) const
    {
        return status == Status::Qualified && bytes.covers(begin, size);
    }
    const char* reason() const
    {
        switch (status) {
            case Status::UnknownSemantics:
                return "unknown-semantics";
            case Status::NotWholeInstruction:
                return "not-whole-instruction";
            case Status::NonUniqueLocation:
                return "non-unique-location";
            case Status::UnknownDescriptor:
                return "unknown-descriptor";
            case Status::UnsupportedLayout:
                return "unsupported-layout";
            case Status::UnqualifiedView:
                return "unqualified-view";
            case Status::UnknownAddress:
                return "unknown-address";
            case Status::GeometryMismatch:
                return "geometry-mismatch";
            case Status::Qualified:
                return "qualified";
        }
        return "unknown-semantics";
    }
};
} // namespace mlir::pto::frontiersynch
#endif
