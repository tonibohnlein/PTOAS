// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_FRONTIERSYNCH_ORIGINALWRITECOVERAGE_H
#define PTO_FRONTIERSYNCH_ORIGINALWRITECOVERAGE_H
#include "PTO/Transforms/FrontierSynch/OriginalStructure.h"

namespace mlir::pto::frontiersynch {
// The shared op interface supplies the semantic valid-element guarantee;
// this adapter supplies its physical interpretation. Existing translated
// def/use vectors, conservative aliases and read-before-write effects survive.
class OriginalWriteCoverage {
public:
    explicit OriginalWriteCoverage(const SyncTileDescriptorState& descriptors) : descriptors(descriptors) {}

    WriteCoverage qualify(const PhysicalOperation& phase, const BaseMemInfo& memory)
    {
        using Status = WriteCoverage::Status;
        auto failure = [](Status status) { return WriteCoverage{status, {}}; };
        auto* operation = phase.instruction->elementOp;
        auto contract = dyn_cast<TileWriteCoverageOpInterface>(operation);
        const Value output = memory.baseBuffer;
        if (!output || !contract || !contract.writesAllValidTileElements(output)) {
            return failure(Status::UnknownSemantics);
        }
        // A whole-operation guarantee cannot be copied to each internal phase.
        if (!phase.beforeExecutable || !phase.afterExecutable) {
            return failure(Status::NotWholeInstruction);
        }
        unsigned locations = 0;
        for (const auto* effect : phase.instruction->defVec) {
            locations += effect && effect->baseBuffer == output;
        }
        if (locations != 1 || memory.aliasesUnknownRange || memory.baseAddresses.size() != 1) {
            return failure(Status::NonUniqueLocation);
        }
        const auto valid = descriptors.at(output, operation);
        if (!valid) {
            return failure(Status::UnknownDescriptor);
        }
        const auto type = dyn_cast<TileBufType>(output.getType());
        const auto layout = getLayout(type);
        if (!layout) {
            return failure(Status::UnsupportedLayout);
        }
        if ((*valid)[0] > layout->rows || (*valid)[1] > layout->cols) {
            return failure(Status::UnknownDescriptor);
        }

        Value root = output;
        uint64_t offset = 0;
        if (auto view = output.getDefiningOp<SubViewOp>()) {
            // One-level, one-major-segment views have an unambiguous contiguous
            // physical interpretation. In particular, never use the narrower result
            // shape as its parent's row stride. Multirow/nested/dynamic views keep
            // their may effects until that strided-view contract is qualified.
            root = view.getSource();
            const auto parentType = dyn_cast<TileBufType>(root.getType());
            const auto parent = getLayout(parentType);
            if (!parent || !root.getDefiningOp<AllocTileOp>() || parentType.getElementType() != type.getElementType() ||
                parent->rowMajor != layout->rowMajor || (layout->rowMajor ? layout->rows : layout->cols) != 1 ||
                view.getOffsets().size() != 2 || view.getSizes().size() != 2) {
                return failure(Status::UnqualifiedView);
            }
            const auto row = SyncSlotMapping::evaluateConstant(view.getOffsets()[0], constants);
            const auto col = SyncSlotMapping::evaluateConstant(view.getOffsets()[1], constants);
            const auto rows = cast<IntegerAttr>(view.getSizes()[0]).getInt();
            const auto cols = cast<IntegerAttr>(view.getSizes()[1]).getInt();
            if (!row || !col || rows <= 0 || cols <= 0 || uint64_t(rows) != layout->rows ||
                uint64_t(cols) != layout->cols || *row > parent->rows || *col > parent->cols ||
                layout->rows > parent->rows - *row || layout->cols > parent->cols - *col) {
                return failure(Status::UnqualifiedView);
            }
            using namespace write_coverage_detail;
            const auto major = multiply(layout->rowMajor ? *row : *col, parent->stride);
            const auto minor = multiply(layout->rowMajor ? *col : *row, parent->elementBytes);
            const auto displacement = major && minor ? add(*major, *minor) : std::nullopt;
            if (!displacement) {
                return failure(Status::UnqualifiedView);
            }
            offset = *displacement;
        }
        auto allocation = root.getDefiningOp<AllocTileOp>();
        if (!allocation || !allocation.getAddr() || memory.rootBuffer != root || !memory.hasKnownPhysicalAddresses ||
            memory.scope == AddressSpace::GM || memory.scope == AddressSpace::Zero) {
            return failure(Status::UnknownAddress);
        }
        // Requalify the original scalar. The shared may-address extraction can
        // strip casts or retain alternative addresses; neither is a must proof.
        const auto base = SyncSlotMapping::evaluateConstant(allocation.getAddr(), constants);
        const auto begin = base ? write_coverage_detail::add(*base, offset) : std::nullopt;
        const auto rootLayout = getLayout(dyn_cast<TileBufType>(root.getType()));
        if (!begin || !rootLayout || !write_coverage_detail::add(*base, rootLayout->span)) {
            return failure(Status::UnknownAddress);
        }
        const auto allocated = StridedWriteRange::get(
            *begin, layout->rowMajor ? layout->rows : layout->cols, layout->minorBytes, layout->stride);
        if (!allocated || *begin != memory.baseAddresses.front()) {
            return failure(Status::GeometryMismatch);
        }
        // getLayout() checked the span. A subview above has one segment, so this
        // equality also checks the translator's parent-relative displacement.
        if (layout->span != memory.allocateSize) {
            return failure(Status::GeometryMismatch);
        }
        const uint64_t validMajor = (*valid)[layout->rowMajor ? 0 : 1];
        const uint64_t validMinor = (*valid)[layout->rowMajor ? 1 : 0];
        const auto minorBytes = write_coverage_detail::multiply(validMinor, layout->elementBytes);
        const auto bytes =
            minorBytes ? StridedWriteRange::get(*begin, validMajor, *minorBytes, layout->stride) : std::nullopt;
        if (!bytes) {
            return failure(Status::GeometryMismatch);
        }
        return {Status::Qualified, *bytes};
    }

private:
    struct Layout {
        uint64_t rows, cols, elementBytes, minorBytes, stride, span;
        bool rowMajor;
    };
    static std::optional<Layout> getLayout(TileBufType type)
    {
        if (!type || type.getShape().size() != 2 || type.getShape()[0] <= 0 || type.getShape()[1] <= 0 ||
            type.getSLayoutValueI32() != static_cast<int32_t>(SLayout::NoneBox)) {
            return {};
        }
        unsigned bits = 0;
        if (auto integer = dyn_cast<IntegerType>(type.getElementType())) {
            bits = integer.getWidth();
        } else if (auto floating = dyn_cast<FloatType>(type.getElementType())) {
            bits = floating.getWidth();
        }
        if (!bits || bits % 8) {
            return {}; // Packed/sub-byte and opaque element layouts need their own map.
        }
        const auto blockLayout = type.getBLayoutValueI32();
        if (blockLayout != 0 && blockLayout != 1) {
            return {};
        }
        const auto compact = type.getCompactModeI32();
        if (compact < 0 || compact > 2) {
            return {};
        }
        const bool rowMajor = blockLayout == static_cast<int32_t>(BLayout::RowMajor);
        const uint64_t rows = type.getShape()[0], cols = type.getShape()[1];
        const uint64_t major = rowMajor ? rows : cols, minor = rowMajor ? cols : rows;
        using namespace write_coverage_detail;
        const auto elements = add(minor, compact == static_cast<int32_t>(CompactMode::RowPlusOne));
        const auto stride = elements ? multiply(*elements, bits / 8) : std::nullopt;
        const auto minorBytes = multiply(minor, bits / 8);
        const auto distance = stride ? multiply(major - 1, *stride) : std::nullopt;
        const auto span = distance && minorBytes ? add(*distance, *minorBytes) : std::nullopt;
        if (!span) {
            return {};
        }
        return Layout{rows, cols, bits / 8, *minorBytes, *stride, *span, rowMajor};
    }
    const SyncTileDescriptorState& descriptors;
    SyncSlotMapping::ConstantCache constants;
};
} // namespace mlir::pto::frontiersynch
#endif
