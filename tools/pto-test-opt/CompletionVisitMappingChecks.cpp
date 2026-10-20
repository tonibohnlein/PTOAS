// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Check active physical-map equivalence without occurrence/activity premises.
// The reference evaluates the original table independently of the adapter.
#include "../../lib/PTO/Transforms/FrontierSynch/Shared/CompletionVisitMapping.h"
using namespace mlir;
namespace fs = pto::frontiersynch;

namespace {
uint64_t mappingOffset(const fs::PhysicalRecordAllocation& record, ArrayRef<uint64_t> coordinates)
{
    if (!record.tupleRule) {
        return (record.stride * (coordinates.front() % record.ids.size()) + record.phase) % record.ids.size();
    }
    uint64_t offset = record.tupleRule->base;
    for (const auto& term : record.tupleRule->terms) {
        offset += term.scale * ((term.stride * (coordinates[term.coordinate] % term.modulus) +
                                 term.phase) % term.modulus);
    }
    return offset;
}
bool sameVisitAssignment(const fs::PhysicalRecordAllocation& record, uint64_t arity)
{
    auto converted = fs::completionVisitMapping(record, arity, 2);
    if (!converted || converted->record != record.record || converted->sourcePipe != record.sourcePipe ||
        converted->targetPipe != record.targetPipe || !converted->tupleRule ||
        converted->tupleRule->coordinateCount != arity + 2) { return false; }
    // The converted tuple evaluates global palette offsets; the original tuple
    // selects a table entry. Test active maps without any activity assumptions.
    for (uint64_t value : std::initializer_list<uint64_t>{0, 1, 2, 3, 7, 255, uint64_t(INT64_MAX), UINT64_MAX}) {
        for (uint64_t varying = 0; varying < arity + 2; ++varying) {
            SmallVector<uint64_t> coordinates(arity + 2, 0);
            coordinates[varying] = value;
            if (static_cast<uint64_t>(record.ids[mappingOffset(record, coordinates)]) !=
                mappingOffset(*converted, coordinates)) { return false; }
        }
    }
    return true;
}
}
bool runCompletionVisitMappingChecks()
{
    fs::PhysicalRecordAllocation invalid;
    invalid.ids = {0, 0};
    if (fs::completionVisitMapping(invalid, 0, 1) ||
        fs::completionVisitMapping(invalid, 1, UINT64_MAX)) { return false; }
    invalid.ids = {-1, -1};
    if (fs::completionVisitMapping(invalid, 1, 1)) { return false; }
    for (uint64_t arity : {1U, 2U, 4U}) {
        for (int64_t id : {0, 3, 5}) {
            for (unsigned width : {1U, 2U, 4U}) {
                fs::PhysicalRecordAllocation record;
                record.record = 17; record.sourcePipe = 4; record.targetPipe = 3;
                record.ids.assign(width, id); record.stride = 1; record.phase = 1;
                if (!sameVisitAssignment(record, arity)) { return false; }
                record.tupleRule = fs::PhysicalTupleRule{arity, 0, {{arity - 1, 1, 0, width, 1}}};
                if (!sameVisitAssignment(record, arity)) { return false; }
                record.tupleRule->coordinateCount = arity + 1;
                if (fs::completionVisitMapping(record, arity, 2)) { return false; }
                record.tupleRule->coordinateCount = arity;
                record.tupleRule->base = width;
                if (fs::completionVisitMapping(record, arity, 2)) { return false; }
                record.tupleRule->base = 0;
                record.tupleRule->terms.front().modulus = 0;
                if (fs::completionVisitMapping(record, arity, 2)) { return false; }
            }
        }
        fs::PhysicalRecordAllocation nonconstant;
        nonconstant.ids = {2, 3}; nonconstant.stride = 1; nonconstant.phase = 1;
        if (!sameVisitAssignment(nonconstant, arity)) { return false; }
        nonconstant.tupleRule = fs::PhysicalTupleRule{arity, 0, {{arity - 1, 1, 0, 2, 1}}};
        if (!sameVisitAssignment(nonconstant, arity)) { return false; }
        nonconstant.ids = {2, 4};
        if (fs::completionVisitMapping(nonconstant, arity, 2)) { return false; }
        nonconstant.ids.clear();
        if (fs::completionVisitMapping(nonconstant, arity, 2)) { return false; }
    }
    return true;
}
