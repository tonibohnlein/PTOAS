// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#include "PTO/Transforms/FrontierSynch/FrontierSynch.h"
#include "OriginalReadQueries.h"
#include "PTO/Transforms/FrontierSynch/OccurrenceQueries.h"
#include "PTO/Transforms/FrontierSynch/OriginalLifetimes.h"
#include "PTO/Transforms/FrontierSynch/OriginalStructure.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include <set>

namespace mlir::pto::frontiersynch {
LogicalResult run(func::FuncOp function, const SyncInput& input)
{
    OriginalStructure original;
    if (failed(importOriginalStructure(function, input, original))) {
        return failure();
    }
    OriginalReadQueries readers(original);
    OccurrenceQueries occurrences(original);
    OriginalLifetimes lifetimes(original);
    if (!lifetimes.complete()) {
        return function.emitError("frontier-synch: original lifetime analysis failed: ") << lifetimes.reason();
    }
    std::set<std::pair<std::size_t, PipelineType>> projections;
    for (const auto& operation : original.operations) {
        for (const auto& access : operation.accesses) {
            if (access.read) {
                projections.emplace(access.cell, operation.instruction->kPipeValue);
            }
        }
    }
    std::size_t exactReaders = 0, unknownReaders = 0, bankPermutations = 0;
    for (const auto& [cell, pipe] : projections) {
        ReaderIntervalQuery query;
        query.cell = cell;
        query.reader = pipe;
        const auto& answer = readers.query(query);
        exactReaders += answer.status == OriginalReaderFrontiers::Status::Exact && answer.guardsAvailableAtReadSites;
        unknownReaders += !answer.complete() || (answer.status == OriginalReaderFrontiers::Status::Exact &&
                                                 !answer.guardsAvailableAtReadSites);
    }
    const auto participation = readers.participationDemands();
    std::size_t dueRequirements = 0;
    for (std::size_t site = 0; site < original.operations.size(); ++site) {
        dueRequirements += lifetimes.requirementsAt(site).size();
    }
    for (const auto& relation : original.physicalAddresses) {
        bankPermutations += occurrences.bank(relation).exactPermutation;
    }
    return function.emitError("frontier-synch: imported ")
           << input.instructions().size() << " instruction phases through InsertSync; analyzed "
           << original.cells.size() << " storage cells, " << exactReaders
           << " guardsAvailableAtReadSites reader frontiers, " << unknownReaders << " unresolved reader projections, "
           << participation.size() << " participation demands, and " << bankPermutations
           << " qualified bank relations; indexed " << dueRequirements << " original storage requirements and "
           << lifetimes.stats().accessIncidences << " physical access incidences; construction is not implemented yet";
}
} // namespace mlir::pto::frontiersynch
