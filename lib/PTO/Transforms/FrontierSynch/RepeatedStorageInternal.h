// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Private validated geometry retained by the selector and owner exports.
#ifndef PTO_FRONTIERSYNCH_REPEATEDSTORAGEINTERNAL_H
#define PTO_FRONTIERSYNCH_REPEATEDSTORAGEINTERNAL_H
#include "PTO/Transforms/FrontierSynch/RepeatedStorage.h"
namespace mlir::pto::frontiersynch {
struct RepeatedStorage::State {
    using Id = RegionExpressions::Id;
    struct Piece {
        uint64_t begin = 0, end = 0;
        RegionalAccessBoundary access;
        SyncAccessMode mode = SyncAccessMode::Read;
        uint32_t pipe = 0;
        std::optional<RepeatedStorageInnerRun> inner;
    };
    struct Family {
        RepeatedStorageFamily spec;
        Id origin = RegionExpressions::invalid;
        std::vector<Piece> pieces;
        uint64_t extent = 0;
    };
    RegionalAnalysis body;
    scf::ForOp loop;
    Id trips = RegionExpressions::invalid;
    std::vector<Family> families;
    std::vector<RepeatedPersistentStorage> persistent;
    std::vector<std::size_t> effectIds;
    std::string error;
    RegionExpressions& expressions() const { return *body.expressions; }
};
} // namespace mlir::pto::frontiersynch
#endif
