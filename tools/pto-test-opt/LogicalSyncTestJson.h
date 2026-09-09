// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_LOGICAL_SYNC_TEST_JSON_H
#define PTO_LOGICAL_SYNC_TEST_JSON_H
#include "PTO/Transforms/InsertSync/LogicalSyncRelations.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir::pto::logical_sync::testing {
inline llvm::json::Object encode(const Relation& relation)
{
    using llvm::json::Array;
    using llvm::json::Object;
    Array pieces;
    for (const auto& p : relation.getAllDisjuncts()) {
        Array eq, ge;
        for (bool equality : {true, false}) {
            unsigned count = equality ? p.getNumEqualities() : p.getNumInequalities();
            for (unsigned i = 0; i < count; ++i) {
                Array row;
                for (unsigned j = 0; j < p.getNumCols(); ++j) {
                    std::string coefficient;
                    llvm::raw_string_ostream stream(coefficient);
                    stream << (equality ? p.atEq(i, j) : p.atIneq(i, j));
                    row.push_back(coefficient);
                }
                (equality ? eq : ge).push_back(std::move(row));
            }
        }
        pieces.push_back(Object{{"locals", p.getNumLocalVars()}, {"eq", std::move(eq)}, {"ge", std::move(ge)}});
    }
    return Object{
        {"d", relation.getNumDomainVars()},
        {"r", relation.getNumRangeVars()},
        {"s", relation.getNumSymbolVars()},
        {"pieces", std::move(pieces)}};
}
} // namespace mlir::pto::logical_sync::testing
#endif
