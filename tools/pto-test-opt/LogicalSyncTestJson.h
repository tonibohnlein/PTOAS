// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under the CANN Open Software License Agreement Version 2.0.
// See LICENSE for details. Provided AS IS, WITHOUT WARRANTIES OF ANY KIND.
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
