// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Exact query outputs consumed under one immutable incoming activity condition.
#ifndef PTO_FRONTIERSYNCH_SEQUENCEINCOMINGQUERY_H
#define PTO_FRONTIERSYNCH_SEQUENCEINCOMINGQUERY_H
#include "SequenceAnalysisInternal.h"
namespace mlir::pto::frontiersynch {
// One predeletion incoming graph and append-only arena. No callback executes
// during proof rollback; only original expression IDs and literal facts survive.
// Public answers are activity-masked and must never enter the unrestricted query cache.
class SequenceIncomingQuery {
public:
    explicit SequenceIncomingQuery(SequenceAnalysisState& state) : state(state) {}
    std::optional<Expr> query(std::size_t source, std::size_t target, Expr activity);
    std::optional<Expr> query(SequenceEvent source, SequenceEvent target, Expr activity);
    // One alternative path, masked once. Prefix qualification still precedes
    // suffix qualification; a false suffix does not evaluate the prefix.
    std::optional<Expr> path(std::size_t source, std::size_t entrySource,
                            std::size_t entryTarget, std::size_t target, Expr activity);
private:
    // Meaningful only under activity. Never expose this answer to callers or
    // place it in the unrestricted regional reachability cache.
    std::optional<Expr> restricted(SequenceEvent source, SequenceEvent target, Expr activity);
    std::optional<uint64_t> fact(Expr activity, Expr condition);
    Expr coordinate(Expr activity, Expr root);
    SequenceEvent occurrence(Expr activity, SequenceEvent event);
    SequenceAnalysisState& state;
    std::map<std::pair<Expr, Expr>, std::optional<uint64_t>> facts;
    std::map<std::pair<Expr, Expr>, Expr> coordinates;
    using Key = std::tuple<Expr, SequenceAnalysisState::EventKey, SequenceAnalysisState::EventKey>;
    std::map<Key, Expr> answers;
    std::map<Key, Expr> maskedAnswers;
};
} // namespace mlir::pto::frontiersynch
#endif
