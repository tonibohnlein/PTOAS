// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Invocation-wide publication consumes certified incoming ordinals; no demand analysis.
#ifndef PTO_FRONTIERSYNCH_INVOCATIONPREFIX_H
#define PTO_FRONTIERSYNCH_INVOCATIONPREFIX_H
#include "StructuredPrivateState.h"
#include <cstdint>
#include <map>
namespace mlir::pto::frontiersynch {
class PrefixExpressionProfile;
struct PrefixDirection {
    uint32_t sourcePipe = 0, targetPipe = 0;
    bool operator<(const PrefixDirection& other) const;
};
struct PrefixSource {
    Operation* original = nullptr;
    uint32_t pipe = 0;
    // Optional source-rank snapshot actions run after the count increment and
    // outgoing publications. They receive that actual executed-source ordinal.
    std::function<LogicalResult(OpBuilder&, const IRMapping&, llvm::MutableArrayRef<Value>,
                                Value, std::string&)> snapshot;
};
using PrefixIncomingSelector = std::function<FailureOr<Value>(
    OpBuilder&, const IRMapping&, llvm::ArrayRef<Value>, std::string&)>;
struct PrefixConsumer {
    Operation* original = nullptr;
    PrefixDirection direction;
    PrefixIncomingSelector incoming;
};
struct InvocationPrefixRecipe {
    int64_t planId = 0;
    // Explicit target capability: scalar bookkeeping and SCF acquisition loops.
    bool privateScalarControl = false;
    // Caller-certified upper bounds on all executed source phases in this
    // invocation. Bounds include branch alternatives and nested visit counts;
    // INT64_MAX-1 is the largest bound (exclusive N+1 loop sentinel must fit).
    std::map<uint32_t, uint64_t> sourceBounds;
    // Contribution actions/selectors see only these slots; coordinator counters
    // and watermarks are private. Same-pipe barriers and the fixed exit interface
    // remain caller-supplied actions; this constructor handles cross-pipe flags.
    std::vector<PrivateStateSlot> slots;
    std::vector<PrivateStateAction> actions;
    std::vector<PrefixSource> sources;
    std::vector<PrefixConsumer> consumers;
    std::vector<PrefixDirection> directions;
    std::shared_ptr<PrefixExpressionProfile> expressionProfile;
};
// Internal proof boundary: supplied selectors must denote the maximum EXACT
// retained incoming executed-source ordinal (zero if absent), be available at
// this consumer, and lie in [0,current N]. Sources must enumerate native
// completion order on each pipe exactly once. The coordinator must collect all
// records in each selected direction; mixing direct and prefix owners is illegal.
// This constructor validates structure/representability, not those mathematics.
// Returns an owned verified clone. No original mutation or physical event IDs.
FailureOr<std::unique_ptr<StructuredPrivateState>> materializeInvocationPrefix(
    func::FuncOp original, const InvocationPrefixRecipe& recipe, PrivateStateError& error);
} // namespace mlir::pto::frontiersynch
#endif
