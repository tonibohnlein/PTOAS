// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Logical identities include an ordered member tuple and remain separate from
// physical event attributes. ODS checks index operand types. Domain membership
// and matching across SET/WAIT sites require the family certificate, not a local
// operation verifier.
#include "PTO/IR/PTO.h"

namespace mlir::pto {
namespace {
bool supportedPipe(PIPE pipe)
{
    switch (pipe) {
    case PIPE::PIPE_S:
    case PIPE::PIPE_V:
    case PIPE::PIPE_M:
    case PIPE::PIPE_MTE1:
    case PIPE::PIPE_MTE2:
    case PIPE::PIPE_MTE3:
    case PIPE::PIPE_MTE4:
    case PIPE::PIPE_MTE5:
    case PIPE::PIPE_V2:
    case PIPE::PIPE_FIX:
        return true;
    default:
        return false;
    }
}

LogicalResult verifyLogicalSync(Operation* operation, PIPE source, PIPE target, int64_t plan, int64_t record)
{
    if (plan < 0 || record < 0) {
        return operation->emitOpError("plan_id and record_id must be nonnegative");
    }
    const bool supported = supportedPipe(source) && supportedPipe(target);
    if (!supported) {
        return operation->emitOpError("requires concrete source and destination pipes");
    }
    if (source == target) {
        return operation->emitOpError("source and destination pipes must differ");
    }
    return success();
}
} // namespace

LogicalResult LogicalSetOp::verify()
{
    return verifyLogicalSync(getOperation(), getSrcPipe().getPipe(), getDstPipe().getPipe(),
                             getPlanIdAttr().getInt(), getRecordIdAttr().getInt());
}

LogicalResult LogicalWaitOp::verify()
{
    return verifyLogicalSync(getOperation(), getSrcPipe().getPipe(), getDstPipe().getPipe(),
                             getPlanIdAttr().getInt(), getRecordIdAttr().getInt());
}
} // namespace mlir::pto
