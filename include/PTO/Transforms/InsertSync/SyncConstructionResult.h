// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_SYNCCONSTRUCTIONRESULT_H
#define PTO_TRANSFORMS_INSERTSYNC_SYNCCONSTRUCTIONRESULT_H
#include <cstdint>
#include <string>
namespace mlir::pto::logical_sync {
// Shared outcome only. No occurrence representation is part of this interface.
struct ConstructionResult {
    enum Status { Applied, Unsupported, AnalysisLimit, Unproved, AllocationFailure, InternalError };
    Status status = Unsupported;
    std::string reason;
    uint64_t work = 0;
    unsigned requirements = 0, handoffs = 0, barriers = 0;
};
}
#endif
