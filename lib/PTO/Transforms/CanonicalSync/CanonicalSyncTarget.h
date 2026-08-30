// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED
// ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR
// FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software
// repository for the full text of the License.

#ifndef PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCTARGET_H
#define PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCTARGET_H

#include "PTO/Transforms/CanonicalSync/CanonicalSyncAnalysis.h"

namespace mlir {
namespace pto {
namespace canonical_sync_detail {

/// Resolve the function's authoritative target and return the versioned
/// completion capabilities exposed to CanonicalSync analysis. Unsupported,
/// missing, or conflicting targets return a disabled capability set.
CanonicalSyncTargetCapabilities
getCanonicalSyncTargetCapabilities(func::FuncOp function);

} // namespace canonical_sync_detail
} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_CANONICALSYNC_CANONICALSYNCTARGET_H
