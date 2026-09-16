// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_IR_SYNCPROTOCOLMODEL_H
#define PTO_IR_SYNCPROTOCOLMODEL_H
#include "PTO/IR/PTO.h"
#include <optional>
#include <string>

namespace mlir::pto {
// Lowering-owned description of a preserved protocol component. Byte effects
// describe this core only. Peer matching/progress remain the original protocol
// contract; no peer receipt grants local completion in an autosync constructor.
struct SyncProtocolModel {
  enum Kind { Initialize, Send, Receive, Release, Collective } kind = Initialize;
  Value handle, tile, globalStorage, localBase;
  PIPE pipeline = PIPE::PIPE_UNASSIGNED;
  SmallVector<Value> reads, writes;
  unsigned localSlots = 0;
  // Cross-core flags are a distinct namespace from directional local events.
  unsigned crossCoreFlagBase = 0, crossCoreFlagCount = 0;
  // Preserved intrinsic behavior, not extra generated commands or checker
  // credit. A local drain neither consumes local events nor proves peer
  // participation. The current constructors conservatively retain residuals
  // across this boundary instead of exploiting its completion.
  bool localDrainBefore = false;
  std::optional<SyncCoreType> participants;
  std::string contract, gap;
  bool complete() const { return gap.empty(); }
};
// Only qualified lowering families return a complete model. A recognized but
// unqualified variant returns a model with a diagnostic, never an empty effect
// list interpreted as completeness. No synchronization policy lives here.
std::optional<SyncProtocolModel> getSyncProtocolModel(Operation *operation);
} // namespace mlir::pto
#endif
