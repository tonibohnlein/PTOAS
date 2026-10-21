// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Typed constant-role acknowledgment recipe. No IR, allocation or JSON.
#ifndef PTO_FRONTIERSYNCH_PREFIXRECIPE_H
#define PTO_FRONTIERSYNCH_PREFIXRECIPE_H
#include "PTO/Transforms/FrontierSynch/Shared/ServicePruning.h"
namespace mlir::pto::frontiersynch {
struct PrefixRole {
    uint32_t source = 0, target = 0;
    bool reverse = false;
    uint64_t ordinal = 0;
};
enum class PrefixCommandKind { Set, Wait, Barrier };
struct PrefixCommand {
    PrefixCommandKind kind;
    uint32_t source = 0, target = 0;
    bool reverse = false;
};
// A role key is (actual direction, F/R), not the service orientation. Compute
// once from the baseline; deletion never recolors surviving commands.
std::vector<PrefixRole> prefixRoles(const std::vector<SourcePipeSite>& sites, const ServiceMask& masks);
std::vector<PrefixCommand> prefixCommands(uint32_t target, const std::vector<uint32_t>& sources);
} // namespace mlir::pto::frontiersynch
#endif
