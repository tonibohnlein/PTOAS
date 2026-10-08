// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_CHAININTERFACEINTERNAL_H
#define PTO_FRONTIERSYNCH_CHAININTERFACEINTERNAL_H
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
namespace mlir::pto::frontiersynch::chain {
NumericalChainInterface initialize(std::vector<std::vector<uint32_t>> chains);
void invert(NumericalChainInterface& index);
std::optional<std::vector<bool>> reduce(const NumericalChainInterface& left,
    const NumericalChainInterface& right, const std::vector<NumericalCrossing>& edges, uint64_t& operations);
} // namespace mlir::pto::frontiersynch::chain
#endif
