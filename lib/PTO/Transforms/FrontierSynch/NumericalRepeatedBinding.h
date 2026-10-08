// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_FRONTIERSYNCH_NUMERICALREPEATEDBINDING_H
#define PTO_FRONTIERSYNCH_NUMERICALREPEATEDBINDING_H
#include "PTO/Transforms/FrontierSynch/RepeatedRegion.h"
#include "PTO/Transforms/FrontierSynch/NumericalWeightedRepetition.h"
namespace mlir::pto::frontiersynch {
struct NumericalBindingWork {
    uint64_t bodyQueries = 0, orderingQueries = 0;
    NumericalWeightedRepetitionCost index;
};
struct NumericalRepeatedBinding {
    NumericalWeightedRepetition analysis;
    std::vector<uint32_t> ports;
    std::vector<RegionalEvent> events;
    std::vector<RepeatedCrossing> covers;
};
// Evaluate a supplied regional context. Unknown guards/order stay on the shared
// guarded route. Ordering/context evaluation is charged separately from the
// numerical theorem, whose input is an evaluated ordered port frame.
std::optional<NumericalRepeatedBinding> bindNumericalRepeated(
    const RegionalAnalysis& body, const std::vector<RegionalEvent>& ports,
    const std::vector<RepeatedCrossing>& crossings, NumericalBindingWork& work);
} // namespace mlir::pto::frontiersynch
#endif
