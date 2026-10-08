// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALNUMERICALINTERFACE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REGIONALNUMERICALINTERFACE_H
#include "PTO/Transforms/FrontierSynch/RegionalAnalysis.h"
#include "PTO/Transforms/FrontierSynch/ChainInterface.h"
namespace mlir::pto::frontiersynch {
struct RegionalNumericalInterface {
    std::shared_ptr<const NumericalChainInterface> index;
    // Dense event IDs, including distinct start/completion ports. Numerical
    // exports require evaluated leaf ordinals and enclosing visit coordinates.
    std::vector<RegionalEvent> events;
    std::vector<std::pair<uint32_t, PeriodicEventKind>> chainKeys;
    std::function<std::optional<bool>(RegionalEvent, RegionalEvent, NumericalChainQueryCost&)> query;
    // False is event->ports, true is ports->event. Queries descend only the
    // event's child path; each merge propagates one vector with numerical indices.
    std::function<std::optional<std::vector<uint32_t>>(RegionalEvent, bool, NumericalChainQueryCost&)> thresholds;
};
} // namespace mlir::pto::frontiersynch
#endif
