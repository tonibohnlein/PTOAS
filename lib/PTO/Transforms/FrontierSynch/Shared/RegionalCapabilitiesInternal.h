// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Common adapter checks. Specialized query, selector and command representations
// remain owned by their producers; this boundary does not promise capabilities.
#ifndef PTO_FRONTIERSYNCH_REGIONALCAPABILITIESINTERNAL_H
#define PTO_FRONTIERSYNCH_REGIONALCAPABILITIESINTERNAL_H
#include "PTO/Transforms/FrontierSynch/Shared/RegionalAnalysis.h"
namespace mlir::pto::frontiersynch {
struct RegionalContextView {
    const RegionExpressions* expressions = nullptr;
    const SyncStorageEffects* accessModel = nullptr;
    GMAliasPolicy gmAliasPolicy = GMAliasPolicy::MayAlias;
};
RegionalContextView regionalContext(const RegionalAnalysis& region);
bool matchesRegionalContext(const RegionalAnalysis& region, RegionalContextView expected);
// Validate all coordinate frames once when constructing an adapter, never on
// every query. Empty-anchor flat producers retain callback-owned type checking.
bool validRegionalOccurrenceSchema(const RegionalAnalysis& region);
// An identity/presence/order transformation must rebuild these certificates.
// Modeled accesses, original anchors and producer-owned demand recipes survive.
void invalidateDerivedRegionalCertificates(RegionalAnalysis& region);
} // namespace mlir::pto::frontiersynch
#endif
