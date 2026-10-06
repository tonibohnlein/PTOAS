// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Checked byte reservations and exact selectors for evolving repeated storage.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_REPEATEDSTORAGE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_REPEATEDSTORAGE_H
#include "PTO/Transforms/FrontierSynch/RepeatedRegion.h"
namespace mlir::pto::frontiersynch {
enum class RepeatedStorageKind { VisitOwned, SharedReadOnly };
struct RepeatedStorageFamily {
    RepeatedStorageKind kind = RepeatedStorageKind::VisitOwned;
    AddressSpace space = AddressSpace::GM;
    // Zero-dimensional invariant origin; byteOffset and symbols describe a.
    SyncAccessRegion origin;
    uint64_t stride = 0;
    std::vector<std::size_t> effects;
};
struct RepeatedStorageOwner {
    RegionExpressions::Id present, visit, localByte;
};
class RepeatedStorage {
public:
    struct State;
    explicit RepeatedStorage(std::shared_ptr<State> state) : state(std::move(state)) {}
    std::optional<RepeatedStorageOwner> owner(std::size_t family, RegionalByteAddress address) const;
    std::optional<RegionalStorageSelectors> selectors(RegionalByteAddress address) const;
    // Only the inter-visit bridge view removes these effects. The original
    // body, its internal demands, and the exported effects remain intact.
    bool contains(std::size_t effect) const;
    const RegionalAnalysis& body() const;
    scf::ForOp loop() const;
    RegionExpressions::Id trips() const;
    ArrayRef<std::size_t> effects() const;
private:
    std::shared_ptr<State> state;
};
struct RepeatedStorageResult {
    std::string error;
    std::shared_ptr<RepeatedStorage> storage;
};
// Validate supplied modeled maps, not inferred instruction footprints. Each
// selected effect's complete union must equal a+t*S+local; local maps must have
// finite constant ranges independent of all body occurrences. The body's exact
// first/last effect selectors then apply to each covered byte. More general
// byte-dependent occurrence selectors need another adapter, not a discharge.
RepeatedStorageResult buildRepeatedStorage(const RegionalAnalysis& body, scf::ForOp loop,
    RegionExpressions::Id trips, ArrayRef<RepeatedStorageFamily> families);
// Propose and validate the constant-stride subclass directly from shared maps.
// Missing body effect selectors remain an error, distinct from geometric success.
RepeatedStorageResult recognizeRepeatedStorage(const RegionalAnalysis& body, scf::ForOp loop,
    RegionExpressions::Id trips);
// Uses the exact immutable body captured by the certificate. Only inter-visit
// storage witnesses are projected; queries and internal recipes stay complete.
RepeatedRegionAnalysis repeatEvolvingRegion(func::FuncOp function, std::shared_ptr<RepeatedStorage> storage);
} // namespace mlir::pto::frontiersynch
#endif
