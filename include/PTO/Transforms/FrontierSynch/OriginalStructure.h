// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License. Original physical/control records extracted from OAHS Plan.h. The translated phases and
// source IR are borrowed from SyncInput and must outlive this result.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALSTRUCTURE_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_ORIGINALSTRUCTURE_H
#include "PTO/Transforms/FrontierSynch/SyncTileDescriptorState.h"
#include "PTO/Transforms/InsertSync/SyncInput.h"
#include <limits>
#include <string>
#include <vector>

namespace mlir::pto::frontiersynch {
inline constexpr std::size_t NoControlId = std::numeric_limits<std::size_t>::max();
struct Access {
  std::size_t cell = 0;
  bool read = false, write = false;
  // Full-cell overwrite requires a separate effect-coverage proof.
  bool definiteWrite = false;
  // Original translated effect identity; the cell alone does not identify
  // which independent physical selector produced this use.
  const BaseMemInfo *memory = nullptr;
  // The specific qualified address relation for this translated effect.
  std::size_t physicalRelation = NoControlId;
};
struct Cell {
  bool exclusive = false;
  std::string addressSpace;
  std::string provenance;
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  bool unknownRange = false;
  enum class Storage { Abstract, CanonicalInterval, OverlapWitness };
  Storage storage = Storage::Abstract;
  std::string coordinateSpace;
  std::vector<std::size_t> storageOrigins;
};
struct PhysicalOperation {
  const CompoundInstanceElement *instruction = nullptr;
  std::size_t original = NoControlId;
  std::vector<Access> accesses;
};
struct Region {
  enum Kind { Sequence, Choice, For, While, Operation } kind = Sequence;
  std::vector<Region> children;
  std::size_t operation = 0;
  bool zeroTripPossible = false;
  std::size_t originalOwner = NoControlId;
  bool qualifiedCounted = false;
};
// An address relation records physical alternatives only. Neither its finite
// period nor its root identity establishes a predecessor-use correspondence.
struct PhysicalAddressRelation {
  std::size_t owner = NoControlId;
  Value address;
  const BaseMemInfo *memory = nullptr;
  std::vector<SmallVector<uint64_t>> addresses;
};
struct OriginalStructure {
  func::FuncOp function;
  Region body;
  std::vector<PhysicalOperation> operations;
  std::vector<Cell> cells;
  std::vector<mlir::Operation *> originalSites;
  std::vector<Value> storageRoots;
  std::vector<PhysicalAddressRelation> physicalAddresses;
  std::unique_ptr<SyncTileDescriptorState> descriptors;
};
// Requires verified original IR. Failure leaves the caller's result unchanged.
LogicalResult importOriginalStructure(func::FuncOp function, const SyncInput &input,
                                      OriginalStructure &result);
} // namespace mlir::pto::frontiersynch
#endif
