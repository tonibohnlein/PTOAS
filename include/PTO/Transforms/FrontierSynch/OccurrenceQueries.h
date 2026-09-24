// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and
// conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
// the License for details. You may not use this file except in compliance with the License. THIS
// SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
// PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
// License.
#ifndef PTO_TRANSFORMS_FRONTIERSYNCH_OCCURRENCEQUERIES_H
#define PTO_TRANSFORMS_FRONTIERSYNCH_OCCURRENCEQUERIES_H
#include "PTO/Transforms/FrontierSynch/OriginalStructure.h"
#include <string>
namespace mlir::pto::frontiersynch {
// These are original-use facts. Unknown retains the original physical effects.
struct PhysicalBankCorrespondence {
  bool exactPermutation = false;
  std::size_t owner = NoControlId;
  const BaseMemInfo *memory = nullptr;
  std::size_t distance = 0;
  std::vector<std::size_t> participatingOperations;
  std::string reason;
};
struct FixedVisitCorrespondence {
  bool exact = false;
  std::size_t source = NoControlId, target = NoControlId;
  std::string reason;
};
struct ChildOccurrence {
  std::size_t owner = NoControlId, child = NoControlId, position = 0;
  std::vector<std::size_t> operations;
  bool canSkip = false, canRepeat = false;
};
class OccurrenceQueries {
public:
  explicit OccurrenceQueries(const OriginalStructure &original);
  ~OccurrenceQueries();
  PhysicalBankCorrespondence bank(std::size_t relationId) const;
  FixedVisitCorrespondence fixedVisit(std::size_t source, std::size_t target,
                                      std::size_t cell) const;
  std::vector<ChildOccurrence> children(std::size_t owner) const;

private:
  const OriginalStructure &original;
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlir::pto::frontiersynch
#endif
