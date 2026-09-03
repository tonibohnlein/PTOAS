// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOISADuration.h - Exact PTO-ISA formula durations ------*- C++ -*-===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_KERNELSCHEDULING_PTOISADURATION_H
#define MLIR_DIALECT_PTO_TRANSFORMS_KERNELSCHEDULING_PTOISADURATION_H

#include "PTO/IR/PTO.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace pto {

/// A fully specified formula-model query. The A2/A3 PTO-ISA formula table is
/// linear in rows for an exact (opcode, dtype, cols) entry.
struct PTOISADurationSignature {
  std::string opcode;
  std::string dtype;
  int64_t rows = 0;
  int64_t cols = 0;
};

struct PTOISADurationEstimate {
  uint64_t cycles = 0;
  PTOISADurationSignature signature;
};

/// Loads the pinned public PTO-ISA formula CSV. Missing signatures are
/// unavailable: this class intentionally has no operation-family fallback.
class PTOISADurationTable {
public:
  static FailureOr<PTOISADurationTable> loadFromFile(llvm::StringRef path);
  static FailureOr<PTOISADurationTable> parseCSV(llvm::StringRef csv);

  std::optional<PTOISADurationEstimate>
  estimate(const PTOISADurationSignature &signature) const;

private:
  struct FormulaRow {
    int64_t cols = 0;
    double slope = 0.0;
    double bias = 0.0;
  };
  /// Key: lower-case `opcode|dtype`.
  llvm::StringMap<std::vector<FormulaRow>> rows_;
};

/// Derive a formula signature only for operations whose formula work is
/// defined by their first static tile operand. Operations with extra semantic
/// parameters, dynamic shapes, or transfer-specific models return nullopt.
std::optional<PTOISADurationSignature>
getPTOISADurationSignature(Operation *operation);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_KERNELSCHEDULING_PTOISADURATION_H
