// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/KernelScheduling/PTOISADuration.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cmath>
#include <algorithm>

using namespace mlir;

namespace mlir {
namespace pto {
namespace {

static std::string lower(llvm::StringRef value) {
  return value.trim().lower();
}

static std::string makeKey(llvm::StringRef opcode, llvm::StringRef dtype) {
  return (lower(opcode) + "|" + lower(dtype));
}

static std::optional<int64_t> parseInt(llvm::StringRef value) {
  int64_t result = 0;
  if (value.trim().getAsInteger(10, result)) {
    return std::nullopt;
  }
  return result;
}

static std::optional<double> parseDouble(llvm::StringRef value) {
  double result = 0.0;
  if (value.trim().getAsDouble(result)) {
    return std::nullopt;
  }
  return result;
}

static std::optional<std::string> getFormulaOpcode(Operation *operation) {
  const llvm::StringRef name = operation->getName().getStringRef();
  return llvm::StringSwitch<std::optional<std::string>>(name)
      .Case("pto.tadds", "TADDS")
      .Case("pto.tsub", "TSUB")
      .Case("pto.tsubs", "TSUBS")
      .Case("pto.tmuls", "TMULS")
      .Case("pto.tdivs", "TDIVS")
      .Case("pto.tmins", "TMINS")
      .Case("pto.tmul", "TMUL")
      .Case("pto.trowsum", "TROWSUM")
      .Case("pto.trowmax", "TROWMAX")
      .Case("pto.tcolsum", "TCOLSUM")
      .Case("pto.tcolmax", "TCOLMAX")
      .Case("pto.trowexpand", "TROWEXPAND")
      .Case("pto.texp", "TEXP")
      .Case("pto.tsqrt", "TSQRT")
      .Default(std::nullopt);
}

static std::optional<std::string> getFormulaDType(Type type) {
  if (type.isF32()) {
    return "fp32";
  }
  if (type.isF16()) {
    return "fp16";
  }
  if (type.isBF16()) {
    return "bf16";
  }
  if (auto integer = dyn_cast<IntegerType>(type)) {
    const unsigned width = integer.getWidth();
    if (width == 32) {
      return integer.isUnsigned() ? "u32" : "i32";
    }
    if (width == 16) {
      return integer.isUnsigned() ? "u16" : "i16";
    }
    if (width == 8) {
      return integer.isUnsigned() ? "u8" : "i8";
    }
  }
  return std::nullopt;
}

} // namespace

FailureOr<PTOISADurationTable>
PTOISADurationTable::loadFromFile(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    return failure();
  }
  return parseCSV(buffer.get()->getBuffer());
}

FailureOr<PTOISADurationTable>
PTOISADurationTable::parseCSV(llvm::StringRef csv) {
  PTOISADurationTable table;
  llvm::SmallVector<llvm::StringRef, 8> columns;
  llvm::SmallVector<llvm::StringRef, 128> lines;
  csv.split(lines, '\n');
  bool firstLine = true;
  for (llvm::StringRef line : lines) {
    line = line.trim();
    if (line.empty() || line.starts_with("#")) {
      continue;
    }
    if (firstLine) {
      firstLine = false;
      if (line != "op,dtype,cols,slope,bias") {
        return failure();
      }
      continue;
    }
    columns.clear();
    line.split(columns, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
    if (columns.size() != 5) {
      return failure();
    }
    std::optional<int64_t> cols = parseInt(columns[2]);
    std::optional<double> slope = parseDouble(columns[3]);
    std::optional<double> bias = parseDouble(columns[4]);
    if (!cols || *cols <= 0 || !slope || !bias) {
      return failure();
    }
    table.rows_[makeKey(columns[0], columns[1])].push_back(
        {*cols, *slope, *bias});
  }
  if (firstLine || table.rows_.empty()) {
    return failure();
  }
  for (auto &entry : table.rows_) {
    llvm::sort(entry.getValue(), [](const FormulaRow &lhs,
                                    const FormulaRow &rhs) {
      return lhs.cols < rhs.cols;
    });
    if (std::adjacent_find(entry.getValue().begin(), entry.getValue().end(),
                           [](const FormulaRow &lhs, const FormulaRow &rhs) {
                             return lhs.cols == rhs.cols;
                           }) != entry.getValue().end()) {
      return failure();
    }
  }
  return table;
}

std::optional<PTOISADurationEstimate>
PTOISADurationTable::estimate(const PTOISADurationSignature &signature) const {
  if (signature.rows <= 0 || signature.cols <= 0) {
    return std::nullopt;
  }
  const auto entry = rows_.find(makeKey(signature.opcode, signature.dtype));
  if (entry == rows_.end()) {
    return std::nullopt;
  }
  const auto row = llvm::find_if(entry->getValue(), [&](const FormulaRow &row) {
    return row.cols == signature.cols;
  });
  if (row == entry->getValue().end()) {
    return std::nullopt;
  }
  const double cycles = row->slope * static_cast<double>(signature.rows) *
                            static_cast<double>(signature.cols) +
                        row->bias;
  if (cycles <= 0.0 || !std::isfinite(cycles)) {
    return std::nullopt;
  }
  // This is PTO-ISA's formula_backend_compute.hpp::RoundToCycles semantics.
  return PTOISADurationEstimate{static_cast<uint64_t>(std::llround(cycles)),
                                signature};
}

std::optional<PTOISADurationSignature>
getPTOISADurationSignature(Operation *operation) {
  if (!operation) {
    return std::nullopt;
  }
  std::optional<std::string> opcode = getFormulaOpcode(operation);
  if (!opcode) {
    return std::nullopt;
  }
  for (Value operand : operation->getOperands()) {
    auto tile = dyn_cast<TileBufType>(operand.getType());
    if (!tile) {
      continue;
    }
    auto shape = tile.getShape();
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
      return std::nullopt;
    }
    std::optional<std::string> dtype = getFormulaDType(tile.getElementType());
    if (!dtype) {
      return std::nullopt;
    }
    return PTOISADurationSignature{*opcode, *dtype, shape[0], shape[1]};
  }
  return std::nullopt;
}

} // namespace pto
} // namespace mlir
