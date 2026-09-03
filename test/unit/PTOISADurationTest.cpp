// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/KernelScheduling/PTOISADuration.h"

#include <iostream>
#include <string_view>

namespace {

bool check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "PTOISADurationTest failure: " << message << '\n';
  }
  return condition;
}

bool testExactFormulaLookup() {
  constexpr std::string_view csv = R"(op,dtype,cols,slope,bias
TADDS,fp32,32,0.0156,25
TROWSUM,fp16,64,0.0078,42
TEXP,any,*,0.0314,30.1
)";
  auto table = mlir::pto::PTOISADurationTable::parseCSV(csv);
  if (!check(succeeded(table), "formula CSV parses")) {
    return false;
  }
  auto adds = table->estimate({"tadds", "fp32", 10, 32});
  auto sum = table->estimate({"TROWSUM", "fp16", 5, 64});
  auto exp = table->estimate({"TEXP", "bf16", 4, 128});
  return check(adds.has_value(), "case-insensitive exact key") &&
         check(adds->cycles == 30, "PTO-ISA llround formula semantics") &&
         check(sum.has_value() && sum->cycles == 44, "second exact formula") &&
         check(exp.has_value() && exp->cycles == 46,
               "PTO-ISA explicit any/star formula is accepted") &&
         check(!table->estimate({"TADDS", "fp32", 10, 33}).has_value(),
               "unsupported column count fails closed") &&
         check(!table->estimate({"TADDS", "bf16", 10, 32}).has_value(),
               "unsupported dtype fails closed") &&
         check(!table->estimate({"TEXP", "fp32", 0, 32}).has_value(),
               "dynamic/invalid rows fail closed");
}

bool testRejectMalformedOrAmbiguousTables() {
  constexpr std::string_view duplicate = R"(op,dtype,cols,slope,bias
TADDS,fp32,32,0.0156,25
TADDS,fp32,32,0.0200,24
)";
  constexpr std::string_view duplicateWildcard = R"(op,dtype,cols,slope,bias
TEXP,any,*,0.0314,30.1
TEXP,any,*,0.0314,30.1
)";
  constexpr std::string_view badHeader = R"(opcode,dtype,cols,slope,bias
TADDS,fp32,32,0.0156,25
)";
  return check(failed(mlir::pto::PTOISADurationTable::parseCSV(duplicate)),
               "ambiguous exact formula is rejected") &&
         check(failed(mlir::pto::PTOISADurationTable::parseCSV(duplicateWildcard)),
               "ambiguous wildcard formula is rejected") &&
         check(failed(mlir::pto::PTOISADurationTable::parseCSV(badHeader)),
               "wrong formula schema is rejected");
}

} // namespace

int main() {
  return testExactFormulaLookup() && testRejectMalformedOrAmbiguousTables()
             ? 0
             : 1;
}
