// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// RUN: pto-sync-input-test --region-expression-checks %s | FileCheck %s
// RUN: pto-sync-input-test --numerical-hierarchy-checks %s
// RUN: pto-sync-input-test --numerical-repeated-squaring-checks
// RUN: pto-sync-input-test --bounding-contract-checks %s
// RUN: pto-sync-input-test --certified-partial-reduction-checks
// RUN: pto-sync-input-test --control-origin-distance-checks
// RUN: pto-sync-input-test --compact-writer-reader-checks
// RUN: pto-sync-input-test --compact-lower-facts-checks
// RUN: pto-sync-input-test --boundary-excess-checks
// RUN: pto-sync-input-test --repeated-excess-checks
// RUN: pto-sync-input-test --periodic-excess-checks
// CHECK: regional expression checks passed

module {
  func.func @expressions(%index: index, %condition: i1) {
    %late = arith.addi %index, %index {test.cut} : index
    scf.if %condition {
      %hidden = arith.addi %late, %index {test.hidden} : index
    }
    %end = arith.addi %late, %index {test.cut} : index
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %three = arith.constant 3 : index
    %four = arith.constant 4 : index
    scf.for %i = %zero to %three step %one {
      scf.for %j = %zero to %four step %one {
        %nested = arith.addi %i, %j {test.nested} : index
      }
    }
    return
  }
}
