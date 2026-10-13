# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Opaque-call/depth family with an independent dependency-visit bound."""
import itertools

print("!tile = !pto.tile_buf<vec, 8x8xf32, valid=?x?>")
print("module {")
print("  func.func private @opaque()")
for descriptors, calls, depth in itertools.product((4, 8), (1, 16, 64), (1, 3, 6)):
    name = f"dependencies_{descriptors}_{calls}_{depth}"
    print(f"  func.func @{name}(%count: index) attributes {{"
          f"test.max_universal_dependency_probes = {descriptors * depth} : i64}} {{")
    print("    %zero = arith.constant 0 : index")
    print("    %one = arith.constant 1 : index")
    print("    %eight = arith.constant 8 : index")
    print("    %addr = arith.constant 0 : i64")
    for descriptor in range(descriptors):
        print(f"    %t{descriptor} = pto.alloc_tile addr = %addr "
              "valid_row = %eight valid_col = %eight : !tile")
    for level in range(depth):
        print(f"    scf.for %i{level} = %zero to %count step %one {{")
    for _ in range(calls):
        print("      func.call @opaque() : () -> ()")
    for descriptor in range(descriptors):
        print(f'      %r{descriptor} = "pto.tgetval"(%t{descriptor}, %zero) '
              '{test.shape = array<i64: -1, -1>} : (!tile, index) -> f32')
    for _ in range(depth):
        print("    }")
    print('    %exit = "pto.tgetval"(%t0, %zero) '
          '{test.shape = array<i64: -1, -1>} : (!tile, index) -> f32')
    print("    return\n  }")
print("}")
