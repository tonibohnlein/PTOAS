# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate wide accesses over a finely partitioned physical tile."""


def main():
    """Emit fixed scaling families for the independent compiled byte oracle."""
    for size in (8, 32, 128, 512):
        print(f"!tile{size} = !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols={size}, "
              f"v_row=1, v_col={size}, blayout=row_major, slayout=none_box, fractal=512, pad=0>")
    print("module {")
    for size in (8, 32, 128, 512):
        tile = f"!tile{size}"
        print(f"func.func @split{size}() attributes {{test.membership_expect = "
              f"array<i64: {size}, {2 * size}, {2 * size}, {size * (size + 1)}>}} {{")
        print("%zero = arith.constant 0 : i64")
        print("%one = arith.constant 1.0 : f32")
        print(f"%tile = pto.alloc_tile addr = %zero : {tile}")
        for index in range(size):
            print(f"%i{index} = arith.constant {index} : index")
            print(f"pto.tsetval ins(%i{index}, %one : index, f32) outs(%tile : {tile})")
        for _ in range(size):
            print(f"pto.texpands ins(%one : f32) outs(%tile : {tile})")
        print("return\n}")
    print("}")


if __name__ == "__main__":
    main()
