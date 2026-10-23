# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Small supplied-layout references, independent of PTOAS cells and generators.

Payload inputs are initialized on entry. These are host models of the named
layout/operation contracts, not independent native instruction qualification.
"""

from compact_persistent_oracle import merge
from check_periodic_demands import native, closure


def tile(space, rows, cols, dtype="f32", layout="row_major", inner="none_box", fractal=512):
    return (f"!pto.tile_buf<{space}, {rows}x{cols}x{dtype}, blayout={layout}, "
            f"slayout={inner}, fractal={fractal}>")


def boxed(rows, cols, width, row, col, height, length, *, outer_row=False, inner_row=True, fractal=512):
    """Enumerate logical elements through the supplied fractal layout."""
    ir, ic = (16, 16) if fractal == 1024 else ((16, 32 // width) if inner_row else (32 // width, 16))
    ranges = []
    for r in range(row, row + height):
        for c in range(col, col + length):
            outer = (r // ir) * (cols // ic) + c // ic if outer_row else (c // ic) * (rows // ir) + r // ir
            inner = (r % ir) * ic + c % ic if inner_row else (c % ic) * ir + r % ir
            start = width * (outer * ir * ic + inner)
            ranges.append((start, start + width))
    return merge(ranges)


def access(space, mode, spans, base=-1):
    return dict(space=space, base=base, mode=mode, intervals=spans)


def item(phase, pipe, effects, coordinates=(), active=True):
    return dict(phase=phase, pipe=pipe, effects=effects, coordinates=list(coordinates), active=active)


def module(name, arguments, types, body, core="cube"):
    return ("\n".join(f"!{key} = {value}" for key, value in types.items()) +
            '\nmodule attributes {pto.target_arch = "a3"} {\n' +
            f"  func.func @{name}({arguments}) attributes {{pto.kernel_kind = #pto.kernel_kind<{core}>}} {{\n" +
            body + "\n    return\n  }\n}\n")


def l1_case(count=1, later=64, second_window=False):
    name = f"l1_{count}_{later}" + ("_two_windows" if second_window else "")
    types = dict(pool=tile("mat", 128, 128, "i8", "col_major", "row_major"),
                 strip=tile("mat", 32, 128, "i8", "col_major", "row_major"),
                 transposed=tile("mat", 128, 128, "i8", "row_major", "col_major"),
                 right=tile("right", 128, 64, "i8", "row_major", "col_major"))
    extra = ("      pto.textract ins(%transposed, %c0, %c64 : !transposed, index, index) "
             "outs(%right2 : !right)") if second_window else ""
    body = f"""    %zero = arith.constant 0 : i64
    %other = arith.constant 8192 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %later = arith.constant {later} : index
    %count = arith.constant {count} : index
    %view = pto.make_tensor_view %src, shape = [%c32, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xi8>
    %input = pto.partition_view %view, offsets = [%c0, %c0], sizes = [%c32, %c128]
      : !pto.tensor_view<?x?xi8> -> !pto.partition_tensor_view<32x128xi8>
    %pool = pto.alloc_tile addr = %zero : !pool
    %transposed = pto.alloc_tile addr = %zero : !transposed
    %right = pto.alloc_tile addr = %zero : !right
    %right2 = pto.alloc_tile addr = %other : !right
    %first = pto.subview %pool[%c0, %c0] sizes [32, 128] : !pool -> !strip
    %later_view = pto.subview %pool[%later, %c0] sizes [32, 128] : !pool -> !strip
    scf.for %visit = %c0 to %count step %c1 {{
      pto.tload ins(%input : !pto.partition_tensor_view<32x128xi8>) outs(%first : !strip)
      pto.textract ins(%transposed, %c0, %c0 : !transposed, index, index) outs(%right : !right)
      pto.tload ins(%input : !pto.partition_tensor_view<32x128xi8>) outs(%later_view : !strip)
{extra}
    }}"""
    gm = access(1, "read", [(0, 4096)], 0)
    first = access(2, "write", boxed(128, 128, 1, 0, 0, 32, 128))
    read = access(2, "read", boxed(128, 128, 1, 0, 0, 128, 64, outer_row=True, inner_row=False))
    last = access(2, "write", boxed(128, 128, 1, later, 0, 32, 128))
    word = []
    for visit in range(count):
        word.extend((item(0, 4, [gm, first], (visit,)),
                     item(1, 3, [read, access(4, "write", [(0, 8192)])], (visit,)),
                     item(2, 4, [gm, last], (visit,))))
        if second_window:
            read2 = boxed(128, 128, 1, 0, 64, 128, 64, outer_row=True, inner_row=False)
            word.append(item(3, 3, [access(2, "read", read2), access(4, "write", [(8192, 16384)])], (visit,)))
    return name, module(name, "%src: !pto.ptr<i8, gm>", types, body), word


def acc_case(overlap=False):
    offset = 0 if overlap else 16
    name = "acc_overlap" if overlap else "acc_disjoint"
    types = dict(pool=tile("acc", 16, 64, layout="col_major", inner="row_major", fractal=1024),
                 acc=tile("acc", 16, 16, layout="col_major", inner="row_major", fractal=1024),
                 left=tile("left", 16, 16, "f16", "row_major", "row_major"),
                 right=tile("right", 16, 16, "f16", "row_major", "col_major"))
    body = f"""    %zero = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %column = arith.constant {offset} : index
    %pool = pto.alloc_tile addr = %zero : !pool
    %left = pto.alloc_tile addr = %zero : !left
    %right = pto.alloc_tile addr = %zero : !right
    %first = pto.subview %pool[%c0, %c0] sizes [16, 16] : !pool -> !acc
    %second = pto.subview %pool[%c0, %column] sizes [16, 16] : !pool -> !acc
    pto.tmatmul ins(%left, %right : !left, !right) outs(%first : !acc)
    pto.tstore ins(%first : !acc) outs(%out0 : !pto.partition_tensor_view<16x16xf32>)
    pto.tmatmul ins(%left, %right : !left, !right) outs(%second : !acc)
    pto.tstore ins(%second : !acc) outs(%out1 : !pto.partition_tensor_view<16x16xf32>)
    pto.tmatmul ins(%left, %right : !left, !right) outs(%first : !acc)"""
    reads = [access(3, "read", [(0, 512)]), access(4, "read", [(0, 512)])]
    first = boxed(16, 64, 4, 0, 0, 16, 16, fractal=1024)
    second = boxed(16, 64, 4, 0, offset, 16, 16, fractal=1024)
    word = [item(0, 2, [*reads, access(5, "write", first)]),
            item(1, 10, [access(5, "read", first), access(1, "write", [(0, 1024)], 0)]),
            item(2, 2, [*reads, access(5, "write", second)]),
            item(3, 10, [access(5, "read", second), access(1, "write", [(0, 1024)], 1)]),
            item(4, 2, [*reads, access(5, "write", first)])]
    args = "%out0: !pto.partition_tensor_view<16x16xf32>, %out1: !pto.partition_tensor_view<16x16xf32>"
    return name, module(name, args, types, body), word


def row_case(count=3, resets=1, same=False, step=1, base=0, rows=8):
    name = f"rows_{count}_{resets}_{int(same)}_{step}_{base}_{rows}"
    seed_base = base + (rows + 8) * 512
    types = dict(parent=tile("vec", rows, 128, "f32"), row=tile("vec", 1, 128, "f32"))
    coordinate = "%c0" if same else "%row"
    body = f"""    %zero = arith.constant {base} : i64
    %other = arith.constant {seed_base} : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant {rows} : index
    %c128 = arith.constant 128 : index
    %count = arith.constant {count} : index
    %step = arith.constant {step} : index
    %resets = arith.constant {resets} : index
    %value = arith.constant 0.0 : f32
    %source_view = pto.make_tensor_view %src, shape = [%c1, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xf32>
    %source = pto.partition_view %source_view, offsets = [%c0, %c0], sizes = [%c1, %c128]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x128xf32>
    %output_view = pto.make_tensor_view %out, shape = [%c8, %c128], strides = [%c128, %c1] : !pto.tensor_view<?x?xf32>
    %output = pto.partition_view %output_view, offsets = [%c0, %c0], sizes = [%c8, %c128]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<{rows}x128xf32>
    %parent = pto.alloc_tile addr = %zero : !parent
    %seed = pto.alloc_tile addr = %other : !row
    pto.texpands ins(%value : f32) outs(%parent : !parent)
    pto.texpands ins(%value : f32) outs(%seed : !row)
    scf.for %reset = %c0 to %resets step %c1 {{
      scf.for %row = %c0 to %count step %step {{
        %target = pto.subview %parent[{coordinate}, %c0] sizes [1, 128] : !parent -> !row
        %parity = arith.remui %row, %c2 : index
        %take = arith.cmpi eq, %parity, %c0 : index
        scf.if %take {{
          pto.tload ins(%source : !pto.partition_tensor_view<1x128xf32>) outs(%target : !row)
        }} else {{
          pto.tmov ins(%seed : !row) outs(%target : !row)
        }}
      }}
    }}
    pto.tstore ins(%parent : !parent) outs(%output : !pto.partition_tensor_view<{rows}x128xf32>)"""
    seed = access(6, "read", [(seed_base, seed_base + 512)])
    word = [item(0, 1, [access(6, "write", [(base, base + rows * 512)])]),
            item(1, 1, [access(6, "write", [(seed_base, seed_base + 512)])])]
    for reset in range(resets):
        for ordinal, row in enumerate(range(0, count, step)):
            address = base + (0 if same else row) * 512
            write = access(6, "write", [(address, address + 512)])
            word.extend((item(2, 4, [access(1, "read", [(0, 512)], 0), write],
                              (reset, ordinal), row % 2 == 0),
                         item(3, 1, [seed, write], (reset, ordinal), row % 2 != 0)))
    word.append(item(4, 5, [access(6, "read", [(base, base + rows * 512)]),
                            access(1, "write", [(0, rows * 512)], 1)]))
    args = "%src: !pto.ptr<f32, gm>, %out: !pto.ptr<f32, gm>"
    return name, module(name, args, types, body, "vector"), word


def required(word):
    """Declarative native graph plus all concrete conflicting access pairs."""
    active = [i for i, event in enumerate(word) if event["active"]]
    edges = native([word[i]["pipe"] for i in active])
    for a, i in enumerate(active):
        for b in range(a + 1, len(active)):
            if any((x["mode"] == "write" or y["mode"] == "write") and
                   (x["space"], x["base"]) == (y["space"], y["base"]) and
                   any(lo < hi2 and lo2 < hi for lo, hi in x["intervals"] for lo2, hi2 in y["intervals"])
                   for x in word[i]["effects"] for y in word[active[b]]["effects"]):
                edges.add((2*a + 1, 2*b))
    reach, covers = closure(2 * len(active), edges)
    rows = [0] * (2 * len(word))
    for a, i in enumerate(active):
        for b, j in enumerate(active):
            for k in (0, 1):
                for h in (0, 1):
                    rows[2*i+k] |= ((reach[2*a+k] >> (2*b+h)) & 1) << (2*j+h)
    return rows, {(active[a//2], active[b//2]) for a, b in covers}
