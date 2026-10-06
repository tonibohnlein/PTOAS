#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Lossless cast's generic 64x64 scale-factor configuration.

This is a supported source factory configuration, NOT the default max4 fast
path. Keep that distinction in benchmark reporting and device comparisons.
"""

from port_ir import Port
from port_conv import cast


def linear(p, i, stride, j):
    return p.binary("addi", p.binary("muli", i, p.const(stride)), j)


def sf_reduce(p, exponents, out_exp):
    with p.loop(p.const(0), p.const(64)) as i:
        with p.loop(p.const(0), p.const(4)) as j:
            target = p.binary("divui", j, p.const(2))
            old = p.get(out_exp, target, "i32")
            value = p.get(exponents, linear(p, i, 8, j), "i32")
            p.put(out_exp, target, p.binary("maxsi", old, value, "i32"))
    with p.loop(p.const(0), p.const(2)) as j:
        value = p.binary("subi", p.get(out_exp, j, "i32"), p.const(6, "i32"), "i32")
        p.put(out_exp, j, p.binary("maxsi", value, p.const(0, "i32"), "i32"))


def sf_store(p, out_exp, out_float, row, col):
    with p.loop(p.const(0), p.const(2)) as j:
        bits = p.binary("shli", p.get(out_exp, j, "i32"), p.const(23, "i32"), "i32")
        value = p.value(f"arith.bitcast {bits} : i32 to f32", "f32")
        p.put(out_float, j, value)
        value = p.get(out_float, j)
        offset = linear(p, row, 4, p.binary("addi", col, j))
        p.emit(f"pto.store {value}, %out_sf[{offset}] : !pto.ptr<f32, gm>, f32")


def relative_scales(p, exp, out_exp, relative):
    with p.loop(p.const(0), p.const(64)) as i:
        with p.loop(p.const(0), p.const(4)) as j:
            index = linear(p, i, 8, j)
            value = p.get(exp, index, "i32")
            target = p.binary("divui", j, p.const(2))
            value = p.binary("subi", value, p.get(out_exp, target, "i32"), "i32")
            p.put(exp, index, p.binary("addi", value, p.const(127, "i32"), "i32"))
    with p.loop(p.const(0), p.const(64)) as i:
        with p.loop(p.const(0), p.const(128)) as j:
            source = linear(p, i, 8, p.binary("divui", j, p.const(32)))
            bits = p.binary("shli", p.get(exp, source, "i32"), p.const(23, "i32"), "i32")
            value = p.value(f"arith.bitcast {bits} : i32 to f32", "f32")
            p.put(relative, linear(p, i, 128, j), value)


def lossless_block_cast():
    args = [("%x_sf", "!pto.ptr<f32, gm>"), ("%X", "!pto.ptr<bf16, gm>"),
            ("%Y", "!pto.ptr<f32, gm>"), ("%out_sf", "!pto.ptr<f32, gm>")]
    p = Port("lossless_block_cast", args)
    x, y = p.view("%X", 128, 256, "bf16"), p.view("%Y", 128, 256)
    sf = p.view("%x_sf", 128, 8)
    cid = p.index(p.value("pto.get_block_idx", "i64"))
    vid = p.index(p.value("pto.get_subblock_idx", "i64"))
    task_m = p.binary("divui", cid, p.const(2))
    task_n = p.binary("remui", cid, p.const(2))
    row = p.binary("muli", task_m, p.const(64))
    col = p.binary("muli", task_n, p.const(128))
    with p.when(p.compare("eq", vid, p.const(0))):
        loaded = p.tile(64, 8, valid=(64, 4))
        # A typed alias preserves reinterpretation without a numeric conversion.
        bits = p.tile(64, 8, "i32", p.const(0, "i64"), valid=(64, 4))
        exp = p.tile(64, 8, "i32", valid=(64, 4))
        inp, out, relative = p.tile(64, 128, "bf16"), p.tile(64, 128), p.tile(64, 128)
        out_exp = p.tile(1, 8, "i32", valid=(1, 2))
        out_float = p.tile(1, 8, valid=(1, 2))
        # The source initializes this output before loading/decode/reduction.
        with p.loop(p.const(0), p.const(2)) as j:
            p.put(out_exp, j, p.const(0, "i32"))
        p.load(loaded, sf, row, p.binary("muli", task_n, p.const(4)))
        p.barrier()
        p.op("tshrs", [bits, p.const(23, "i32")], [exp])
        p.barrier("V")
        # A3 lacks i32 TANDS. Apply an i16 TAND to both halves of each
        # word using the bit representation of an i32 mask [255, 0].
        mask = p.tile(64, 8, "i32", valid=(64, 4))
        masked = p.tile(64, 8, "i32", valid=(64, 4))
        aliases = [p.tile(64, 16, "i16", p.addresses[tile], valid=(64, 8))
                   for tile in (exp, mask, masked)]
        p.op("texpands", [p.const(255, "i32")], [mask])
        p.barrier("V")
        p.op("tand", aliases[:2], [aliases[2]])
        p.barrier("V")
        p.op("tmov", [masked], [exp])
        sf_reduce(p, exp, out_exp)
        sf_store(p, out_exp, out_float, task_m, p.binary("muli", task_n, p.const(2)))
        relative_scales(p, exp, out_exp, relative)
        p.load(inp, x, row, col)
        p.barrier()
        cast(p, out, inp)
        p.op("tmul", [out, relative], [out])
        p.barrier()
        p.store(out, y, row, col)
    p.barrier()
    return p, {"shape": [128, 256], "in_sf_block": [1, 32], "out_sf_block": [64, 64],
               "block_m": 64, "block_k": 128, "input_dtype": "bfloat16", "output_dtype": "float32",
               "sf_layout": "plain row major, fp32 powers of two", "launch_blocks": 4, "threads": 2,
               "factory_path": "generic vid==0", "input_sf_is_tile_packed": False,
               "scope": "Supported generic configuration; optimized default max4 path remains unported.",
               "requirements": ["Valid finite positive power-of-two scales; padded dimensions are exact.",
                                "Do not label this as the default max4 HPC configuration."]}
