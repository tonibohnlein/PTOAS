#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Audited vector-add and chunk-prefix benchmark specializations."""

from port_ir import Port


def coordinates(p):
    return (p.index(p.value("pto.get_block_idx", "i64")),
            p.index(p.value("pto.get_subblock_idx", "i64")))


def elementwise_pipeline():
    p = Port("elementwise_pipeline", [("%A", "!pto.ptr<f32, gm>"),
             ("%B", "!pto.ptr<f32, gm>"), ("%C", "!pto.ptr<f32, gm>")])
    a, b, c = [p.view(v, 1024, 1024) for v in ("%A", "%B", "%C")]
    cid, vid = coordinates(p)
    bx = p.binary("divui", cid, p.const(8))
    by = p.binary("remui", cid, p.const(8))
    row = p.binary("addi", p.binary("muli", bx, p.const(128)), p.binary("muli", vid, p.const(16)))
    col = p.binary("muli", by, p.const(128))
    for event in (0, 1):
        p.flag("set", "MTE3", "MTE2", p.const(event))
    p.flag("wait", "MTE3", "MTE2", p.const(0))
    for view, base in ((a, 0), (b, 16384)):
        p.load(p.bank(p.const(0), 16, 128, base), view, row, col)
    p.flag("set", "MTE2", "V", p.const(0))
    with p.loop(p.const(0), p.const(4)) as i:
        cur = p.binary("remui", i, p.const(2))
        next_i = p.binary("addi", i, p.const(1))
        nxt = p.binary("remui", next_i, p.const(2))
        with p.when(p.compare("ult", i, p.const(3))):
            p.flag("wait", "MTE3", "MTE2", nxt)
            next_row = p.binary("addi", row, p.binary("muli", next_i, p.const(32)))
            for view, base in ((a, 0), (b, 16384)):
                p.load(p.bank(nxt, 16, 128, base), view, next_row, col)
            p.flag("set", "MTE2", "V", nxt)
        p.flag("wait", "MTE2", "V", cur)
        aa, bb, cc = [p.bank(cur, 16, 128, base) for base in (0, 16384, 32768)]
        p.op("tadd", [aa, bb], [cc])
        p.flag("set", "V", "MTE3", cur)
        p.flag("wait", "V", "MTE3", cur)
        out_row = p.binary("addi", row, p.binary("muli", i, p.const(32)))
        p.store(cc, c, out_row, col)
        p.flag("set", "MTE3", "MTE2", cur)
    for event in (0, 1):
        p.flag("wait", "MTE3", "MTE2", p.const(event))
    p.barrier()
    return p, {"shape": [1024, 1024], "tile": [128, 128], "sub_m": 32,
               "dtype": "float32", "launch_blocks": 64, "vector_lanes": 2,
               "scope": "Whole source shape; fixed disjoint UB banks; terminal completion added."}


def gdn_chunk_cumsum():
    p = Port("gdn_chunk_cumsum", [("%G", "!pto.ptr<f32, gm>"), ("%S", "!pto.ptr<f32, gm>")])
    g, s = [p.view(v, 32, 16384) for v in ("%G", "%S")]
    cid, vid = coordinates(p)
    chunk = p.binary("remui", cid, p.const(16))
    head = p.binary("addi", p.binary("muli", p.binary("divui", cid, p.const(16)), p.const(2)), vid)
    col = p.binary("muli", chunk, p.const(1024))
    gb, sb = p.tile(1, 1024), p.tile(1, 1024)
    p.op("texpands", [p.const("0.0", "f32")], [sb])
    p.load(gb, g, head, col)
    p.handoff("MTE2", "V")
    with p.loop(p.const(0), p.const(8)) as ii:
        offset = p.binary("muli", ii, p.const(128))
        p.handoff("V", "S")
        p.put(sb, offset, p.get(gb, offset))
        with p.loop(p.const(1), p.const(128)) as i:
            pos = p.binary("addi", offset, i)
            prev = p.binary("subi", pos, p.const(1))
            total = p.binary("addf", p.get(sb, prev), p.get(gb, pos), "f32")
            p.put(sb, pos, total)
    p.handoff("V", "MTE3")
    p.store(sb, s, head, col)
    p.barrier()
    return p, {"shape": [2, 16, 16384], "chunk": 128, "chunks_per_task": 8,
               "dtype": "float32", "launch_blocks": 256, "vector_lanes": 2,
               "scope": ("Full source test configuration; retains eight scalar chunks, "
                         "one final full-buffer store and original flags.")}


def sigmoid(p, dst, src):
    """Pinned PTO TSIGMOID helper mutates src and has three A3 V barriers."""
    dtype = p.shapes[src][2]
    p.op("tmuls", [src, p.const("-1.0", dtype)], [src])
    p.barrier("V")
    p.op("texp", [src], [src])
    p.barrier("V")
    p.op("tadds", [src, p.const("1.0", dtype)], [src])
    p.barrier("V")
    p.op("trecip", [src], [dst])


def silu(p, dst, src, tmp):
    p.op("tmov", [src], [tmp])
    p.barrier("V")
    sigmoid(p, dst, src)
    p.barrier("V")
    p.op("tmul", [tmp, dst], [dst])


def mul_add(p, dst, left, right, tmp):
    p.op("tmul", [left, right], [tmp])
    p.barrier("V")
    p.op("tadd", [dst, tmp], [dst])


def mhc_head_mix():
    arguments = [(name, "!pto.ptr<f32, gm>") for name in ("%X", "%scale", "%base", "%Y")]
    p = Port("mhc_head_mix", arguments)
    x, y = [p.view(v, 2048, 16) for v in ("%X", "%Y")]
    scale, base = p.view("%scale", 1, 1), p.view("%base", 1, 4)
    cid, vid = coordinates(p)
    row = p.binary("addi", p.binary("muli", cid, p.const(128)), p.binary("muli", vid, p.const(64)))
    inp, out, bc = [p.tile(64, 16) for _ in range(3)]
    scale_tile = p.tile(1, 8, valid=(1, 1))
    base_addr = p.next_address
    base_full = p.tile(1, 16)
    base_load = p.tile(1, 16, address=p.const(base_addr, "i64"), valid=(1, 4))
    p.handoff("MTE3", "MTE2")
    p.load(inp, x, row, p.const(0))
    p.load(scale_tile, scale, p.const(0), p.const(0))
    p.load(base_load, base, p.const(0), p.const(0))
    p.handoff("MTE2", "V")
    with p.loop(p.const(1), p.const(4)) as r:
        with p.loop(p.const(0), p.const(4)) as j:
            pos = p.binary("addi", p.binary("muli", r, p.const(4)), j)
            p.put(base_full, pos, p.get(base_full, j))
    p.op("tcolexpand", [base_full], [bc])
    p.op("taxpy", [inp, p.get(scale_tile, p.const(0))], [bc])
    sigmoid(p, out, bc)
    p.op("tadds", [out, p.const("0.01", "f32")], [out])
    p.handoff("V", "MTE3")
    p.store(out, y, row, p.const(0))
    p.handoff("MTE3", "MTE2")
    p.barrier()
    return p, {"direction": "forward", "num_original_tokens": 8192, "mhc_mult": 4,
               "reshape_factor": 4, "view_shape": [2048, 16], "launch_blocks": 16,
               "vector_lanes": 2, "dtype": "float32", "epsilon": 0.01,
               "scope": "Source test_fwd configuration; PTO sigmoid expansion and its A3 V barriers included."}
