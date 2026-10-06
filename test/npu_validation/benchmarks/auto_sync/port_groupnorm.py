#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Serial GroupNorm factory specialization; both source phases are retained."""

from port_ir import Port


def column(p, count):
    return p.tile(count, 1, layout="col_major")


def row_alias(p, column_tile, count):
    # Row reductions write a compact column. The following reduction consumes
    # the same physical values as a row, matching the source's 1-D result.
    return p.tile(1, count, address=p.addresses[column_tile])


def group_norm_phase1(p, x, row, total, total_sq):
    accum, squares, data, scratch = [p.tile(16, 64) for _ in range(4)]
    sums, sum_sq = column(p, 16), column(p, 16)
    for tile in (accum, squares):
        p.op("texpands", [p.const("0.0", "f32")], [tile])
    p.load(p.bank(p.const(0), 16, 64, 0), x, row, p.const(0))
    p.barrier()
    with p.loop(p.const(0), p.const(4)) as i:
        cur = p.binary("remui", i, p.const(2))
        ni = p.binary("addi", i, p.const(1))
        with p.when(p.compare("slt", ni, p.const(4))):
            nxt = p.binary("remui", ni, p.const(2))
            p.load(p.bank(nxt, 16, 64, 0), x, row, p.binary("muli", ni, p.const(64)))
        p.op("tmov", [p.bank(cur, 16, 64, 0)], [data])
        p.op("tadd", [accum, data], [accum])
        p.op("tmul", [data, data], [data])
        p.op("tadd", [squares, data], [squares])
        p.barrier()
    p.op("trowsum", [accum, scratch], [sums])
    p.op("trowsum", [squares, scratch], [sum_sq])
    small_scratch = p.tile(1, 16)
    p.op("trowsum", [row_alias(p, sums, 16), small_scratch], [total])
    p.op("trowsum", [row_alias(p, sum_sq, 16), small_scratch], [total_sq])


def group_norm_affine(p, gamma, beta, group):
    broadcasts = []
    for view in (gamma, beta):
        raw, cal = column(p, 16), column(p, 16)
        full, bc = p.tile(16, 64), p.tile(16, 64)
        broadcasts.append((raw, cal, full, bc))
        p.load(raw, view, p.binary("muli", group, p.const(16)), p.const(0))
    p.barrier()
    for raw, cal, full, bc in broadcasts:
        p.op("tmov", [raw], [cal])
        p.op("trowexpand", [cal], [full])
        p.op("tmov", [full], [bc])
    p.barrier()
    return [entry[-1] for entry in broadcasts]


def group_norm_phase2(p, x, y, row, mean, std, gamma, beta):
    data = p.tile(16, 64)
    for event in (0, 1):
        p.flag("set", "MTE3", "MTE2", p.const(event))
    p.flag("wait", "MTE3", "MTE2", p.const(0))
    p.load(p.bank(p.const(0), 16, 64, 8192), x, row, p.const(0))
    p.flag("set", "MTE2", "V", p.const(0))
    with p.loop(p.const(0), p.const(4)) as i:
        cur = p.binary("remui", i, p.const(2))
        ni = p.binary("addi", i, p.const(1))
        with p.when(p.compare("slt", ni, p.const(4))):
            nxt = p.binary("remui", ni, p.const(2))
            p.flag("wait", "MTE3", "MTE2", nxt)
            p.load(p.bank(nxt, 16, 64, 8192), x, row, p.binary("muli", ni, p.const(64)))
            p.flag("set", "MTE2", "V", nxt)
        p.flag("wait", "MTE2", "V", cur)
        p.op("tmov", [p.bank(cur, 16, 64, 8192)], [data])
        for op, value in (("tsub", mean), ("tdiv", std), ("tmul", gamma), ("tadd", beta)):
            p.op(op, [data, value], [data])
        out = p.bank(cur, 16, 64, 16384)
        p.op("tmov", [data], [out])
        p.flag("set", "V", "MTE3", cur)
        p.flag("wait", "V", "MTE3", cur)
        p.store(out, y, row, p.binary("muli", i, p.const(64)))
        p.flag("set", "MTE3", "MTE2", cur)
    for event in (0, 1):
        p.flag("wait", "MTE3", "MTE2", p.const(event))


def group_norm():
    p = Port("group_norm", [(v, "!pto.ptr<f32, gm>") for v in ("%X", "%gamma", "%beta", "%Y")])
    x, y = [p.view(v, 256, 256) for v in ("%X", "%Y")]
    gamma, beta = [p.view(v, 64, 1) for v in ("%gamma", "%beta")]
    cid = p.index(p.value("pto.get_block_idx", "i64"))
    vid = p.index(p.value("pto.get_subblock_idx", "i64"))
    row, group = p.binary("muli", cid, p.const(16)), p.binary("remui", cid, p.const(4))
    p.next_address = 24576
    with p.when(p.compare("eq", vid, p.const(0))):
        total, total_sq, mean_sq, var, std_val = [p.tile(1, 8, valid=(1, 1)) for _ in range(5)]
        group_norm_phase1(p, x, row, total, total_sq)
        for value in (total, total_sq):
            p.op("tdivs", [value, p.const("4096.0", "f32")], [value])
        p.op("tmul", [total, total], [mean_sq])
        p.op("tsub", [total_sq, mean_sq], [var])
        p.op("tadds", [var, p.const("1.0e-5", "f32")], [var])
        p.op("tsqrt", [var], [std_val])
        expanded = []
        for scalar in (total, std_val):
            col, bc = column(p, 16), p.tile(16, 64)
            p.op("texpands", [p.get(scalar, p.const(0))], [row_alias(p, col, 16)])
            p.op("trowexpand", [col], [bc])
            expanded.append(bc)
        gb, bb = group_norm_affine(p, gamma, beta, group)
        group_norm_phase2(p, x, y, row, *expanded, gb, bb)
    p.barrier()
    return p, {"factory": "group_norm_kernel_serial", "shape": [4, 64, 16, 16], "groups": 4,
               "cpg": 16, "cpg_padded": 16, "S_orig": 256, "S_padded": 256, "block_S": 64,
               "s_num": 4, "dtype": "float32", "eps": 1e-5, "launch_blocks": 16, "threads": 2,
               "scope": "Serial factory, no padding specialization; channel-pipeline factory not ported."}
