#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Causal convolution ports with the original cache indices and state effects."""

from port_ir import Port
from port_vector import mul_add, silu


def gm_index(p, ptr, offset):
    value = p.value(f"pto.load {ptr}[{offset}] : {p.types[ptr]} -> i32", "i32")
    return p.index(value)


def cast(p, dst, src, mode="NONE"):
    p.emit(f"pto.tcvt ins({src} {{rmode = #pto<round_mode {mode}>}} : {p.types[src]}) "
           f"outs({dst} : {p.types[dst]})")


def causal_conv1d_decode():
    arguments = [(n, "!pto.ptr<bf16, gm>") for n in ("%X", "%W", "%state")]
    arguments += [(n, "!pto.ptr<i32, gm>") for n in ("%read_indices", "%write_indices", "%initial")]
    arguments += [(n, "!pto.ptr<bf16, gm>") for n in ("%bias", "%Y")]
    p = Port("causal_conv1d_decode", arguments)
    x, y = [p.view(v, 8, 2048, "bf16") for v in ("%X", "%Y")]
    w = p.view("%W", 4, 2048, "bf16")
    state = p.view("%state", 24, 2048, "bf16")
    bias = p.view("%bias", 1, 2048, "bf16")
    batch = p.index(p.value("pto.get_block_idx", "i64"))
    read = gm_index(p, "%read_indices", batch)
    write = gm_index(p, "%write_indices", batch)
    initial = gm_index(p, "%initial", batch)
    halves = [p.tile(1, 2048, "bf16") for _ in range(13)]
    wh, bh, hh, xh, yh, save = halves[:4], halves[4], halves[5:8], halves[8], halves[9], halves[10:]
    full = [p.tile(1, 2048) for _ in range(12)]
    wf, bf, hist, xf, accum, tmp, yf = full[:4], full[4], full[5:8], full[8], full[9], full[10], full[11]
    mul_tmp, silu_tmp = p.tile(1, 2048), p.tile(1, 2048)
    for i in range(4):
        p.load(wh[i], w, p.const(i), p.const(0))
    p.load(bh, bias, p.const(0), p.const(0))
    p.handoff("MTE2", "V", 1)
    for dst, src in zip([*wf, bf], [*wh, bh]):
        cast(p, dst, src)
    for tile in hist:
        p.op("texpands", [p.const("0.0", "f32")], [tile])
    with p.when(p.compare("ne", initial, p.const(0))):
        for i in range(3):
            row = p.binary("addi", p.binary("muli", read, p.const(3)), p.const(i))
            p.load(hh[i], state, row, p.const(0))
            p.handoff("MTE2", "V", i + 2)
            cast(p, hist[i], hh[i])
    p.load(xh, x, batch, p.const(0))
    p.handoff("MTE2", "V", 5)
    cast(p, xf, xh)
    p.op("tmul", [wf[0], hist[0]], [accum])
    for i in (1, 2):
        p.op("tmul", [wf[i], hist[i]], [tmp])
        p.op("tadd", [accum, tmp], [accum])
    mul_add(p, accum, xf, wf[3], mul_tmp)
    p.op("tadd", [accum, bf], [tmp])
    silu(p, yf, tmp, silu_tmp)
    cast(p, yh, yf, "CAST_RINT")
    p.handoff("V", "MTE3", 0)
    p.store(yh, y, batch, p.const(0))
    for dst, src in zip(save, [hist[1], hist[2], xf]):
        cast(p, dst, src, "CAST_RINT")
    p.handoff("V", "MTE3", 6)
    for i, tile in enumerate(save):
        row = p.binary("addi", p.binary("muli", write, p.const(3)), p.const(i))
        p.store(tile, state, row, p.const(0))
    p.barrier()
    return p, {"batch": 8, "dim": 2048, "width": 4, "state_shape": [8, 3, 2048],
               "dtype": "bfloat16", "activation": "silu", "launch_blocks": 8,
               "threads": 2, "mutates": ["state"],
               "scope": ("One feature-dimension chunk per batch; dynamic read/write cache indices "
                         "and initial-state flags retained."),
               "requirements": ["Cache read/write row unions are disjoint across tasks; indices are in range.",
                                "Authored ID 6 requires target/API qualification.",
                                "PTO helper expansion included; native AscendC equivalence needs device checking."]}


def prefill_row(p, slot, row, base):
    index = p.binary("addi", p.binary("muli", slot, p.const(4)), p.const(row))
    return p.bank(index, 1, 128, base, "f16")


def prefill_load(p, x, start, slot):
    for j in range(4):
        row = p.binary("addi", start, p.const(j))
        p.load(prefill_row(p, slot, j, 0), x, row, p.const(0))


def causal_conv1d_prefill():
    args = [(n, "!pto.ptr<f16, gm>") for n in ("%X", "%W", "%state")]
    args += [(n, "!pto.ptr<i32, gm>") for n in ("%seqlens", "%cache", "%initial")]
    args += [("%Y", "!pto.ptr<f16, gm>")]
    p = Port("causal_conv1d_prefill", args)
    x, y = [p.view(v, 32, 128, "f16") for v in ("%X", "%Y")]
    w, state = p.view("%W", 4, 128, "f16"), p.view("%state", 6, 128, "f16")
    batch = p.index(p.value("pto.get_block_idx", "i64"))
    start = gm_index(p, "%seqlens", batch)
    end = gm_index(p, "%seqlens", p.binary("addi", batch, p.const(1)))
    count = p.binary("subi", end, start)
    cache, initial = gm_index(p, "%cache", batch), gm_index(p, "%initial", batch)
    p.next_address = 4096  # Two banks each of 4x128 half inputs and outputs.
    first, last = p.tile(1, 8, "i32", valid=(1, 1)), p.tile(1, 8, "i32", valid=(1, 1))
    p.op("texpands", [p.const(1, "i32")], [first])
    p.op("texpands", [p.const(1, "i32")], [last])
    s = [p.tile(1, 128, "f16") for _ in range(3)]
    h = [p.tile(1, 128, "f16") for _ in range(3)]
    weights = [p.tile(1, 128, "f16") for _ in range(4)]
    tmp = p.tile(1, 128, "f16")
    save = [p.tile(1, 128, "f16") for _ in range(3)]
    helper_mul = [p.tile(1, 128, "f16") for _ in range(4)]
    helper_silu = [p.tile(1, 128, "f16") for _ in range(4)]
    for j in range(4):
        p.load(weights[j], w, p.const(j), p.const(0))
    p.barrier()
    for tile in h:
        p.op("texpands", [p.const("0.0", "f16")], [tile])
    is_first = p.compare("ne", p.get(first, p.const(0), "i32"), p.const(0, "i32"))
    has_initial = p.compare("ne", initial, p.const(0))
    condition = p.binary("andi", is_first, has_initial, "i1")
    with p.when(condition):
        for j in range(3):
            row = p.binary("addi", p.binary("muli", cache, p.const(3)), p.const(j))
            p.load(h[j], state, row, p.const(0))
    not_first = p.compare("eq", p.get(first, p.const(0), "i32"), p.const(0, "i32"))
    with p.when(not_first):
        for j in range(3):
            p.load(h[j], x, p.binary("addi", start, p.const(j)), p.const(0))
    p.barrier()
    p.op("tmul", [weights[0], h[2]], [s[2]])
    p.op("tmul", [weights[0], h[1]], [s[1]])
    p.op("tmul", [weights[1], h[2]], [tmp])
    p.op("tadd", [s[1], tmp], [s[1]])
    p.op("tmul", [weights[0], h[0]], [s[0]])
    for j in (1, 2):
        p.op("tmul", [weights[j], h[j]], [tmp])
        p.op("tadd", [s[0], tmp], [s[0]])
    iterations = p.binary("divui", p.binary("addi", count, p.const(3)), p.const(4))
    for event in (0, 1):
        p.flag("set", "MTE3", "MTE2", p.const(event))
    p.flag("wait", "MTE3", "MTE2", p.const(0))
    with p.when(p.compare("sgt", count, p.const(0))):
        prefill_load(p, x, start, p.const(0))
        p.flag("set", "MTE2", "V", p.const(0))
    with p.loop(p.const(0), iterations) as i:
        cur = p.binary("remui", i, p.const(2))
        ni = p.binary("addi", i, p.const(1))
        nxt = p.binary("remui", ni, p.const(2))
        output_row = p.binary("addi", start, p.binary("muli", i, p.const(4)))
        with p.when(p.compare("slt", ni, iterations)):
            p.flag("wait", "MTE3", "MTE2", nxt)
            prefill_load(p, x, p.binary("addi", start, p.binary("muli", ni, p.const(4))), nxt)
            p.flag("set", "MTE2", "V", nxt)
        p.flag("wait", "MTE2", "V", cur)
        for j in range(4):
            inp, out = prefill_row(p, cur, j, 0), prefill_row(p, cur, j, 2048)
            mul_add(p, s[0], inp, weights[3], helper_mul[j])
            silu(p, out, s[0], helper_silu[j])
            p.op("tmul", [weights[2], inp], [tmp])
            p.op("tadd", [tmp, s[1]], [s[0]])
            p.op("tmul", [weights[1], inp], [tmp])
            p.op("tadd", [tmp, s[2]], [s[1]])
            p.op("tmul", [weights[0], inp], [s[2]])
        p.flag("set", "V", "MTE3", cur)
        p.flag("wait", "V", "MTE3", cur)
        remain = p.binary("subi", count, p.binary("muli", i, p.const(4)))
        for j in range(4):
            with p.when(p.compare("sge", remain, p.const(j + 1))):
                p.store(prefill_row(p, cur, j, 2048), y, p.binary("addi", output_row, p.const(j)), p.const(0))
        p.flag("set", "MTE3", "MTE2", cur)
    for event in (0, 1):
        p.flag("wait", "MTE3", "MTE2", p.const(event))
    is_last = p.compare("ne", p.get(last, p.const(0), "i32"), p.const(0, "i32"))
    condition = p.binary("andi", is_last, p.compare("sgt", count, p.const(0)), "i1")
    with p.when(condition):
        for tile in save:
            p.op("texpands", [p.const("0.0", "f16")], [tile])
        for j in (2, 1, 0):
            p.load(save[j], x, p.binary("subi", end, p.const(3 - j)), p.const(0))
        p.barrier()
        for j, tile in enumerate(save):
            row = p.binary("addi", p.binary("muli", cache, p.const(3)), p.const(j))
            p.store(tile, state, row, p.const(0))
    p.barrier()
    return p, {"batch": 2, "max_total_tokens": 32, "dim": 128, "width": 4,
               "dim_num": 1, "dtype": "float16", "state_shape": [2, 3, 128],
               "launch_blocks": 2, "vector_lanes": 2, "mutates": ["state"],
               "scope": "One sequence partition per batch; dynamic sequence/cache/initial data preserved.",
               "requirements": ["Positive sequence lengths divisible by four; sum at most 32.",
                                "Distinct valid cache indices; initialize output padding for checking.",
                                "Source ignores vid; launch mapping must be checked against native TileLang."]}
