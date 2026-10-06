#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Variable-length recurrent gating, with the source's QKV prefetch protocol."""

from port_ir import Port
from port_conv import cast, gm_index
from port_groupnorm import column, row_alias
from port_vector import mul_add


def gating_buffers(p):
    p.next_address = 320  # Q[2,32], K[2,32], V[2,16] in fp16.
    shapes = {"q": (1, 32), "k": (1, 32), "k1": (1, 32), "v": (1, 16),
              "kb": (16, 32), "work": (16, 32), "h": (16, 32), "pred": (1, 16),
              "delta": (1, 16), "normsq": (1, 32), "reduce_tmp": (16, 32),
              "mul_tmp": (16, 32)}
    b = {name: p.tile(*shape) for name, shape in shapes.items()}
    b.update({name: p.tile(1, 16, valid=(1, 1)) for name in ("s1", "s2", "sp", "alpha", "norm", "recip_tmp")})
    b.update({name: p.tile(1, 16, "f16", valid=(1, 1)) for name in ("half1", "half2")})
    b.update({name: p.tile(16, 32, "f16") for name in ("hload", "hstore")})
    b.update({name: column(p, 16) for name in ("predcol", "deltacol")})
    b["outbase"] = p.next_address
    p.next_address += 64
    return b


def prefetch(p, views, token, offset, slot):
    for name, base, count, col in (("query", 0, 32, p.const(0)),
                                   ("key", 128, 32, p.const(0)), ("value", 256, 16, offset)):
        p.load(p.bank(slot, 1, count, base, "f16"), views[name], token, col)
    p.flag("set", "MTE2", "V", p.const(6))


def gate_scalars(p, b, views, token, dt, exp_a):
    p.load(b["half1"], views["a"], token, p.const(0))
    p.load(b["half2"], views["beta"], token, p.const(0))
    p.handoff("MTE2", "V", 5)
    cast(p, b["s1"], b["half1"])
    cast(p, b["s2"], b["half2"])
    x = p.binary("addf", p.get(b["s1"], p.const(0)), dt, "f32")
    beta = p.get(b["s2"], p.const(0))
    gt = p.value(f"arith.cmpf ogt, {x}, {p.const('20.0', 'f32')} : f32", "i1")
    with p.when(gt):
        p.put(b["sp"], p.const(0), x)
        p.depth -= 1
        p.emit("} else {")
        p.depth += 1
        p.put(b["s1"], p.const(0), x)
        p.op("texp", [b["s1"]], [b["alpha"]])
        p.op("tadds", [b["alpha"], p.const("1.0", "f32")], [b["alpha"]])
        p.op("tlog", [b["alpha"]], [b["alpha"]])
        p.put(b["sp"], p.const(0), p.get(b["alpha"], p.const(0)))
    neg_a = p.value(f"arith.negf {exp_a} : f32", "f32")
    value = p.binary("mulf", neg_a, p.get(b["sp"], p.const(0)), "f32")
    p.put(b["s1"], p.const(0), value)
    p.op("texp", [b["s1"]], [b["alpha"]])
    p.put(b["s1"], p.const(0), p.value(f"arith.negf {beta} : f32", "f32"))
    p.op("texp", [b["s1"]], [b["s1"]])
    p.op("tadds", [b["s1"], p.const("1.0", "f32")], [b["s1"]])
    p.op("trecip", [b["s1"]], [b["recip_tmp"]])
    p.barrier("V")
    p.op("tmov", [b["recip_tmp"]], [b["s1"]])
    return p.get(b["s1"], p.const(0))


def normalize(p, b, vector):
    p.op("tmul", [vector, vector], [b["normsq"]])
    p.op("trowsum", [b["normsq"], p.tile(1, 32)], [b["norm"]])
    p.op("tadds", [b["norm"], p.const("1.0e-12", "f32")], [b["norm"]])
    p.op("trsqrt", [b["norm"]], [b["norm"]])
    p.op("tmuls", [vector, p.get(b["norm"], p.const(0))], [vector])


def contraction(p, b, vector):
    p.op("tmov", [vector], [b["k1"]])
    p.op("tcolexpand", [b["k1"]], [b["kb"]])
    p.op("tmul", [b["h"], b["kb"]], [b["work"]])
    p.op("trowsum", [b["work"], b["reduce_tmp"]], [b["predcol"]])


def token_loop(p, b, views, start, length, offset, dt, exp_a):
    with p.loop(p.const(0), length) as i:
        token = p.binary("addi", start, i)
        slot = p.binary("remui", i, p.const(2))
        p.flag("wait", "MTE2", "V", p.const(6))
        gate = gate_scalars(p, b, views, token, dt, exp_a)
        for name, base, count in (("q", 0, 32), ("k", 128, 32), ("v", 256, 16)):
            cast(p, b[name], p.bank(slot, 1, count, base, "f16"))
        ni = p.binary("addi", i, p.const(1))
        with p.when(p.compare("slt", ni, length)):
            prefetch(p, views, p.binary("addi", token, p.const(1)), offset,
                     p.binary("remui", ni, p.const(2)))
        for name in ("q", "k"):
            normalize(p, b, b[name])
        p.op("tmuls", [b["q"], p.const("0.1767766952966369", "f32")], [b["q"]])
        p.op("tmuls", [b["h"], p.get(b["alpha"], p.const(0))], [b["h"]])
        contraction(p, b, b["k"])
        p.op("tmov", [row_alias(p, b["predcol"], 16)], [b["pred"]])
        p.op("tsub", [b["v"], b["pred"]], [b["delta"]])
        p.op("tmuls", [b["delta"], gate], [b["delta"]])
        p.op("tmov", [b["delta"]], [row_alias(p, b["deltacol"], 16)])
        p.op("trowexpand", [b["deltacol"]], [b["work"]])
        mul_add(p, b["h"], b["kb"], b["work"], b["mul_tmp"])
        contraction(p, b, b["q"])
        out = p.bank(slot, 1, 16, b["outbase"], "f16")
        cast(p, out, row_alias(p, b["predcol"], 16), "CAST_RINT")
        p.handoff("V", "MTE3", 0)
        p.store(out, views["out"], token, offset)


def gating_visit(p, b, views, seq, offset):
    start = gm_index(p, "%cu_seqlens", seq)
    end = gm_index(p, "%cu_seqlens", p.binary("addi", seq, p.const(1)))
    length = p.binary("subi", end, start)
    state = gm_index(p, "%state_indices", seq)
    p.op("texpands", [p.const("0.0", "f32")], [b["h"]])
    with p.when(p.compare("sge", state, p.const(0))):
        row = p.binary("addi", p.binary("muli", state, p.const(32)), offset)
        p.load(b["hload"], views["init_state"], row, p.const(0))
        p.handoff("MTE2", "V", 1)
        cast(p, b["h"], b["hload"])
    p.load(b["half1"], views["A_log"], p.const(0), p.const(0))
    p.handoff("MTE2", "V", 2)
    cast(p, b["s1"], b["half1"])
    p.op("texp", [b["s1"]], [b["sp"]])
    exp_a = p.get(b["sp"], p.const(0))
    p.handoff("V", "MTE2", 3)
    p.load(b["half1"], views["dt_bias"], p.const(0), p.const(0))
    p.handoff("MTE2", "V", 4)
    cast(p, b["s1"], b["half1"])
    dt = p.get(b["s1"], p.const(0))
    prefetch(p, views, start, offset, p.const(0))
    token_loop(p, b, views, start, length, offset, dt, exp_a)
    cast(p, b["hstore"], b["h"], "CAST_RINT")
    p.handoff("V", "MTE3", 5)
    row = p.binary("addi", p.binary("muli", seq, p.const(32)), offset)
    p.store(b["hstore"], views["final_state"], row, p.const(0))


def fused_gating_delta():
    half_names = ("A_log", "a", "dt_bias", "query", "key", "value", "beta", "init_state")
    args = [("%" + name, "!pto.ptr<f16, gm>") for name in half_names]
    args += [("%" + name, "!pto.ptr<i32, gm>") for name in ("state_indices", "cu_seqlens")]
    args += [("%" + name, "!pto.ptr<f16, gm>") for name in ("out", "final_state")]
    p = Port("fused_gating_delta", args)
    shapes = {"A_log": (1, 1), "a": (192, 1), "dt_bias": (1, 1), "query": (192, 32),
              "key": (192, 32), "value": (192, 32), "beta": (192, 1),
              "init_state": (768, 32), "final_state": (768, 32), "out": (192, 32)}
    views = {name: p.view("%" + name, *shape, "f16") for name, shape in shapes.items()}
    cid = p.index(p.value("pto.get_block_idx", "i64"))
    vid = p.index(p.value("pto.get_subblock_idx", "i64"))
    offset = p.binary("muli", vid, p.const(16))
    b = gating_buffers(p)
    with p.loop(p.const(0), p.const(2)) as work:
        with p.when(p.compare("slt", work, p.const(1))):
            gating_visit(p, b, views, p.binary("addi", cid, work), offset)
    p.barrier()
    return p, {"num_seqs": 24, "num_cache_slots": 24, "total_tokens_padded": 192,
               "max_seq_len": 8, "nk": 1, "nv": 1, "dk": 32, "dv": 32, "block_v": 32,
               "scale": 0.1767766952966369, "use_qk_l2norm": True, "softplus_beta": 1.0,
               "dtype": "float16", "launch_blocks": 24, "threads": 2,
               "requirements": ["Each sequence has 1..8 tokens; prefix sum <=192.",
                                "Initial state index is -1 or in [0,24).",
                                "Authored ID 6 and upstream auto-sync defaults require device qualification."]}
