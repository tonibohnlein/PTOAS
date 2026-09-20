#!/usr/bin/env python3
"""PTO transcription of pinned manual PTO-ISA GEMM. See README.md.

Only synchronization differs between manual.pto and input.pto. The original
four-wide K panel is represented as panel/substep loops in BOTH files.
"""
import argparse
import json
from pathlib import Path

PIN = "c0d7148e95ef73bd12a73165fdce4b723a3b7e72"
CASES = {
    "smoke": dict(m=256, n=512, k=512, cm=256, cn=512, cores=1),
    "reference": dict(m=6144, n=6144, k=6144, cm=1536, cn=1024, cores=24),
}


def generate(c, manual, banked_ready=False):
    m, n, k, cm, cn = (c[x] for x in ("m", "n", "k", "cm", "cn"))
    assert k % 512 == 0  # Original flags return to zero between output tiles.
    assert m % cm == n % cn == cm % 128 == cn % 256 == 0
    assert c["cores"] == (m // cm) * (n // cn)
    lines = [f"// PTO-ISA {PIN}: manual/a2a3/gemm_performance.",
             "// Panel/substep normalization, unchanged payload and event sequence.",
             'module attributes {pto.backend = "emitc", pto.kernel_kind = #pto.kernel_kind<cube>, pto.target_arch = "a3"} {',
             '  func.func @reference_gemm(%out: !pto.ptr<f32, gm>, %a: !pto.ptr<f16, gm>, %b: !pto.ptr<f16, gm>, %core: i32) attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {']
    def emit(s): lines.append("    " + s)
    values = {0, 1, 2, 3, 4, 64, 128, 256, 32768, 65536, 131072, m, n, k, cm, cn,
              m // cm, cm // 128, cn // 256, k // 256}
    for v in sorted(values): emit(f"%c{v} = arith.constant {v} : index")
    emit("%zero64 = arith.constant 0 : i64")
    types = {
        "at": "!pto.tile_buf<mat, 128x256xf16, valid=?x?, blayout=col_major, slayout=row_major>",
        "bt": "!pto.tile_buf<mat, 256x256xf16, valid=?x?, blayout=row_major, slayout=col_major>",
        "left": "!pto.tile_buf<left, 128x64xf16, valid=?x?, slayout=row_major>",
        "right": "!pto.tile_buf<right, 64x256xf16, valid=?x?, slayout=col_major>",
        "acc": "!pto.tile_buf<acc, 128x256xf32, valid=?x?, blayout=col_major, slayout=row_major, fractal=1024>",
    }
    def alloc(name, addr, rows, cols):
        emit(f"%{name} = pto.alloc_tile addr = %{addr} valid_row = %c{rows} valid_col = %c{cols} : {types[name]}")
    def flag(action, src, dst, key):
        if not manual: return
        def word(i): emit(f"pto.{action}_flag[<PIPE_{src}>, <PIPE_{dst}>, <EVENT_ID{i}>]")
        if isinstance(key, int): word(key)
        else:
            emit(f"scf.if %{key} {{")
            word(0)
            emit("} else {")
            word(1)
            emit("}")
    def ready(action, operand):
        if banked_ready:
            emit("scf.if %bank0 {")
            flag(action, "MTE2", "MTE1", operand)
            emit("} else {")
            flag(action, "MTE2", "MTE1", operand+2)
            emit("}")
        else:
            flag(action, "MTE2", "MTE1", operand)
    for name, ptr, rows, cols, rs, cs, dtype, layout in (
        ("av", "a", m, k, k, 1, "f16", "nd"),
        ("bv", "b", k, n, 1, k, "f16", "dn"),
        ("cv", "out", m, n, n, 1, "f32", "nd"),
    ):
        emit(f"%{name} = pto.make_tensor_view %{ptr}, shape = [%c{rows}, %c{cols}], strides = [%c{rs}, %c{cs}] {{layout = #pto.layout<{layout}>}} : !pto.tensor_view<?x?x{dtype}>")
    alloc("acc", "zero64", 128, 256)
    emit("%cid = arith.index_cast %core : i32 to index")
    emit(f"%mi = arith.remsi %cid, %c{m // cm} : index")
    emit(f"%ni = arith.divsi %cid, %c{m // cm} : index")
    emit(f"%mbase = arith.muli %mi, %c{cm} : index")
    emit(f"%nbase = arith.muli %ni, %c{cn} : index")
    for src, dst in (("MTE1", "MTE2"), ("M", "MTE1")):
        for key in (0, 1): flag("set", src, dst, key)
    emit(f"scf.for %i = %c0 to %c{cm // 128} step %c1 {{")
    emit(f"scf.for %j = %c0 to %c{cn // 256} step %c1 {{")
    emit("%ir = arith.muli %i, %c128 : index")
    emit("%jr = arith.muli %j, %c256 : index")
    emit("%row = arith.addi %mbase, %ir : index")
    emit("%col = arith.addi %nbase, %jr : index")
    emit(f"%panel_result = scf.for %panel = %c0 to %c{k // 256} step %c1 iter_args(%previous_bank = %c1) -> (index) {{")
    emit("%next_bank = arith.addi %previous_bank, %c1 : index")
    emit("%bank = arith.remsi %next_bank, %c2 : index")
    emit("%bank0 = arith.cmpi eq, %bank, %c0 : index")
    emit("%ko = arith.muli %panel, %c256 : index")
    emit("%aa = arith.muli %bank, %c65536 : index")
    emit("%ab = arith.muli %bank, %c131072 : index")
    emit("%bb = arith.addi %ab, %c131072 : index")
    emit("%addr_a = arith.index_cast %aa : index to i64")
    emit("%addr_b = arith.index_cast %bb : index to i64")
    alloc("at", "addr_a", 128, 256)
    alloc("bt", "addr_b", 256, 256)
    emit("%ap = pto.partition_view %av, offsets = [%row, %ko], sizes = [%c128, %c256] : !pto.tensor_view<?x?xf16>")
    emit("%bp = pto.partition_view %bv, offsets = [%ko, %col], sizes = [%c256, %c256] : !pto.tensor_view<?x?xf16>")
    flag("wait", "MTE1", "MTE2", "bank0")
    emit(f"pto.tload ins(%ap : !pto.partition_tensor_view<128x256xf16>) outs(%at : {types['at']}) {{layout = #pto.layout<nd>}}")
    ready("set", 0)
    emit(f"pto.tload ins(%bp : !pto.partition_tensor_view<256x256xf16>) outs(%bt : {types['bt']}) {{layout = #pto.layout<dn>}}")
    ready("set", 1)
    emit("%inner_result = scf.for %sub = %c0 to %c4 step %c1 iter_args(%previous_l0 = %c1) -> (index) {")
    emit("%next_l0 = arith.addi %previous_l0, %c1 : index")
    emit("%l0 = arith.remsi %next_l0, %c2 : index")
    emit("%l00 = arith.cmpi eq, %l0, %c0 : index")
    emit("%lo = arith.muli %l0, %c32768 : index")
    emit("%addr_l0 = arith.index_cast %lo : index to i64")
    alloc("left", "addr_l0", 128, 64)
    alloc("right", "addr_l0", 64, 256)
    emit("%slice = arith.muli %sub, %c64 : index")
    emit("%first = arith.cmpi eq, %sub, %c0 : index")
    emit("%last = arith.cmpi eq, %sub, %c3 : index")
    flag("wait", "M", "MTE1", "l00")
    if manual:
        emit("scf.if %first {")
        ready("wait", 0)
        emit("}")
    emit(f"pto.textract ins(%at, %c0, %slice : {types['at']}, index, index) outs(%left : {types['left']})")
    if manual:
        emit("scf.if %first {")
        ready("wait", 1)
        emit("}")
    emit(f"pto.textract ins(%bt, %slice, %c0 : {types['bt']}, index, index) outs(%right : {types['right']})")
    if manual:
        emit("scf.if %last {")
        flag("set", "MTE1", "MTE2", "bank0")
        emit("}")
    flag("set", "MTE1", "M", "l00")
    flag("wait", "MTE1", "M", "l00")
    emit("%first_panel = arith.cmpi eq, %panel, %c0 : index")
    emit("%initialize = arith.andi %first_panel, %first : i1")
    emit("scf.if %initialize {")
    emit(f"pto.tmatmul ins(%left, %right : {types['left']}, {types['right']}) outs(%acc : {types['acc']})")
    emit("} else {")
    emit(f"pto.tmatmul.acc ins(%acc, %left, %right : {types['acc']}, {types['left']}, {types['right']}) outs(%acc : {types['acc']})")
    emit("}")
    flag("set", "M", "MTE1", "l00")
    emit("scf.yield %l0 : index")
    emit("}")
    emit("scf.yield %bank : index")
    emit("}")
    flag("set", "M", "FIX", 0)
    flag("wait", "M", "FIX", 0)
    emit("%cp = pto.partition_view %cv, offsets = [%row, %col], sizes = [%c128, %c256] : !pto.tensor_view<?x?xf32>")
    emit(f"pto.tstore ins(%acc : {types['acc']}) outs(%cp : !pto.partition_tensor_view<128x256xf32>) {{layout = #pto.layout<nd>}}")
    flag("set", "FIX", "M", 0)
    flag("wait", "FIX", "M", 0)
    emit("}")
    emit("}")
    for src, dst in (("M", "MTE1"), ("MTE1", "MTE2")):
        for key in (0, 1): flag("wait", src, dst, key)
    emit("return")
    lines += ["  }", "}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    for name, config in CASES.items():
        dest = args.output / name
        dest.mkdir(parents=True, exist_ok=True)
        for manual, filename in ((True, "manual.pto"), (False, "input.pto")):
            (dest / filename).write_text(generate(config, manual))
        (dest / "manual_banked_keys.pto").write_text(generate(config, True, True))
        (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
