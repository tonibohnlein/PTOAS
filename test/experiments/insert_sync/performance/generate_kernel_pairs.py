#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Source-derived benchmark extractions. See KERNEL_PAIRS.md for scope/contracts."""

from pathlib import Path

ROOT = Path(__file__).parent
LICENSE = "\n".join("//" + line[1:] for line in Path(__file__).read_text().split('"""', 1)[0].splitlines()[1:]) + "\n"


def tile(space, rows, cols, dtype="f32", valid=None):
    config = ", blayout=col_major, slayout=row_major" if space in ("mat", "acc") else ""
    if space == "left":
        config = ", blayout=row_major, slayout=row_major"
    elif space == "right":
        config = ", blayout=row_major, slayout=col_major"
    elif space == "acc":
        config += ", fractal=1024"
    if valid is not None:
        config += f", valid={rows}x{valid}"
    return f"!pto.tile_buf<{space}, {rows}x{cols}x{dtype}{config}>"


class Builder:
    def __init__(self, name, arguments, kind):
        self.lines = [
            "module {",
            f"  func.func @{name}({arguments})",
            f"      attributes {{pto.kernel_kind = #pto.kernel_kind<{kind}>}} {{",
        ]
        self.types = {}
        self.constants = {}
        self.declarations = {}
        self.indent = "    "

    def add(self, line):
        self.lines.append(self.indent + line)

    def const(self, value, dtype="index"):
        key = value, dtype
        if key not in self.constants:
            name = f"%c{str(value).replace('-', 'n')}_{dtype}"
            self.lines.insert(3, f"    {name} = arith.constant {value} : {dtype}")
            self.constants[key] = name
        return self.constants[key]

    def alloc(self, name, address, ty):
        addr = self.const(address, "i64")
        self.add(f"%{name} = pto.alloc_tile addr = {addr} : {ty}")
        self.types[name] = ty

    def unary(self, op, src, dst):
        self.add(f"pto.{op} ins(%{src} : {self.types[src]}) outs(%{dst} : {self.types[dst]})")

    def flag(self, action, src, dst, event):
        self.add(f"pto.{action}_flag[<PIPE_{src}>, <PIPE_{dst}>, <EVENT_ID{event}>]")

    def barrier(self):
        self.add("pto.barrier <PIPE_V>")

    def call(self, name, operands):
        types = [ty for _, ty in operands]
        self.declarations[name] = f"  func.func private @{name}({', '.join(types)})"
        self.add(f"func.call @{name}({', '.join(value for value, _ in operands)}) : ({', '.join(types)}) -> ()")

    def view(self, name, pointer, rows, cols, dtype, stride=None):
        r, c, one = self.const(rows), self.const(cols), self.const(1)
        st = self.const(stride or cols)
        self.add(
            f"%{name} = pto.make_tensor_view %{pointer}, shape = [{r}, {c}], "
            f"strides = [{st}, {one}] : !pto.tensor_view<{rows}x{cols}x{dtype}>"
        )
        self.types[name] = f"!pto.tensor_view<{rows}x{cols}x{dtype}>"

    def part(self, name, source, row, col, rows, cols, dtype):
        rr, cc = self.const(rows), self.const(cols)
        self.types[name] = f"!pto.partition_tensor_view<{rows}x{cols}x{dtype}>"
        self.add(
            f"%{name} = pto.partition_view %{source}, offsets = [{row}, {col}], "
            f"sizes = [{rr}, {cc}] : {self.types[source]} -> {self.types[name]}"
        )

    def finish(self):
        self.add("return")
        return LICENSE + "\n".join(self.lines + ["  }"] + list(self.declarations.values()) + ["}"]) + "\n"


def topk():
    # Source runTOPK<float>: no-tail specialization validCol=128, topk=128,
    # SINGLE_LOOP_ROW=2, four rows per iteration, original backing offsets.
    b = Builder(
        "topk_128",
        "%src: !pto.ptr<f32>, %scores: !pto.ptr<f32>, %indices: !pto.ptr<ui32>, %inidx: !pto.ptr<ui32>, %groups: index",
        "vector",
    )
    z, one, four = b.const(0), b.const(1), b.const(4)
    b.view("srcview", "src", 64, 128, "f32")
    b.view("scoreview", "scores", 64, 128, "f32")
    b.view("indexview", "indices", 64, 128, "ui32")
    b.view("idxview", "inidx", 1, 128, "ui32")
    b.part("idxpart", "idxview", z, z, 1, 128, "ui32")
    b.alloc("idx", 8192, tile("vec", 1, 128, "ui32"))
    for slot in range(2):
        for row in range(2):
            suffix = f"{slot}_{row}"
            for name, addr, cols, dtype in (
                ("src", 14848 + slot * 1024 + row * 512, 128, "f32"),
                ("sort", slot * 2048 + row * 1024, 256, "f32"),
                ("merge", 4096 + slot * 2048 + row * 1024, 256, "f32"),
                ("data", 10752 + slot * 1024 + row * 512, 128, "f32"),
                ("index", 12800 + slot * 1024 + row * 512, 128, "ui32"),
            ):
                # TGATHER requires dst valid width == physical width in PTO.
                # Materialize the active 128-wide score view at the same address.
                ty = tile("vec", 1, 256, dtype, valid=128) if name == "src" else tile("vec", 1, cols, dtype)
                b.alloc(name + suffix, addr, ty)
        b.alloc(f"tmp{slot}", 8704 + slot * 1024, tile("vec", 1, 256))
        b.alloc(f"sorttmp{slot}", 8704 + slot * 1024, tile("vec", 1, 256, valid=128))
    b.unary("tload", "idxpart", "idx")
    for slot in range(2):
        b.flag("set", "V", "MTE2", slot)
    b.add(f"scf.for %group = {z} to %groups step {one} {{")
    b.indent += "  "
    b.add(f"%base = arith.muli %group, {four} : index")
    for slot in range(2):
        b.flag("wait", "V", "MTE2", slot)
        # The original TLOAD spans two rows. Use one 2x128 allocation/view.
        b.alloc(f"load{slot}", 14848 + slot * 1024, tile("vec", 2, 128))
        off = b.const(slot * 2)
        b.add(f"%row{slot} = arith.addi %base, {off} : index")
        b.part(f"loadpart{slot}", "srcview", f"%row{slot}", z, 2, 128, "f32")
        b.unary("tload", f"loadpart{slot}", f"load{slot}")
        b.flag("set", "MTE2", "V", slot)
        b.flag("wait", "MTE2", "V", slot)
        for row in range(2):
            suf = f"{slot}_{row}"
            src, idx, tmp, dst = f"src{suf}", "idx", f"sorttmp{slot}", f"sort{suf}"
            b.add(
                f"pto.tsort32 ins(%{src}, %{idx}, %{tmp} : {b.types[src]}, {b.types[idx]}, {b.types[tmp]}) "
                f"outs(%{dst} : {b.types[dst]})"
            )
            b.barrier()
        b.flag("set", "V", "MTE2", slot)
        b.barrier()
        for row in range(2):
            suf = f"{slot}_{row}"
            src, tmp = f"sort{suf}", f"tmp{slot}"
            length = b.const(64, "i32")
            b.add(f"pto.tmrgsort ins(%{src}, {length} : {b.types[src]}, i32) outs(%{tmp} : {b.types[tmp]})")
            b.barrier()
            b.unary("tmov", tmp, src)
            b.barrier()
            b.barrier()
            b.unary("tmov", src, f"merge{suf}")
        b.barrier()
        for row in range(2):
            src, dst = f"merge{slot}_{row}", f"data{slot}_{row}"
            b.add(
                f'pto.tgather ins(%{src}, {{maskPattern = #pto.mask_pattern<P0101>}} : {b.types[src]}, "row") '
                f"outs(%{dst} : {b.types[dst]})"
            )
        b.flag("set", "V", "MTE3", slot)
        b.barrier()
        for row in range(2):
            src, dst = f"merge{slot}_{row}", f"index{slot}_{row}"
            cast = src + "_bits"
            b.types[cast] = tile("vec", 1, 256, "ui32")
            b.add(f"%{cast} = pto.bitcast %{src} : {b.types[src]} -> {b.types[cast]}")
            b.add(
                f'pto.tgather ins(%{cast}, {{maskPattern = #pto.mask_pattern<P1010>}} : {b.types[cast]}, "row") '
                f"outs(%{dst} : {b.types[dst]})"
            )
        b.flag("set", "V", "MTE3", slot + 2)
        for name, target, dtype, addr, event in (
            ("data", "scoreview", "f32", 10752 + slot * 1024, slot),
            ("index", "indexview", "ui32", 12800 + slot * 1024, slot + 2),
        ):
            b.flag("wait", "V", "MTE3", event)
            b.alloc(f"{name}store{slot}", addr, tile("vec", 2, 128, dtype))
            b.part(f"{name}part{slot}", target, f"%row{slot}", z, 2, 128, dtype)
            b.unary("tstore", f"{name}store{slot}", f"{name}part{slot}")
    b.indent = b.indent[:-2]
    b.add("}")
    for slot in range(2):
        b.flag("wait", "V", "MTE2", slot)
    return b.finish()


def conv2d():
    # Interior mIter=3: rows 384..511, hinStart=3, hinCount=4,
    # woutStart=0, pad=[1,1,0,0]. Source baseM/K/N=128/48/256.
    b = Builder(
        "conv2d_interior_tile",
        "%fmap: !pto.ptr<f16>, %weight: !pto.ptr<f16>, %out: !pto.ptr<f16>, %panels: index",
        "cube",
    )
    z, one, two, three = (b.const(v) for v in (0, 1, 2, 3))
    for slot in range(2):
        b.alloc(f"fmap{slot}", slot * 18432, tile("mat", 128, 48, "f16"))
        b.alloc(f"weight{slot}", 131072 + slot * 73728, tile("mat", 144, 256, "f16"))
        b.alloc(f"left{slot}", slot * 32768, tile("left", 128, 48, "f16"))
        b.alloc(f"right{slot}", slot * 32768, tile("right", 48, 256, "f16"))
    b.alloc("acc", 0, tile("acc", 128, 256).replace(">", ", compact=1>"))
    b.call("benchmark_conv_setfmatrix", [])
    for slot in range(2):
        b.flag("set", "MTE1", "MTE2", slot)
        b.flag("set", "M", "MTE1", slot)
    b.add(f"scf.for %panel = {z} to %panels step {one} {{")
    b.indent += "  "
    b.add(f"%pslot = arith.remui %panel, {two} : index")
    for panel_slot in range(2):
        sv = b.const(panel_slot)
        b.add(f"%pactive{panel_slot} = arith.cmpi eq, %pslot, {sv} : index")
        b.add(f"scf.if %pactive{panel_slot} {{")
        b.indent += "  "
        b.flag("wait", "MTE1", "MTE2", panel_slot)
        b.call(
            "benchmark_conv_load_fmap",
            [("%fmap", "!pto.ptr<f16>"), (f"%fmap{panel_slot}", b.types[f"fmap{panel_slot}"]), ("%panel", "index")],
        )
        b.flag("set", "MTE2", "MTE1", 0)
        b.call(
            "benchmark_conv_load_weight",
            [
                ("%weight", "!pto.ptr<f16>"),
                (f"%weight{panel_slot}", b.types[f"weight{panel_slot}"]),
                ("%panel", "index"),
            ],
        )
        b.flag("set", "MTE2", "MTE1", 1)
        for use in range(3):
            offset = b.const(use * 48)
            # Three uses per panel invert L0 parity between adjacent L1 panels.
            l0 = (panel_slot * 3 + use) % 2
            b.flag("wait", "M", "MTE1", l0)
            if use == 0:
                b.flag("wait", "MTE2", "MTE1", 0)
            b.call(
                "benchmark_conv_img2col",
                [
                    (f"%fmap{panel_slot}", b.types[f"fmap{panel_slot}"]),
                    (f"%left{l0}", b.types[f"left{l0}"]),
                    (offset, "index"),
                ],
            )
            if use == 0:
                b.flag("wait", "MTE2", "MTE1", 1)
            b.call(
                "benchmark_conv_extract_weight",
                [
                    (f"%weight{panel_slot}", b.types[f"weight{panel_slot}"]),
                    (f"%right{l0}", b.types[f"right{l0}"]),
                    (offset, "index"),
                ],
            )
            if use == 2:
                b.flag("set", "MTE1", "MTE2", panel_slot)
            b.flag("set", "MTE1", "M", l0)
            b.flag("wait", "MTE1", "M", l0)
            ins = f"%left{l0}, %right{l0} : {b.types[f'left{l0}']}, {b.types[f'right{l0}']}"
            if use == 0:
                b.add(f"%first{panel_slot} = arith.cmpi eq, %panel, {z} : index")
                b.add(f"scf.if %first{panel_slot} {{")
                b.add(f"  pto.tmatmul ins({ins}) outs(%acc : {b.types['acc']})")
                b.add("} else {")
            b.add(
                f"  pto.tmatmul.acc ins(%acc, {ins.replace(':', ': ' + b.types['acc'] + ',', 1)}) "
                f"outs(%acc : {b.types['acc']})"
            )
            if use == 0:
                b.add("}")
            b.flag("set", "M", "MTE1", l0)
        b.indent = b.indent[:-2]
        b.add("}")
    b.indent = b.indent[:-2]
    b.add("}")
    b.flag("set", "M", "FIX", 0)
    b.flag("wait", "M", "FIX", 0)
    b.call("benchmark_conv_store", [("%out", "!pto.ptr<f16>"), ("%acc", b.types["acc"])])
    b.flag("set", "FIX", "M", 0)
    b.flag("wait", "FIX", "M", 0)
    for slot in range(2):
        b.flag("wait", "M", "MTE1", slot)
        b.flag("wait", "MTE1", "MTE2", slot)
    return b.finish()


def flash_cube():
    # compute_qk + compute_pv, noncausal / UF_ENABLE=0 / preload=0 /
    # one subtile / one L0 K segment. FIFO operations remain semantic payload.
    b = Builder(
        "flash_attention_cube",
        "%q: !pto.ptr<f16>, %k: !pto.ptr<f16>, %v: !pto.ptr<f16>, "
        "%qk_fifo: !pto.ptr<f32>, %p_fifo: !pto.ptr<f16>, %pv_fifo: !pto.ptr<f32>, %tiles: index",
        "cube",
    )
    z, one = b.const(0), b.const(1)
    b.view("qview", "q", 16, 16, "f16")
    b.view("kview", "k", 16, 256, "f16", 1)
    # K is transposed [HEAD,S1]: strides [1,HEAD].
    b.lines[-1] = b.lines[-1].replace(f"strides = [{one}, {one}]", f"strides = [{one}, {b.const(16)}]")
    b.lines[-1] = b.lines[-1].replace(" : !pto.tensor_view", " {layout = #pto.layout<dn>} : !pto.tensor_view")
    b.view("vview", "v", 256, 16, "f16")
    for name, ptr, dtype, size, direction, base in [
        ("qk", "qk_fifo", "f32", 1024, 1, 0),
        ("p", "p_fifo", "f16", 512, 2, 2),
        ("pv", "pv_fifo", "f32", 1024, 1, 4),
    ]:
        b.view(name + "desc", ptr, 16, 16, dtype)
        b.add(
            f'%{name}pipe = "pto.initialize_l2g2l_pipe"(%{name}desc) <{{dir_mask = {direction} : i8, '
            f"slot_size = {size} : i32, slot_num = 8 : i32, flag_base = {base} : i32, "
            f'operandSegmentSizes = array<i32: 1, 0, 0>}}> : ({b.types[name + "desc"]}) -> !pto.pipe'
        )
        b.add(f"%{name}entry = pto.declare_global -> !pto.tensor_view<16x16x{dtype}>")
        b.types[name + "entry"] = f"!pto.tensor_view<16x16x{dtype}>"
    for name, addr in [("qmat", 0), ("kmat", 512), ("pmat", 1024), ("vmat", 1536)]:
        ty = tile("mat", 16, 16, "f16")
        if name in ("kmat", "vmat"):
            ty = ty.replace("blayout=col_major, slayout=row_major", "blayout=row_major, slayout=col_major")
        b.alloc(name, addr, ty)
    for slot in range(2):
        b.alloc(f"left{slot}", 32768 * slot, tile("left", 16, 16, "f16"))
        b.alloc(f"right{slot}", 32768 * slot, tile("right", 16, 16, "f16"))
    b.alloc("qkacc", 0, tile("acc", 16, 16))
    b.alloc("pvacc", 1024, tile("acc", 16, 16))
    for slot in range(2):
        b.flag("set", "M", "MTE1", slot)
        b.flag("set", "MTE1", "MTE2", slot)
        b.flag("set", "FIX", "M", slot)
    b.add(f"scf.for %it = {z} to %tiles step {one} {{")
    b.indent += "  "
    sixteen = b.const(16)
    b.add(f"%off = arith.muli %it, {sixteen} : index")
    b.add("pto.talloc(%qkentry, %qkpipe : !pto.tensor_view<16x16xf32>, !pto.pipe) {split = 1}")
    b.flag("wait", "MTE1", "MTE2", 0)
    b.add(f"%first = arith.cmpi eq, %it, {z} : index")
    b.add("scf.if %first {")
    b.part("qpart", "qview", z, z, 16, 16, "f16")
    b.unary("tload", "qpart", "qmat")
    b.add("}")
    b.part("kpart", "kview", z, "%off", 16, 16, "f16")
    b.unary("tload", "kpart", "kmat")
    for phase, a, right, acc, slot in [("qk", "qmat", "kmat", "qkacc", 0), ("pv", "pmat", "vmat", "pvacc", 1)]:
        if phase == "pv":
            b.add("pto.tpop(%pentry, %ppipe : !pto.tensor_view<16x16xf16>, !pto.pipe) {split = 1}")
            b.flag("wait", "MTE1", "MTE2", 1)
            b.part("vpart", "vview", "%off", z, 16, 16, "f16")
            b.unary("tload", "vpart", "vmat")
            b.part("ppart", "pentry", z, z, 16, 16, "f16")
            b.unary("tload", "ppart", "pmat")
            b.add("pto.tfree(%ppipe : !pto.pipe) {split = 1}")
        b.flag("set", "MTE2", "MTE1", 0)
        b.flag("wait", "MTE2", "MTE1", 0)
        b.flag("wait", "FIX", "M", slot)
        b.flag("wait", "M", "MTE1", slot)
        for src, dst in [(a, f"left{slot}"), (right, f"right{slot}")]:
            b.add(f"pto.textract ins(%{src}, {z}, {z} : {b.types[src]}, index, index) outs(%{dst} : {b.types[dst]})")
        b.flag("set", "MTE1", "M", slot)
        b.flag("wait", "MTE1", "M", slot)
        b.add(
            f"pto.tmatmul ins(%left{slot}, %right{slot} : {b.types[f'left{slot}']}, {b.types[f'right{slot}']}) "
            f"outs(%{acc} : {b.types[acc]})"
        )
        b.flag("set", "M", "MTE1", slot)
        b.flag("set", "MTE1", "MTE2", slot)
        b.flag("set", "M", "FIX", 0)
        b.flag("wait", "M", "FIX", 0)
        if phase == "pv":
            b.add("pto.talloc(%pventry, %pvpipe : !pto.tensor_view<16x16xf32>, !pto.pipe) {split = 1}")
        b.part(phase + "out", phase + "entry", z, z, 16, 16, "f32")
        b.unary("tstore", acc, phase + "out")
        b.add(f"pto.tpush(%{phase}entry, %{phase}pipe : !pto.tensor_view<16x16xf32>, !pto.pipe) {{split = 1}}")
        b.flag("set", "FIX", "M", slot)
    b.indent = b.indent[:-2]
    b.add("}")
    for slot in range(2):
        b.flag("wait", "M", "MTE1", slot)
        b.flag("wait", "MTE1", "MTE2", slot)
        b.flag("wait", "FIX", "M", slot)
    b.add("pto.barrier <PIPE_ALL>")
    return b.finish()


def strip_local_sync(source):
    """Retain queue/ownership operations and all scalar/control/computation."""
    return (
        "\n".join(
            line
            for line in source.splitlines()
            if not any(op in line for op in ("pto.set_flag[", "pto.wait_flag[", "pto.barrier "))
        )
        + "\n"
    )


if __name__ == "__main__":
    for name, source in [("topk_128", topk()), ("conv2d_interior", conv2d()), ("flash_attention_cube", flash_cube())]:
        (ROOT / "inputs" / f"{name}.manual.pto").write_text(source)
        (ROOT / "inputs" / f"{name}.auto.pto").write_text(strip_local_sync(source))
