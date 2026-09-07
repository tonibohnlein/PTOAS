#!/usr/bin/env python3
# The Clear BSD License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Pinned source transcriptions for triangular inverse and GDN/KDA WY stages."""

from generate_kernel_pairs import Builder, LICENSE, ROOT, strip_local_sync, tile


SOURCE_COMMIT = "e118ec71ed0f170111d4f3380a7a93550c8c22ed"


def source_notice(source, filename):
    license_text = (ROOT / "references/pto-kernels-LICENSE").read_text()
    notice = "".join("// " + line + "\n" for line in license_text.splitlines())
    notice += f"// Source: huawei-csl/pto-kernels @ {SOURCE_COMMIT}\n// csrc/kernel/{filename}\n"
    return notice + source.removeprefix(LICENSE)


class RecurrenceBuilder(Builder):
    def fence(self, src, dst, event=0):
        self.flag("set", src, dst, event)
        self.flag("wait", src, dst, event)

    def begin(self, statement):
        self.add(statement + " {")
        self.indent += "  "

    def end(self):
        self.indent = self.indent[:-2]
        self.add("}")

    def otherwise(self):
        self.indent = self.indent[:-2]
        self.add("} else {")
        self.indent += "  "

    def binary(self, op, lhs, rhs, dst):
        self.add(
            f"pto.{op} ins(%{lhs}, %{rhs} : {self.types[lhs]}, {self.types[rhs]}) outs(%{dst} : {self.types[dst]})"
        )

    def cvt(self, src, dst):
        self.add(
            f"pto.tcvt ins(%{src} {{rmode = #pto<round_mode NONE>}} : {self.types[src]}) "
            f"outs(%{dst} : {self.types[dst]})"
        )

    def zero(self, dst):
        value = self.const("0.0", "f32")
        self.add(f"pto.texpands ins({value} : f32) outs(%{dst} : {self.types[dst]})")

    def cross(self, action, pipe, event):
        self.add(f"pto.sync.{action} <PIPE_{pipe}>, {event} {{ffts_mode = 2 : i32}}")


def triangular_inverse():
    # runTTriInv<float,16>, one already assigned contiguous per-AIV range.
    b = RecurrenceBuilder(
        "triangular_inverse_16",
        "%src: !pto.ptr<f32>, %dst: !pto.ptr<f32>, %matrices: index",
        "vector",
    )
    z, one, size = b.const(0), b.const(1), b.const(16)
    fone, minus = b.const("1.0", "f32"), b.const("-1.0", "f32")
    b.view("input", "src", 256, 16, "f32")
    b.view("output", "dst", 256, 16, "f32")
    b.alloc("matrix", 0, tile("vec", 16, 16))
    b.alloc("basis", 1024, tile("vec", 1, 16))
    b.alloc("inverse", 1088, tile("vec", 16, 16))
    b.flag("set", "MTE3", "MTE2", 0)
    b.flag("set", "MTE3", "V", 0)
    b.begin(f"scf.for %matrix_id = {z} to %matrices step {one}")
    b.flag("wait", "MTE3", "V", 0)
    b.zero("inverse")
    b.fence("V", "MTE2")
    b.flag("wait", "MTE3", "MTE2", 0)
    b.add(f"%row = arith.muli %matrix_id, {size} : index")
    b.part("inpart", "input", "%row", z, 16, 16, "f32")
    b.part("outpart", "output", "%row", z, 16, 16, "f32")
    b.unary("tload", "inpart", "matrix")
    b.fence("MTE2", "V")
    # Compile-time unroll of the two descending loops; preserves all fences.
    for j in range(15, -1, -1):
        b.add(f"// Source column sweep j={j}.")
        b.zero("basis")
        b.fence("V", "S")
        offset = b.const(j)
        b.add(f"pto.tsetval ins({offset}, {fone} : index, f32) outs(%basis : {b.types['basis']})")
        x = f"x{j}"
        b.alloc(x, 1088 + j * 64, tile("vec", 1, 16))
        if j == 0:
            b.add(f"pto.tsetval ins({z}, {fone} : index, f32) outs(%{x} : {b.types[x]})")
            continue
        a = f"a{j}_{j}"
        b.alloc(a, j * 64, tile("vec", 1, 16))
        b.fence("S", "V")
        b.add(f"pto.taxpy ins(%{a}, {minus} : {b.types[a]}, f32) outs(%basis : {b.types['basis']})")
        b.fence("V", "S")
        off = b.const(j - 1)
        xk = f"%value{j}_{j - 1}"
        b.add(f"{xk} = pto.tgetval ins(%basis, {off} : {b.types['basis']}, index) outs : f32")
        xkp1 = fone
        b.fence("V", "S")
        for k in range(j - 1, 0, -1):
            a = f"a{j}_{k}"
            b.alloc(a, k * 64, tile("vec", 1, 16))
            b.barrier()
            b.fence("S", "V")
            neg = f"%neg{j}_{k}"
            b.add(f"{neg} = arith.negf {xk} : f32")
            b.add(f"pto.taxpy ins(%{a}, {neg} : {b.types[a]}, f32) outs(%basis : {b.types['basis']})")
            b.fence("V", "S")
            off = b.const(k + 1)
            b.add(f"pto.tsetval ins({off}, {xkp1} : index, f32) outs(%{x} : {b.types[x]})")
            xkp1 = xk
            off = b.const(k - 1)
            xk = f"%value{j}_{k - 1}"
            b.add(f"{xk} = pto.tgetval ins(%basis, {off} : {b.types['basis']}, index) outs : f32")
        b.add(f"pto.tsetval ins({one}, {xkp1} : index, f32) outs(%{x} : {b.types[x]})")
        b.add(f"pto.tsetval ins({z}, {xk} : index, f32) outs(%{x} : {b.types[x]})")
    b.fence("V", "MTE3")
    b.unary("tstore", "inverse", "outpart")
    b.flag("set", "MTE3", "MTE2", 0)
    b.flag("set", "MTE3", "V", 0)
    b.end()
    b.flag("wait", "MTE3", "MTE2", 0)
    b.flag("wait", "MTE3", "V", 0)
    return source_notice(b.finish(), "kernel_tri_inv_col_sweep.cpp")


def padded_tile(space, rows, cols, dtype, valid_rows=None, valid_cols=None):
    base = tile(space, rows, cols, dtype)
    return base[:-1] + f", valid={valid_rows or rows}x{valid_cols or cols}, pad=1>"


def wy_builder(kda, kind):
    name = "kda_wy" if kda else "gdn_wy"
    # H=Hg=1, C=D=128. Input/output GM backing covers sixteen chunks.
    args = (
        "%k: !pto.ptr<f16>, %v: !pto.ptr<f16>, %beta: !pto.ptr<f16>, "
        "%g: !pto.ptr<f32>, %a: !pto.ptr<f16>, %ws2: !pto.ptr<f16>, "
        "%ws1: !pto.ptr<f16>, %u: !pto.ptr<f16>, %w: !pto.ptr<f16>, "
        "%ffts: !pto.ptr<i64>, %chunks: index, %tail_half: i1"
    )
    if kind == "vector":
        args += ", %stripe: index"
    b = RecurrenceBuilder(name + "_" + kind, args, kind)
    # PTOAS requires same-function FFTS setup; upstream pto-kernels assumes caller setup.
    b.add("pto.set_ffts %ffts : !pto.ptr<i64>")
    for n, dtype in (
        ("k", "f16"),
        ("v", "f16"),
        ("a", "f16"),
        ("u", "f16"),
        ("w", "f16"),
    ):
        b.view(n + "view", n, 2048, 128, dtype)
    b.view("betaview", "beta", 1, 2048, "f16")
    b.view("gview", "g", 2048 if kda else 1, 128 if kda else 2048, "f32")
    for n in ("ws1", "ws2"):
        b.view(n + "view", n, 128, 128, "f16")
    return b


def wy_loop(b, vector):
    z, one, width, half = b.const(0), b.const(1), b.const(128), b.const(64)
    b.begin(f"scf.for %chunk = {z} to %chunks step {one}")
    b.add(f"%base = arith.muli %chunk, {width} : index")
    b.add(f"%next = arith.addi %chunk, {one} : index")
    b.add("%last = arith.cmpi eq, %next, %chunks : index")
    b.add("%tail = arith.andi %last, %tail_half : i1")
    b.add(f"%reuse = arith.cmpi sgt, %chunk, {z} : index")
    if vector:
        b.add(f"%stripe_row = arith.muli %stripe, {half} : index")
        b.add("%row = arith.addi %base, %stripe_row : index")
        b.add(f"%lower = arith.cmpi eq, %stripe, {one} : index")
        b.add("%empty = arith.andi %tail, %lower : i1")
        b.add(f"%live = arith.cmpi eq, %empty, {b.const(0, 'i1')} : i1")


def load_padded(b, name, source, dst, addr, space, rows, cols, dtype, row, col, tail_rows, tail_cols):
    # Source full-vs-half tail. Explicit descriptor aliases preserve TASSIGN.
    b.begin("scf.if %tail")
    ty = padded_tile(space, rows, cols, dtype, tail_rows, tail_cols)
    b.alloc(name + "_tail", addr, ty)
    b.part(name + "_tail_part", source, row, col, tail_rows, tail_cols, dtype)
    b.unary("tload", name + "_tail_part", name + "_tail")
    # Source cube TFILLPAD uses the same descriptor; vector expands into full.
    b.unary("tfillpad", name + "_tail", name + "_tail" if space == "mat" else dst)
    b.otherwise()
    b.part(name + "_full_part", source, row, col, rows, cols, dtype)
    b.unary("tload", name + "_full_part", dst)
    b.end()


def publish(b, src, workspace, ready, free):
    b.begin("scf.if %reuse")
    b.cross("wait", "MTE3", free)
    b.end()
    b.fence("V", "MTE3")
    part = src + "_workspace"
    b.part(part, workspace + "view", "%stripe_row", b.const(0), 64, 128, "f16")
    b.unary("tstore", src, part)
    b.add("pto.barrier <PIPE_ALL>")
    b.cross("set", "MTE3", ready)


def wy_vector(kda):
    b = wy_builder(kda, "vector")
    z = b.const(0)
    if kda:
        plan = [
            ("bh", 0, 1, 128, "f16"),
            ("br", 512, 1, 128, "f32"),
            ("b2", 1024, 64, 128, "f32"),
            ("af", 33792, 64, 128, "f32"),
            ("a2", 66560, 64, 128, "f32"),
            ("a2h", 99328, 64, 128, "f16"),
            ("kf", 0, 64, 128, "f32"),
            ("gf", 32768, 64, 128, "f32"),
            ("eff", 65536, 64, 128, "f32"),
            ("effh", 98304, 64, 128, "f16"),
        ]
    else:
        plan = [
            ("bh", 0, 1, 128, "f16"),
            ("a1h", 256, 64, 128, "f16"),
            ("bf", 16640, 1, 128, "f32"),
            ("br", 17152, 1, 128, "f32"),
            ("b2", 17664, 64, 128, "f32"),
            ("a1", 75008, 64, 128, "f32"),
            ("a2", 107776, 64, 128, "f32"),
            ("a2h", 140544, 64, 128, "f16"),
            ("gf", 156928, 1, 128, "f32"),
            ("gr", 157440, 1, 128, "f32"),
            ("g2", 157952, 64, 128, "f32"),
        ]
    for name, addr, rows, cols, dtype in plan:
        padded = name in ({"bh", "af", "kf", "gf"} if kda else {"bh", "a1h", "gf"})
        b.alloc(
            name,
            addr,
            padded_tile("vec", rows, cols, dtype) if padded else tile("vec", rows, cols, dtype),
        )
    if kda:
        # Same address as output half tiles: phase-dependent staging aliases.
        b.alloc("ah", 99328, padded_tile("vec", 64, 128, "f16"))
        b.alloc("kh", 98304, padded_tile("vec", 64, 128, "f16"))
    wy_loop(b, True)
    load_padded(b, "beta_load", "betaview", "bh", 0, "vec", 1, 128, "f16", z, "%base", 1, 64)
    b.begin("scf.if %live")
    ah = "ah" if kda else "a1h"
    b.part("a_part", "aview", "%row", z, 64, 128, "f16")
    b.unary("tload", "a_part", ah)
    b.otherwise()
    b.zero("a2" if kda else "a1")
    b.barrier()
    b.cvt("a2" if kda else "a1", "a2h" if kda else "a1h")
    b.end()
    b.fence("MTE2", "V")  # One bundled publication for beta and A loads.
    if kda:
        b.begin("scf.if %live")
        b.cvt("bh", "br")
        b.barrier()
        b.cvt("ah", "af")
        b.barrier()
    else:
        b.cvt("bh", "bf")
        b.barrier()
        b.unary("tmov", "bf", "br")
        b.barrier()
    b.unary("tcolexpand", "br", "b2")
    if not kda:
        b.cvt("a1h", "a1")
    b.binary("tmul", "af" if kda else "a1", "b2", "a2")
    b.cvt("a2", "a2h")
    if kda:
        b.end()
    publish(b, "a2h", "ws2", 10 if kda else 2, 12 if kda else 3)
    if kda:
        b.begin("scf.if %live")
        b.part("k_part", "kview", "%row", z, 64, 128, "f16")
        b.unary("tload", "k_part", "kh")
        b.fence("MTE2", "V")
        b.cvt("kh", "kf")
        b.barrier()
        b.part("g_part", "gview", "%row", z, 64, 128, "f32")
        b.unary("tload", "g_part", "gf")
        b.fence("MTE2", "V")
        b.unary("texp", "gf", "gf")
        b.barrier()
        b.binary("tmul", "kf", "gf", "eff")
        b.barrier()
        b.cvt("eff", "effh")
        b.otherwise()
        b.zero("eff")
        b.barrier()
        b.cvt("eff", "effh")
        b.end()
    else:
        load_padded(
            b,
            "gate_load",
            "gview",
            "gf",
            156928,
            "vec",
            1,
            128,
            "f32",
            z,
            "%base",
            1,
            64,
        )
        b.fence("MTE2", "V")
        b.unary("texp", "gf", "gf")
        b.barrier()
        b.binary("tmul", "gf", "bf", "gf")
        b.barrier()
        b.unary("tmov", "gf", "gr")
        b.barrier()
        b.unary("tcolexpand", "gr", "g2")
        b.binary("tmul", "a1", "g2", "a1")
        b.cvt("a1", "a1h")
    publish(b, "effh" if kda else "a1h", "ws1", 11 if kda else 1, 13 if kda else 4)
    b.end()
    # KDA explicitly drains; GDN wy_fast source leaves trailing free credits.
    if kda:
        b.add(f"%worked = arith.cmpi sgt, %chunks, {z} : index")
        b.begin("scf.if %worked")
        b.cross("wait", "MTE3", 12)
        b.cross("wait", "MTE3", 13)
        b.end()
    return b.finish()


def wy_cube(kda):
    b = wy_builder(kda, "cube")
    z = b.const(0)
    plan = (
        [("v1", 0), ("a2", 32768), ("other", 65536)]
        if kda
        else [("k1", 0), ("v1", 32768), ("a2", 65536), ("other", 98304)]
    )
    for n, addr in plan:
        b.alloc(n, addr, padded_tile("mat", 128, 128, "f16"))
    for n, space, addr, dtype in [
        ("left", "left", 0, "f16"),
        ("right", "right", 0, "f16"),
        ("accu", "acc", 0, "f32"),
        ("accw", "acc", 65536, "f32"),
    ]:
        b.alloc(
            n,
            addr,
            tile(space, 128, 128, dtype) if space == "acc" else padded_tile(space, 128, 128, dtype),
        )
    wy_loop(b, False)
    if not kda:
        load_padded(b, "kload", "kview", "k1", 0, "mat", 128, 128, "f16", "%base", z, 64, 128)
    load_padded(
        b,
        "vload",
        "vview",
        "v1",
        0 if kda else 32768,
        "mat",
        128,
        128,
        "f16",
        "%base",
        z,
        64,
        128,
    )
    for stage, ready, free, lhs, rhs, acc, dst in (
        (2, 10 if kda else 2, 12 if kda else 3, "a2", "v1", "accu", "u"),
        (
            1,
            11 if kda else 1,
            13 if kda else 4,
            "a2" if kda else "other",
            "other" if kda else "k1",
            "accw",
            "w",
        ),
    ):
        b.cross("wait", "MTE2", ready)
        b.part(f"ws{stage}_part", f"ws{stage}view", z, z, 128, 128, "f16")
        b.unary("tload", f"ws{stage}_part", "a2" if stage == 2 else "other")
        b.fence("FIX", "M")
        # gemm_v0 K=128: single tail segment, with its exact flag placement.
        b.fence("MTE2", "MTE1", 1)
        b.fence("M", "MTE1", 1)
        for src, target in ((lhs, "left"), (rhs, "right")):
            b.add(
                f"pto.textract ins(%{src}, {z}, {z} : {b.types[src]}, index, index) outs(%{target} : {b.types[target]})"
            )
        b.fence("MTE1", "M", 1)
        b.binary("tmatmul", "left", "right", acc)
        b.fence("MTE1", "MTE2", 1)
        b.fence("M", "FIX", 1)
        b.begin("scf.if %tail")
        ty = tile("acc", 128, 128, "f32")[:-1] + ", valid=64x128>"
        b.alloc(acc + "_tail", 0 if stage == 2 else 65536, ty)
        b.part(dst + "_tail_part", dst + "view", "%base", z, 64, 128, "f16")
        b.unary("tstore", acc + "_tail", dst + "_tail_part")
        b.otherwise()
        b.part(dst + "_part", dst + "view", "%base", z, 128, 128, "f16")
        b.unary("tstore", acc, dst + "_part")
        b.end()
        b.cross("set", "FIX", free)
    b.end()
    return b.finish()


def wy(kda=False):
    # Two entry functions preserve separate engines. A device wrapper must
    # launch the cube function and both vector stripes as cooperating peers.
    vector, cube = wy_vector(kda), wy_cube(kda)
    source = vector.rsplit("}", 1)[0] + cube.split("module {", 1)[1]
    return source_notice(source, "kernel_kda_wy.cpp" if kda else "kernel_gdn_wy_fast.cpp")


GENERATORS = {
    "triangular_inverse_16": triangular_inverse,
    "gdn_wy": lambda: wy(False),
    "kda_wy": lambda: wy(True),
}

if __name__ == "__main__":
    for name, generate in GENERATORS.items():
        source = generate()
        (ROOT / "inputs" / f"{name}.manual.pto").write_text(source)
        (ROOT / "inputs" / f"{name}.auto.pto").write_text(strip_local_sync(source))
