#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small text builder for explicit benchmark ports, not a production importer.

Only synchronization-tagged lines differ between the expert and input versions.
All allocations are fixed, disjoint and 32-byte aligned unless a port explicitly
supplies a bank address. Per-port metadata describes the chosen specialization.
"""

from contextlib import contextmanager
from pathlib import Path


class Port:
    def __init__(self, name, arguments, kind="vector"):
        self.name = name
        self.arguments = arguments
        self.kind = kind
        self.types = dict(arguments)
        self.constants = {}
        self.lines = []
        self.depth = 2
        self.counter = 0
        self.next_address = 0
        self.shapes = {}
        self.addresses = {}

    def emit(self, text, sync=False):
        self.lines.append(("  " * self.depth + text, sync))

    def value(self, rhs, typ):
        name = f"%v{self.counter}"
        self.counter += 1
        self.types[name] = typ
        self.emit(f"{name} = {rhs}")
        return name

    def const(self, value, typ="index"):
        key = (value, typ)
        if key not in self.constants:
            self.constants[key] = f"%c{len(self.constants)}"
            self.types[self.constants[key]] = typ
        return self.constants[key]

    def binary(self, op, left, right, typ="index"):
        return self.value(f"arith.{op} {left}, {right} : {typ}", typ)

    def compare(self, predicate, left, right):
        return self.value(f"arith.cmpi {predicate}, {left}, {right} : {self.types[left]}", "i1")

    def index(self, value):
        return self.value(f"arith.index_cast {value} : {self.types[value]} to index", "index")

    @contextmanager
    def loop(self, start, end, step=None):
        variable = f"%iv{self.counter}"
        self.counter += 1
        self.types[variable] = "index"
        step = self.const(1) if step is None else step
        self.emit(f"scf.for {variable} = {start} to {end} step {step} {{")
        self.depth += 1
        yield variable
        self.depth -= 1
        self.emit("}")

    @contextmanager
    def when(self, condition):
        self.emit(f"scf.if {condition} {{")
        self.depth += 1
        yield
        self.depth -= 1
        self.emit("}")

    def tile(self, rows, cols, dtype="f32", address=None, valid=None, layout="row_major"):
        vr, vc = valid or (rows, cols)
        typ = (f"!pto.tile_buf<loc=vec, dtype={dtype}, rows={rows}, cols={cols}, "
               f"v_row={vr}, v_col={vc}, blayout={layout}, slayout=none_box, fractal=512, pad=0>")
        if address is None:
            address = self.const(self.next_address, "i64")
            self.next_address += ((rows * cols * (2 if dtype in ("f16", "bf16", "i16") else 4) + 31) // 32) * 32
        elif self.types[address] != "i64":
            address = self.value(f"arith.index_cast {address} : index to i64", "i64")
        name = self.value(f"pto.alloc_tile addr = {address} : {typ}", typ)
        self.shapes[name] = (rows, cols, dtype, vr, vc)
        self.addresses[name] = address
        return name

    def bank(self, index, rows, cols, base, dtype="f32"):
        size = rows * cols * (2 if dtype in ("f16", "bf16", "i16") else 4)
        offset = self.binary("muli", index, self.const(size))
        offset = self.binary("addi", offset, self.const(base))
        return self.tile(rows, cols, dtype, address=offset)

    def view(self, pointer, rows, cols, dtype="f32"):
        typ = f"!pto.tensor_view<{rows}x{cols}x{dtype}>"
        return self.value(f"pto.make_tensor_view {pointer}, shape = [{self.const(rows)}, {self.const(cols)}], "
                          f"strides = [{self.const(cols)}, {self.const(1)}] : {typ}", typ)

    def part(self, view, row, col, rows, cols, dtype="f32"):
        typ = f"!pto.partition_tensor_view<{rows}x{cols}x{dtype}>"
        return self.value(f"pto.partition_view {view}, offsets = [{row}, {col}], "
                          f"sizes = [{self.const(rows)}, {self.const(cols)}] : {self.types[view]} -> {typ}", typ)

    def op(self, name, inputs, outputs, attrs=""):
        ins = ", ".join(inputs)
        itypes = ", ".join(self.types[v] for v in inputs)
        outs = ", ".join(outputs)
        otypes = ", ".join(self.types[v] for v in outputs)
        self.emit(f"pto.{name} ins({ins} : {itypes}) outs({outs} : {otypes}){attrs}")

    def load(self, tile, view, row, col):
        _, _, dtype, rows, cols = self.shapes[tile]
        part = self.part(view, row, col, rows, cols, dtype)
        self.op("tload", [part], [tile])

    def store(self, tile, view, row, col):
        _, _, dtype, rows, cols = self.shapes[tile]
        part = self.part(view, row, col, rows, cols, dtype)
        self.op("tstore", [tile], [part])

    def flag(self, action, src, dst, event):
        self.emit(f"pto.{action}_flag_dyn[<PIPE_{src}>, <PIPE_{dst}>, {event}]", sync=True)

    def handoff(self, src, dst, event=0):
        for action in ("set", "wait"):
            self.flag(action, src, dst, self.const(event))

    def barrier(self, pipe="ALL"):
        self.emit(f"pto.barrier #pto.pipe<PIPE_{pipe}>", sync=True)

    def get(self, tile, offset, dtype="f32"):
        return self.value(f"pto.tgetval ins({tile}, {offset} : {self.types[tile]}, index) outs : {dtype}", dtype)

    def put(self, tile, offset, value):
        self.op("tsetval", [offset, value], [tile])

    def render(self, expert):
        args = ", ".join(f"{v}: {t}" for v, t in self.arguments)
        text = [f'module attributes {{pto.target_arch = "a3", pto.kernel_kind = #pto.kernel_kind<{self.kind}>}} {{',
                f'  func.func @{self.name}({args}) attributes {{pto.kernel_kind = #pto.kernel_kind<{self.kind}>}} {{']
        for (value, typ), variable in self.constants.items():
            text.append(f"    {variable} = arith.constant {value} : {typ}")
        text.extend(line for line, sync in self.lines if expert or not sync)
        text.extend(["    return", "  }", "}", ""])
        header = Path(__file__).read_text(encoding="utf-8").splitlines()[1:8]
        license_text = "\n".join("//" + line[1:] for line in header)
        return license_text + "\n\n" + "\n".join(text)
