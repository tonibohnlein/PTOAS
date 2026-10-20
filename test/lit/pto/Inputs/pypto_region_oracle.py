# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent bounded interpreter of the surveyed physical-storage PTO subset.

Reads original address/stride/type declarations, never compiler cells, effects,
records or queries. Vector scratch operands use their supplied full-tile envelope;
that is not an assertion of exact native scratch accesses. Unsupported operations
and data-dependent addresses refuse, never manufacture disjointness or emptiness.
"""

import itertools
import re
from copy import deepcopy

from check_periodic_demands import closure, native

VALUE = r"%[A-Za-z_0-9][A-Za-z_0-9-]*"
SPACES = {"gm": 1, "mat": 2, "left": 3, "right": 4, "acc": 5, "vec": 6}
WIDTHS = {"bf16": 2, "f16": 2, "f32": 4, "i32": 4}


def values(text):
    return re.findall(VALUE, text)


def merge(ranges):
    result = []
    for begin, end in sorted(ranges):
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((begin, end))
    return result


def tile_ranges(tile, window=None):
    rows, cols = tile["valid"]
    row, col = 0, 0
    if window:
        row, col, rows, cols = window
    width, start = tile["width"], tile["address"]
    shape = tile["shape"]
    if row < 0 or col < 0 or row + rows > shape[0] or col + cols > shape[1]:
        raise ValueError("Tile access exceeds supplied shape")
    if tile["space"] == 2:
        # Supplied NZ map: column-major outer 16x16 blocks, row-major inner.
        if not tile["column"] or tile["subcolumn"] or shape[0] % 16 or shape[1] % 16:
            raise ValueError("Unsupported matrix layout")
        ranges = []
        for c in range(col, col + cols, 16):
            end_col = min(col + cols, (c // 16 + 1) * 16)
            for r in range(row, row + rows):
                element = (
                    ((c // 16) * (shape[0] // 16) + r // 16) * 256
                    + (r % 16) * 16
                    + c % 16
                )
                ranges.append(
                    (start + width * element, start + width * (element + end_col - c))
                )
        return merge(ranges)
    if window:
        raise ValueError("Only matrix extraction windows are supported")
    # Full L0 tiles are contiguous irrespective of their internal permutation.
    if tile["space"] in (3, 4, 5):
        return [(start, start + shape[0] * shape[1] * width)]
    if tile["space"] != 6:
        raise ValueError("Unsupported tile space")
    if tile["column"]:
        return merge(
            (start + width * c * shape[0], start + width * (c * shape[0] + rows))
            for c in range(cols)
        )
    return merge(
        (start + width * r * shape[1], start + width * (r * shape[1] + cols))
        for r in range(rows)
    )


def interpret(text, constants=None):
    source_lines = [
        i
        for i, line in enumerate(text.splitlines(), 1)
        if line.strip() and not line.strip().startswith("//")
    ]
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("//")
    ]
    phase, types = 0, {}
    for i, line in enumerate(lines):
        if re.search(r"pto\.t[\w.]+ ins", line):
            types[i] = phase
            phase += 1
    items, loops = [], []

    def block(begin, end, env):
        i = begin
        while i < end:
            line = lines[i]
            if line.startswith(("scf.for", "scf.if")):
                depth, j, alternative = 1, i + 1, None
                while j < end and depth:
                    if lines[j].startswith(("scf.for", "scf.if")):
                        depth += 1
                    elif lines[j] == "} else {" and depth == 1:
                        alternative = j
                    elif lines[j] == "}":
                        depth -= 1
                    j += 1
                if depth:
                    raise ValueError("Unbalanced supported control")
                if line.startswith("scf.for"):
                    names = values(line)
                    if len(names) != 4 or "iter_args" in line:
                        raise ValueError("Unsupported carried loop")
                    iv, lo, hi, step = names
                    if env[step] <= 0:
                        raise ValueError("Unsupported loop step")
                    for n in range(env[lo], env[hi], env[step]):
                        nested = deepcopy(env)
                        nested[iv] = n
                        loops.append(n)
                        block(i + 1, j - 1, nested)
                        loops.pop()
                elif env[values(line)[0]]:
                    block(
                        i + 1,
                        alternative if alternative is not None else j - 1,
                        deepcopy(env),
                    )
                elif alternative is not None:
                    block(alternative + 1, j - 1, deepcopy(env))
                i = j
                continue
            if line.startswith(("func.func", "module ")) or line in ("return", "}"):
                i += 1
                continue
            lhs, rhs = line.split(" = ", 1) if line.startswith("%") else (None, line)
            args = values(rhs)
            if rhs.startswith("arith.constant"):
                val = rhs.split()[1]
                env[lhs] = float(val) if any(c in val for c in ".eE") else int(val)
            elif rhs.startswith("arith.index_cast"):
                env[lhs] = env[args[0]]
            elif rhs.startswith("arith."):
                op = rhs.split()[0][6:]
                a, b = [env[x] for x in args[:2]]
                if op == "cmpi":
                    pred = rhs.split()[1].rstrip(",")
                    env[lhs] = {
                        "eq": a == b,
                        "ne": a != b,
                        "slt": a < b,
                        "sle": a <= b,
                        "sgt": a > b,
                        "sge": a >= b,
                    }[pred]
                else:
                    if op in ("divsi", "divui", "remsi", "remui") and (a < 0 or b <= 0):
                        raise ValueError(
                            "Oracle only admits nonnegative division operands"
                        )
                    operations = {
                        "addi": lambda: a + b,
                        "subi": lambda: a - b,
                        "muli": lambda: a * b,
                        "divsi": lambda: a // b,
                        "divui": lambda: a // b,
                        "remui": lambda: a % b,
                        "remsi": lambda: a % b,
                        "maxsi": lambda: max(a, b),
                        "minsi": lambda: min(a, b),
                        "andi": lambda: a & b,
                        "ori": lambda: a | b,
                    }
                    if op not in operations:
                        raise ValueError("Unsupported scalar operation " + op)
                    env[lhs] = operations[op]()
            elif rhs.startswith("pto.make_tensor_view"):
                groups = re.findall(r"\[([^]]*)\]", rhs)
                env[lhs] = {
                    "base": int(args[0][4:]),
                    "strides": [env[x] for x in values(groups[1])],
                    "origin": 0,
                    "width": WIDTHS[re.search(r"x(bf16|f32|f16|i32)", rhs)[1]],
                }
            elif rhs.startswith("pto.partition_view"):
                parent = env[args[0]]
                groups = re.findall(r"\[([^]]*)\]", rhs)
                offsets = [env[x] for x in values(groups[0])]
                shape = [env[x] for x in values(groups[1])]
                env[lhs] = {
                    **parent,
                    "origin": parent["origin"]
                    + sum(a * b for a, b in zip(offsets, parent["strides"])),
                    "shape": shape,
                }
            elif rhs.startswith("pto.alloc_tile"):
                match = re.search(
                    r"!pto.tile_buf<(\w+), (\d+)x(\d+)x(bf16|f16|f32|i32)", rhs
                )
                if not match or len(args) != 3:
                    raise ValueError("Unsupported physical tile declaration")
                env[lhs] = {
                    "space": SPACES[match[1]],
                    "shape": [int(match[2]), int(match[3])],
                    "width": WIDTHS[match[4]],
                    "address": env[args[0]],
                    "valid": [env[args[1]], env[args[2]]],
                    "column": "blayout=col_major" in rhs,
                    "subcolumn": "slayout=col_major" in rhs,
                }
            elif rhs.startswith("pto.t"):
                op = rhs.split()[0]
                ins = re.search(r"ins\((.*?)\s*:", rhs)[1]
                outs = re.search(r"outs\((.*?)\s*:", rhs)[1]
                reads, writes = values(ins), values(outs)
                if op not in {
                    "pto.tload",
                    "pto.tstore",
                    "pto.textract",
                    "pto.tmatmul",
                    "pto.tmatmul.acc",
                    "pto.trowexpanddiv",
                    "pto.tcvt",
                    "pto.texpands",
                    "pto.tmul",
                    "pto.tadd",
                    "pto.trowsum",
                    "pto.tmuls",
                    "pto.tadds",
                    "pto.trsqrt",
                    "pto.trowexpandmul",
                    "pto.tcolexpandmul",
                }:
                    raise ValueError("Unsupported payload " + op)
                effects = []

                def access(value, write, window=None, envelope=False):
                    data = env[value]
                    if not isinstance(data, dict):
                        return
                    if "base" in data:
                        indices = itertools.product(*(range(n) for n in data["shape"]))
                        origin, width = data["origin"], data["width"]
                        ranges = merge(
                            (
                                width
                                * (
                                    origin
                                    + sum(a * b for a, b in zip(point, data["strides"]))
                                ),
                                width
                                * (
                                    origin
                                    + sum(a * b for a, b in zip(point, data["strides"]))
                                    + 1
                                ),
                            )
                            for point in indices
                        )
                        space, base = 1, data["base"]
                    else:
                        ranges = tile_ranges(data, window)
                        space, base = data["space"], -1
                    effects.append(
                        {
                            "space": space,
                            "base": base,
                            "mode": "write" if write else "read",
                            "intervals": ranges,
                            "qualification": "supplied-scratch-envelope"
                            if envelope
                            else "supplied-layout",
                        }
                    )

                for n, value in enumerate(reads):
                    if op == "pto.textract" and n == 0:
                        dst = env[writes[0]]
                        access(
                            value, False, [env[reads[1]], env[reads[2]], *dst["valid"]]
                        )
                    elif value in env:
                        access(
                            value,
                            False,
                            envelope=op in ("pto.trowsum", "pto.trsqrt") and n == 1,
                        )
                if op in ("pto.trowsum", "pto.trsqrt"):
                    access(reads[1], True, envelope=True)
                if op == "pto.tstore" and "atomic_add" in rhs:
                    access(writes[0], False)
                for value in writes:
                    access(value, True)
                pipe = {"pto.tload": 4, "pto.textract": 3}.get(op, 0)
                if op == "pto.tstore":
                    pipe = 10 if env[reads[0]]["space"] == 5 else 5
                if op in ("pto.tmatmul", "pto.tmatmul.acc"):
                    pipe = 2
                elif pipe == 0:
                    pipe = 1
                items.append(
                    {
                        "phase": types[i],
                        "coordinates": list(loops),
                        "pipe": pipe,
                        "operation": op,
                        "active": True,
                        "effects": effects,
                    }
                )
            else:
                raise ValueError("Unsupported line " + line)
            if (
                constants is not None
                and lhs
                and rhs.startswith("arith.")
                and isinstance(env.get(lhs), (int, bool))
            ):
                constants.setdefault(source_lines[i], set()).add(int(env[lhs]))
            i += 1

    block(0, len(lines), {})
    return items


def mathematical(items):
    edges = native([x["pipe"] for x in items])
    groups, group = [], 0
    for item in items:
        if item["operation"] == "pto.tmatmul":
            group += 1
        elif (
            any(x["space"] == 5 for x in item["effects"])
            and item["operation"] != "pto.tmatmul.acc"
        ):
            group += 1
        groups.append(group)
    for b, second in enumerate(items):
        for a, first in enumerate(items[:b]):
            for x in first["effects"]:
                for y in second["effects"]:
                    protected = (
                        x["space"] == 5
                        and groups[a] == groups[b]
                        and all(
                            i["operation"].startswith("pto.tmatmul")
                            for i in (first, second)
                        )
                    )
                    if (
                        not protected
                        and (x["mode"] == "write" or y["mode"] == "write")
                        and (x["space"], x["base"]) == (y["space"], y["base"])
                        and any(
                            lo < end and start < hi
                            for lo, hi in x["intervals"]
                            for start, end in y["intervals"]
                        )
                    ):
                        edges.add((2 * a + 1, 2 * b))
    return closure(2 * len(items), edges)
