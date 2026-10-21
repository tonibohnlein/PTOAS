# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bounded original integer control/GM views; vector arithmetic is not interpreted."""

import re
from copy import deepcopy

VALUE = r"%[A-Za-z_0-9]+"


def names(text):
    return re.findall(VALUE, text)


def source_gm_accesses(text):
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("//")
    ]
    result = []

    def scalar(rhs, env):
        args = names(rhs)
        op = rhs.split()[0]
        if op == "arith.constant":
            return (
                float(rhs.split()[1])
                if "f16" in rhs or "f32" in rhs
                else int(rhs.split()[1])
            )
        if op == "arith.index_cast":
            return env[args[0]]
        a, b = [env[x] for x in args[:2]]
        if op == "arith.cmpi":
            pred = rhs.split()[1].rstrip(",")
            return {
                "ne": a != b,
                "eq": a == b,
                "slt": a < b,
                "sgt": a > b,
                "sge": a >= b,
                "sle": a <= b,
            }[pred]
        if op in ("arith.divui", "arith.remui") and (a < 0 or b <= 0):
            raise ValueError(
                "reference requires qualified nonnegative unsigned operands"
            )
        operations = {
            "arith.addi": lambda: a + b,
            "arith.subi": lambda: a - b,
            "arith.muli": lambda: a * b,
            "arith.divui": lambda: a // b,
            "arith.remui": lambda: a % b,
            "arith.andi": lambda: bool(a and b),
        }
        if op not in operations:
            raise ValueError(f"unsupported prefill integer operation {op}")
        return operations[op]()

    def block(begin, end, env, coordinates=()):
        at = begin
        while at < end:
            line = lines[at]
            if line.startswith(("scf.for", "scf.if")):
                depth, stop, otherwise = 1, at + 1, None
                while depth:
                    current = lines[stop]
                    if current.startswith(("scf.for", "scf.if")):
                        depth += 1
                    elif current == "} else {" and depth == 1:
                        otherwise = stop
                    elif current == "}":
                        depth -= 1
                    stop += 1
                operands = names(line)
                if line.startswith("scf.for"):
                    iv, lower, upper, step = operands
                    for value in range(env[lower], env[upper], env[step]):
                        nested = deepcopy(env)
                        nested[iv] = value
                        block(at + 1, stop - 1, nested, coordinates + (value,))
                elif env[operands[0]]:
                    block(
                        at + 1,
                        otherwise if otherwise is not None else stop - 1,
                        deepcopy(env),
                        coordinates,
                    )
                elif otherwise is not None:
                    block(otherwise + 1, stop - 1, deepcopy(env), coordinates)
                at = stop
                continue
            lhs, rhs = line.split(" = ", 1) if line.startswith("%") else (None, line)
            args = names(rhs)
            if rhs.startswith("arith."):
                env[lhs] = scalar(rhs, env)
            elif rhs.startswith("pto.make_tensor_view"):
                arrays = re.findall(r"\[([^]]*)\]", rhs)
                env[lhs] = {
                    "root": args[0],
                    "shape": [env[x] for x in names(arrays[0])],
                    "strides": [env[x] for x in names(arrays[1])],
                    "offsets": [0, 0],
                }
            elif rhs.startswith("pto.partition_view"):
                arrays = re.findall(r"\[([^]]*)\]", rhs)
                parent = env[args[0]]
                env[lhs] = {
                    **parent,
                    "offsets": [env[x] for x in names(arrays[0])],
                    "sizes": [env[x] for x in names(arrays[1])],
                }
            elif rhs.startswith(("pto.tload ", "pto.tstore ")):
                view_name = args[0] if rhs.startswith("pto.tload ") else args[1]
                view = env[view_name]
                if not all(
                    0 <= off and off + size <= dim
                    for off, size, dim in zip(
                        view["offsets"], view["sizes"], view["shape"]
                    )
                ):
                    raise ValueError(f"original GM view exceeds supplied shape: {view}")
                result.append(
                    {
                        "write": rhs.startswith("pto.tstore "),
                        "view": view,
                        "coordinates": coordinates,
                    }
                )
            elif rhs.startswith("pto."):
                # Metadata loads/getvals have explicit independently checked
                # trace bindings immediately after their preserved observation.
                # No opaque floating result is used to decide original control.
                pass
            at += 1

    block(0, len(lines), {})
    return result
