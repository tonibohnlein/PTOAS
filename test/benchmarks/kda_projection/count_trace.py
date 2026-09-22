# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Count explicit executed PTO instructions using original scalar control.

This is an instruction-population diagnostic, NOT a memory/queue correctness
oracle or latency model. PTO queue operations are counted as opaque operations;
their lowering-internal instructions are not included.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "oahs"))
from check_carried_slot_trace import parse


def execute(nodes, env, counts, trace=None):
    for line, body, other in nodes:
        loop = re.search(r"scf.for (%\w+) = (%\w+) to (%\w+) step (%\w+)", line)
        if loop:
            assert "iter_args" not in line, "unsupported carried scalar"
            iv, lower, upper, step = loop.groups()
            for value in range(env[lower], env[upper], env[step]):
                execute(body, {**env, iv: value}, counts, trace)
            continue
        branch = re.search(r"scf.if (%\w+)", line)
        if branch:
            assert " = scf.if" not in line, "unsupported scalar result"
            execute(body if env[branch[1]] else other, dict(env), counts, trace)
            continue
        if not line or line.startswith("//") or line in ("return", "scf.yield"):
            continue
        flag = re.search(
            r"pto.(set_flag|wait_flag)\[<PIPE_(\w+)>, <PIPE_(\w+)>, <EVENT_ID(\d+)>\]",
            line,
        )
        if flag:
            counts[f"{flag[1]}:{flag[2]}->{flag[3]}"] += 1
            if trace is not None:
                trace.sync(flag[1], flag[2], flag[3], int(flag[4]))
            continue
        barrier = re.search(r"pto.barrier <PIPE_(\w+)>", line)
        if barrier:
            counts[f"barrier:{barrier[1]}"] += 1
            if trace is not None:
                trace.sync("barrier", barrier[1])
            continue
        operation = re.search(r"\bpto\.(\w+(?:\.acc)?)", line)
        if operation:
            counts["pto:" + operation[1]] += 1
            if trace is not None:
                trace.operation(operation[1])
            continue
        assert " = " in line, line
        name, expression = line.split(" = ", 1)
        operands = re.findall(r"%\w+", expression.split(":")[0])
        constant = re.match(r"arith.constant (-?\d+)", expression)
        if constant:
            env[name] = int(constant[1])
        elif expression.startswith("arith.index_cast"):
            env[name] = env[operands[0]]
        elif expression.startswith("arith.cmpi"):
            a, b = (env[x] for x in operands)
            predicate = expression.split()[1].rstrip(",")
            env[name] = {
                "eq": a == b,
                "ne": a != b,
                "slt": a < b,
                "sle": a <= b,
                "sgt": a > b,
                "sge": a >= b,
            }[predicate]
        else:
            a, b = (env[x] for x in operands)
            op = expression.split()[0]
            if op == "arith.addi":
                env[name] = a + b
            elif op == "arith.andi":
                env[name] = a & b
            elif op == "arith.subi":
                env[name] = a - b
            elif op == "arith.muli":
                env[name] = a * b
            elif op == "arith.divsi":
                env[name] = a // b
            elif op == "arith.maxsi":
                env[name] = max(a, b)
            elif op == "arith.minsi":
                env[name] = min(a, b)
            else:
                raise AssertionError(expression)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--function", required=True)
    parser.add_argument("--bindings", required=True, help="JSON scalar argument map")
    args = parser.parse_args()
    lines = re.sub(r"%c-(\d+)", r"%cneg\1", args.input.read_text()).splitlines()
    start = (
        next(
            i for i, line in enumerate(lines) if f"func.func @{args.function}(" in line
        )
        + 1
    )
    nodes, _ = parse(lines, start)
    counts = Counter()
    bindings = json.loads(args.bindings)
    execute(nodes, bindings.copy(), counts)
    print(
        json.dumps(
            dict(
                input=str(args.input),
                function=args.function,
                bindings=bindings,
                scope="Explicit executed PTO operations only; queue lowering opaque",
                counts=dict(sorted(counts.items())),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
