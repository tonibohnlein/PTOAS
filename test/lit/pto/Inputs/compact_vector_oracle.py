# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent interpreter of the three pinned vector ports' supplied layout.
No PTOAS generators/cells/queries are inputs. Not native footprint validation.
Only row/column-major f32 vector tiles, fixed for/if and listed ops are admitted.
"""

import re
from check_periodic_demands import closure, native
from check_logical_insertion import closure_with_commands
from check_finite_allocation import physical_check

V = r"%[A-Za-z_][A-Za-z_0-9]*"


def variables(s):
    return re.findall(V, s)


def model(text, block=0, subblock=0, *, scalar_payload=None):
    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip() and not x.strip().startswith("//")
    ]
    env = dict(getattr(scalar_payload, "initial_env", {}))
    deps = {}
    items = []
    phase = 0
    types = {}
    loops = []
    controls = []
    for i, line in enumerate(lines):
        if re.search(r"pto\.t\w+ (?:ins|outs)", line) or re.search(r" = pto\.load ", line):
            types[i] = phase
            phase += 1

    def evaluate(begin, end):
        i = begin
        while i < end:
            line = lines[i]
            if line.startswith(("scf.for", "scf.if")):
                level = 1
                j = i + 1
                while j < end and level:
                    if lines[j].startswith(("scf.for", "scf.if")):
                        level += 1
                    elif lines[j] == "}":
                        level -= 1
                    j += 1
                if level:
                    raise ValueError("unbalanced control")
                if line.startswith("scf.for"):
                    iv, lo, hi, step = variables(line)
                    for n in range(env[lo], env[hi], env[step]):
                        env[iv] = n
                        loops.append(n)
                        controls.append(set().union(*(deps.get(v, set()) for v in (lo, hi, step))))
                        evaluate(i + 1, j - 1)
                        controls.pop()
                        loops.pop()
                elif env[variables(line)[0]]:
                    controls.append(deps.get(variables(line)[0], set()))
                    evaluate(i + 1, j - 1)
                    controls.pop()
                i = j
                continue
            if line.startswith(("module ", "func.func ")) or line in ("return", "}"):
                i += 1
                continue
            lhs, rhs = line.split(" = ", 1) if " = " in line else (None, line)
            args = variables(rhs)
            if rhs.startswith("arith.constant"):
                value = rhs.split()[1]
                env[lhs] = float(value) if any(x in value for x in ".e") else int(value)
            elif rhs.startswith("pto.get_block_idx"):
                env[lhs] = block
            elif rhs.startswith("pto.get_subblock_idx"):
                env[lhs] = subblock
            elif rhs.startswith(("arith.index_cast", "arith.fptosi")):
                env[lhs] = int(env[args[0]])
                deps[lhs] = deps.get(args[0], set()).copy()
            elif rhs.startswith("arith."):
                op = rhs.split()[0][6:]
                a, b = [env[x] for x in args[:2]]
                if op == "cmpi":
                    pred = rhs.split()[1].rstrip(",")
                    env[lhs] = {
                        "eq": a == b,
                        "ne": a != b,
                        "ult": a < b,
                        "slt": a < b,
                        "ule": a <= b,
                        "sle": a <= b,
                        "ugt": a > b,
                        "sgt": a > b,
                    }[pred]
                elif op == "cmpf":
                    pred = rhs.split()[1].rstrip(",")
                    env[lhs] = {"ogt": a > b, "oge": a >= b, "olt": a < b, "ole": a <= b, "oeq": a == b}[pred]
                else:
                    env[lhs] = {
                        "addi": lambda: a + b,
                        "muli": lambda: a * b,
                        "subi": lambda: a - b,
                        "divui": lambda: a // b,
                        "remui": lambda: a % b,
                        "addf": lambda: a + b,
                        "mulf": lambda: a * b,
                        "divf": lambda: a / b,
                    }[op]()
                deps[lhs] = set().union(*(deps.get(a, set()) for a in args))
            elif rhs.startswith("pto.make_tensor_view"):
                shape, strides = re.findall(r"\[([^]]+)\]", rhs)[:2]
                env[lhs] = (
                    "gm",
                    args[0],
                    [env[x] for x in variables(strides)],
                    0,
                    None,
                )
            elif rhs.startswith("pto.partition_view"):
                offsets, sizes = re.findall(r"\[([^]]+)\]", rhs)[:2]
                _, base, strides, origin, _ = env[args[0]]
                offset = sum(env[v] * s for v, s in zip(variables(offsets), strides))
                sizes = [env[v] for v in variables(sizes)]
                ranges = [
                    (
                        4 * (origin + offset + r * strides[0] + c * strides[1]),
                        4 * (origin + offset + r * strides[0] + c * strides[1]) + 4,
                    )
                    for r in range(sizes[0])
                    for c in range(sizes[1])
                ]
                env[lhs] = ("gm", base, strides, origin + offset, ranges)
            elif rhs.startswith("pto.alloc_tile"):
                data = dict(
                    re.findall(r"(dtype|rows|cols|v_row|v_col|blayout)=([^,>]+)", rhs)
                )
                if data["dtype"] != "f32":
                    raise ValueError("unsupported tile dtype")
                nr, nc, vr, vc = [
                    int(data[k]) for k in ("rows", "cols", "v_row", "v_col")
                ]
                addr = env[args[0]]
                ranges = [
                    (
                        addr
                        + 4
                        * (
                            r * nc + c if data["blayout"] == "row_major" else c * nr + r
                        ),
                        addr
                        + 4
                        * (r * nc + c if data["blayout"] == "row_major" else c * nr + r)
                        + 4,
                    )
                    for r in range(vr)
                    for c in range(vc)
                ]
                env[lhs] = ("vec", None, None, addr, ranges)
            elif i in types:
                op = rhs.split()[0]
                ins = re.search(r"ins\((.*?)\s*:", rhs)
                outs = re.search(r"outs\((.*?)\s*:", rhs)
                reads = args[:2] if op == "pto.load" else variables(ins[1]) if ins else []
                writes = variables(outs[1]) if outs else []
                effects = []

                def access(value, write=False):
                    val = env.get(value)
                    if not isinstance(val, tuple):
                        return
                    ranges = val[4]
                    if op == "pto.load":
                        ranges = [(val[3] + 4 * env[reads[1]], val[3] + 4 * env[reads[1]] + 4)]
                    elif op in ("pto.tgetval", "pto.tsetval"):
                        index = reads[1] if op == "pto.tgetval" else reads[0]
                        ranges = [
                            (val[3] + 4 * env[index], val[3] + 4 * env[index] + 4)
                        ]
                    # Merge adjacent elements without consulting the compiler's partition.
                    merged = []
                    for lo, hi in sorted(ranges):
                        if merged and lo <= merged[-1][1]:
                            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
                        else:
                            merged.append((lo, hi))
                    effects.append(
                        {
                            "space": val[0],
                            "base": val[1],
                            "write": write,
                            "ranges": merged,
                        }
                    )

                for value in reads:
                    access(value)
                if op == "pto.trowsum" and len(reads) > 1:
                    access(reads[1], True)
                for value in writes:
                    access(value, True)
                control_deps = set().union(*controls) if scalar_payload is not None else set()
                scalar = set().union(*(deps.get(a, set()) for a in reads)) | control_deps
                items.append(
                    {
                        "operation": op,
                        "pipe": 4
                        if op == "pto.tload"
                        else 5
                        if op == "pto.tstore"
                        else 0
                        if op in ("pto.tgetval", "pto.tsetval", "pto.load")
                        else 1,
                        "coordinates": list(loops),
                        "effects": effects,
                        "scalar": scalar,
                        **({"control": control_deps} if scalar_payload is not None else {}),
                        "type": types[i],
                    }
                )
                observed = scalar_payload(op, reads, writes, env) if scalar_payload is not None else 1.0
                if lhs:
                    if observed is None:
                        raise ValueError("missing independent scalar payload value")
                    env[lhs] = observed
                    deps[lhs] = {len(items) - 1}
            elif rhs not in ("return", "}") and not rhs.startswith(
                ("module ", "func.func ")
            ):
                raise ValueError("unsupported original line: " + line)
            i += 1

    evaluate(0, len(lines))
    return items


def required(items):
    edges = native([e["pipe"] for e in items])
    extra = set()
    for b, item in enumerate(items):
        for a in item["scalar"]:
            edges.add((2 * a + 1, 2 * b))
            extra.add((a, b))
        for a, first in enumerate(items[:b]):
            if item["pipe"] == first["pipe"] == 0:
                continue
            if any(
                (x["write"] or y["write"])
                and x["space"] == y["space"]
                and x["base"] == y["base"]
                and any(
                    lo < end and start < hi
                    for lo, hi in x["ranges"]
                    for start, end in y["ranges"]
                )
                for x in first["effects"]
                for y in item["effects"]
            ):
                edges.add((2 * a + 1, 2 * b))
    rows, covers = closure(2 * len(items), edges)
    return rows, covers, extra


def command_order(
    trace, scalar=(), *, include_commands=False, native_serial_pipes=(), command_prerequisites=()
):
    events = trace["events"]
    commands = []
    pending = {}
    counter = 0
    logical = []
    for e in events:
        old = dict(e)
        if e["kind"] == "payload":
            logical.append(old)
            continue
        cmd = dict(e)
        if e["kind"] != "barrier":
            if "physical_id" in e:
                key = (e["source_pipe"], e["target_pipe"], e["physical_id"])
                if e["kind"] == "set":
                    if key in pending:
                        raise AssertionError("flag republished before wait")
                    pending[key] = counter
                    counter += 1
                ident = pending[key]
                if e["kind"] == "wait":
                    del pending[key]
                old.update(plan=0, record=ident, source_ordinal=0, members=[])
            cmd["identity"] = (
                old["plan"],
                old["record"],
                old["source_ordinal"],
                tuple(old.get("members", [])),
            )
        commands.append(cmd)
        logical.append(old)
    if pending:
        raise AssertionError("unconsumed flags")
    pipes = [e["pipe"] for e in events if e["kind"] == "payload"]
    # Scalar prerequisites are supplied as native edges by the test interpreter.
    # The independent source model checks their values separately.
    native_pairs = [
        (a, b)
        for b, e in enumerate(e for e in events if e["kind"] == "payload")
        for a, native in e.get("prerequisites", [])
        if native
    ]
    rows = closure_with_commands(
        pipes, commands, include_commands=True, native_serial_pipes=native_serial_pipes,
        command_prerequisites=command_prerequisites
    )
    edges = {
        (a, b) for a, row in enumerate(rows) for b in range(len(rows)) if row >> b & 1
    }
    edges.update((2 * a + 1, 2 * b) for a, b in native_pairs)
    edges.update((2 * a + 1, 2 * b) for a, b in scalar)
    rows = [0] * len(rows)
    for a, b in edges:
        rows[a] |= 1 << b
    for k in range(len(rows)):
        for a in range(len(rows)):
            if rows[a] >> k & 1:
                rows[a] |= rows[k]
    if not include_commands:
        rows = rows[: 2 * len(pipes)]
    reuse = {"status": "not_requested"}
    if any("physical_id" in e for e in events if e["kind"] == "set"):
        try:
            physical_check(
                {
                    "accepted": True,
                    "allocated": True,
                    "trace": {**trace, "events": logical},
                    "physical": trace,
                },
                set(range(6)),
                native_serial_pipes=native_serial_pipes,
            )
            reuse = {"status": "passed"}
        except AssertionError as error:
            reuse = {"status": "unqualified", "diagnostic": str(error)}
    return rows, reuse
