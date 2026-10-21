# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent scalar layouts and event oracle for indexed finite-loop checks."""

import re
from compact_persistent_oracle import boundary
from check_periodic_demands import native, closure


def source(root, name, n, l=8):
    s = (
        (
            root
            / "test/npu_validation/benchmarks/auto_sync/prepared"
            / name
            / "input.pto"
        )
        .read_text()
        .replace("pto.get_block_idx", "arith.constant 0 : i64")
        .replace("pto.get_subblock_idx", "arith.constant 0 : i64")
    )
    if name == "lossless_block_cast":
        s = (
            s.replace("rows=64", f"rows={n}")
            .replace("v_row=64", f"v_row={n}")
            .replace("<64x", f"<{n}x")
            .replace("constant 64 : index", f"constant {n} : index")
        )
    else:
        s = (
            s.replace("constant 8 : index", f"constant {n} : index")
            .replace("constant 128 : index", f"constant {l} : index")
            .replace("1024", str(n * l))
        )
    return s


def words(kind, n, l):
    result = []

    def add(phase, coords, write, address):
        result.append(
            {
                "phase": phase,
                "coordinates": list(coords),
                "pipe": 0,
                "effects": [
                    {
                        "space": 6,
                        "base": -1,
                        "mode": "write" if write else "read",
                        "intervals": [(address, address + 4)],
                    }
                ],
            }
        )

    if kind == "cumsum":
        for c in range(n):
            add(2, (c,), False, 4 * (c * l))
            add(3, (c,), True, 4096 + 4 * c * l)
            for j in range(1, l):
                add(4, (c, j), False, 4096 + 4 * (c * l + j - 1))
                add(5, (c, j), False, 4 * (c * l + j))
                add(6, (c, j), True, 4096 + 4 * (c * l + j))
    elif kind in ("reduce", "correct"):
        for i in range(n):
            for j in range(4):
                if kind == "reduce":
                    add(6, (i, j), False, 86016 + 4 * (j // 2))
                    add(7, (i, j), False, 2048 + 4 * (8 * i + j))
                    add(8, (i, j), True, 86016 + 4 * (j // 2))
                else:
                    add(15, (i, j), False, 2048 + 4 * (8 * i + j))
                    add(16, (i, j), False, 86016 + 4 * (j // 2))
                    add(17, (i, j), True, 2048 + 4 * (8 * i + j))
    else:
        for i in range(n):
            for j in range(128):
                add(18, (i, j), False, 2048 + 4 * (8 * i + j // 32))
                add(19, (i, j), True, 53248 + 4 * (128 * i + j))
    return result


def constants(s):
    env = {}
    rows = []
    for number, line in enumerate(s.splitlines(), 1):
        m = re.search(r"(%\w+) = (arith\..*)", line)
        if not m:
            continue
        name, rhs = m.groups()
        v = re.findall(r"%\w+", rhs)
        try:
            if rhs.startswith("arith.constant") and not any(
                x in rhs for x in ("f32", "f16")
            ):
                value = int(rhs.split()[1])
            elif rhs.startswith("arith.index_cast"):
                value = env[v[0]]
            elif rhs.startswith("arith.cmpi eq"):
                value = int(env[v[0]] == env[v[1]])
            elif rhs.startswith("arith.addi"):
                value = env[v[0]] + env[v[1]]
            elif rhs.startswith("arith.muli"):
                value = env[v[0]] * env[v[1]]
            elif rhs.startswith("arith.divui"):
                value = env[v[0]] // env[v[1]]
            elif rhs.startswith("arith.remui"):
                value = env[v[0]] % env[v[1]]
            else:
                continue
        except KeyError:
            continue
        env[name] = value
        rows.append(f"array<i64: {number}, {value}>")
    return "test.contract_constants = [" + ", ".join(rows) + "]"


def check_queries(d, items, full, cells):
    if d.get("error"):
        return {"status": "unavailable", "reason": d["error"]}
    # Same-PIPE_S storage conflicts are native under the supplied scalar contract.
    # SSA readiness is independent of storage extraction.
    edges = native([0] * len(full))
    for i, item in enumerate(full):
        if item["effects"][0]["mode"] == "write":
            if i >= 1:
                edges.add((2 * (i - 1) + 1, 2 * i))
            if i >= 2 and full[i - 2]["effects"][0]["mode"] == "read":
                edges.add((2 * (i - 2) + 1, 2 * i))
    rows, _ = closure(2 * len(full), edges)
    positions = [full.index(x) for x in items]
    rows = [
        sum(
            ((rows[2 * i + k] >> (2 * j + h)) & 1) << (2 * b + h)
            for b, j in enumerate(positions)
            for h in (0, 1)
        )
        for i in positions
        for k in (0, 1)
    ]
    actual = d.get("event_rows", [])
    want = [
        "".join("1" if row >> j & 1 else "0" for j in range(len(rows))) for row in rows
    ]
    result = {
        "status": "checked",
        "all_four_queries_equal": actual == want,
        "retained_covers_equal": (
            d.get("retained_demands") == [] and not d.get("retained_demand_error")
        ),
        "retained_cover_error": d.get("retained_demand_error"),
    }
    b = d.get("boundaries", {})
    result["selector_error"] = b.get("error")
    if not b.get("error"):
        expected = []
        for cell in cells:
            row = boundary([dict(x, active=True) for x in full], cell)
            for key in ("first_writers", "last_writers"):
                row[key] = [positions.index(i) for i in row[key]]
            for key in ("first_readers", "last_readers"):
                row[key] = {
                    p: [positions.index(i) for i in values]
                    for p, values in row[key].items()
                    if values
                }
            expected.append(row)
        actual = b["rows"]
        for row in actual:
            for key in ("first_readers", "last_readers"):
                row[key] = {p: v for p, v in row[key].items() if v}
        result["selectors_equal"] = actual == expected
    return result
