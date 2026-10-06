# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check late same-pipe barriers against an independent order oracle."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native

PREFIX = """!tile = !pto.tile_buf<vec, 1x8xf32>
module attributes {pto.target_arch = "a3"} {
 func.func @local(%n: index, %g: i1)
 attributes {test.trace_arguments = array<i64: TRIPS, GUARD>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %value = arith.constant 1.0 : f32
  %x = arith.constant 0 : i64
  %y = arith.constant 1024 : i64
  %a = pto.alloc_tile addr = %x : !tile
  %b = pto.alloc_tile addr = %y : !tile
"""
A = 'pto.tsetval ins(%zero, %value : index, f32) outs(%a : !tile) {test.label = "A"}'
X = 'pto.tsetval ins(%zero, %value : index, f32) outs(%b : !tile) {test.label = "X"}'
B = '%v = pto.tgetval ins(%a, %zero : !tile, index) outs : f32 {test.label = "B"}'


def loop(body):
    return f"scf.for %i = %zero to %n step %one {{\n{body}\n}}"


def validate(report):
    assert report["accepted"], report
    trace = report["trace"]
    assert not trace["error"], trace
    payloads = [x for x in trace["events"] if x["kind"] == "payload"]
    pipes = [x["pipe"] for x in payloads]
    effects = {"A": ({}, {"x"}), "X": ({}, {"y"}), "B": ({"x"}, {})}
    edges = native(pipes)
    for i, a in enumerate(payloads):
        reads, writes = (set(s) for s in effects[a["label"]])
        for j in range(i + 1, len(payloads)):
            rr, ww = (set(s) for s in effects[payloads[j]["label"]])
            if writes & (rr | ww) or reads & ww:
                edges.add((2 * i + 1, 2 * j))
    _, covers = closure(2 * len(payloads), edges)
    targets = {b // 2 for a, b in covers}
    # A barrier immediately before b completes every preceding payload on b's
    # pipe, including unrelated work between the demand's endpoints.
    for b in targets:
        edges.update((2 * a + 1, 2 * b) for a in range(b) if pipes[a] == pipes[b])
    expected, _ = closure(2 * len(payloads), edges)
    commands = [x for x in trace["events"] if x["kind"] != "payload" and
                not (x["kind"] == "barrier" and x["pipe"] == 6)]
    assert all(x["kind"] == "barrier" for x in commands), commands
    assert {x["gap"] for x in commands} == targets, (commands, targets)
    actual = closure_with_commands(pipes, commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(expected)], (actual, expected)


def main():
    tool = sys.argv[1]
    cases = [
        "\n".join([A, X, B]),
        A + f"\nscf.if %g {{ {X} }}\n" + B,
        A + "\n" + loop(X) + "\n" + B,
        loop("\n".join([A, X, B])),
        loop(f"scf.if %g {{ {A} }}\n{X}\n{B}"),
        loop(f"%first = arith.cmpi eq, %i, %zero : index\nscf.if %first {{ {A} }}\n{X}\n{B}"),
    ]
    count = 0
    with tempfile.TemporaryDirectory(prefix="late-barrier-") as directory:
        path = Path(directory) / "case.pto"
        for case, body in enumerate(cases):
            for n in (0, 1, 3):
                for guard in (0, 1):
                    path.write_text(PREFIX.replace("TRIPS", str(n)).replace("GUARD", str(guard)) +
                                    body + "\nreturn\n}\n}\n")
                    run = subprocess.run([tool, "--structured-trace", str(path)], capture_output=True,
                                         text=True, timeout=45)
                    assert run.returncode == 0, (case, run.stderr, run.stdout)
                    validate(json.loads(run.stdout))
                    count += 1
    print(f"late barriers: {count} explicit, guarded, periodic, arithmetic and regional traces passed")


if __name__ == "__main__":
    main()
