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
import shutil
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
  %z = arith.constant 2048 : i64
  %a = pto.alloc_tile addr = %x : !tile
  %b = pto.alloc_tile addr = %y : !tile
  %c = pto.alloc_tile addr = %z : !tile
"""
A = 'pto.tsetval ins(%zero, %value : index, f32) outs(%a : !tile) {test.label = "A"}'
X = 'pto.tsetval ins(%zero, %value : index, f32) outs(%b : !tile) {test.label = "X"}'
B = '%v = pto.tgetval ins(%a, %zero : !tile, index) outs : f32 {test.label = "B"}'


def loop(body):
    return f"scf.for %i = %zero to %n step %one {{\n{body}\n}}"


def expected_payloads(case, n, guard):
    if case == 0:
        return [(label, []) for label in ("A", "X", "B")]
    if case == 1:
        return [(label, []) for label in (["A", "X", "B"] if guard else ["A", "B"])]
    if case == 2:
        return [("A", [])] + [("X", [i]) for i in range(n)] + [("B", [])]
    labels = []
    for i in range(n):
        if case == 3 or (case == 4 and guard) or (case == 5 and i == 0):
            labels.append(("A", [i]))
        labels.extend([("X", [i]), ("B", [i])])
    return labels


def validate(report, expected_payloads, vector):
    assert report["accepted"], report
    trace = report["trace"]
    assert not trace["error"], trace
    payloads = [x for x in trace["events"] if x["kind"] == "payload"]
    assert [(x["label"], x["coordinates"]) for x in payloads] == expected_payloads, payloads
    assert trace["payloads"] == len(expected_payloads)
    pipes = [x["pipe"] for x in payloads]
    assert all(pipe == (1 if vector else 0) for pipe in pipes), pipes
    effects = {"A": (set(), {"x"}), "X": (set(), {"y"}),
               "B": ({"x"}, {"z"} if vector else set())}
    edges = native(pipes)
    protected, raw_hazards = set(), []
    for i, a in enumerate(payloads):
        reads, writes = (set(s) for s in effects[a["label"]])
        for j in range(i + 1, len(payloads)):
            rr, ww = (set(s) for s in effects[payloads[j]["label"]])
            hazards = {kind for kind, overlap in (("RAW", writes & rr),
                                                  ("WAR", reads & ww), ("WAW", writes & ww)) if overlap}
            if not hazards:
                continue
            raw_hazards.append((i, j, hazards))
            # These original tsetval/tgetval operations are on target PIPE_S:
            # all three same-scalar storage hazard kinds are hardware-protected.
            # The vector variant has no such protection and retains every edge.
            if pipes[i] == pipes[j] == 0:
                protected.update(hazards)
            else:
                edges.add((2 * i + 1, 2 * j))
    _, covers = closure(2 * len(payloads), edges)
    targets = {b // 2 for a, b in covers}
    # A barrier immediately before b completes every preceding payload on b's
    # pipe, including unrelated work between the demand's endpoints.
    for b in targets:
        edges.update((2 * a + 1, 2 * b) for a in range(b) if pipes[a] == pipes[b])
    expected, _ = closure(2 * len(payloads), edges)
    assert all(x["gap"] == len(payloads) for x in trace["events"]
               if x["kind"] == "barrier" and x["pipe"] == 6), trace
    commands = [x for x in trace["events"] if x["kind"] != "payload" and
                not (x["kind"] == "barrier" and x["pipe"] == 6)]
    assert all(x["kind"] == "barrier" and x["pipe"] == (1 if vector else 0) for x in commands), commands
    assert {x["gap"] for x in commands} == targets and len(commands) == len(targets), (commands, targets)
    actual = closure_with_commands(pipes, commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(expected)], (actual, expected)
    # LifetimeScan.h StorageProtectionPolicy::protectsScalar discharges storage
    # hazards only when both original pipes are PIPE_S; it grants no C/I order.
    # Keep every raw conflict in this safety ledger, including protected ones,
    # without inventing completion-to-issue edges in the command graph above.
    for source, target, hazards in raw_hazards:
        hardware_safe = pipes[source] == pipes[target] == 0
        command_ordered = bool(actual[2 * source + 1] & (1 << (2 * target)))
        assert hardware_safe or command_ordered, (source, target, hazards, trace)
    return protected, len(commands)


def cases(a, x, b):
    return [
        "\n".join([a, x, b]),
        a + f"\nscf.if %g {{ {x} }}\n" + b,
        a + "\n" + loop(x) + "\n" + b,
        loop("\n".join([a, x, b])),
        loop(f"scf.if %g {{ {a} }}\n{x}\n{b}"),
        loop(f"%first = arith.cmpi eq, %i, %zero : index\nscf.if %first {{ {a} }}\n{x}\n{b}"),
    ]


def main():
    tool = shutil.which(sys.argv[1])
    assert tool
    tool = str(Path(tool).resolve(strict=True))
    vector_ops = ('pto.texpands ins(%value : f32) outs(%a : !tile) {test.label = "A"}',
                  'pto.texpands ins(%value : f32) outs(%b : !tile) {test.label = "X"}',
                  'pto.tadds ins(%a, %value : !tile, f32) outs(%c : !tile) {test.label = "B"}')
    count, vector_barriers, scalar_hazards = 0, 0, set()
    with tempfile.TemporaryDirectory(prefix="late-barrier-") as directory:
        path = Path(directory) / "case.pto"
        for vector, operations in ((False, (A, X, B)), (True, vector_ops)):
            for case, body in enumerate(cases(*operations)):
                for n in (0, 1, 3):
                    for guard in (0, 1):
                        path.write_text(PREFIX.replace("TRIPS", str(n)).replace("GUARD", str(guard)) +
                                        body + "\nreturn\n}\n}\n")
                        run = subprocess.run([tool, "--structured-trace", str(path)], capture_output=True,
                                             text=True, check=False, timeout=45)
                        assert run.returncode == 0, (vector, case, run.stderr, run.stdout)
                        protected, barriers = validate(json.loads(run.stdout), expected_payloads(case, n, guard),
                                                       vector)
                        if vector:
                            vector_barriers += barriers
                            if case in (0, 1, 2) or (n > 0 and (case in (3, 5) or guard)):
                                assert barriers > 0, (case, n, guard)
                        else:
                            scalar_hazards.update(protected)
                            assert barriers == 0
                        count += 1
    assert scalar_hazards == {"RAW", "WAR", "WAW"} and vector_barriers > 0
    print(f"late barriers: {count} protected-scalar/unprotected-vector traces passed")


if __name__ == "__main__":
    main()
