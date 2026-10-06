# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare emitted compact/guarded orders with a nontransitive alias model."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native

PREFIX = """!tile = !pto.tile_buf<vec, 4x8xf32>
!unknown = !pto.tile_buf<vec, 4x8xf32, valid=?x?>
module attributes {pto.target_arch = "a3"} {
 func.func @modeled(%address: i64, %rows: index, %n: index, %g: i1)
 attributes {test.trace_arguments = array<i64: 64, 2, TRIPS, GUARD>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %cols = arith.constant 8 : index
  %value = arith.constant 1.0 : f32
  %x = arith.constant 1024 : i64
  %y = arith.constant 2048 : i64
  %U = pto.alloc_tile addr = %address valid_row = %rows valid_col = %cols : !unknown
  %A = pto.alloc_tile addr = %x : !tile
  %B = pto.alloc_tile addr = %y : !tile
"""
OPS = {
    "U": 'pto.tadds ins(%U, %value : !unknown, f32) outs(%U : !unknown) {test.label = "U"}',
    "A": 'pto.tsetval ins(%zero, %value : index, f32) outs(%A : !tile) {test.label = "A"}',
    "B": 'pto.tsetval ins(%zero, %value : index, f32) outs(%B : !tile) {test.label = "B"}',
}


def loop(body, iv="i"):
    return f"scf.for %{iv} = %zero to %n step %one {{\n{body}\n}}"


def validate(report, labels):
    assert report["accepted"], report
    trace = report["trace"]
    assert not trace["error"], trace
    payloads = [item for item in trace["events"] if item["kind"] == "payload"]
    assert [p["label"] for p in payloads] == labels, payloads
    pipes = [p["pipe"] for p in payloads]
    edges = native(pipes)
    # Every site writes. A and B are disjoint; U may overlap both but never
    # turns their two ranges into a single storage class.
    for a, x in enumerate(labels):
        for b in range(a + 1, len(labels)):
            if x == labels[b] or "U" in (x, labels[b]):
                edges.add((2 * a + 1, 2 * b))
    required, _ = closure(2 * len(labels), edges)
    commands = []
    for event in trace["events"]:
        if event["kind"] == "payload" or (event["kind"] == "barrier" and event["pipe"] == 6):
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], (labels, actual, required)


def main():
    tool = sys.argv[1]
    word = "\n".join(OPS.values())
    cases = [
        (f'scf.if %g {{ {OPS["U"]} }}\n{OPS["A"]}\n{OPS["B"]}',
         lambda n, g: (["U"] if g else []) + ["A", "B"]),
        (loop(word), lambda n, g: list("UAB") * n),
        (OPS["U"] + "\n" + loop(OPS["A"]) + "\n" + OPS["B"],
         lambda n, g: ["U"] + ["A"] * n + ["B"]),
        (loop(OPS["A"]) + "\n" + loop(OPS["B"], "j") + "\n" + OPS["U"],
         lambda n, g: ["A"] * n + ["B"] * n + ["U"]),
        (loop(OPS["U"]) + f'\nscf.if %g {{ {OPS["A"]} }}\n{OPS["B"]}',
         lambda n, g: ["U"] * n + (["A"] if g else []) + ["B"]),
    ]
    count = 0
    with tempfile.TemporaryDirectory(prefix="modeled-composition-") as directory:
        path = Path(directory) / "case.pto"
        for body, expected in cases:
            for n in (0, 1, 2, 3):
                for g in (0, 1):
                    path.write_text(PREFIX.replace("TRIPS", str(n)).replace("GUARD", str(g)) +
                                    body + "\nreturn\n}\n}\n")
                    run = subprocess.run([tool, "--structured-trace", str(path)], capture_output=True,
                                         text=True, timeout=45)
                    assert run.returncode == 0, run.stderr
                    validate(json.loads(run.stdout), expected(n, g))
                    count += 1
    print(f"modeled composition: {count} independent guarded/periodic/regional traces passed")


if __name__ == "__main__":
    main()
