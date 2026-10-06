# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare compact repeated-region insertion with an independently unfolded DAG."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def run(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout


def check(document, dimensions):
    assert document["accepted"], document
    trace = document["trace"]
    assert not trace["error"], trace
    events = trace["events"]
    payloads = [event for event in events if event["kind"] == "payload"]
    expected = [(0, [])]
    for coordinates in itertools.product(*(range(max(0, count)) for count in dimensions)):
        expected.extend([(1, list(coordinates)), (2, list(coordinates))])
    expected.append((3, []))
    assert [(p["type"], p["coordinates"]) for p in payloads] == expected
    effects = []
    for kind, coordinates in expected:
        slot = coordinates[-1] % 2 if coordinates else 0
        effects.append(({("left", slot), ("right", 0)}, {("acc", 0)}) if kind == 2 else
                       ({("mat", 0)}, {("left", slot)}))
    pipes = [p["pipe"] for p in payloads]
    edges = native(pipes)
    for a, (reads, writes) in enumerate(effects):
        for b in range(a+1, len(effects)):
            later_reads, later_writes = effects[b]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2*a+1, 2*b))
    required = closure(2*len(effects), edges)[0]
    commands = []
    for event in events:
        if event["kind"] == "payload":
            continue
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert event["gap"] == len(payloads), "unexpected region drain"
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], (dimensions, document)


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    tested = 0
    with tempfile.TemporaryDirectory(prefix="repeated-region-") as scratch:
        path = Path(scratch) / "case.pto"
        for n, m in ((0, 0), (0, 3), (2, 0), (1, 1), (2, 1), (2, 2), (2, 3), (3, 4), (2, 5)):
            path.write_text(source.replace("array<i64: 2, 3>", f"array<i64: {n}, {m}>"))
            check(json.loads(run(tool, "--structured-trace", path)), [n, m])
            tested += 1
        path.write_text(source)
        report = json.loads(run(tool, "--sequence-analysis", path))
        assert not report["error"] and report["prepared"] and report["numeric_visits"] == 0, report
        small = run(tool, "--insert-logical", path)
        path.write_text(source.replace("array<i64: 2, 3>", "array<i64: 1000000000, 1000000001>"))
        large = run(tool, "--insert-logical", path)
        assert small == large.replace("1000000000, 1000000001", "2, 3")
        assert small.count("scf.for") == 2 and "version = 4" in small
        # Keep the old allocatable regional numerical path with surrounding work.
        path.write_text(source.replace("%i = %zero to %m", "%i = %zero to %two"))
        fixed = json.loads(run(tool, "--sequence-analysis", path))
        assert not fixed["error"] and fixed["prepared"] and fixed["numeric_visits"] > 0, fixed
        assert "version = 4" not in run(tool, "--insert-logical", path)
        folded = source.replace("%i = %zero to %m", "%i = %zero to %folded")
        folded = folded.replace("    scf.for %visit", "    %folded = arith.addi %one, %one : index\n    scf.for %visit")
        path.write_text(folded)
        folded_report = json.loads(run(tool, "--sequence-analysis", path))
        assert not folded_report["error"] and folded_report["numeric_visits"] > 0, folded_report
        assert "version = 4" not in run(tool, "--insert-logical", path)
        triangular = source.replace("      scf.for %i",
                                    "      scf.for %middle = %zero to %two step %one {\n      scf.for %i")
        triangular = triangular.replace("%i = %zero to %m", "%i = %zero to %middle")
        triangular = triangular.replace("    }\n    pto.textract", "      }\n    }\n    pto.textract")
        path.write_text(triangular)
        triangular_report = json.loads(run(tool, "--sequence-analysis", path))
        assert not triangular_report["error"] and triangular_report["numeric_visits"] > 0, triangular_report
        assert "version = 4" not in run(tool, "--insert-logical", path)
        # Outer-dependent bounds, bank selection, and descriptor rebinding do
        # not satisfy invariant-body repetition. Rejection is of this route.
        negatives = [source.replace("%i = %zero to %m", "%i = %zero to %visit"),
                     source.replace("arith.remui %i, %two", "arith.remui %visit, %two"),
                     source.replace("      scf.for %i", "      %address = arith.index_cast %visit : index to i64\n"
                         "      %rebound = pto.tassign %acc, %address : !acc -> !acc\n      scf.for %i")]
        for candidate in negatives:
            path.write_text(candidate)
            rejected = json.loads(run(tool, "--sequence-analysis", path))
            assert rejected["error"], rejected
        triple = source.replace("%n: index, %m: index", "%p: index, %n: index, %m: index")
        triple = triple.replace("    scf.for %visit",
                                "    scf.for %outer = %zero to %p step %one {\n    scf.for %visit")
        triple = triple.replace("    }\n    pto.textract", "    }\n    }\n    pto.textract")
        for p, n, m in ((0, 2, 3), (2, 0, 3), (2, 2, 0), (2, 2, 3), (3, 1, 2)):
            path.write_text(triple.replace("array<i64: 2, 3>", f"array<i64: {p}, {n}, {m}>"))
            check(json.loads(run(tool, "--structured-trace", path)), [p, n, m])
            tested += 1
    print(f"repeated regions: {tested} nested command closures, zero trips and compact size passed")


if __name__ == "__main__":
    main()
