# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise production owner recognition and compare emitted logical closure."""
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


def check(document, outer, inner):
    assert document["accepted"], document
    trace = document["trace"]
    assert not trace["error"], trace
    payloads = [event for event in trace["events"] if event["kind"] == "payload"]
    expected, effects = [], []
    for visit in range(outer):
        for iteration in range(inner):
            expected.extend([(0, [visit, iteration]), (1, [visit, iteration])])
            effects.extend([({("mat", 0)}, {("left", iteration % 2)}),
                            ({("left", iteration % 2), ("right", 0)}, {("acc", 0)})])
        expected.append((2, [visit]))
        effects.append((set(), {("gm", visit * 8 + byte) for byte in range(4)}))
    assert [(value["type"], value["coordinates"]) for value in payloads] == expected
    pipes = [value["pipe"] for value in payloads]
    edges = native(pipes)
    for a, (reads, writes) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            later_reads, later_writes = effects[b]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * a + 1, 2 * b))
    required = closure(2 * len(effects), edges)[0]
    commands = []
    for event in trace["events"]:
        if event["kind"] == "payload":
            continue
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert event["gap"] == len(payloads)
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    assert closure_with_commands(pipes, commands) == [row & ~(1 << i) for i, row in enumerate(required)]


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    cases = ((0, 0), (0, 3), (1, 0), (3, 0), (1, 1), (2, 3), (3, 4))
    with tempfile.TemporaryDirectory(prefix="repeated-storage-") as scratch:
        path = Path(scratch) / "case.pto"
        path.write_text(source)
        report = json.loads(run(tool, "--sequence-analysis", path))
        assert not report["error"] and report["prepared"] and report["numeric_visits"] == 0, report
        assert report["symbolic_storage_effects"] > 0 and report["storage_selector_interface"], report
        assert report["repeated_regions"] > 0, report
        for outer, inner in cases:
            path.write_text(source.replace("array<i64: 2, 3>", f"array<i64: {outer}, {inner}>"))
            check(json.loads(run(tool, "--structured-trace", path)), outer, inner)
        path.write_text(source)
        emitted = run(tool, "--insert-logical", path)
        assert "version = 4" in emitted and "pto.store" in emitted
        for invalid in (source.replace(" overflow<nsw>", ""),
                        source.replace("    return",
                                       "    %external = pto.load %p[%zero] : !pto.ptr<f32, gm> -> f32\n    return")):
            path.write_text(invalid)
            rejected = json.loads(run(tool, "--sequence-analysis", path))
            assert rejected["error"] and not rejected["queries_available"], rejected
    print(f"repeated storage: {len(cases)} production closures, "
          "zero-inner stores and explicit crossing rejection passed")


if __name__ == "__main__":
    main()
