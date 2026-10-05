# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare nested arithmetic endpoint code against independent physical conflicts."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    count = 0
    with tempfile.TemporaryDirectory(prefix="arithmetic-insertion-") as scratch:
        path = Path(scratch) / "case.pto"
        for enabled, trips in itertools.product((0, 1), (-2, 0, 1, 2, 3, 5)):
            path.write_text(source.replace("array<i64: 1, 3>", f"array<i64: {enabled}, {trips}>"))
            run = subprocess.run([tool, "--structured-trace", str(path)], capture_output=True,
                                 text=True, check=False, timeout=45)
            assert run.returncode == 0, run.stderr + run.stdout
            report = json.loads(run.stdout)
            assert report["accepted"], report
            events = report["trace"]["events"]
            payloads = [event for event in events if event["kind"] == "payload"]
            expected = []
            for outer in range(max(0, trips)):
                if enabled:
                    expected.append((0, [outer]))
                expected.extend((1, [outer, inner]) for inner in range(outer))
            assert [(event["type"], event["coordinates"]) for event in payloads] == expected
            effects = [({"mat"}, {"left"}) if event["type"] == 0 else
                       ({"left", "right"}, {"acc"}) for event in payloads]
            pipes = [event["pipe"] for event in payloads]
            edges = native(pipes)
            for a, (reads, writes) in enumerate(effects):
                for b in range(a + 1, len(effects)):
                    later_reads, later_writes = effects[b]
                    if writes & (later_reads | later_writes) or reads & later_writes:
                        edges.add((2 * a + 1, 2 * b))
            required = closure(2 * len(payloads), edges)[0]
            commands = []
            for event in events:
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
            actual = closure_with_commands(pipes, commands)
            assert actual == [row & ~(1 << i) for i, row in enumerate(required)], (enabled, trips)
            count += 1
    print(f"arithmetic insertion: {count} nested physical-conflict traces passed")


if __name__ == "__main__":
    main()
