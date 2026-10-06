# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check compact nested endpoints against an independent expanded conflict graph."""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def invoke(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    return result.stdout


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    with tempfile.TemporaryDirectory(prefix="arithmetic-compaction-") as directory:
        path = Path(directory) / "input.pto"
        path.write_text(source)
        emitted = invoke(tool, "--insert-logical", path)
        count = len(re.findall(r"\barith\.", emitted))
        assert count <= 207, count
        # Include zero trips, singleton visits and several reset boundaries.
        for trips in [0, 1, 2, 4, 10]:
            path.write_text(source.replace("constant 10 : index", f"constant {trips} : index"))
            report = json.loads(invoke(tool, "--structured-trace", path))
            assert report["accepted"], report
            events = report["trace"]["events"]
            payloads = [event for event in events if event["kind"] == "payload"]
            assert len(payloads) == trips * (trips + 1)
            pipes = [event["pipe"] for event in payloads]
            effects = [({"input"}, {"tile"}) if event["type"] == 0 else
                       ({"tile"}, {"output"}) for event in payloads]
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
            assert closure_with_commands(pipes, commands) == [row & ~(1 << i) for i, row in enumerate(required)]
    print(f"nested arithmetic: {count} operations, five exact expanded-graph comparisons passed")


if __name__ == "__main__":
    main()
