# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Finite overlay covers and emitted endpoint pairing against an independent DAG."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def main():
    tool, fixture = sys.argv[1:3]
    compact = len(sys.argv) > 3
    source = Path(fixture).read_text()
    with tempfile.TemporaryDirectory(prefix="finite-overlay-") as scratch:
        path = Path(scratch) / "case.pto"
        for enabled, trips in ((enabled, trips) for enabled in (0, 1)
                               for trips in ((0, 1, 2, 4) if compact else (1,))):
            if compact:
                text = source.replace("array<i64: 0, 3>", f"array<i64: {enabled}, {trips}>")
            else:
                text = source.replace("array<i64: 0, 0>", f"array<i64: {enabled}, 0>")
            path.write_text(text)
            run = subprocess.run([tool, "--finite-overlay-insertion", str(path)],
                                 capture_output=True, text=True, check=False, timeout=45)
            assert run.returncode == 0, run.stderr + run.stdout
            report = json.loads(run.stdout)
            assert report["accepted"] and report["unchanged_before_insertion"], report
            assert report["no_physical_certificate"], report
            assert not report["trace"]["error"], report
            events = report["trace"]["events"]
            payloads = [event for event in events if event["kind"] == "payload"]
            assert [event["type"] for event in payloads] == [0, 1, 2] * trips
            pipes = [event["pipe"] for event in payloads]
            assert len(set(pipes)) == (3 if trips else 0)
            effects = [({"mat"}, {"left"}), ({"vec"}, {"vec"}), ({"left", "right"}, {"acc"})] * trips
            required_edges = native(pipes)
            for a, (reads, writes) in enumerate(effects):
                for b in range(a + 1, len(effects)):
                    later_reads, later_writes = effects[b]
                    if writes & (later_reads | later_writes) or reads & later_writes:
                        required_edges.add((2*a+1, 2*b))
            if enabled and trips:
                required_edges |= {(1, 2), (3, 4)}
            required = closure(2 * len(payloads), required_edges)[0]
            commands = []
            for event in events:
                if event["kind"] == "payload":
                    continue
                if event["kind"] == "barrier":
                    assert compact and event["pipe"] != 6, event
                command = dict(event)
                if event["kind"] != "barrier":
                    command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                           tuple(event.get("members", [])))
                commands.append(command)
            actual = closure_with_commands(pipes, commands)
            assert actual == [row & ~(1 << i) for i, row in enumerate(required)], (enabled, report)
            # This checks reduction, not just closure preservation: no redundant
            # original A->B pair survives when the two new covers are enabled.
            if not compact:
                assert len(commands) == (4 if enabled else 2), (enabled, commands)
            # The original A_i->B_i family survives for every later visit. Only
            # the first pair is replaced, despite sharing the same static sites.
            elif trips:
                sets = [command for command in commands if command["kind"] == "set"]
                assert len(sets) == 2 * trips - 1 + enabled, (enabled, trips, commands)
    print(f"finite overlay insertion: {'8 compact' if compact else '2 finite'} traces passed")


if __name__ == "__main__":
    main()
