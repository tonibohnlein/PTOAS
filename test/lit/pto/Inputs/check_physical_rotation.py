# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check physical bank aliases across distinct allocation SSA roots."""
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
    with tempfile.TemporaryDirectory(prefix="physical-rotation-") as directory:
        path = Path(directory) / "case.pto"
        for trips in (0, 1, 2, 3, 5, 8):
            path.write_text(source.replace("array<i64: 5>", f"array<i64: {trips}>"))
            run = subprocess.run([tool, "--rotating-analysis", str(path)], capture_output=True,
                                 text=True, check=True, timeout=60)
            result = json.loads(next(line for line in run.stdout.splitlines() if line.startswith("{")))
            if result["error"] or not result["prepared"] or result["trace"]["error"]:
                raise RuntimeError(result)
            if len({item["family"] for item in result["fragments"]}) != 1:
                raise RuntimeError("physical aliases were split by allocation SSA identity")
            pipes = result["periodic"]["payloads"] * trips
            edges = native(pipes)
            for i in range(trips):
                edges.add((4 * i + 1, 4 * i + 2))
                for j in range(i + 2, trips, 2):
                    edges.update(((4 * i + 1, 4 * j), (4 * i + 3, 4 * j),
                                  (4 * i + 1, 4 * j + 2)))
            expected = closure(len(pipes) * 2, edges)[0]
            commands = []
            for event in result["trace"]["events"]:
                if event["kind"] == "payload" or (event["kind"] == "barrier" and event["pipe"] == 6):
                    continue
                command = dict(event)
                if event["kind"] != "barrier":
                    command["identity"] = (event["plan"], event["record"], event["source_ordinal"])
                commands.append(command)
            actual = closure_with_commands(pipes, commands)
            if actual != [row & ~(1 << i) for i, row in enumerate(expected)]:
                raise RuntimeError((trips, actual, expected))
        path.write_text(source.replace("%two = arith.constant 2 : index",
                                       "%two = arith.constant 1000000000 : index"))
        run = subprocess.run([tool, "--sequence-analysis", str(path)], capture_output=True,
                             text=True, check=True, timeout=10)
        rejected = json.loads(run.stdout)
        if "slot expansion" not in rejected["error"] or not rejected["unchanged"]:
            raise RuntimeError(("large compact bank family was materialized", rejected))
    print("physical rotation: six trip counts, shared aliases, and emitted-order checks passed")


if __name__ == "__main__":
    main()
