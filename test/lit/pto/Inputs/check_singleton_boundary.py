# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check single-active-visit wrappers against an independently unfolded graph."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_repeated_region import check, run
from check_frontier_public_pipeline import unchanged_failure


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def wrapper(source, lower, upper, step, active):
    source = source.replace("    scf.for %visit = %zero to %n step %one {", f"""
    %outer_lower = arith.constant {lower} : index
    %outer_upper = arith.constant {upper} : index
    %outer_step = arith.constant {step} : index
    %active = arith.constant {active} : index
    scf.for %visit = %outer_lower to %outer_upper step %outer_step {{
      %selected = arith.cmpi eq, %visit, %active : index
      scf.if %selected {{
      %limit = arith.addi %m, %visit : index""")
    source = source.replace("%i = %zero to %m", "%i = %zero to %limit")
    return source.replace("    }\n    pto.textract", "      }\n    }\n    pto.textract")


def main():
    tool, optimizer, fixture = sys.argv[1:]
    source = Path(fixture).read_text(encoding="utf-8")
    tested = 0
    with tempfile.TemporaryDirectory(prefix="singleton-boundary-") as scratch:
        path = Path(scratch) / "case.pto"
        for lower, upper, step, active in ((0, 2, 1, 0), (0, 2, 1, 1), (7, 11, 2, 9)):
            for inner in (0, 1, 3):
                program = wrapper(source, lower, upper, step, active)
                program = program.replace("array<i64: 2, 3>", f"array<i64: 2, {inner}>")
                path.write_text(program, encoding="utf-8")
                document = json.loads(run(tool, "--structured-trace", path))
                require(document["accepted"], document)
                expected = [(0, [])]
                for visit in range(max(0, inner + active)):
                    expected.extend([(1, [active, visit]), (2, [active, visit])])
                expected.append((3, []))
                actual = [(event["type"], event["coordinates"]) for event in document["trace"]["events"]
                          if event["kind"] == "payload"]
                require(actual == expected, (actual, expected))
                # Reuse the independent byte-conflict/native-order graph with
                # one visit after separately checking the original coordinates.
                normalized = copy.deepcopy(document)
                for event in normalized["trace"]["events"]:
                    if event.get("coordinates"):
                        event["coordinates"][0] = 0
                check(normalized, [1, inner + active])
                logical = subprocess.run([optimizer, "--pto-frontier-analysis", str(path)],
                                         capture_output=True, text=True, check=False, timeout=60)
                require(logical.returncode == 0, logical.stderr)
                require("pto.logical_set" in logical.stdout, logical.stdout)
                tested += 1
        # A nonexistent selected visit contributes no occurrences or commands.
        program = wrapper(source, 0, 2, 1, 5)
        path.write_text(program, encoding="utf-8")
        document = json.loads(run(tool, "--structured-trace", path))
        check(document, [0, 3])
        # More than one active visit must retain genuine recurrence obligations.
        program = wrapper(source, 0, 3, 1, 2).replace(
            "arith.cmpi eq, %visit, %active", "arith.cmpi slt, %visit, %active")
        # An invariant inner domain lets the independent oracle check both
        # visits and every crossing hazard, not only their payload presence.
        program = program.replace("%i = %zero to %limit", "%i = %zero to %m")
        path.write_text(program, encoding="utf-8")
        report = json.loads(run(tool, "--structured-trace", path))
        if report["accepted"]:
            check(report, [2, 3])
        else:
            unchanged_failure(optimizer, program, ["--pto-frontier-analysis"])
    print(f"singleton boundary: {tested} exact command graphs and public logical plans passed")


if __name__ == "__main__":
    main()
