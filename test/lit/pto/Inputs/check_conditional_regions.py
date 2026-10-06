# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Verify branch-selected compact regions against the independent event oracle."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_repeated_region import check
from check_logical_insertion import closure_with_commands


def run(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return json.loads(result.stdout)


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    tested = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "conditional.pto"
        for guard in (0, 1):
            for n, m in ((0, 3), (2, 0), (1, 1), (2, 3)):
                candidate = source.replace("array<i64: 1, 2, 3>", f"array<i64: {guard}, {n}, {m}>")
                path.write_text(candidate)
                report = run(tool, "--sequence-analysis", path)
                assert report["prepared"] and not report["error"], report
                check(run(tool, "--structured-trace", path), [n if guard else 0, m])
                tested += 1
        # Swapping the active arm must preserve its occurrence semantics.
        candidate = source.replace("scf.if %g {", "scf.if %g {\n    } else {")
        for guard in (0, 1):
            path.write_text(candidate.replace("array<i64: 1, 2, 3>", f"array<i64: {guard}, 2, 3>"))
            check(run(tool, "--structured-trace", path), [0 if guard else 2, 3])
            tested += 1
        start = source.index("    scf.for %visit")
        end = source.index("    }\n    pto.textract", start)
        body = source[start:end]
        both = source[:end] + "    } else {\n" + body + source[end:]
        for guard in (0, 1):
            path.write_text(both.replace("array<i64: 1, 2, 3>", f"array<i64: {guard}, 2, 3>"))
            document = run(tool, "--structured-trace", path)
            for event in document.get("trace", {}).get("events", []):
                if event["kind"] == "payload":
                    if event["coordinates"]:
                        event["type"] = 1 if event["pipe"] == 3 else 2
                    elif event["type"] != 0:
                        event["type"] = 3
            check(document, [2, 3])
            tested += 1
        # Shared interfaces must remain economical under repeated branch wrapping.
        nested = source
        for depth in range(2, 5):
            nested = nested.replace("    scf.for %visit", "    scf.if %g {\n    scf.for %visit", 1)
            marker = "    }\n    pto.textract"
            nested = nested.replace(marker, "    }\n" + marker, 1)
            path.write_text(nested)
            report = run(tool, "--sequence-analysis", path)
            assert report["prepared"], report
            check(run(tool, "--structured-trace", path), [2, 3])
            print("conditional depth", depth, "expressions", report["expressions"], "emitted", report["emitted"])
            tested += 1
        # Explicit runs inside a selected arm retain their original block cuts,
        # including a nonempty internal cross-pipe demand before a compact child.
        prefix = """    pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%first : !left)
    pto.tmatmul ins(%first, %right : !left, !right) outs(%acc : !acc)
"""
        for guard in (0, 1):
            candidate = source.replace("scf.if %g {", "scf.if %g {\n" + prefix, 1)
            path.write_text(candidate.replace("array<i64: 1, 2, 3>", f"array<i64: {guard}, 2, 3>"))
            report = run(tool, "--structured-trace", path)
            assert report["accepted"] and not report["trace"]["error"], report
            payloads = [event for event in report["trace"]["events"] if event["kind"] == "payload"]
            assert len(payloads) == (16 if guard else 2), payloads
            commands = []
            for event in report["trace"]["events"]:
                if event["kind"] == "payload" or (event["kind"] == "barrier" and event["pipe"] == 6):
                    continue
                command = dict(event)
                if event["kind"] != "barrier":
                    command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                           tuple(event.get("members", [])))
                commands.append(command)
            actual = closure_with_commands([event["pipe"] for event in payloads], commands)
            if guard:
                # Payloads 1 and 2 are the inserted TEXTRACT and TMATMUL:
                # the latter reads the tile written by the former.
                assert actual[3] & (1 << 4), "missing explicit prefix readiness"
        unavailable = source.replace("%g: i1,", "%flags: !pto.ptr<i1, gm>, %g: i1,")
        unavailable = unavailable.replace("    scf.if %g {",
            "    %predicate = pto.load %flags[%zero] : !pto.ptr<i1, gm> -> i1\n    scf.if %predicate {", 1)
        path.write_text(unavailable)
        report = run(tool, "--sequence-analysis", path)
        assert not report["prepared"] and report["unchanged"], report
    print(f"conditional compact regions: {tested} emitted closures passed")


if __name__ == "__main__":
    main()
