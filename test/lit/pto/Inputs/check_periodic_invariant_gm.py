# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check periodic GM-origin cancellation against independent conflict graphs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_finite_allocation import physical_check
from check_logical_insertion import closure_with_commands, recognized
from check_periodic_demands import unfolded


def run(tool, args, path):
    return subprocess.run([tool, *args, str(path)], capture_output=True, text=True, timeout=45)


def exact_copy_order(report, trips):
    trace = report["trace"]
    payloads = [event for event in trace["events"] if event["kind"] == "payload"]
    assert len(payloads) == 2 * trips
    if not payloads:
        return
    # Every load writes the same UB tile; every store reads it. GM source is
    # read-only and destination slices are disjoint. Enumerate all UB conflicts
    # independently of the producer's partition or exported generator list.
    case = {"pipes": [event["pipe"] for event in payloads[:2]], "word": [
        [{"atom": 0, "read": False, "write": True}],
        [{"atom": 0, "read": True, "write": False}]]}
    required, covers = unfolded(case, len(payloads))
    commands = []
    for event in trace["events"]:
        if event["kind"] == "payload" or (event["kind"] == "barrier" and event["pipe"] == 6):
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands([event["pipe"] for event in payloads], commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)]
    assert len(covers) == 2 * trips - 1
    # For this copy pipeline, additionally prove reuse of the numeric ID across
    # directions. The earlier WAIT must precede the next SET even when the
    # source and target pipes exchange roles.
    reach = closure_with_commands([event["pipe"] for event in payloads], commands, include_commands=True)
    physical = [event for event in report["physical"]["events"] if event["kind"] != "payload"
                and not (event["kind"] == "barrier" and event["pipe"] == 6)]
    live, consumed = set(), {}
    for index, command in enumerate(physical):
        if command["kind"] == "barrier":
            continue
        identity = command["physical_id"]
        vertex = 2 * len(payloads) + index
        if command["kind"] == "set":
            assert identity not in live
            if identity in consumed:
                assert reach[consumed[identity]] >> vertex & 1
            live.add(identity)
        else:
            assert identity in live
            live.remove(identity)
            consumed[identity] = vertex
    assert not live


def main():
    tool, opt, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    with tempfile.TemporaryDirectory(prefix="periodic-invariant-gm-") as directory:
        path = Path(directory) / "input.pto"
        for trips in [0, 1, 2, 7, 20]:
            for args in ["1, 2, 3", "0, -2, 0"]:
                case = source.replace("constant 20 : index", f"constant {trips} : index")
                case = case.replace("array<i64: 1, 2, 3>", f"array<i64: {args}>")
                path.write_text(case)
                if trips:
                    template = recognized(tool, path)
                    assert template["period"] == template["refresh"] == 1
                    assert len(template["payloads"]) == 2
                    assert template["payloads"][1]["effects"][1]["discharge"] == 2
                result = run(tool, ["--structured-trace"], path)
                assert result.returncode == 0, result.stderr
                report = json.loads(result.stdout)
                physical_check(report, set(range(6)))
                exact_copy_order(report, trips)
        # A shift smaller than a slice overlaps between visits and must not be
        # discharged. A clamp that changes branch inside the loop is likewise
        # outside the affine-translation certificate.
        overlapping = source.replace("%2 = arith.muli %arg5, %c256", "%shift = arith.constant 128 : index\n"
                                     "      %2 = arith.muli %arg5, %shift")
        varying = source.replace("%11 = arith.maxsi %2, %c0", "%pred = arith.subi %arg5, %c1 : index\n"
                                 "      %raw = arith.muli %pred, %c256 : index\n"
                                 "      %11 = arith.maxsi %raw, %c0")
        for case in [overlapping, varying]:
            path.write_text(case)
            result = run(tool, ["--recognize"], path)
            assert result.returncode == 0, result.stderr
            documents = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
            attempts = [attempt for node in documents[0]["nodes"] for attempt in node["attempts"]
                        if attempt["route"] == "numeric-template"]
            assert attempts and all(attempt["state"] != "applicable" for attempt in attempts)
        path.write_text(source)
        result = run(opt, ["--mlir-disable-threading", "--pto-frontier-analysis=gm-alias=may-not-alias",
                           "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"], path)
        assert result.returncode == 0, result.stderr
        assert "pto.logical_" not in result.stdout
        for operation in ["scf.for", "pto.tload", "pto.tstore"]:
            assert result.stdout.count(operation) == 1
        # No independence assumption may be smuggled into may-alias mode.
        result = run(tool, ["--gm-alias=may-alias", "--recognize"], path)
        documents = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        attempts = [attempt for node in documents[0]["nodes"] for attempt in node["attempts"]
                    if attempt["route"] == "numeric-template"]
        assert attempts and all(attempt["state"] != "applicable" for attempt in attempts)
    print("periodic invariant GM: 10 exact-order/reuse traces; overlap, clamp and alias rejections passed")


if __name__ == "__main__":
    main()
