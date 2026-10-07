# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check finite allocation against executed logical identities and causal reuse."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_finite_guarded_analysis import load, compute, render, validate
from check_logical_insertion import closure_with_commands


def run(tool, args, path):
    return subprocess.run([tool, *args, str(path)], capture_output=True, text=True, timeout=45)


def physical_check(report, eligible):
    assert report["accepted"] and report["allocated"], report
    logical, physical = report["trace"], report["physical"]
    assert not physical["error"] and len(logical["events"]) == len(physical["events"])
    pipes = [e["pipe"] for e in logical["events"] if e["kind"] == "payload"]
    commands, uses = [], []
    for old, new in zip(logical["events"], physical["events"]):
        if old["kind"] in ("payload", "barrier"):
            assert old == new
        if old["kind"] == "payload" or (old["kind"] == "barrier" and old["pipe"] == 6):
            continue
        assert (old["kind"], old["gap"], old["pipe"]) == (new["kind"], new["gap"], new["pipe"])
        command = dict(old)
        if old["kind"] != "barrier":
            identity = (old["plan"], old["record"], old["source_ordinal"], tuple(old.get("members", [])))
            command["identity"] = identity
            assert new["physical_id"] in eligible
            event = new["physical_id"]
            identity += (new["source_pipe"], new["target_pipe"])
            uses.append((len(commands), new["kind"], event, identity))
        commands.append(command)
    reach = closure_with_commands(pipes, commands, include_commands=True)
    live, consumed = {}, {}
    for index, kind, event, identity in uses:
        vertex = 2 * len(pipes) + index
        if kind == "set":
            assert event not in live
            if event in consumed:
                assert reach[consumed[event]] >> vertex & 1, "reuse without causal consumption"
            live[event] = identity
        else:
            assert live.pop(event, None) == identity
            consumed[event] = vertex
    assert not live


def main():
    tool, opt, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    linear = "\n".join([load("L"), compute("C"), load("E"), compute("D"), load("B"), compute("F")])
    cases = [linear, "", load("L"),
             f"scf.if %g {{\n{load('L')}\n{compute('C')}\n}} else {{\n{load('E')}\n{compute('D')}\n}}",
             f"{load('L')}\n{compute('C')}\nscf.if %g {{\n{load('E')}\n{compute('D')}\n}}\n{load('B')}\n{compute('F')}"]
    effects = {x: ({"mat"}, {"left"}) for x in ("L", "E", "B")}
    effects.update({x: ({"left", "right"}, {"acc"}) for x in ("C", "D", "F")})
    count = 0
    with tempfile.TemporaryDirectory(prefix="finite-allocation-") as scratch:
        path = Path(scratch) / "input.pto"
        for body, g, h in itertools.product(cases, (0, 1), (0, 1)):
            text = render(source, body, g, h).replace(
                "test.trace_arguments =",
                "test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>, test.trace_arguments =")
            path.write_text(text)
            result = run(tool, ["--structured-trace"], path)
            assert result.returncode == 0, result.stderr
            report = json.loads(result.stdout)
            assert report["accepted"], report
            labels = [e["label"] for e in report["trace"]["events"] if e["kind"] == "payload"]
            validate(report, labels, effects)
            physical_check(report, set(range(6)))
            count += 1
        # Two independent publications have overlapping causal lifetimes.
        body = "\n".join([load('L'), load('X', 'second'), compute('C'), compute('Y').replace('%left,', '%second,')])
        path.write_text(render(source, body, 1, 1).replace(
            "test.trace_arguments =", "test.eligible_ids = array<i64: 0>, test.trace_arguments ="))
        report = json.loads(run(tool, ["--structured-trace"], path).stdout)
        assert report["accepted"] and not report["allocated"] and report["allocation_unchanged_on_failure"]
        path.write_text(path.read_text().replace('array<i64: 0>', 'array<i64: 1, 3>'))
        report = json.loads(run(tool, ["--structured-trace"], path).stdout)
        physical_check(report, {1, 3})
        logical = run(opt, ['--mlir-disable-threading', '--pto-frontier-analysis'], path)
        assert logical.returncode == 0, logical.stderr
        assert 'pto.finite_allocation' in logical.stdout
        for old, new in [('evidence = array<i64>', 'evidence = array<i64: 0>'),
                         ('version = 2 : i64', 'version = 1 : i64')]:
            assert old in logical.stdout, logical.stdout
            path.write_text(logical.stdout.replace(old, new, 1))
            result = run(opt, ['--mlir-disable-threading', '--pto-frontier-allocate=eligible-ids=0,1'], path)
            assert result.returncode != 0, result.stdout
        path.write_text(logical.stdout)
        for ids in ['0,0', '6', '7', '-1']:
            result = run(opt, ['--mlir-disable-threading', '--pto-frontier-allocate=eligible-ids='+ids], path)
            assert result.returncode != 0
    print(f"finite allocation: {count} guarded/explicit closure traces, capacity and malformed evidence passed")


if __name__ == "__main__":
    main()
