# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare affine slice plans with independently unfolded byte conflicts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def require(condition, detail):
    if not condition:
        raise RuntimeError(str(detail))


def compare(left, right, predicate):
    return {"slt": left < right, "sle": left <= right, "sgt": left > right,
            "sge": left >= right, "eq": left == right, "ne": left != right}[predicate]


def validate(document, trips, cut, predicate, swapped, slope):
    require(document["accepted"], document)
    require(not document["trace"]["error"], document)
    effects = [(set(), {0})]
    for i in range(max(0, trips)):
        if i + 1 < trips:
            effects.append((set(), {(i + 1) % 2}))
        left, right = cut + slope * i, 0
        if swapped:
            left, right = right, left
        if compare(left, right, predicate):
            effects.append((set(), {i % 2}))
        effects.append(({i % 2}, set()))
    events = document["trace"]["events"]
    pipes = [event["pipe"] for event in events if event["kind"] == "payload"]
    require(len(pipes) == len(effects), (document, effects))
    edges = native(pipes)
    for a, (reads, writes) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            other_reads, other_writes = effects[b]
            if writes & (other_reads | other_writes) or reads & other_writes:
                edges.add((2 * a + 1, 2 * b))
    _, covers = closure(2 * len(effects), edges)
    # The production policy implements a nonadjacent local cover with a barrier
    # immediately before its consumer, adding that native completion prefix.
    for source, target in covers:
        a, b = source // 2, target // 2
        if pipes[a] == pipes[b]:
            previous = max(i for i in range(b) if pipes[i] == pipes[b])
            edges.add((2 * previous + 1, target))
    required = closure(2 * len(effects), edges)[0]
    commands = []
    for event in events:
        if event["kind"] == "payload" or (event["kind"] == "barrier" and event["pipe"] == 6):
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    require(actual == [row & ~(1 << i) for i, row in enumerate(required)],
            (trips, cut, predicate, swapped, slope, document))


def main():
    tool, fixture = sys.argv[1:]
    original = Path(fixture).read_text()
    source = original.replace("%n: index,", "%n32: i32, %cut32: i32,")
    source = source.replace("    %zero =", """    %n = arith.index_cast %n32 : i32 to index
    %cut = arith.index_cast %cut32 : i32 to index
    %four = arith.constant 4 : index
    %zero =""")
    source = source.replace("      %destination =", """      %scaled = arith.muli %i, %four : index
      %varying = arith.subi %cut, %scaled : index
      %active = arith.cmpi PREDICATE, LEFT, RIGHT : index
      scf.if %active {
        pto.texpands ins(%value : f32) outs(%reader : !tile)
      }
      %destination =""")
    checked = 0
    # These source type ranges, not sample values, prove no signed index wrap.
    cases = [(5, cut) for cut in (-2147483648, -1, 0, 3, 4, 8, 2147483647)]
    cases += [(-2, 3), (0, 0), (1, 0)]
    with tempfile.TemporaryDirectory(prefix="affine-boundary-") as directory:
        path = Path(directory) / "case.pto"
        for predicate in ("slt", "sle", "sge", "sgt", "eq", "ne"):
            for trips, cut in cases:
                # Exercise both orientation and slope signs without doubling
                # the full Cartesian test campaign.
                swapped = checked % 2 == 0
                slope = -4 if checked % 3 else 4
                text = source.replace("PREDICATE", predicate)
                text = text.replace("LEFT", "%zero" if swapped else "%varying")
                text = text.replace("RIGHT", "%varying" if swapped else "%zero")
                if slope > 0:
                    text = text.replace("arith.subi %cut, %scaled", "arith.addi %cut, %scaled")
                text = text.replace("array<i64: 5>", f"array<i64: {trips}, {cut}>")
                path.write_text(text)
                result = subprocess.run([tool, "--structured-trace", str(path)],
                                        text=True, capture_output=True, check=False)
                require(result.returncode == 0, (result.stdout, result.stderr))
                validate(json.loads(result.stdout), trips, cut, predicate, swapped, slope)
                checked += 1
        # Small sampled values do not prove the same formula safe for arbitrary
        # index parameters. Reject unproved signed wrap; never assume metadata
        # is sorted or constrain it using the trace's sample arguments.
        unbounded = source.replace("%n32: i32, %cut32: i32,", "%n: index, %cut: index,")
        unbounded = unbounded.replace("    %n = arith.index_cast %n32 : i32 to index\n", "")
        unbounded = unbounded.replace("    %cut = arith.index_cast %cut32 : i32 to index\n", "")
        unbounded = unbounded.replace("PREDICATE", "sge").replace("LEFT", "%varying").replace("RIGHT", "%zero")
        path.write_text(unbounded.replace("array<i64: 5>", "array<i64: 5, 8>"))
        result = subprocess.run([tool, "--sequence-analysis", str(path)],
                                text=True, capture_output=True, check=False)
        require(result.returncode == 0, (result.stdout, result.stderr))
        rejected = json.loads(result.stdout)
        require(rejected["error"] and rejected["unchanged"], rejected)
    print(f"affine boundary cuts: {checked} independently unfolded closures passed")


if __name__ == "__main__":
    main()
