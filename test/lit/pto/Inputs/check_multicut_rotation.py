# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check symbolic cut order and equality slices against unfolded byte conflicts."""
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def validate(document, trips, cut, predicate):
    assert document["accepted"], document
    assert not document["trace"]["error"], document
    effects = [(set(), {0})]
    for i in range(max(0, trips)):
        if i + 1 < trips:
            effects.append((set(), {(i + 1) % 2}))
        active = {"slt": i < cut, "sge": i >= cut, "eq": i == cut, "ne": i != cut}[predicate]
        if active:
            effects.append((set(), {i % 2}))
        effects.append(({i % 2}, set()))
    events = document["trace"]["events"]
    pipes = [event["pipe"] for event in events if event["kind"] == "payload"]
    assert len(pipes) == len(effects), (document, effects)
    edges = native(pipes)
    for a, (reads, writes) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            other_reads, other_writes = effects[b]
            if writes & (other_reads | other_writes) or reads & other_writes:
                edges.add((2 * a + 1, 2 * b))
    required, covers = closure(2 * len(effects), edges)
    # Implementation policy places a same-pipe barrier at the consumer even
    # for a nonadjacent cover. Account for exactly that extra native prefix.
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
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], (trips, cut, predicate, document)


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text().replace("%n: index,", "%n: index, %cut: index,")
    source = source.replace("      %destination =", """      %active = arith.cmpi slt, %i, %cut : index
      scf.if %active {
        pto.texpands ins(%value : f32) outs(%reader : !tile)
      }
      %destination =""")
    checked = 0
    with tempfile.TemporaryDirectory(prefix="multicut-rotation-") as directory:
        path = Path(directory) / "case.pto"
        for predicate in ("slt", "sge", "eq", "ne"):
            for trips in (0, 1, 3, 5):
                for cut in (-1, 0, 2, 6):
                    text = source.replace("cmpi slt, %i, %cut", f"cmpi {predicate}, %i, %cut")
                    text = text.replace("array<i64: 5>", f"array<i64: {trips}, {cut}>")
                    path.write_text(text)
                    result = subprocess.run([tool, "--structured-trace", str(path)],
                                            text=True, capture_output=True, check=False)
                    assert result.returncode == 0, (trips, cut, predicate, result.stdout, result.stderr)
                    validate(json.loads(result.stdout), trips, cut, predicate)
                    checked += 1
        sizes = []
        for trips in (5, 1000000000):
            path.write_text(source.replace("array<i64: 5>", f"array<i64: {trips}, 2>"))
            result = subprocess.run([tool, "--insert-logical", str(path)],
                                    text=True, capture_output=True, check=True)
            assert len(re.findall(r"^\s*scf\.for\s", result.stdout, re.MULTILINE)) == 1
            sizes.append(len(result.stdout.splitlines()))
        assert sizes[0] == sizes[1], sizes
    print(f"multiple boundary cuts: {checked} unfolded closures and compact-size checks passed")


if __name__ == "__main__":
    main()
