# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check actual sequence insertion against independent physical conflict graphs."""
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def invoke(tool, mode, path):
    result = subprocess.run([tool, mode, str(path)], capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout


def validate(document, trips, reader_only=False, rmw=False, inner_visits=0, partial=False):
    assert document["accepted"], document
    trace = document["trace"]
    assert not trace["error"], trace
    def written(slot):
        return {("left", slot, 0), ("left", slot, 1)} if partial else {("left", slot)}
    expected = [(0, [], ({("mat", 0)}, written(0)))]
    width = 1 if reader_only else 2
    for region, count in enumerate(trips):
        for i in range(max(0, count)):
            for j in range(inner_visits or 1):
                slot = 0 if reader_only else (j if inner_visits else i) % 2
                coordinates = [i, j] if inner_visits else [i]
                if not reader_only:
                    expected.append((1 + width * region, coordinates, ({("mat", 0)}, written(slot))))
                reads = written(slot) | {("right", 0)}
                if rmw:
                    reads.add(("acc", 0))
                expected.append((width + width * region, coordinates, (reads, {("acc", 0)})))
    expected.append((1 + width * len(trips), [], ({("mat", 0)}, {("left", 0, 0)} if partial else written(0))))
    payloads = [event for event in trace["events"] if event["kind"] == "payload"]
    assert [(x["type"], x["coordinates"]) for x in payloads] == [(x[0], x[1]) for x in expected]
    pipes = [x["pipe"] for x in payloads]
    edges = native(pipes)
    for a, (_, _, (reads, writes)) in enumerate(expected):
        for b in range(a + 1, len(expected)):
            later_reads, later_writes = expected[b][2]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * a + 1, 2 * b))
    required, _ = closure(2 * len(expected), edges)
    commands = []
    gap = 0
    for event in trace["events"]:
        if event["kind"] == "payload":
            gap += 1
            continue
        assert event["gap"] == gap
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert gap == len(expected), "child region inserted a completion barrier"
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands(pipes, commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], "sequence changed required order"


def operation_count(text, name):
    """Count original textual operations, excluding names in diagnostic attributes."""
    pattern = r"^\s*(?:%[^=\n]+=[ \t]*)?" + re.escape(name) + r"\b"
    return len(re.findall(pattern, text, re.MULTILINE))


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    start = source.index("    scf.for")
    # Find the outer closing brace following the two payload body lines.
    end = source.index("    }", start) + len("    }\n")
    second = source[start:end].replace("%i", "%j").replace("%n", "%m")
    two = source[:end] + second + source[end:]
    two = two.replace("%n: index)", "%n: index, %m: index)")
    checked = 0
    with tempfile.TemporaryDirectory(prefix="sequence-analysis-") as directory:
        path = Path(directory) / "case.pto"
        for n in [-2, 0, 1, 2, 3, 5, 9]:
            path.write_text(source.replace("array<i64: 5>", f"array<i64: {n}>"))
            validate(json.loads(invoke(tool, "--structured-trace", path)), [n])
            checked += 1
        for n, m in [(0, 0), (0, 3), (4, 0), (1, 1), (2, 3), (5, 4)]:
            path.write_text(two.replace("array<i64: 5>", f"array<i64: {n}, {m}>"))
            validate(json.loads(invoke(tool, "--structured-trace", path)), [n, m])
            checked += 1
        for reader_only, rmw in [(True, False), (False, True), (True, True)]:
            variant = source
            if reader_only:
                variant = variant.replace(
                    '      pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%left : !left)\n', '')
                variant = variant.replace('pto.tmatmul ins(%left, %right', 'pto.tmatmul ins(%first, %right')
            if rmw:
                operand = '%first' if reader_only else '%left'
                variant = variant.replace(f'pto.tmatmul ins({operand}, %right : !left, !right)',
                    f'pto.tmatmul.acc ins(%acc, {operand}, %right : !acc, !left, !right)')
            for n in [0, 1, 4]:
                path.write_text(variant.replace("array<i64: 5>", f"array<i64: {n}>"))
                validate(json.loads(invoke(tool, "--structured-trace", path)), [n], reader_only, rmw)
                checked += 1
        numeric = source.replace('      %slot = arith.remui %i, %two : index',
            '      scf.for %j = %zero to %two step %one {\n      %slot = arith.remui %j, %two : index')
        numeric = numeric.replace('    }\n    pto.textract', '      }\n    }\n    pto.textract')
        for n in [0, 1, 4]:
            path.write_text(numeric.replace("array<i64: 5>", f"array<i64: {n}>"))
            validate(json.loads(invoke(tool, "--structured-trace", path)), [n], inner_visits=2)
            checked += 1
        alias = source
        last = alias.rfind('    pto.textract')
        alias = (alias[:last] + '    %alias = pto.alloc_tile addr = %base : !left\n' +
                 alias[last:].replace('%first', '%alias'))
        path.write_text(alias)
        validate(json.loads(invoke(tool, "--structured-trace", path)), [5])
        checked += 1
        partial = source.replace("left, 16x16xf16", "left, 32x16xf16")
        partial = partial.replace("mat, 16x16xf16", "mat, 32x16xf16")
        partial = partial.replace("acc, 16x16xf32", "acc, 32x16xf32")
        partial = partial.replace("module attributes",
            "!half = !pto.tile_buf<left, 16x16xf16, slayout=row_major>\nmodule attributes", 1)
        tail = partial.rfind('    pto.textract')
        partial = partial[:tail] + (
            '    %tail = pto.alloc_tile addr = %base : !half\n') + (
            partial[tail:].replace('outs(%first : !left)', 'outs(%tail : !half)'))
        for n in (0, 1, 5):
            path.write_text(partial.replace("array<i64: 5>", f"array<i64: {n}>"))
            validate(json.loads(invoke(tool, "--structured-trace", path)), [n], partial=True)
            checked += 1
        for n in (0, 1, 5):
            recomposed = source.replace("attributes {test.trace_arguments",
                "attributes {test.recompose, test.trace_arguments")
            path.write_text(recomposed.replace("array<i64: 5>", f"array<i64: {n}>"))
            validate(json.loads(invoke(tool, "--structured-trace", path)), [n])
            checked += 1
        # The whole dispatcher can replay a pure scalar upper bound through an
        # endpoint-capable route. The direct sequence adapter still exports its
        # queries if that value is unavailable at its earlier SET cut.
        unavailable = source.replace("    scf.for %i = %zero to %n",
            "    %late = arith.addi %n, %one : index\n    scf.for %i = %zero to %late")
        unavailable = unavailable.replace(
            '      pto.textract ins(%mat, %zero, %zero : !mat, index, index) outs(%left : !left)\n', '')
        unavailable = unavailable.replace('pto.tmatmul ins(%left, %right', 'pto.tmatmul ins(%first, %right')
        path.write_text(unavailable)
        replayed = json.loads(invoke(tool, "--structured-trace", path))
        assert replayed["accepted"] and not replayed["trace"]["error"], replayed
        report = json.loads(invoke(tool, "--sequence-analysis", path))
        assert not report["error"] and not report["prepared"], report
        assert report["queries_available"] and report["unchanged"], report
        # Recognition/insertion size must not depend on runtime trip count.
        path.write_text(source.replace("array<i64: 5>", "array<i64: 1000000000>"))
        large = invoke(tool, "--insert-logical", path)
        path.write_text(source)
        small = invoke(tool, "--insert-logical", path)
        assert large.replace("1000000000", "5") == small
        # Diagnostic attributes can name scf.for without adding an operation.
        assert operation_count(large, "scf.for") == operation_count(source, "scf.for")
        assert "pto.logical_set" in large and "pto.set_flag" not in large
        # Constant trip counts change analysis input, unlike trace arguments.
        constant_sizes = []
        constant_analysis = []
        for bound in (101, 1000000101):
            constant = source.replace("    scf.for %i = %zero to %n",
                f"    %bound = arith.constant {bound} : index\n    scf.for %i = %zero to %bound")
            path.write_text(constant)
            emitted = invoke(tool, "--insert-logical", path)
            report = json.loads(invoke(tool, "--sequence-analysis", path))
            assert not report["error"] and report["prepared"], report
            constant_analysis.append(tuple(report[k] for k in ("ports", "cells", "expressions", "emitted")))
            assert operation_count(emitted, "scf.for") == 1
            assert operation_count(emitted, "pto.textract") == operation_count(source, "pto.textract")
            constant_sizes.append((len(emitted.splitlines()), emitted.count("pto.logical_")))
        assert constant_sizes[0] == constant_sizes[1], constant_sizes
        assert constant_analysis[0] == constant_analysis[1], constant_analysis
    print(f"sequence composition: {checked} actual command closures and trip-independent insertion passed")


if __name__ == "__main__":
    main()
