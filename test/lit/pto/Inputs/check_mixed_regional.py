# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check mixed regional insertion against independently unfolded physical effects."""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


TRACE_ARGUMENTS = "array<i64: 1, 1, 3, 2, 0>"


def invoke(tool, path, mode="--structured-trace"):
    result = subprocess.run([tool, mode, str(path)], capture_output=True,
                            text=True, check=False, timeout=90)
    assert result.returncode == 0, result.stderr + result.stdout
    return result.stdout if mode == "--insert-logical" else json.loads(result.stdout)


def expected_payloads(g, h, n, m, offset, partial):
    # These cells describe the fixture's concrete byte overlaps. No analysis
    # boundary selector, source record, or inferred effect is used by this oracle.
    def left(slot):
        return {("left", slot, 0), ("left", slot, 1)} if partial else {("left", slot)}

    def writer(label, coordinates, slot, tail=False):
        writes = {("left", 0, 0)} if partial and tail else left(slot)
        return label, coordinates, {("mat", 0)}, writes

    def reader(label, coordinates, slot):
        return label, coordinates, left(slot) | {("right", 0)}, {("acc", 0)}

    result = [writer("E", [], 0), reader("I", [], 0)]
    for i in range(max(0, n)):
        if g:
            result.append(writer("L" if h else "B", [i], (i + offset) % 2))
            result.append(reader("C", [i], (i + offset) % 2))
    if h:
        result.append(writer("M", [], 0))
        if g:
            result.append(reader("Q", [], 0))
    else:
        result.append(reader("R", [], 0))
    result.append(reader("U", [], 0))
    for j in range(max(0, m)):
        if h:
            result.append(writer("S", [j], (j + offset) % 2))
            result.append(reader("T", [j], (j + offset) % 2))
    result.append(writer("F", [], 0, tail=True))
    return result


def validate(document, arguments, partial=False):
    assert document["accepted"], (arguments, document)
    trace = document["trace"]
    assert not trace["error"], trace
    expected = expected_payloads(*arguments, partial)
    payloads = [event for event in trace["events"] if event["kind"] == "payload"]
    assert [(event["label"], event["coordinates"]) for event in payloads] == [item[:2] for item in expected]
    pipes = [event["pipe"] for event in payloads]
    edges = native(pipes)
    for a, (_, _, reads, writes) in enumerate(expected):
        for b in range(a + 1, len(expected)):
            later_reads, later_writes = expected[b][2:]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * a + 1, 2 * b))
    required = closure(2 * len(payloads), edges)[0]
    commands = []
    gap = 0
    drains = 0
    for event in trace["events"]:
        if event["kind"] == "payload":
            gap += 1
            continue
        assert event["gap"] == gap
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert gap == len(payloads), "regional child drained the invocation"
            drains += 1
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
            endpoint_pipe = "source_pipe" if event["kind"] == "set" else "target_pipe"
            assert event["pipe"] == event[endpoint_pipe]
        commands.append(command)
    assert drains == 1, "mixed invocation must have exactly one completion barrier"
    actual = closure_with_commands(pipes, commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], (arguments, actual, required)


def seeded(source, arguments):
    return source.replace(TRACE_ARGUMENTS, "array<i64: " + ", ".join(map(str, arguments)) + ">")


def partial_alias(source):
    source = source.replace("left, 16x16xf16", "left, 32x16xf16")
    source = source.replace("mat, 16x16xf16", "mat, 32x16xf16")
    source = source.replace("acc, 16x16xf32", "acc, 32x16xf32")
    source = source.replace("module attributes",
                            "!half = !pto.tile_buf<left, 16x16xf16, slayout=row_major>\nmodule attributes", 1)
    tail = source.rfind("    pto.textract")
    return (source[:tail] + "    %tail = pto.alloc_tile addr = %base : !half\n" +
            source[tail:].replace("outs(%alias : !left)", "outs(%tail : !half)"))


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    count = 0
    # Only small text inputs live in tmpfs. Large trip counts are analyzed but
    # never unfolded by either the production driver or this finite oracle.
    with tempfile.TemporaryDirectory(prefix="mixed-regional-") as scratch:
        path = Path(scratch) / "case.pto"
        trip_pairs = ((0, 0), (0, 3), (4, 0), (1, 1), (2, 3), (5, 4), (-2, 3))
        for g, h, trips, offset in itertools.product((0, 1), (0, 1), trip_pairs, (-1, 0, 1)):
            arguments = (g, h, *trips, offset)
            path.write_text(seeded(source, arguments))
            validate(invoke(tool, path), arguments)
            count += 1
        for g, h, offset in itertools.product((0, 1), (0, 1), (2, 1000000001)):
            arguments = (g, h, 2, 3, offset)
            path.write_text(seeded(source, arguments))
            validate(invoke(tool, path), arguments)
            count += 1
        # Public export/recomposition also destroys the original local wrapper
        # and inserts empty siblings, exposing dangling ownership or lost guards.
        recomposed = source.replace("attributes {test.trace_arguments",
                                    "attributes {test.recompose, test.trace_arguments")
        for g, h in itertools.product((0, 1), repeat=2):
            arguments = (g, h, 3, 2, 1)
            path.write_text(seeded(recomposed, arguments))
            validate(invoke(tool, path), arguments)
            count += 1
        # Repartition one concrete ring cell at a later region's half-size alias.
        partial = partial_alias(source)
        for g, h in itertools.product((0, 1), repeat=2):
            arguments = (g, h, 2, 3, 1)
            path.write_text(seeded(partial, arguments))
            validate(invoke(tool, path), arguments, partial=True)
            count += 1
        metrics = []
        for bound in (101, 1000000101):
            constant = source.replace("    scf.for %i = %zero to %n",
                f"    %bound = arith.constant {bound} : index\n    scf.for %i = %zero to %bound")
            constant = constant.replace("scf.for %j = %zero to %m", "scf.for %j = %zero to %bound")
            path.write_text(constant)
            report = invoke(tool, path, "--sequence-analysis")
            assert not report["error"] and report["prepared"], report
            emitted = invoke(tool, path, "--insert-logical")
            assert emitted.count("scf.for") == 2
            assert emitted.count("pto.textract") == source.count("pto.textract")
            assert emitted.count("pto.tmatmul") == source.count("pto.tmatmul")
            metrics.append(tuple(report[key] for key in ("ports", "cells", "expressions", "emitted")) +
                           (len(emitted.splitlines()), emitted.count("pto.logical_")))
        assert metrics[0] == metrics[1], metrics
        # Without the boundary readers, and with different read/write banks,
        # the regional placement gate cannot prove every local crossing adjacent.
        # Keep this unsupported placement transactional instead of inserting an
        # over-ordering local barrier. The positive fixture above supplies the
        # cross-pipe reader paths explicitly.
        unsupported = "\n".join(line for line in source.splitlines()
                                if 'test.label = "I"' not in line and 'test.label = "U"' not in line)
        unsupported = unsupported.replace("%slot = arith.remui %sum, %banks",
                                          "%slot = arith.remui %i, %banks", 1)
        position = unsupported.index("    scf.for %j")
        unsupported = unsupported[:position] + unsupported[position:].replace(
            "%readslot = arith.remui %sum, %banks", "%readslot = arith.remui %j, %banks", 1)
        path.write_text(unsupported)
        rejected = invoke(tool, path)
        assert not rejected["accepted"] and rejected["unchanged_on_failure"], rejected
    print(f"mixed regional: {count} physical-conflict closures, trip-independent sizes and placement rejection passed")


if __name__ == "__main__":
    main()
