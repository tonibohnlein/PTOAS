# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare guarded rotating commands with all concrete physical conflicts."""
import itertools
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def invoke(tool, path, mode="--structured-trace"):
    run = subprocess.run([tool, mode, str(path)], capture_output=True,
                         text=True, check=False, timeout=90)
    assert run.returncode == 0, run.stderr + run.stdout
    return run.stdout if mode == "--insert-logical" else json.loads(run.stdout)


def validate(report, g, h, n, offset):
    assert report["accepted"], report
    trace = report["trace"]
    assert not trace["error"], trace
    payloads = [event for event in trace["events"] if event["kind"] == "payload"]
    assert [event["label"] for event in payloads] == (["L" if h else "B", "C"] * n if g else [])
    effects = []
    for event in payloads:
        iteration, = event["coordinates"]
        if event["label"] in ("L", "B"):
            effects.append(({("mat", 0)}, {("left", iteration % 2)}))
        else:
            effects.append(({("left", (iteration + offset) % 2), ("right", 0)}, {("acc", 0)}))
    edges = native([event["pipe"] for event in payloads])
    for a, (reads, writes) in enumerate(effects):
        for b in range(a + 1, len(effects)):
            later_reads, later_writes = effects[b]
            if writes & (later_reads | later_writes) or reads & later_writes:
                edges.add((2 * a + 1, 2 * b))
    required = closure(2 * len(payloads), edges)[0]
    commands = []
    gap = 0
    for event in trace["events"]:
        if event["kind"] == "payload":
            gap += 1
            continue
        assert event["gap"] == gap
        if event["kind"] == "barrier" and event["pipe"] == 6:
            assert gap == len(payloads)
            continue
        command = dict(event)
        if event["kind"] != "barrier":
            command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                   tuple(event.get("members", [])))
        commands.append(command)
    actual = closure_with_commands([event["pipe"] for event in payloads], commands)
    assert actual == [row & ~(1 << i) for i, row in enumerate(required)], (g, h, n, offset, actual, required)


def main():
    tool, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    count = 0
    with tempfile.TemporaryDirectory(prefix="guarded-rotating-") as scratch:
        path = Path(scratch) / "case.pto"
        # All guard valuations; zero/one/short/multiple rotations; offset residues
        # plus negative and large encoded offsets. Neither oracle imports the scan.
        for g, h, n, offset in itertools.product((0, 1), (0, 1), (0, 1, 2, 5), (-1, 0, 1, 2, 1000000001)):
            program = source.replace("array<i64: 1, 1, 3, 0>", f"array<i64: {g}, {h}, {n}, {offset}>")
            path.write_text(program)
            validate(invoke(tool, path), g, h, n, offset)
            count += 1
        # A branch-local deterministic guard must be replayed/masked at the entry
        # cut, while each endpoint stays in its original arm.
        local = source.replace("// LOCAL_GUARD", "%local = arith.xori %g, %h : i1")
        local = local.replace("scf.if %h", "scf.if %local")
        for g, h in itertools.product((0, 1), repeat=2):
            path.write_text(local.replace("array<i64: 1, 1, 3, 0>", f"array<i64: {g}, {h}, 3, 0>"))
            validate(invoke(tool, path), g, g ^ h, 3, 0)
            count += 1
        # Inactive arm-local overflow poison must be masked before entering
        # invocation-level quotient arithmetic. MLIR folding does not model
        # poison, so verify the emitted masking structure as well as the trace.
        poison = source.replace("%offset: index)", "%offset: index, %x: i64)")
        poison = poison.replace("array<i64: 1, 1, 3, 0>", "array<i64: 0, 0, 3, 0, 9223372036854775807>")
        poison = poison.replace("// LOCAL_GUARD", "%one64 = arith.constant 1 : i64\n"
                                "        %overflow = arith.addi %x, %one64 overflow<nsw> : i64\n"
                                "        %poison_guard = arith.cmpi slt, %overflow, %one64 : i64")
        poison = poison.replace("scf.if %h", "scf.if %poison_guard")
        path.write_text(poison)
        validate(invoke(tool, path), 0, 0, 3, 0)
        emitted = invoke(tool, path, "--insert-logical")
        additions = set(re.findall(r"(%[\w]+) = arith.addi [^\n]+overflow<nsw>", emitted))
        comparisons = {result for result, operand in re.findall(
            r"(%[\w]+) = arith.cmpi slt, (%[\w]+),", emitted) if operand in additions}
        assert any(re.search(r"arith.select %arg0, " + re.escape(value) + r", %false", emitted)
                   for value in comparisons), emitted
        offset_poison = source.replace("array<i64: 1, 1, 3, 0>",
                                       "array<i64: 0, 0, 3, 9223372036854775807>")
        declarations = []
        kept = []
        for line in offset_poison.splitlines():
            if any(token in line for token in ("%sum =", "%readslot =", "%readleft =")):
                declarations.append(line)
            else:
                kept.append(line)
        offset_poison = "\n".join(kept).replace("// LOCAL_GUARD",
            "%local_offset = arith.addi %offset, %one overflow<nsw> : index\n"
            + "\n".join(declarations).replace("%i, %offset", "%i, %local_offset"))
        path.write_text(offset_poison)
        validate(invoke(tool, path), 0, 0, 3, 0)
        emitted = invoke(tool, path, "--insert-logical")
        # Exact modular normalization may avoid replaying the flagged addition
        # altogether. If it is replayed before the loop, track its dependency
        # through normalization and require the outer guard to mask that value.
        entry_emitted, loop_emitted = emitted.split("scf.for", 1)
        assert "overflow<nsw>" in loop_emitted, emitted
        tainted = set()
        masked = False
        for line in entry_emitted.splitlines():
            match = re.search(r"(%[\w]+) = (arith\.[\w]+) (.*)", line)
            if not match:
                continue
            value, operation, operands = match.groups()
            inputs = set(re.findall(r"%[\w]+", operands))
            if "overflow<nsw>" in operands or inputs & tainted:
                if operation == "arith.select" and operands.startswith("%arg0,"):
                    masked = True
                else:
                    tainted.add(value)
        assert "overflow<nsw>" not in entry_emitted or masked, emitted
        count += 2
        # Repeatedly alternating presence is outside immutable guards and finite
        # monotone boundary slicing (a single equality is now supported there).
        varying = source.replace("// LOCAL_GUARD", "%parity = arith.remui %i, %banks : index\n"
                                 "        %varying = arith.cmpi eq, %parity, %zero : index")
        varying = varying.replace("scf.if %h", "scf.if %varying")
        # Removing the consumer leaves the writer's distance-two WAW cover
        # non-adjacent on its pipe; a barrier would impose an extra order.
        nonadjacent = "\n".join(line for line in source.splitlines() if "pto.tmatmul" not in line)
        path.write_text(varying)
        report = invoke(tool, path)
        assert not report["accepted"] and report["unchanged_on_failure"], report
        path.write_text(nonadjacent)
        report = invoke(tool, path)
        assert report["accepted"], report
    print(f"guarded rotating: {count} physical-conflict traces and nonadjacent insertion and guard rejection passed")


if __name__ == "__main__":
    main()
