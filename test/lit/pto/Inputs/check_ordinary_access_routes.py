# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise shared ordinary footprints through exact analysis and insertion."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands
from check_periodic_demands import closure, native


def require(condition, detail):
    if not condition:
        raise AssertionError(detail)


def run(tool, path, mode, alias="may-not-alias"):
    return subprocess.run([tool, "--gm-alias=" + alias, mode, str(path)],
                          capture_output=True, text=True, timeout=30)


def document(output):
    lines = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    require(len(lines) == 1, output)
    return lines[0]


def trace_existing(tool, path, source):
    lines = source.splitlines()
    functions = [i for i, line in enumerate(lines) if "func.func @ordinary(" in line]
    require(len(functions) == 1, "missing original ordinary function")
    index = functions[0]
    if "attributes {" in lines[index]:
        lines[index] = lines[index].replace("attributes {", "attributes {test.trace_existing, ", 1)
    else:
        require(lines[index].endswith(" {"), lines[index])
        lines[index] = lines[index][:-2] + " attributes {test.trace_existing} {"
    path.write_text("\n".join(lines) + "\n")
    result = run(tool, path, "--structured-trace")
    require(result.returncode == 0, result.stderr)
    report = document(result.stdout)
    require(report["accepted"] and not report["trace"]["error"], report)
    return report["trace"]


def check_original_commands(tool, path, source, prefix_barrier=False):
    # This original barrier precedes every payload. It is preserved protocol,
    # not a discharge of the subsequent tile load/add/store hazards.
    if prefix_barrier:
        source = source.replace("    %base =", "    pto.barrier #pto.pipe<PIPE_ALL>\n    %base =")
    path.write_text(source)
    inserted = run(tool, path, "--insert-logical")
    require(inserted.returncode == 0, inserted.stderr)
    original = trace_existing(tool, path, source)
    actual = trace_existing(tool, path, inserted.stdout)
    payloads = [event for event in original["events"] if event["kind"] == "payload"]
    require([event for event in actual["events"] if event["kind"] == "payload"] == payloads,
            "insertion changed an original payload")
    pipes = [event["pipe"] for event in payloads]
    require(pipes == [4, 1, 5] and all(event["coordinates"] == [] for event in payloads), payloads)
    require([(event["pipe"], event["gap"]) for event in original["events"]
             if event["kind"] == "barrier"] == ([(6, 0)] if prefix_barrier else []), original)
    commands = []
    barriers = []
    for event in actual["events"]:
        if event["kind"] == "payload":
            continue
        command = dict(event)
        if event["kind"] == "barrier":
            barriers.append((event["pipe"], event["gap"]))
        else:
            require(event["kind"] in ("set", "wait"), event)
            command["identity"] = (event["plan"], event["record"], event["source_pipe"], event["target_pipe"],
                                   event["source_ordinal"], tuple(event.get("members", [])))
        commands.append(command)
    expected_barriers = [(6, 0)] if prefix_barrier else []
    require(barriers in (expected_barriers, expected_barriers + [(6, 3)]), barriers)
    # Independently enumerate all conflicts of the one tile: W, RW, R.
    required = closure(6, native(pipes) | {(1, 2), (1, 4), (3, 4)})[0]
    require(closure_with_commands(pipes, commands) == [row & ~(1 << i) for i, row in enumerate(required)],
            "existing barrier masked a missing or excessive payload dependency")


def check_metadata_update(tool, path, source, effects, inserted):
    # Original valid shape changes from2x64 to1x64 between load and add.
    # Check each tile operand's descriptor at its original use, not one cached
    # allocation-wide shape. An unresolved GM store remains conservative.
    blocks = re.split(r"(?m)^  effect ", effects.split("\nordinary: phases=", 1)[0])[1:]
    require(len(blocks) == 6, effects)
    path.write_text(inserted)
    emitted_effects = run(tool, path, "--storage-effects")
    require(emitted_effects.returncode == 0, emitted_effects.stderr)
    emitted_blocks = re.split(r"(?m)^  effect ", emitted_effects.stdout.split("\nordinary: phases=", 1)[0])[1:]
    require(emitted_blocks == blocks, "insertion changed original effect geometry or descriptor state")
    for index, operation, mode, rows in ((1, "tload", "write", 2), (2, "tadds", "read", 1),
                                        (3, "tadds", "write", 1), (4, "tstore", "read", 1)):
        require(blocks[index].startswith(f"{index} pto.{operation} {mode} "), blocks[index])
        descriptor = next((line for line in blocks[index].splitlines() if line.startswith("    descriptor ")), "")
        require("base=absolute bytes=4 " in descriptor and descriptor.endswith(f"extents={rows}, 64 loops=0"),
                descriptor)
    original_ops = re.findall(r"(?m)^\s*(pto\.(?:tload|tadds|tstore|set_validshape))\b", source)
    inserted_ops = re.findall(r"(?m)^\s*(pto\.(?:tload|tadds|tstore|set_validshape))\b", inserted)
    require(original_ops == inserted_ops == ["pto.tload", "pto.set_validshape", "pto.tadds", "pto.tstore"],
            inserted_ops)
    path.write_text(source)
    recognition = run(tool, path, "--recognize")
    require(recognition.returncode == 0 and "source-unchanged; synchronization-insertion-not-run" in recognition.stdout,
            recognition.stderr + recognition.stdout)
    report = document(recognition.stdout)
    require([item["operation"] for item in report["payloads"]] == ["pto.tload", "pto.tadds", "pto.tstore"], report)
    check_original_commands(tool, path, source)


def check_unsupported_effect(tool, path, source):
    declaration = "  func.func private @unmodeled(!pto.ptr<f32, gm>)\n"
    source = source.replace("  func.func @", declaration + "  func.func @", 1)
    source = source.replace("    return", "    func.call @unmodeled(%input) : (!pto.ptr<f32, gm>) -> ()\n    return")
    path.write_text(source)
    refused = run(tool, path, "--insert-logical")
    require(refused.returncode != 0, "external effect without closed-child evidence accepted")
    boundary = subprocess.run([tool, "--recognition-boundary", str(path)], capture_output=True,
                              text=True, timeout=30)
    require(boundary.returncode == 0 and "recognition-boundary: rejected; source-unchanged" in boundary.stdout,
            boundary.stderr + boundary.stdout)


def main():
    tool, fixture = sys.argv[1:]
    tool = shutil.which(tool)
    require(tool is not None, "missing tracer executable")
    tool = str(Path(tool).resolve(strict=True))
    original = Path(fixture).read_text()
    with tempfile.TemporaryDirectory(prefix="ordinary-access-") as directory:
        path = Path(directory) / "case.pto"
        path.write_text(original)
        for alias, accepted in [("may-not-alias", True), ("may-alias", True)]:
            analysis = run(tool, path, "--explicit-analysis", alias)
            require(analysis.returncode == 0, analysis.stderr)
            doc = document(analysis.stdout)
            require(bool(doc["error"]) != accepted, doc)
            inserted = run(tool, path, "--insert-logical", alias)
            require((inserted.returncode == 0) == accepted, inserted.stderr)
            if accepted:
                require(doc["retained"] == [[0, 1], [1, 2]], doc)
                require(inserted.stdout.count("pto.logical_set") == 2, inserted.stdout)
                require(inserted.stdout.count("pto.logical_wait") == 2, inserted.stdout)
        check_original_commands(tool, path, original, prefix_barrier=True)
        # The same base's read/write aliases remain exact under either policy.
        same = original.replace("pto.make_tensor_view %output", "pto.make_tensor_view %input")
        path.write_text(same)
        for alias in ["may-alias", "may-not-alias"]:
            doc = document(run(tool, path, "--explicit-analysis", alias).stdout)
            require(not doc["error"] and doc["retained"] == [[0, 1], [1, 2]], doc)
        # A conditional transfer with a live outside store needs a guarded
        # handoff and based-GM storage, not an independent-GM discharge.
        guarded = same.replace("%scalar: f32)", "%scalar: f32, %guard: i1)")
        guarded = guarded.replace("    pto.tload", "    scf.if %guard {\n      pto.tload")
        guarded = guarded.replace("    pto.tadds ins(%tile, %scalar : !tile, f32) outs(%tile : !tile)", "    }")
        path.write_text(guarded)
        inserted = run(tool, path, "--insert-logical", "may-alias")
        require(inserted.returncode == 0, inserted.stderr)
        require(inserted.stdout.count("pto.logical_set") == 1, inserted.stdout)
        require(inserted.stdout.count("pto.logical_wait") == 1, inserted.stdout)
        # Runtime metadata and vector tails retain their shared buffer bound when
        # a narrower map is unavailable. Descriptor-only strong updates are modeled.
        changed = original.replace("    pto.tadds", "    pto.set_validshape %tile, %one, %cols : !tile\n    pto.tadds")
        oversized = original.replace("2x64", "1024x4096").replace("constant 2 : index", "constant 1024 : index")
        oversized = oversized.replace("constant 64 : index", "constant 4096 : index")
        wide = original.replace("2x64", "2x16384").replace("constant 64 : index", "constant 16384 : index")
        tail = original.replace("partition_tensor_view<2x64", "partition_tensor_view<2x56")
        tail = tail.replace("constant 64 : index", "constant 56 : index")
        for text in [changed, tail, oversized, wide]:
            path.write_text(text)
            effects = run(tool, path, "--storage-effects")
            require(effects.returncode == 0 and "all-materialized=" in effects.stdout, effects.stderr + effects.stdout)
            inserted = run(tool, path, "--insert-logical")
            require(inserted.returncode == 0, inserted.stderr)
            if text == changed:
                check_metadata_update(tool, path, text, effects.stdout, inserted.stdout)
        check_unsupported_effect(tool, path, original)
    print("ordinary access routes: exact, alias, guarded and metadata cases passed")


if __name__ == "__main__":
    main()
