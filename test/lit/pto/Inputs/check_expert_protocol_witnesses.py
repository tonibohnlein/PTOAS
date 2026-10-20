# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Pin minimal expert-port counterexamples under the supplied event model.

This is a model regression, never a native-kernel miscompilation claim. It reads
original port layouts, not Frontier generators. Native lowering/issue guarantees
not represented here remain qualification obligations.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from compact_vector_oracle import command_order, model, required


def describe(items, index):
    item = items[index]
    return {"payload": index, **{k: v for k, v in item.items() if k != "scalar"}}


def overlap(first, second):
    result = []
    for a in first["effects"]:
        for b in second["effects"]:
            if (a["space"], a["base"]) != (b["space"], b["base"]):
                continue
            if not (a["write"] or b["write"]):
                continue
            for lo, hi in a["ranges"]:
                for start, end in b["ranges"]:
                    if max(lo, start) < min(hi, end):
                        result.append(
                            {
                                "space": a["space"],
                                "base": a["base"],
                                "bytes": [max(lo, start), min(hi, end)],
                            }
                        )
    return result


def reuse_witness(trace, graph):
    payloads = sum(e["kind"] == "payload" for e in trace["events"])
    commands = [e for e in trace["events"] if e["kind"] != "payload"]
    consumed = {}
    for i, command in enumerate(commands):
        if command["kind"] == "barrier":
            continue
        key = (command["source_pipe"], command["target_pipe"], command["physical_id"])
        vertex = 2 * payloads + i
        if command["kind"] == "wait":
            consumed[key] = i
        elif key in consumed:
            previous = consumed[key]
            if not (graph[2 * payloads + previous] >> vertex & 1):
                return {
                    "resource": list(key),
                    "prior_wait_command": previous,
                    "next_set_command": i,
                    "prior_wait": commands[previous],
                    "next_set": command,
                    "consumption_reaches_republication": False,
                    "scalar_value_prerequisites_included": True,
                }
    return None


def check(tool, prepared, output, name):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    root = prepared / name
    original = (root / "input.pto").read_text()
    expert = (root / "expert.pto").read_text()
    if name == "gdn_chunk_cumsum":

        def small(text):
            return text.replace(
                "%c10 = arith.constant 8", "%c10 = arith.constant 2"
            ).replace("%c11 = arith.constant 128", "%c11 = arith.constant 3")

        original, expert = small(original), small(expert)
    expected = model(original)
    _, covers, scalar = required(expected)
    annotated = expert.replace("pto.get_block_idx", "arith.constant 0 : i64").replace(
        "pto.get_subblock_idx", "arith.constant 0 : i64"
    )
    annotated, count = re.subn(
        r"(func\.func\s+@\w+\([^\n]*?\)\s*)attributes\s*\{",
        r"\1attributes {test.trace_existing, ",
        annotated,
        count=1,
    )
    if count != 1:
        raise ValueError("Cannot annotate expert port")
    path = directory / "expert.pto"
    path.write_text(annotated)
    command = [str(tool), "--structured-trace", str(path)]
    result = subprocess.run(
        command, text=True, capture_output=True, check=False, timeout=25
    )
    (directory / "trace.json").write_text(result.stdout)
    (directory / "trace.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(result.stderr)
    document = json.loads(result.stdout)
    trace = document["trace"]
    if not document["accepted"] or trace["error"]:
        raise ValueError(document)
    payloads = [e for e in trace["events"] if e["kind"] == "payload"]
    if [(e["type"], e["coordinates"], e["pipe"]) for e in payloads] != [
        (e["type"], e["coordinates"], e["pipe"]) for e in expected
    ]:
        raise ValueError("Original occurrence mapping differs")
    graph, reuse = command_order(trace, scalar, include_commands=True)
    missing = sorted(
        (a, b)
        for a, b in covers
        if a % 2 == 1 and b % 2 == 0 and not (graph[a] >> b & 1)
    )
    if not missing:
        raise AssertionError(
            "Expected uncovered minimal cover disappeared; requalify baseline"
        )
    a, b = missing[0]
    witness = {
        "event_pair": [a, b],
        "source": describe(expected, a // 2),
        "target": describe(expected, b // 2),
        "overlap": overlap(expected[a // 2], expected[b // 2]),
        "is_required_graph_cover": True,
        "covered_by_expert_commands": False,
    }
    if not witness["overlap"]:
        raise AssertionError("Witness must have a direct modeled physical conflict")
    if name == "group_norm":
        assert (a, b) == (9, 10)
        assert [witness[x]["operation"] for x in ("source", "target")] == [
            "pto.tmov",
            "pto.tadd",
        ]
    else:
        assert (a, b) == (35, 36)
        assert [witness[x]["operation"] for x in ("source", "target")] == [
            "pto.tsetval",
            "pto.tstore",
        ]
    failed_reuse = reuse_witness(trace, graph)
    if name == "gdn_chunk_cumsum":
        assert failed_reuse and failed_reuse["resource"] == [1, 0, 0]
        assert (
            failed_reuse["prior_wait_command"],
            failed_reuse["next_set_command"],
        ) == (3, 4)
    elif failed_reuse:
        raise AssertionError("Unexpected GroupNorm reuse discrepancy")
    pins = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [root / "input.pto", root / "expert.pto", root / "port.json", tool]
    }
    return {
        "name": name,
        "qualification": "expert-port-unqualified-under-supplied-event-model",
        "native_compilation_or_execution": "not_performed",
        "command": command,
        "pins": pins,
        "payloads": len(expected),
        "missing_ci_covers": len(missing),
        "minimal_cover": witness,
        "physical_reuse": reuse,
        "reuse_witness_including_scalar_prerequisites": failed_reuse,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [
        check(args.trace_tool.resolve(), args.prepared_root, args.output, name)
        for name in ("group_norm", "gdn_chunk_cumsum")
    ]
    (args.output / "witnesses.json").write_text(json.dumps(results, indent=2) + "\n")
    print("expert-port minimal covers and cumsum causal-reuse witnesses reproduced")


if __name__ == "__main__":
    main()
