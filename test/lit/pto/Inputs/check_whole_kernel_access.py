# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent concrete requirements versus complete small production plans."""

import argparse
import json
import os
from pathlib import Path
import re
import resource

from assess_whole_kernel_access import annotate, compile_stage, digest, invoke
from check_compact_persistent_contract import annotated, variant
from compact_persistent_oracle import (
    boundary,
    occurrences,
    required as persistent_required,
)
from compact_vector_oracle import command_order, model, required as vector_required
from pypto_region_oracle import interpret, mathematical
from survey_pypto_regions import local_probes, specialize


def inputs(root):
    prepared = root / "test/npu_validation/benchmarks/auto_sync/prepared"
    survey = root / "test/lit/pto/Inputs/pypto_region_survey"
    original = (prepared / "persistent_gemm/input.pto").read_text()
    for trips in (0, 1, 2):
        yield (
            "persistent_gemm",
            f"k{trips}",
            variant(original, 2, trips, 2, 0),
            "persistent",
            trips,
        )
    for name, limit in (
        ("elementwise_pipeline", "%c8"),
        ("group_norm", "%c4"),
        ("gdn_chunk_cumsum", "%c10"),
    ):
        original = (prepared / name / "input.pto").read_text()
        for trips in (0, 1, 2):
            text = original.replace(
                "    %c0 =",
                f"    %assess_limit = arith.constant {trips} : index\n    %c0 =",
                1,
            )
            text = text.replace(f"to {limit} step", "to %assess_limit step")
            if name == "gdn_chunk_cumsum":
                text = text.replace(
                    "    %c0 =",
                    "    %assess_inner = arith.constant 3 : index\n    %c0 =",
                    1,
                )
                text = text.replace("to %c11 step", "to %assess_inner step")
            text = text.replace("pto.get_block_idx", "arith.constant 0 : i64")
            text = text.replace("pto.get_subblock_idx", "arith.constant 0 : i64")
            yield name, f"trips{trips}", text, "vector", trips
    for name in ("qwen-down", "qwen-finalize", "deepseek-rms"):
        original = (survey / name / "input.pto").read_text()
        for trips in (0, 1, 2):
            if name == "qwen-down":
                args = {
                    "%arg3": 20,
                    "%arg4": trips,
                    "%arg5": 0,
                    "%arg6": 0,
                    "%arg7": 4352,
                    "%arg8": 0,
                    "%arg9": 20,
                }
                prefix = None
                # K0 does not initialize the accumulator: keep the outcome visible.
            elif name == "qwen-finalize":
                args = {"%arg3": 128, "%arg4": 0, "%arg5": 0, "%arg6": 48}
                prefix = (r"to %c1024 step", trips * 48)
            else:
                args = {"%arg3": 8, "%arg4": 0, "%arg5": 4}
                prefix = (r"to %c7168 step", trips * 3584)
            yield (
                name,
                f"trips{trips}",
                specialize(original, args, prefix),
                "pypto",
                trips,
            )
        if name == "qwen-finalize":
            args["%arg3"] = 0
            yield (
                name,
                "skipped",
                specialize(original, args, (r"to %c1024 step", 96)),
                "pypto",
                2,
            )


def independent(text, strategy, trips):
    if strategy == "persistent":
        items = occurrences(2, trips, 2, 0)
        items, reach, covers, _ = persistent_required(items)
        for item in items:
            item["operation"] = (
                "pto.tload"
                if item["phase"] < 4
                else "pto.textract"
                if item["phase"] < 6
                else "pto.tmatmul"
                if item["phase"] == 6
                else "pto.tmatmul.acc"
                if item["phase"] == 7
                else "pto.tstore"
            )
        return items, reach, covers, set()
    if strategy == "pypto":
        items = interpret(text)
        reach, covers = mathematical(items)
        return items, reach, covers, set()
    word = model(text)
    reach, covers, scalar = vector_required(word)
    arguments = re.findall(
        r"(%[\w]+): !pto.ptr", text.split("func.func", 1)[1].split(" attributes", 1)[0]
    )
    items = [
        {
            "phase": item["type"],
            "coordinates": item["coordinates"],
            "pipe": item["pipe"],
            "active": True,
            "operation": item["operation"],
            "effects": [
                {
                    "space": 1 if e["space"] == "gm" else 6,
                    "base": arguments.index(e["base"]) if e["base"] else -1,
                    "mode": "write" if e["write"] else "read",
                    "intervals": e["ranges"],
                }
                for e in item["effects"]
            ],
        }
        for item in word
    ]
    return items, reach, covers, scalar


def compare_commands(args, folder, source, expected, items, scalar):
    path = folder / (source.stem + "-trace.pto")
    path.write_text(annotate(source.read_text(), "test.trace_existing"))
    row = invoke(
        [args.trace_tool, "--structured-trace", path],
        path.with_suffix(".json"),
        args.timeout,
    )
    if row["status"] != "success":
        return row
    doc = json.loads(Path(row["stdout"]).read_text())
    if not doc.get("accepted") or doc.get("trace", {}).get("error"):
        row.update(
            status="trace-unavailable",
            diagnostic=doc.get("trace", {}).get("error", doc),
        )
        return row
    trace = doc["trace"]
    payload = [e for e in trace["events"] if e["kind"] == "payload"]
    keys = [(e["type"], e["coordinates"], e["pipe"]) for e in payload]
    wanted = [(e["phase"], e["coordinates"], e["pipe"]) for e in items]
    if keys != wanted:
        row.update(
            status="payload-map-mismatch", actual=len(keys), expected=len(wanted)
        )
        return row
    try:
        observed, reuse = command_order(trace, scalar)
        missing = [
            (a, b)
            for a, bits in enumerate(expected)
            for b in range(len(expected))
            if a != b and bits >> b & 1 and not observed[a] >> b & 1
        ]
        extra = [
            (a, b)
            for a, bits in enumerate(expected)
            for b in range(len(expected))
            if a != b and observed[a] >> b & 1 and not bits >> b & 1
        ]
        row.update(
            status="passed" if not missing else "coverage-mismatch",
            matching="passed",
            reuse=reuse,
            missing_event_pairs=len(missing),
            missing_examples=missing[:4],
            excess_ci=sum(a % 2 == 1 and b % 2 == 0 for a, b in extra),
            extra_examples=extra[:4],
            commands=len(trace["events"]) - len(payload),
        )
    except (ValueError, KeyError, AssertionError) as error:
        row.update(status="command-check-refusal", diagnostic=str(error))
    return row


def contract(args, folder, text, alias, items, expected, covers):
    cells = local_probes(items)
    # Include requested GM boundaries; unavailable stays unavailable.
    cells += sorted(
        {
            (e["space"], e["base"], lo)
            for item in items
            for e in item["effects"]
            if e["space"] == 1
            for lo, _ in e["intervals"]
        }
    )[:8]
    path = folder / "contract.pto"
    path.write_text(
        annotate(annotated(text, items, cells), f'test.contract_gm_alias = "{alias}"')
    )
    row = invoke(
        [args.trace_tool, "--compact-region-contract", path],
        folder / "contract.json",
        args.timeout,
    )
    if row["status"] != "success":
        return row
    result = json.loads(Path(row["stdout"]).read_text())
    if result.get("error"):
        row.update(
            status="oracle-valuation-unavailable"
            if result["error"] == "nonconstant occurrence presence"
            else "exact-export-unavailable",
            diagnostic=result["error"],
        )
        return row
    if result.get("present") != [True] * len(items):
        row.update(
            status="occurrence-export-unqualified",
            present=result.get("present"),
            cover_export_error=result.get("retained_demand_error"),
        )
        return row
    want = {(a // 2, b // 2) for a, b in covers}
    got = {tuple(p) for p in result.get("retained_demands", [])}
    rows = result.get("event_rows", [])
    row.update(
        cover_export_error=result.get("retained_demand_error"),
        actual_covers=len(got),
        concrete_covers=len(want),
        covers_equal=got == want,
        covers_extra=sorted(got - want)[:4],
        covers_missing=sorted(want - got)[:4],
        concrete_closure_covered=all(
            rows[a][b] == "1"
            for a, bits in enumerate(expected)
            for b in range(len(expected))
            if bits >> b & 1
        ),
        event_queries=len(expected) ** 2,
    )
    boundaries = result.get("boundaries", {})
    row["selectors"] = "unavailable" if boundaries.get("error") else "available"
    if not boundaries.get("error"):
        actual = [
            {
                k: {p: v for p, v in value.items() if v}
                if isinstance(value, dict)
                else value
                for k, value in r.items()
            }
            for r in boundaries["rows"]
        ]
        row["selectors_equal"] = actual == [boundary(items, c) for c in cells]
    return row


def run(args, name, label, text, strategy, trips, alias):
    folder = args.output / name / label / alias
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "input.pto"
    source.write_text(text)
    row = {
        "kernel": name,
        "configuration": label,
        "alias": alias,
        "input_sha256": digest(source),
        "scope": "Whole-kernel bounded derivative, disjoint GM allocations under supplied layouts",
    }
    try:
        items, reach, covers, scalar = independent(text, strategy, trips)
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as error:
        row.update(status="independent-access-unavailable", diagnostic=str(error))
        return row
    row.update(
        payloads=len(items),
        concrete_covers=len(covers),
        access_records=sum(len(i["effects"]) for i in items),
    )
    row["contract"] = contract(args, folder, text, alias, items, reach, covers)
    for policy in ("ordinary", "completion"):
        option = " completion-only=true" if policy == "completion" else ""
        logical = compile_stage(
            args,
            folder,
            policy,
            source,
            f"--pto-frontier-analysis=gm-alias={alias} allocation-ids=0,1,2,3,4,5{option}",
        )
        row[policy] = logical
        if logical["status"] == "success":
            row[policy]["semantics"] = compare_commands(
                args, folder, Path(logical["output"]), reach, items, scalar
            )
            physical = compile_stage(
                args,
                folder,
                policy + "-physical",
                Path(logical["output"]),
                "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5",
            )
            row[policy]["physical"] = physical
            if physical["status"] == "success":
                physical["semantics"] = compare_commands(
                    args, folder, Path(physical["output"]), reach, items, scalar
                )
    row["status"] = "checked-with-explicit-capability-outcomes"
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("optimizer", "trace-tool", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--timeout", type=int, default=35)
    p.add_argument("--cases", nargs="+")
    args = p.parse_args()
    if args.timeout < 1:
        p.error("Positive watchdog required")
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = []
    for name, label, text, strategy, trips in inputs(Path.cwd()):
        if args.cases and name not in args.cases:
            continue
        for alias in ("may-not-alias", "may-alias"):
            result.append(run(args, name, label, text, strategy, trips, alias))
            (args.output / "results.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(name, label, alias, result[-1]["status"], flush=True)


if __name__ == "__main__":
    main()
