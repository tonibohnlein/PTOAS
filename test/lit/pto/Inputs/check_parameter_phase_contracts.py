# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Qualify boundary/phase premises without assuming them in production analysis."""

import argparse
import json
from pathlib import Path
from assess_whole_kernel_access import annotate, digest, invoke
from check_prefill_protocol_capacity import bind_prefill
from check_compact_persistent_contract import annotated
from prefill_boundary_reference import source_gm_accesses
from pypto_region_oracle import interpret
from reaching_access_research import require
from survey_pypto_regions import child_source, specialize, validate


def signed_div(a, b):
    require(b > 0, "positive phase divisor required")
    return a // b if a >= 0 else -((-a) // b)


def decoded(w):
    micro = signed_div(w, 64)
    remainder = w - 64 * micro
    lane = signed_div(remainder, 8)
    group = remainder - 8 * lane
    return micro, lane, group, 8 * micro + lane


def phase_contract():
    comparisons = 0
    for start in range(48):
        for t in range((1023 - start) // 48 + 1):
            w = start + 48 * t
            a, b, c, dt = decoded(w)
            require(
                dt == start // 8 + 6 * t and c == start % 8, "active-prefix identity"
            )
            a4, b4, c4, dt4 = decoded(w + 192)
            require((a4, b4, c4, dt4) == (a + 3, b, c, dt + 24), "four-phase identity")
            require(
                0 <= 1024 * a + 128 * c + 16 * b <= 16384 - 16, "GM input row domain"
            )
            require(0 <= c * 640 <= 5120 - 640, "GM output column domain")
            comparisons += 1
    negative = {"start": -1, "first": decoded(-1), "four_visits_later": decoded(191)}
    require(
        decoded(-1)[2] != decoded(191)[2], "negative-start counterexample disappeared"
    )
    return {
        "comparisons": comparisons,
        "negative_counterexample": negative,
        "premise": "start in [0,47], positive step 48; caller evidence not supplied as input range metadata",
        "production_rewrite": False,
    }


def prefill_contract(root):
    path = (
        root
        / "test/npu_validation/benchmarks/auto_sync/prepared/causal_conv1d_prefill/input.pto"
    )
    text = path.read_text()
    rows = []
    for length in (4, 8, 12, 32):
        for initial in (0, 1):
            bound, bindings = bind_prefill(text, length, initial)
            effects = source_gm_accesses(bound)
            outputs = [
                x["view"]["offsets"][0]
                for x in effects
                if x["write"] and x["view"]["root"] == "%Y"
            ]
            cache_rows = [
                x["view"]["offsets"][0]
                for x in effects
                if x["write"] and x["view"]["root"] == "%state"
            ]
            require(
                outputs == list(range(length)),
                "prefill startup/final output rows differ",
            )
            require(cache_rows == [0, 1, 2], "prefill final cache writeback differs")
            state_reads = [
                x for x in effects if not x["write"] and x["view"]["root"] == "%state"
            ]
            require(
                len(state_reads) == (3 if initial else 0),
                "prefill initial-state path differs",
            )
            rows.append(
                {
                    "length": length,
                    "initial": initial,
                    "groups": length // 4,
                    "gm_transfers": len(effects),
                    "output_rows": outputs,
                    "cache_writeback_rows": cache_rows,
                    "bindings": bindings,
                }
            )
    return {
        "input_sha256": digest(path),
        "cases": rows,
        "scope": "independent integer control/GM boundary views, not full demand or native checks",
    }


def finite_finalize(args, folder):
    path = args.root / "test/lit/pto/Inputs/pypto_region_survey/qwen-finalize/input.pto"
    original = path.read_text()
    rows = []
    for work in (0, 16, 32, 48, 64, 1008):
        text = child_source(
            specialize(original, {"%arg3": 128, "%arg4": 0, "%arg5": 0, "%arg6": 48}),
            "qwen-finalize",
            work,
        )
        observations = {}
        items = interpret(text, observations)
        cells = sorted(
            {
                (e["space"], e["base"], lo)
                for item in items
                for e in item["effects"]
                for lo, _ in e["intervals"]
            }
        )
        supplied = annotated(text, items, cells)
        constants = ", ".join(
            f"array<i64: {line}, {next(iter(values))}>"
            for line, values in observations.items()
            if len(values) == 1
        )
        supplied = annotate(supplied, "test.contract_constants = [" + constants + "]")
        fixture = folder / f"work{work}.pto"
        fixture.write_text(supplied)
        run = invoke(
            [args.trace_tool, "--compact-region-contract", fixture],
            fixture.with_suffix(".json"),
            35,
        )
        require(run["status"] == "success", f"finite phase input refused: {run}")
        result = json.loads(Path(run["stdout"]).read_text())
        outcome = validate(result, items, cells)
        require(
            all(outcome.get(k) for k in ("presence", "closure", "covers")), str(outcome)
        )
        # Complete GM selectors may remain explicitly unavailable. Never empty.
        if not outcome["selector_error"]:
            require(outcome["selectors"], str(outcome))
        rows.append(
            {
                "work": work,
                "run": run,
                "input_sha256": digest(fixture),
                "checked": outcome,
            }
        )
    return rows


def finite_crossings(args, folder):
    original = (
        args.root / "test/lit/pto/Inputs/pypto_region_survey/qwen-finalize/input.pto"
    ).read_text()
    rows = []
    for count, active in ((0, 8), (1, 8), (2, 8), (4, 8), (2, 0)):
        text = specialize(
            original,
            {"%arg3": active, "%arg4": 0, "%arg5": 0, "%arg6": 48},
            (r"to %c1024 step", count * 48),
        )
        observations = {}
        items = interpret(text, observations)
        cells = sorted(
            {
                (e["space"], e["base"], lo)
                for item in items
                for e in item["effects"]
                for lo, _ in e["intervals"]
            }
        )
        valued = annotated(text, items, cells)
        bindings = ", ".join(
            f"array<i64: {line}, {next(iter(values))}>"
            for line, values in observations.items()
            if len(values) == 1
        )
        attrs = (
            'test.contract_strategy = "expanded", test.contract_loop = 0 : i64, '
            "test.contract_exports = 3 : i64, test.contract_constants = ["
            + bindings
            + "]"
        )
        path = folder / f"crossings-{count}-{active}.pto"
        path.write_text(annotate(valued, attrs))
        run = invoke(
            [args.trace_tool, "--compact-region-contract", path],
            path.with_suffix(".json"),
            35,
        )
        data = (
            json.loads(Path(run["stdout"]).read_text())
            if run["status"] == "success"
            else {}
        )
        outcome = (
            validate(data, items, cells) if data else {"status": "process-unavailable"}
        )
        if data and not data.get("error"):
            require(
                all(outcome.get(key) for key in ("presence", "closure", "covers")),
                str(outcome),
            )
            if not outcome["selector_error"]:
                require(outcome["selectors"], str(outcome))
        rows.append({"trips": count, "active": active, "run": run, "checked": outcome})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.root = Path(__file__).resolve().parents[4]
    args.output.mkdir(parents=True, exist_ok=True)
    data = {
        "phase": phase_contract(),
        "prefill": prefill_contract(args.root),
        "finite_finalize": finite_finalize(args, args.output),
        "finite_crossings": finite_crossings(args, args.output),
        "production_changes": False,
    }
    (args.output / "bounded.json").write_text(json.dumps(data, indent=2) + "\n")
    print("parameter and phase premises checked; no unsupplied bounds installed")


if __name__ == "__main__":
    main()
