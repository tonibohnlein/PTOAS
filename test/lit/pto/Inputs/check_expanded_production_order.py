# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare actual Elementwise plans under explicit disjoint/overlapping layouts."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from assess_whole_kernel_access import compile_stage, digest
from check_compact_persistent_contract import trace
from compact_vector_oracle import model, required, command_order
from reaching_access_research import require


def layout(items, alias):
    result = deepcopy(items)
    if alias:
        for item in result:
            for effect in item["effects"]:
                if effect["space"] == "gm" and effect["base"] == "%C":
                    effect["base"] = "%A"
    return result


def compare_trace(args, text, items, folder):
    observed = trace(
        args,
        text.replace("pto.get_block_idx", "arith.constant 0 : i64").replace(
            "pto.get_subblock_idx", "arith.constant 0 : i64"
        ),
        folder / "trace.pto",
    )
    payloads = [x for x in observed["events"] if x["kind"] == "payload"]
    require(
        [(x["type"], x["coordinates"], x["pipe"]) for x in payloads]
        == [(x["type"], x["coordinates"], x["pipe"]) for x in items],
        "production payload mapping differs",
    )
    expected, _, scalar = required(items)
    actual, reuse = command_order(observed, scalar)
    actual = [row | 1 << i for i, row in enumerate(actual)]
    require(
        all((x & y) == x for x, y in zip(expected, actual)),
        "production plan misses modeled requirements",
    )
    pairs = {
        (a, b)
        for a in range(len(items))
        for b in range(a + 1, len(items))
        if actual[2 * a + 1] >> (2 * b) & 1
    }
    needs = {
        (a, b)
        for a in range(len(items))
        for b in range(a + 1, len(items))
        if expected[2 * a + 1] >> (2 * b) & 1
    }
    return pairs, {
        "payloads": len(items),
        "commands": len(observed["events"]) - len(items),
        "excess_pairs": sorted(pairs - needs),
        "physical_reuse": reuse,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--baseline-optimizer", type=Path, required=True)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = (
        Path(__file__).resolve().parents[4]
        / "test/npu_validation/benchmarks/auto_sync/prepared/elementwise_pipeline/input.pto"
    )
    original = model(source.read_text())
    rows = []
    for policy in ("may-not-alias", "may-alias"):
        folder = args.output / policy
        folder.mkdir(exist_ok=True)
        previous = compile_stage(
            SimpleNamespace(optimizer=args.baseline_optimizer, timeout=35),
            folder,
            "previous",
            source,
            "--pto-frontier-analysis=completion-only=true "
            f"gm-alias={policy} prepare-allocation=false",
        )
        require(previous["status"] == "success", str(previous))
        ordinary = compile_stage(
            SimpleNamespace(optimizer=args.baseline_optimizer, timeout=35),
            folder,
            "previous-ordinary",
            source,
            f"--pto-frontier-analysis=gm-alias={policy} prepare-allocation=false",
        )
        current = args.production / "elementwise_pipeline" / policy
        for overlap in (False, True) if policy == "may-alias" else (False,):
            items = layout(original, overlap)
            directory = folder / ("overlap" if overlap else "disjoint")
            directory.mkdir(exist_ok=True)
            plans = {}
            for label, path in (
                ("previous-completion", Path(previous["output"])),
                ("production-logical", current / "production/logical.pto"),
                ("production-physical", current / "production/physical.pto"),
                ("current-completion", current / "completion/logical.pto"),
            ):
                at = directory / label
                at.mkdir(exist_ok=True)
                pairs, metrics = compare_trace(args, path.read_text(), items, at)
                plans[label] = (pairs, metrics)
            same_policy = {"previous_status": ordinary["status"], "run": ordinary}
            if ordinary["status"] == "success":
                previous_folder = directory / "previous-ordinary"
                previous_folder.mkdir(exist_ok=True)
                old_pairs, old_metrics = compare_trace(
                    args, Path(ordinary["output"]).read_text(), items, previous_folder
                )
                now_pairs = plans["production-logical"][0]
                same_policy.update(
                    removed=sorted(old_pairs - now_pairs),
                    added=sorted(now_pairs - old_pairs),
                    previous=old_metrics,
                )
            require(
                plans["previous-completion"][0] == plans["current-completion"][0],
                "explicit completion changed prerequisite relation",
            )
            prior = plans["previous-completion"][0]
            after = plans["production-logical"][0]
            require(
                after == plans["production-physical"][0],
                "allocation changed payload prerequisites",
            )
            require(
                plans["production-physical"][1]["physical_reuse"]["status"] == "passed",
                "causal six-ID reuse not established",
            )
            rows.append(
                {
                    "policy": policy,
                    "layout": "C aliases A" if overlap else "disjoint",
                    "layout_contract": "structural stress outside disjoint-argument port ABI"
                    if overlap
                    else "admitted disjoint-argument port ABI",
                    "same_policy": same_policy,
                    "comparison_policies": "ordinary exact/selected-model production vs previous explicit completion",
                    "previous_run": previous,
                    "removed_pairs": sorted(prior - after),
                    "added_pairs": sorted(after - prior),
                    "plans": {name: values[1] for name, values in plans.items()},
                }
            )
    receipt = {
        "rows": rows,
        "source_sha256": digest(source),
        "tool_sha256": digest(args.trace_tool),
        "baseline_optimizer_sha256": digest(args.baseline_optimizer),
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        "Elementwise production coverage, logical matching and causal six-ID reuse passed"
    )


if __name__ == "__main__":
    main()
