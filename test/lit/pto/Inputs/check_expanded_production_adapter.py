# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check original-frame production endpoints against independent byte effects."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from assess_whole_kernel_access import annotate, compile_stage, digest, invoke
from check_finite_allocation import physical_check
from check_whole_kernel_access import independent
from compact_vector_oracle import command_order
from reaching_access_research import require
from whole_kernel_reference import reference


def word(args, text, folder):
    path = folder / "trace.pto"
    path.write_text(annotate(text, "test.trace_existing"))
    run = invoke(
        [args.trace_tool, "--structured-trace", path], folder / "trace.json", 35
    )
    require(run["status"] == "success", str(run))
    data = json.loads(Path(run["stdout"]).read_text())
    require(data.get("accepted") and not data["trace"]["error"], str(data))
    return data["trace"]


def ci(rows, count):
    return {
        (a, b)
        for a in range(count)
        for b in range(a + 1, count)
        if rows[2 * a + 1] >> (2 * b) & 1
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("optimizer", "trace-tool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-optimizer", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[4]
    rows = []
    for name, limit, full in (
        ("elementwise_pipeline", "%c8", 8),
        ("group_norm", "%c4", 4),
    ):
        source = (
            root
            / "test/npu_validation/benchmarks/auto_sync/prepared"
            / name
            / "input.pto"
        )
        original = source.read_text()
        for count, subblock in ((0, 0), (1, 0), (2, 0), (full, 0), (full, 1)):
            text = original.replace("pto.get_block_idx", "arith.constant 0 : i64")
            text = text.replace(
                "pto.get_subblock_idx", f"arith.constant {subblock} : i64"
            )
            if count != full:
                text = text.replace(
                    "    %c0 =",
                    f"    %point_limit = arith.constant {count} : index\n    %c0 =",
                    1,
                )
                text = text.replace(f"to {limit} step", "to %point_limit step")
            items, exact, _, scalar = independent(text, "vector", count)
            for policy in ("may-not-alias", "may-alias"):
                folder = args.output / name / f"n{count}-s{subblock}" / policy
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / "input.pto"
                path.write_text(original if count == full else text)
                config = SimpleNamespace(optimizer=args.optimizer, timeout=35)
                logical = compile_stage(
                    config,
                    folder,
                    "logical",
                    path,
                    f"--pto-frontier-analysis=gm-alias={policy}",
                )
                require(logical["status"] == "success", str(logical))
                physical = compile_stage(
                    config,
                    folder,
                    "physical",
                    Path(logical["output"]),
                    "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5",
                )
                if physical["status"] != "success":
                    require(
                        count != full
                        and "allocation adapter not implemented yet"
                        in Path(physical["stderr"]).read_text(),
                        str(physical),
                    )
                traces = []
                for label, run in (("logical", logical), ("physical", physical)):
                    if run["status"] == "success":
                        at = folder / label
                        at.mkdir(exist_ok=True)
                        traces.append(
                            word(
                                args,
                                Path(run["output"])
                                .read_text()
                                .replace("pto.get_block_idx", "arith.constant 0 : i64")
                                .replace(
                                    "pto.get_subblock_idx",
                                    f"arith.constant {subblock} : i64",
                                ),
                                at,
                            )
                        )
                expected = [(i["phase"], i["coordinates"], i["pipe"]) for i in items]
                require(
                    [
                        (e["type"], e["coordinates"], e["pipe"])
                        for e in traces[0]["events"]
                        if e["kind"] == "payload"
                    ]
                    == expected,
                    "original occurrence mismatch",
                )
                if len(traces) == 2:
                    physical_check(
                        {
                            "accepted": True,
                            "allocated": True,
                            "trace": traces[0],
                            "physical": traces[1],
                        },
                        set(range(6)),
                    )
                actual, _ = command_order(traces[0], scalar)
                upper, _ = reference(
                    items, scalar, possible_alias=policy == "may-alias"
                )
                for wanted in (exact, upper):
                    require(
                        all(
                            (bits & ~(actual[i] | (1 << i))) == 0
                            for i, bits in enumerate(wanted)
                        ),
                        "modeled prerequisite missing",
                    )
                selected = ci(actual, len(items))
                row = dict(
                    kernel=name,
                    count=count,
                    subblock=subblock,
                    alias=policy,
                    input_sha256=digest(source),
                    logical=logical,
                    physical=physical,
                    payloads=len(items),
                    executed_commands=len(traces[0]["events"]) - len(items),
                    matching="passed",
                    causal_reuse="passed"
                    if len(traces) == 2
                    else "unavailable-allocation-export",
                    selected_model_coverage="passed",
                    excess_ci=len(selected - ci(exact, len(items))),
                )
                if args.baseline_optimizer and count == full and subblock == 0:
                    prior = compile_stage(
                        SimpleNamespace(optimizer=args.baseline_optimizer, timeout=35),
                        folder,
                        "previous-completion",
                        path,
                        f"--pto-frontier-analysis=completion-only=true gm-alias={policy}",
                    )
                    require(prior["status"] == "success", str(prior))
                    at = folder / "previous-trace"
                    at.mkdir(exist_ok=True)
                    previous_word = word(
                        args,
                        Path(prior["output"])
                        .read_text()
                        .replace("pto.get_block_idx", "arith.constant 0 : i64")
                        .replace(
                            "pto.get_subblock_idx", f"arith.constant {subblock} : i64"
                        ),
                        at,
                    )
                    previous, _ = command_order(previous_word, scalar)
                    old = ci(previous, len(items))
                    old_ordinary = compile_stage(
                        SimpleNamespace(optimizer=args.baseline_optimizer, timeout=35),
                        folder,
                        "previous-ordinary",
                        path,
                        f"--pto-frontier-analysis=gm-alias={policy}",
                    )
                    row["same_policy"] = {"run": old_ordinary}
                    if old_ordinary["status"] == "success":
                        at = folder / "previous-ordinary-trace"
                        at.mkdir(exist_ok=True)
                        prior_text = (
                            Path(old_ordinary["output"])
                            .read_text()
                            .replace("pto.get_block_idx", "arith.constant 0 : i64")
                            .replace(
                                "pto.get_subblock_idx",
                                f"arith.constant {subblock} : i64",
                            )
                        )
                        prior_trace = word(args, prior_text, at)
                        prior_reach, _ = command_order(prior_trace, scalar)
                        prior_pairs = ci(prior_reach, len(items))
                        row["same_policy"].update(
                            removed=sorted(prior_pairs - selected),
                            added=sorted(selected - prior_pairs),
                        )
                    row["comparison"] = dict(
                        policy="previous explicit completion vs current ordinary production",
                        previous=prior,
                        removed=sorted(old - selected),
                        added=sorted(selected - old),
                        previous_commands=len(previous_word["events"]) - len(items),
                        previous_excess_ci=len(old - ci(exact, len(items))),
                    )
                rows.append(row)
    report = dict(
        cases=rows,
        optimizer_sha256=digest(args.optimizer),
        tool_sha256=digest(args.trace_tool),
        scope=(
            "Original disjoint-argument port contracts; scratch remains a supplied enclosure. "
            "May-alias additionally covers the independent all-root possible-overlap upper model."
        ),
    )
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "expanded production endpoints: 20 guard/trip/alias cases; "
        "matching passed, physical outcomes recorded separately"
    )


if __name__ == "__main__":
    main()
