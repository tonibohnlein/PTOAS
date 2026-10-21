# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Original-point adapter checks; no alternate demand reduction or allocation."""

import argparse
import json
from pathlib import Path
from assess_whole_kernel_access import annotate, digest, invoke
from check_compact_persistent_contract import trace
from compact_vector_oracle import command_order, model, required
from check_compact_persistent_contract import annotated
from reaching_access_research import require


def fixture(outer, inner):
    return f"""module attributes {{pto.target_arch = "a3"}} {{
func.func @points(%A: !pto.ptr<f32, gm>, %C: !pto.ptr<f32, gm>) attributes {{pto.kernel_kind = \
#pto.kernel_kind<vector>}} {{
 %zero = arith.constant 0 : index
 %one = arith.constant 1 : index
 %two = arith.constant 2 : index
 %three = arith.constant 3 : index
 %outer = arith.constant {2 + 2 * outer} : index
 %inner = arith.constant {1 + 2 * inner} : index
 %eight = arith.constant 8 : index
 %rows = arith.constant 64 : index
 %base = arith.constant 0 : i64
 %input = pto.make_tensor_view %A, shape = [%rows, %eight], strides = [%eight, %one] : !pto.tensor_view<64x8xf32>
 %output = pto.make_tensor_view %C, shape = [%rows, %eight], strides = [%eight, %one] : !pto.tensor_view<64x8xf32>
 %tile = pto.alloc_tile addr = %base : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, \
v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>
 scf.for %i = %two to %outer step %two {{
  scf.for %j = %one to %inner step %two {{
   %row = arith.addi %i, %j : index
   %source = pto.partition_view %input, offsets = [%row, %zero], sizes = [%one, %eight] : \
!pto.tensor_view<64x8xf32> -> !pto.partition_tensor_view<1x8xf32>
   pto.tload ins(%source : !pto.partition_tensor_view<1x8xf32>) outs(%tile : !pto.tile_buf<loc=vec, \
dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
   pto.tadd ins(%tile, %tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, \
blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, \
cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%tile : \
!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, \
fractal=512, pad=0>)
   %target = pto.partition_view %output, offsets = [%row, %zero], sizes = [%one, %eight] : \
!pto.tensor_view<64x8xf32> -> !pto.partition_tensor_view<1x8xf32>
   pto.tstore ins(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, \
blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%target : !pto.partition_tensor_view<1x8xf32>)
  }}
 }}
 return
}}
}}
"""


def qualify_trace(args, original, emitted, directory):
    expected = model(original)
    actual = trace(args, emitted, directory / "trace.pto")
    payloads = [x for x in actual["events"] if x["kind"] == "payload"]
    require(
        [(x["type"], x["coordinates"], x["pipe"]) for x in payloads]
        == [(x["type"], x["coordinates"], x["pipe"]) for x in expected],
        "occurrences/cuts changed",
    )
    wanted, _, scalar = required(expected)
    observed, reuse = command_order(actual, scalar)
    # The command oracle returns strict reachability; required() is reflexive.
    observed = [row | (1 << i) for i, row in enumerate(observed)]
    require(
        all((a & b) == a for a, b in zip(wanted, observed)),
        "emitted plan lost modeled order",
    )
    # command_order checks unique publications/consumers and graph acyclicity.
    excess = [
        (a, b)
        for a in range(len(expected))
        for b in range(a + 1, len(expected))
        if observed[2 * a + 1] >> (2 * b) & 1 and not wanted[2 * a + 1] >> (2 * b) & 1
    ]
    return {
        "payloads": len(payloads),
        "commands": len(actual["events"]) - len(payloads),
        "matching": True,
        "physical_reuse": reuse,
        "excess_pairs": excess,
    }, actual


def run_case(args, text, folder, points=False):
    folder.mkdir(exist_ok=True)
    attrs = (
        'test.contract_math_only, test.contract_strategy = "expanded", '
        "test.contract_loop = 0 : i64, test.contract_exports = 4 : i64, test.contract_emit"
    )
    path = folder / "input.pto"
    path.write_text(annotate(text, attrs))
    run = invoke(
        [args.trace_tool, "--compact-region-contract", path], folder / "logical.pto", 35
    )
    require(run["status"] == "success", f"original-point preparation failed: {run}")
    result, _ = qualify_trace(args, text, Path(run["stdout"]).read_text(), folder)
    result["run"] = run
    items = model(text)
    query = annotated(text, [dict(item, phase=item["type"]) for item in items], [])
    query = annotate(
        query,
        'test.contract_strategy = "expanded", test.contract_loop = 0 : i64, '
        "test.contract_exports = 1 : i64",
    )
    query_path = folder / "covers.pto"
    query_path.write_text(query)
    query_run = invoke(
        [args.trace_tool, "--compact-region-contract", query_path],
        folder / "covers.json",
        35,
    )
    require(query_run["status"] == "success", str(query_run))
    data = json.loads(Path(query_run["stdout"]).read_text())
    reach, covers, _ = required(items)
    require(not data.get("error") and not data.get("retained_demand_error"), str(data))
    require(
        {tuple(edge) for edge in data["retained_demands"]}
        == {(a // 2, b // 2) for a, b in covers},
        "actual expanded covers differ",
    )
    expected_rows = [
        "".join("1" if row >> j & 1 else "0" for j in range(len(reach)))
        for row in reach
    ]
    require(data["event_rows"] == expected_rows, "expanded all-event queries differ")
    result["actual_covers"] = len(covers)
    if points:
        path = folder / "invalid-points.pto"
        attrs = (
            'test.contract_math_only, test.contract_strategy = "expanded", '
            "test.contract_loop = 0 : i64, test.contract_exports = 0 : i64, test.contract_point_checks"
        )
        path.write_text(annotate(text, attrs))
        check = invoke(
            [args.trace_tool, "--compact-region-contract", path],
            folder / "points.json",
            35,
        )
        require(check["status"] == "success", "point rejection probe failed")
        data = json.loads(Path(check["stdout"]).read_text())
        require(
            all(
                data.get(k)
                for k in (
                    "valid",
                    "missing_frame_refused",
                    "duplicate_frame_refused",
                    "negative_point_refused",
                    "changing_activity_refused",
                    "retained_owner_reusable",
                    "false_filter_empty",
                )
            ),
            str(data),
        )
        result["point_checks"] = data
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for outer, inner in ((0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)):
        row = run_case(
            args,
            fixture(outer, inner),
            args.output / f"o{outer}-i{inner}",
            points=bool(outer and inner),
        )
        row.update(outer=outer, inner=inner)
        rows.append(row)
    report = {
        "cases": rows,
        "optimizer_sha256": digest(args.optimizer),
        "tool_sha256": digest(args.trace_tool),
        "scope": "bounded original-point matching and modeled order; physical allocation separate",
    }
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print("expanded original-coordinate adapter checks passed")


if __name__ == "__main__":
    main()
