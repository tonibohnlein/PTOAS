# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Alias attribution fixture with an explicit overlap-permitting pointer ABI.

The prepared production ports require disjoint GM allocations. Overlap here is
synthetic evidence for the shared importer/transfer, not qualification of those
ports under a wider ABI. No expert protocol or production input is rewritten.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from assess_whole_kernel_access import compile_stage
from check_whole_kernel_access import compare_commands
from whole_kernel_reference import reference

SOURCE = """module attributes {pto.target_arch = "a3", pto.kernel_kind = #pto.kernel_kind<vector>} {
func.func @alias_fixture(%A: !pto.ptr<f32, gm>, %B: !pto.ptr<f32, gm>)
 attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
 %zero = arith.constant 0 : index
 %one = arith.constant 1 : index
 %n = arith.constant 16 : index
 %addr = arith.constant 0 : i64
 %far = arith.constant 4096 : i64
 %a = pto.make_tensor_view %A, shape = [%n, %n], strides = [%n, %one] : !pto.tensor_view<16x16xf32>
 %b = pto.make_tensor_view %B, shape = [%n, %n], strides = [%n, %one] : !pto.tensor_view<16x16xf32>
 %pa = pto.partition_view %a, offsets = [%zero, %zero], sizes = [%n, %n] : !pto.tensor_view<16x16xf32>
 %pb = pto.partition_view %b, offsets = [%zero, %zero], sizes = [%n, %n] : !pto.tensor_view<16x16xf32>
 %src = pto.alloc_tile addr = %addr : !pto.tile_buf<vec, 16x16xf32>
 %dst = pto.alloc_tile addr = %far : !pto.tile_buf<vec, 16x16xf32>
 pto.tstore ins(%src : !pto.tile_buf<vec, 16x16xf32>) outs(%pa : !pto.partition_tensor_view<16x16xf32>)
 pto.tload ins(%pb : !pto.partition_tensor_view<16x16xf32>) outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
 pto.tload ins(%pa : !pto.partition_tensor_view<16x16xf32>) outs(%dst : !pto.tile_buf<vec, 16x16xf32>)
 return
}
}
"""


def items():
    def effect(space, base, mode, begin):
        return {
            "space": space,
            "base": base,
            "mode": mode,
            "intervals": [(begin, begin + 1024)],
        }

    return [
        {
            "phase": i,
            "coordinates": [],
            "pipe": pipe,
            "operation": operation,
            "active": True,
            "effects": effects,
        }
        for i, (pipe, operation, effects) in enumerate(
            [
                (5, "pto.tstore", [effect(6, -1, "read", 0), effect(1, 0, "write", 0)]),
                (
                    4,
                    "pto.tload",
                    [effect(1, 1, "read", 0), effect(6, -1, "write", 4096)],
                ),
                (
                    4,
                    "pto.tload",
                    [effect(1, 0, "read", 0), effect(6, -1, "write", 4096)],
                ),
            ]
        )
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("optimizer", "trace-tool", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=35)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("Positive watchdog required")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "input.pto"
    source.write_text(SOURCE)
    report = []
    for policy, option in (("ordinary", ""), ("completion", " completion-only=true")):
        stage = compile_stage(
            args,
            args.output,
            policy,
            source,
            "--pto-frontier-analysis=gm-alias=may-alias allocation-ids=0,1,2,3,4,5"
            + option,
        )
        row = {
            "policy": policy,
            "logical": stage,
            "layouts": [],
            "scope": "synthetic overlap-permitting ABI",
        }
        if stage["status"] == "success":
            for layout in ("disjoint", "B=A"):
                mapped = deepcopy(items())
                if layout == "B=A":
                    mapped[1]["effects"][0]["base"] = 0
                reach, _ = reference(mapped)
                folder = args.output / (policy + "-" + layout.replace("=", "-"))
                folder.mkdir(exist_ok=True)
                check = compare_commands(
                    args, folder, Path(stage["output"]), reach, mapped, set()
                )
                row["layouts"].append({"layout": layout, "check": check})
                if check["status"] != "passed":
                    raise RuntimeError(f"Alias coverage check failed: {check}")
        report.append(row)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if not any(row["layouts"] for row in report):
        raise RuntimeError("No executable plan was checked")


if __name__ == "__main__":
    main()
