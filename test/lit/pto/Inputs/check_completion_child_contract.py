# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Reproduce the compact K-child adapter and six-ID certificate obstruction."""

import argparse
import json
import os
from pathlib import Path
import resource
import shutil

from assess_whole_kernel_access import annotate, digest, invoke


def diagnostic(args, bound):
    label = "bound" if bound else "unbound"
    folder = args.output / label
    folder.mkdir(parents=True, exist_ok=True)
    attributes = (
        'test.contract_gm_alias = "may-not-alias", test.contract_math_only, '
        "test.contract_region = 5 : i64, test.contract_completion_probe"
    )
    if bound:
        attributes += ", test.contract_completion_bound"
    source = folder / "input.pto"
    source.write_text(annotate(args.input.read_text(), attributes))
    row = invoke(
        [args.trace_tool, "--compact-region-contract", source],
        folder / "result.json",
        35,
    )
    if row["status"] != "success":
        raise AssertionError(row)
    result = json.loads(Path(row["stdout"]).read_text())
    if result.get("backend") != "sequence" or result.get("required_outer_loops") != 1:
        raise AssertionError(result)
    if bound:
        expected = {
            "prepared": True,
            "allocation_certificate": True,
            "materialized": True,
            "choice_families": 0,
            "outside_endpoints": 0,
            "private_state": False,
            "prepared_complete_invocation": False,
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise AssertionError(result)
        if not result["nonsingleton_endpoints"]:
            raise AssertionError("Grouped record contract was not exercised")
        if not result.get("physical_certificate_decoded"):
            if "regional lane assignment not certified within directed capacity" not in Path(row["stderr"]).read_text():
                raise AssertionError("A different certificate failure requires investigation")
        elif not result.get("physical_assignment") or any(
            flag not in range(6) for record in result["physical_assignment"] for flag in record["ids"]
        ):
            raise AssertionError("Accepted certificate has no checked six-ID mapping")
    elif result.get("prepared") or "enclosing repeat coordinates" not in result.get("preparation_error", ""):
        raise AssertionError(result)
    row.update(input_sha256=digest(source), result=result)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("trace-tool", "input", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    args.trace_tool = Path(shutil.which(str(args.trace_tool)) or args.trace_tool).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    receipt = {
        "scope": "Explicit diagnostic requests; not production timing or a capacity lower bound",
        "tool_sha256": digest(args.trace_tool),
        "input_sha256": digest(args.input),
        "watchdog_seconds": 35,
        "workers": 1,
        "unbound": diagnostic(args, False),
        "bound": diagnostic(args, True),
    }
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("compact completion child: original-frame qualification and physical certificate outcome checked")


if __name__ == "__main__":
    main()
