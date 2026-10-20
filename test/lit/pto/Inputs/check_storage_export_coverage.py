# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression for exact cells, proven absence and unavailable GM exports."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tool = Path(shutil.which(str(args.trace_tool)) or args.trace_tool).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    source = r"""module attributes {pto.target_arch = "a3"} {
  func.func @finite(%p: !pto.ptr<f32, gm>, %q: !pto.ptr<f32, gm>)
      attributes {test.contract_occurrences = [array<i64: 0>],
                  test.contract_cells = [CELLS]} {
    %zero = arith.constant 0 : index
    %value = pto.load %p[%zero] : !pto.ptr<f32, gm> -> f32
    return
  }
}
"""
    rows = []
    for name, cells, alias in [
        ("present", "array<i64: 1, 0, 0>", "may-not-alias"),
        ("absent-byte", "array<i64: 1, 0, 8>", "may-not-alias"),
        ("absent-domain", "array<i64: 1, 1, 0>", "may-not-alias"),
        ("alias-unknown-domain", "array<i64: 1, 1, 0>", "may-alias"),
        ("invalid-policy", "array<i64: 1, 0, 0>", "invalid"),
    ]:
        path = args.output / (name + ".pto")
        path.write_text(
            source.replace("CELLS", cells).replace(
                "test.contract_occurrences",
                f'test.contract_gm_alias = "{alias}", test.contract_occurrences',
                1,
            )
        )
        command = [str(tool), "--compact-region-contract", str(path)]
        run = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        path.with_suffix(".json").write_text(run.stdout)
        path.with_suffix(".stderr").write_text(run.stderr)
        if name == "invalid-policy":
            if run.returncode != 1:
                raise AssertionError("Invalid alias policy was not rejected")
            rows.append({"case": name, "command": command, "status": "rejected"})
            continue
        if run.returncode:
            raise RuntimeError(run.stderr)
        result = json.loads(run.stdout)
        if name == "alias-unknown-domain":
            if not (result.get("error") or result.get("boundaries", {}).get("error")):
                raise AssertionError(
                    "Unknown cross-root displacement became an exact answer"
                )
            rows.append({"case": name, "command": command, "status": "unavailable"})
            continue
        if result.get("error") or result.get("boundaries", {}).get("error"):
            raise AssertionError(result)
        answer = result["boundaries"]["rows"][0]
        readers = [values for values in answer["first_readers"].values() if values]
        if name == "present":
            if readers != [[0]] or answer["first_writers"] or answer["last_writers"]:
                raise AssertionError(answer)
        elif any(
            any(value.values()) if isinstance(value, dict) else bool(value)
            for value in answer.values()
        ):
            raise AssertionError(answer)
        rows.append({"case": name, "command": command, "answer": answer})
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
