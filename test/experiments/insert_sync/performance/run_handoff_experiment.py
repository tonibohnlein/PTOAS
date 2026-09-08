#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Run the native handoff experiment on frozen emitted benchmark plans.

This does not regenerate InsertSync seeds or run a device. Both seed and trial
use the same parser/printer, so changed SSA numbering is not an optimization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mechanisms(path):
    from measure import measure
    counts = measure(path, [], mode="static", normalize_gm_pipes=True)["static"]["counts"]
    barriers = {}
    for key, count in counts.items():
        if key.startswith("barrier:"):
            pipe = re.search(r"PIPE_[A-Z0-9]+", key).group()
            barriers[pipe] = barriers.get(pipe, 0) + count
    return {"sets": counts.get("pto.set_flag", 0), "waits": counts.get("pto.wait_flag", 0),
            "PIPE_ALL": barriers.pop("PIPE_ALL", 0), "named": barriers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pto-test-opt", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_root.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    binary = args.pto_test_opt.resolve()
    binary_hash = digest(binary)
    keys = set()
    for campaign in args.campaign:
        inputs = sorted(campaign.glob("*/*/output.pto"))
        if not inputs:
            raise ValueError(f"no emitted seeds in {campaign}")
        for source in inputs:
            case, arm = source.parent.parent.name, source.parent.name
            key = f"{case}/{arm}"
            if key in keys:
                raise ValueError(f"duplicate seed {key}")
            keys.add(key)
            directory = args.output / key
            directory.mkdir(parents=True)
            frozen = directory / "input.pto"
            frozen.write_bytes(source.read_bytes())
            row = {"fixture": case, "arm": arm, "source": str(source.resolve()),
                   "source_sha256": digest(frozen), "status": "pass", "commands": []}
            # Reuse the benchmark's existing exact syntax bridge for GM-only
            # pipe initialization. Preserve original bytes and record the bridge;
            # both arms receive identical operation attributes/operands.
            from measure import normalize_gm_pipe_assembly
            assembly, normalized = normalize_gm_pipe_assembly(frozen.read_text())
            parsed = frozen
            if normalized:
                parsed = directory / "parse.pto"
                parsed.write_text(assembly)
                row["parser_bridge"] = {"operations": normalized, "sha256": digest(parsed),
                                        "kind": "existing-gm-only-generic-assembly"}
            for optimize in (False, True):
                name = "trial" if optimize else "baseline"
                output = directory / f"{name}.pto"
                command = [str(binary), "--mlir-disable-threading", "--mlir-print-op-on-diagnostic=false",
                           str(parsed), "-o", str(output)]
                if optimize:
                    command.append("--pto-experiment-handoff-planning=mmad-chains=true")
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                (directory / f"{name}.stderr").write_text(result.stderr)
                row["commands"].append({"argv": command, "exit": result.returncode})
                if result.returncode:
                    row["status"] = "failed"
                    break
            if row["status"] == "pass":
                row["baseline_sha256"] = digest(directory / "baseline.pto")
                row["trial_sha256"] = digest(directory / "trial.pto")
                row["changed"] = row["baseline_sha256"] != row["trial_sha256"]
                row["before"] = mechanisms(directory / "baseline.pto")
                row["after"] = mechanisms(directory / "trial.pto")
                row["decisions"] = [line.split("remark: ", 1)[1] for line in result.stderr.splitlines()
                                    if "remark: InsertSync handoff experiment:" in line]
            rows.append(row)
    report = {"pto_test_opt_sha256": binary_hash, "binary_unchanged": digest(binary) == binary_hash,
              "scope": "native experiment on frozen emitted plans; no device execution",
              "rows": rows}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    failed = sum(row["status"] != "pass" for row in rows)
    changed = sum(row.get("changed", False) for row in rows)
    print(f"{len(rows)} rows; {failed} failures; {changed} changed plans")
    return bool(failed or not report["binary_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
