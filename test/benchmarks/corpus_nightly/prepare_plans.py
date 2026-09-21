#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Prepare paired default-handoff/existing plans; no device execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--python", required=True, help="Configured compiler Python interpreter")
    parser.add_argument("--ptoas", type=Path, help="Defaults to BUILD/tools/ptoas/ptoas")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", help="Case IDs; default is every captured A3 module")
    parser.add_argument("--timeout", type=int, default=180, help="Seconds per host stage")
    args = parser.parse_args()
    package, build, out = args.package.resolve(), args.build.resolve(), args.out.resolve()
    if out.exists():
        raise SystemExit("Use a fresh output directory")
    out.mkdir(parents=True)
    cases = json.loads((package / "manifest.json").read_text())["cases"]
    known = {c["id"] for c in cases if c["target"] == "a3"}
    if args.cases and not set(args.cases) <= known:
        raise SystemExit("Unknown/non-A3 case IDs: " + str(set(args.cases) - known))
    driver = build / "tools/pto-test-opt/pto-oahs-selected-test"
    opt = build / "tools/pto-test-opt/pto-test-opt"
    ptoas = (args.ptoas or build / "tools/ptoas/ptoas").resolve()
    rows = []

    def run(command, stdout, stderr):
        start = time.monotonic()
        with stdout.open("w") as output, stderr.open("w") as log:
            try:
                rc = subprocess.run(command, stdout=output, stderr=log, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = 124
        return dict(command=list(map(str, command)), rc=rc, wall_seconds=time.monotonic()-start)

    for case in cases:
        if case["target"] != "a3" or (args.cases and case["id"] not in args.cases):
            continue
        source = package / case["input"]
        assert sha(source) == case["sha256"], case["id"]
        for arm in ("handoff", "existing"):
            dest = out / case["id"] / arm
            dest.mkdir(parents=True)
            plan = dest / "plan.pto"
            cmd = [driver, "--construct", source] if arm == "handoff" else [opt, "--mlir-disable-threading", "--pto-insert-sync=algorithm=existing", source]
            result = run(cmd, plan, dest / "construction.log")
            row = dict(case=case["id"], arm=arm, input_sha256=sha(source), construction=result)
            if result["rc"] == 0:
                row["plan_sha256"] = sha(plan)
                row["pin_match"] = arm != "handoff" or sha(plan) == case["handoff_pin_sha256"]
                if arm == "handoff":
                    functions = [dict(t.split("=", 1) for t in line.split() if "=" in t) for line in (dest / "construction.log").read_text().splitlines() if line.startswith("function=")]
                    row["functions"] = functions
                    row["reconstruction_pass"] = bool(functions) and all(f.get("construction") == f.get("reconstruction") == "1" for f in functions)
                if row["pin_match"] and row.get("reconstruction_pass", True):
                    cpp = dest / "kernel.cpp"
                    row["lowering"] = run([args.python, ptoas, "--pto-level=level3", "--pto-arch=a3", plan, "-o", cpp], dest / "lowering.stdout", dest / "lowering.log")
                    if row["lowering"]["rc"] == 0:
                        row["cpp_sha256"] = sha(cpp)
            row["passed"] = result["rc"] == 0 and row.get("pin_match", False) and row.get("reconstruction_pass", True) and row.get("lowering", {}).get("rc") == 0
            rows.append(row)
            temporary = out / "summary.json.new"
            temporary.write_text(json.dumps(rows, indent=2)+"\n")
            os.replace(temporary, out / "summary.json")
            print(case["id"], arm, "PASS" if row["passed"] else "FAIL", flush=True)
    raise SystemExit(0 if rows and all(r["passed"] for r in rows) else 1)


if __name__ == "__main__":
    main()
