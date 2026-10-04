# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Serial Step 0 audit on hashed, prepared inputs accepted by InsertSync."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, directory, name, timeout):
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (directory / (name + ".stdout")).open("w") as out, (directory / (name + ".stderr")).open("w") as err:
        with subprocess.Popen(command, stdout=out, stderr=err, env=env, start_new_session=True) as child:
            try:
                status = "ok" if child.wait(timeout=timeout) == 0 else "error"
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                status = "timeout"
    return {"command": command, "status": status, "returncode": child.returncode,
            "seconds": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    tool = str(args.tool.resolve())
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["cases"]
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"attempted": len(cases), "accepted": 0, "step0_passed": 0,
               "functions": 0, "failures": [], "families": {}, "totals": {},
               "unphased_kinds": {}, "manifest_sha256": digest(args.manifest),
               "tool_sha256": digest(tool), "semantics": "shared-input preservation, not exact F*"}
    totals = collections.Counter()
    unphased = collections.Counter()
    families = collections.defaultdict(collections.Counter)
    for number, case in enumerate(cases):
        directory = args.output / f"{number:04d}"
        directory.mkdir()
        prepared = Path(case["input"])
        if digest(prepared) != case["sha256"]:
            raise RuntimeError(f"prepared input hash changed: {prepared}")
        if case.get("source") and digest(case["source"]) != case["source_sha256"]:
            raise RuntimeError(f"original input hash changed: {case['source']}")
        result = {"case": case}
        result["existing"] = run([tool, "--existing-check", str(prepared)], directory, "existing", args.timeout)
        family = families[case["family"]]
        family["attempted"] += 1
        if result["existing"]["status"] == "ok":
            summary["accepted"] += 1
            family["accepted"] += 1
        passed = True
        for policy in ("may-not-alias", "may-alias"):
            name = "step0-" + policy
            result[name] = run([tool, "--gm-alias=" + policy, "--step0-json", str(prepared)],
                               directory, name, args.timeout)
            passed &= result[name]["status"] == "ok"
        if passed:
            first = [json.loads(line) for line in (directory / "step0-may-not-alias.stdout").read_text().splitlines()]
            second = [json.loads(line) for line in (directory / "step0-may-alias.stdout").read_text().splitlines()]
            if first != second or not all(f["step0"] == "preserved" for f in first):
                raise RuntimeError(f"accesses differ by alias policy: {case['id']}")
            if len(first) != case["functions"]:
                raise RuntimeError(f"function count changed: {case['id']}")
            summary["step0_passed"] += 1
            family["step0_passed"] += 1
            summary["functions"] += len(first)
            for function in first:
                unphased.update(function["unphased_kinds"])
                totals.update({k: v for k, v in function.items() if isinstance(v, int)})
        if result["existing"]["status"] != "ok" or not passed:
            summary["failures"].append({"id": case["id"], "directory": str(directory)})
        if digest(prepared) != case["sha256"]:
            raise RuntimeError(f"analysis mutated input: {prepared}")
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if number % 100 == 0:
            print(f"{number + 1}/{len(cases)}: failures={len(summary['failures'])}", flush=True)
    summary["families"] = dict(families)
    summary["totals"] = totals
    summary["unphased_kinds"] = unphased
    if digest(tool) != summary["tool_sha256"]:
        raise RuntimeError("audit binary changed during campaign")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return bool(summary["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
