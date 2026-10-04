# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Serial structural-recognition coverage on a frozen Step 0 corpus."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path, help="Completed Step 0 corpus directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", default=30.0, type=float)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    tool = args.tool.resolve()
    tool_hash = digest(tool)
    cases = json.loads(args.manifest.read_text())["cases"]
    args.output.mkdir(parents=True, exist_ok=False)
    counts = collections.Counter()
    issues = collections.Counter()
    failures = []
    for number, case in enumerate(cases):
        source = Path(case["input"])
        if digest(source) != case["sha256"]:
            raise RuntimeError(f"changed corpus input: {source}")
        directory = args.output / f"{number:04d}"
        directory.mkdir()
        command = [str(tool), "--recognize", str(source)]
        with (directory / "report.jsonl").open("w") as out, (directory / "stderr.txt").open("w") as err:
            try:
                process = subprocess.run(command, stdout=out, stderr=err, timeout=args.timeout, check=False)
                ok = process.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
        if not ok:
            failures.append({"case": case["id"], "reason": "recognition failed or timed out"})
            continue
        documents = [json.loads(line) for line in (directory / "report.jsonl").read_text().splitlines()
                     if line.startswith("{")]
        baseline = {value["function"]: value for line in
                    (args.baseline / f"{number:04d}" / "step0-may-not-alias.stdout").read_text().splitlines()
                    if line.startswith("{") for value in [json.loads(line)]}
        if set(baseline) != {doc["function"] for doc in documents}:
            raise RuntimeError(f"lost function in {case['id']}")
        for doc in documents:
            effects = [effect["id"] for payload in doc["payloads"] for effect in payload["effects"]]
            expected = baseline[doc["function"]]
            if (len(doc["payloads"]) != expected["phases"] or
                    sorted(effects) != list(range(expected["effects"]))):
                raise RuntimeError(f"lost/duplicated shared records in {case['id']}:{doc['function']}")
            counts["functions"] += 1
            counts["payloads"] += len(doc["payloads"])
            counts["effects"] += len(effects)
            counts["nodes"] += len(doc["nodes"])
            for node in doc["nodes"]:
                if not node["payload_count"]:
                    continue
                for attempt in node["attempts"]:
                    counts[attempt["route"] + ":" + attempt["state"]] += 1
                    analysis = attempt.get("analysis")
                    if analysis is not None:
                        state = "failed" if analysis["error"] else "ready"
                        counts["minimum-analysis:" + state] += 1
                        counts["minimum-analysis:records"] += len(analysis["retained"])
                    for issue in attempt["issues"]:
                        issues[attempt["route"] + ":" + issue["issue"]] += 1
            arithmetic = doc["arithmetic"]
            counts["arithmetic:" + arithmetic["state"]] += 1
            for issue in arithmetic.get("issues", []):
                issues["arithmetic:" + issue["issue"]] += 1
        if digest(source) != case["sha256"]:
            raise RuntimeError(f"recognition changed input: {source}")
        if (number + 1) % 100 == 0:
            print(f"{number + 1}/{len(cases)} failures={len(failures)}", flush=True)
    if digest(tool) != tool_hash:
        raise RuntimeError("diagnostic executable changed during run")
    summary = {"attempted": len(cases), "passed": len(cases) - len(failures), "failures": failures,
               "counts": dict(counts), "issues": dict(issues), "tool_sha256": tool_hash,
               "manifest_sha256": digest(args.manifest),
               "scope": "structural recognition and shared-record preservation; demand correctness checked separately"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
