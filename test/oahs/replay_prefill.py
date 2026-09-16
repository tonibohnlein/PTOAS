#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Replay the pinned real Qwen3 prefill captures through native synchronization.

This is a compiler integration cohort, not all PyPTO/pypto-lib coverage or device
correctness. Preserve refusals/timeouts in the denominator. Never insert sync a
second time during C++ emission. Each subprocess runs serially without MLIR
threading; timeout is a harness limit, not a constructor fallback.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def inventory(path):
    text = path.read_text()
    directions = Counter(re.findall(
        r"pto\.(set_flag|wait_flag)\[<([^>]+)>,\s*<([^>]+)>,\s*<([^>]+)>\]", text))
    barriers = Counter(re.findall(r"pto\.barrier\s*<([^>]+)>", text))
    return {
        "set": len(re.findall(r"\bpto\.set_flag\b", text)),
        "wait": len(re.findall(r"\bpto\.wait_flag\b", text)),
        "barrier": len(re.findall(r"\bpto\.barrier\b", text)),
        "functions": len(re.findall(r"\bfunc\.func\b", text)),
        "events": [{"kind": k[0], "source": k[1], "target": k[2], "key": k[3], "count": v}
                   for k, v in sorted(directions.items())],
        "barriers_by_pipe": dict(sorted(barriers.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    repo = Path(__file__).resolve().parents[2]
    build = args.build.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tools = {
        "ptoas": build / "tools/ptoas/ptoas",
        "opt": build / "tools/pto-test-opt/pto-test-opt",
        "analysis": build / "tools/pto-test-opt/pto-oahs-native-test",
        "selected": build / "tools/pto-test-opt/pto-oahs-selected-test",
    }
    # LLVM derives its default pool size from Linux affinity. Restrict child
    # compilers as well as explicit test-driver/opt threading to one worker.
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    cohort = repo / "test/samples/Qwen3_14BPrefillA3"
    report = {
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "tracked_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=repo)).hexdigest(),
        "changed_sources": {name: digest(repo / name) for name in
                            subprocess.check_output(["git", "ls-files", "-m", "-o", "--exclude-standard"],
                                                    cwd=repo, text=True).splitlines()
                            if (repo / name).is_file()},
        "tools": {key: {"path": str(path), "sha256": digest(path)}
                  for key, path in tools.items()},
        "runtime_libraries": {str(path.relative_to(build)): digest(path)
                              for pattern in ("python/ptoas/_core*.so",
                                              "python/ptoas/mlir/_mlir_libs/libPTOASCompiler.so")
                              for path in build.glob(pattern)},
        "provenance": (cohort / "README.md").read_text(),
        "timeout_seconds": args.timeout,
        "rows": [],
    }

    def run(directory, name, command, stdout=None):
        command = [str(x) for x in command]
        log = directory / (name + ".log")
        start = time.monotonic()
        with log.open("w") as err, (stdout or directory / (name + ".stdout")).open("w") as out:
            proc = subprocess.Popen(command, stdout=out, stderr=err,
                                    start_new_session=True)
            try:
                status = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                status = "TIMEOUT"
        result = {"command": command, "status": status,
                  "seconds": time.monotonic() - start, "log": str(log)}
        if status == 0 and stdout:
            result["sha256"] = digest(stdout)
        return result

    for source in sorted((cohort / "kernels").rglob("*.pto")):
        directory = output / source.stem
        directory.mkdir()
        prepared = directory / "prepared.pto"
        row = {"name": source.stem, "source": str(source), "sha256": digest(source)}
        row["prepare"] = run(directory, "prepare", [tools["ptoas"],
            "--pto-level=level3", "--pto-arch=a3", "--emit-pto-ir", source, "-o", prepared])
        if row["prepare"]["status"] == 0:
            row["prepared_sha256"] = digest(prepared)
            row["analysis"] = run(directory, "analysis", [tools["analysis"], "--analyze", prepared])
            for arm in ("existing", "default", "handoff"):
                synchronized = directory / (arm + ".pto")
                option = "--pto-insert-sync" + ("=algorithm=" + arm if arm != "default" else "")
                command = [tools["opt"], "--mlir-disable-threading", option, prepared]
                row[arm] = run(directory, arm, command, synchronized)
                if row[arm]["status"] == 0:
                    row[arm]["inventory"] = inventory(synchronized)
                    if arm != "default":
                        row[arm + "_lowering"] = run(directory, arm + "_lowering", [
                            tools["ptoas"], "--pto-level=level3", "--pto-arch=a3",
                            synchronized, "-o", directory / (arm + ".cpp")])
            row["default_equal_existing"] = (
                row["existing"]["status"] == row["default"]["status"] == 0 and
                (directory / "existing.pto").read_bytes() == (directory / "default.pto").read_bytes())
        report["rows"].append(row)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(source.stem, {k: v["status"] for k, v in row.items()
                            if isinstance(v, dict) and "status" in v}, flush=True)
    # A partial cohort stays available for diagnosis, but is not a passing gate.
    return 0 if report["rows"] and all(
        row.get("handoff_lowering", {}).get("status") == 0 and
        row.get("analysis", {}).get("status") == 0 and
        row.get("existing_lowering", {}).get("status") == 0 and
        row.get("default_equal_existing") for row in report["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
