# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Serial compiler/control audit. No device compilation or execution is performed.

Inputs and tools are pinned by content; every attempted stage keeps its output,
diagnostics and resource receipt. Watchdogs are measurement limits, not class
exclusions or evidence of event-ID infeasibility.
"""
import argparse
from collections import Counter
import hashlib
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import shutil
import subprocess
import time


OP = re.compile(r'^\s*(?:%[^=\n]+\s*=\s*)?"?((?:pto|arith|scf)\.[\w_]+)\b')
PROFILES = ("frontier-profile", "frontier-control-profile", "frontier-allocation-work")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def metrics(path):
    counts = Counter()
    records = set()
    with path.open() as stream:
        for line in stream:
            match = OP.match(line)
            if match:
                counts[match[1]] += 1
            if "pto.endpoint_families =" in line:
                records.update(re.findall(r"\brecord = (\d+) : i64", line))
    return {"bytes": path.stat().st_size, "operations": dict(counts),
            "arithmetic": sum(n for name, n in counts.items() if name.startswith("arith.")),
            "selects": counts["arith.select"], "ifs": counts["scf.if"],
            "sets": sum(counts[k] for k in ("pto.logical_set", "pto.set_flag", "pto.set_flag_dyn")),
            "waits": sum(counts[k] for k in ("pto.logical_wait", "pto.wait_flag", "pto.wait_flag_dyn")),
            "barriers": counts["pto.barrier"], "textual_record_labels": len(records)}


def run_stage(directory, name, command, timeout, memory_mib):
    # Affinity also bounds LLVM's automatic CPU-count discovery in an external
    # backend whose command line does not expose --mlir-disable-threading.
    cpu = min(os.sched_getaffinity(0))

    def limits():
        limit = memory_mib * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.sched_setaffinity(0, {cpu})

    def terminate(child):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()

    start = time.monotonic()
    status = "finished"
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[key] = "1"
    with (directory / (name + ".out")).open("wb") as out, (directory / (name + ".err")).open("wb") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, env=environment,
                                 start_new_session=True, preexec_fn=limits)
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "measurement_timeout"
            terminate(child)
        except BaseException:
            terminate(child)
            raise
    report = {"command": command, "status": status, "returncode": child.returncode,
              "success": status == "finished" and child.returncode == 0,
              "seconds": time.monotonic() - start, "timeout_seconds": timeout,
              "memory_limit_bytes": memory_mib * 1024 * 1024, "cpu_affinity": [cpu],
              "outcome_kind": ("timeout" if status != "finished" else
                               "signal" if child.returncode < 0 else
                               "success" if child.returncode == 0 else "nonzero_exit")}
    if not report["success"]:
        with (directory / (name + ".err")).open(errors="replace") as stream:
            report["diagnostics"] = [line[:1800].rstrip() for line in islice(
                (line for line in stream if "error:" in line), 4)]
    save(directory / (name + ".json"), report)
    return report


def profiles(path):
    result = {}
    with path.open(errors="replace") as stream:
        for line in stream:
            for prefix in PROFILES:
                if line.startswith(prefix + " "):
                    result.setdefault(prefix, []).append(json.loads(line[len(prefix) + 1:]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, required=True)
    parser.add_argument("--emitter", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--memory-mib", type=int, default=4096)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt("campaign interrupted by signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.memory_mib <= 0:
        parser.error("resource limits must be positive")
    root = Path(__file__).resolve().parents[4]
    git = shutil.which("git")
    if git is None:
        parser.error("git executable was not found")
    planner = args.planner.resolve(strict=True)
    emitter = args.emitter.resolve(strict=True) if args.emitter else None
    cases = json.loads(args.manifest.read_text())
    if isinstance(cases, dict):
        cases = cases["cases"]
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in ids):
        parser.error("case IDs must be unique, simple directory names")
    inputs = [(root / case["input"]).resolve(strict=True) for case in cases]
    for case, source in zip(cases, inputs):
        if digest(source) != case["sha256"]:
            parser.error("input hash changed: " + case["id"])
    tools = {str(path): digest(path) for path in (planner, emitter) if path}
    args.output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=root, text=True, timeout=30).strip()
    working_diff = subprocess.check_output([git, "diff", "--no-ext-diff"], cwd=root, timeout=30)
    metadata = {"checkout_revision": revision,
                "working_diff_sha256": hashlib.sha256(working_diff).hexdigest(),
                "script_sha256": digest(Path(__file__)), "tools": tools, "inputs": cases,
                "scope": "serial compiler/control audit; no device execution", "alias_policy": "may-not-alias",
                "eligible_ids": list(range(6)), "timeout_seconds": args.timeout,
                "memory_limit_bytes": args.memory_mib * 1024 * 1024,
                "cpu_affinity": [min(os.sched_getaffinity(0))], "peak_rss_measured": False}
    save(args.output / "metadata.json", metadata)
    rows = []
    for case, source in zip(cases, inputs):
        directory = args.output / case["id"]
        directory.mkdir()
        row = {"id": case["id"], "sha256": case["sha256"], "stages": {}}
        stages = [("logical", [str(planner), "--mlir-disable-threading",
                   "--pto-frontier-analysis=profile-stages=true gm-alias=may-not-alias allocation-ids=0,1,2,3,4,5",
                   str(source)]),
                  ("allocation", [str(planner), "--mlir-disable-threading",
                   "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5", str(directory / "logical.out")])]
        if emitter:
            stages.append(("cpp", [str(emitter), "--pto-level=level3", "--enable-insert-sync=false",
                           "--enable-plan-memory=false", str(directory / "allocation.out"),
                           "-o", str(directory / "kernel.cpp")]))
        for name, command in stages:
            report = run_stage(directory, name, command, args.timeout, args.memory_mib)
            row["stages"][name] = report
            if not report["success"]:
                break
            if name != "cpp":
                row[name + "_metrics"] = metrics(directory / (name + ".out"))
            if name == "logical":
                row["profiles"] = profiles(directory / "logical.err")
            if name == "cpp":
                kernel = directory / "kernel.cpp"
                row["cpp_metrics"] = {"bytes": kernel.stat().st_size,
                                      "lines": sum(1 for _ in kernel.open())}
        rows.append(row)
        save(args.output / "results.json", rows)
        print(case["id"], {k: v["success"] for k, v in row["stages"].items()}, flush=True)
    for path, expected in tools.items():
        if digest(Path(path)) != expected:
            raise RuntimeError("tool changed during campaign: " + path)
    for case, source in zip(cases, inputs):
        if digest(source) != case["sha256"]:
            raise RuntimeError("input changed during campaign: " + case["id"])
    (args.output / "COMPLETE").write_text("All attempted outcomes retained; no device execution.\n")


if __name__ == "__main__":
    main()
