# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run pinned prepared inputs serially through analysis, allocation, and emission.

Timeouts are campaign cancellations, never recognizer class mismatches. The emitter
may be a separate build; both executable hashes are recorded explicitly.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time


def sha256(path):
    """Hash a pinned input or executable without loading its entire contents."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def invoke(arguments, directory, name, timeout):
    """Persist subprocess output and distinguish cancellation from compiler failure."""
    start = time.monotonic()
    with (directory / (name + ".out")).open("w") as stdout:
        with (directory / (name + ".err")).open("w") as stderr:
            try:
                result = subprocess.run(arguments, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
                status = "success" if result.returncode == 0 else "compiler_failure"
                if result.returncode < 0:
                    status = "process_signal"
                code = result.returncode
            except subprocess.TimeoutExpired:
                status = "campaign_timeout"
                code = None
    return {"status": status, "exit": code, "seconds": time.monotonic() - start}


def run_case(row, compiler, emitter, output, timeout):
    """Validate the pin and run one module through the three distinct passes."""
    item = row["case"]
    source = Path(item["input"]).resolve(strict=True)
    if sha256(source) != item["sha256"]:
        raise ValueError("Prepared input hash changed for case " + str(row["index"]))
    index = int(row["index"])
    if index < 0:
        raise ValueError("Negative case index")
    directory = output / str(index)
    directory.mkdir()
    result = {"index": index, "id": item["id"], "family": item["family"],
              "input": str(source), "sha256": item["sha256"]}
    result["logical"] = invoke([str(compiler), "--mlir-disable-threading",
        "--pto-frontier-analysis=gm-alias=may-not-alias", str(source)], directory, "logical", timeout)
    result["allocation"] = {"status": "not_run"}
    result["emission"] = {"status": "not_run"}
    if result["logical"]["status"] == "success":
        result["allocation"] = invoke([str(compiler), "--mlir-disable-threading",
            "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5", str(directory / "logical.out")],
            directory, "physical", timeout)
        if result["allocation"]["status"] == "success":
            result["emission"] = invoke([str(emitter), "--pto-level=level3",
                "--enable-insert-sync=false",
                "--enable-plan-memory=false", str(directory / "physical.out"),
                "-o", str(directory / "kernel.cpp")], directory, "emission", timeout)
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    """Write an incremental, reproducible campaign; never launch parallel children."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("compiler", type=Path)
    parser.add_argument("emitter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="External campaign watchdog in seconds, not an analysis budget")
    parser.add_argument("--cpu", type=int, help="One allowed CPU; LLVM derives one worker from this affinity")
    args = parser.parse_args()
    # The standalone emitter does not register MLIR's threading CLI option.
    # LLVM's Linux thread-count query honors this inherited affinity mask.
    allowed = os.sched_getaffinity(0)
    cpu = min(allowed) if args.cpu is None else args.cpu
    if cpu not in allowed:
        parser.error("Requested CPU is outside this process's allowed affinity")
    os.sched_setaffinity(0, {cpu})
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Timeout must be positive")
    compiler = args.compiler.resolve(strict=True)
    emitter = args.emitter.resolve(strict=True)
    manifest = args.manifest.resolve(strict=True)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    indices = [int(row["index"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate case index in manifest")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pins = {"manifest": sha256(manifest), "compiler": sha256(compiler), "emitter": sha256(emitter),
            "workers": 1, "cpu_affinity": [cpu], "timeout_seconds": args.timeout,
            "address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
            "analysis_options": ["--mlir-disable-threading", "--pto-frontier-analysis=gm-alias=may-not-alias"],
            "allocation_options": ["--mlir-disable-threading", "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"],
            "emission_options": ["--pto-level=level3", "--enable-insert-sync=false",
                                 "--enable-plan-memory=false"]}
    (output / "pins.json").write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    results = []
    for row in rows:
        result = run_case(row, compiler, emitter, output, args.timeout)
        results.append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(result["index"], *(result[step]["status"] for step in ("logical", "allocation", "emission")),
              flush=True)

    if sha256(compiler) != pins["compiler"] or sha256(emitter) != pins["emitter"]:
        raise RuntimeError("A compiler executable changed during the campaign")


if __name__ == "__main__":
    main()
