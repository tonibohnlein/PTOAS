# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Production scaling sweeps; serial requests, no audit-mode timing."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from check_compact_persistent_contract import variant


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command, stderr_path, timeout):
    start = time.perf_counter()
    try:
        with stderr_path.open("w") as stderr:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        status = "success" if result.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        status = "watchdog"
    return status, time.perf_counter() - start


def profile(command, directory, mode, timeout):
    command = list(command)
    command[2] += " profile-stages=true"
    path = directory / (mode + "-profile.stderr")
    status, seconds = invoke(command, path, timeout)
    selected = []
    for line in path.read_text().splitlines():
        for prefix in ("frontier-profile ", "frontier-retained-inventory "):
            if line.startswith(prefix):
                selected.append(
                    {"kind": prefix.strip(), "value": json.loads(line[len(prefix) :])}
                )
    (directory / (mode + "-profile.json")).write_text(
        json.dumps(selected, indent=2) + "\n"
    )
    return {"status": status, "seconds": seconds}


def check_case(args, name, text):
    directory = args.output / name
    directory.mkdir(exist_ok=True)
    source = directory / "input.pto"
    source.write_text(text)
    result = {"name": name, "input_sha256": digest(source), "runs": {}}
    for mode in ("logical-only", "allocation-support"):
        options = "gm-alias=may-not-alias " + (
            "prepare-allocation=false"
            if mode == "logical-only"
            else "allocation-ids=0,1,2,3,4,5"
        )
        command = [
            str(args.optimizer),
            "--mlir-disable-threading",
            "--pto-frontier-analysis=" + options,
            str(source),
            "-o",
            str(directory / (mode + ".pto")),
        ]
        times = []
        for repetition in range(args.repetitions):
            status, seconds = invoke(
                command, directory / f"{mode}-{repetition}.stderr", args.timeout
            )
            times.append(seconds)
            if status != "success":
                break
        run = {"status": status, "launcher_seconds": times, "command": command}
        result["runs"][mode] = run
        if status != "success":
            break  # Experiment policy: do not repeat an unresolved earlier stage for allocation.
        run["profile"] = profile(command, directory, mode, args.timeout)
    print(
        name,
        {mode: value["status"] for mode, value in result["runs"].items()},
        flush=True,
    )
    return result


def static_stages(args, source, stages):
    text = variant(source, 2, 4, stages, 0)
    opening = text.index("{", text.index("scf.for %kk"))
    depth, end = 1, opening + 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise RuntimeError("Unbalanced fixture loop")
    text = text[:end] + ' {pto.unroll = "full"}' + text[end:]
    path = args.output / f"static-{stages}-before.pto"
    path.write_text(text)
    target = args.output / f"static-{stages}.pto"
    command = [
        str(args.optimizer),
        "--mlir-disable-threading",
        "--pto-unroll-loops",
        str(path),
        "-o",
        str(target),
    ]
    status, _ = invoke(command, target.with_suffix(".stderr"), args.timeout)
    if status != "success":
        raise RuntimeError(f"Test fixture unrolling failed: {status}")
    return target.read_text()


def cases(args, source):
    if args.sweep in ("all", "trips"):
        for trips in (2, 4, 8, 16, 32):
            yield f"trips-{trips}", variant(source, 2, trips, 4, 0)
    if args.sweep in ("all", "stages"):
        for stages in (1, 2, 4):
            yield f"static-stages-{stages}", static_stages(args, source, stages)
    if args.sweep in ("all", "nesting"):
        for levels in (0, 1, 2):
            text = variant(source, 2, 2, 2, 0)
            for level in range(levels):
                text = text.replace(
                    "    %c0 =",
                    f"    %nest_limit_{level} = arith.constant 2 : index\n    %c0 =",
                    1,
                )
                text = text.replace(
                    "    scf.for %wave",
                    f"    scf.for %nest_{level} = %c0 to %nest_limit_{level} step %c1 {{\n    scf.for %wave",
                    1,
                )
                text = text.replace(
                    "    // Uniform invocation", "    }\n    // Uniform invocation", 1
                )
            yield f"nesting-{levels}", text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("optimizer", "input", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--require-compact",
        action="store_true",
        help="Require K>=4 derivatives to retain the same nonempty phase/type inventory",
    )
    parser.add_argument(
        "--sweep", choices=("all", "trips", "stages", "nesting"), default="all"
    )
    args = parser.parse_args()
    if args.timeout <= 0 or args.repetitions < 1:
        parser.error("Positive timeout and repetition count required")
    resolved = (
        str(args.optimizer.resolve())
        if args.optimizer.is_file()
        else shutil.which(str(args.optimizer))
    )
    if not resolved:
        parser.error("Optimizer unavailable")
    args.optimizer = Path(resolved)
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    receipt = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "optimizer_sha256": digest(args.optimizer),
        "cases": [],
        "timing_scope": "Launcher wall time including timeout polling; profiled runs separate; no speedup claim",
    }
    for name, text in cases(args, args.input.read_text()):
        receipt["cases"].append(check_case(args, name, text))
        (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if args.require_compact:
        signatures = []
        for case in receipt["cases"]:
            if (
                not case["name"].startswith("trips-")
                or int(case["name"].split("-")[-1]) < 4
            ):
                continue
            if case["runs"]["logical-only"]["status"] != "success":
                raise RuntimeError("Compact-trip production request failed")
            document = json.loads(
                (args.output / case["name"] / "logical-only-profile.json").read_text()
            )
            rows = next(
                item["value"]["rows"]
                for item in document
                if item["kind"] == "frontier-retained-inventory"
            )
            repeated = next(
                (
                    row
                    for row in rows
                    if row.get("representation") == "regional-repetition"
                ),
                None,
            )
            numerical = [
                row["input_sites"]
                for row in rows
                if row.get("representation") == "finite-numerical-body"
            ]
            if not repeated or not numerical:
                raise RuntimeError("Compact repeated/numerical representation was lost")
            signatures.append((repeated["query_types"], max(numerical)))
        if len(signatures) < 2 or len(set(signatures)) != 1:
            raise RuntimeError(
                f"Retained size grows with K or comparison is missing: {signatures}"
            )
        receipt["compact_trip_signature"] = signatures[0]
        (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
