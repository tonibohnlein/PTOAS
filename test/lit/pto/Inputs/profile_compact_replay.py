# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Production-only retained-size sweep for the admitted counter-reset pipeline."""

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import time
from pathlib import Path

from check_compact_persistent_contract import variant


def cases(source):
    for visits in (0, 1, 2, 3, 8):
        text = variant(source, 2, 2, 2, 0, 2).replace(
            "%epoch_limit = arith.constant 2", f"%epoch_limit = arith.constant {visits}"
        )
        yield f"replays-{visits}", text
    for trips in (4, 8, 16):
        yield f"inner-K-{trips}", variant(source, 2, trips, 4, 0, 3)
    for inner in (1, 2, 4):
        yield f"inner-kk-{inner}", variant(source, 2, 4, inner, 0, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("optimizer", "input", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    args.optimizer = Path(shutil.which(str(args.optimizer)) or args.optimizer).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    receipt = {
        "optimizer_sha256": hashlib.sha256(args.optimizer.read_bytes()).hexdigest(),
        "allocation": "not requested",
        "scope": "single production profiles; no speedup claim",
        "cases": [],
    }
    for name, text in cases(args.input.read_text()):
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        source, target = directory / "input.pto", directory / "logical.pto"
        source.write_text(text)
        command = [
            str(args.optimizer),
            "--mlir-disable-threading",
            "--pto-frontier-analysis=gm-alias=may-not-alias prepare-allocation=false profile-stages=true",
            str(source),
            "-o",
            str(target),
        ]
        start = time.perf_counter()
        run = subprocess.run(
            command, capture_output=True, text=True, timeout=35, check=False
        )
        directory.joinpath("stderr").write_text(run.stderr)
        if run.returncode:
            raise RuntimeError(run.stderr[-2000:])
        inventory = next(
            json.loads(line.split(" ", 1)[1])
            for line in run.stderr.splitlines()
            if line.startswith("frontier-retained-inventory ")
        )
        rows = inventory["rows"]
        repeat = [
            r.get("query_types", 0)
            for r in rows
            if r.get("representation") == "regional-repetition"
        ]
        finite = [
            r.get("input_sites", 0)
            for r in rows
            if r.get("representation") == "finite-numerical-body"
        ]
        backend = re.search(
            r'selected_logical_backend = "([^"\n]+)"', target.read_text()
        )
        if not backend or backend[1] == "completion-transfer":
            raise AssertionError(
                "Conservative completion is not an exact scaling success"
            )
        row = {
            "name": name,
            "command": command,
            "seconds": time.perf_counter() - start,
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "backend": backend[1],
            "max_repeated_types": max(repeat, default=0),
            "largest_numerical_body": max(finite, default=0),
            "arena_nodes": max((r.get("arena_nodes", 0) for r in rows), default=0),
            "inventory": inventory,
        }
        receipt["cases"].append(row)
        (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            name,
            row["max_repeated_types"],
            row["largest_numerical_body"],
            row["arena_nodes"],
            flush=True,
        )
    for prefix, minimum in [("replays-", 2), ("inner-K-", 4)]:
        rows = [
            r
            for r in receipt["cases"]
            if r["name"].startswith(prefix)
            and int(r["name"].removeprefix(prefix)) >= minimum
        ]
        signatures = {
            (r["max_repeated_types"], r["largest_numerical_body"]) for r in rows
        }
        if len(signatures) != 1 or not next(iter(signatures))[0]:
            raise AssertionError("Compact representation was lost: " + str(signatures))
    receipt["structural_gate"] = "passed"
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
