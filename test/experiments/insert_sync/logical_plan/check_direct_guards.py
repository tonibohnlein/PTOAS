#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0.
"""Exercise the live direct-row edge check and optional-guard fallback."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
from observations import SERIAL_DRIVER, population


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case = next(item for item in population() if item["case_id"] == "two_buffer")
    output = args.output / "two_buffer.direct-budget.pto"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PTOAS_LOGICAL_TRACE="1", PTOAS_LOGICAL_DIRECT_ATTEMPT_ALLOWANCE="1",
               PTOAS_LOGICAL_DIRECT_SELF_TEST="1")
    command = [sys.executable, "-c", SERIAL_DRIVER, str(args.python_root.resolve()),
               "--pto-arch=a3", "--pto-level=level3", "--enable-insert-sync",
               "--insert-sync-planner=logical", "--insert-sync-gm-alias=assume-disjoint-arguments",
               "--emit-pto-ir", str(case["source"].resolve()), "-o", str(output.resolve())]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
    (args.output / "stdout").write_text(result.stdout)
    (args.output / "stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stderr[-4000:]
    matches = re.findall(
        r"logical guard domains (\d+)/(\d+) ranges (\d+)/(\d+) direct (\d+)/(\d+) "
        r"forms (\d+) rows (\d+) matched (\d+) grids (\d+) work (\d+) remaining (\d+) self_test ([01])",
        result.stderr)
    assert len(matches) == 1, result.stderr[-8000:]
    fields = list(map(int, matches[0]))
    direct_accepted, direct_attempts, legacy_grids, self_test = fields[4], fields[5], fields[10], fields[12]
    assert direct_attempts > 0 and legacy_grids > 0, matches
    assert self_test == 1, matches
    summary = {"status": "passed", "direct_accepted": direct_accepted,
               "direct_attempts": direct_attempts, "legacy_grids": legacy_grids,
               "direct_self_test": bool(self_test),
               "claim": "live directRow rejects INT64_MIN sign normalization and optional exhaustion falls back"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
