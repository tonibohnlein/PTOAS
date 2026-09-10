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


GUARD_TRACE = re.compile(
    r"logical guard domains \d+/\d+ ranges \d+/\d+ "
    r"direct (?P<direct_accepted>\d+)/(?P<direct_attempts>\d+) "
    r"forms \d+ rows \d+ matched \d+ grids (?P<legacy_grids>\d+) "
    r"work (?P<direct_work>\d+) remaining \d+ self_test (?P<self_test>[01])\b")


def check_fallback_trace(stderr):
    matches = list(GUARD_TRACE.finditer(stderr))
    assert len(matches) == 1, stderr[-8000:]
    fields = {key: int(value) for key, value in matches[0].groupdict().items()}
    # With a one-unit attempt allowance, every nontrivial direct proposal
    # must decline. Successful strict compilation plus real grid visits then
    # establishes fallback; positive work alone establishes neither.
    assert fields["direct_attempts"] > 0 and fields["legacy_grids"] > 0, fields
    assert fields["direct_accepted"] == 0, fields
    assert 0 < fields["direct_work"] <= fields["direct_attempts"], fields
    assert fields["self_test"] == 1, fields
    return fields


def check_trace_parser():
    good = ("logical guard domains 0/0 ranges 0/0 direct 0/2 forms 1 rows 0 "
            "matched 0 grids 2 work 2 remaining 10 self_test 1\n")
    fields = check_fallback_trace(good)
    assert fields["legacy_grids"] == 2 and fields["direct_work"] == 2
    for bad in (good.replace("grids 2", "grids 0"),
                good.replace("direct 0/2", "direct 1/2"),
                good.replace("direct 0/2", "direct 0/0"),
                good.replace("work 2", "work 3"),
                good.replace("self_test 1", "self_test 0"), "", good + good):
        try:
            check_fallback_trace(bad)
        except AssertionError:
            continue
        raise AssertionError("Fallback checker accepted a negative trace: " + bad)


def main():
    if not __debug__:
        raise RuntimeError("Direct guard checks require assertions enabled")
    check_trace_parser()
    from observations import SERIAL_DRIVER, population
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
    fields = check_fallback_trace(result.stderr)
    summary = {"status": "passed", "direct_accepted": fields["direct_accepted"],
               "direct_attempts": fields["direct_attempts"], "legacy_grids": fields["legacy_grids"],
               "direct_work": fields["direct_work"],
               "direct_self_test": bool(fields["self_test"]),
               "claim": "live directRow rejects INT64_MIN sign normalization and optional exhaustion falls back"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
