# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Exercise native fitting and scarce directed-domain event-key trial policies."""
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
from check_scalability import source_for


def alternating_streams(count):
    # N V readers of one load need only ONE incoming MTE2 prefix. Overwrite the
    # shared input between V consumers to create distinct ready/release streams.
    lines, load, consumers = [], None, 0
    for line in source_for(count, False).splitlines():
        if line.strip().startswith('pto.tload '):
            load = line
        if line.strip().startswith('pto.tabs '):
            if consumers:
                assert load is not None
                lines.append(load)
            consumers += 1
        lines.append(line)
    assert consumers == count
    return '\n'.join(lines) + '\n'


def compile_case(case, source, output, python_root, env):
    command = [sys.executable, "-c", SERIAL_DRIVER, str(python_root.resolve()),
               "--pto-arch=a3", "--pto-level=level3", "--enable-insert-sync",
               "--insert-sync-planner=logical", "--insert-sync-gm-alias=assume-disjoint-arguments",
               "--emit-pto-ir", str(source.resolve()), "-o", str(output.resolve())]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
    (output.parent / (case + ".stdout")).write_text(result.stdout)
    (output.parent / (case + ".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stderr[-4000:]
    matches = re.findall(r"logical allocation dedicated (\d+) sharing_trials (\d+) streams (\d+)", result.stderr)
    assert len(matches) == 1, result.stderr[-8000:]
    return command, tuple(map(int, matches[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    case = next(item for item in population() if item["case_id"] == "two_buffer")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PTOAS_LOGICAL_TRACE="1")
    fit_command, fit_values = compile_case("two_buffer", case["source"],
                                            args.output / "two_buffer.logical.pto",
                                            args.python_root, env)
    dedicated, sharing_trials, streams = fit_values
    assert 0 < streams <= 8, fit_values
    assert dedicated == streams and sharing_trials == 0, fit_values
    over_source = args.output / "over_capacity.pto"
    over_source.write_text(alternating_streams(9))
    over_command, over_values = compile_case("over_capacity", over_source,
                                               args.output / "over_capacity.logical.pto",
                                               args.python_root, env)
    over_dedicated, over_sharing, over_streams = over_values
    # Nine MTE2->V readiness streams are scarce; eight V->MTE2 release
    # streams fit their SEPARATE directed pool. Sharing-first dedicates only
    # the first ready stream; the fitting reverse domain dedicates all eight.
    # A global <=8 dedicated bound would incorrectly combine those pools.
    assert over_streams == 17, over_values
    assert over_dedicated == 9 and over_sharing >= 8, over_values
    (args.output / "summary.json").write_text(json.dumps({
        "fitting_pool": {"case": case["case_id"], "dedicated": dedicated,
                          "sharing_trials": sharing_trials, "streams": streams,
                          "command": fit_command},
        "over_capacity": {"dedicated": over_dedicated, "sharing_trials": over_sharing,
                           "streams": over_streams, "command": over_command},
    }, indent=2) + "\n")
    print(f"native fitting-pool allocation passed: {dedicated} dedicated assignments, "
          f"{sharing_trials} occupied-key trials, {streams} streams; "
          f"over-capacity case: {over_dedicated} dedicated, {over_sharing} occupied-key trials, "
          f"{over_streams} streams")


if __name__ == "__main__":
    main()
