# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""Exercise the native fitting-pool event-key trial policy."""
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
    over_source.write_text(source_for(9, False))
    over_command, over_values = compile_case("over_capacity", over_source,
                                               args.output / "over_capacity.logical.pto",
                                               args.python_root, env)
    over_dedicated, over_sharing, over_streams = over_values
    assert over_streams > 8, over_values
    assert over_dedicated <= 8 and over_sharing > 0, over_values
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
