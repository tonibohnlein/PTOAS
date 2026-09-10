# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""Serial ordinary two-/three-buffer compile comparison; device-free."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
from measure import measure
from observations import SERIAL_DRIVER, analyze, population


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_root.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cases = {item["case_id"]: item for item in population()}
    summary = []
    for case_id in ("two_buffer", "three_buffer"):
        case = cases[case_id]
        row = {"case": case_id, "arms": {}}
        for arm, flags in (("existing", []), ("logical", ["--insert-sync-planner=logical"])):
            output = args.output / f"{case_id}.{arm}.pto"
            command = [sys.executable, "-c", SERIAL_DRIVER, str(args.python_root.resolve()),
                       "--pto-arch=a3", "--pto-level=level3", "--enable-insert-sync",
                       "--insert-sync-gm-alias=assume-disjoint-arguments", *flags,
                       "--emit-pto-ir", str(case["source"].resolve()), "-o", str(output.resolve())]
            start = time.monotonic()
            result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
            seconds = time.monotonic() - start
            (args.output / f"{case_id}.{arm}.stdout").write_text(result.stdout)
            (args.output / f"{case_id}.{arm}.stderr").write_text(result.stderr)
            (args.output / f"{case_id}.{arm}.command.json").write_text(
                json.dumps({"command": command, "exit": result.returncode, "seconds": seconds}, indent=2) + "\n")
            assert result.returncode == 0, (case_id, arm, result.stderr[-4000:])
            projection = analyze(output)
            row["arms"][arm] = {"seconds": seconds, "projection": projection}
        cpp = args.output / f"{case_id}.logical.cpp"
        command = [sys.executable, "-c", SERIAL_DRIVER, str(args.python_root.resolve()),
                   "--pto-arch=a3", "--pto-level=level3",
                   str(args.output / f"{case_id}.logical.pto"), "-o", str(cpp.resolve())]
        start = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=180)
        seconds = time.monotonic() - start
        (args.output / f"{case_id}.cpp.stdout").write_text(result.stdout)
        (args.output / f"{case_id}.cpp.stderr").write_text(result.stderr)
        (args.output / f"{case_id}.cpp.command.json").write_text(
            json.dumps({"command": command, "exit": result.returncode, "seconds": seconds}, indent=2) + "\n")
        assert result.returncode == 0, (case_id, "cpp", result.stderr[-4000:])
        row["logical_cpp_seconds"] = seconds
        row["logical_metrics"] = measure(args.output / f"{case_id}.logical.pto", case.get("scenarios", []))
        for key in ("payload", "allocations", "views", "abi"):
            assert row["arms"]["existing"]["projection"][key] == row["arms"]["logical"]["projection"][key], (case_id, key)
        summary.append(row)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in summary:
        existing = row["arms"]["existing"]["seconds"]
        logical = row["arms"]["logical"]["seconds"]
        print(f"{row['case']}: existing={existing:.3f}s logical={logical:.3f}s "
              f"ratio={logical / existing:.2f}x cpp={row['logical_cpp_seconds']:.3f}s")


if __name__ == "__main__":
    main()
