#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Verify benchmark pins; optionally regenerate expert/frontier/existing C++.

Source-only candidates remain explicitly pending. Snapshots are never imported.
Generation is serial and records one-shot wall times, not device measurements.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parent


def pinned_file(record):
    path = (ROOT / record.get("snapshot", record.get("path", ""))).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f"Invalid pinned file: {path}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Hash mismatch: {path}")
    return path


def verify(manifest):
    if manifest["variants"] != ["expert", "frontier", "existing"]:
        raise ValueError("Expected expert, frontier and existing variants")
    pinned_file(manifest["sources_license"])
    for record in manifest.get("lowering_sources", []):
        pinned_file(record)
    seen = set()
    for case in manifest["benchmarks"]:
        name = case["id"]
        if name in seen or not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid or duplicate benchmark ID: {name}")
        seen.add(name)
        for source in case["sources"]:
            if "snapshot" in source:
                path = pinned_file(source)
                if source["path"].endswith(".py"):
                    ast.parse(path.read_text(encoding="utf-8"), filename=source["path"])
        if "port_metadata" in case:
            pinned_file(case["port_metadata"])
        for record in case.get("prepared", {}).values():
            pinned_file(record)
        if case["status"] == "prepared_pto" and set(case.get("prepared", {})) != {"input.pto", "expert.pto"}:
            raise ValueError(f"Missing prepared input pair: {name}")


def executable(value):
    path = shutil.which(value)
    if path is None:
        raise ValueError(f"Executable not found: {value}")
    return str(Path(path).resolve())


def run_command(command, output, timeout):
    start = time.perf_counter()
    with output.open("w", encoding="utf-8") as stdout, output.with_suffix(output.suffix + ".err").open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    return {"command": command, "returncode": code, "wall_seconds": time.perf_counter() - start}


def generate_variant(case, variant, output, opt, compiler, timeout):
    stem = output / variant
    steps = []
    if variant == "expert":
        shutil.copyfile(pinned_file(case["prepared"]["expert.pto"]), stem.with_suffix(".pto"))
    else:
        source = pinned_file(case["prepared"]["input.pto"])
        passes = ["--pto-insert-sync=gm-alias=may-not-alias"]
        if variant == "frontier":
            passes = ["--pto-frontier-analysis=gm-alias=may-not-alias",
                      "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"]
        steps.append(run_command([opt, "--mlir-disable-threading", *passes, str(source)],
                                 stem.with_suffix(".pto"), timeout))
    if not steps or steps[-1]["returncode"] == 0:
        command = [compiler, "--pto-level=level3", "--enable-insert-sync=false",
                   "--enable-plan-memory=false", str(stem.with_suffix(".pto")),
                   "-o", str(stem.with_suffix(".cpp"))]
        steps.append(run_command(command, stem.with_suffix(".codegen.log"), timeout))
    return {"benchmark": case["id"], "variant": variant, "steps": steps,
            "success": all(step["returncode"] == 0 for step in steps)}


def generate(manifest, args):
    opt, compiler = executable(args.pto_test_opt), executable(args.ptoas)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(ROOT / "benchmarks.json", output / "benchmarks.json")
    results = []
    for case in manifest["benchmarks"]:
        if case["status"] != "prepared_pto":
            continue
        directory = output / case["id"]
        directory.mkdir()
        if "port_metadata" in case:
            shutil.copyfile(pinned_file(case["port_metadata"]), directory / "port.json")
        shutil.copyfile(pinned_file(case["prepared"]["input.pto"]), directory / "input.pto")
        for variant in manifest["variants"]:
            results.append(generate_variant(case, variant, directory, opt, compiler, args.timeout))
    report = {"tools": {tool: hashlib.sha256(Path(tool).read_bytes()).hexdigest()
                        for tool in (opt, compiler)},
              "manifest_sha256": hashlib.sha256((ROOT / "benchmarks.json").read_bytes()).hexdigest(),
              "pending": [c["id"] for c in manifest["benchmarks"] if c["status"] != "prepared_pto"],
              "artifacts": {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(output.rglob("*")) if path.is_file()},
              "results": results}
    (output / "generation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {sum(r['success'] for r in results)}/{len(results)} variants; report: {output}")
    return 0 if all(result["success"] for result in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory; omission only verifies and lists")
    parser.add_argument("--pto-test-opt", default="pto-test-opt")
    parser.add_argument("--ptoas", default="ptoas")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        manifest = json.loads((ROOT / "benchmarks.json").read_text(encoding="utf-8"))
        verify(manifest)
        for case in manifest["benchmarks"]:
            print(f"{case['id']:28} {case['group']:10} {case['status']}")
        return generate(manifest, args) if args.output is not None else 0
    except (OSError, ValueError, KeyError, SyntaxError) as error:
        parser.exit(1, f"Benchmark preparation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
