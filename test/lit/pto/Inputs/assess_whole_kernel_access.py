# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bounded serial whole-kernel assessment; no recognition or allocation changes."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time


PREPARED = ("persistent_gemm", "elementwise_pipeline", "group_norm", "gdn_chunk_cumsum")
SURVEY = ("qwen-down", "qwen-finalize", "deepseek-rms")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command, path, timeout):
    start = time.perf_counter()
    with path.open("w") as out, path.with_suffix(".stderr").open("w") as err:
        try:
            result = subprocess.run(
                list(map(str, command)),
                stdout=out,
                stderr=err,
                timeout=timeout,
                check=False,
            )
            status = "success" if result.returncode == 0 else "failure"
            code = result.returncode
        except subprocess.TimeoutExpired:
            status, code = "timeout", None
    return {
        "command": list(map(str, command)),
        "status": status,
        "returncode": code,
        "seconds": time.perf_counter() - start,
        "stdout": str(path),
        "stderr": str(path.with_suffix(".stderr")),
    }


def annotate(text, attrs):
    text, count = re.subn(
        r"(func\.func\s+@\w+\([^\n]*?\)\s*)attributes\s*\{",
        lambda m: m[1] + "attributes {" + attrs + ", ",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Expected one attributed function")
    return text


def profile(path):
    result = []
    for line in path.read_text().splitlines():
        for prefix in (
            "frontier-profile ",
            "frontier-retained-inventory ",
            "frontier-control-profile ",
        ):
            if line.startswith(prefix):
                result.append(
                    {"kind": prefix.strip(), "value": json.loads(line[len(prefix) :])}
                )
    return result


def inventory(path):
    text = path.read_text()
    counts = {
        name: len(re.findall(pattern, text))
        for name, pattern in {
            "sets": r"\bpto\.(?:logical_set|set_flag(?:_dyn)?)\b",
            "waits": r"\bpto\.(?:logical_wait|wait_flag(?:_dyn)?)\b",
            "barriers": r"\bpto\.barrier\b",
            "arithmetic": r"\barith\.\w+",
        }.items()
    }
    attrs = {}
    for key in (
        "backend",
        "effect_order",
        "reduction",
        "realization_order",
        "access_model",
    ):
        values = re.findall(r"\b" + key + r'\s*=\s*"([^"]*)"', text)
        if values:
            attrs[key] = sorted(set(values))
    return {
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "static_ir": counts,
        "guarantees": attrs,
    }


def compile_stage(args, folder, name, source, flag):
    target = folder / (name + ".pto")
    row = invoke(
        [args.optimizer, "--mlir-disable-threading", flag, source, "-o", target],
        folder / (name + ".stdout"),
        args.timeout,
    )
    row["profiles"] = profile(Path(row["stderr"]))
    if row["status"] == "success":
        row.update(inventory(target))
        row["output"] = str(target)
    else:
        row["diagnostic"] = Path(row["stderr"]).read_text()[-12000:]
    return row


def access_stage(args, folder, source, alias):
    row = invoke(
        [args.trace_tool, "--gm-alias=" + alias, "--storage-effects", source],
        folder / "access.txt",
        args.timeout,
    )
    counts, fallbacks = Counter(), Counter()
    for line in Path(row["stdout"]).read_text().splitlines():
        m = re.search(
            r" effect \d+ .* (read|write) (intervals|symbolic|unresolved).*"
            r"description=([^ ]+) fallback=(\d+) space=(\d+)",
            line,
        )
        if m:
            mode, form, description, fallback, space = m.groups()
            counts[f"{space}/{mode}/{description}/{form}"] += 1
            fallbacks[fallback] += 1
    row.update(
        effect_counts=dict(counts),
        fallback_counts=dict(fallbacks),
        precision_scope="Supplied model provenance, not native footprint qualification",
    )
    return row


def diagnostic(args, folder, source, alias, name, exports=None):
    attrs = f'test.contract_gm_alias = "{alias}", test.contract_math_only'
    if exports is not None:
        attrs += f", test.contract_exports = {exports} : i64"
    target = folder / (name + "-input.pto")
    target.write_text(annotate(source.read_text(), attrs))
    row = invoke(
        [args.trace_tool, "--compact-region-contract", target],
        folder / (name + ".json"),
        args.timeout,
    )
    if row["status"] == "success":
        row["result"] = json.loads(Path(row["stdout"]).read_text())
    return row


def run_case(args, source, alias):
    folder = args.output / source.parent.name / alias
    folder.mkdir(parents=True, exist_ok=True)
    # Survey copies all have input.pto under distinct kernel directories.
    row = {
        "kernel": source.parent.name,
        "alias": alias,
        "input": str(source),
        "input_sha256": digest(source),
        "stages": {},
    }
    stages = row["stages"]
    stages["access"] = access_stage(args, folder, source, alias)
    stages["logical"] = compile_stage(
        args,
        folder,
        "logical",
        source,
        f"--pto-frontier-analysis=gm-alias={alias} prepare-allocation=false profile-stages=true",
    )
    print(row["kernel"], alias, "logical", stages["logical"]["status"], flush=True)
    if stages["logical"]["status"] == "success":
        stages["allocation-support"] = compile_stage(
            args,
            folder,
            "allocation-support",
            source,
            f"--pto-frontier-analysis=gm-alias={alias} allocation-ids=0,1,2,3,4,5 profile-stages=true",
        )
        support = stages["allocation-support"]
        if support["status"] == "success":
            stages["physical"] = compile_stage(
                args,
                folder,
                "physical",
                Path(support["output"]),
                "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5",
            )
            if stages["physical"]["status"] == "success":
                output = folder / "kernel.cpp"
                stages["cpp"] = invoke(
                    [
                        args.python,
                        args.cli,
                        "--pto-arch=a3",
                        "--pto-level=level3",
                        "--enable-insert-sync=false",
                        "--enable-plan-memory=false",
                        stages["physical"]["output"],
                        "-o",
                        output,
                    ],
                    folder / "cpp.stdout",
                    args.timeout,
                )
                if stages["cpp"]["status"] == "success":
                    stages["cpp"].update(
                        bytes=output.stat().st_size, sha256=digest(output)
                    )
    for name, mask in (("mathematics", None), ("queries", 1), ("selectors", 2)):
        stages[name] = diagnostic(args, folder, source, alias, name, mask)
        print(row["kernel"], alias, name, stages[name]["status"], flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("optimizer", "trace-tool", "cli", "python", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=35)
    parser.add_argument("--cases", nargs="+", default=PREPARED + SURVEY)
    args = parser.parse_args()
    if args.timeout < 1 or any(k not in PREPARED + SURVEY for k in args.cases):
        parser.error("Positive watchdog and known cases required")
    args.output = args.output.resolve()
    for key in ("optimizer", "trace_tool", "cli", "python"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    report = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_source": True,
        "watchdog_seconds": args.timeout,
        "workers": 1,
        "tools": {
            str(getattr(args, key)): digest(getattr(args, key))
            for key in ("optimizer", "trace_tool", "cli")
        },
        "cases": [],
        "timing_scope": "Separate process wall times; inclusive profile columns are not additive",
        "device_compile": False,
        "device_execution": False,
    }
    for name in args.cases:
        source = (
            Path("test/npu_validation/benchmarks/auto_sync/prepared")
            / name
            / "input.pto"
        )
        if name in SURVEY:
            source = (
                Path("test/lit/pto/Inputs/pypto_region_survey") / name / "input.pto"
            )
        for alias in ("may-not-alias", "may-alias"):
            report["cases"].append(run_case(args, source.resolve(), alias))
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
