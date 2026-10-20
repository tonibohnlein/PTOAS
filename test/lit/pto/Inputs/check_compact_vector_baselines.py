# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bounded prepared-port baselines; unsupported stages are recorded, not repaired.

Coverage starts from an independent supplied-layout interpreter. This does not
qualify native instruction footprints. Scalar-result readiness is a shared
native prerequisite contract, separate from memory hazards and emitted commands.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time
from compact_vector_oracle import model, required, command_order


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(command, output, timeout):
    start = time.perf_counter()
    code = None
    with output.open("w") as stdout, output.with_suffix(output.suffix + ".stderr").open(
        "w"
    ) as stderr:
        try:
            result = subprocess.run(
                list(map(str, command)),
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
            code = result.returncode
            status = (
                "success"
                if code == 0
                else "process_signal"
                if code < 0
                else "compiler_failure"
            )
        except subprocess.TimeoutExpired:
            status = "watchdog"
    return {
        "status": status,
        "returncode": code,
        "signal": -code if code is not None and code < 0 else None,
        "seconds": time.perf_counter() - start,
        "command": list(map(str, command)),
    }


def compile_stage(args, directory, stage, source, flag):
    target = directory / (stage + ".pto")
    if not source.is_file() or not source.stat().st_size:
        return {"status": "not_run_missing_input"}
    command = [args.optimizer, "--mlir-disable-threading", flag, source, "-o", target]
    result = invoke(command, directory / (stage + ".stdout"), args.timeout)
    result["output"] = str(target)
    if result["status"] == "success":
        text = target.read_text()
        result["static_ir"] = {
            "sets": len(
                re.findall(r"\bpto\.(?:logical_set|set_flag(?:_dyn)?)\b", text)
            ),
            "waits": len(
                re.findall(r"\bpto\.(?:logical_wait|wait_flag(?:_dyn)?)\b", text)
            ),
            "barriers": len(re.findall(r"\bpto\.barrier\b", text)),
            "arithmetic": len(re.findall(r"\barith\.\w+", text)),
        }
    result["profiles"] = []
    log = (directory / (stage + ".stdout.stderr")).read_text()
    for line in log.splitlines():
        for prefix in (
            "frontier-profile ",
            "frontier-retained-inventory ",
            "frontier-control-profile ",
        ):
            if line.startswith(prefix):
                result["profiles"].append(
                    {"kind": prefix.strip(), "value": json.loads(line[len(prefix) :])}
                )
    return result


def semantic(args, directory, source, expert, logical, physical, name):
    text = source.read_text()
    if name == "gdn_chunk_cumsum":

        def small(value):
            return value.replace(
                "%c10 = arith.constant 8", "%c10 = arith.constant 2"
            ).replace("%c11 = arith.constant 128", "%c11 = arith.constant 3")

        text = small(text)
        source = directory / "small.pto"
        source.write_text(text)
        expert_small = directory / "small-expert.pto"
        expert_small.write_text(small(expert.read_text()))
        expert = expert_small
        compiled = compile_stage(
            args,
            directory,
            "small-logical",
            source,
            "--pto-frontier-analysis=gm-alias=may-not-alias allocation-ids=0,1,2,3,4,5",
        )
        logical = Path(compiled["output"]) if compiled["status"] == "success" else None
        physical = None
        if logical:
            allocated = compile_stage(
                args,
                directory,
                "small-physical",
                logical,
                "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5",
            )
            if allocated["status"] == "success":
                physical = Path(allocated["output"])
    rows = []
    for block, subblock in ((0, 0), (1, 0), (0, 1)):
        row = {"block": block, "subblock": subblock, "variants": {}}
        expected = model(text, block, subblock)
        wanted, covers, scalar = required(expected)
        row.update(
            payloads=len(expected),
            nonnative_covers=len(covers),
            scalar_edges=len(scalar),
        )
        for variant, path in (
            ("input", source),
            ("expert", expert),
            ("generated", logical),
            ("physical", physical),
        ):
            if not path or not path.is_file() or not path.stat().st_size:
                continue
            current = (
                path.read_text()
                .replace("pto.get_block_idx", f"arith.constant {block} : i64")
                .replace("pto.get_subblock_idx", f"arith.constant {subblock} : i64")
            )
            current, count = re.subn(
                r"(func\.func\s+@\w+\([^\n]*?\)\s*)attributes\s*\{",
                r"\1attributes {test.trace_existing, ",
                current,
                count=1,
            )
            if count != 1:
                raise RuntimeError("Cannot annotate function")
            trace_source = directory / f"{variant}-{block}-{subblock}.pto"
            trace_source.write_text(current)
            output = trace_source.with_suffix(".json")
            result = invoke(
                [args.trace_tool, "--structured-trace", trace_source],
                output,
                args.timeout,
            )
            if result["status"] == "success":
                try:
                    document = json.loads(output.read_text())
                    if not document.get("accepted") or document["trace"]["error"]:
                        raise ValueError(str(document)[:500])
                    trace = document["trace"]
                    payloads = [e for e in trace["events"] if e["kind"] == "payload"]
                    if [(e["type"], e["coordinates"], e["pipe"]) for e in payloads] != [
                        (e["type"], e["coordinates"], e["pipe"]) for e in expected
                    ]:
                        raise ValueError("Original payload mapping mismatch")
                    observed, reuse = command_order(trace, scalar)
                    missing, excess = [], []
                    for a in range(len(wanted)):
                        for b in range(len(wanted)):
                            if a == b:
                                continue
                            if wanted[a] >> b & 1 and not observed[a] >> b & 1:
                                missing.append((a, b))
                            if observed[a] >> b & 1 and not wanted[a] >> b & 1:
                                excess.append((a, b))
                    result.update(
                        commands=len(trace["events"]) - len(payloads),
                        matching="passed",
                        physical_reuse=reuse,
                        missing_event_pairs=len(missing),
                        extra_event_pairs=len(excess),
                        missing_examples=missing[:5],
                        extra_examples=excess[:5],
                        missing_ci=sum(a % 2 == 1 and b % 2 == 0 for a, b in missing),
                        extra_ci=sum(a % 2 == 1 and b % 2 == 0 for a, b in excess),
                    )
                except (ValueError, KeyError, AssertionError) as error:
                    result.update(status="oracle_unavailable", diagnostic=str(error))
            row["variants"][variant] = result
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("optimizer", "trace-tool", "prepared-root", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("elementwise_pipeline", "group_norm", "gdn_chunk_cumsum"),
        default=("elementwise_pipeline", "group_norm"),
    )
    parser.add_argument("--timeout", type=int, default=25)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("Positive watchdog required")
    args.optimizer = args.optimizer.resolve()
    args.trace_tool = args.trace_tool.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    receipt = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "optimizer_sha256": digest(args.optimizer),
        "trace_sha256": digest(args.trace_tool),
        "target": "A3 vector",
        "alias": "may-not-alias",
        "eligible_ids": list(range(6)),
        "timeout_seconds": args.timeout,
        "cpp": "not_requested",
        "device": "not_requested",
        "cases": [],
    }
    for name in args.cases:
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        source = args.prepared_root / name / "input.pto"
        expert = source.with_name("expert.pto")
        case = {
            "name": name,
            "input_sha256": digest(source),
            "expert_sha256": digest(expert),
            "stages": {},
        }
        for stage, path, flag in (
            (
                "logical-only",
                source,
                "--pto-frontier-analysis=gm-alias=may-not-alias prepare-allocation=false profile-stages=true",
            ),
            (
                "allocation-support",
                source,
                "--pto-frontier-analysis=gm-alias=may-not-alias allocation-ids=0,1,2,3,4,5 profile-stages=true",
            ),
            (
                "physical",
                directory / "allocation-support.pto",
                "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5",
            ),
            ("existing", source, "--pto-insert-sync=gm-alias=may-not-alias"),
        ):
            case["stages"][stage] = compile_stage(args, directory, stage, path, flag)
            print(name, stage, case["stages"][stage]["status"], flush=True)
        case["semantics"] = semantic(
            args,
            directory,
            source,
            expert,
            directory / "logical-only.pto",
            Path(case["stages"]["physical"]["output"])
            if case["stages"]["physical"]["status"] == "success"
            else None,
            name,
        )
        receipt["cases"].append(case)
        (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
