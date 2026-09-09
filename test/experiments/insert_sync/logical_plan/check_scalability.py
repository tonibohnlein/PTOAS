#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native sparse-requirement scalability gate, with bounded generated inputs.

One load supplies N independent vector readers, each writing a disjoint tile.
Physical phases and required relations grow linearly, while readiness needs one
logical stream/key. Direct pto-test-opt invocation avoids the test observer's
all-phase-pair export. Count envelopes reject avoidable quadratic provider or
endpoint work; timing and total checked work are recorded separately, not used
as noisy global complexity proofs. This is not device performance measurement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

from compare_boundaries import Boundaries
from measure import children, replay
from observations import analyze

HERE = Path(__file__).resolve().parent
SIZES = (8, 16, 32, 64)
STAGES = ("discovery", "handoffs", "barriers", "realization")
PRIMITIVES = {"normalize", "compose", "subtract", "contains", "restrictEndpoints",
              "subtractQualification", "subtractPartition", "subtractImplication"}
COUNTERS = ("order_requests", "order_built", "endpoint_comparisons")


def source_for(readers, scalar_noise):
    lines = [
        'module attributes {pto.target_arch = "a3"} {',
        '  func.func @reader_fanout(%src: !pto.ptr<f16>)',
        '      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {',
        '    %c0 = arith.constant 0 : index',
        '    %c1 = arith.constant 1 : index',
        '    %c16 = arith.constant 16 : index',
        '    %a0 = arith.constant 0 : i64',
        '    %input = pto.alloc_tile addr = %a0 : !pto.tile_buf<vec, 16x16xf16>',
        '    %view = pto.make_tensor_view %src, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<?x?xf16>',
        '    %part = pto.partition_view %view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x16xf16>',
    ]
    for n in range(readers):
        # Input occupies [0,512); every reader writes another exact 512-byte
        # physical interval. The largest fixture uses only 33,280 local bytes.
        lines += [f'    %a{n + 1} = arith.constant {512 * (n + 1)} : i64',
                  f'    %output{n} = pto.alloc_tile addr = %a{n + 1} : !pto.tile_buf<vec, 16x16xf16>']
    lines.append('    pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%input : !pto.tile_buf<vec, 16x16xf16>)')
    for n in range(readers):
        if scalar_noise:
            previous = "%c0"
            for j in range(4):
                value = f"%unrelated_{n}_{j}"
                lines.append(f'    {value} = arith.addi {previous}, %c1 : index')
                previous = value
        lines.append(f'    pto.tabs ins(%input : !pto.tile_buf<vec, 16x16xf16>) outs(%output{n} : !pto.tile_buf<vec, 16x16xf16>)')
    return "\n".join(lines + ['    return', '  }', '}', ''])


def parse_trace(text, required):
    stages, substages, primitives, cache, discovery = {}, [], {}, {}, {}
    for line in text.splitlines():
        match = re.fullmatch(
            r"logical stage (\S+) work (\d+) complete ([01]) seconds (\S+) "
            r"order_requests (\d+) order_built (\d+) endpoint_comparisons (\d+)", line)
        if match:
            name, work, complete, seconds, requests, built, comparisons = match.groups()
            assert name not in stages, ("duplicate constructor stage", name)
            stages[name] = {"work": int(work), "complete": complete == "1", "seconds": float(seconds),
                            "order_requests": int(requests), "order_built": int(built),
                            "endpoint_comparisons": int(comparisons)}
        match = re.fullmatch(r"logical substage (\S+) work (\d+) seconds (\S+)", line)
        if match:
            name, work, seconds = match.groups()
            substages.append({"name": name, "work": int(work), "seconds": float(seconds)})
        match = re.fullmatch(
            r"logical primitive (\S+) calls (\d+) nanoseconds (\d+) max_input_pieces (\d+)", line)
        if match:
            name, calls, nanoseconds, pieces = match.groups()
            assert name not in primitives, ("duplicate constructor primitive", name)
            primitives[name] = {"calls": int(calls), "nanoseconds": int(nanoseconds),
                                "max_input_pieces": int(pieces)}
        match = re.fullmatch(r"logical cache composition_builds (\d+) composition_pieces (\d+) endpoint_projections (\d+) endpoint_pieces (\d+)", line)
        if match:
            cache = dict(zip(("composition_builds", "composition_pieces", "endpoint_projections", "endpoint_pieces"),
                             map(int, match.groups())))
        match = re.fullmatch(r"logical discovery intervals (\d+) candidate_visits (\d+) pairs (\d+)", line)
        if match:
            discovery = dict(zip(("intervals", "candidate_visits", "pairs"), map(int, match.groups())))
    if required:
        assert cache and discovery, ("missing native preprocessing profile", cache, discovery)
        assert tuple(stages) == STAGES and all(s["complete"] for s in stages.values()), stages
        assert set(primitives) == PRIMITIVES, ("missing primitive profile", primitives)
        previous = {key: 0 for key in COUNTERS}
        for stage in stages.values():
            assert stage["order_built"] <= stage["order_requests"], stage
            stage["counter_delta"] = {key: stage[key] - previous[key] for key in COUNTERS}
            assert all(value >= 0 for value in stage["counter_delta"].values()), stage
            previous = {key: stage[key] for key in COUNTERS}
    return {"stages": stages, "substages": substages, "primitives": primitives, "cache": cache, "discovery": discovery}


def main():
    if not __debug__:
        raise RuntimeError("Scalability acceptance requires assertions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", required=True, type=Path)
    parser.add_argument("--python-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-opt", type=Path,
                        help="Optional older strict-constructor binary; failures are reported, never counted as current coverage")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 5 or not 0 < args.timeout <= 300:
        parser.error("Use 1..5 serial repetitions and a timeout in (0,300] seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.python_root.resolve()))
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PTOAS_LOGICAL_TRACE="1")
    provenance = {
        "opt_sha256": hashlib.sha256(args.opt.read_bytes()).hexdigest(),
        "python_library_sha256": hashlib.sha256((args.python_root / "ptoas/mlir/_mlir_libs/libPTOASCompiler.so").read_bytes()).hexdigest(),
        "baseline_opt_sha256": hashlib.sha256(args.baseline_opt.read_bytes()).hexdigest() if args.baseline_opt else None,
        "sizes": SIZES, "repetitions": args.repetitions, "device": "not-run",
        "counter_gate": "order_built and endpoint_comparisons: W64 <= 12*W8; W32 <= 3*W16; W64 <= 3*W32",
        "timing_contract": "Process time includes parsing/serialization; stage seconds are exclusive stage wall time. Primitive times are inclusive and must not be summed.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows, commands = [], []

    def save():
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")

    def run(binary, source, name, current):
        command = [str(binary.resolve()), "--mlir-disable-threading",
                   "--pto-insert-sync=planner=logical gm-alias=assume-disjoint-arguments", str(source.resolve())]
        start = time.monotonic()
        try:
            process = subprocess.run(command, text=True, capture_output=True, env=env, timeout=args.timeout)
            stdout, stderr, code = process.stdout, process.stderr, process.returncode
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
            code = "timeout"
        seconds = time.monotonic() - start
        (args.output / (name + ".stdout")).write_text(stdout)
        (args.output / (name + ".stderr")).write_text(stderr)
        commands.append({"name": name, "command": command, "exit": code, "seconds": seconds})
        save()
        if current:
            assert code == 0, (name, code, stderr[-4000:])
        trace = parse_trace(stderr, current and code == 0)
        return {"exit": code, "seconds": seconds, "trace": trace}, stdout

    for scalar_noise in (False, True):
        variant = "scalar_noise" if scalar_noise else "readers"
        for count in SIZES:
            name = f"{variant}.{count}"
            source = args.output / (name + ".input.pto")
            source.write_text(source_for(count, scalar_noise))
            row = {"variant": variant, "readers": count, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "physical_phases": count + 1, "distinct_output_tiles": count, "runs": []}
            rows.append(row)
            output = args.output / (name + ".logical.pto")
            first_text = None
            for repetition in range(args.repetitions):
                run_result, text = run(args.opt, source, name + f".current.{repetition}", True)
                row["runs"].append(run_result)
                if first_text is None:
                    first_text = text
                    output.write_text(text)
                else:
                    assert text == first_text, (name, "nondeterministic emitted output or checked-work metadata")
                save()
            old, new = analyze(source), analyze(output)
            for key in ("payload", "allocations", "views", "abi"):
                assert old[key] == new[key], (name, "changed", key)
            assert new["mechanisms"] == {"sets": 1, "waits": 1, "PIPE_ALL": 1, "named": {}}, (name, new["mechanisms"])
            assert len(new["event_ids_by_direction"]) == 1
            assert sum(len(keys) for keys in new["event_ids_by_direction"].values()) == 1
            with ir.Context() as context:
                context.enable_multithreading(False)
                pto.register_dialect(context, load=True)
                modules = [ir.Module.parse(path.read_text()) for path in (source, output)]
                functions = [next(op for op in children(m.operation) if op.name == "func.func") for m in modules]
                function = functions[1]
                integer = lambda key: ir.IntegerAttr(function.attributes[key]).value
                assert ir.StringAttr(function.attributes["pto.insert_sync.producer"]).value == "logical"
                assert ir.StringAttr(function.attributes["pto.insert_sync.logical_status"]).value == "applied"
                assert integer("pto.insert_sync.logical_requirements") == 2 * count + 1
                assert integer("pto.insert_sync.logical_streams") == 1
                row["requirements"] = integer("pto.insert_sync.logical_requirements")
                row["work"] = integer("pto.insert_sync.logical_work")
                observer = Boundaries()
                for _ in range(3):
                    original = replay(functions[0], ["src"])
                    emitted = replay(function, ["src"], observer=observer.observe)
                    assert original["payload_sha256"] == emitted["payload_sha256"], name
                    assert emitted["counts"]["pto.tload"] == 1 and emitted["counts"]["pto.tabs"] == count
                    assert emitted["counts"]["pto.set_flag"] == emitted["counts"]["pto.wait_flag"] == 1
                    assert not observer.tokens
                    assert all(observer.drained.get(pipe, -1) >= point for pipe, point in observer.issued.items())
            traces = [run_result["trace"] for run_result in row["runs"]]
            row["counters"] = {key: traces[0]["stages"]["realization"][key] for key in COUNTERS}
            row["stage_work"] = {stage: traces[0]["stages"][stage]["work"] for stage in STAGES}
            assert sum(row["stage_work"].values()) == row["work"], name
            for trace in traces[1:]:
                assert {key: trace["stages"]["realization"][key] for key in COUNTERS} == row["counters"], name
                assert {stage: trace["stages"][stage]["work"] for stage in STAGES} == row["stage_work"], name
            row["median_process_seconds"] = statistics.median(run_result["seconds"] for run_result in row["runs"])
            row["median_stage_seconds"] = {stage: statistics.median(trace["stages"][stage]["seconds"] for trace in traces)
                                           for stage in STAGES}
            row["mechanisms"] = new["mechanisms"]
            row["event_ids_by_direction"] = new["event_ids_by_direction"]
            if args.baseline_opt:
                row["baseline_runs"] = []
                for repetition in range(args.repetitions):
                    baseline, text = run(args.baseline_opt, source, name + f".baseline.{repetition}", False)
                    row["baseline_runs"].append(baseline)
                    if baseline["exit"] == 0:
                        path = args.output / (name + f".baseline.{repetition}.pto")
                        path.write_text(text)
                        projection = analyze(path)
                        for key in ("payload", "allocations", "views", "abi"):
                            assert old[key] == projection[key], (name, "baseline changed", key)
                        baseline["mechanisms"] = projection["mechanisms"]
                if all(sample["exit"] == 0 for sample in row["baseline_runs"]):
                    row["baseline_median_process_seconds"] = statistics.median(sample["seconds"] for sample in row["baseline_runs"])
            save()
            print(name, "work", row["work"], "counters", row["counters"], flush=True)

    gates = []
    for variant in ("readers", "scalar_noise"):
        family = {row["readers"]: row for row in rows if row["variant"] == variant}
        for counter in ("order_built", "endpoint_comparisons"):
            values = {n: family[n]["counters"][counter] for n in SIZES}
            assert values[8] > 0, (variant, counter, "vacuous instrumentation")
            for small, large, bound in ((8, 64, 12), (16, 32, 3), (32, 64, 3)):
                passed = values[large] <= bound * values[small]
                gates.append({"variant": variant, "counter": counter, "small": small, "large": large,
                              "small_value": values[small], "large_value": values[large],
                              "allowed_ratio": bound, "actual_ratio": values[large] / values[small], "passed": passed})
    for n in SIZES:
        clean, noise = (next(row for row in rows if row["readers"] == n and row["variant"] == variant)
                        for variant in ("readers", "scalar_noise"))
        for counter in ("order_built", "endpoint_comparisons"):
            gates.append({"case": f"unrelated_scalars.{n}", "counter": counter,
                          "clean": clean["counters"][counter], "noise": noise["counters"][counter],
                          "passed": clean["counters"][counter] == noise["counters"][counter]})
    (args.output / "growth_gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    failed = [gate for gate in gates if not gate["passed"]]
    assert not failed, ("Avoidable provider/endpoint growth", failed)
    print("Native scalability gate passed: eight strict inputs, linear semantic scale, constant stream/key count, bounded provider/endpoint growth")


if __name__ == "__main__":
    main()
