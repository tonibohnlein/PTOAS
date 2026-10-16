# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Paired finite pipeline qualification; artifacts and timings belong to an explicit output directory."""
import argparse
import hashlib
import itertools
import json
import os
import resource
import sys
from pathlib import Path
import re
import statistics
import time

from check_deferred_family_output import arithmetic, executable, invoke
from check_finite_allocation import physical_check
from check_finite_family_pipeline import render, trace
from check_finite_guarded_analysis import compute, load, validate

FLAGS = "gm-alias=may-not-alias allocation-ids=0,1,2,3,4,5 profile-stages=true"
ALLOCATE = "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5"


def body_for(form, size):
    chunks, effects = [], {}
    if form == "guarded":
        chunks.append(load("L0"))
        effects["L0"] = ({"mat"}, {"left"})
    for i in range(size):
        a, b = f"L{i}", f"C{i}"
        if form != "guarded":
            effects[a] = ({"mat"}, {"left"})
        effects[b] = ({"left", "right"}, {f"private{i}"} if form == "guarded" else {"acc"})
        if form == "concrete":
            chunks += [load(a), compute(b)]
        elif form == "guarded":
            chunks += [f"%addr{i} = arith.constant {1024 * (i + 1)} : i64",
                       f"%out{i} = pto.alloc_tile addr = %addr{i} : !acc",
                       "scf.if %" + ("g" if i % 2 else "h") + " { " +
                       compute(b).replace("outs(%acc", f"outs(%out{i}") + " }"]
        elif form == "writers":
            chunks += ["scf.if %" + ("g" if i % 2 else "h") + " { " + load(a) + " }"]
        else:
            chunks += ["scf.if %g { " + load(a) + " }", "scf.if %h { " + compute(b) + " }"]
    if form == "writers":
        chunks.append(compute("tail"))
        effects["tail"] = ({"left", "right"}, {"acc"})
    return "\n".join(chunks), effects


def labels_for(form, size, g, h):
    labels = ["L0"] if form == "guarded" else []
    for i in range(size):
        active = g if i % 2 else h
        if form == "concrete" or (form == "composed" and g) or (form == "writers" and active):
            labels.append(f"L{i}")
        if form == "concrete" or (form == "guarded" and active) or (form == "composed" and h):
            labels.append(f"C{i}")
    if form == "writers":
        labels.append("tail")
    return labels


def profiles(stderr, prefix="frontier-control-profile "):
    result = []
    for line in stderr.splitlines():
        if line.startswith(prefix):
            result.append(json.loads(line.removeprefix(prefix)))
    return result


def compile_route(optimizer, source, route):
    command = [optimizer, "--mlir-disable-threading"]
    option = " deferred-output=true" if route == "saved" else " physical-output=true"
    start = time.perf_counter()
    first = invoke([*command, "--pto-frontier-analysis=" + FLAGS + option, "-"], source)
    first_seconds = time.perf_counter() - start
    if route == "saved":
        middle = time.perf_counter()
        second = invoke([*command, ALLOCATE, "-"], first.stdout)
        second_seconds = time.perf_counter() - middle
        if "pto.logical_set" in first.stdout or arithmetic(first.stdout) != arithmetic(source):
            raise RuntimeError("saved finite output eagerly materialized synchronization")
    else:
        second, second_seconds = first, 0.0
    if "pto.logical_" in second.stdout or "pto.deferred_family" in second.stdout:
        raise RuntimeError("physical output contains unallocated synchronization")
    metrics = {"seconds": first_seconds + second_seconds, "producer_seconds": first_seconds,
               "allocation_process_seconds": second_seconds,
               "profiles": profiles(first.stderr) + (profiles(second.stderr) if route == "saved" else []),
               "stage_profiles": profiles(first.stderr, "frontier-profile "),
               "producer_stderr": first.stderr,
               "allocation_stderr": second.stderr if route == "saved" else "",
               "arithmetic_total": arithmetic(second.stdout),
               "arithmetic_added": arithmetic(second.stdout) - arithmetic(source),
               "sets": len(re.findall(r"\bpto\.set_flag(?:_dyn)?\b", second.stdout)),
               "waits": len(re.findall(r"\bpto\.wait_flag(?:_dyn)?\b", second.stdout))}
    return second.stdout, metrics


def signature(events):
    identities, live, out = {}, set(), []
    for event in events:
        kind = event["kind"]
        if kind == "payload":
            out.append((kind, event["label"], event["pipe"], tuple(event.get("coordinates", []))))
        elif kind == "barrier":
            out.append((kind, event["gap"], event["pipe"]))
        else:
            identity = (event["plan"], event["record"], event["source_ordinal"],
                        tuple(event.get("members", [])), event["source_pipe"], event["target_pipe"])
            if kind == "set":
                if identity in identities:
                    raise RuntimeError("duplicate publication identity")
                identities[identity] = len(identities)
                live.add(identity)
            elif identity not in live:
                raise RuntimeError("missing or duplicate logical consumption")
            else:
                live.remove(identity)
            coordinates = event.get("members", [])
            # endpoint_piece marks the first tuple field as a provenance record label.
            # Keep it in the matching key above; only cross-plan comparison renames it.
            if event.get("record_label"):
                if not coordinates:
                    raise RuntimeError("record-labelled command has no record field")
                coordinates = coordinates[1:]
            out.append((kind, event["gap"], event["pipe"], event["source_pipe"], event["target_pipe"],
                        identities[identity], event["source_ordinal"], tuple(coordinates)))
    if live:
        raise RuntimeError("unconsumed logical publication")
    return out


def compile_regional(args, source, folder, outputs, metrics):
    logical = None
    for route in ("logical", "saved", "combined"):
        if source.count("test.trace_arguments =") != 1:
            raise RuntimeError("composition fixture needs one function binding marker")
        supplied = source.replace("test.trace_arguments =",
            f'test.finite_output = "{route}", test.trace_arguments =', 1)
        regional = invoke([args.trace_tool, "--gm-alias=may-not-alias",
                           "--finite-composed-output", "-"], supplied)
        structural = profiles(regional.stderr, "finite-composition-profile ")
        if len(structural) != 1 or structural[0]["children"] < 2 or not structural[0]["crossings"]:
            raise RuntimeError("composition was not exercised")
        if route == "logical":
            logical = regional.stdout
        else:
            lowered = (invoke([args.trace_tool, "--profile-deferred-families", "-"], regional.stdout)
                       if route == "saved" else regional)
            outputs["regional-" + route] = lowered.stdout
            (folder / ("regional-" + route + "-physical.pto")).write_text(lowered.stdout, encoding="utf-8")
            metrics["regional-" + route] = {"composition": structural,
                "profiles": profiles(regional.stderr) + (profiles(lowered.stderr) if route == "saved" else []),
                "producer_stderr": regional.stderr,
                "allocation_stderr": lowered.stderr if route == "saved" else "",
                "arithmetic_total": arithmetic(lowered.stdout),
                "arithmetic_added": arithmetic(lowered.stdout) - arithmetic(source)}
        (folder / ("regional-" + route + ".pto")).write_text(regional.stdout, encoding="utf-8")
    return logical


def qualify(args, source, form, size, effects, folder):
    variants, signatures = {}, {}
    for name, optimizer in (("baseline", args.baseline), ("candidate", args.optimizer)):
        logical = invoke([optimizer, "--mlir-disable-threading",
                          "--pto-frontier-analysis=" + FLAGS, "-"], source).stdout
        (folder / (name + "-logical.pto")).write_text(logical, encoding="utf-8")
        outputs, metrics = {}, {}
        for route in ("saved", "combined"):
            outputs[route], metrics[route] = compile_route(optimizer, source, route)
            (folder / (name + "-" + route + ".pto")).write_text(outputs[route], encoding="utf-8")
        if name == "candidate":
            saved = invoke([optimizer, "--mlir-disable-threading", "--pto-frontier-analysis=" +
                            FLAGS + " deferred-output=true", "-"], source).stdout
            diagnostic = invoke([args.trace_tool, "--profile-deferred-families", "-"], saved)
            if diagnostic.stdout.strip() != outputs["saved"].strip():
                raise RuntimeError("profiling changed saved allocation output")
            metrics["saved"]["allocation_profiles"] = profiles(diagnostic.stderr)
        regional_logical = None
        if name == "candidate" and form == "composed":
            regional_logical = compile_regional(args, source, folder, outputs, metrics)
        signatures[name] = []
        executions = []
        for g, h in itertools.product((0, 1), repeat=2):
            values = (1, g, h, 3)
            original = trace(args.trace_tool, source, values)
            logical_trace = trace(args.trace_tool, logical, values)
            validate({"accepted": True, "trace": logical_trace}, labels_for(form, size, g, h), effects)
            signatures[name].append(signature(logical_trace["events"]))
            work = {"binding": list(values), "source_arithmetic": original["arithmetic_visits"], "routes": {}}
            regional_trace = trace(args.trace_tool, regional_logical, values) if regional_logical else None
            if regional_trace:
                validate({"accepted": True, "trace": regional_trace}, labels_for(form, size, g, h), effects)
                if signature(regional_trace["events"]) != signature(logical_trace["events"]):
                    raise RuntimeError("flat/composed communication mismatch")
            for route, physical in outputs.items():
                actual = trace(args.trace_tool, physical, values)
                expected = regional_trace if route.startswith("regional-") else logical_trace
                physical_check({"accepted": True, "allocated": True, "trace": expected,
                                "physical": actual}, set(range(6)))
                counts = {key: actual[key] for key in ("arithmetic_visits", "select_visits", "guard_visits")}
                counts["added_arithmetic_visits"] = actual["arithmetic_visits"] - original["arithmetic_visits"]
                work["routes"][route] = counts
            executions.append(work)
        variants[name] = {"routes": metrics, "executions": executions}
    if signatures["baseline"] != signatures["candidate"]:
        raise RuntimeError(f"baseline/candidate ordered communication mismatch: {form}/{size}")
    return variants


def timed_pairs(args, source):
    result = {}
    for route in ("saved", "combined"):
        # One unrecorded warm-up of each binary and complete route.
        for optimizer in (args.baseline, args.optimizer):
            compile_route(optimizer, source, route)
        pairs = []
        for pair in range(args.pairs):
            sample = {}
            order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
            for name in order:
                optimizer = args.baseline if name == "baseline" else args.optimizer
                _, measurement = compile_route(optimizer, source, route)
                sample[name] = measurement
            pairs.append(sample)
        result[route] = {"pairs": pairs, "median_candidate_over_baseline": statistics.median(
            p["candidate"]["seconds"] / p["baseline"]["seconds"] for p in pairs)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("optimizer", "trace-tool", "fixture", "output"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--sizes", default="8,16,32,64")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--timing-size", type=int, default=32)
    args = parser.parse_args()
    args.optimizer = executable(args.optimizer)
    args.baseline = executable(args.baseline or args.optimizer)
    args.trace_tool = executable(args.trace_tool)
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or min(sizes) < 1 or max(sizes) > 4096 or args.pairs < 0 or args.pairs > 20:
        parser.error("sizes must be 1..4096 and pairs 0..20")
    if args.pairs and args.timing_size not in sizes:
        parser.error("timing size must be one of the qualified sizes")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    template = Path(args.fixture).read_text(encoding="utf-8")
    receipt = {"scope": "Static finite fixtures; four admitted guard bindings per case; no device execution",
               "binaries": {key: {"path": value, "sha256": hashlib.sha256(Path(value).read_bytes()).hexdigest()}
                            for key, value in (("baseline", args.baseline), ("candidate", args.optimizer),
                                               ("trace_tool", args.trace_tool))},
               "flags": FLAGS, "warmup": "one complete route per binary before requested interleaved pairs",
               "requested_pairs": args.pairs, "status": "running",
               "memory": "peak memory not measured",
               "execution": {"affinity": sorted(os.sched_getaffinity(0)),
                             "address_space_limit": list(resource.getrlimit(resource.RLIMIT_AS)),
                             "python": sys.version,
                             "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
               "cases": []}
    (output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for form, size in itertools.product(("concrete", "guarded", "writers", "composed"), sizes):
        body, effects = body_for(form, size)
        source = render(template, body)
        folder = output / f"{form}-{size}"
        folder.mkdir(exist_ok=True)
        (folder / "input.pto").write_text(source, encoding="utf-8")
        item = {"form": form, "size": size, "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "variants": qualify(args, source, form, size, effects, folder)}
        if args.pairs and size == args.timing_size:
            item["timings"] = timed_pairs(args, source)
        receipt["cases"].append(item)
        (output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"finite emission qualified: {form}/{size}", flush=True)
    receipt["status"] = "complete"
    (output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
