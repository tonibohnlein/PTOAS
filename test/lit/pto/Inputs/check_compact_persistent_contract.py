# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Section 5 validation on small original nested pipelines; never a device test."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import shutil
import subprocess
import time
from compact_persistent_oracle import boundary, merge, occurrences, probes, required
from check_logical_insertion import closure_with_commands
from check_finite_allocation import physical_check


def run(command, output, timeout=120):
    start = time.perf_counter()
    with output.open("w") as stdout, output.with_suffix(output.suffix + ".stderr").open(
        "w"
    ) as stderr:
        result = subprocess.run(
            list(map(str, command)),
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"{command[0]} failed: {output.with_suffix(output.suffix + '.stderr').read_text()[-2000:]}"
        )
    return time.perf_counter() - start


def variant(source, waves, trips, inner, core, epochs=1):
    text = source.replace("%c13 = arith.constant 13", f"%c13 = arith.constant {waves}")
    text = text.replace("%c32 = arith.constant 32", f"%c32 = arith.constant {trips}")
    text = text.replace(
        "%cid_i64 = pto.get_block_idx", f"%cid_i64 = arith.constant {core} : i64"
    )
    text = text.replace(
        "    %c0 =", f"    %inner_limit = arith.constant {inner} : index\n    %c0 =", 1
    )
    text = text.replace("%kk = %c0 to %c4 step", "%kk = %c0 to %inner_limit step")
    if epochs != 1:
        text = text.replace(
            "    %c0 =",
            f"    %epoch_limit = arith.constant {epochs} : index\n    %c0 =",
            1,
        )
        text = text.replace(
            "    scf.for %wave",
            "    scf.for %epoch = %c0 to %epoch_limit step %c1 {\n    scf.for %wave",
            1,
        )
        text = text.replace(
            "    // Uniform invocation", "    }\n    // Uniform invocation", 1
        )
    return text


def annotated(text, items, cells):
    def array(values):
        return "array<i64: " + ", ".join(map(str, values)) + ">"

    attrs = [
        "test.contract_occurrences = ["
        + ", ".join(array([x["phase"], *x["coordinates"]]) for x in items)
        + "]",
        "test.contract_cells = [" + ", ".join(array(p) for p in cells) + "]",
    ]
    return text.replace(
        "attributes {pto.kernel_kind",
        "attributes {" + ", ".join(attrs) + ", pto.kernel_kind",
        1,
    )


def trace(args, text, path):
    # Function, not module, owns the interpreter request.
    text, count = re.subn(
        r"(func\.func\s+@\w+\([^\n]*?\)\s*)attributes\s*\{",
        r"\1attributes {test.trace_existing, ",
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Cannot annotate function for tracing")
    path.write_text(text)
    output = path.with_suffix(".trace.json")
    run([args.trace_tool, "--structured-trace", path], output)
    result = json.loads(output.read_text())
    if not result["accepted"] or result["trace"]["error"]:
        raise RuntimeError(str(result))
    return result["trace"]


def emitted_order(trace):
    commands = []
    for event in trace["events"]:
        if event["kind"] == "payload":
            continue
        command = dict(event)
        if command["kind"] != "barrier":
            command["identity"] = (
                event["plan"],
                event["record"],
                event["source_ordinal"],
                tuple(event.get("members", [])),
            )
        commands.append(command)
    pipes = [e["pipe"] for e in trace["events"] if e["kind"] == "payload"]
    return closure_with_commands(pipes, commands)


def check_accesses(args, original, word, directory, required_order=required):
    # Annotate only the known test fixture's counted loops. Production input
    # remains nested; this flattening is solely an independent extraction test.
    closes = []
    for position in range(len(original)):
        if not original.startswith("scf.for ", position):
            continue
        opening = original.index("{", position)
        depth, end = 1, opening + 1
        while depth and end < len(original):
            depth += (original[end] == "{") - (original[end] == "}")
            end += 1
        if depth:
            raise RuntimeError("Unbalanced fixture loop")
        closes.append(end)
    text = original
    for end in sorted(closes, reverse=True):
        text = text[:end] + ' {pto.unroll = "full"}' + text[end:]
    path = directory / "access-unroll-0.pto"
    path.write_text(text)
    for level in range(len(closes) + 1):
        target = directory / f"access-unroll-{level + 1}.pto"
        run(
            [
                args.optimizer,
                "--mlir-disable-threading",
                "--canonicalize",
                "--pto-unroll-loops",
                path,
                "-o",
                target,
            ],
            directory / f"unroll-{level}.stdout",
        )
        path = target
    output = directory / "accesses.json"
    run([args.trace_tool, "--gm-alias=may-not-alias", "--step1-json", path], output)
    recovered = json.loads(output.read_text())["access_records"]
    expected = [access for item in word for access in item["effects"]]
    if len(recovered) != len(expected):
        raise RuntimeError(
            f"Access count mismatch: {len(recovered)} != {len(expected)}"
        )
    comparisons = 0
    for index, (actual, reference) in enumerate(zip(recovered, expected)):
        if actual["space"] != reference["space"] or actual["mode"] != reference["mode"]:
            raise RuntimeError(f"Access mode/space mismatch at {index}")
        if not actual["ranges_materialized"]:
            raise RuntimeError(f"Concrete access not materialized at {index}")
        intervals = merge((value["begin"], value["end"]) for value in actual["ranges"])
        if intervals != reference["intervals"]:
            raise RuntimeError(
                f"Supplied-layout access mismatch at {index}: {intervals[:3]} != {reference['intervals'][:3]}"
            )
        if reference["space"] == 1:
            if any(
                value["base"].get("number") != reference["base"]
                or not value["base"].get("block_argument")
                for value in actual["ranges"]
            ):
                raise RuntimeError(f"GM base mismatch at {index}")
        comparisons += 1
    explicit = directory / "flat-explicit.json"
    run(
        [args.trace_tool, "--gm-alias=may-not-alias", "--explicit-analysis", path],
        explicit,
    )
    analysis = json.loads(explicit.read_text().splitlines()[0])
    _, closure, covers, _ = required_order(word)
    expected_covers = {(a // 2, b // 2) for a, b in covers}
    if (
        analysis.get("error")
        or {tuple(edge) for edge in analysis["retained"]} != expected_covers
    ):
        raise RuntimeError(f"Independent flat-cover mismatch: {analysis.get('error')}")
    expected_queries = [
        [bool(row >> b & 1) for b in range(len(closure))] for row in closure
    ]
    if analysis["event_reachable"] != expected_queries:
        raise RuntimeError("Independent flat reachability mismatch")
    return comparisons


def unroll_inner(args, text, directory):
    opening = text.index("{", text.index("scf.for %kk"))
    depth, end = 1, opening + 1
    while depth and end < len(text):
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    if depth:
        raise RuntimeError("Unbalanced inner-loop fixture")
    path = directory / "before-inner-unroll.pto"
    path.write_text(text[:end] + ' {pto.unroll = "full"}' + text[end:])
    target = directory / "inner-unrolled.pto"
    run(
        [
            args.optimizer,
            "--mlir-disable-threading",
            "--pto-unroll-loops",
            path,
            "-o",
            target,
        ],
        directory / "inner-unroll.stdout",
    )
    return target.read_text()


def check_case(args, source, spec, static=False):
    waves, trips, inner, core, epochs = spec
    name = f"w{waves}-k{trips}-i{inner}-c{core}-e{epochs}" + (
        "-static" if static else ""
    )
    directory = args.output / name
    directory.mkdir(parents=True, exist_ok=True)
    original = variant(source, *spec)
    oracle_items = occurrences(*spec)
    if getattr(args, "replay_only", False):
        original = variant(source, waves, trips, inner, core, 2).replace(
            "%epoch_limit = arith.constant 2", f"%epoch_limit = arith.constant {epochs}"
        )
        if epochs == 1:
            oracle_items = [
                x
                for x in occurrences(waves, trips, inner, core, 2)
                if x["coordinates"][0] == 0
            ]
    items = [dict(item) for item in oracle_items]
    if static:
        original = unroll_inner(args, original, directory)
        for item in items:
            if item["phase"] == 8:
                item["phase"] = 4 + 4 * inner
            elif item["phase"] in (4, 5, 6, 7):
                item["phase"] += 4 * item["coordinates"][-1]
                item["coordinates"] = item["coordinates"][:-1]
    cells = probes(oracle_items)
    path = directory / "input.pto"
    path.write_text(annotated(original, items, cells))
    output = directory / "contract.json"
    seconds = run([args.trace_tool, "--compact-region-contract", path], output)
    result = json.loads(output.read_text())
    if result.get("error"):
        raise RuntimeError(f"{name}: {result}")
    expected_present = [x["active"] for x in items]
    if result["present"] != expected_present:
        first = next(
            i
            for i, pair in enumerate(zip(result["present"], expected_present))
            if pair[0] != pair[1]
        )
        raise RuntimeError(
            f"{name}: presence mismatch {items[first]} actual={result['present'][first]}"
        )
    word, reach, covers, full = required(oracle_items)
    expected = [
        "".join("1" if row >> b & 1 else "0" for b in range(len(full))) for row in full
    ]
    if result["event_rows"] != expected:
        first = next(
            (a, b, observed[b], expected[a][b])
            for a, observed in enumerate(result["event_rows"])
            for b in range(len(expected))
            if observed[b] != expected[a][b]
        )
        raise RuntimeError(f"{name}: all-event query mismatch {first}")
    if result.get("retained_demand_error"):
        raise RuntimeError(f"{name}: {result['retained_demand_error']}")
    active = [i for i, item in enumerate(items) if item["active"]]
    expected_records = {(active[a // 2], active[b // 2]) for a, b in covers}
    actual_records = {tuple(edge) for edge in result["retained_demands"]}
    if actual_records != expected_records:
        raise RuntimeError(
            f"{name}: regional cover mismatch; "
            f"extra={sorted(actual_records - expected_records)[:4]}, "
            f"missing={sorted(expected_records - actual_records)[:4]}"
        )
    if result["boundaries"].get("error"):
        raise RuntimeError(f"{name}: {result['boundaries']}")
    if len(result["boundaries"]["rows"]) != len(cells):
        raise RuntimeError(f"{name}: incomplete selector output")
    for cell, actual in zip(cells, result["boundaries"]["rows"]):
        expected_boundary = boundary(oracle_items, cell)
        actual = {
            k: {p: v for p, v in value.items() if v}
            if isinstance(value, dict)
            else value
            for k, value in actual.items()
        }
        if actual != expected_boundary:
            raise RuntimeError(
                f"{name}: selector mismatch {cell}: {actual} != {expected_boundary}"
            )
    access_comparisons = check_accesses(args, original, word, directory)
    plain = directory / "plain.pto"
    plain.write_text(original)
    logical = directory / "logical.pto"
    logical_seconds = run(
        [
            args.optimizer,
            "--mlir-disable-threading",
            "--pto-frontier-analysis=gm-alias=may-not-alias allocation-ids=0,1,2,3,4,5",
            plain,
            "-o",
            logical,
        ],
        directory / "logical.stdout",
    )
    physical = directory / "physical.pto"
    allocation_seconds = run(
        [
            args.optimizer,
            "--mlir-disable-threading",
            "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5",
            logical,
            "-o",
            physical,
        ],
        directory / "physical.stdout",
    )
    logical_trace = trace(args, logical.read_text(), directory / "logical-trace.pto")
    physical_trace = trace(args, physical.read_text(), directory / "physical-trace.pto")
    physical_check(
        {
            "accepted": True,
            "allocated": True,
            "trace": logical_trace,
            "physical": physical_trace,
        },
        set(range(6)),
    )
    payloads = [e for e in logical_trace["events"] if e["kind"] == "payload"]
    expected_payloads = [(x["phase"], x["coordinates"]) for x in items if x["active"]]
    if [(x["type"], x["coordinates"]) for x in payloads] != expected_payloads:
        raise RuntimeError("Different payload sequence")
    emitted = emitted_order(logical_trace)
    required_ci = {
        (a, b)
        for a in range(len(word))
        for b in range(len(word))
        if reach[2 * a + 1] >> (2 * b) & 1
    }
    actual_ci = {
        (a, b)
        for a in range(len(word))
        for b in range(len(word))
        if emitted[2 * a + 1] >> (2 * b) & 1
    }
    if required_ci - actual_ci:
        raise RuntimeError(
            f"{name}: missing payload prerequisites {sorted(required_ci - actual_ci)[:3]}"
        )
    extra = sorted(actual_ci - required_ci)
    witness = None
    if args.quick and extra:
        raise RuntimeError(
            f"{name}: unexpected extra payload order in qualified fixture"
        )
    if extra:
        a, b = extra[0]
        witness = {
            "source": {k: word[a][k] for k in ("phase", "coordinates", "pipe")},
            "target": {k: word[b][k] for k in ("phase", "coordinates", "pipe")},
        }
    return {
        "name": name,
        "spec": spec,
        "input_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "selected_backend": result["backend"],
        "access_comparisons": access_comparisons,
        "potential_occurrences": len(items),
        "payloads": len(word),
        "event_query_comparisons": len(full) ** 2,
        "storage_probes": len(cells),
        "minimum_covers": len(covers),
        "cover_scope": (
            "Independent interval graph covers agree with flat explicit retained edges; "
            "actual retained regional records are expanded and deduplicated against the same covers"
        ),
        "extra_ci": len(extra),
        "extra_witness": witness,
        "missing_ci": 0,
        "physical_reuse": "passed",
        "seconds": {
            "contract_probe": seconds,
            "production_logical": logical_seconds,
            "allocation": allocation_seconds,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("optimizer", "trace-tool", "input", "output"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Validate explicit zero-through-three counter-reset visits",
    )
    parser.add_argument(
        "--static-stages",
        action="store_true",
        help="Also validate one/two-stage flat K bodies",
    )
    args = parser.parse_args()
    for name in ("optimizer", "trace_tool"):
        supplied = getattr(args, name)
        resolved = (
            str(supplied.resolve())
            if supplied.is_file()
            else shutil.which(str(supplied))
        )
        if not resolved:
            parser.error(f"Unavailable tool: {supplied}")
        setattr(args, name, Path(resolved))
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    source = args.input.read_text()
    specs = [(1, 1, 1, 0, 1), (2, 2, 2, 0, 1), (2, 3, 4, 255, 1)]
    if not args.quick:
        specs += [
            (1, 0, 2, 0, 1),
            (0, 3, 4, 0, 1),
            (2, 1, 0, 0, 1),
            (2, 3, 4, 0, 1),
            (2, 4, 4, 0, 1),
            (2, 2, 2, 0, 2),
        ]
    if args.replay_only:
        specs = [(2, 2, 2, 0, epochs) for epochs in (0, 1, 2, 3)]
    receipt = {
        "explicit_replay": args.replay_only,
        "scope": (
            "Supplied NZ layout, explicit GM disjointness and ACC hardware protection; "
            "no native/device qualification"
        ),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "optimizer_sha256": hashlib.sha256(args.optimizer.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(args.trace_tool.read_bytes()).hexdigest(),
        "cases": [],
    }
    experiments = [(spec, False) for spec in specs]
    if args.static_stages:
        experiments += [((2, 4, stages, 0, 1), True) for stages in (1, 2)]
    for spec, static in experiments:
        try:
            case = check_case(args, source, spec, static)
            case["status"] = "passed"
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            case = {"spec": spec, "status": "unresolved", "diagnostic": str(error)}
            receipt["cases"].append(case)
            (args.output / "results.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            if args.quick:
                raise
            print(spec, str(error), flush=True)
            continue
        receipt["cases"].append(case)
        (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            case["name"],
            "queries/selectors/reuse passed; extra CI",
            case["extra_ci"],
            flush=True,
        )
    receipt["complete"] = True
    receipt["passed"] = sum(case["status"] == "passed" for case in receipt["cases"])
    receipt["unresolved"] = len(receipt["cases"]) - receipt["passed"]
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
