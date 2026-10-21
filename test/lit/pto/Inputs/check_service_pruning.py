# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Actual optional emission plus independent finite delivery and command oracles."""

import argparse
import copy
import json
import re
from pathlib import Path
from assess_whole_kernel_access import invoke, digest, annotate
from check_current_cut_services import fixture, qualify
from check_expanded_endpoint_adapter import fixture as pipeline
from check_operation_current_cut_services import fixture as scalar_fixture, Memory
from reaching_access_research import Graph, require
from current_cut_delta_oracle import check_delta, normalized_payload


def profile(run):
    for line in Path(run["stderr"]).read_text().splitlines():
        if line.startswith("current-cut-work "):
            return json.loads(line.removeprefix("current-cut-work "))
        if line.startswith("frontier-prefix-result "):
            return json.loads(line.removeprefix("frontier-prefix-result "))["work"]
    raise AssertionError("no service profile")


def emit_mode(args, mode, source, output, *, alias="may-not-alias"):
    experimental = invoke(
        [args.trace_tool, "--gm-alias=" + alias, mode, source],
        output.with_suffix(".experimental.pto"),
        35,
    )
    require(experimental["status"] == "success", str(experimental))
    if not getattr(args, "production_tool", None):
        output.write_bytes(Path(experimental["stdout"]).read_bytes())
        experimental["stdout"] = str(output)
        return experimental
    pool = re.search(
        r"test.current_cut_ids = array<i64:\s*([^>]*)>", source.read_text()
    )
    options = "conservative-prefix=true physical-output=true gm-alias=" + alias
    if mode == "--current-cut-pruned":
        options += " prune-prefix-services=true"
    if pool:
        options += " allocation-ids=" + pool[1].replace(" ", "")
    production = invoke(
        [args.production_tool, "--pto-frontier-analysis=" + options, source], output, 35
    )
    require(
        production["status"] == "success",
        str(production) + Path(production["stderr"]).read_text(),
    )
    require(
        Path(experimental["stdout"]).read_bytes().rstrip(b"\n") == output.read_bytes().rstrip(b"\n"),
        "production/experimental command or payload difference",
    )
    # MlirOptMain appends one extra terminal newline. Nothing inside the IR
    # (including commands, operands, cuts and payloads) is normalized.
    production["comparison"] = "identical-except-terminal-newlines"
    production["experimental"] = experimental
    return production


def reference_delivery(data, full, kept, protected):
    """Whole-state iteration, independent of native delta-worklist implementation."""
    if any(p not in kept[i] for i, row in enumerate(protected) for p in row):
        return False
    if any(p not in full[i] for i, row in enumerate(kept) for p in row):
        return False
    sites = data["sites"]
    if any(
        s["pipe"] in full[i] and s["pipe"] not in kept[i] for i, s in enumerate(sites)
    ):
        return False
    graph = Graph(data["tree"])
    pipes = {s["pipe"] for s in sites}
    incoming, outgoing = [set() for _ in graph.nodes], [set() for _ in graph.nodes]

    def delivered(site, debt):
        return (
            site is not None
            and debt[3] == sites[site]["pipe"]
            and debt[2] in kept[site]
        )

    changed = True
    while changed:
        changed = False
        for node, site in enumerate(graph.nodes):
            before = set().union(*(outgoing[p] for p in graph.predecessors[node]))
            incoming[node] = before
            after = {d for d in before if not delivered(site, d)}
            if site is not None:
                after |= {
                    (c, mode, sites[site]["pipe"], q)
                    for c, bits in enumerate(sites[site]["facts"])
                    for mode in (1, 2)
                    if bits & mode
                    for q in pipes
                }
            if after != outgoing[node]:
                outgoing[node] = after
                changed = True
    return not any(
        not delivered(site, debt)
        and debt[3] == sites[site]["pipe"]
        and sites[site]["facts"][debt[0]]
        and (debt[1] == 2 or sites[site]["facts"][debt[0]] & 2)
        for node, site in enumerate(graph.nodes)
        if site is not None
        for debt in incoming[node]
    )


def model_case(args, data, name):
    path = args.output / (name + ".json")
    data = dict(data, prune=True)
    path.write_text(json.dumps(data))
    run = invoke(
        [args.trace_tool, "--source-pipe-json", path], path.with_suffix(".out.json"), 35
    )
    require(run["status"] == "success", str(run))
    result = json.loads(Path(run["stdout"]).read_text())
    require(not result["error"], str(result))
    prune = result["pruning"]
    if "kept" in prune:
        full = data.get("baseline_services", result["possible"])
        protected = data.get("protected_services", [[] for _ in data["sites"]])
        delivered = reference_delivery(data, full, prune["kept"], protected)
        # Debt propagation sees a set of deliveries. Joint acceptance also
        # requires preserving their program order, which a set cannot prove.
        ordered = all(
            [p for p in full[i] if p in row] == row
            for i, row in enumerate(prune["kept"])
        )
        require(prune["accepted"] == (delivered and ordered), str(prune))
    if "fact_transmissions" in prune:
        graph = Graph(data["tree"])
        h, k = data["classes"], len({s["pipe"] for s in data["sites"]})
        bound = 2 * h * k * k
        require(
            prune["facts_retained"] <= bound * len(graph.nodes),
            "certificate state bound",
        )
        require(
            prune["fact_transmissions"] <= bound * sum(map(len, graph.successors)),
            "certificate edge bound",
        )
    return prune


def joint_negatives(args):
    sites = [
        dict(pipe=4, facts=[2, 0]),
        dict(pipe=1, facts=[0, 1]),
        dict(pipe=1, facts=[1, 0]),
    ]
    data = dict(
        classes=2,
        sites=sites,
        tree=[dict(kind="payload", site=i) for i in range(3)],
        baseline_services=[[], [4], [4]],
    )
    anchored = model_case(args, data, "retained-anchor")
    require(anchored["kept"] == [[], [4], []], "deleted service became an anchor")
    rows = []
    for name, mask, expected in (
        ("first", [[], [], [4]], True),
        ("second", [[], [4], []], True),
        ("both", [[], [], []], False),
    ):
        row = model_case(args, dict(data, candidate_services=mask), "mutual-" + name)
        require(row["accepted"] == expected, str(row))
        rows.append(row)
    # Empty effects cannot identify a separately supplied non-storage path.
    empty = dict(
        classes=0,
        sites=[dict(pipe=4, facts=[]), dict(pipe=1, facts=[])],
        tree=[dict(kind="payload", site=0), dict(kind="payload", site=1)],
        baseline_services=[[], [4]],
        candidate_services=[[], []],
    )
    require(
        model_case(args, empty, "storage-only-empty")["accepted"],
        "unexpected storage debt",
    )
    protected = model_case(
        args, dict(empty, protected_services=[[], [4]]), "protected-empty"
    )
    require(
        not protected["accepted"] and "protected" in protected["reason"], str(protected)
    )
    refresh = copy.deepcopy(data)
    refresh.pop("baseline_services")
    refresh["sites"][0]["refresh"] = [0]
    row = model_case(args, refresh, "refresh")
    require(
        not row["accepted"]
        and row["certificate_runs"] == 0
        and "refresh" in row["reason"],
        str(row),
    )
    reordered = dict(
        classes=1,
        sites=[
            dict(pipe=4, facts=[2]),
            dict(pipe=5, facts=[2]),
            dict(pipe=1, facts=[1]),
        ],
        tree=[dict(kind="payload", site=i) for i in range(3)],
        candidate_services=[[], [4], [5, 4]],
    )
    order = model_case(args, reordered, "reordered-services")
    require(not order["accepted"] and "ordered subset" in order["reason"], str(order))
    return dict(mutual=rows, protected=protected, refresh=row, reordered=order)


def run_pair(args, name, text, counts=None):
    source = args.output / (name + ".pto")
    source.write_text(text)
    runs, checks, profiles = [], [], []
    for mode in ("--current-cut-output", "--current-cut-pruned"):
        output = args.output / f"{name}{mode}.pto"
        run = emit_mode(args, mode, source, output)
        require(run["status"] == "success", str(run) + Path(run["stderr"]).read_text())
        work = profile(run)
        kept = work.get("pruning", {}).get("kept")
        check = qualify(
            args,
            text,
            output.read_text(),
            output.with_suffix(".trace.pto"),
            native_serial_pipes=(0,),
            scalar_payload=Memory(counts) if counts is not None else None,
            retained_services=kept,
            return_relation=True,
        )
        require(work["protocol_certified"], "missing physical certificate")
        runs.append(run)
        checks.append(check)
        profiles.append(work)
    before, after = [set(map(tuple, c.pop("payload_event_relation"))) for c in checks]
    require(not after - before, "pruning added payload order")
    require(checks[1]["commands"] <= checks[0]["commands"], "pruning added commands")
    pruning = profiles[1]["pruning"]
    require(pruning["accepted"] and pruning["certificate_runs"] == 1, str(pruning))
    return dict(
        name=name,
        input_sha256=digest(source),
        runs=runs,
        checks=checks,
        work=profiles,
        added_pairs=len(after - before),
        removed_pairs=len(before - after),
    )


def delta_case(args):
    source = args.delta
    text = source.read_text()
    annotated = args.output / "delta-input.pto"
    annotated.write_text(annotate(text, "test.reaching_access"))
    run = invoke(
        [args.trace_tool, "--step1-json", annotated],
        args.output / "delta-input.json",
        35,
    )
    require(run["status"] == "success", str(run))
    data = json.loads(Path(run["stdout"]).read_text())["reaching_access"]
    programs, runs, work = [], [], []
    for mode in ("--current-cut-output", "--current-cut-pruned"):
        output = args.output / f"delta{mode}.pto"
        result = emit_mode(args, mode, source, output)
        require(result["status"] == "success", str(result))
        programs.append(output.read_text())
        runs.append(result)
        work.append(profile(result))
    require(work[1]["pruning"]["accepted"], str(work[1]))
    cases = check_delta(text, data, programs)
    return dict(
        input_sha256=digest(source),
        runs=runs,
        work=work,
        cases=cases,
        scope=(
            "actual emitted commands and independent original SSA/control; reservation-model effects, "
            "not native footprint recovery or numerical Delta execution"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", type=Path)
    parser.add_argument("--production-tool", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    negatives = joint_negatives(args)
    normalization = "%arg = source\n%bound = arith.constant 3 : index\n%test = arith.cmpi eq, %arg, %bound : index"
    require(
        normalized_payload(normalization)
        != normalized_payload(normalization.replace("constant 3", "constant 4")),
        "normalizer hid a used constant change",
    )
    require(
        normalized_payload(normalization)
        != normalized_payload(normalization.replace("cmpi eq", "cmpi ne")),
        "normalizer hid a predicate change",
    )
    require(
        normalized_payload(normalization)
        == normalized_payload("%dead = arith.constant 9 : index\n" + normalization),
        "unused map constant was not isolated",
    )
    cases = []
    for outer, inner in ((0, 2), (1, 0), (1, 1), (2, 3)):
        for guard in ("all", "none", "first"):
            cases.append(
                run_pair(
                    args,
                    f"nested-{outer}-{inner}-{guard}",
                    fixture(outer, inner, guard, stages=4),
                )
            )
    base = fixture(2, 3, "first", stages=4)
    add = next(s for s in base.splitlines() if "pto.tadd " in s)
    load = next(s for s in base.splitlines() if "pto.tload " in s)
    conditional = base.replace(add, f"scf.if %choice {{\n{add}\n}}", 1)
    cases.append(run_pair(args, "skipped-anchor", conditional))
    # One source pipe writes X, delivers X to V, then writes disjoint Y.
    # The later V reader of X must not acquire the unrelated Y completion.
    disjoint = fixture(2, 2, "all", stages=2)
    tile = next(
        line.split(" : ", 1)[1]
        for line in disjoint.splitlines()
        if "%tile = pto.alloc_tile" in line
    )
    disjoint = disjoint.replace(
        " %unit =",
        f" %ybase = arith.constant 64 : i64\n %other = pto.alloc_tile addr = %ybase : {tile}\n %unit =",
        1,
    )
    first_add = next(line for line in disjoint.splitlines() if "pto.tadd " in line)
    source_load = next(line for line in disjoint.splitlines() if "pto.tload " in line)
    other_load = source_load.replace("outs(%tile", "outs(%other")
    disjoint = disjoint.replace(first_add, first_add + "\n" + other_load, 1)
    reduced = run_pair(args, "disjoint-intervening-access", disjoint)
    require(reduced["removed_pairs"] > 0, "disjoint intervening work still ordered")
    require(
        len(reduced["checks"][1]["extra_pairs"])
        < len(reduced["checks"][0]["extra_pairs"]),
        "no excess-prerequisite improvement",
    )
    cases.append(reduced)
    conflict = base.replace(add, add + "\n" + load, 1)
    cases.append(run_pair(args, "intervening-writer", conflict))
    cases.append(
        run_pair(
            args,
            "conditional-writer",
            base.replace(add, add + f"\nscf.if %choice {{\n{load}\n}}", 1),
        )
    )
    multi = pipeline(2, 2).replace(
        "attributes {pto.kernel_kind",
        "attributes {test.current_cut_ids = array<i64: 3, 5>, pto.kernel_kind",
    )
    line = next(s for s in multi.splitlines() if "pto.tstore " in s)
    cases.append(run_pair(args, "third-pipe", multi.replace(line, line + "\n" + line)))
    cases.append(
        run_pair(args, "scalar-control", scalar_fixture(3, stages=3), [2, 0, 1])
    )
    scaling = []
    for stages in (1, 2, 4, 8, 16):
        for trips in (1, 256):
            text = fixture(2, trips, "all", stages=stages)
            source = args.output / f"scale-{stages}-{trips}.pto"
            source.write_text(text)
            run = emit_mode(
                args, "--current-cut-pruned", source, source.with_suffix(".out.pto")
            )
            require(run["status"] == "success", str(run))
            scaling.append(dict(stages=stages, trips=trips, run=run, work=profile(run)))
    for stages in (1, 2, 4, 8, 16):
        a, b = [r["work"] for r in scaling if r["stages"] == stages]
        require(
            a["pruning"]["proposed_deletions"] == stages - 1,
            "repeated acquire pruning regressed",
        )
        for field in ("out_facts", "ordered_command_sites", "selected_cross_services"):
            require(a[field] == b[field], "trip-count expansion")
        for field in (
            "proposal_nodes",
            "service_tests",
            "fact_transmissions",
            "proposed_deletions",
        ):
            require(a["pruning"][field] == b["pruning"][field], "pruning unfolds trips")
    (args.output / "receipt.json").write_text(
        json.dumps(
            dict(
                cases=cases,
                negatives=negatives,
                scaling=scaling,
                tool_sha256=digest(args.trace_tool),
                delta=delta_case(args) if args.delta else None,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "joint pruning: actual commands, guarded/reset coverage, no added order and causal reuse passed"
    )


if __name__ == "__main__":
    main()
