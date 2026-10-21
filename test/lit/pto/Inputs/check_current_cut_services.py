# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Real IR current-cut lowering: independent bytes, command causality and reuse."""

import argparse
import json
from pathlib import Path
import re
from assess_whole_kernel_access import annotate, digest, invoke
from check_compact_persistent_contract import trace
from check_expanded_endpoint_adapter import fixture as point_fixture
from compact_vector_oracle import command_order, model, required
from reaching_access_research import require
from check_source_pipe_services import compare
from check_periodic_demands import closure, native


def fixture(outer, inner, guard, stages=1):
    text = (
        point_fixture(outer, inner)
        .replace("64x8xf32", "2048x8xf32")
        .replace("constant 64 : index", "constant 2048 : index")
    )
    begin = text.index("   %target =")
    end = text.index("  }\n }\n return", begin)
    text = text[:begin] + text[end:]
    tile = re.search(r"%tile = pto.alloc_tile[^\n]* : (.+)", text)[1]
    initialize = f" %unit = arith.constant 1.0 : f32\n pto.texpands ins(%unit : f32) outs(%tile : {tile})\n"
    text = text.replace(" scf.for %i", initialize + " scf.for %i", 1)
    start = text.index("   pto.tload")
    after = text.index("   pto.tadd", start)
    load = text[start:after]
    predicate = {"all": "sle", "none": "slt", "first": "eq"}[guard]
    # sle(j,j) and slt(j,j) give exact taken/skipped controls. No history state.
    right = "%one" if guard == "first" else "%j"
    guarded = f"   %choice = arith.cmpi {predicate}, %j, {right} : index\n   scf.if %choice {{\n{load}   }}\n"
    text = text[:start] + guarded + text[after:]
    if stages > 1:
        start = text.index("   pto.tadd")
        end = text.index("\n", start)
        text = text[:start] + "\n".join([text[start:end]] * stages) + text[end:]
    # The emitter's role ordinals map into this explicitly eligible pool.
    return text.replace(
        "attributes {pto.kernel_kind",
        "attributes {test.current_cut_ids = array<i64: 3, 5>, pto.kernel_kind",
        1,
    )


def two_class_fixture(guard):
    text = fixture(2, 3, guard)
    tile = re.search(r"%tile = pto.alloc_tile[^\n]* : (.+)", text)[1]
    text = text.replace(
        " %unit =",
        f" %otherbase = arith.constant 64 : i64\n %other = pto.alloc_tile addr = %otherbase : {tile}\n %unit =",
        1,
    )
    text = text.replace(
        " scf.for %i",
        f" pto.texpands ins(%unit : f32) outs(%other : {tile})\n scf.for %i",
        1,
    )
    start = text.index("   pto.tadd")
    end = text.index("\n", start)
    independent = text[start:end].replace("%tile", "%other")
    return text[:end] + "\n" + independent + text[end:]


def qualify(
    args, source, target, path, *, native_serial_pipes=(), scalar_payload=None,
    retained_services=None, return_relation=False
):
    source_path = path.with_suffix(".input.pto")
    source_path.write_text(annotate(source, "test.reaching_access"))
    access_run = invoke(
        [args.trace_tool, "--step1-json", source_path],
        path.with_suffix(".input.json"),
        35,
    )
    require(access_run["status"] == "success", str(access_run))
    data = json.loads(Path(access_run["stdout"]).read_text())["reaching_access"]
    compare(data, data["projected"])
    planned = {}

    def collect(tree):
        for item in tree:
            if item["kind"] == "service":
                planned[item["site"]] = item["commands"]
            else:
                for key in ("body",) if item["kind"] == "loop" else ("then", "else"):
                    collect(item[key])

    collect(data["projected"]["physical_plan"])
    selected = data["projected"]["possible"] if retained_services is None else retained_services
    if retained_services is not None:
        for site, commands in planned.items():
            kept = []
            cursor = 0
            while cursor < len(commands):
                command = commands[cursor]
                if command["op"] == "B":
                    kept.append(command)
                    cursor += 1
                    continue
                source_pipe = int(command["source"].split(":")[1])
                if source_pipe in selected[site]:
                    kept.extend(commands[cursor:cursor + 4])
                cursor += 4
            planned[site] = kept
    role_ids = {
        f"{r['role']}:{r['source']}:{r['target']}": (3, 5)[r["id"]]
        for r in data["projected"]["role_ids"]
    }

    expected = model(source, scalar_payload=scalar_payload)
    if scalar_payload is not None:
        observations = ", ".join(getattr(scalar_payload, "trace_observations", [
            f"{float(value):.1f} : f32" for value in scalar_payload.observations
        ]))
        target = annotate(target, f"test.trace_scalar_observations = [{observations}]")
    executed = trace(args, target, path)
    payload = [e for e in executed["events"] if e["kind"] == "payload"]
    require(
        [(e["type"], e["coordinates"], e["pipe"]) for e in payload]
        == [(e["type"], e["coordinates"], e["pipe"]) for e in expected],
        "payload/control changed",
    )
    wanted, _, scalar = required(expected)
    corrected = native([e["pipe"] for e in payload], native_serial_pipes)
    corrected.update(
        (a, b)
        for a, row in enumerate(wanted)
        for b in range(a + 1, len(wanted))
        if row >> b & 1
    )
    wanted, _ = closure(2 * len(payload), corrected)
    command_controls = [
        (source, index)
        for index, event in enumerate(
            e for e in executed["events"] if e["kind"] != "payload"
        )
        if event["gap"] < len(expected)
        for source in expected[event["gap"]].get("control", ())
    ]
    rows, reuse = command_order(
        executed,
        scalar,
        include_commands=True,
        native_serial_pipes=native_serial_pipes,
        command_prerequisites=command_controls,
    )
    require(
        all(not (row >> i & 1) for i, row in enumerate(rows)), "command graph cycle"
    )
    require(
        all(
            ((a & ~(1 << i)) & b) == (a & ~(1 << i))
            for i, (a, b) in enumerate(zip(wanted, rows))
        ),
        "concrete prerequisite lost",
    )
    prefix_edges = native([e["pipe"] for e in payload], native_serial_pipes)
    prefix_edges.update((2 * a + 1, 2 * b) for a, b in scalar)
    for b, target_event in enumerate(payload):
        sources = selected[target_event["type"]]
        prefix_edges.update(
            (2 * a + 1, 2 * b) for a in range(b) if payload[a]["pipe"] in sources
        )
    prefix, _ = closure(2 * len(payload), prefix_edges)
    count = 2 * len(payload)
    prefix_pairs = {
        (a, b)
        for a in range(count)
        for b in range(count)
        if a != b and prefix[a] >> b & 1
    }
    physical_pairs = {
        (a, b)
        for a in range(count)
        for b in range(count)
        if a != b and rows[a] >> b & 1
    }
    require(not (prefix_pairs - physical_pairs), "acknowledgments lost prefix order")
    sets = [e for e in executed["events"] if e["kind"] == "set"]
    if sets:
        require(reuse["status"] == "passed", str(reuse))
        require(
            all(e["physical_id"] in (3, 5) for e in sets), "eligible pool mapping lost"
        )
    expected_commands = []
    for gap, event in enumerate(payload):
        for command in planned[event["type"]]:
            if command["op"] == "B":
                if command["pipe"] in native_serial_pipes:
                    continue
                expected_commands.append(
                    ("barrier", gap, command["pipe"], None, None, None)
                )
            else:
                _, p, q = command["source"].split(":")
                expected_commands.append(
                    (
                        "set" if command["op"] == "S" else "wait",
                        gap,
                        command["pipe"],
                        int(p),
                        int(q),
                        role_ids[command["source"]],
                    )
                )
    expected_commands.append(("barrier", len(payload), 6, None, None, None))
    actual_commands = [
        (
            e["kind"],
            e["gap"],
            e["pipe"],
            e.get("source_pipe"),
            e.get("target_pipe"),
            e.get("physical_id"),
        )
        for e in executed["events"]
        if e["kind"] != "payload"
    ]
    require(
        expected_commands == actual_commands,
        "native code differs from once-generated static plan",
    )
    # Every physical SET/WAIT matches within one static service; the trace's
    # direction/ID FIFO pairing and the causal check are independent of GEN.
    commands = [
        e for e in executed["events"] if e["kind"] not in ("payload", "barrier")
    ]
    require(len(commands) % 4 == 0, "incomplete service")
    for offset in range(0, len(commands), 4):
        fset, fwait, rset, rwait = commands[offset : offset + 4]
        require(
            [e["kind"] for e in (fset, fwait, rset, rwait)]
            == ["set", "wait", "set", "wait"],
            "service order",
        )
        require(
            fset["gap"] == fwait["gap"] == rset["gap"] == rwait["gap"],
            "shared cut lost",
        )
        require(
            fset["source_pipe"] == rset["target_pipe"]
            and fset["target_pipe"] == rset["source_pipe"],
            "ack direction",
        )
    native_comparison = {}
    if native_serial_pipes:
        # Oracle-only proposed barriers: never emit PIPE_S barriers to the target.
        # Compare both programs under the same corrected native/control contract.
        abstract_events = []
        gap = 0
        for event in executed["events"]:
            if event["kind"] == "payload":
                if any(
                    c["op"] == "B" and c["pipe"] in native_serial_pipes
                    for c in planned[event["type"]]
                ):
                    abstract_events.append(
                        dict(kind="barrier", gap=gap, pipe=event["pipe"])
                    )
                gap += 1
            abstract_events.append(event)
        abstract_controls = [
            (source, index)
            for index, event in enumerate(
                e for e in abstract_events if e["kind"] != "payload"
            )
            if event["gap"] < len(expected)
            for source in expected[event["gap"]].get("control", ())
        ]
        abstract_rows, _ = command_order(
            {**executed, "events": abstract_events},
            scalar,
            include_commands=True,
            native_serial_pipes=native_serial_pipes,
            command_prerequisites=abstract_controls,
        )
        abstract_pairs = {
            (a, b)
            for a in range(count)
            for b in range(count)
            if a != b and abstract_rows[a] >> b & 1
        }
        native_comparison = dict(
            native_discharge_added_event_pairs=sorted(physical_pairs - abstract_pairs),
            native_discharge_removed_event_pairs=sorted(
                abstract_pairs - physical_pairs
            ),
        )
    extra = [
        (a, b)
        for a in range(len(payload))
        for b in range(len(payload))
        if rows[2 * a + 1] >> (2 * b) & 1 and not wanted[2 * a + 1] >> (2 * b) & 1
    ]
    return dict(
        **native_comparison,
        payloads=len(payload),
        commands=len(executed["events"]) - len(payload),
        matching="closed",
        supplied_command_control_edges=len(command_controls),
        causal_reuse=reuse,
        extra_pairs=extra,
        physical_added_event_pairs=sorted(physical_pairs - prefix_pairs),
        physical_removed_event_pairs=[],
        **({"payload_event_relation": sorted(physical_pairs)} if return_relation else {}),
        ids=sorted(
            {(e["source_pipe"], e["target_pipe"], e["physical_id"]) for e in sets}
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results, scaling = [], []
    fixtures = [
        (f"o{outer}-i{inner}-{guard}", fixture(outer, inner, guard))
        for outer, inner in ((0, 2), (1, 0), (1, 1), (1, 3), (2, 1), (2, 3), (3, 2))
        for guard in ("all", "none", "first")
    ]
    fixtures += [
        (f"two-class-{guard}", two_class_fixture(guard)) for guard in ("all", "first")
    ]
    fixtures.append(
        (
            "empty-invocation",
            'module attributes {pto.target_arch = "a3"} {\n'
            "func.func @empty() attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {\n"
            " return\n}\n}\n",
        )
    )
    for name, source in fixtures:
        path = args.output / f"{name}.pto"
        path.write_text(source)
        emitted = args.output / f"{name}.physical.pto"
        run = invoke([args.trace_tool, "--current-cut-output", path], emitted, 35)
        require(run["status"] == "success", str(run) + Path(run["stderr"]).read_text())
        target = emitted.read_text()
        profile_lines = [
            line[len("current-cut-work ") :]
            for line in Path(run["stderr"]).read_text().splitlines()
            if line.startswith("current-cut-work ")
        ]
        require(len(profile_lines) == 1, "missing exclusive construction work")
        run["work"] = json.loads(profile_lines[0])
        checked = qualify(args, source, target, args.output / f"{name}.trace.pto")
        results.append(
            dict(
                name=name,
                input_sha256=digest(path),
                output_sha256=digest(emitted),
                run=run,
                **checked,
            )
        )
    for stages in (1, 2, 4, 8, 16):
        for trips in (1, 256):
            source = fixture(1, trips, "first", stages)
            path = args.output / f"scale-{stages}-{trips}.pto"
            path.write_text(source)
            emitted = path.with_suffix(".physical.pto")
            run = invoke([args.trace_tool, "--current-cut-output", path], emitted, 35)
            require(run["status"] == "success", str(run))
            text = emitted.read_text()
            # Strip quoted attribute values before lexical operation accounting.
            operations = re.sub(r'"[^"\n]*"', '""', text)
            scaling.append(
                dict(
                    stages=stages,
                    trips=trips,
                    seconds=run["seconds"],
                    original_arithmetic=len(
                        re.findall(r"arith\.\w+", re.sub(r'"[^"\n]*"', '""', source))
                    ),
                    added_id_arithmetic=len(re.findall(r"arith\.\w+", operations))
                    - len(
                        re.findall(r"arith\.\w+", re.sub(r'"[^"\n]*"', '""', source))
                    ),
                    introduced_guard_nodes=0,
                    retained_control_tests=len(
                        re.findall(r"arith\.cmpi\b", operations)
                    ),
                    flags=len(re.findall(r"pto\.(?:set_flag|wait_flag)\b", operations)),
                    barriers=len(re.findall(r"pto\.barrier\b", operations)),
                    arithmetic=len(re.findall(r"arith\.\w+", operations)),
                    bytes=len(text),
                )
            )
    for stages in (1, 2, 4, 8, 16):
        a, b = [row for row in scaling if row["stages"] == stages]
        require(
            (a["flags"], a["barriers"], a["arithmetic"])
            == (b["flags"], b["barriers"], b["arithmetic"]),
            "trip counts enlarged static representation",
        )
    negative = args.output / "resident-commands.pto"
    source = fixture(1, 2, "all").replace(" return", " pto.barrier <PIPE_ALL>\n return")
    negative.write_text(source)
    rejected = invoke(
        [args.trace_tool, "--current-cut-output", negative],
        args.output / "rejected.pto",
        35,
    )
    require(
        rejected["status"] != "success", "resident command boundary silently flattened"
    )
    negatives = []
    for name, source in (
        ("one-role", fixture(1, 2, "all").replace("array<i64: 3, 5>", "array<i64: 3>")),
        (
            "ineligible",
            fixture(1, 2, "all").replace("array<i64: 3, 5>", "array<i64: 6>"),
        ),
        (
            "unsupported-target",
            fixture(1, 2, "all").replace(
                'pto.target_arch = "a3"', 'pto.target_arch = "a5"'
            ),
        ),
    ):
        path = args.output / f"negative-{name}.pto"
        path.write_text(source)
        negative_run = invoke(
            [args.trace_tool, "--current-cut-output", path],
            path.with_suffix(".output.pto"),
            35,
        )
        require(negative_run["status"] != "success", f"unsupported {name} accepted")
        require(
            not Path(negative_run["stdout"]).read_text().strip(),
            "partial lowering escaped transaction",
        )
        negatives.append(dict(name=name, run=negative_run))
    cpp = {"status": "not-requested"}
    if args.cli:
        require(args.python is not None, "C++ request needs explicit workspace Python")
        output = args.output / "current-cut.cpp"
        cpp = invoke(
            [
                args.python,
                args.cli,
                "--pto-arch=a3",
                "--pto-level=level3",
                "--enable-insert-sync=false",
                "--enable-plan-memory=false",
                args.output / "o2-i3-first.physical.pto",
                "-o",
                output,
            ],
            args.output / "cpp.stdout",
            35,
        )
        if cpp["status"] == "success":
            cpp.update(
                output=str(output), sha256=digest(output), bytes=output.stat().st_size
            )
    report = dict(
        tool_sha256=digest(args.trace_tool),
        cases=results,
        scaling=scaling,
        resident_command_negative=rejected,
        qualification_negatives=negatives,
        cpp=cpp,
        scope="host model; isolated command-free A3 AIV invocation",
    )
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "qualified current-cut IR: coverage, closed matching, causal reuse and compact trips passed"
    )


if __name__ == "__main__":
    main()
