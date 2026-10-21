# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Contract-driven admission: real pointer loads, memory-driven loops and mixed S work."""

import argparse
import json
from pathlib import Path
from assess_whole_kernel_access import annotate, digest, invoke
from check_current_cut_services import qualify
from check_scalar_current_cut_services import Memory as ScalarMemory, TILE, lower
from reaching_access_research import require
from compact_vector_oracle import model, command_order
from check_compact_persistent_contract import trace


class Memory(ScalarMemory):
    def __init__(self, counts, floating=False):
        super().__init__()
        self.initial_env = {"%bounds": ("gm", "%bounds", None, 0, None)}
        self.trace_observations = []
        self.pointer_type = "f32" if floating else "i32"
        for i, value in enumerate(counts):
            self.values["gm", "%bounds", 4 * i] = float(value) if floating else value

    def keys(self, value):
        space, base, _, _, ranges = value
        require(space in ("gm", "vec"), "unsupported declared fixture storage")
        return [
            (space, base, offset) for lo, hi in ranges for offset in range(lo, hi, 4)
        ]

    def __call__(self, op, reads, writes, env):
        if op == "pto.load":
            pointer = env[reads[0]]
            key = (pointer[0], pointer[1], pointer[3] + 4 * env[reads[1]])
            require(key in self.values, "uninitialized pointer load")
            value = self.values[key]
            self.observations.append(value)
            self.trace_observations.append(f"{value} : {self.pointer_type}")
            return value
        if op == "pto.tsub":
            left, right = [self.keys(env[value]) for value in reads]
            target = self.keys(env[writes[0]])
            values = [self.values[a] - self.values[b] for a, b in zip(left, right)]
            require(len(target) == len(values), "subtraction shape mismatch")
            self.values.update(zip(target, values))
            return None
        if op == "pto.tstore":
            source, target = self.keys(env[reads[0]]), self.keys(env[writes[0]])
            require(len(source) == len(target), "transfer shape mismatch")
            values = [self.values[key] for key in source]
            self.values.update(zip(target, values))
            return None
        result = super().__call__(op, reads, writes, env)
        if op == "pto.tgetval":
            self.trace_observations.append(f"{float(result):.1f} : f32")
        return result


def fixture(outer=3, stages=1, transfer=False, skipped=False):
    kind = "f32" if transfer else "i32"
    setup = ""
    producer = ""
    if transfer:
        setup = (
            " %eight = arith.constant 8 : index\n"
            " %view = pto.make_tensor_view %bounds, shape = [%step, %eight], "
            "strides = [%eight, %step] : !pto.tensor_view<1x8xf32>\n"
            " %part = pto.partition_view %view, offsets = [%zero, %zero], "
            "sizes = [%step, %eight] : !pto.tensor_view<1x8xf32> -> !pto.partition_tensor_view<1x8xf32>\n"
        )
        producer = f"""  %publish = arith.cmpi {"slt" if skipped else "sle"}, %i, %i : index
  scf.if %publish {{
   pto.tstore ins(%tile : {TILE}) outs(%part : !pto.partition_tensor_view<1x8xf32>)
  }}
"""
    load_index = "%zero" if transfer else "%i"
    cast = "  %integer = arith.fptosi %count : f32 to i32\n" if transfer else ""
    limit = "%integer" if transfer else "%count"
    adds = "\n".join(
        [f"    pto.tadd ins(%tile, %tile : {TILE}, {TILE}) outs(%tile : {TILE})"]
        * stages
    )
    attrs = "pto.kernel_kind = #pto.kernel_kind<vector>, test.current_cut_ids = array<i64: 3, 5>"
    return f"""module attributes {{pto.target_arch = "a3"}} {{
func.func @load_services(%bounds: !pto.ptr<{kind}, gm>) attributes {{{attrs}}} {{
 %zero = arith.constant 0 : index
 %step = arith.constant 1 : index
 %outer = arith.constant {outer} : index
 %base = arith.constant 0 : i64
 %unit = arith.constant 1.0 : f32
 %tile = pto.alloc_tile addr = %base : {TILE}
 pto.texpands ins(%unit : f32) outs(%tile : {TILE})
{setup} scf.for %i = %zero to %outer step %step {{
{producer}  %count = pto.load %bounds[{load_index}] : !pto.ptr<{kind}, gm> -> {kind}
{cast}  %limit = arith.index_cast {limit} : i32 to index
  scf.for %j = %zero to %limit step %step {{
   %read = pto.tgetval ins(%tile, %zero : {TILE}, index) outs : f32
   pto.tsetval ins(%zero, %read : index, f32) outs(%tile : {TILE})
   %active = arith.cmpi eq, %j, %zero : index
   scf.if %active {{
{adds}
   }}
  }}
 }}
 return
}}
}}
"""


def access_coverage(args, name, text, counts, transfer):
    reference = model(text, scalar_payload=Memory(counts, floating=transfer))
    data = json.loads((args.output / f"{name}.trace.input.json").read_text())[
        "reaching_access"
    ]
    checks = 0
    for occurrence in reference:
        summary = data["sites"][occurrence["type"]]["effects"]
        for effect in occurrence["effects"]:
            candidates = [
                entry
                for entry in summary
                if entry["space"] == {"gm": 1, "vec": 6}[effect["space"]]
                and entry["write"] == effect["write"]
            ]
            require(candidates, "declared access mode/domain omitted")
            for lo, hi in effect["ranges"]:
                require(
                    any(
                        not entry["known"]
                        or any(
                            base <= lo and hi <= base + entry["bytes"]
                            for base in entry["bases"]
                        )
                        for entry in candidates
                    ),
                    "bounded access outside declared reservation enclosure",
                )
                checks += 1
    return dict(
        intervals=checks,
        gm_contract="whole known GM space; unknown absolute bases are not disjoint",
        scope="declared source layout/access model; native footprint unverified",
    )


def missing_vector_scalar_service(args):
    text = fixture(1)
    source, output, run = lower(args, text, "vector-negative-base")
    require(run["status"] == "success", str(run))
    memory = Memory([1])
    items = model(text, scalar_payload=memory)
    executed = trace(
        args,
        annotate(
            output.read_text(),
            "test.trace_scalar_observations = ["
            + ", ".join(memory.trace_observations)
            + "]",
        ),
        args.output / "vector-negative.trace.pto",
    )
    load = next(i for i, item in enumerate(items) if item["operation"] == "pto.tgetval")
    producer = next(
        i for i, item in enumerate(items) if item["operation"] == "pto.texpands"
    )
    removed = [
        event
        for event in executed["events"]
        if event["kind"] in ("set", "wait")
        and event["gap"] == load
        and {event["source_pipe"], event["target_pipe"]} == {0, 1}
    ]
    require(len(removed) == 4, "missing isolated vector-to-scalar service")
    kept = [event for event in executed["events"] if event not in removed]
    rows, _ = command_order(
        {**executed, "events": kept}, native_serial_pipes=(0,), include_commands=True
    )
    require(
        not (rows[2 * producer + 1] >> (2 * load) & 1),
        "deleted service has an alternative cover",
    )
    return dict(
        removed_commands=len(removed),
        uncovered=[producer, load],
        source_sha256=digest(source),
    )


def run_case(args, name, text, counts, transfer=False):
    source, output, run = lower(args, text, name)
    require(run["status"] == "success", str(run) + Path(run["stderr"]).read_text())
    memory = Memory(counts, floating=transfer)
    checked = qualify(
        args,
        text,
        output.read_text(),
        args.output / f"{name}.trace.pto",
        native_serial_pipes=(0,),
        scalar_payload=memory,
    )
    require("pto.barrier <PIPE_S>" not in output.read_text(), "scalar barrier emitted")
    profile = next(
        line[len("current-cut-work ") :]
        for line in Path(run["stderr"]).read_text().splitlines()
        if line.startswith("current-cut-work ")
    )
    return dict(
        name=name,
        counts=counts,
        input_sha256=digest(source),
        output_sha256=digest(output),
        run=run,
        work=json.loads(profile),
        scalar_observations=memory.trace_observations,
        access_coverage=access_coverage(args, name, text, counts, transfer),
        **checked,
    )


def negatives(args):
    good = fixture(1)
    cases = {
        "vector-load": good.replace("!pto.ptr<i32, gm>", "!pto.ptr<vector<2xi32>, gm>")
        .replace("-> i32", "-> vector<2xi32>")
        .replace(
            "%limit = arith.index_cast %count : i32 to index",
            "%limit = arith.constant 1 : index",
        ),
        "unknown-effects": good.replace(
            " %unit =",
            " %unknown = pto.alloc_multi_tile addr = %base : !pto.multi_tile_buf<vec, 1x8xf32, count=2>\n %unit =",
            1,
        ),
        "loop-carried": good.replace(
            "scf.for %j = %zero to %limit step %step {",
            "scf.for %j = %zero to %limit step %step iter_args(%carry = %unit) -> (f32) {",
        ).replace("   }\n  }", "   }\n   scf.yield %carry : f32\n  }"),
    }
    for skipped in (False, True):
        cases[f"gm-write-visibility-{skipped}"] = fixture(
            1, transfer=True, skipped=skipped
        )
    results = []
    for name, text in cases.items():
        _, _, run = lower(args, text, f"negative-{name}")
        require(run["status"] != "success", f"unsupported {name} accepted")
        diagnostic = Path(run["stderr"]).read_text()
        require(
            "current-cut:" in diagnostic,
            f"negative {name} failed before contract admission: {diagnostic}",
        )
        expected = (
            "gm-read-visibility"
            if name.startswith("gm-write-visibility")
            else {
                "vector-load": "native-prefix",
                "unknown-effects": "effects-or-phase",
                "loop-carried": "structured-control",
            }[name]
        )
        require(
            expected in diagnostic, f"wrong contract refusal for {name}: {diagnostic}"
        )
        results.append(dict(name=name, run=run, diagnostic=diagnostic))
    return results


def alias_contracts(args):
    """Visibility qualification is independent of directional event capacity."""
    readonly = fixture(1)
    distinct = (
        fixture(1, transfer=True)
        .replace(
            "(%bounds: !pto.ptr<f32, gm>)",
            "(%bounds: !pto.ptr<f32, gm>, %output: !pto.ptr<f32, gm>)",
        )
        .replace("pto.make_tensor_view %bounds", "pto.make_tensor_view %output")
    )
    rows = []
    for name, text in (("readonly", readonly), ("distinct-writer", distinct)):
        source = args.output / f"alias-{name}.pto"
        source.write_text(text)
        for policy in ("may-alias", "may-not-alias"):
            run = invoke(
                [
                    args.trace_tool,
                    f"--gm-alias={policy}",
                    "--current-cut-contracts",
                    source,
                ],
                args.output / f"alias-{name}-{policy}.json",
                35,
            )
            require(run["status"] == "success", str(run))
            inventory = json.loads(Path(run["stdout"]).read_text())[0]
            issues = inventory["issues"]
            expected = name == "distinct-writer" and policy == "may-alias"
            require(
                bool(issues) == expected, f"alias qualification differs: {inventory}"
            )
            require(
                all(i["contract"] == "gm-read-visibility" for i in issues), str(issues)
            )
            refusal = None
            if expected:
                refusal = invoke(
                    [
                        args.trace_tool,
                        f"--gm-alias={policy}",
                        "--current-cut-output",
                        source,
                    ],
                    args.output / f"alias-{name}-{policy}-output.pto",
                    35,
                )
                require(refusal["status"] == "failure", str(refusal))
                diagnostic = Path(refusal["stderr"]).read_text()
                require(
                    "gm-read-visibility" in diagnostic
                    and "eligible role reservation" not in diagnostic,
                    diagnostic,
                )
            rows.append(
                dict(
                    name=name,
                    policy=policy,
                    inventory=inventory,
                    run=run,
                    emission_refusal=refusal,
                )
            )
    empty = readonly.replace("array<i64: 3, 5>", "array<i64>")
    source = args.output / "alias-empty-pool.pto"
    source.write_text(empty)
    contracts = invoke(
        [args.trace_tool, "--current-cut-contracts", source],
        args.output / "alias-empty-pool.json",
        35,
    )
    require(contracts["status"] == "success", str(contracts))
    require(
        not json.loads(Path(contracts["stdout"]).read_text())[0]["issues"],
        "capacity polluted contracts",
    )
    output = invoke(
        [args.trace_tool, "--current-cut-output", source],
        args.output / "alias-empty-pool-output.pto",
        35,
    )
    require(output["status"] == "failure", str(output))
    diagnostic = Path(output["stderr"]).read_text()
    require("eligible role reservation unavailable" in diagnostic, diagnostic)
    require("gm-read-visibility" not in diagnostic, diagnostic)
    rows.append(dict(name="empty-pool", contracts=contracts, output=output))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for outer, values in (
        (0, []),
        (1, [-1]),
        (1, [0]),
        (1, [1]),
        (1, [3]),
        (2, [0, 2]),
        (3, [2, 0, 1]),
    ):
        cases.append(
            run_case(
                args,
                f"loads-{outer}-{'-'.join(map(str, values))}",
                fixture(outer),
                values,
            )
        )
    # This ordinary vector primitive was absent from the removed name gate.
    cases.append(
        run_case(
            args,
            "generic-subtraction",
            fixture(2).replace("pto.tadd", "pto.tsub"),
            [2, 1],
        )
    )
    scaling = []
    for stages in (1, 2, 4, 8):
        for trips in (1, 256):
            # Dynamic counts change only independent memory, never source IR.
            text = fixture(2, stages=stages)
            _, output, run = lower(args, text, f"scale-{stages}-{trips}")
            require(run["status"] == "success", str(run))
            work = json.loads(
                next(
                    x[len("current-cut-work ") :]
                    for x in Path(run["stderr"]).read_text().splitlines()
                    if x.startswith("current-cut-work ")
                )
            )
            scaling.append(
                dict(
                    stages=stages,
                    runtime_trip_sample=trips,
                    work=work,
                    output_sha256=digest(output),
                    bytes=output.stat().st_size,
                )
            )
    for stages in (1, 2, 4, 8):
        a, b = [row for row in scaling if row["stages"] == stages]
        require(
            a["output_sha256"] == b["output_sha256"],
            "trip count entered static construction",
        )
    negative = negatives(args)
    vector_negative = missing_vector_scalar_service(args)
    cpp = dict(status="not-requested")
    if args.cli:
        require(args.python is not None, "workspace Python required")
        cpp = invoke(
            [
                args.python,
                args.cli,
                "--pto-arch=a3",
                "--pto-level=level3",
                "--enable-insert-sync=false",
                "--enable-plan-memory=false",
                args.output / "loads-3-2-0-1.physical.pto",
                "-o",
                args.output / "load-services.cpp",
            ],
            args.output / "cpp.stdout",
            35,
        )
        require(cpp["status"] == "success", str(cpp))
    (args.output / "receipt.json").write_text(
        json.dumps(
            dict(
                cases=cases,
                scaling=scaling,
                negatives=negative,
                alias_contracts=alias_contracts(args),
                vector_service_negative=vector_negative,
                cpp=cpp,
                tool_sha256=digest(args.trace_tool),
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "contract admission: pointer-load control, mixed scalar ordering and causal services passed"
    )


if __name__ == "__main__":
    main()
