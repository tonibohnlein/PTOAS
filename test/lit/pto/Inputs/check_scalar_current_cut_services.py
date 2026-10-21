# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native scalar-prefix discharge; real GETVAL/SETVAL and independent memory values."""

import argparse
import json
from pathlib import Path
import re
from assess_whole_kernel_access import digest, invoke
from check_current_cut_services import qualify
from compact_vector_oracle import model, command_order, required
from check_compact_persistent_contract import trace
from assess_whole_kernel_access import annotate
from reaching_access_research import require

TILE = (
    "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, "
    "blayout=row_major, slayout=none_box, fractal=512, pad=0>"
)


class Memory:
    """Serial declared-layout reference. No compiler partition/generator inputs."""

    def __init__(self):
        self.values = {}
        self.observations = []

    def keys(self, value):
        space, base, _, _, ranges = value
        require(space == "vec", "scalar fixture requires known local vector storage")
        return [(space, base, lo) for lo, hi in ranges for lo in range(lo, hi, 4)]

    def __call__(self, op, reads, writes, env):
        if op == "pto.tgetval":
            tile = env[reads[0]]
            key = (tile[0], tile[1], tile[3] + 4 * env[reads[1]])
            require(key in self.values, "uninitialized scalar read")
            value = self.values[key]
            self.observations.append(value)
            return value
        if op == "pto.tsetval":
            tile = env[writes[0]]
            self.values[tile[0], tile[1], tile[3] + 4 * env[reads[0]]] = env[reads[1]]
        elif op == "pto.texpands":
            for key in self.keys(env[writes[0]]):
                self.values[key] = env[reads[0]]
        elif op == "pto.tadd":
            a, b = [self.keys(env[value]) for value in reads]
            values = [self.values[x] + self.values[y] for x, y in zip(a, b)]
            keys = self.keys(env[writes[0]])
            require(len(keys) == len(values), "vector shape mismatch")
            self.values.update(zip(keys, values))
        else:
            raise RuntimeError(f"unsupported independent scalar fixture operation {op}")
        return None


def fixture(outer, inner, positive=True, skip="first", stages=1):
    read = f"%read = pto.tgetval ins(%tile, %zero : {TILE}, index) outs : f32"
    read2 = f"%again = pto.tgetval ins(%tile, %zero : {TILE}, index) outs : f32"
    write = f"pto.tsetval ins(%zero, %one : index, f32) outs(%tile : {TILE})"
    add = f"pto.tadd ins(%tile, %tile : {TILE}, {TILE}) outs(%tile : {TILE})"
    predicate = {"all": "sle", "none": "slt", "first": "eq"}[skip]
    rhs = "%zero" if skip == "first" else "%j"
    attributes = "pto.kernel_kind = #pto.kernel_kind<vector>, test.current_cut_ids = array<i64: 3, 5>"
    return f"""module attributes {{pto.target_arch = "a3"}} {{
func.func @scalar_services() attributes {{{attributes}}} {{
 %zero = arith.constant 0 : index
 %step = arith.constant 1 : index
 %outer = arith.constant {outer} : index
 %inner = arith.constant {inner} : index
 %base = arith.constant 0 : i64
 %one = arith.constant 1.0 : f32
 %limit = arith.constant {0.0 if positive else 1000000.0} : f32
 %tile = pto.alloc_tile addr = %base : {TILE}
 pto.texpands ins(%one : f32) outs(%tile : {TILE})
 scf.for %i = %zero to %outer step %step {{
  scf.for %j = %zero to %inner step %step {{
   %producer = arith.cmpi {predicate}, %j, {rhs} : index
   scf.if %producer {{
    pto.texpands ins(%one : f32) outs(%tile : {TILE})
   }}
   {read}
   {read2}
   %active = arith.cmpf ogt, %read, %limit : f32
   scf.if %active {{
    {write}
    {chr(10).join([add] * stages)}
   }}
  }}
 }}
 return
}}
}}
"""


def scalar_only(outer, inner):
    text = fixture(outer, inner)
    text = text[: text.index(" pto.texpands")]
    return (
        text
        + f""" %twof = arith.constant 2.0 : f32
 pto.tsetval ins(%zero, %one : index, f32) outs(%tile : {TILE})
 scf.for %i = %zero to %outer step %step {{
  scf.for %j = %zero to %inner step %step {{
   %read = pto.tgetval ins(%tile, %zero : {TILE}, index) outs : f32
   pto.tsetval ins(%zero, %twof : index, f32) outs(%tile : {TILE})
   %again = pto.tgetval ins(%tile, %zero : {TILE}, index) outs : f32
  }}
 }}
 return
}}
}}
"""
    )


def sensitivity(args):
    # Use actual emitted traces with no bypass path through another service.
    base = fixture(1, 1)
    prefix = base[: base.index(" scf.for")]
    scalar_read = f" %read = pto.tgetval ins(%tile, %zero : {TILE}, index) outs : f32\n"
    add = f" pto.tadd ins(%tile, %tile : {TILE}, {TILE}) outs(%tile : {TILE})\n"
    results = []
    for name, body in (("vector-barrier", add), ("vector-to-scalar", scalar_read)):
        source = prefix + body + " return\n}\n}\n"
        _, path, run = lower(args, source, f"negative-base-{name}")
        require(run["status"] == "success", str(run))
        memory = Memory()
        items = model(source, scalar_payload=memory)
        observations = ", ".join(f"{value:.1f} : f32" for value in memory.observations)
        emitted = annotate(
            path.read_text(), f"test.trace_scalar_observations = [{observations}]"
        )
        original = trace(args, emitted, args.output / f"negative-{name}.trace.pto")
        if name == "vector-barrier":
            removed = [
                e
                for e in original["events"]
                if e["kind"] == "barrier" and e["gap"] == 1 and e["pipe"] == 1
            ]
            kept = [e for e in original["events"] if e not in removed]
            require(len(removed) == 1, "vector hazard lacks local protection")
        else:
            removed = [
                e
                for e in original["events"]
                if e["kind"] in ("set", "wait") and e["gap"] == 1
            ]
            kept = [e for e in original["events"] if e not in removed]
            require(len(removed) == 4, "cross-pipe service not isolated")
        rows, _ = command_order(
            {**original, "events": kept},
            native_serial_pipes=(0,),
            include_commands=True,
        )
        require(
            not (rows[1] >> 2 & 1),
            "removed service still covered by an alternative path",
        )
        wanted, _, _ = required(items)
        require(wanted[1] >> 2 & 1, "negative lacks a concrete required conflict")
        results.append(dict(name=name, removed_commands=len(removed), uncovered=[1, 2]))
    # Scalar-only access coverage must rely on the explicitly supplied native
    # edge, not on accidental vector handshakes.
    source = scalar_only(1, 1)
    _, path, run = lower(args, source, "scalar-only-native-negative")
    require(run["status"] == "success", str(run))
    memory = Memory()
    model(source, scalar_payload=memory)
    observed = ", ".join(f"{value:.1f} : f32" for value in memory.observations)
    actual = trace(
        args,
        annotate(path.read_text(), f"test.trace_scalar_observations = [{observed}]"),
        args.output / "scalar-only-native-negative.trace.pto",
    )
    weak, _ = command_order(actual, include_commands=True)
    strong, _ = command_order(actual, native_serial_pipes=(0,), include_commands=True)
    require(
        not (weak[1] >> 2 & 1) and strong[1] >> 2 & 1,
        "scalar edge not independently exercised",
    )
    results.append(
        dict(name="omitted-native-scalar-edge", uncovered_without_premise=[1, 2])
    )
    return results


def lower(args, text, name):
    source = args.output / f"{name}.pto"
    source.write_text(text)
    output = source.with_suffix(".physical.pto")
    run = invoke([args.trace_tool, "--current-cut-output", source], output, 35)
    return source, output, run


def run_case(args, text, name):
    source, output, run = lower(args, text, name)
    require(run["status"] == "success", str(run) + Path(run["stderr"]).read_text())
    emitted = output.read_text()
    require("pto.barrier <PIPE_S>" not in emitted, "fictitious scalar barrier emitted")
    work = [
        json.loads(line[len("current-cut-work ") :])
        for line in Path(run["stderr"]).read_text().splitlines()
        if line.startswith("current-cut-work ")
    ][0]
    checked = qualify(
        args,
        text,
        emitted,
        args.output / f"{name}.trace.pto",
        native_serial_pipes=(0,),
        scalar_payload=Memory(),
    )
    require(work["native_self_discharges"] > 0, "scalar self-obligation not exercised")
    return dict(
        name=name,
        input_sha256=digest(source),
        output_sha256=digest(output),
        run=run,
        work=work,
        **checked,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cli", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for outer, inner in ((0, 2), (1, 0), (1, 1), (1, 3), (2, 2), (3, 1)):
        for positive, skip in ((True, "all"), (True, "first"), (False, "none")):
            name = f"o{outer}-i{inner}-p{int(positive)}-{skip}"
            cases.append(run_case(args, fixture(outer, inner, positive, skip), name))
    for outer, inner in ((0, 2), (1, 1), (2, 3)):
        cases.append(
            run_case(args, scalar_only(outer, inner), f"scalar-only-{outer}-{inner}")
        )
    negative_controls = sensitivity(args)
    scaling = []
    for stages in (1, 2, 4, 8):
        for trips in (1, 256):
            source, output, run = lower(
                args, fixture(2, trips, stages=stages), f"scale-{stages}-{trips}"
            )
            require(run["status"] == "success", str(run))
            text = output.read_text()
            profile = next(
                line[len("current-cut-work ") :]
                for line in Path(run["stderr"]).read_text().splitlines()
                if line.startswith("current-cut-work ")
            )
            scaling.append(
                dict(
                    stages=stages,
                    trips=trips,
                    work=json.loads(profile),
                    flags=len(re.findall(r"pto\.(?:set_flag|wait_flag)\b", text)),
                    barriers=len(re.findall(r"pto\.barrier\b", text)),
                    bytes=len(text),
                )
            )
    for stages in (1, 2, 4, 8):
        a, b = [row for row in scaling if row["stages"] == stages]
        require(
            (
                a["flags"],
                a["barriers"],
                a["work"]["out_facts"],
                a["work"]["native_self_discharges"],
            )
            == (
                b["flags"],
                b["barriers"],
                b["work"]["out_facts"],
                b["work"]["native_self_discharges"],
            ),
            "trip count unfolded representation",
        )
    # A valid scalar opcode on an unsupported storage domain is not a native
    # prefix certificate just because its nominal pipe is S.
    bad = fixture(1, 1).replace(
        f"pto.tsetval ins(%zero, %one : index, f32) outs(%tile : {TILE})",
        "pto.tsetval ins(%zero, %one : index, f32) outs(%acc : !pto.tile_buf<acc, 1x8xf32>)",
    )
    bad = bad.replace(
        " %one =",
        " %acc = pto.alloc_tile addr = %base : !pto.tile_buf<acc, 1x8xf32>\n %one =",
        1,
    )
    _, _, negative = lower(args, bad, "unknown-scalar-domain")
    require(negative["status"] != "success", "unknown scalar prefix accepted")
    cpp = dict(status="not-requested")
    if args.cli:
        require(args.python is not None, "explicit workspace Python required")
        source = args.output / "o2-i2-p1-first.physical.pto"
        output = args.output / "scalar-services.cpp"
        cpp = invoke(
            [
                args.python,
                args.cli,
                "--pto-arch=a3",
                "--pto-level=level3",
                "--enable-insert-sync=false",
                "--enable-plan-memory=false",
                source,
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
    result = dict(
        tool_sha256=digest(args.trace_tool),
        native_scalar_pipe=0,
        contract="supplied A3 AIV synchronous vec GETVAL/SETVAL serialization; no cross-pipe exemption",
        cases=cases,
        scaling=scaling,
        negative_controls=negative_controls,
        unknown_scalar_negative=negative,
        cpp=cpp,
    )
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        "scalar native-prefix discharge: bounded memory/control, matching, order and reuse passed"
    )


if __name__ == "__main__":
    main()
