#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Retained finite-width address SSA must not hide real physical dependencies.

Invoke InsertSync directly in all three modes, without canonicalization. An
independent straight-line observer checks actual payload ordering. The native
read-only address mode challenges translated physical intervals and both alias
queries before either planner runs. This does not simulate device execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from compare_boundaries import Boundaries
from measure import attrs, children
from observations import project

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TILE = "!pto.tile_buf<vec, 16x16xf16>"
LEFT = "!pto.tile_buf<loc=left, dtype=f32, rows=1, cols=8, v_row=1, v_col=8, blayout=row_major, slayout=none_box, fractal=512, pad=0>"
MAT = LEFT.replace("loc=left", "loc=mat")


def program(name, arithmetic, expected_second=0, *, argument="", subview=False):
    address = "%address"
    allocations = f"%a = pto.alloc_tile addr = {address} : {TILE}"
    if subview:
        allocations = f"""%root = pto.alloc_tile addr = %zero : !pto.tile_buf<vec, 16x32xf16>
    %a = pto.subview %root[%c0, %offset] sizes [16, 16] :
      !pto.tile_buf<vec, 16x32xf16> -> {TILE}"""
    return f"""module attributes {{pto.target_arch = "a3"}} {{
  func.func @{name}(%src: !pto.ptr<f16>{argument})
      attributes {{pto.kernel_kind = #pto.kernel_kind<vector>}} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0 : i64
    %other = arith.constant {expected_second} : i64
    %output_address = arith.constant 1048576 : i64
    {arithmetic}
    {allocations}
    %b = pto.alloc_tile addr = %other : {TILE}
    %output = pto.alloc_tile addr = %output_address : {TILE}
    %view = pto.make_tensor_view %src, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<?x?xf16>
    %part = pto.partition_view %view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x16xf16>
    pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%a : {TILE})
    pto.tabs ins(%b : {TILE}) outs(%output : {TILE})
    return
  }}
}}"""


def left_program():
    multi = f"!pto.multi_tile_buf<{LEFT}, count=2>"
    return f"""module attributes {{pto.target_arch = "a3"}} {{
  func.func @explicit_left_stride() attributes {{pto.kernel_kind = #pto.kernel_kind<cube>}} {{
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : index
    %base512 = arith.constant 512 : i64
    %mat = pto.alloc_tile addr = %zero : {MAT}
    %slots = pto.alloc_multi_tile addr = %zero : {multi}
    %a = pto.multi_tile_get %slots[%one] : {multi} -> {LEFT}
    %b = pto.alloc_tile addr = %base512 : {LEFT}
    pto.tmov ins(%mat : {MAT}) outs(%a : {LEFT})
    pto.tmov ins(%mat : {MAT}) outs(%b : {LEFT})
    return
  }}
}}"""


def cases():
    trunc = "%large = arith.constant 65536 : i64\n    %narrow = arith.trunci %large : i64 to i16\n    %address = arith.extui %narrow : i16 to i64"
    signed = "%narrow = arith.constant 32736 : i16\n    %address = arith.extsi %narrow : i16 to i64"
    rows = [
        ("trunc_extui_alias", trunc, 0, 0, True, {}),
        ("trunc_extsi_alias", trunc.replace("arith.extui", "arith.extsi"), 0, 0, True, {}),
        ("signed_extension_alias", signed, 32736, 32736, True, {}),
        ("unsigned_extension_distinct", signed.replace("32736", "-32768").replace("arith.extsi", "arith.extui"), 32768, 0, False, {}),
        ("narrow_index_cast", "%large = arith.constant 65536 : index\n    %narrow = arith.index_cast %large : index to i16\n    %address = arith.extui %narrow : i16 to i64", 0, 0, True, {}),
        ("unsigned_index_extension", "%narrow = arith.constant -32768 : i16\n    %wide = arith.index_castui %narrow : i16 to index\n    %address = arith.index_cast %wide : index to i64", 32768, 32768, True, {}),
        ("wrapping_add", "%a16 = arith.constant 32767 : i16\n    %one16 = arith.constant 1 : i16\n    %sum = arith.addi %a16, %one16 : i16\n    %address = arith.extui %sum : i16 to i64", None, 32768, True, {}),
        ("wrapping_multiply", "%a16 = arith.constant -32768 : i16\n    %two16 = arith.constant 2 : i16\n    %product = arith.muli %a16, %two16 : i16\n    %address = arith.extui %product : i16 to i64", None, 0, True, {}),
        ("runtime_cross_root", "", None, 0, True, {"argument": ", %address: i64"}),
        ("runtime_cast_cross_root", "%narrow = arith.trunci %runtime : i64 to i16\n    %address = arith.extui %narrow : i16 to i64", None, 0, True, {"argument": ", %runtime: i64"}),
        ("subview_offset_cast", trunc + "\n    %offset = arith.index_cast %address : i64 to index", 0, 0, True, {"subview": True}),
        ("negative_address", "%small = arith.constant -32768 : i16\n    %address = arith.extsi %small : i16 to i64", "reject", 0, True, {}),
        ("address_end_overflow", "%address = arith.constant 9223372036854775776 : i64", "reject", 0, True, {}),
    ]
    result = []
    for name, expression, base, second, conflict, options in rows:
        source = program(name, expression, second, **options)
        result.append({"name": name, "source": source, "base": base, "second": second,
                       "conflict": conflict, "lane": "PIPE_MTE2", "target_lane": "PIPE_V", "target_write": False,
                       "subview": options.get("subview", False),
                       "retained_ops": [op for op in ("arith.trunci", "arith.extsi", "arith.extui", "arith.index_cast", "arith.index_castui",
                                                              "arith.addi", "arith.muli") if op in expression]})
    result.append({"name": "explicit_left_stride", "source": left_program(), "base": 512, "second": 512,
                   "conflict": True, "lane": "PIPE_MTE1", "target_lane": "PIPE_MTE1", "target_write": True,
                   "subview": False, "retained_ops": ["pto.multi_tile_get"]})
    # The same immutable address DAG is requested by many distinct descriptors.
    # Known and unsupported expressions must both use the bounded memo table.
    for seed in (result[0], next(row for row in result if row["name"] == "runtime_cast_cross_root")):
        case = dict(seed)
        case["name"] = seed["name"] + "_fanout"
        descriptors = "\n    ".join(f"%unused{i} = pto.alloc_tile addr = %address : {TILE}" for i in range(64))
        case["source"] = seed["source"].replace("@" + seed["name"], "@" + case["name"]).replace("    %b =", "    " + descriptors + "\n    %b =")
        case["fanout"] = True
        result.append(case)
    return result


def observe(source, emitted, case, ir, pto):
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        modules = [ir.Module.parse(text) for text in (source, emitted)]
        # This campaign intentionally uses the default may-alias GM contract.
        # The shared observer's historical fixtures use disjoint arguments;
        # normalize only this explicitly checked default metadata here.
        for module in modules:
            function = next(op for op in children(module.operation) if op.name == "func.func")
            if "pto.gm_alias" in function.attributes:
                assert str(function.attributes["pto.gm_alias"]) == '"may-alias"'
                del function.attributes["pto.gm_alias"]
        # We intentionally avoid constant folding or scalar replay: a truncated
        # constant must remain in the exact input/output SSA projection.
        projections = [project(module) for module in modules]
        for key in ("payload", "allocations", "views", "abi"):
            assert projections[0][key] == projections[1][key], (case["name"], key)
        function = next(op for op in children(modules[1].operation) if op.name == "func.func")
        result = Boundaries()
        for op in children(function):
            if op.name.startswith("scf."):
                raise ValueError("straight-line address test unexpectedly needs control interpretation")
            if op.name in ("pto.tload", "pto.tmov", "pto.tabs"):
                lane = case["lane"] if not result.payload else case["target_lane"]
                result.action(op.name, attrs(op), lane, [op.name, len(result.payload)])
            elif op.name in ("pto.set_flag", "pto.wait_flag", "pto.barrier"):
                result.observe(op, 0, None)
        assert len(result.before) == 2 and not result.tokens
        prefix = result.before[1]["completed"].get(case["lane"], -1)
        assert (prefix >= 0) == case["conflict"], (case["name"], "wrong physical conflict ordering", prefix)
        for name in case["retained_ops"]:
            assert name in emitted, (case["name"], "retained SSA disappeared", name)
        return {"target_completed_source": prefix, "payload_occurrences": len(result.before)}


def check_translation(data, case):
    assert data["mode"] == "physical-addresses"
    if case["base"] == "reject":
        assert data["admission"] == "reject" and not data["translated"], data
        return {"admission": data["admission"], "reason": data["reason"]}
    assert data["translated"] and data["admission"] in ("safe", "conservative"), data
    assert data["address_evaluation_repeated_visits"] == data["address_evaluation_visits"], data
    if case.get("fanout"):
        assert data["address_evaluation_requests"] == 67 and data["address_evaluation_visits"] <= 5, data
    relevant = [[a for a in data["accesses"] if a["phase"] == p and a["write"] == write]
                for p, write in ((0, True), (1, case["target_write"]))]
    assert all(relevant) and len({a["phase"] for a in data["accesses"]}) == 2, data
    a, b = relevant[0][0], relevant[1][0]
    assert a["root"] != b["root"], ("test lost cross-root coverage", case["name"], a, b)
    if case["base"] is None:
        assert not a["physical"] and a["unknown_range"], (case["name"], a)
    else:
        assert a["physical"] and not a["unknown_range"]
        assert case["base"] in list(map(int, a["base_addresses"])), (case["name"], a)
        if not case["subview"]:
            assert list(map(int, a["base_addresses"])) == [case["base"]], (case["name"], a)
            assert int(a["allocation_bytes"]) == (32 if case["lane"] == "PIPE_MTE1" else 512)
    assert int(b["base_addresses"][0]) == case["second"] and b["physical"], b
    answers = [q for q in data["aliases"] if q["source"] in {a["id"] for a in relevant[0]} and
               q["target"] in {b["id"] for b in relevant[1]}]
    assert answers
    assert any(q["legacy"] for q in answers) == case["conflict"], (case["name"], answers)
    assert any(q["logical"] for q in answers) == case["conflict"], (case["name"], answers)
    return {"admission": data["admission"], "effects": relevant, "alias_queries": answers}


def main():
    if not __debug__:
        raise RuntimeError("Physical address acceptance requires assertions")
    parser = argparse.ArgumentParser()
    for option in ("opt", "driver", "python-root", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.python_root.resolve()))
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    hashes = {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() for name in ("opt", "driver")}
    selected = cases()
    if args.case:
        assert set(args.case) <= {case["name"] for case in selected}
        selected = [case for case in selected if case["name"] in args.case]
    results, commands, evaluator_results = [], [], []

    def run(command, label):
        start = time.monotonic()
        process = subprocess.run([str(arg) for arg in command], capture_output=True, text=True, timeout=45, env=env)
        (args.output / (label + ".stdout")).write_text(process.stdout)
        (args.output / (label + ".stderr")).write_text(process.stderr)
        commands.append({"command": list(map(str, command)), "exit": process.returncode, "seconds": time.monotonic()-start})
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        assert process.returncode in (0, 1), (label, process.returncode, process.stderr[-4000:])
        return process

    # Pure value queries avoid conflating i1 semantics with physical-address
    # alignment. A conflicting index data layout must decline all index casts,
    # while retaining exact ordinary integer constants in the same function.
    value_cases = [
        ("boolean_extensions", """module {
  func.func @boolean_extensions() {
    %true = arith.constant true
    %signed = arith.extsi %true : i1 to i64
    %unsigned = arith.extui %true : i1 to i64
    %false = arith.constant false
    %signed_zero = arith.extsi %false : i1 to i64
    %unsigned_zero = arith.extui %false : i1 to i64
    return
  }
}""", [(-1, 1), (-1, 64), (1, 64), (0, 1), (0, 64), (0, 64)]),
        ("index32_declined", """module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>} {
  func.func @index32_declined() {
    %constant = arith.constant 65536 : i64
    %narrow = arith.index_cast %constant : i64 to index
    %wide = arith.index_cast %narrow : index to i64
    %negative = arith.constant -32768 : i16
    %unsigned = arith.index_castui %negative : i16 to index
    return
  }
}""", [(65536, 64), None, None, (-32768, 16), None]),
    ]
    for name, text, expected in value_cases:
        source = args.output / (name + ".pto")
        source.write_text(text)
        result = run([args.driver, source, "address-values"], name)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["mode"] == "address-values" and data["visits"] == data["repeated_visits"]
        assert len(data["values"]) == len(expected)
        for actual, value in zip(data["values"], expected, strict=True):
            assert actual["known"] == (value is not None), (name, actual, value)
            if value is not None:
                number, width = value
                assert (int(actual["signed"]), actual["bits"]) == (number, width), (name, actual, value)
                assert int(actual["unsigned"]) == number % (1 << width), (name, actual, value)
        evaluator_results.append({"case": name, **data})
    (args.output / "evaluator-results.json").write_text(json.dumps(evaluator_results, indent=2) + "\n")

    for case in selected:
        source = args.output / (case["name"] + ".pto")
        source.write_text(case["source"])
        if case["base"] != "reject" and case["conflict"]:
            # A successful compile is insufficient: independently establish
            # that this exact observer rejects an unordered payload pair.
            try:
                observe(case["source"], case["source"], case, ir, pto)
            except AssertionError as error:
                assert "wrong physical conflict ordering" in str(error), str(error)
            else:
                raise AssertionError((case["name"], "observer accepted unsynchronized conflicting effects"))
        probe = run([args.driver, source, "physical-addresses"], case["name"] + ".translation")
        assert probe.returncode == 0, probe.stderr
        translated = check_translation(json.loads(probe.stdout), case)
        row = {"case": case["name"], "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "translation": translated, "modes": []}
        for mode in ("existing", "logical", "logical-or-existing"):
            result = run([args.opt, "--mlir-disable-threading", "--pto-insert-sync=planner=" + mode, source],
                         case["name"] + "." + mode)
            expected_failure = case["base"] == "reject" or (case["base"] is None and mode == "logical")
            assert (result.returncode != 0) == expected_failure, (case["name"], mode, result.stderr[-3000:])
            if expected_failure:
                assert any(text in result.stderr for text in ("physical-address admission failed", "unproved write footprint", "unproved read footprint", "unqualified physical effects")), result.stderr
                status = {"explicit_failure": True, "diagnostic": result.stderr}
            else:
                status = observe(case["source"], result.stdout, case, ir, pto)
                if mode != "existing" and case["base"] is not None:
                    assert 'pto.insert_sync.logical_status = "applied"' in result.stdout
                    assert 'pto.insert_sync.producer = "logical"' in result.stdout
                if mode == "logical-or-existing" and case["base"] is None:
                    assert 'pto.insert_sync.producer = "existing-fallback"' in result.stdout, result.stdout
            row["modes"].append({"mode": mode, **status})
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(case["name"], "translated and all three direct planners checked", flush=True)
    assert all(hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() == expected for name, expected in hashes.items())
    (args.output / "summary.json").write_text(json.dumps({"status":"passed", "cases":results, "evaluator_cases":evaluator_results,
        "binary_sha256":hashes, "device":"not run", "gm_contract":"may-alias", "no_canonicalization":True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
