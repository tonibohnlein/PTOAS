#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare optional native scalar occurrence domains with original-input isl sets.

Unknown optional expressions must preserve the core occurrence import. Expected
sets are supplied from each original program, not recomputed from native rows.
This is normalization parity, not proof of physical-slot mapping or event reuse.
"""
import argparse
import json
from pathlib import Path
import subprocess

from check_relations import ISL, isl_text


def marker(value, point=0, lower=0, upper=1, ty="index", budget=None):
    attrs = f"point = {point} : i64, lower = {lower} : i64, upper = {upper} : i64"
    if budget is not None:
        attrs += f", budget = {budget} : i64"
    return f'"test.scalar"({value}) {{{attrs}}} : ({ty}) -> ()'


def loop(body, low=0, high=8, ty="index", args="", root="", tail=""):
    annotation = "" if ty == "index" else f" : {ty}"
    return f'''module {{ func.func @f({args}) {{
  %lo = arith.constant {low} : {ty}
  %hi = arith.constant {high} : {ty}
  %one = arith.constant 1 : {ty}
  %two = arith.constant 2 : {ty}
  {root}
  scf.for %i = %lo to %hi step %one{annotation} {{
    {body}
  }}
  {tail}
  return
}} }}'''


def main():
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    isl = ISL()
    cases = []

    def run(name, source, expected, *, parameters=None, import_limit=None):
        path = args.output / (name + ".mlir")
        path.write_text(source)
        process = subprocess.run([str(args.driver.resolve()), str(path)], text=True,
                                 capture_output=True, timeout=60)
        (args.output / (name + ".stdout")).write_text(process.stdout)
        (args.output / (name + ".stderr")).write_text(process.stderr)
        assert process.returncode == 0, (name, process.stderr)
        result = json.loads(process.stdout)[0]
        if import_limit is not None:
            assert not result["complete"] and result["reason"] == import_limit, (name, result)
            assert "scalars" not in result, (name, result)
            cases.append({"case": name, "complete": False, "reason": result["reason"]})
            (args.output / "results.json").write_text(json.dumps(cases, indent=2) + "\n")
            return result
        assert result["complete"], (name, result)
        assert len(result["scalars"]) == len(expected), (name, result)
        if parameters is not None:
            assert len(result["parameters"]) == parameters, (name, result["parameters"])
        for native, wanted in zip(result["scalars"], expected):
            if wanted in ("unsupported", "budget-exhausted"):
                assert native["status"] == wanted and "relation" not in native, (name, native)
            else:
                assert native["status"] == "proved", (name, native)
                actual, exact = isl.map(isl_text(native["relation"])), isl.map(wanted)
                assert actual.equal(exact), (name, str(actual), str(exact))
        cases.append({"case": name, "queries": result["scalars"],
                      "parameters": result["parameters"], "complete": result["complete"]})
        (args.output / "results.json").write_text(json.dumps(cases, indent=2) + "\n")
        return result

    phase = '"test.phase"() : () -> ()'
    expected = ["{[] -> [0,i]: 0<=i<8}",
                "{[] -> [0,i]: 0<=i<8 and i%2=0}",
                "{[] -> [0,i]: 0<=i<8 and i%2=1}"]
    for op, operand in (("remsi", "%two"), ("remui", "%two"), ("andi", "%one")):
        run("parity_" + op, loop(f"%slot = arith.{op} %i, {operand} : index\n{phase}\n" +
            "\n".join(marker("%slot", lower=a, upper=b) for a, b in ((0, 1), (0, 0), (1, 1)))),
            expected, parameters=0)

    run("safe_next_slot", loop("%next = arith.addi %i, %one : index\n"
        "%slot = arith.remsi %next, %two : index\n" + phase + "\n" + marker("%slot", lower=0, upper=0)),
        ["{[] -> [0,i]: 0<=i<8 and (i+1)%2=0}"], parameters=0)

    # Original signed definition domain includes negative i. A later nonnegative
    # use guard cannot validate a different mathematical remainder at definition.
    bad = "%slot = arith.remsi %i, %two : i8\n%p = arith.cmpi sge, %i, %zero : i8\n" + \
          "scf.if %p {\n" + phase + "\n" + marker("%slot", ty="i8") + "\n}"
    run("negative_definition_later_safe_use", loop(bad, low=-2, high=4, ty="i8",
        root="%zero = arith.constant 0 : i8"), ["unsupported"], parameters=0)
    good = "%p = arith.cmpi sge, %i, %zero : i8\nscf.if %p {\n" + \
           "%slot = arith.remsi %i, %two : i8\n" + phase + "\n" + marker("%slot", lower=1, upper=1, ty="i8") + "\n}"
    run("qualified_definition", loop(good, low=-2, high=4, ty="i8",
        root="%zero = arith.constant 0 : i8"), ["{[] -> [0,i]: 0<=i<4 and i%2=1}"], parameters=0)

    wrap = "%slot = arith.addi %i, %eight : i8\n%p = arith.cmpi slt, %i, %limit : i8\n" + \
           "scf.if %p {\n" + phase + "\n" + marker("%slot", lower=127, upper=127, ty="i8") + "\n}"
    run("wrapping_definition_later_safe_use", loop(wrap, low=119, high=124, ty="i8",
        root="%eight = arith.constant 8 : i8\n%limit = arith.constant 120 : i8"),
        ["unsupported"], parameters=0)

    run("unknown_loop_select", loop("%slot = arith.select %p, %lo, %one : index\n" + phase + "\n" +
        marker("%slot"), args="%p: i1"), ["unsupported"], parameters=0)

    # Root arithmetic denotes one actual runtime SSA value. Keep it opaque,
    # rather than asserting x+1 in unbounded mathematical integers.
    root = run("opaque_root_wrapping_value", loop(phase + "\n" + marker("%slot", ty="i8"),
        high=4, ty="i8", args="%x: i8", root="%slot = arith.addi %x, %one : i8"),
        ["[p0] -> {[] -> [0,i]: 0<=i<4 and 0<=p0<=1}"], parameters=1)
    assert "ssa" in root["parameters"][0] and "argument" not in root["parameters"][0]

    run("loop_value_at_outer_point", loop("%slot = arith.remsi %i, %two : index\n" + phase + "\n" +
        marker("%slot", point=1), tail=phase), ["unsupported"], parameters=0)
    run("value_before_definition", loop(phase + "\n%slot = arith.remsi %i, %two : index\n" +
        marker("%slot")), ["unsupported"], parameters=0)
    run("zero_trip", loop("%slot = arith.remsi %i, %two : index\n" + phase + "\n" +
        marker("%slot"), high=0), ["{[] -> [0,i]: false}"], parameters=0)
    run("zero_query_budget", loop("%slot = arith.remsi %i, %two : index\n" + phase + "\n" +
        marker("%slot", budget=0)), ["budget-exhausted"], parameters=0)
    run("empty_requested_interval", loop(phase + "\n" + marker("%i", lower=2, upper=1)),
        ["{[] -> [0,i]: false}"], parameters=0)
    run("full_signed_query_bounds", loop(phase + "\n" + marker("%i", lower=-(2**63), upper=2**63-1)),
        ["{[] -> [0,i]: 0<=i<8}"], parameters=0)

    # Repeated operands form a small SSA DAG, not an exponentially large input.
    # A failed optional normalization must remain cheap and must not poison a
    # later supported query in the same immutable original definition domains.
    def unknown_dag(depth):
        return "%u0 = arith.xori %i, %one : i64\n" + "\n".join(
            f"%u{k} = arith.xori %u{k-1}, %u{k-1} : i64" for k in range(1, depth + 1))

    run("shared_unknown_optional_dag", loop(unknown_dag(40) +
        "\n%slot = arith.remsi %i, %two : i64\n" + phase + "\n" +
        marker("%u40", ty="i64") + "\n" + marker("%u40", ty="i64") + "\n" +
        marker("%slot", lower=1, upper=1, ty="i64"), ty="i64"),
        ["unsupported", "unsupported", "{[] -> [0,i]: 0<=i<8 and i%2=1}"], parameters=0)

    # The control walk cannot cache definition-dependent unknowns yet. Each
    # recursive visit still spends work, so this shared DAG has bounded refusal.
    run("shared_unknown_control_dag", loop(unknown_dag(24) +
        "\n%p = arith.cmpi eq, %u24, %lo : i64\nscf.if %p {\n" + phase + "\n}", ty="i64"),
        [], import_limit="occurrence expression work limit")

    chain = "%s0 = arith.addi %i, %one : i64\n" + "\n".join(
        f"%s{k} = arith.addi %s{k-1}, %one : i64" for k in range(1, 72))
    run("optional_expression_depth_limit", loop(chain + "\n" + phase + "\n" +
        marker("%s71", ty="i64"), ty="i64"), [],
        import_limit="occurrence expression nesting limit")
    print(f"Optional scalar domains: {len(cases)} cases passed; unknown precision preserves core import")


if __name__ == "__main__":
    main()
