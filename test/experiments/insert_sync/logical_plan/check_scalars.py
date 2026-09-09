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


def equality(left, right, source=0, target=1, ty="index", right_ty=None, budget=None,
             source_lower=None, target_upper=None, parameter_equals=None,
             empty_filter=False, wrong_filter_space=False):
    attrs = [f"source = {source} : i64", f"target = {target} : i64"]
    for name, value in (("budget", budget), ("source_lower", source_lower), ("target_upper", target_upper),
                        ("parameter_equals", parameter_equals)):
        if value is not None:
            attrs.append(f"{name} = {value} : i64")
    if empty_filter:
        attrs.append("empty_filter")
    if wrong_filter_space:
        attrs.append("wrong_filter_space")
    return f'"test.equal_scalars"({left}, {right}) {{{", ".join(attrs)}}} : ({ty}, {right_ty or ty}) -> ()'


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

    def run(name, source, expected, *, parameters=None, import_limit=None, equalities=()):
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
        assert len(result["equalities"]) == len(equalities), (name, result)
        for native, wanted in zip(result["equalities"], equalities):
            options = wanted if isinstance(wanted, dict) else {"relation": wanted}
            wanted = options["relation"]
            if "reason" in options:
                assert native["reason"] == options["reason"], (name, native)
            if wanted in ("unsupported", "budget-exhausted"):
                assert native["status"] == wanted and "relation" not in native, (name, native)
            else:
                assert native["status"] == "proved", (name, native)
                actual, exact = isl.map(isl_text(native["relation"])), isl.map(wanted)
                assert actual.equal(exact), (name, str(actual), str(exact))
                if "max_pieces" in options:
                    assert len(native["relation"]["pieces"]) <= options["max_pieces"], (name, native)
            if "filter_status" not in native:
                assert options.get("invalid_point"), (name, native)
                continue
            filter_status = options.get("filter_status", native["status"])
            assert native["filter_status"] == filter_status, (name, native)
            if filter_status != "proved":
                assert "filtered_relation" not in native, (name, native)
                if "reason" in options:
                    assert native["filter_reason"] == options["reason"], (name, native)
                continue
            original = isl.map(isl_text(native["original_occurrences"]))
            if "original" in options:
                assert original.equal(isl.map(options["original"])), (name, str(original), options["original"])
            filtered = isl.map(isl_text(native["filtered_relation"]))
            filter_equality = isl.map(options.get("filter_equality", wanted))
            assert filtered.equal(original & filter_equality), (name, str(filtered), str(original & filter_equality))
        cases.append({"case": name, "queries": result["scalars"],
                      "equalities": result["equalities"],
                      "synthetic_equality_dag_depth": result.get("synthetic_equality_dag_depth"),
                      "equality_work_scope": "query work only; excludes native import and synthetic expression construction",
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

    # Two executions of one SSA definition have different loop coordinates.
    # Equality is an exact relation; it supplies neither scalar execution order
    # nor a physical-slot mapping. The filtered form starts from native ordered
    # occurrences and must equal their intersection with the independent isl
    # equality, including any additional caller-retained restrictions.
    for depth in (2, 3, 16, 64):
        relation = f"{{[0,i] -> [1,j]: 0<=i<8 and 0<=j<8 and i%{depth}=j%{depth}}}"
        source = loop(
            "%slot = arith.remui %i, %depth : index\n" + phase + "\n" + phase + "\n" +
            equality("%slot", "%slot"), root=f"%depth = arith.constant {depth} : index")
        result = run(f"equal_same_ssa_depth{depth}", source, [],
            parameters=0, equalities=[{"relation": relation, "max_pieces": 1,
                "original": "{[0,i] -> [1,j]: 0<=i<8 and 0<=j<8 and i<=j}"}])
        if depth == 2:
            costs = result["equalities"][0]
            assert costs["filter_work"] < costs["work"], costs
            limited = source.replace(equality("%slot", "%slot"),
                                     equality("%slot", "%slot", budget=costs["filter_work"]))
            run("equal_filtered_budget_advantage", limited, [], parameters=0,
                equalities=[{"relation": "budget-exhausted", "filter_status": "proved", "filter_equality": relation}])

    same_phase = "{[0,i] -> [0,j]: 0<=i<8 and 0<=j<8 and i%2=j%2}"
    run("equal_same_static_phase", loop("%slot = arith.remui %i, %two : index\n" + phase + "\n" +
        equality("%slot", "%slot", target=0)), [], parameters=0,
        equalities=[{"relation": same_phase, "original": "{[0,i] -> [0,j]: 0<=i<j<8}"}])

    mixed = "%slot = arith.remui %i, %two : index\n%mask = arith.andi %i, %one : index\n" + phase + "\n" + phase
    run("equal_equivalent_spellings", loop(mixed + "\n" + equality("%slot", "%mask")), [], parameters=0,
        equalities=[{"relation": "{[0,i] -> [1,j]: 0<=i<8 and 0<=j<8 and i%2=j%2}", "max_pieces": 1}])
    shifted = "%next = arith.addi %i, %one : index\n%slot = arith.remui %i, %three : index\n" + \
              "%later = arith.remui %next, %three : index\n" + phase + "\n" + phase
    relation = "{[0,i] -> [1,j]: 0<=i<8 and 0<=j<8 and i%3=(j+1)%3}"
    run("equal_shifted_guarded_filter", loop(shifted + "\n" + equality("%slot", "%later", source_lower=2, target_upper=5),
        root="%three = arith.constant 3 : index"), [], parameters=0,
        equalities=[{"relation": relation, "max_pieces": 1,
            "original": "{[0,i] -> [1,j]: 2<=i<8 and 0<=j<=5 and i<=j}"}])

    guarded = "%slot = arith.remui %i, %two : index\n%even = arith.cmpi eq, %slot, %lo : index\n" + \
              "scf.if %even {\n" + phase + "\n}\n" + phase
    run("equal_guarded_source", loop(guarded + "\n" + equality("%slot", "%slot")), [], parameters=0,
        equalities=[{"relation": "{[0,i] -> [1,j]: 0<=i<8 and 0<=j<8 and i%2=0 and j%2=0}",
            "original": "{[0,i] -> [1,j]: 0<=i<8 and 0<=j<8 and i%2=0 and i<=j}"}])
    exclusive = "scf.if %take {\n" + phase + "\n} else {\n" + phase + "\n}\n" + equality("%i", "%i")
    run("equal_exclusive_original_guard", loop(exclusive, args="%take: i1"), [], parameters=1,
        equalities=["[p0] -> {[0,i] -> [1,j]: false}"])

    nested = "%slot = arith.remui %i, %two : index\n" + phase + "\n" + \
        "scf.for %k = %lo to %three step %one {\n" + \
        "%sum = arith.addi %i, %k : index\n%inner = arith.remui %sum, %two : index\n" + phase + "\n" + \
        equality("%slot", "%inner") + "\n}"
    run("equal_distinct_nested_invocations", loop(nested, high=4,
        root="%three = arith.constant 3 : index"), [], parameters=0,
        equalities=[{"relation": "{[0,i,k] -> [1,j,l]: 0<=i<4 and k=0 and 0<=j<4 and 0<=l<3 and i%2=(j+l)%2}",
            "original": "{[0,i,k] -> [1,j,l]: 0<=i<4 and k=0 and 0<=j<4 and 0<=l<3 and i<=j}"}])
    run("equal_zero_trip", loop(mixed + "\n" + equality("%slot", "%mask"), high=0), [], parameters=0,
        equalities=["{[0,i] -> [1,j]: false}"])

    run("equal_shared_parameter", loop(phase + "\n" + phase + "\n" + equality("%x", "%x", ty="i8"),
        args="%x: i8"), [], parameters=1,
        equalities=["[p0] -> {[0,i] -> [1,j]: -128<=p0<=127 and 0<=i<8 and 0<=j<8}"])
    run("equal_filtered_parameter_guard", loop(phase + "\n" + phase + "\n" +
        equality("%x", "%x", ty="i8", parameter_equals=1), args="%x: i8"), [], parameters=1,
        equalities=[{"relation": "[p0] -> {[0,i] -> [1,j]: -128<=p0<=127 and 0<=i<8 and 0<=j<8}",
            "original": "[p0] -> {[0,i] -> [1,j]: p0=1 and 0<=i<8 and 0<=j<8 and i<=j}"}])
    run("equal_distinct_parameters", loop(phase + "\n" + phase + "\n" + equality("%x", "%y", ty="i8"),
        args="%x: i8, %y: i8"), [], parameters=2,
        equalities=["[p0,p1] -> {[0,i] -> [1,j]: -128<=p0<=127 and -128<=p1<=127 and p0=p1 and 0<=i<8 and 0<=j<8}"])
    run("equal_boolean_parameter", loop(phase + "\n" + phase + "\n" + equality("%take", "%true", ty="i1"),
        args="%take: i1", root="%true = arith.constant true"), [], parameters=1,
        equalities=["[p0] -> {[0,i] -> [1,j]: p0=1 and 0<=i<8 and 0<=j<8}"])
    run("equal_extreme_literals", loop(phase + "\n" + phase + "\n" + equality("%minimum", "%maximum", ty="i64"),
        root=f"%minimum = arith.constant {-2**63} : i64\n%maximum = arith.constant {2**63-1} : i64"), [],
        parameters=2, equalities=["[p0,p1] -> {[0,i] -> [1,j]: false}"])

    run("equal_before_source_definition", loop(phase + "\n%slot = arith.remui %i, %two : index\n" + phase + "\n" +
        equality("%slot", "%slot")), [], parameters=0, equalities=["unsupported"])
    run("equal_target_outside_definition", loop("%slot = arith.remui %i, %two : index\n" + phase + "\n" +
        equality("%slot", "%slot"), tail=phase), [], parameters=0, equalities=["unsupported"])
    run("equal_unknown_target", loop("%slot = arith.remui %i, %two : index\n%unknown = arith.xori %i, %one : index\n" +
        phase + "\n" + phase + "\n" + equality("%slot", "%unknown")), [], parameters=0,
        equalities=["unsupported"])
    run("equal_unqualified_negative_remainder", loop(
        "%slot = arith.remsi %i, %two : i8\n" + phase + "\n" + phase + "\n" + equality("%slot", "%i", ty="i8"),
        low=-2, high=4, ty="i8"), [], parameters=0, equalities=["unsupported"])
    run("equal_exhausted_query_then_supported", loop(mixed + "\n" +
        "\n".join(equality("%slot", "%mask", budget=budget) for budget in (0, 7, 1000000))), [], parameters=0,
        equalities=["budget-exhausted", "budget-exhausted", "{[0,i] -> [1,j]: 0<=i<8 and 0<=j<8 and i%2=j%2}"])
    run("equal_invalid_point", loop(mixed + "\n" + equality("%slot", "%mask", target=99)), [], parameters=0,
        equalities=[{"relation": "unsupported", "invalid_point": True}])
    run("equal_empty_qualified_filter", loop(mixed + "\n" + equality("%slot", "%mask", empty_filter=True)), [],
        parameters=0, equalities=[{"relation": "{[0,i] -> [1,j]: 0<=i<8 and 0<=j<8 and i%2=j%2}",
            "original": "{[0,i] -> [1,j]: false}"}])
    run("equal_wrong_filter_space", loop(mixed + "\n" + equality("%slot", "%mask", wrong_filter_space=True)), [],
        parameters=0, equalities=[{"relation": "{[0,i] -> [1,j]: 0<=i<8 and 0<=j<8 and i%2=j%2}",
            "filter_status": "unsupported"}])

    # Test only the query's representation-cost qualification here. The native
    # loop has the singleton domain i=0, and each constructed floor diamond
    # denotes that same zero SSA value. This is not extra importer coverage.
    def diamond(depth, markers, *, chain=False):
        source = loop(phase + "\n" + phase + "\n" + "\n".join(markers), high=1)
        attrs = f"test.equality_dag_depth = {depth} : i64" + (", test.equality_chain" if chain else "")
        return source.replace("func.func @f() {", f"func.func @f() attributes {{{attrs}}} {{")

    run("equal_shared_diamond_small", diamond(8, [equality("%i", "%i")]), [], parameters=0,
        equalities=["{[0,i] -> [1,j]: i=0 and j=0}"])
    run("equal_shared_diamond_budget", diamond(26, [equality("%i", "%i", budget=b) for b in (40, 256, 1000000)]), [],
        parameters=0, equalities=[
            {"relation": "budget-exhausted", "reason": "scalar equality coefficient traversal unavailable"},
            {"relation": "budget-exhausted", "reason": "scalar equality affine traversal budget"},
            {"relation": "budget-exhausted", "reason": "scalar equality affine traversal budget"}])
    # A chain tests the same depth limit without making MLIR's construction of
    # the synthetic test expression itself expand a depth-32 diamond first.
    run("equal_expression_depth_limit", diamond(32, [equality("%i", "%i")], chain=True), [], parameters=0,
        equalities=[{"relation": "unsupported", "reason": "scalar equality coefficient traversal unavailable"}])
    print(f"Optional scalar domains: {len(cases)} cases passed; unknown precision preserves core import")


if __name__ == "__main__":
    main()
