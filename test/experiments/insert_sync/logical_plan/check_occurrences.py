# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Challenge native occurrence import with original IR and independent isl domains."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from check_relations import ISL, isl_text


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

    def run(name, source, complete=True):
        path = args.output / (name + ".mlir")
        path.write_text(source)
        process = subprocess.run([str(args.driver.resolve()), str(path)], text=True,
                                 capture_output=True, timeout=60)
        (args.output / (name + ".stdout")).write_text(process.stdout)
        (args.output / (name + ".stderr")).write_text(process.stderr)
        assert process.returncode == 0, (name, process.stderr)
        answer = json.loads(process.stdout)[0]
        assert answer["complete"] is complete, (name, answer)
        cases.append({"name": name, "complete": complete, "reason": answer["reason"]})
        return answer

    def equals(native, wanted):
        actual, expected = isl.map(isl_text(native)), isl.map(wanted)
        assert actual.equal(expected), (str(actual), str(expected))

    def boolean(predicate, constant):
        return f'''module {{ func.func @f(%p: i1) {{
  %c = arith.constant {constant}
  %q = arith.cmpi {predicate}, %p, %c : i1
  scf.if %q {{ "test.phase"() : () -> () }} else {{ "test.phase"() : () -> () }}
  return
}} }}'''

    for predicate, constant, then in (("eq", "true", 1), ("ne", "true", 0),
                                      ("eq", "false", 0), ("slt", "false", 1),
                                      ("sgt", "true", 0)):
        result = run(f"i1_{predicate}_{constant}", boolean(predicate, constant))
        equals(result["points"][0], f"[p0] -> {{[] -> [0]: p0={then}}}")
        equals(result["points"][1], f"[p0] -> {{[] -> [1]: p0={1-then}}}")
        for edge in result["orders"]:
            equals(edge["relation"], "[p0] -> {[a] -> [b]: false}")

    minimum = -2**63
    result = run("minimum_i64", f'''module {{ func.func @f(%x: i64) {{
  %min = arith.constant {minimum} : i64
  %q = arith.cmpi eq, %x, %min : i64
  scf.if %q {{ "test.phase"() : () -> () }}
  return
}} }}''')
    equals(result["points"][0], f"[p0,p1] -> {{[] -> [0]: p0={minimum} and p1={minimum}}}")

    run("overflowing_i32_increment", '''module { func.func @f() {
  %l = arith.constant 2147483646 : i32
  %h = arith.constant 2147483647 : i32
  %s = arith.constant 2 : i32
  scf.for %i = %l to %h step %s : i32 { "test.phase"() : () -> () }
  return
} }''', False)
    run("wide_induction", '''module { func.func @f(%n: i128) {
  %l = arith.constant 0 : i128
  %s = arith.constant 1 : i128
  scf.for %i = %l to %n step %s : i128 { "test.phase"() : () -> () }
  return
} }''', False)
    run("coefficient_growth", '''module { func.func @f() {
  %l = arith.constant 0 : index
  %h = arith.constant 1 : index
  %k = arith.constant 2305843009213693951 : index
  scf.for %i = %l to %h step %h {
    %x = arith.muli %i, %k : index
    %q = arith.cmpi eq, %x, %l : index
    scf.if %q { "test.phase"() : () -> () }
  }
  return
} }''', False)

    def difference_source(lower, inside_guard=False):
        difference = '%d = arith.subi %n, %i : i8\n%p = arith.cmpi sge, %d, %three : i8'
        body = ('%safe = arith.cmpi sge, %i, %z : i8\nscf.if %safe {\n' + difference +
                '\nscf.if %p { "test.phase"() : () -> () }\n}') if inside_guard else (
                difference + '\n%safe = arith.cmpi sge, %i, %z : i8\nscf.if %safe {\n' +
                'scf.if %p { "test.phase"() : () -> () }\n}')
        return f'''module {{ func.func @f(%n: i8) {{
  %z = arith.constant 0 : i8
  %lo = arith.constant {lower} : i8
  %one = arith.constant 1 : i8
  %three = arith.constant 3 : i8
  scf.for %i = %lo to %n step %one : i8 {{ {body} }}
  return
}} }}'''

    for name, lower, guarded in (("difference_loop_domain", 0, False),
                                 ("difference_definition_domain", -1, True)):
        result = run(name, difference_source(lower, guarded))
        equals(result["points"][0], "[p0] -> {[] -> [0,i]: 0<=i and i+3<=p0<=127}")
    # A safe use of the result does not make its earlier overflowing definition
    # mathematical: n=127,i=-1 wraps before the later i>=0 branch is evaluated.
    run("difference_unsafe_definition_safe_use", difference_source(-1), False)

    result = run("correlated_safe_addition", '''module { func.func @f(%n: i8) {
  %z = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  scf.for %i = %z to %n step %one : i8 {
    %d = arith.subi %n, %i : i8
    %sum = arith.addi %d, %i : i8
    %p = arith.cmpi eq, %sum, %n : i8
    scf.if %p { "test.phase"() : () -> () }
  }
  return
} }''')
    equals(result["points"][0], "[p0] -> {[] -> [0,i]: 0<=i<p0<=127}")
    run("addition_unsafe_definition_safe_use", '''module { func.func @f(%n: i8) {
  %z = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  scf.for %i = %z to %n step %one : i8 {
    %sum = arith.addi %n, %i : i8
    %safe = arith.cmpi eq, %i, %z : i8
    scf.if %safe {
      %p = arith.cmpi eq, %sum, %n : i8
      scf.if %p { "test.phase"() : () -> () }
    }
  }
  return
} }''', False)

    def division_source(opcode, guarded, divisor=3):
        math = f'%x = arith.addi %n, %i : i8\n%r = {opcode} %x, %d : i8'
        check = '%p = arith.cmpi eq, %r, %z : i8\nscf.if %p { "test.phase"() : () -> () }'
        body = (f'scf.if %safe {{ {math}\n{check} }}' if guarded else
                f'{math}\nscf.if %safe {{ {check} }}')
        return f'''module {{ func.func @f(%n: i8) {{
  %z = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  %d = arith.constant {divisor} : i8
  %safe = arith.cmpi sge, %n, %z : i8
  scf.for %i = %z to %one step %one : i8 {{ {body} }}
  return
}} }}'''

    for opcode in ("arith.remsi", "arith.divsi"):
        name = opcode.split(".")[1]
        result = run(name + "_guarded_definition", division_source(opcode, True))
        wanted = "p0 mod 3=0" if name == "remsi" else "p0<3"
        equals(result["points"][0], f"[p0] -> {{[] -> [0,i]: i=0 and 0<=p0<=127 and {wanted}}}")
        run(name + "_unsafe_definition_safe_use", division_source(opcode, False), False)
        for divisor in (0, -1):
            run(name + "_invalid_divisor_" + str(divisor), division_source(opcode, True, divisor), False)

    for spelling in ("arith.remui", "arith.andi"):
        divisor = 2 if spelling.endswith("remui") else 1
        result = run(spelling.replace(".", "_"), f'''module {{ func.func @f(%n: index) {{
  %z = arith.constant 0 : index
  %one = arith.constant 1 : index
  %k = arith.constant {divisor} : index
  "test.phase"() : () -> ()
  scf.for %i = %z to %n step %one {{
    %r = {spelling} %i, %k : index
    %p = arith.cmpi eq, %r, %z : index
    scf.if %p {{ "test.phase"() : () -> () }} else {{ "test.phase"() : () -> () }}
  }}
  "test.phase"() : () -> ()
  return
}} }}''')
        equals(result["points"][1], f"[p0] -> {{[] -> [1,i]: 0<=i<p0<= {2**63-1} and i mod 2=0}}")
        equals(result["points"][2], f"[p0] -> {{[] -> [2,i]: 0<=i<p0<= {2**63-1} and i mod 2=1}}")
        for edge in result["orders"]:
            if edge["source"] == 1 and edge["target"] == 2:
                equals(edge["relation"], f"[p0] -> {{[1,i] -> [2,j]: 0<=i<j<p0<={2**63-1} and i mod 2=0 and j mod 2=1}}")

    result = run("boolean_negation", '''module { func.func @f(%p: i1) {
  %true = arith.constant true
  %q = arith.xori %p, %true : i1
  scf.if %q { "test.phase"() : () -> () }
  return
} }''')
    equals(result["points"][0], "[p0] -> {[] -> [0]: p0=0}")
    result = run("nested_reversed_identities", '''module {
func.func @f(%p: i1) attributes {test.reverse_phases} {
  %z = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  scf.for %i = %z to %three step %one {
    "test.phase"() : () -> ()
    scf.for %j = %z to %two step %one {
      scf.if %p { "test.phase"() : () -> () }
    }
    "test.phase"() : () -> ()
  }
  return
} }''')
    equals(result["points"][2], "[p0] -> {[] -> [2,i,0]: 0<=i<3 and 0<=p0<=1}")
    equals(result["points"][1], "[p0] -> {[] -> [1,i,j]: 0<=i<3 and 0<=j<2 and p0=1}")
    for edge in result["orders"]:
        if edge["source"] == 2 and edge["target"] == 1:
            equals(edge["relation"], "[p0] -> {[2,i,0] -> [1,k,j]: 0<=i<=k<3 and 0<=j<2 and p0=1}")
        if edge["source"] == 1 and edge["target"] == 2:
            equals(edge["relation"], "[p0] -> {[1,i,j] -> [2,k,0]: 0<=i<k<3 and 0<=j<2 and p0=1}")
        if edge["source"] == 2 and edge["target"] == 0:
            equals(edge["relation"], "[p0] -> {[2,i,0] -> [0,k,0]: 0<=i<=k<3 and 0<=p0<=1}")

    result = run("nonunit_step", '''module { func.func @f() {
  %one = arith.constant 1 : index
  %three = arith.constant 3 : index
  %eight = arith.constant 8 : index
  scf.for %i = %one to %eight step %three { "test.phase"() : () -> () }
  return
} }''')
    equals(result["points"][0], "{[] -> [0,i]: i=1 or i=4 or i=7}")
    equals(result["orders"][0]["relation"], "{[0,1] -> [0,4]; [0,1] -> [0,7]; [0,4] -> [0,7]}")

    # The native adapter must qualify the entire selected population, and use
    # actual schedule ranks even when point IDs are deliberately reversed.
    def periodic_source(period=2, attributes="", body=None, step=1):
        if body is None:
            body = '''%r = arith.remui %i, %d : index
    %p = arith.cmpi eq, %r, %z : index
    scf.if %p { "test.phase"() : () -> ()
                "test.phase"() : () -> () }'''
        return f'''module {{ func.func @f(%n: index) attributes {{test.periodic {attributes}}} {{
  %z = arith.constant 0 : index
  %step = arith.constant {step} : index
  %d = arith.constant {period} : index
  scf.for %i = %z to %n step %step {{ {body} }}
  return
}} }}'''

    def check_periodic(name, source, status="proved", reversed_ids=False):
        answer = run(name, source)
        periodic = answer["periodic"]
        assert periodic["status"] == status, (name, periodic)
        if status != "proved":
            assert "relation" not in periodic
            return periodic
        domain = isl.map(isl_text(periodic["domain"]))
        selected = domain.reverse().then(domain)
        before = isl.map("[p0] -> {[a,i] -> [b,j]: false}")
        for edge in answer["orders"]:
            before = before | isl.map(isl_text(edge["relation"]))
        count = len(answer["points"])
        rank = f"{count - 1}-p" if reversed_ids else "p"
        schedule = isl.map(f"[p0] -> {{[p,i] -> [i,{rank}]}}")
        wanted = (before & selected).then(schedule).lexmin().then(schedule.reverse())
        actual = isl.map(isl_text(periodic["relation"]))
        assert actual.equal(wanted), (name, str(actual), str(wanted))
        return periodic

    check_periodic("periodic_mod2", periodic_source())
    check_periodic("periodic_mod3", periodic_source(3))
    check_periodic("periodic_reverse_ranks", periodic_source(attributes=", test.reverse_phases"), reversed_ids=True)
    check_periodic("periodic_selected_subdomain", periodic_source(attributes=", test.periodic_lower = 3 : i64, test.periodic_upper = 12 : i64"))
    check_periodic("periodic_empty", periodic_source(attributes=", test.periodic_empty"))
    check_periodic("periodic_exhausted", periodic_source(attributes=", test.periodic_budget = 0 : i64"), "budget-exhausted")
    check_periodic("periodic_nonunit", periodic_source(step=2).replace(
        "scf.for %i = %z to %n", "%hi = arith.constant 8 : index\n  scf.for %i = %z to %hi"), "unsupported")
    check_periodic("periodic_nested", periodic_source(body='''scf.for %j = %z to %n step %step {
      "test.phase"() : () -> () }'''), "unsupported")
    check_periodic("periodic_once", periodic_source(body='"test.phase"() : () -> ()'))
    # Local inequalities encode holes without a single selected residue. The
    # candidate extractor drops those constraints; the final equality gate must
    # reject the resulting overly broad period-one proposal.
    refusal = check_periodic("periodic_unrepresented_holes", periodic_source(3).replace(
        "arith.cmpi eq, %r, %z", "arith.cmpi ne, %r, %z"), "unsupported")
    assert "whole-population equality" in refusal["reason"], refusal
    assert not isl.map(isl_text(refusal["domain"])).empty(), refusal

    # Synthetic conjuncts refine actual phase domains. These tests challenge
    # exact integer elimination independently of imported scalar precision.
    def cell_case(name, locals_count, equations=(), inequalities=(), *, lower=0,
                  status="proved", wanted=None, budget=None, successor=False):
        def rows_attribute(label, rows):
            return ", " + label + " = [" + ", ".join(
                "[" + ", ".join(f"{value} : i64" for value in row) + "]" for row in rows) + "]"
        attributes = f", test.periodic_cell_locals = {locals_count} : i64, test.periodic_cell_probe"
        if not successor:
            attributes += ", test.periodic_cell_only"
        attributes += rows_attribute("test.periodic_cell_eq", equations)
        attributes += rows_attribute("test.periodic_cell_ge", inequalities)
        if budget is not None:
            attributes += f", test.periodic_budget = {budget} : i64"
        source = periodic_source(attributes=attributes, body='"test.phase"() : () -> ()')
        source = source.replace("%z = arith.constant 0 : index", f"%z = arith.constant {lower} : index")
        answer = run(name, source)
        assert answer["synthetic_periodic_refinement"] is True, answer
        fixed = ["phase", "i", "p0"]
        local_names = [f"l{k}" for k in range(locals_count)]
        def expression(row):
            assert len(row) == len(fixed) + locals_count + 1
            terms = [f"({coefficient}*{variable})" for coefficient, variable
                     in zip(row[:-1], fixed + local_names) if coefficient]
            return "(" + " + ".join(terms + [str(row[-1])]) + ")"
        constraints = [expression(row) + " = 0" for row in equations]
        constraints += [expression(row) + " >= 0" for row in inequalities]
        condition = " and ".join(constraints) if constraints else "true"
        if locals_count:
            condition = "exists (" + ",".join(local_names) + " : " + condition + ")"
        bounds = f"phase=0 and {lower}<=i<p0 and {-2**63}<=p0<={2**63-1}"
        expected = isl.map("[p0] -> {[] -> [phase,i] : " + bounds + " and " + condition + "}")
        probes = answer["periodic_cells"]
        assert len(probes) == 1, (name, probes)
        probe = probes[0]
        assert probe["status"] == status, (name, probe)
        original = isl.map(isl_text(probe["original"]))
        assert original.equal(expected), (name, "synthetic row placement", str(original), str(expected))
        if status == "proved":
            simplified = isl.map(isl_text(probe["simplified"]))
            assert simplified.equal(expected), (name, "integer elimination", str(simplified), str(expected))
            if wanted:
                target = isl.map("[p0] -> {[] -> [phase,i] : " + bounds + " and " + wanted + "}")
                assert simplified.equal(target), (name, str(simplified), str(target))
        else:
            assert "simplified" not in probe, (name, probe)
        if successor:
            periodic = answer["periodic"]
            assert periodic["status"] == "proved", (name, periodic)
            # A single actual phase: strict iteration order gives the complete
            # reference successor, independent of the native cell algorithm.
            order = isl.map("[p0] -> {[phase,i] -> [other,j] : i<j}")
            expected_next = (expected.reverse().then(expected) & order).lexmin()
            assert isl.map(isl_text(periodic["relation"])).equal(expected_next), (name, periodic)
        cases[-1]["cell_status"] = status
        cases[-1]["cell_work"] = probe["work"]
        return probe

    cell_case("cell_gcd_negative_constant", 0, inequalities=[[0, 2, 2, -3]], wanted="i+p0>=2")
    cell_case("cell_unit_lower_frontier", 1,
              inequalities=[[0,-1,0,1,0], [0,0,1,-3,0]], wanted="3*i<=p0", lower=-5)
    cell_case("cell_unit_upper_frontier", 1,
              inequalities=[[0,-1,0,3,0], [0,0,1,-1,0]], wanted="i<=3*p0", lower=-5)
    cell_case("cell_nonunit_both_frontiers", 1,
              inequalities=[[0,-1,0,3,0], [0,0,1,-2,0]], status="unsupported")
    cell_case("cell_nonunit_coupled_equality", 1,
              equations=[[0,-1,-1,2,0]], status="unsupported")
    cell_case("cell_free_local", 1, wanted="true", lower=-5)
    cell_case("cell_one_sided_lower", 1,
              inequalities=[[0,-1,0,3,0]], wanted="true", lower=-5)
    cell_case("cell_one_sided_upper", 1,
              inequalities=[[0,0,1,-3,0]], wanted="true", lower=-5)
    cell_case("cell_gcd_impossible_equality", 1,
              equations=[[0,2,0,2,1]], wanted="false")
    cell_case("cell_implicit_congruence", 1,
              inequalities=[[0,1,0,-2,-1], [0,-1,0,2,1]],
              wanted="i mod 2=1", lower=-7, successor=True)
    cell_case("cell_contradictory_opposite_bounds", 1,
              inequalities=[[0,1,0,-2,0], [0,-1,0,2,-1]], wanted="false")
    cell_case("cell_all_empty_population", 1,
              equations=[[0,2,0,2,1]], wanted="false", successor=True)
    cell_case("cell_empty_equality", 0, equations=[[0,0,0,1]], wanted="false")
    cell_case("cell_empty_inequality", 0, inequalities=[[0,0,0,-1]], wanted="false")
    cell_case("cell_shifted_negative_quotient", 2,
              equations=[[0,1,0,2,0,3], [0,0,0,-1,1,0]],
              wanted="i mod 2=1", lower=-7)
    cell_case("cell_redundant_quotients", 2,
              equations=[[0,1,0,-3,0,0], [0,1,0,0,-3,0]],
              wanted="i mod 3=0", lower=-7)

    def hidden_next(period):
        # Original locals a,b,j,c: i=p*a, j=p*b, c=floor((i+1)/p),
        # i<j<n, j<=2*i-p*c+p. On the selected residue c=a and j=i+p.
        equations = [[0,1,0,-period,0,0,0,0], [0,0,0,0,-period,1,0,0]]
        inequalities = [[0,-1,0,0,0,1,0,-1], [0,0,1,0,0,-1,0,-1],
                        [0,2,0,0,0,-1,-period,period],
                        [0,1,0,0,0,0,-period,1], [0,-1,0,0,0,0,period,period-2]]
        return equations, inequalities

    for period in (2,3,2147483647):
        eq, ge = hidden_next(period)
        receipt = cell_case(f"cell_hidden_next_{period}", 4, eq, ge,
                            wanted=f"i mod {period}=0 and i+{period}<p0", successor=True)
        if period == 2:
            cell_case("cell_hidden_next_negative", 4, eq, ge, lower=-7,
                      wanted="i mod 2=0 and i+2<p0", successor=True)
            cell_case("cell_budget_before_output", 4, eq, ge,
                      budget=receipt["work"]-1, status="budget-exhausted")

    growth = [[0,-k,0,1,0] for k in range(1,18)]
    growth += [[0,0,k,-1,1] for k in range(1,18)]
    cell_case("cell_fm_output_growth", 1, inequalities=growth, status="unsupported")
    check_periodic("periodic_cell_interval_envelope", periodic_source(body='''%r = arith.remui %i, %d : index
    %p = arith.cmpi eq, %r, %z : index
    scf.if %p { "test.phase"() : () -> () } else { "test.phase"() : () -> () }'''))
    mixed = check_periodic("periodic_mixed_empty_residues", periodic_source(body='''%one = arith.constant 1 : index
    %r = arith.remui %i, %d : index
    %p = arith.cmpi eq, %r, %z : index
    %q = arith.cmpi eq, %r, %one : index
    scf.if %p { "test.phase"() : () -> () } else {
      scf.if %q { "test.phase"() : () -> () } else { "test.phase"() : () -> () }
    }'''))
    mixed_domain = isl.map(isl_text(mixed["domain"]))
    for phase in (0, 1, 2):
        selected_phase = mixed_domain & isl.map(f"[p0] -> {{[] -> [phase,i] : phase={phase}}}")
        assert selected_phase.empty() == (phase == 2), (phase, mixed)
    # The weakest interval spanning both cells is only a proposal. Here it
    # would publish both phases at every iteration, adding real occurrences.
    mismatch = check_periodic("periodic_mismatched_phase_intervals", periodic_source(body='''%four = arith.constant 4 : index
    %p = arith.cmpi slt, %i, %four : index
    scf.if %p { "test.phase"() : () -> () } else { "test.phase"() : () -> () }'''), "unsupported")
    assert "whole-population equality" in mismatch["reason"], mismatch
    mismatch_domain = isl.map(isl_text(mismatch["domain"]))
    for phase in (0, 1):
        phase_filter = isl.map(f"[p0] -> {{[] -> [phase,i] : phase={phase}}}")
        assert not (mismatch_domain & phase_filter).empty(), (phase, mismatch)

    root = Path(__file__).resolve().parents[4]
    fixture = root / "test/samples/Qwen3DecodeA3/kernels/aiv/online_softmax.pto"
    source = fixture.read_text()
    result = run("unchanged_online_softmax", source)
    assert result["parameters"] == [{"argument": 4}], result["parameters"]
    # This frozen input has one loop, 1 <= i < arg4. Read its original
    # indentation to enumerate preloads/body/tail; no native schedule is reused.
    roles = ["body" if line.startswith("    ") else "root"
             for line in source.splitlines() if line.lstrip().startswith("pto.t")]
    assert len(roles) == len(result["points"])
    first, last = roles.index("body"), len(roles) - 1 - roles[::-1].index("body")
    bounds = f"{-2**63}<=p0<={2**63-1}"
    for edge in result["orders"]:
        p, q = edge["source"], edge["target"]
        pi, qi = first <= p <= last, first <= q <= last
        domain = ("1<=i<p0" if pi else "i=0") + " and " + ("1<=j<p0" if qi else "j=0")
        if pi and qi:
            precedes = "i<j" + (" or i=j" if p < q else "")
        else:
            precedes = "true" if p < q else "false"
        equals(edge["relation"], f"[p0] -> {{[{p},i] -> [{q},j]: {bounds} and {domain} and ({precedes})}}")
    cases[-1]["phase_pairs"] = len(result["orders"])
    summary = {"status": "passed", "cases": cases, "isl": isl.version,
               "driver_sha256": hashlib.sha256(args.driver.read_bytes()).hexdigest()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(cases)} native occurrence cases passed; {len(result['orders'])} unchanged softmax phase pairs checked")


if __name__ == "__main__":
    main()
