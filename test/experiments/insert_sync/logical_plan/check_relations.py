# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Check actual native queries against libisl and explicit expected relations.

No PTO parsing, compiler subprocess, or isl dependency is added to production.
All output belongs to the supplied fresh disk-backed campaign directory.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
from islwrap import ISL


def relation(d=1, r=1, s=0, eq=(), ge=(), locals=0, pieces=None):
    return {"d": d, "r": r, "s": s, "pieces": pieces if pieces is not None else
            [{"locals": locals, "eq": list(eq), "ge": list(ge)}]}


def isl_text(value):
    dims = [f"d{i}" for i in range(value["d"])]
    ranges = [f"r{i}" for i in range(value["r"])]
    params = [f"p{i}" for i in range(value["s"])]
    pieces = []
    for piece in value["pieces"]:
        locals = [f"e{i}" for i in range(piece["locals"])]
        names = dims + ranges + params + locals
        clauses = []
        for kind, operator in (("eq", "="), ("ge", ">=")):
            for row in piece[kind]:
                coefficients = [int(x) for x in row]
                assert len(coefficients) == len(names) + 1
                terms = [f"({c}*{name})" for c, name in zip(coefficients, names) if c]
                terms.append(str(coefficients[-1]))
                clauses.append("(" + "+".join(terms) + f") {operator} 0")
        condition = " and ".join(clauses) or "true"
        if locals:
            condition = "exists (" + ",".join(locals) + ": " + condition + ")"
        pieces.append("[" + ",".join(dims) + "] -> [" + ",".join(ranges) + "]: " + condition)
    return "[" + ",".join(params) + "] -> {" + "; ".join(pieces) + "}"


def check_periodic_successors(run, isl, output):
    """Challenge the compact query against general native extrema and isl.

    Only these small reference cases enumerate candidate phase pairs. Production
    scaling cases below never construct a quadratic reference population.
    """
    def request(common, atoms, period, **kw):
        return dict(op="periodic_successors", a=common, atoms=[dict(phase=p, rank=k, residue=r)
                    for p, k, r in atoms], period=period, phase_coordinate=0, iteration_coordinate=1, **kw)

    def challenge(name, atoms, period, lower, upper):
        common = relation(d=0, r=2, s=1, ge=[[0, 1, 0, -lower], [0, -1, 1, upper]])
        pieces = []
        for p, rank, residue in atoms:
            for q, next_rank, next_residue in atoms:
                pieces.append(dict(locals=2, eq=[
                    [1, 0, 0, 0, 0, 0, 0, -p], [0, 0, 1, 0, 0, 0, 0, -q],
                    [0, 1, 0, 0, 0, -period, 0, -residue],
                    [0, 0, 0, 1, 0, 0, -period, -next_residue]], ge=[
                    [0, 1, 0, 0, 0, 0, 0, -lower], [0, -1, 0, 0, 1, 0, 0, upper],
                    [0, 0, 0, 1, 0, 0, 0, -lower], [0, 0, 0, -1, 1, 0, 0, upper],
                    [0, -1, 0, 1, 0, 0, 0, 0 if rank < next_rank else -1]]))
        following = relation(d=2, r=2, s=1, pieces=pieces)
        schedule = isl.map("[p0] -> {" + ";".join(
            f"[p,i] -> [i,{rank}]: p={p}" for p, rank in sorted({(p,k) for p,k,r in atoms})) + "}")
        expected = isl.map(isl_text(following)).then(schedule).lexmin().then(schedule.reverse())
        answer = run(name, request(common, atoms, period), str(expected))
        # The native general query independently removes dominated successors.
        # The retained general implementation times out on the INT64-period
        # case. isl remains an exact independent oracle there; do not make the
        # automatic gate run that known expensive reference computation.
        if period <= 7:
            run(name + "_general", dict(op="first", a=following, b=following), str(expected))
        assert answer["periodic_output_pieces"] == len({(p,r) for p,k,r in atoms}), answer

    cases = [
        ("periodic_single", [(17,0,0)], 1, 0, -1),
        ("periodic_phase_rank", [(90,0,0),(3,1,0),(40,2,0)], 2, -3, 0),
        ("periodic_multiple_residues", [(7,0,0),(7,0,2),(2,1,1),(7,0,0)], 3, -7, 2),
        ("periodic_short_interval", [(4,0,1),(9,1,4)], 7, 0, -1),
        ("periodic_large_period", [(-2**63+1,0,0),(2**63-1,1,2**63-2)], 2**63-1, -1, 0),
    ]
    rng = random.Random(20260910)
    for i in range(12):
        period = rng.choice((1,2,3,5))
        atoms = [(phase, rank, rng.randrange(period)) for rank,phase in enumerate(rng.sample(range(40),3))]
        cases.append((f"periodic_random_{i}", atoms, period, rng.randrange(-6,3), rng.randrange(-3,4)))
    for case in cases:
        challenge(*case)

    singleton = relation(d=0, r=2, eq=[[0,1,-2]])
    run("periodic_singleton_rank", request(singleton, [(9,0,0),(3,1,0)], 2),
        "{[9,2] -> [3,2]}")
    run("periodic_extreme_phase", request(singleton, [(-2**63,0,0),(2**63-1,1,0)], 2),
        "{[-9223372036854775808,2] -> [9223372036854775807,2]}")
    run("periodic_singleton_no_next", request(singleton, [(9,0,0)], 2), "{[p,i] -> [q,j]: false}")
    empty_interval = relation(d=0, r=2, ge=[[0,1,0],[0,-1,-1]])
    run("periodic_empty_interval", request(empty_interval, [(9,0,0)], 2), "{[p,i] -> [q,j]: false}")
    run("periodic_empty_atoms", request(singleton, [], 2), "{[p,i] -> [q,j]: false}")
    run("periodic_empty_common", request(relation(d=0,r=2,pieces=[]), [(9,0,-1)], 2),
        "{[p,i] -> [q,j]: false}")
    existential = relation(d=0,r=2,s=1,locals=1,eq=[[0,0,1,-2,0]],
                           ge=[[0,1,0,0,1],[0,-1,0,0,3]])
    run("periodic_parameter_local", request(existential, [(9,0,0)], 2),
        "[p0] -> {[9,i] -> [9,j]: p0 mod 2=0 and i mod 2=0 and j=i+2 and -1<=i and j<=3}")
    fixed = relation(d=0,r=3,eq=[[0,0,1,-7]],ge=[[0,1,0,0],[0,-1,0,4]])
    run("periodic_fixed_invocation", request(fixed, [(9,0,0)], 2),
        "{[9,i,7] -> [9,j,7]: i mod 2=0 and j=i+2 and 0<=i and j<=4}")
    contradictory = relation(d=0,r=3,eq=[[0,0,1,-7],[0,0,1,-8]],ge=[[0,1,0,0],[0,-1,0,4]])
    run("periodic_contradictory_invocation", request(contradictory, [(9,0,0)], 2),
        "{[p,i,x] -> [q,j,y]: false}")
    contradictory_parameter = relation(d=0,r=2,s=1,eq=[[0,0,1,-2],[0,0,1,-3]],
                                       ge=[[0,1,0,0],[0,-1,0,4]])
    run("periodic_contradictory_parameter", request(contradictory_parameter, [(9,0,0)], 2),
        "[p0] -> {[p,i] -> [q,j]: false}")
    invalid = [
        ("phase_constrained", relation(d=0,r=2,eq=[[1,0,-9],[0,1,-2]]), [(9,0,0)], 2),
        ("nonunit_interval", relation(d=0,r=2,ge=[[0,2,0],[0,-1,3]]), [(9,0,0)], 2),
        ("iv_local", relation(d=0,r=2,locals=1,eq=[[0,1,-2,0]],ge=[[0,1,0,0],[0,-1,0,4]]), [(9,0,0)], 2),
        ("unfixed_invocation", relation(d=0,r=3,ge=[[0,1,0,0],[0,-1,0,4]]), [(9,0,0)], 2),
        ("rank_conflict", singleton, [(9,0,0),(9,1,1)], 2),
        ("rank_tie", singleton, [(9,0,0),(3,0,1)], 2),
        ("residue_out_of_range", singleton, [(9,0,2)], 2),
        ("period_nonpositive", singleton, [(9,0,0)], 0),
        ("missing_upper", relation(d=0,r=2,ge=[[0,1,0]]), [(9,0,0)], 2),
        ("missing_lower", relation(d=0,r=2,ge=[[0,-1,4]]), [(9,0,0)], 2),
        ("multiple_pieces", relation(d=0,r=2,pieces=singleton["pieces"]*2), [(9,0,0)], 2),
    ]
    for name, common, atoms, period in invalid:
        answer = run("periodic_refuse_" + name, request(common, atoms, period), status="unsupported")
        assert "relation" not in answer, answer
    for coordinates in ((2,1),(0,2),(0,0)):
        query = request(singleton, [(9,0,0)], 2)
        query.update(phase_coordinate=coordinates[0],iteration_coordinate=coordinates[1])
        run(f"periodic_invalid_coordinates_{coordinates[0]}_{coordinates[1]}", query, status="unsupported")
    wide = relation(d=0,r=2,s=1,eq=[[0,0,1,2**62]],ge=[[0,1,0,0],[0,-1,0,4]])
    answer = run("periodic_wide_coefficient", request(wide, [(9,0,0)], 2, template_scale=2),
                 status="unsupported")
    assert "coefficient exceeds int64" in answer["reason"], answer
    atoms = [(100-i,i,i % 3) for i in range(32)]
    unlimited = run("periodic_budget_reference", request(singleton, atoms, 3))
    for budget in (0,20,100,600,unlimited["work"]-1):
        answer = run(f"periodic_budget_{budget}", request(singleton, atoms, 3, budget=budget),
                     status="budget-exhausted")
        assert "relation" not in answer, answer
    scaling = []
    for count in (8,32,128,512):
        for period in (1,2**31-1):
            atoms = [(10000-i,i,0) for i in range(count)]
            common = relation(d=0,r=2,ge=[[0,1,0],[0,-1,period]])
            answer = run(f"periodic_scaling_{count}_{period}", request(common, atoms, period))
            # Compare each nonempty piece, then require the exact population.
            # A 512-piece union-equality check itself exceeds the reference's
            # solver allowance; these disjoint fixed-phase pieces avoid that
            # unrelated test-oracle Cartesian work.
            actual = answer["relation"]
            assert len(actual["pieces"]) == count, answer
            for i, piece in enumerate(actual["pieces"]):
                expected = (f"{{[{10000-i},t] -> [{10000-i-1},t]: t=0 or t={period}}}"
                            if i+1 < count else f"{{[{10000-count+1},0] -> [10000,{period}]}}")
                single = dict(actual, pieces=[piece])
                assert isl.map(isl_text(single)).equal(isl.map(expected)), (count,period,i)
            assert answer["periodic_atom_visits"] == count, answer
            assert answer["periodic_output_pieces"] == count, answer
            assert answer["periodic_sort_comparisons"] <= count * (count-1).bit_length(), answer
            assert answer["work"] <= 200*count*((count-1).bit_length()+1), answer
            scaling.append(dict(atoms=count,period=period,work=answer["work"],
                                comparisons=answer["periodic_sort_comparisons"],pieces=answer["periodic_output_pieces"]))
    (output / "periodic_scaling.json").write_text(json.dumps(scaling,indent=2)+"\n")


def main():
    if not __debug__:
        raise RuntimeError("Run relation checks without Python -O; assertions are required")
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    isl = ISL()
    results = []

    def run(name, request, expected=None, status="proved"):
        result = subprocess.run([str(args.driver.resolve())], input=json.dumps(request),
                                text=True, capture_output=True, timeout=45)
        (args.output / (name + ".request.json")).write_text(json.dumps(request, indent=2) + "\n")
        (args.output / (name + ".stdout")).write_text(result.stdout)
        (args.output / (name + ".stderr")).write_text(result.stderr)
        assert result.returncode == 0, (name, result.returncode, result.stderr)
        answer = json.loads(result.stdout)
        if request["op"] != "completion":
            assert answer["status"] == status, (name, answer)
        if expected is not None:
            actual = isl.map(isl_text(answer["relation"]))
            wanted = isl.map(expected)
            assert actual.equal(wanted), (name, str(actual), str(wanted))
        results.append({"case": name, "passed": True, "work": answer["work"]})
        return answer

    before = relation(ge=[[-1, 1, -1]])
    through = relation(ge=[[-1, 1, 0]])
    before_param = relation(s=1, ge=[[-1, 1, 0, -1]])
    through_param = relation(s=1, ge=[[-1, 1, 0, 0]])
    parity = relation(s=1, locals=1, eq=[[1, -1, 0, 2, 0]],
                      ge=[[1, 0, 0, 0, 0], [0, -1, 1, 0, -1], [-1, 1, 0, 0, -1]])
    run("latest_carried_parity", {"op": "latest", "a": parity, "b": before_param},
        "[p0] -> {[d0] -> [r0]: 0 <= d0 and d0 = r0-2 and r0 < p0}")
    run("next_same_slot", {"op": "first", "a": parity, "b": before_param},
        "[p0] -> {[d0] -> [r0]: 0 <= d0 and r0 = d0+2 and r0 < p0}")
    # Algebra oracle uses isl's lexmax rather than the native dominance formula.
    latest_isl = isl.map(isl_text(parity)).reverse().lexmax().reverse()
    run("isl_latest_parity", {"op": "latest", "a": parity, "b": before_param}, str(latest_isl))

    # Keep byte as the second sink coordinate until after latest-source selection.
    partial = relation(r=2, pieces=[
        {"locals": 0, "eq": [[1,0,0,0], [0,1,0,-2]], "ge": [[0,0,1,0], [0,0,-1,7]]},
        {"locals": 0, "eq": [[1,0,0,-1], [0,1,0,-2]], "ge": [[0,0,1,0], [0,0,-1,3]]}])
    run("partial_replacement", {"op": "latest", "a": partial, "b": before},
        "{[d0] -> [r0,r1]: r0=2 and ((d0=0 and 4<=r1<8) or (d0=1 and 0<=r1<4))}")
    # A may-write must stay in the conservative conflict contract; no kill query.
    run("may_write_keeps_all_conflicts", {"op": "subtract", "a": partial,
        "b": relation(r=2, pieces=[])},
        "{[d0] -> [r0,r1]: r0=2 and ((d0=0 and 0<=r1<8) or (d0=1 and 0<=r1<4))}")

    guarded = relation(s=1, pieces=[
        {"locals":0, "eq":[[1,0,0,0], [0,1,0,-1], [0,0,1,-1]], "ge":[]},
        {"locals":0, "eq":[[1,0,0,0], [0,1,0,-2]], "ge":[[0,0,1,0], [0,0,-1,1]]}])
    run("guarded_first_acquisition", {"op":"staircase", "a":guarded,
        "b":through_param, "c":before_param},
        "[p0] -> {[d0] -> [r0]: d0=0 and ((p0=1 and r0=1) or (p0=0 and r0=2))}")

    nested = relation(r=2, s=1, eq=[[1,-1,0,0,0]],
                      ge=[[1,0,0,0,0], [0,0,1,0,0], [0,0,-1,1,-1]])
    target_before = relation(d=2, r=2, s=1, pieces=[
        {"locals":0, "eq":[], "ge":[[-1,0,1,0,0,-1]]},
        {"locals":0, "eq":[[1,0,-1,0,0,0]], "ge":[[0,-1,0,1,0,-1]]}])
    run("nested_preload_first_use", {"op":"staircase", "a":nested,
        "b":through_param, "c":target_before},
        "[p0] -> {[d0] -> [r0,r1]: d0=r0 and d0>=0 and r1=0 and p0>0}")
    release = relation(d=2, s=1, eq=[[1,0,-1,0,1]],
                       ge=[[1,0,0,0,0], [0,1,0,0,0], [0,-1,0,1,-1]])
    run("nested_last_reader", {"op":"latest", "a":release, "b":target_before},
        "[p0] -> {[d0,d1] -> [r0]: d0>=0 and r0=d0+1 and d1=p0-1 and p0>0}")

    even = relation(eq=[[1,-1,0,0],[1,0,-2,0]], locals=1)
    diagonal = relation(eq=[[1,-1,0]])
    run("exact_existential_even_composition", {"op":"compose", "a":even, "b":diagonal},
        "{[d0] -> [r0]: d0=r0 and d0 mod 2=0}")
    shift = relation(eq=[[1,-1,1]])
    run("unit_local_substitution", {"op":"compose", "a":shift, "b":shift},
        "{[d0] -> [r0]: r0=d0+2}")
    run("nonunit_local_survives_substitution", {"op":"compose", "a":even, "b":shift},
        "{[d0] -> [r0]: r0=d0+1 and d0 mod 2=0}")
    run("exact_even_complement", {"op":"subtract", "a":diagonal, "b":even},
        "{[d0] -> [r0]: d0=r0 and d0 mod 2=1}")
    run("even_is_not_all_integers", {"op":"contains", "a":even, "b":diagonal}, status="not-established")

    # Challenge sufficient rational containment separately from exact integer
    # subtraction. In particular, unmatched existential locals must never be
    # identified just because the two relations have equally many columns.
    def contains_case(name, supply, requirement):
        remainder = isl.map(isl_text(requirement)) - isl.map(isl_text(supply))
        status = "proved" if remainder.empty() else "not-established"
        run(name, {"op": "contains", "a": supply, "b": requirement}, status=status)

    odd = relation(eq=[[1,-1,0,0], [1,0,-2,-1]], locals=1)
    # A single candidate antecedent serves several optimum queries. These
    # exercise both signs of equalities, late failures and a fresh candidate
    # after failure, so reusing its tableau cannot retain a tested RHS row.
    multi_needed = relation(d=2, r=2, s=1,
        eq=[[1,0,-1,0,0,0], [0,1,0,-1,0,0], [1,0,0,0,-1,0], [0,1,0,0,1,0]],
        ge=[[0,0,0,0,1,0], [0,0,0,0,-1,7]])
    multi_supply = relation(d=2, r=2, s=1,
        eq=[[1,1,0,0,0,0], [0,0,1,1,0,0], [2,0,-2,0,0,0], [0,-3,0,3,0,0],
            [1,0,0,0,-1,0]],
        ge=[[1,0,0,0,0,0], [0,-1,0,0,0,0], [0,0,-1,0,0,7], [0,0,0,1,0,7]])
    contains_case("reused_simplex_multiple_equalities", multi_supply, multi_needed)
    late_equality = relation(d=2, r=2, s=1,
        eq=multi_supply["pieces"][0]["eq"] + [[0,0,0,0,1,-3]],
        ge=multi_supply["pieces"][0]["ge"])
    contains_case("reused_simplex_late_equality_failure", late_equality, multi_needed)
    late_inequality = relation(d=2, r=2, s=1,
        eq=multi_supply["pieces"][0]["eq"],
        ge=multi_supply["pieces"][0]["ge"] + [[0,0,0,0,-1,3]])
    contains_case("reused_simplex_late_inequality_failure", late_inequality, multi_needed)
    contains_case("reused_simplex_candidate_reset", relation(d=2, r=2, s=1,
        pieces=late_equality["pieces"] + multi_supply["pieces"]), multi_needed)
    contains_case("reused_simplex_unbounded_directions", relation(d=2, r=2,
        eq=[[1,1,-1,-1,0], [2,0,-2,0,0], [0,-3,0,3,0]]),
        relation(d=2, r=2, eq=[[1,0,-1,0,0], [0,1,0,-1,0]]))

    contains_case("different_parity_witnesses", even, odd)
    contains_case("same_parity_redundant_constraints", even,
                  relation(eq=[[1,-1,0,0], [1,0,-2,0]], ge=[[1,0,0,0]], locals=1))
    contains_case("unrelated_existential_witnesses", even,
                  relation(eq=[[1,-1,0,0]], ge=[[0,0,1,0]], locals=1))
    integer_empty = relation(eq=[[1,-1,0], [2,0,-1]])
    contains_case("integer_empty_rationally_nonempty", even, integer_empty)
    contains_case("rationally_empty_antecedent", diagonal,
                  relation(ge=[[1,0,-2], [-1,0,1]]))
    pieces = relation(pieces=[
        {"locals":0, "eq":[[1,-1,0]], "ge":[[1,0,0], [-1,0,1]]},
        {"locals":0, "eq":[[1,-1,0]], "ge":[[1,0,-2], [-1,0,3]]}])
    contains_case("integer_union_only_coverage", pieces,
                  relation(eq=[[1,-1,0]], ge=[[1,0,0], [-1,0,3]]))
    contains_case("sample_preserves_parameter", relation(s=1, eq=[[1,-1,0,0], [0,0,1,0]]),
                  relation(s=1, eq=[[1,-1,0,0], [0,0,1,-1]]))
    large = 1 << 62
    contains_case("sample_preserves_large_integer", relation(eq=[[1,0,0]]),
                  relation(eq=[[1,-large,0], [0,1,-large]]))
    contains_case("sample_in_second_union_piece", pieces,
                  relation(eq=[[1,0,-2], [0,1,-2]]))

    # Complement must preserve residue restrictions in tightened bounds used
    # to discover quotient locals. Negating only the other constraints is unsound.
    tightened = relation(locals=1, ge=[[1, 0, -3, -1], [-1, 0, 3, 2]])
    quotient_eq = relation(locals=1, eq=[[1, 0, -3, -1]])
    contains_case("tightened_floor_not_universal", tightened, diagonal)
    contains_case("tightened_floor_matching_witness", tightened, quotient_eq)
    contains_case("tightened_floor_shifted_witness", tightened,
                  relation(locals=1, eq=[[1,0,-3,0]]))
    nested_divs = relation(locals=2, ge=[
        [1, 0, -3, 0, 0], [-1, 0, 3, 0, 2],
        [0, 0, 1, -2, -1], [0, 0, -1, 2, 1]])
    for name, rhs in (("tightened_division", tightened), ("equality_quotient", quotient_eq),
                      ("nested_division", nested_divs)):
        expected = str((isl.map(isl_text(diagonal)) - isl.map(isl_text(rhs))))
        run(name + "_complement", {"op": "subtract", "a": diagonal, "b": rhs}, expected)
        reordered = {**rhs, "pieces": [
            {**p, "eq": list(reversed(p["eq"] * 2)), "ge": list(reversed(p["ge"] * 2))}
            for p in rhs["pieces"]]}
        run(name + "_reordered", {"op": "subtract", "a": diagonal, "b": reordered}, expected)
    union_residues = relation(pieces=tightened["pieces"] + quotient_eq["pieces"])
    run("residue_union_complement", {"op": "subtract", "a": even, "b": union_residues},
        str((isl.map(isl_text(even)) - isl.map(isl_text(union_residues)))))
    existential_lhs = relation(locals=1, ge=[[1, 0, -3, 0], [0, 1, 2, 0]])
    run("arbitrary_left_witness_complement", {"op": "subtract", "a": existential_lhs, "b": tightened},
        str((isl.map(isl_text(existential_lhs)) - isl.map(isl_text(tightened)))))
    for name, low, high in (("matching", 0, 2), ("shifted", 1, 1)):
        lhs = relation(locals=1, ge=[[1, 0, -3, low], [-1, 0, 3, high], [0, -1, 1, 0]])
        run(name + "_floor_witness", {"op": "subtract", "a": lhs, "b": tightened},
            str(isl.map(isl_text(lhs)) - isl.map(isl_text(tightened))))
    captured = json.loads((HERE / "q_projection_difference.json").read_text())
    run("q_projection_native_difference", captured,
        str((isl.map(isl_text(captured["a"])) - isl.map(isl_text(captured["b"])))))
    run("q_projection_difference_exhaustion", {**captured, "budget": 20000}, status="budget-exhausted")
    # Symbolic-domain conversion may expose another non-division witness.
    # Keep qualification explicit rather than trusting one conversion call.
    bounded_witness = relation(d=0, r=2, s=1, locals=1, ge=[
        [0,0,0,1,-1], [0,0,1,-1,-1], [0,0,-1,0,9223372036854775807],
        [0,0,-1,0,3], [0,0,-1,0,2], [1,0,0,0,-38], [-1,0,0,0,38],
        [0,1,0,0,0], [0,-1,0,0,0]])
    nested_witness = relation(d=0, r=2, locals=5, ge=[
        [0,-1,2,0,0,0,0,1], [0,1,-2,0,0,0,0,0], [0,1,0,0,0,0,0,0],
        [0,-1,0,0,0,0,0,63], [0,-1,0,2,0,0,0,-1], [0,-1,0,0,0,0,0,1],
        [0,0,0,-1,0,0,0,30], [0,0,0,1,0,-1,0,0], [0,0,0,-1,0,1,0,0],
        [0,1,0,0,-2,0,0,1], [0,-1,0,0,2,0,0,0], [0,0,0,-1,0,0,0,1],
        [1,0,0,0,0,0,0,-6], [-1,0,0,0,0,0,0,6], [0,-1,0,0,0,0,0,0],
        [0,-1,0,0,0,0,2,0], [0,1,0,0,0,0,-2,1]])
    for name, rhs in (("bounded", bounded_witness), ("nested", nested_witness)):
        universe = relation(d=rhs["d"], r=rhs["r"], s=rhs["s"])
        run(name + "_witness_requalification", {"op": "subtract", "a": universe, "b": rhs},
            str(isl.map(isl_text(universe)) - isl.map(isl_text(rhs))))
    previous = relation(eq=[[1,-1,-1]])
    run("invocation_identity", {"op":"compose", "a":diagonal, "b":previous},
        "{[d0] -> [r0]: d0=r0+1}")
    run("wrong_invocation_not_supplied", {"op":"contains", "a":previous, "b":diagonal}, status="not-established")
    unbounded = relation(eq=[[0,1,0]], ge=[[1,0,0]])
    run("missing_maximum_is_not_no_requirement", {"op":"latest", "a":unbounded, "b":before}, status="not-established")

    def edges(pairs):
        return relation(pieces=[{"locals":0,"eq":[[1,0,-a],[0,1,-b]],"ge":[]} for a,b in pairs])
    primitive = edges([(0,1),(1,2),(2,3)])
    sparse = run("sparse_completion_joins_issue_order", {
        "op":"completion", "a":edges([(1,3),(5,7)]),
        "issue_order":edges([(0,0),(0,1),(1,1),(3,3),(3,4),(3,5),(4,4),(4,5),(5,5),(7,7),(7,8),(8,8)]),
        "needs":[{"relation":edges([(0,8)])}, {"relation":edges([(4,3)])}]})
    assert [v["status"] for v in sparse["answers"]] == ["proved","not-established"], sparse
    no_completion = run("issue_order_alone_is_not_completion", {
        "op":"completion", "a":edges([]), "issue_order":edges([(0,0),(0,1),(1,1)]),
        "needs":[{"relation":edges([(0,1)])}]})
    assert no_completion["answers"][0]["status"] == "not-established", no_completion
    screened = run("absent_boundary_is_not_fixed_point", {
        "op":"completion", "a":edges([(0,2),(3,5)]),
        "issue_order":edges([(0,0),(0,1),(1,1),(2,2),(2,3),(3,3),(5,5)]),
        "global_order":through,
        "needs":[{"relation":edges([(1,2)])},{"relation":edges([(0,5)])}]})
    assert screened["answers"][0] == {"status":"not-established","fixed":False}, screened
    assert screened["answers"][1]["status"] == "proved", screened
    necessary_only = run("global_issue_screen_supplies_nothing", {
        "op":"completion", "a":edges([(3,5)]),
        "issue_order":edges([(1,1),(3,3),(5,5)]), "global_order":through,
        "needs":[{"relation":edges([(1,5)])}]})
    assert necessary_only["answers"][0]["status"] == "not-established", necessary_only
    # Independent finite reachability challenges cached source scopes across
    # many successive queries, including branches with no completion supply.
    # Plain lane issue edges may join handoffs but cannot prove completion alone.
    rng = random.Random(8127)
    for case in range(12):
        size = 7
        lanes = [rng.randrange(3) for _ in range(size)]
        issue = {(a,b) for a in range(size) for b in range(a,size) if lanes[a] == lanes[b]}
        handoffs = {(a,b) for a in range(size) for b in range(a+1,size) if rng.randrange(5) == 0}
        def compose_pairs(left, right):
            return {(a,c) for a,b in left for x,c in right if b == x}
        supplied = compose_pairs(compose_pairs(issue, handoffs), issue)
        while True:
            following = supplied | compose_pairs(supplied, supplied)
            if following == supplied:
                break
            supplied = following
        pairs = [(a,b) for a in range(size) for b in range(size)]
        rng.shuffle(pairs)
        answer = run(f"scoped_finite_completion_{case}", {
            "op":"completion", "a":edges(sorted(handoffs)), "issue_order":edges(sorted(issue)),
            "global_order":through, "needs":[{"relation":edges([pair])} for pair in pairs]})
        expected = ["proved" if pair in supplied else "not-established" for pair in pairs]
        assert [a["status"] for a in answer["answers"]] == expected, (case, answer, expected)

    def guarded_edges(values):
        return relation(s=1, pieces=[{"locals":0,"eq":[[1,0,0,-a],[0,1,0,-b],[0,0,1,-guard]],"ge":[]}
                                    for a,b,guard in values])
    scoped_guard = run("scoped_cache_keeps_other_execution_domains", {
        "op":"completion", "a":guarded_edges([(0,2,1),(0,3,0)]),
        "issue_order":guarded_edges([(p,p,g) for p in (0,2,3) for g in (0,1)]),
        "needs":[{"relation":guarded_edges([edge])} for edge in ((0,2,1),(0,3,0),(0,2,0))]})
    assert [a["status"] for a in scoped_guard["answers"]] == ["proved","proved","not-established"], scoped_guard

    chain = edges([(0,1),(1,2),(2,3),(3,4)])
    continued_sparse = run("sparse_early_success_preserves_pending", {
        "op":"completion", "a":chain, "issue_order":edges([(i,i) for i in range(5)]),
        "needs":[{"relation":edges([(0,1)]),"rounds":0},
                 {"relation":edges([(0,4)]),"rounds":2},
                 {"relation":edges([(4,0)])}, {"relation":edges([(1,4)]),"rounds":2}]})
    assert [a["status"] for a in continued_sparse["answers"]] == ["proved","proved","not-established","proved"], continued_sparse
    # Nonconstant coordinate zero uses the whole actual O domain. A first
    # narrow query must not truncate the cached sources for a later query.
    positive_shift = relation(eq=[[1,-1,1]], ge=[[1,0,0]])
    broad_cache = run("sparse_nonconstant_source_domain_expands", {
        "op":"completion", "a":positive_shift, "issue_order":diagonal, "global_order":through,
        "needs":[{"relation":relation(eq=[[1,-1,1]],ge=[[1,0,2],[1,0,0],[0,-1,4]])},
                 {"relation":positive_shift},
                 {"relation":relation(eq=[[1,-1,1]],ge=[[1,0,2]])}]})
    assert [a["status"] for a in broad_cache["answers"]] == ["proved","proved","not-established"], broad_cache
    replacement = run("replacement_keeps_only_immutable_order_cache", {
        "op":"completion", "a":edges([(0,1),(1,2)]), "issue_order":edges([(i,i) for i in range(3)]),
        "needs":[{"relation":edges([(0,2)])},
                 {"relation":edges([(0,2)]),"replace_handoffs":edges([(1,2)])},
                 {"relation":edges([(0,2)]),"replace_handoffs":edges([(0,1),(1,2)])}]})
    assert [a["status"] for a in replacement["answers"]] == ["proved","not-established","proved"], replacement

    addition = run("new_handoff_reopens_saturated_source", {
        "op": "completion", "a": edges([(0, 1)]), "issue_order": edges([(i, i) for i in range(4)]),
        "needs": [{"relation": edges([(0, 3)])},
                  {"relation": edges([(0, 3)]), "add_handoffs": edges([(1, 2), (2, 3)])},
                  {"relation": edges([(3, 0)])},
                  {"relation": edges([(0, 3)]), "replace_handoffs": edges([(1, 2), (2, 3)])}]})
    assert [a["status"] for a in addition["answers"]] == ["not-established", "proved", "not-established", "not-established"], addition
    guarded_addition = run("new_handoff_retains_occurrence_guard", {
        "op": "completion", "a": guarded_edges([(0, 1, 0), (0, 1, 1)]),
        "issue_order": guarded_edges([(i, i, g) for i in range(3) for g in (0, 1)]),
        "needs": [{"relation": guarded_edges([(0, 2, 0), (0, 2, 1)])},
                  {"relation": guarded_edges([(0, 2, 1)]), "add_handoffs": guarded_edges([(1, 2, 1)])},
                  {"relation": guarded_edges([(0, 2, 0)])}]})
    assert [a["status"] for a in guarded_addition["answers"]] == ["not-established", "proved", "not-established"], guarded_addition
    new_direct = run("new_handoff_enters_cached_empty_scope", {
        "op": "completion", "a": edges([]), "issue_order": edges([(i, i) for i in range(3)]),
        "needs": [{"relation": edges([(0, 2)])},
                  {"relation": edges([(0, 2)]), "add_handoffs": edges([(0, 2)])}]})
    assert [a["status"] for a in new_direct["answers"]] == ["not-established", "proved"], new_direct
    screened_addition = run("added_handoff_updates_negative_screen", {
        "op": "completion", "a": edges([(3, 5)]),
        "issue_order": edges([(i, i) for i in (1, 3, 5)]), "global_order": through,
        "needs": [{"relation": edges([(1, 5)])},
                  {"relation": edges([(1, 5)]), "add_handoffs": edges([(1, 3)])}]})
    assert [a["status"] for a in screened_addition["answers"]] == ["not-established", "proved"], screened_addition

    # The explicit relation constructor remains an oracle for an owning lazy
    # provider. Compare each retained supply, not only the eventual statuses.
    def provider_parity(name, request, expected):
        eager = run(name + "_eager", {**request, "record_steps": True})
        lazy = run(name + "_lazy", {**request, "record_steps": True, "lazy": True})
        assert eager["answers"] == lazy["answers"], (name, eager, lazy)
        assert [x["status"] for x in eager["answers"]] == expected, (name, eager)
        assert len(eager["steps"]) == len(lazy["steps"]) == len(expected)
        for left, right in zip(eager["steps"], lazy["steps"]):
            assert isl.map(isl_text(left["supply"])).equal(isl.map(isl_text(right["supply"]))), name
        return eager, lazy

    provider_parity("provider_empty_and_issue_only", {
        "op": "completion", "a": edges([]), "issue_order": edges([(0,0),(0,1),(1,1)]),
        "needs": [{"relation": edges([])}, {"relation": edges([(0,1)])}]},
        ["proved", "not-established"])
    provider_parity("provider_addition_to_cached_empty_scope", {
        "op": "completion", "a": edges([]), "issue_order": edges([(i,i) for i in range(3)]),
        "needs": [{"relation": edges([(0,2)])},
                  {"relation": edges([(0,2)]), "add_handoffs": edges([(0,2)])},
                  {"relation": edges([(0,2)]), "replace_handoffs": edges([])}]},
        ["not-established", "proved", "not-established"])
    provider_parity("provider_resumed_and_reset", {
        "op": "completion", "a": chain, "issue_order": edges([(i,i) for i in range(5)]),
        "needs": [{"relation": edges([(0,1)]), "rounds": 0},
                  {"relation": edges([(0,4)]), "rounds": 2},
                  {"relation": edges([(4,0)])},
                  {"relation": edges([(0,4)]), "replace_handoffs": edges([(1,2),(2,3),(3,4)])},
                  {"relation": edges([(0,4)]), "add_handoffs": edges([(0,1)])}]},
        ["proved", "proved", "not-established", "not-established", "proved"])
    provider_parity("provider_guarded_addition", {
        "op": "completion", "a": guarded_edges([(0,1,0),(0,1,1)]),
        "issue_order": guarded_edges([(i,i,g) for i in range(3) for g in (0,1)]),
        "needs": [{"relation": guarded_edges([(0,2,0),(0,2,1)])},
                  {"relation": guarded_edges([(0,2,1)]), "add_handoffs": guarded_edges([(1,2,1)])},
                  {"relation": guarded_edges([(0,2,0)])}]},
        ["not-established", "proved", "not-established"])
    provider_parity("provider_unknown_scope_narrow_then_broad", {
        "op": "completion", "a": positive_shift, "issue_order": diagonal, "global_order": through,
        "needs": [{"relation": relation(eq=[[1,-1,1]], ge=[[1,0,0],[0,-1,4]])},
                  {"relation": positive_shift},
                  {"relation": relation(eq=[[1,-1,1]], ge=[[1,0,2]])}]},
        ["proved", "proved", "not-established"])
    provider_parity("provider_screen_is_not_completion", {
        "op": "completion", "a": edges([(3,5)]),
        "issue_order": edges([(i,i) for i in (1,3,5)]), "global_order": through,
        "needs": [{"relation": edges([(1,5)])},
                  {"relation": edges([(1,5)]), "add_handoffs": edges([(1,3)])}]},
        ["not-established", "proved"])
    provider_parity("provider_ids_are_not_schedule_order", {
        "op": "completion", "a": edges([(9,3),(3,1)]), "issue_order": edges([(i,i) for i in (9,3,1)]),
        "global_order": edges([(9,9),(9,3),(9,1),(3,3),(3,1),(1,1)]),
        "needs": [{"relation": edges([(9,1)])}, {"relation": edges([(1,9)])}]},
        ["proved", "not-established"])
    # Unknown ORDER coordinates must remain candidates for a fixed query, and
    # unknown query coordinates must select all applicable fixed-order blocks.
    wildcard_order = relation(pieces=diagonal["pieces"] + edges([(0,2)])["pieces"])
    provider_parity("provider_wildcard_order", {
        "op": "completion", "a": edges([(2,3)]), "issue_order": wildcard_order,
        "needs": [{"relation": edges([(0,3)])}, {"relation": edges([(2,3)])}]},
        ["proved", "proved"])
    provider_parity("provider_wildcard_query", {
        "op": "completion", "a": edges([(0,1),(2,3)]),
        "issue_order": edges([(i,i) for i in range(4)]),
        "needs": [{"relation": relation(eq=[[1,-1,1]], ge=[[1,0,0],[-1,0,2]])}]},
        ["not-established"])

    # A finite independent occurrence model with phase, outer invocation, inner
    # iteration, a trip parameter and a skipped-reader parameter. Returning to
    # the same phase in another invocation must not reuse the first query's
    # narrower guard/domain. Compute expected supply by ordinary tuple closure.
    def occurrence_edges(values):
        pieces = []
        for source, target, trips, take in values:
            equalities = []
            for column, value in enumerate((*source, *target, trips, take)):
                row = [0] * 9
                row[column], row[-1] = 1, -value
                equalities.append(row)
            pieces.append({"locals": 0, "eq": equalities, "ge": []})
        return relation(d=3, r=3, s=2, pieces=pieces)

    occurrence_order, occurrence_global, occurrence_handoffs, expected_supply = [], [], [], set()
    for trips in (0,1,2):
        for take in (0,1):
            points = [(phase, outer, inner) for outer in range(2) for phase in range(3)
                      for inner in (range(trips) if phase == 1 and take else (() if phase == 1 else (0,)))]
            schedule = lambda point: (point[1], point[0], point[2])
            order = {(a,b) for a in points for b in points if a[0] == b[0] and schedule(a) <= schedule(b)}
            all_order = {(a,b) for a in points for b in points if schedule(a) <= schedule(b)}
            handoffs = ({((0,o,0),(1,o,0)) for o in range(2)} |
                        {((1,o,trips-1),(2,o,0)) for o in range(2)}) if trips and take else set()
            compose = lambda a,b: {(x,z) for x,y in a for v,z in b if y == v}
            supplied = compose(compose(order, handoffs), order)
            while True:
                next_supply = supplied | compose(supplied, supplied)
                if next_supply == supplied: break
                supplied = next_supply
            occurrence_order.extend((a,b,trips,take) for a,b in sorted(order))
            occurrence_global.extend((a,b,trips,take) for a,b in sorted(all_order))
            occurrence_handoffs.extend((a,b,trips,take) for a,b in sorted(handoffs))
            expected_supply.update((a,b,trips,take) for a,b in supplied)
    requested_occurrences = [((0,0,0),(2,0,0),1,1), ((0,1,0),(2,1,0),2,1),
                             ((0,0,0),(2,0,0),0,1), ((0,0,0),(2,0,0),2,0),
                             ((0,1,0),(2,0,0),2,1), ((0,0,0),(2,1,0),2,1)]
    provider_parity("provider_nested_skipped_and_zero", {
        "op": "completion", "a": occurrence_edges(occurrence_handoffs),
        "issue_order": occurrence_edges(occurrence_order), "global_order": occurrence_edges(occurrence_global),
        "needs": [{"relation": occurrence_edges([item])} for item in requested_occurrences]},
        ["proved" if item in expected_supply else "not-established" for item in requested_occurrences])

    # Warm both caches, then challenge one narrow query in an order universe
    # with 1,024 independent phase blocks. The per-query metadata probes must
    # depend on requested endpoint buckets, not scan the universe again.
    eager, lazy = provider_parity("provider_sparse_lookup_work", {
        "op": "completion", "a": edges([(0,1)]),
        "issue_order": edges([(i,i) for i in range(1024)]),
        "needs": [{"relation": edges([(0,1)])}, {"relation": edges([(0,1)])}]},
        ["proved", "proved"])
    assert eager["steps"][1]["index_lookups"] - eager["steps"][0]["index_lookups"] <= 4, eager["steps"]
    assert lazy["steps"][1]["provider_block_lookups"] - lazy["steps"][0]["provider_block_lookups"] <= 2, lazy["steps"]
    assert lazy["steps"][1]["provider_returned_pieces"] - lazy["steps"][0]["provider_returned_pieces"] == 1
    independent = edges([(2*i,2*i+1) for i in range(16)])
    eager, lazy = provider_parity("provider_sparse_rows_many_endpoints", {
        "op": "completion", "a": independent,
        "issue_order": edges([(i,i) for i in range(1024)]),
        "needs": [{"relation": independent}]}, ["proved"])
    assert eager["steps"][0]["index_lookups"] <= 64, eager["steps"]
    assert lazy["steps"][0]["provider_block_lookups"] <= 64, lazy["steps"]
    for fault in ("unsupported", "budget-exhausted"):
        answer = run("provider_reports_" + fault, {
            "op": "completion", "a": edges([(0,1)]), "issue_order": diagonal, "lazy": True,
            "provider_fault": fault, "needs": [{"relation": edges([(0,1)])}]})
        assert answer["answers"][0]["status"] == fault, answer

    # Structural deduplication must retain distinct pieces and confirm equality
    # inside fingerprint buckets. Repeated pieces provide actual duplicate
    # candidates; the independent isl union checks semantic preservation.
    duplicates = edges([(i,i) for i in range(256)] * 3)
    dedup = run("normalize_fingerprinted_pieces", {"op": "normalize", "a": duplicates},
                str(isl.map(isl_text(duplicates))))
    assert len(dedup["relation"]["pieces"]) == 256, dedup
    assert dedup["work"] <= 32 * len(duplicates["pieces"]), dedup["work"]
    endpoint_scaling = []
    for size in (32,128,512):
        selected = relation(d=0, pieces=[{"locals":0,"eq":[[1,-i]],"ge":[]} for i in range(size)])
        ordered = edges([(i,i) for i in range(size)])
        answer = run(f"indexed_endpoint_scaling_{size}", {
            "op": "restrict_endpoints", "a": ordered, "sources": selected, "targets": selected},
            str(isl.map(isl_text(ordered))))
        assert answer["endpoint_comparisons"] == 2 * size, answer
        assert answer["work"] <= 25 * size + 10, answer
        endpoint_scaling.append({"pieces": size, "comparisons": answer["endpoint_comparisons"], "work": answer["work"]})
    # The fast first-coordinate lookup must retain tests of OTHER coordinates;
    # a wildcard in either order or candidate set must remain conservative.
    two_coordinates = relation(d=2,r=2, eq=[[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,-1],[0,0,0,1,0]])
    wrong_inner = relation(d=0,r=2, eq=[[1,0,0],[0,1,-1]])
    target_point = relation(d=0,r=2, eq=[[1,0,-1],[0,1,0]])
    run("indexed_endpoint_checks_full_coordinates", {
        "op":"restrict_endpoints", "a":two_coordinates, "sources":wrong_inner, "targets":target_point},
        "{[d0,d1] -> [r0,r1]: false}")
    unknown_phase = relation(d=0,r=2, eq=[[0,1,0]])
    run("indexed_endpoint_unknown_candidate", {
        "op":"restrict_endpoints", "a":two_coordinates, "sources":unknown_phase, "targets":target_point},
        str(isl.map(isl_text(two_coordinates))))
    # Dense +/-1 terms after a long zero prefix do not define fixed
    # coordinates. The row scan must also retain a genuinely isolated late
    # coordinate, including its negative coefficient and range offset.
    def wide_row(width, terms, constant=0):
        row = [0] * (width + 1)
        for column, coefficient in terms.items():
            row[column] = coefficient
        row[-1] = constant
        return row
    wide = relation(d=32,r=32,eq=[
        wide_row(64,{0:1},-3), wide_row(64,{32:1},-7),
        wide_row(64,{column:(1 if column % 2 == 0 else -1) for column in range(48,64)})])
    wide_source = relation(d=0,r=32,eq=[wide_row(32,{0:1},-3)])
    wide_target = relation(d=0,r=32,eq=[wide_row(32,{0:1},-7),wide_row(32,{16:1},-2)])
    run("indexed_endpoint_dense_late_coefficients_remain_unknown", {
        "op":"restrict_endpoints", "a":wide,"sources":wide_source,"targets":wide_target},
        str(isl.map(isl_text(wide))))
    late_fixed = relation(d=32,r=32,eq=wide["pieces"][0]["eq"]+[wide_row(64,{63:-1},3)])
    late_mismatch = relation(d=0,r=32,eq=[wide_row(32,{0:1},-7),wide_row(32,{31:1},-4)])
    run("indexed_endpoint_late_isolated_coordinate_still_filters", {
        "op":"restrict_endpoints", "a":late_fixed,"sources":wide_source,"targets":late_mismatch},
        str(isl.map(isl_text(relation(d=32,r=32,pieces=[])))))
    (args.output / "endpoint_scaling.json").write_text(json.dumps(endpoint_scaling, indent=2) + "\n")

    answer = run("continued_completion_query", {"op":"completion", "a":primitive, "needs":[
        {"relation":edges([(0,1)]), "rounds":0},
        {"relation":edges([(0,3)]), "rounds":2},
        {"relation":edges([(3,0)]), "rounds":8}]})
    assert answer["answers"] == [{"status":"proved","fixed":False},
                                  {"status":"proved","fixed":False},
                                  {"status":"not-established","fixed":True}], answer
    barrier = run("selected_barrier_supplies_later_need", {"op":"completion", "a":edges([(0,2),(1,2),(2,3)]),
        "needs":[{"relation":edges([(1,3)])}]})
    assert barrier["answers"][0]["status"] == "proved", barrier
    # Without either barrier, mere command issue order supplies no completion.
    run("deleted_barriers_cannot_prove_themselves", {"op":"contains", "a":edges([]),
        "b":edges([(0,2),(1,3)])}, status="not-established")
    run("exhausted_query", {"op":"compose", "a":even, "b":diagonal, "budget":0}, status="budget-exhausted")
    run("incompatible_spaces", {"op":"compose", "a":diagonal, "b":target_before}, status="unsupported")
    # Symbolic expected-relation checks for varied slot counts.
    for slots in (1,2,3,4):
        value = relation(s=1, locals=1, eq=[[1,-1,0,slots,0]],
                         ge=[[1,0,0,0,0],[0,-1,1,0,-1],[-1,1,0,0,-1]])
        run(f"slot_count_{slots}", {"op":"latest","a":value,"b":before_param},
            f"[p0] -> {{[d0] -> [r0]: 0<=d0 and d0=r0-{slots} and r0<p0}}")
    # The prepared RHS is an owning, immutable operand. Compare every result
    # and exact budget charge with the uncached implementation, including
    # wildcard coordinates, shifted division witnesses and RHS replacement.
    def cached_composition(name, rhs, inputs):
        request = {"op": "compose_sequence", "a": inputs[0]["left"], "b": rhs, "inputs": inputs}
        ordinary = run(name + "_ordinary", request)
        cached = run(name + "_prepared", dict(request, prepared=True))
        current = rhs
        assert len(ordinary["compose_steps"]) == len(inputs) == len(cached["compose_steps"])
        for item, plain, fast in zip(inputs, ordinary["compose_steps"], cached["compose_steps"]):
            current = item.get("rhs", current)
            assert (plain["status"], plain["reason"], plain["work"]) == (fast["status"], fast["reason"], fast["work"])
            assert plain["status"] == "proved", (name, plain, fast)
            wanted = isl.map(isl_text(item["left"])).then(isl.map(isl_text(current)))
            assert isl.map(isl_text(plain["relation"])).equal(wanted), (name, plain)
            assert isl.map(isl_text(fast["relation"])).equal(wanted), (name, fast)
        return ordinary, cached

    plain, cached = cached_composition("prepared_wildcard_and_rhs_reset", diagonal, [
        {"left": edges([(0,0)])}, {"left": even}, {"left": diagonal},
        {"left": diagonal, "rhs": shift}, {"left": shift},
        {"left": diagonal, "rhs": even}, {"left": shift}])
    assert plain["composition_index_builds"] == 7, plain
    assert cached["composition_index_builds"] == 3, cached
    cached_composition("prepared_guarded_invocations", guarded_edges([(0,1,0),(0,1,1)]), [
        {"left": guarded_edges([(0,0,0)])},
        {"left": guarded_edges([(0,0,1)])},
        {"left": guarded_edges([(0,0,0),(0,0,1)])}])
    for name, left, right, budget, outcome in (
        ("empty", edges([]), diagonal, 8000000, "proved"),
        ("exhausted", even, diagonal, 0, "budget-exhausted"),
        ("incompatible", diagonal, target_before, 8000000, "unsupported")):
        ordinary = run("prepared_" + name + "_ordinary", {"op":"compose", "a":left, "b":right,
            "budget":budget}, status=outcome)
        cached = run("prepared_" + name + "_cached", {"op":"compose", "a":left, "b":right,
            "budget":budget, "prepared":True}, status=outcome)
        assert ordinary == cached, (name, ordinary, cached)

    preprocessing = []
    for size in (8,32,128):
        handoffs = edges([(2*i,2*i+1) for i in range(size)])
        answer = run(f"source_scope_preprocessing_{size}", {
            "op":"completion", "a":handoffs, "issue_order":edges([(i,i) for i in range(2*size)]),
            "lazy":True, "record_steps":True,
            "needs":[{"relation":edges([(2*i,2*i+1)])} for i in range(size)]})
        assert all(x["status"] == "proved" for x in answer["answers"]), answer
        # One full primitive index plus one one-piece target order per query;
        # one primitive endpoint set plus one reached endpoint per source scope.
        assert answer["composition_index_builds"] == size + 1, answer
        assert answer["composition_index_pieces"] == 2 * size, answer
        assert answer["endpoint_projections"] == size + 1, answer
        assert answer["endpoint_projection_pieces"] == 2 * size, answer
        for i, step in enumerate(answer["steps"]):
            assert isl.map(isl_text(step["supply"])).equal(isl.map(isl_text(edges([(2*i,2*i+1)])))), step
        preprocessing.append({"independent_sources":size, "index_builds":answer["composition_index_builds"],
            "index_piece_visits":answer["composition_index_pieces"],
            "endpoint_projections":answer["endpoint_projections"],
            "projection_input_pieces":answer["endpoint_projection_pieces"], "work":answer["work"]})
    (args.output / "preprocessing_scaling.json").write_text(json.dumps(preprocessing, indent=2) + "\n")

    # Common rows are tested only after local witnesses have been aligned.
    # Check exact integer difference independently, including the sign change
    # allowed for equalities and the sign change forbidden for inequalities.
    def common_difference(name, left, right, minimum_skipped=0):
        expected = isl.map(isl_text(left)) - isl.map(isl_text(right))
        answer = run(name, {"op":"subtract", "a":left, "b":right}, str(expected))
        assert answer["difference_common_rows"] >= minimum_skipped, answer
        return answer
    common_left = relation(d=2,r=2,s=1,
        eq=[[1,0,0,0,0,-7],[0,0,1,0,0,-11],[0,1,0,-1,0,0]],
        ge=[[0,1,0,0,0,0],[0,-1,0,0,1,-1]])
    common_right = relation(d=2,r=2,s=1,
        eq=[[-x for x in row] for row in common_left["pieces"][0]["eq"]],
        ge=common_left["pieces"][0]["ge"]+[[0,1,0,0,0,-4]])
    common_difference("difference_common_guarded_invocation_rows", common_left, common_right, 3)
    common_difference("difference_all_rows_already_hold", common_left, common_left, 3)
    common_difference("difference_opposite_inequality_is_not_common",
        relation(eq=[[1,-1,0]],ge=[[1,0,0]]),
        relation(eq=[[-1,1,0]],ge=[[-1,0,0]]), 1)
    common_difference("difference_total_floor_not_tightened_membership", quotient_eq, tightened)
    common_difference("difference_shifted_floor_keeps_original_witness", even, odd)
    common_difference("difference_union_updates_prefix_rows", common_left, relation(d=2,r=2,s=1,
        pieces=common_right["pieces"]+[{
            "locals":0, "eq":common_left["pieces"][0]["eq"],
            "ge":common_left["pieces"][0]["ge"]+[[0,-1,0,0,0,1]]}]), 3)

    # Endpoint indexing is only a necessary fixed-coordinate filter. Unknown
    # source/target coordinates retain wildcard buckets, invocation coordinates
    # still participate in the final comparison, and source/sink dimensions are
    # never conflated for Presburger sets.
    indexed_cases = [
        ("supply_unknown_source", edges([(2,5)]), relation(eq=[[0,1,-5]])),
        ("supply_unknown_target", edges([(2,5)]), relation(eq=[[1,0,-2]])),
        ("supply_unknown_both", edges([(2,5)]), diagonal),
        ("requirement_unknown_source", relation(eq=[[0,1,-5]]), edges([(2,5),(3,5)])),
        ("requirement_unknown_target", relation(eq=[[1,0,-2]]), edges([(2,5),(2,6)])),
        ("requirement_unknown_both", diagonal, edges([(2,2),(3,3)])),
        ("guard_symbols_remain_exact", guarded_edges([(2,5,0)]), guarded_edges([(2,5,1)])),
        ("other_invocation_coordinate", relation(d=2,r=2,
            eq=[[1,0,0,0,-2],[0,1,0,0,-1],[0,0,1,0,-5],[0,0,0,1,-1]]),
            relation(d=2,r=2, eq=[[1,0,0,0,-2],[0,1,0,0,0],[0,0,1,0,-5],[0,0,0,1,0]])),
        ("set_has_no_source_coordinate", relation(d=0,eq=[[1,-2]]), relation(d=0,eq=[[1,-3]])),
        ("domain_has_no_target_coordinate", relation(r=0,eq=[[1,-2]]), relation(r=0,eq=[[1,-3]])),
    ]
    for name, requirement, supply in indexed_cases:
        run("indexed_difference_" + name, {"op":"subtract", "a":requirement, "b":supply},
            str(isl.map(isl_text(requirement)) - isl.map(isl_text(supply))))
        contains_case("indexed_contains_" + name, supply, requirement)
    # A wildcard family preceding a fixed-key family must retain that priority.
    # The eighth compatible failed query still exhausts the same cheap attempt
    # population, then the complete exact subtraction proves the union.
    priorities = relation(pieces=[relation(eq=[[1,-1,j]])["pieces"][0] for j in range(1,10)] +
                         edges([(7,7)])["pieces"])
    priority = run("indexed_contains_preserves_candidate_budget_order", {
        "op":"contains", "a":priorities, "b":edges([(7,7)])})
    assert priority["containment_endpoint_comparisons"] == 9, priority
    assert priority["difference_endpoint_comparisons"] == 10, priority

    endpoint_population_scaling = []
    for size in (32,128,512):
        left = edges([(i,0) for i in range(size)])
        unrelated = edges([(i+size,0) for i in range(size)])
        disjoint = run(f"indexed_difference_disjoint_population_{size}",
            {"op":"subtract", "a":left, "b":unrelated}, str(isl.map(isl_text(left))))
        matched = run(f"indexed_difference_matched_population_{size}",
            {"op":"subtract", "a":left, "b":left}, "{[d0] -> [r0]: false}")
        supplied = relation(pieces=unrelated["pieces"]+list(reversed(left["pieces"])))
        covered = run(f"indexed_contains_matched_population_{size}",
            {"op":"contains", "a":supplied, "b":left})
        assert disjoint["difference_endpoint_comparisons"] == 0, disjoint
        assert disjoint["relation_endpoint_index_pieces"] == size, disjoint
        assert matched["difference_endpoint_comparisons"] == size, matched
        assert matched["relation_endpoint_index_pieces"] == size, matched
        assert covered["containment_endpoint_comparisons"] == size, covered
        assert covered["relation_endpoint_index_pieces"] == 2*size, covered
        assert covered["difference_endpoint_comparisons"] == 0, covered
        for answer in (disjoint,matched,covered):
            assert answer["relation_endpoint_bucket_lookups"] == 4*size, answer
        endpoint_population_scaling.append({"required_pieces":size,
            "difference_disjoint_comparisons":disjoint["difference_endpoint_comparisons"],
            "difference_matched_comparisons":matched["difference_endpoint_comparisons"],
            "contains_matched_comparisons":covered["containment_endpoint_comparisons"],
            "contains_index_input_pieces":covered["relation_endpoint_index_pieces"],
            "bucket_lookups":covered["relation_endpoint_bucket_lookups"]})
    (args.output / "endpoint_population_scaling.json").write_text(json.dumps(endpoint_population_scaling,indent=2)+"\n")

    # Emptiness-only unit substitution must not change the returned relation's
    # source/range/symbol coordinates or erase nonunit divisibility. All these
    # differences include correlated occurrence coordinates and symbolic bounds.
    for modulus in (2,3,4):
        for shift in (-1,0,1):
            for residue in (0,1):
                eqs = [[1,0,0,0,0,0,0,-7], [0,0,-1,0,0,0,0,11],
                       [0,1,0,-1,0,0,0,shift],
                       [0,1,0,0,0,-modulus,0,-residue],
                       [0,0,0,0,1,0,-2,-1]]
                bounds = [[0,1,0,0,0,0,0,2], [0,-1,0,0,1,0,0,-1],
                          [0,0,0,0,1,0,0,-1], [0,0,0,0,-1,0,0,7]]
                source = relation(d=2,r=2,s=1,locals=2,eq=eqs,ge=bounds)
                kept = relation(d=2,r=2,s=1,locals=2,eq=eqs,
                    ge=bounds+[[0,1,0,0,0,0,0,-1]])
                run(f"unit_emptiness_keeps_occurrence_scope_{modulus}_{shift}_{residue}",
                    {"op":"subtract","a":source,"b":kept},
                    str(isl.map(isl_text(source))-isl.map(isl_text(kept))))
    # No unit coefficient in 2*x=1: the integer-empty equality cannot be
    # rationally substituted away. A fixed parameter is still part of output.
    nonunit_impossible = relation(d=2,r=1,s=1,
        eq=[[1,0,0,0,-4],[0,2,0,0,-1],[0,0,-1,0,8],[0,0,0,1,-3]])
    nonunit_possible = relation(d=2,r=1,s=1,
        eq=[[1,0,0,0,-4],[0,2,0,0,-2],[0,0,-1,0,8],[0,0,0,1,-3]])
    for name,left,right in (("nonunit_empty",nonunit_impossible,nonunit_possible),
                            ("nonunit_nonempty",nonunit_possible,nonunit_impossible)):
        run("unit_emptiness_"+name,{"op":"subtract","a":left,"b":right},
            str(isl.map(isl_text(left))-isl.map(isl_text(right))))

    check_periodic_successors(run, isl, args.output)

    summary = {"status":"passed", "checks":results, "isl_version":isl.version,
               "driver_sha256":hashlib.sha256(args.driver.read_bytes()).hexdigest()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(results)} native MLIR/reference relation checks passed")


if __name__ == "__main__":
    main()
