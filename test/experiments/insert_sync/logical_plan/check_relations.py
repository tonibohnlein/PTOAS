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
sys.path.insert(0, str(HERE.parent / "event_model"))
from eventlab.islwrap import ISL  # noqa: E402


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
    summary = {"status":"passed", "checks":results, "isl_version":isl.version,
               "driver_sha256":hashlib.sha256(args.driver.read_bytes()).hexdigest()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{len(results)} native MLIR/reference relation checks passed")


if __name__ == "__main__":
    main()
