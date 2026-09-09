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
