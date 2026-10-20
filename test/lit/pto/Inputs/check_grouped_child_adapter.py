# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate the supplied two-bank closed child through conservative composition."""

import argparse
import json
import os
from pathlib import Path
import resource
import shutil

from assess_whole_kernel_access import annotate, digest, invoke, compile_stage
from compact_vector_oracle import command_order
from check_finite_allocation import physical_check
from pypto_region_oracle import interpret, mathematical
from survey_pypto_regions import specialize


def run_checked(command, output):
    row = invoke(command, output, 35)
    if row["status"] != "success":
        raise AssertionError((row, Path(row["stderr"]).read_text()[-2000:]))
    return row


def trace(args, text, path):
    path.write_text(annotate(text, "test.trace_existing"))
    run = run_checked([args.trace_tool, "--structured-trace", path], path.with_suffix(".json"))
    result = json.loads(Path(run["stdout"]).read_text())
    if not result.get("accepted") or result["trace"].get("error"):
        raise AssertionError(result)
    return result["trace"]


def check(args, logical, physical, original, visits, folder, imported):
    values = {"%arg1": visits}
    # Strip only the diagnostic loop marker; it has no execution semantics.
    items = interpret(specialize(original.replace(" {test.grouped_visit}", ""), values))
    required, _ = mathematical(items)
    word = trace(args, specialize(logical, values), folder / f"logical-{visits}.pto")
    machine = trace(args, specialize(physical, values), folder / f"physical-{visits}.pto")
    payloads = [e for e in word["events"] if e["kind"] == "payload"]
    if [(e["type"], e["coordinates"]) for e in payloads] != [(e["phase"], e["coordinates"]) for e in items]:
        raise AssertionError("Payload sequence changed")
    reach, _ = command_order(word, include_commands=True)
    if any(row >> i & 1 for i, row in enumerate(reach)):
        raise AssertionError("Command cycle")
    missing = [(a,b) for a in range(len(items)) for b in range(len(items))
               if required[2*a+1] >> (2*b) & 1 and not reach[2*a+1] >> (2*b) & 1]
    if missing:
        raise AssertionError(("Concrete coverage", missing))
    physical_check({"accepted":True,"allocated":True,"trace":word,"physical":machine}, set(range(6)))
    if imported:
        if sum(e["kind"] == "set" for e in word["events"]) != 2*visits:
            raise AssertionError("Publication multiplicity")
        for old, new in zip(word["events"], machine["events"]):
            if old["kind"] in ("payload", "barrier"):
                continue
            # The leading finite WAW child owns record 0. The grouped child's
            # records are 1/2; its source ordinal is still the unshifted j=0/1.
            if old["members"][0] != old["source_ordinal"] + 1:
                raise AssertionError("Dynamic record selector was replaced or mistranslated")
            if new["physical_id"] != old["source_ordinal"]:
                raise AssertionError("Namespace translation changed the source-tuple map")
            if len(old["members"]) != 2 or not 0 <= old["members"][1] < visits:
                raise AssertionError("Enclosing invocation coordinate lost")
    ci = {(a,b) for a in range(len(items)) for b in range(len(items)) if reach[2*a+1] >> (2*b) & 1}
    wanted = {(a,b) for a in range(len(items)) for b in range(len(items)) if required[2*a+1] >> (2*b) & 1}
    return {"visits":visits,"payloads":len(items),"commands":len(word["events"])-len(items),
            "matching":"passed","reuse":"passed","excess_ci":len(ci-wanted),"missing_ci":0}, ci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace-tool", "optimizer", "input", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--baseline-tool", type=Path)
    args = parser.parse_args()
    for name in ("trace_tool", "optimizer", "baseline_tool"):
        value = getattr(args,name)
        if value:
            setattr(args,name,Path(shutil.which(str(value)) or value).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    args.timeout = 35
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    original_path = args.output / "canonical-original.pto"
    run_checked([args.trace_tool,"--roundtrip",args.input],original_path)
    receipt = {"input_sha256":digest(args.input),
               "tools":{str(p):digest(p) for p in (args.trace_tool,args.optimizer)},"routes":[]}
    orders = {}
    routes = [("candidate",args.trace_tool,False,True),("bad-arity",args.trace_tool,True,False)]
    if args.baseline_tool:
        routes.append(("baseline",args.baseline_tool,False,False))
    for name, tool, bad, imported in routes:
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        attrs = ('test.contract_math_only, test.contract_region = 5 : i64, '
                 'test.contract_gm_alias = "may-not-alias", test.contract_completion_parent')
        if bad:
            attrs += ", test.bad_child_arity"
        source = folder / "input.pto"
        source.write_text(annotate(args.input.read_text(), attrs))
        logical_path = folder / "logical.pto"
        run = run_checked([tool,"--compact-region-contract",source],logical_path)
        logical = logical_path.read_text()
        if ("test.imported_grouped_child = true" in logical) != imported:
            raise AssertionError("Unexpected import qualification")
        if imported and "test.retained_eventful_children = 1" not in logical:
            raise AssertionError("Parent discarded the qualified child")
        if imported and any(mark not in logical for mark in (
            "test.pending_child_payloads = true", "test.transparent_child = true"
        )):
            raise AssertionError("Closed notifications incorrectly cleared payload effects")
        physical_run = compile_stage(args,folder,"physical",logical_path,
                                     "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5")
        if physical_run["status"] != "success":
            raise AssertionError(physical_run)
        physical = Path(physical_run["output"]).read_text()
        rows=[]
        for visits in (0,1,2,3):
            row, ci = check(args,logical,physical,original_path.read_text(),visits,folder,imported)
            rows.append(row);orders[(name,visits)]=ci
        receipt["routes"].append({"name":name,"run":run,"physical":physical_run,"cases":rows})
    if args.baseline_tool:
        receipt["comparison"] = [
            {"visits":v, "added_ci":len(orders[("candidate",v)]-orders[("baseline",v)]),
             "removed_ci":len(orders[("baseline",v)]-orders[("candidate",v)])}
            for v in (0,1,2,3)]
    (args.output/"results.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("grouped closed child: selectors, ordinal map, invocation scope, coverage, matching and reuse passed")


if __name__ == "__main__":
    main()
