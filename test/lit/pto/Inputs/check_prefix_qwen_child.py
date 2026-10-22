# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare an actual retained symbolic K recipe with its imported commands."""

import argparse
import json
from pathlib import Path

from assess_whole_kernel_access import annotate, inventory
from check_closed_child_logic import child_phases, identity
from check_prefix_children import BASE, checked, profile, run, traced
from compact_vector_oracle import command_order
from pypto_region_oracle import interpret
from reaching_access_research import require
from survey_pypto_regions import specialize


def symbolic_child(args):
    attrs = "test.contract_math_only, test.contract_region = 5 : i64, "
    original = args.input.read_text()
    source = args.output / "child-probe.pto"
    source.write_text(annotate(original, attrs +
        "test.contract_completion_probe, test.contract_completion_bound"))
    probe = run([args.trace_tool, "--compact-region-contract", source], args.output / "child-probe.json")
    data = json.loads(Path(probe["stdout"]).read_text())
    require(data["closed_child_qualified"] and data["physical_certificate_decoded"], "original child not qualified")
    source = args.output / "child-logical.input.pto"
    source.write_text(annotate(original, attrs + "test.contract_completion_emit"))
    emitted = run([args.trace_tool, "--compact-region-contract", source], args.output / "child-logical.pto")
    return Path(emitted["stdout"]).read_text(), {r["record"]: r for r in data["physical_assignment"]}, probe


def child_agreement(args, original, logical, actual, work, decoded, values, path):
    reference = specialize(original, values)
    expected = traced(args, specialize(logical, values), path)
    palettes = {(p["source"], p["target"]): p["id_mask"] for p in work["child_palettes"]}
    selected = [e for e in actual["events"] if e["kind"] in ("set", "wait")
                and palettes.get((e["source_pipe"], e["target_pipe"]), 0) & (1 << e["physical_id"])]
    wanted = [e for e in expected["events"] if e["kind"] in ("set", "wait")]
    require(len(selected) == len(wanted), "child endpoint activity changed")
    pending, seen = set(), set()
    for old, new in zip(wanted, selected):
        require(all(old[k] == new[k] for k in ("kind", "gap", "source_pipe", "target_pipe")),
                "child command order, guard or endpoint changed")
        key = identity(old)
        if old["kind"] == "set":
            require(key not in seen, "child invocation identity repeated")
            seen.add(key)
            pending.add(key)
        else:
            require(key in pending, "unpublished/cross-invocation child WAIT")
            pending.remove(key)
        record = decoded[old["members"][0]]
        rule = record.get("tuple")
        coords = [old["source_ordinal"], *old["members"][1:]]
        if rule:
            offset = rule["base"] + sum(scale*((stride*coords[c]+phase) % modulus)
                                       for c, stride, phase, modulus, scale in rule["terms"])
        else:
            offset = (record["stride"]*old["source_ordinal"] + record["phase"]) % len(record["ids"])
        require(new["physical_id"] == record["ids"][offset], "child assignment changed")
    require(not pending, "child invocation has open notifications")
    # Compare all four event combinations within each whole child visit.
    phases = child_phases(reference)
    payloads = interpret(reference)
    indices = [i for i, item in enumerate(payloads) if item["phase"] in phases]
    before, _ = command_order(expected, include_commands=True)
    after, _ = command_order(actual, include_commands=True)
    comparisons = 0
    for a in indices:
        for b in indices:
            if payloads[a]["coordinates"][0] != payloads[b]["coordinates"][0]:
                continue
            for p in (0, 1):
                for q in (0, 1):
                    require(bool(before[2*a+p] >> (2*b+q) & 1) == bool(after[2*a+p] >> (2*b+q) & 1),
                            "wrapper changed child internal order")
                    comparisons += 1
    return dict(executed_endpoints=len(selected), original_assignment="identical",
                matching="closed-with-original-invocation-identities", event_comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace-tool", "optimizer", "input", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before_run = run([args.optimizer, "--pto-frontier-analysis="+BASE, args.input], args.output/"before.pto")
    after_run = run([args.optimizer, "--pto-frontier-analysis="+BASE+" prefix-child-region=5", args.input],
                    args.output/"after.pto")
    work = profile(after_run)["work"]
    palettes = {(p["source"], p["target"]): p["id_mask"] for p in work["child_palettes"]}
    for role in work["reserved_role_ids"]:
        require(not palettes.get((role["source"], role["target"]), 0) & (1 << role["id"]),
                "parent/child palette collision")
    logical, decoded, probe = symbolic_child(args)
    rows = []
    for visits, block, k in ((0, 0, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2),
                             (1, 19, 3), (1, 19, 4), (1, 19, 17), (2, 0, 2)):
        values = {"%arg3":20*visits, "%arg4":k, "%arg5":0, "%arg6":0,
                  "%arg7":17408, "%arg8":block, "%arg9":20}
        folder = args.output/f"visits{visits}-block{block}-K{k}"
        folder.mkdir(exist_ok=True)
        old, before, _ = checked(args, args.input.read_text(), Path(before_run["stdout"]).read_text(),
                                  values, folder/"before.pto")
        new, after, word = checked(args, args.input.read_text(), Path(after_run["stdout"]).read_text(),
                                   values, folder/"after.pto")
        child = child_agreement(args, args.input.read_text(), logical, word, work, decoded, values, folder/"child.pto")
        scope = ("structural reset stress; caller/layout applicability unqualified" if visits > 1 else
                 "pinned original caller" if (visits, block, k) == (1, 19, 17) else
                 "bounded zero/short in-range layout derivative")
        rows.append(dict(visits=visits, block=block, K=k, scope=scope, before=old, after=new, child=child,
                         removed_ci=len(before-after), added_ci=len(after-before)))
    receipt = dict(before=before_run, after=after_run, work=work, original_child=probe, cases=rows,
                   before_size=inventory(Path(before_run["stdout"])), after_size=inventory(Path(after_run["stdout"])))
    (args.output/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print("Qwen prefix child: original symbolic endpoints, guards, assignment and internal order preserved")


if __name__ == "__main__":
    main()
