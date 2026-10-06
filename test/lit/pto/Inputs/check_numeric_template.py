# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate generic numeric templates and their preserved occurrence identities."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


def candidate(document):
    candidates = [attempt for node in document["nodes"] if node["kind"] == "loop" and not node["loops"]
                  for attempt in node["attempts"] if attempt["route"] == "numeric-template"]
    assert len(candidates) == 1
    return candidates[0]


def byte_set(ranges):
    return {(r["space"], byte) for r in ranges for byte in range(r["begin"], r["end"])}


def coordinates(payload):
    return tuple(item["value"] for item in payload["coordinates"])


def check_endpoints(template):
    plan = template["logical_endpoints"]
    assert not plan["error"] and plan["selection_recipes_ready"]
    assert not plan["physical_ids_ready"] and not plan["ir_emitted"]
    anchors = plan["anchors"]
    assert len(anchors) == len(template["payloads"])
    static = {}
    for anchor, payload in zip(anchors, template["payloads"]):
        assert anchor["phase"] == payload["phase"] and anchor["coordinates"] == payload["coordinates"]
        cuts = anchor["before_cut"], anchor["after_cut"]
        assert cuts[0] != cuts[1]
        assert static.setdefault(anchor["phase"], cuts) == cuts
    groups = plan["groups"]
    assert len({group["cut"] for group in groups}) == len(groups)
    indices = [index for group in groups for index in group["recipes"]]
    assert sorted(indices) == list(range(len(plan["recipes"])))
    for group in groups:
        for index in group["recipes"]:
            recipe = plan["recipes"][index]
            producer = recipe["kind"] == "set"
            anchor = anchors[recipe["source"] if producer else recipe["target"]]
            assert group["cut"] == anchor["after_cut" if producer else "before_cut"]
    # Each actual inner visit matches only its own full tuple, even when many
    # expanded types share one original operation and static insertion point.
    for identity, visit in enumerate(anchors):
        selected = [i for i, anchor in enumerate(anchors)
                    if anchor["phase"] == visit["phase"] and anchor["coordinates"] == visit["coordinates"]]
        assert selected == [identity]


def check_common(document, template):
    assert not document["analysis_ready"] and not template["interfaces_ready"]
    assert template["scope"] == "whole-function"
    assert template["period"] == template["refresh"] == 1
    payloads = template["payloads"]
    check_endpoints(template)
    assert template["counted_payloads"] >= len(payloads)
    assert template["counted_visits"] >= template["counted_payloads"]
    keys = [(p["phase"], coordinates(p)) for p in payloads]
    assert len(keys) == len(set(keys))
    original_effects = {effect["id"] for p in document["payloads"] for effect in p["effects"]}
    atoms = template["atoms"]
    for first, second in zip(atoms, atoms[1:]):
        assert first["space"] < second["space"] or first["end"] <= second["begin"]
    for payload in payloads:
        for effect in payload["effects"]:
            assert effect["source_effect"] in original_effects
            assert effect["maps"]
            if effect["discharge"] == 0:
                assert byte_set(effect["ranges"]) == byte_set([atoms[i] for i in effect["atoms"]])
            else:
                assert not effect["atoms"]


def check_examples(documents, policy):
    counts = {"nested_recurrence": 6, "nested_coordinates": 4, "zero_outer": 0, "inner_then_outer": 3,
              "zero_inner": 1, "branch_selection": 5, "readonly_global": 2,
              "streaming_global": 1, "negative_stream": 1, "local_reuse": 2}
    if policy == "may-not-alias":
        counts["distinct_globals"] = 2
    for name, count in counts.items():
        template = candidate(documents[name])
        assert template["state"] == "applicable", (name, template["issues"])
        assert len(template["payloads"]) == count, name
        check_common(documents[name], template)
    recurrence = candidate(documents["nested_recurrence"])
    assert [coordinates(p) for p in recurrence["payloads"]] == [(i,) for i in range(3) for _ in range(2)]
    for index, payload in enumerate(recurrence["payloads"]):
        effect, = payload["effects"]
        assert effect["mode"] == ("write" if index % 2 == 0 else "read")
        start = 32 * ((index // 2) % 2)
        assert byte_set(effect["ranges"]) == {(6, byte) for byte in range(start, start + 4)}
    recurrence_anchors = recurrence["logical_endpoints"]["anchors"]
    for i in range(0, 6, 2):
        assert recurrence_anchors[i]["after_cut"] == recurrence_anchors[i + 1]["before_cut"]
    assert recurrence_anchors[0]["before_cut"] == recurrence_anchors[2]["before_cut"]
    assert recurrence_anchors[0]["coordinates"] != recurrence_anchors[2]["coordinates"]
    # The inner loop's end and the following outer-body operation are distinct cuts.
    boundary_anchors = candidate(documents["inner_then_outer"])["logical_endpoints"]["anchors"]
    assert boundary_anchors[1]["after_cut"] != boundary_anchors[2]["before_cut"]
    empty = candidate(documents["zero_outer"])
    assert empty["empty_invocation"] and not empty["payloads"] and not empty["atoms"]
    assert not empty["counted_visits"] and not empty["logical_endpoints"]["recipes"]
    nested = candidate(documents["nested_coordinates"])
    assert nested["lower"] == 1 and nested["step"] == 2
    assert [coordinates(p) for p in nested["payloads"]] == [(i, j) for i in (1, 3) for j in (0, 1)]
    branch = candidate(documents["branch_selection"])
    assert [coordinates(p) for p in branch["payloads"]] == [(0,), (0,), (1,), (1,), (2,)]
    assert [p["effects"][0]["mode"] for p in branch["payloads"]] == ["write"] + ["read"] * 4
    reuse = candidate(documents["local_reuse"])
    assert len(reuse["atoms"]) == 1
    reuse_anchors = reuse["logical_endpoints"]["anchors"]
    assert reuse_anchors[0]["after_cut"] != reuse_anchors[1]["before_cut"]
    assert [p["effects"][0]["atoms"] for p in reuse["payloads"]] == [[0], [0]]
    for name, stride in (("streaming_global", 16), ("negative_stream", -16)):
        effect, = candidate(documents[name])["payloads"][0]["effects"]
        assert effect["discharge"] == 2 and effect["outer_stride"] == stride
        assert effect["maps"][0]["base_argument"] == 0
    for payload in candidate(documents["readonly_global"])["payloads"]:
        assert payload["effects"][0]["discharge"] == 1
    # TGETVAL/TSETVAL scalar ordering is native (the shared prerequisite
    # implementation predates this test update), not an admission failure.
    scalar = candidate(documents["extra_prerequisite"])
    assert scalar["state"] == "applicable" and not scalar["analysis"]["retained"]
    rejected = {"parametric_inner": "loop-domain", "outer_guard": "unsupported-control",
                "external_payload": "template-context", "global_reader_writer": "gm-discharge",
                "expanded_stores": "gm-discharge", "unknown_state": "loop-carried-state",
                "expansion_limit": "template-expansion-limit",
                "unknown_nsw_bound": "loop-domain"}
    assert candidate(documents["distinct_globals"])["state"] == "applicable"
    for name, issue in rejected.items():
        template = candidate(documents[name])
        assert template["state"] != "applicable", name
        assert issue in {item["issue"] for item in template["issues"]}, (name, template["issues"])
        assert not template["payloads"] and not template["atoms"]
        assert template["period"] == template["refresh"] == 0


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing recognition executable")
    source = Path(sys.argv[2]).resolve()
    before = source.read_bytes()
    for policy in ("may-not-alias", "may-alias"):
        run = subprocess.run([tool, "--gm-alias=" + policy, "--recognize", str(source)],
                             check=True, capture_output=True, text=True, timeout=60)
        documents = {d["function"]: d for line in run.stdout.splitlines() if line.startswith("{")
                     for d in [json.loads(line)]}
        assert len(documents) == 21
        check_examples(documents, policy)
        rejected = [attempt for node in documents["multiple_outer_loops"]["nodes"]
                    for attempt in node["attempts"] if attempt["route"] == "numeric-template"]
        assert len(rejected) == 3
        for attempt in rejected:
            assert attempt["state"] != "applicable"
            assert [item["issue"] for item in attempt["issues"]] == ["template-context"]
            assert not attempt["payloads"] and not attempt["atoms"]
    assert source.read_bytes() == before
    print("numeric template identities, byte ranges and discharge checks passed")


if __name__ == "__main__":
    main()
