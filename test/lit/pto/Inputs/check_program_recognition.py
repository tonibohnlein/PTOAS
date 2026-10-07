# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Shared scalar declarations make finite guarded effects exact. Rotating
# recognition consumes the same exact symbolic maps.
"""Check structure, context and shared-record identities independently of JSON layout."""
import json
from pathlib import Path
import shutil
import subprocess
import sys


def check_common(document):
    nodes = document["nodes"]
    payloads = document["payloads"]
    guards = document["guards"]
    assert not document["analysis_ready"]
    assert nodes[0]["parent"] == -1
    assert nodes[0]["payload_count"] == len(payloads)
    owned = []
    for node in nodes:
        assert node["payload_count"] == len(node["payloads"]) + sum(
            nodes[child]["payload_count"] for child in node["children"])
        for child in node["children"]:
            assert nodes[child]["parent"] == node["id"]
        for payload in node["payloads"]:
            assert payloads[payload]["node"] == node["id"]
            owned.append(payload)
        if node["guard"] >= 0:
            assert node["guard"] < len(guards)
        if node["kind"] == "loop":
            assert all(key in node for key in ("lower", "upper", "step", "induction"))
    assert sorted(owned) == list(range(len(payloads)))
    effects = [effect["id"] for payload in payloads for effect in payload["effects"]]
    assert len(effects) == len(set(effects))
    assert document["arithmetic"]["scope"] == "whole-function"


def routes(document, name):
    return [attempt for node in document["nodes"] for attempt in node["attempts"]
            if attempt["route"] == name]


def main():
    tool = shutil.which(sys.argv[1])
    if tool is None:
        raise RuntimeError("missing diagnostic executable")
    source = Path(sys.argv[2]).resolve()
    before = source.read_bytes()
    run = subprocess.run([tool, "--recognize", str(source)], check=True,
                         text=True, capture_output=True, timeout=60)
    assert source.read_bytes() == before
    documents = {value["function"]: value for line in run.stdout.splitlines()
                 if line.startswith("{") for value in [json.loads(line)]}
    assert documents
    for value in documents.values():
        check_common(value)
    mode = sys.argv[3] if len(sys.argv) > 3 else "structure"
    if mode == "structure":
        doc = documents["structure"]
        assert len(doc["payloads"]) == 7
        nodes = doc["nodes"]
        root_children = [nodes[i] for i in nodes[0]["children"]]
        assert [n["kind"] for n in root_children] == ["explicit-run", "loop", "explicit-run"]
        assert len(root_children[0]["payloads"]) == 2
        assert len(root_children[2]["payloads"]) == 1
        assert max(len(n["loops"]) for n in nodes) == 2
        assert sorted(a["state"] for a in routes(doc, "rotating")) == ["applicable", "not-applicable"]
        assert any(g["available_before_loops"] == [False] for g in doc["guards"])
        assert any(g["available_before_loops"] == [True] for g in doc["guards"])
        assert any(n.get("arm") == "else" and not n["children"] for n in nodes)
        assert doc["arithmetic"]["state"] == "applicable"
        assert doc["arithmetic"]["class"] == "differences"
        assert not documents["empty"]["payloads"]
        assert any(a["state"] == "applicable" for a in routes(documents["fixed_storage_loop"], "rotating"))
    elif mode == "guarded":
        for name in ("nested_arms", "invariant_loop", "varying_loop"):
            assert documents[name]["arithmetic"]["state"] == "applicable"
        late_arithmetic = documents["late_finite"]["arithmetic"]
        assert late_arithmetic["state"] == "not-applicable"
        assert {issue["issue"] for issue in late_arithmetic["issues"]} == {
            "unsupported-control"}
        assert any(a["state"] == "applicable" for a in routes(documents["nested_arms"], "finite-guarded"))
        assert any(a["state"] == "applicable" and a["entry_guards_available"]
                   for a in routes(documents["invariant_loop"], "guarded-rotating"))
        assert any(any(i["issue"] == "guard-invariance" for i in a["issues"])
                   for a in routes(documents["varying_loop"], "guarded-rotating"))
        assert any(not a["entry_guards_available"] for a in routes(documents["late_finite"], "finite-guarded"))
    elif mode == "arithmetic":
        assert documents["triangular"]["arithmetic"]["state"] == "applicable"
        assert documents["ssa_prerequisite"]["arithmetic"]["state"] == "applicable"
        assert documents["opaque_bound"]["arithmetic"]["state"] == "applicable"
    else:
        raise ValueError("unknown check mode")
    print("program recognition checks passed:", mode)


if __name__ == "__main__":
    main()
