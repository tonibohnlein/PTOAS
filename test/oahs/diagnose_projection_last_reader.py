# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Diagnose one blocked final-reader boundary in the original down_proj plan.

This supplies a controlled endpoint mutation, not a compiler implementation or
an unbounded protocol proof. All payloads, fences and dynamic event counts stay
fixed. Refuse a changed fixture rather than guessing a different event/site.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from check_projection_trace import run


def payload_order(trace):
    vertices = [v for i, p in enumerate(trace.payloads) for v in (trace.starts[i], p[1])]
    return {(i, j) for i, a in enumerate(vertices) for j, b in enumerate(vertices)
            if trace.ancestors[b] & (1 << a)}


def diagnose(path, out):
    out.mkdir(parents=True, exist_ok=True)
    source = path.read_text()
    release = "        pto.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]\n"
    anchors = [line for line in source.splitlines(True)
               if "pto.textract ins(%16, %c0, %39 " in line]
    assert len(anchors) == 1 and source.count(release) == 1, "changed A0 source/release fixture"
    assert "%c256 = arith.constant 256 : index" in source
    assert "%c128 = arith.constant 128 : index" in source
    assert "scf.for %arg12 = %c0 to %c256 step %c128" in source
    predicate = "%releaseA0_final = arith.cmpi eq, %arg12, %c128 : index"
    endpoint = "          " + predicate + "\n          scf.if %releaseA0_final {\n    " + release + "          }\n"
    early = source.replace(release, "").replace(anchors[0], anchors[0] + endpoint)
    early_path = out / "early.pto"
    early_path.write_text(early)
    def payloads(text):
        return [line for line in text.splitlines()
                if re.match(r"\s*pto\.(?:tload|textract|tmatmul|tstore)\b", line)]
    assert payloads(source) == payloads(early), "payload mutation"
    rows = []
    for chunks in (0, 1, 2, 3, 4, 17):
        for tiles in (0, 1, 2):
            before = run(path, chunks, tiles, True, True)
            after = run(early_path, chunks, tiles, True, True)
            assert [(p[0], p[2], p[3]) for p in before.payloads] == [(p[0], p[2], p[3]) for p in after.payloads]
            assert before.sync_counts == after.sync_counts, "changed executed synchronization population"
            a, b = payload_order(before), payload_order(after)
            assert not b-a, "mutation adds payload ordering"
            rows.append(dict(chunks=chunks, tiles=tiles, payloads=len(before.payloads),
                             local_conflicts=before.required_checks, before=len(a), after=len(b),
                             removed=len(a-b), added=len(b-a)))
    assert sum(row["removed"] for row in rows) > 0, "opportunity is not discriminating"
    # Identify the actual removed edge, not only a smaller relation count.
    before = run(path, 4, 1, True, True)
    after = run(early_path, 4, 1, True, True)
    removed = payload_order(before)-payload_order(after)
    assert removed == {(26,56), (26,57), (27,56), (27,57)}
    assert before.payloads[13][0] == "textract" and before.payloads[28][0] == "tload"
    assert (("mat", 65536, 131072), False) in before.payloads[13][2]
    assert (("mat", 0, 65536), True) in before.payloads[28][2]
    witness = dict(source_visit=13, source=before.payloads[13], destination_visit=28,
                   destination=before.payloads[28], removed=sorted(removed))
    mutations = {
        "first_visit": (early.replace(predicate, predicate.replace("%c128", "%c0")), "missing local completion"),
        "missing_release": (early.replace("    " + release, ""), "unseeded wait"),
        "every_visit": (early.replace(predicate, predicate.replace("%c128", "%arg12")), "occupied publication"),
        "unit_distance": (early.replace(predicate,
            "%last_distance = arith.subi %c256, %arg12 : index\n          "
            "%releaseA0_final = arith.cmpi sle, %last_distance, %c1 : index"), "unseeded wait"),
    }
    negatives = {}
    for name, (text, reason) in mutations.items():
        mutated = out / (name + ".pto")
        mutated.write_text(text)
        try:
            run(mutated, 4, 1, True, True)
        except AssertionError as error:
            assert reason in str(error), (name, error)
            negatives[name] = str(error)
        else:
            raise AssertionError("negative unexpectedly passed: " + name)
    report = dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  scope="finite local-effect/event graph; supplied endpoint mutation, not compiler output",
                  rows=rows, witness=witness, negatives=negatives)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(paths=len(rows), removed=sum(r["removed"] for r in rows), added=0,
                         local_conflicts=sum(r["local_conflicts"] for r in rows),
                         negative_cases=len(negatives)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.plan, args.out)
