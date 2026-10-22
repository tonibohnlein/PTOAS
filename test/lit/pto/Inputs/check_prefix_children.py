# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent emitted-graph checks for closed children in prefix parents.

Original layout operands supply access requirements. Compiler demand records,
cell partitions and generators are never inputs to this reference. Physical
causality is checked on the actual emitted assignment, including parent roles.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from assess_whole_kernel_access import annotate, digest, invoke
from compact_vector_oracle import command_order
from pypto_region_oracle import interpret, mathematical
from reaching_access_research import require
from survey_pypto_regions import specialize


BASE = "conservative-prefix=true physical-output=true gm-alias=may-not-alias"


def run(command, output, *, expected=None):
    row = invoke([str(v) for v in command], output, 35)
    error = Path(row["stderr"]).read_text()
    if expected:
        require(row["status"] == "failure" and expected in error, str(row) + error)
    else:
        require(row["status"] == "success", str(row) + error)
    return row


def traced(args, text, path):
    path.write_text(annotate(text, "test.trace_existing"))
    row = run([args.trace_tool, "--structured-trace", path], path.with_suffix(".json"))
    result = json.loads(Path(row["stdout"]).read_text())
    require(result.get("accepted") and not result["trace"].get("error"), str(result))
    return result["trace"]


def checked(args, original, emitted, values, path):
    reference = interpret(specialize(original, values))
    required, _ = mathematical(reference)
    word = traced(args, specialize(emitted, values), path)
    payloads = [e for e in word["events"] if e["kind"] == "payload"]
    require([(e["type"], e["coordinates"], e["pipe"]) for e in payloads]
            == [(e["phase"], e["coordinates"], e["pipe"]) for e in reference],
            "payload/control sequence changed")
    reach, reuse = command_order(word, include_commands=True)
    require(not any(row >> i & 1 for i, row in enumerate(reach)), "command cycle")
    commands = [e for e in word["events"] if e["kind"] != "payload"]
    if any(e["kind"] == "set" for e in commands):
        require(reuse["status"] == "passed", str(reuse))
    wanted = {(a, b) for a in range(len(payloads)) for b in range(len(payloads))
              if required[2*a+1] >> (2*b) & 1}
    actual = {(a, b) for a in range(len(payloads)) for b in range(len(payloads))
              if reach[2*a+1] >> (2*b) & 1}
    require(not wanted - actual, "missing concrete prerequisites: " + str(sorted(wanted-actual)[:5]))
    row = dict(payloads=len(payloads), commands=dict(Counter(e["kind"] for e in commands)),
               excess_ci=len(actual-wanted), missing_ci=0, acyclic=True,
               matching="passed", causal_reuse=reuse)
    return row, actual, word


def profile(run_result):
    text = Path(run_result["stderr"]).read_text()
    return json.loads(next(line.removeprefix("frontier-prefix-result ") for line in text.splitlines()
                           if line.startswith("frontier-prefix-result ")))


def small_sources(fixture):
    original = fixture.read_text().replace(" {test.grouped_visit}", "")
    # The supplied child still has two independent banks and a dynamic label.
    # Outer/replay counts and the enclosing skip are varied after preparation.
    original = original.replace(
        "%arg1: index)", "%arg1: index, %arg2: index, %arg3: index)")
    before = original.index("  scf.for %outer")
    after = original.rindex("  return")
    body = original[before:after]
    body = body.replace(
        "   %context =", "   %take = arith.cmpi ne, %arg2, %z : index\n   scf.if %take {\n   %context =")
    # Close the skip before the ordinary post-child access, which must observe
    # pending effects even when the invocation's notifications were closed.
    marker = "   pto.textract ins(%mat0"
    body = body.replace(marker, "   }\n" + marker)
    body = "  scf.for %reset = %z to %arg3 step %one {\n" + body + "  }\n"
    original = original[:before] + body + original[after:]
    request = original.replace("   scf.for %j =", "   scf.for %j =")
    # The marker belongs only to the test provider; it supplies a known valid
    # assignment rather than altering any production recognition premise.
    child_end = request.index("   }\n   }\n   pto.textract")
    request = request[:child_end] + request[child_end:].replace("   }", "   } {test.grouped_visit}", 1)
    return original, request


def small_fixture(args):
    original, request = small_sources(args.fixture)
    source = args.output / "small.symbolic.pto"
    source.write_text(annotate(request,
        "test.contract_math_only, test.contract_region = 5 : i64, test.contract_prefix_parent"))
    canonical = run([args.trace_tool, "--roundtrip", source], args.output / "small.canonical.pto")
    original = Path(canonical["stdout"]).read_text().replace(" {test.grouped_visit}", "")
    candidate = run([args.trace_tool, "--compact-region-contract", source], args.output / "small.candidate.pto")
    candidate_text = Path(candidate["stdout"]).read_text()
    baseline = run([args.optimizer, "--pto-frontier-analysis=" + BASE, source], args.output / "small.baseline.pto")
    baseline_text = Path(baseline["stdout"]).read_text()
    work = profile(candidate)
    require(work["child_closed_notifications"] and not work["child_payload_completion"], "pending effects lost")
    cases = []
    for visits, resets, take in ((0, 2, 1), (1, 1, 1), (3, 1, 1), (2, 3, 1),
                                (2, 2, 0), (0, 0, 0)):
        values = {"%arg1": visits, "%arg2": take, "%arg3": resets}
        folder = args.output / f"small-{visits}-{resets}-{take}"
        folder.mkdir(exist_ok=True)
        old, before, _ = checked(args, original, baseline_text, values, folder / "before.pto")
        new, after, word = checked(args, original, candidate_text, values, folder / "after.pto")
        child_commands = [e for e in word["events"] if e["kind"] in ("set", "wait")
                          and (e["source_pipe"], e["target_pipe"]) == (4, 3)
                          and e["physical_id"] in (0, 1)]
        require(sum(e["kind"] == "set" for e in child_commands) == 2*visits*resets*take,
                "child publications changed")
        require(sum(e["kind"] == "wait" for e in child_commands) == 2*visits*resets*take,
                "child matching changed")
        cases.append(dict(visits=visits, resets=resets, take=take, before=old, after=new,
                          removed_ci=len(before-after), added_ci=len(after-before)))
    negatives = []
    for label, attrs, expected in (
        ("palette-collision", "test.prefix_pool = array<i64: 0, 1>", "resource-reservation"),
        ("bad-arity", "test.bad_child_arity", "closed child qualification unavailable"),
        ("guard-escape", "test.prefix_guard_escape", "closed child qualification unavailable"),
        ("alias-contract", "test.prefix_alias_mismatch", "child access-model alias policy differs from parent"),
        ("missing-target-direction", "", "modeled pipe is unavailable"),
    ):
        failed_source = args.output / (label + ".pto")
        failed_text = source.read_text()
        if label == "missing-target-direction":
            failed_text = failed_text.replace("#pto.kernel_kind<cube>", "#pto.kernel_kind<vector>")
        failed_source.write_text(annotate(failed_text, attrs) if attrs else failed_text)
        refused = run([args.trace_tool, "--compact-region-contract", failed_source],
                      args.output / (label + ".out"), expected=expected)
        require("prefix child refusal preserved original IR" in Path(refused["stderr"]).read_text(),
                "missing transactional refusal check")
        negatives.append(refused)
    return dict(candidate=candidate, baseline=baseline, work=work, cases=cases, negatives=negatives)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace-tool", "optimizer", "fixture", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(tools={str(tool): digest(tool) for tool in (args.trace_tool, args.optimizer)},
                  watchdog_seconds=35, small=small_fixture(args))
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print("prefix child: original accesses, pending effects, matching, reset and causal reuse passed")


if __name__ == "__main__":
    main()
