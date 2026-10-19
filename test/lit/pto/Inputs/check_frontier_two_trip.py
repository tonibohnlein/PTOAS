# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check bounded pre-analysis unrolling, carried values, order and ID reuse."""
import json
from pathlib import Path
import re
import sys

from check_deferred_family_output import executable, invoke
from check_finite_allocation import physical_check
from check_finite_guarded_analysis import required
from check_logical_insertion import closure_with_commands


def trace(checker, text, trips):
    text = re.sub(r"test.trace_arguments = array<i64: [^>]+>",
                  f"test.trace_existing, test.trace_arguments = array<i64: {trips}>", text)
    report = json.loads(invoke([checker, "--structured-trace", "-"], text).stdout)
    assert report["accepted"] and not report["trace"]["error"], report
    return report["trace"]


def main():
    optimizer, checker, fixture = sys.argv[1:]
    optimizer, checker = map(executable, (optimizer, checker))
    source = Path(fixture).read_text(encoding="utf-8")
    logical = invoke([optimizer, "--pto-frontier-analysis=normalize-two-trip-loops=true", "-"], source).stdout
    unchanged_coordinates = invoke(
        [optimizer, "--pto-frontier-analysis=normalize-two-trip-loops=false", "-"], source).stdout
    assert len(re.findall(r"^\s*(?:%[^=]+ = )?scf\.for\b", unchanged_coordinates, re.MULTILINE)) == 2
    assert logical.count("scf.for") == 1
    assert "WRONG_RESULT" not in logical
    assert "conservative-completion" not in logical
    physical = invoke([optimizer, "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5", "-"], logical).stdout
    effects = {label: (set(), {"x"}) for label in ("FIRST", "SECOND", "RESULT")}
    effects["READ"] = ({"x"}, {"acc"})
    for trips in (0, 1, 2, 5):
        original, planned, allocated = [trace(checker, text, trips)
                                        for text in (source, logical, physical)]
        payloads = [event for event in planned["events"] if event["kind"] == "payload"]
        labels = [event["label"] for event in payloads]
        assert labels == [event["label"] for event in original["events"]
                          if event["kind"] == "payload"]
        assert labels == ["FIRST", "READ", "SECOND", "READ", "RESULT"] * trips
        commands = []
        for event in planned["events"]:
            if event["kind"] == "payload":
                continue
            command = dict(event)
            if event["kind"] != "barrier":
                command["identity"] = (event["plan"], event["record"], event["source_ordinal"],
                                       tuple(event.get("members", [])))
            commands.append(command)
        actual = closure_with_commands([event["pipe"] for event in payloads], commands)
        expected = required(payloads, effects)
        # Demand generation and native event semantics are independent of the
        # unroller. Require equality, including completion/start asymmetry.
        assert all(row == needed & ~(1 << i) for i, (row, needed)
                   in enumerate(zip(actual, expected)))
        physical_check({"accepted": True, "allocated": True,
                        "trace": planned, "physical": allocated}, set(range(6)))
    # Scalar-only probes isolate normalization from synchronization analysis.
    def scalar(bound, step=1, start=0, nested=False):
        inner = """%r = scf.for %i = %lo to %hi step %step iter_args(%v = %lo) -> index {
          %next = arith.addi %v, %step : index
          scf.yield %next : index
        }"""
        if nested:
            inner = """%r = scf.for %outer = %lo to %hi step %step iter_args(%out = %lo) -> index {
              """ + inner.replace("%r =", "%inner =").replace("%v = %lo", "%v = %out") + """
              scf.yield %inner : index
            }"""
        return f"""module attributes {{pto.target_arch = "a3"}} {{
          func.func @scalar(%n: index) -> index {{
            %lo = arith.constant {start} : index
            %hi = arith.constant {bound} : index
            %step = arith.constant {step} : index
            {inner}
            return %r : index
          }}
        }}"""
    for bound, step, start, expected_loops in ((2, 1, 0, 0), (3, 2, 0, 0),
                                               (0, 1, -2, 1), (0, 1, 0, 1),
                                               (3, 1, 0, 1)):
        result = invoke([optimizer, "--pto-frontier-analysis=normalize-two-trip-loops=true", "-"],
                        scalar(bound, step, start)).stdout
        assert result.count("scf.for") == expected_loops, result
    dynamic = scalar(2).replace("to %hi step", "to %n step")
    narrow = scalar(2).replace("index", "i32").replace(
        "iter_args(%v = %lo) -> i32 {", "iter_args(%v = %lo) -> i32 : i32 {")
    for text in (dynamic, narrow, scalar(9223372036854775807, 4611686018427387904)):
        result = invoke([optimizer, "--pto-frontier-analysis=normalize-two-trip-loops=true", "-"], text).stdout
        assert result.count("scf.for") == 1
    nested = invoke([optimizer, "--pto-frontier-analysis=normalize-two-trip-loops=true", "-"],
                    scalar(2, nested=True)).stdout
    assert nested.count("scf.for") == 1  # Do not recursively expand newly exposed parents.
    flags = ("pto.set_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]\n"
             "pto.wait_flag[<PIPE_MTE1>, <PIPE_M>, <EVENT_ID0>]\n")
    manual = scalar(2).replace("return %r", flags + "return %r")
    result = invoke([optimizer, "--pto-frontier-analysis=normalize-two-trip-loops=true", "-"], manual).stdout
    assert result.count("scf.for") == 1  # Existing synchronized code remains owned by its producer.
    print("two-trip normalization: payloads, iter_args, exact event order, zero/short visits, "
          "six-ID reuse, nesting and manual synchronization checked")


if __name__ == "__main__":
    main()
