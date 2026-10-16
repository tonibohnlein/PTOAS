# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Whole-invocation zero-ID proofs must preserve original IR and emit nothing."""
import json
from pathlib import Path
import re
import shutil
import sys

from check_frontier_public_pipeline import run, unchanged_failure
from check_logical_insertion import without_function_metadata


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def bounded(source, lower, upper):
    original = "    scf.for %t = %one to %n step %two {"
    replacement = (f"    %empty_lower = arith.constant {lower} : index\n"
                   f"    %empty_upper = arith.constant {upper} : index\n"
                   "    scf.for %t = %empty_lower to %empty_upper step %two {")
    return replace_once(source, original, replacement)


def assert_no_commands(text):
    command = r"\bpto\.(?:logical_set|logical_wait|set_flag(?:_dyn)?|wait_flag(?:_dyn)?|barrier)\b"
    assert not re.search(command, text), text


def empty_case(optimizer, tracer, program):
    baseline = run(optimizer, program, []).stdout
    logical = run(optimizer, program, ["--pto-frontier-analysis"]).stdout
    assert 'kind = "finite"' in logical and "handoffs = []" in logical, logical
    assert "pto.cyclic_allocation" not in logical and "pto.endpoint_families" not in logical
    split = run(optimizer, logical, ["--pto-frontier-allocate"]).stdout
    direct = run(optimizer, program, ["--pto-frontier-analysis=physical-output=true"]).stdout
    for output in (split, direct):
        assert "pto.finite_allocation" not in output and "pto.endpoint_families" not in output, output
    for name, output in (("logical", logical), ("split", split), ("direct", direct)):
        assert_no_commands(output)
        assert without_function_metadata(output) == without_function_metadata(baseline), name
        # Bind observation to the function, leaving module attributes unchanged.
        pattern = r"(func\.func @two_slots\([^\n]+\) attributes \{)"
        observed, count = re.subn(pattern, r"\1test.trace_existing, ", output)
        assert count == 1
        report = json.loads(run(tracer, observed, ["--structured-trace"]).stdout)
        assert report["accepted"] and not report["trace"]["error"], report
        assert report["trace"]["payloads"] == 0 and report["trace"]["events"] == [], report


def main():
    optimizer, tracer = (shutil.which(name) for name in sys.argv[1:3])
    assert optimizer and tracer
    optimizer, tracer = (str(Path(name).resolve(strict=True)) for name in (optimizer, tracer))
    source = Path(sys.argv[3]).read_text()
    source = replace_once(source, "test.eligible_ids = array<i64: 1, 3>", "test.eligible_ids = array<i64>")
    for lower, upper in ((1, 1), (1, 0), (-1, -1), (-1, -2),
                         ((1 << 63) - 1, (1 << 63) - 1), (-(1 << 63), -(1 << 63))):
        empty_case(optimizer, tracer, bounded(source, lower, upper))
    # Dynamic bounds cannot use a static empty proof, even when a trace happens
    # to bind an empty value. Nonempty constants and surrounding payloads also
    # retain their ordinary allocation obligations.
    dynamic_empty_trace = replace_once(source, "array<i64: 8>", "array<i64: 1>")
    surrounding = bounded(source, 1, 1)
    prefix = ("    pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%tile0 : !tile)\n"
              "    pto.textract ins(%tile0, %zero, %zero : !tile, index, index) outs(%left0 : !left)\n")
    surrounding = replace_once(surrounding, "    %empty_lower =", prefix + "    %empty_lower =")
    with_barrier = replace_once(surrounding, "    %empty_lower =",
                                "    pto.barrier #pto.pipe<PIPE_V>\n    %empty_lower =")
    for program in (source, dynamic_empty_trace, bounded(source, 1, 4), surrounding, with_barrier):
        logical = run(optimizer, program, ["--pto-frontier-analysis"]).stdout
        unchanged_failure(optimizer, logical, ["--pto-frontier-allocate"])
    # Explicit existing protocols retain their public manual-skip behavior; an
    # empty original loop must not erase these executing SET/WAIT commands.
    commands = ("    pto.set_flag [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_V>, #pto.event<EVENT_ID0>]\n"
                "    pto.wait_flag [#pto.pipe<PIPE_MTE2>, #pto.pipe<PIPE_V>, #pto.event<EVENT_ID0>]\n")
    manual = replace_once(bounded(source, 1, 1), "    %empty_lower =", commands + "    %empty_lower =")
    baseline = run(optimizer, manual, []).stdout
    for flags in (["--pto-frontier-analysis", "--pto-frontier-allocate"],
                  ["--pto-frontier-analysis=physical-output=true"]):
        output = run(optimizer, manual, flags).stdout
        assert without_function_metadata(output) == without_function_metadata(baseline)
        assert len(re.findall(r"\bpto\.set_flag\s*\[", output)) == 1
        assert len(re.findall(r"\bpto\.wait_flag\s*\[", output)) == 1
    print("empty invocation: six signed domains, zero used IDs, five atomic refusals and manual protocol passed")


if __name__ == "__main__":
    main()
