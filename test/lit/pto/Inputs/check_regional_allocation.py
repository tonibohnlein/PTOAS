# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Validate composed cyclic IDs against command reachability, including loop boundaries."""
import json
from pathlib import Path
import subprocess
import re
import sys
import tempfile

from check_finite_allocation import physical_check
from check_frontier_public_pipeline import unchanged_failure
from check_sequence_analysis import validate
from check_finite_guarded_analysis import load, compute, render


def invoke(tool, arguments, path):
    return subprocess.run([tool, *arguments, str(path)], capture_output=True, text=True, timeout=90)


def field_parts(text):
    """Keep nested certificates and typed arrays inside their owning field."""
    fields, stack, start = [], [], 0
    quoted = escaped = False
    closing = {"}": "{", "]": "[", ">": "<", ")": "("}
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "{[<(":
            stack.append(char)
        elif char in "}]>)":
            assert stack and stack.pop() == closing[char], "unbalanced allocation field"
        elif char == "," and not stack:
            fields.append(text[start:index].strip())
            start = index + 1
    assert not stack and not quoted, "unterminated allocation field"
    fields.append(text[start:].strip())
    result = {}
    for field in fields:
        name, separator, value = field.partition(" = ")
        assert name and name not in result, "duplicate allocation field"
        result[name] = value if separator else None
    return result


def finite_certificate(text):
    markers = list(re.finditer(r"\bpto\.finite_allocation\s*=\s*\{", text))
    assert len(markers) == 1, "expected one owning finite certificate"
    start = markers[0].end() - 1
    quoted = escaped = False
    depth = 0
    for position in range(start, len(text)):
        char = text[position]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, position + 1, field_parts(text[start + 1:position])
    raise AssertionError("unterminated owning finite certificate")


def replace_certificate(text, fields):
    start, end, _ = finite_certificate(text)
    contents = ", ".join(name if value is None else name + " = " + value for name, value in fields.items())
    return text[:start] + "{" + contents + "}" + text[end:]


def check_primary_corruptions(opt, path, logical, flags):
    _, _, fields = finite_certificate(logical)
    assert fields["strategy"] == '"regional-palettes"'
    primary = {name: value for name, value in fields.items()
               if name not in ("certified_candidates", "regular_candidate")}
    isolated = replace_certificate(logical, primary)
    path.write_text(isolated)
    baseline = invoke(opt, flags, path)
    assert baseline.returncode == 0, baseline.stderr
    assert "pto.logical_" not in baseline.stdout
    # A semantically invalid optional candidate does not invalidate its valid
    # primary. Verify this boundary separately instead of treating it as a
    # decoder failure or corrupting whichever printed version appears first.
    if "certified_candidates" in fields:
        alternatives = fields["certified_candidates"]
        assert "version = 2 : i64" in alternatives
        damaged = dict(fields)
        damaged["certified_candidates"] = alternatives.replace("version = 2 : i64", "version = 1 : i64", 1)
        path.write_text(replace_certificate(logical, damaged))
        fallback = invoke(opt, flags, path)
        assert fallback.returncode == 0 and "pto.logical_" not in fallback.stdout, fallback.stderr
    for section, old, new in [
            ("version", "2 : i64", "1 : i64"),
            ("groups", "budget = 2 : i64", "budget = 0 : i64"),
            ("groups", "conflicts = array<i64>", "conflicts = array<i64: 999999>"),
            ("groups", "phases = array<i64: 0>", "phases = array<i64: -1>"),
            ("groups", "strides = array<i64: 1>", "strides = array<i64: -1>"),
            ("groups", "strides = array<i64: 1>", "strides = array<i64>"),
            ("lane_evidence", "local = 1 : i64", "local = 0 : i64"),
            ("lane_evidence", "successors = array<i64>", "successors = array<i64: 999999>"),
            ("groups", "sources = array<i64: 2>", "sources = array<i64: 2, 3>")]:
        assert old in primary[section], (section, old)
        broken = dict(primary)
        broken[section] = primary[section].replace(old, new, 1)
        refusal = unchanged_failure(opt, replace_certificate(isolated, broken), flags)
        assert "error:" in refusal.stderr, (section, old, refusal.stderr)


def main():
    tool, opt, fixture = sys.argv[1:]
    source = Path(fixture).read_text()
    start = source.index("    scf.for")
    end = source.index("    }", start) + len("    }\n")
    second = source[start:end].replace("%i", "%j").replace("%n", "%m")
    two = (source[:end] + second + source[end:]).replace("%n: index)", "%n: index, %m: index)")
    cases = [(source, [n]) for n in [-2, 0, 1, 2, 3, 5, 9]]
    cases += [(two, [n, m]) for n, m in [(0, 0), (0, 3), (4, 0), (1, 1), (2, 3), (5, 4)]]
    with tempfile.TemporaryDirectory(prefix="regional-allocation-") as directory:
        path = Path(directory) / "case.pto"
        for text, trips in cases:
            text = text.replace("array<i64: 5>", "array<i64: " + ", ".join(map(str, trips)) + ">")
            text = text.replace("test.trace_arguments =",
                                "test.eligible_ids = array<i64: 0, 1, 2, 3, 4, 5>, test.trace_arguments =")
            path.write_text(text)
            run = invoke(tool, ["--structured-trace"], path)
            assert run.returncode == 0, run.stderr
            report = json.loads(run.stdout)
            validate(report, trips)
            physical_check(report, set(range(6)))
        # L->D is retained only when the optional C is absent; endpoint
        # presence alone misses that it is exclusive with L->C. One ID suffices
        # because the regional adapter simplifies the integer rank guards.
        guarded = (Path(fixture).parent / "sync_finite_guarded_analysis.pto").read_text()
        body = load("L") + "\nscf.if %g {\n" + compute("C") + "\n}\n" + compute("D")
        for guard in [0, 1]:
            text = render(guarded, body, guard, 0).replace(
                "test.trace_arguments =", "test.eligible_ids = array<i64: 0>, test.trace_arguments =")
            path.write_text(text)
            run = invoke(tool, ["--structured-trace"], path)
            assert run.returncode == 0, run.stderr
            physical_check(json.loads(run.stdout), {0})
        path.write_text(source)
        logical = invoke(opt, ["--mlir-disable-threading", "--pto-frontier-analysis"], path)
        assert logical.returncode == 0, logical.stderr
        assert 'strategy = "regional-palettes"' in logical.stdout
        path.write_text(logical.stdout)
        args = ["--mlir-disable-threading", "--pto-frontier-allocate=eligible-ids=0,3,5"]
        physical = invoke(opt, args, path)
        assert physical.returncode == 0, physical.stderr
        loops = sum(line.strip().startswith("scf.for") for line in physical.stdout.splitlines())
        assert "pto.logical_" not in physical.stdout and loops == 1
        # Damage the selected primary, never a printed optional alternative.
        check_primary_corruptions(opt, path, logical.stdout, args)
        path.write_text(source.replace("test.trace_arguments =",
                                       "test.eligible_ids = array<i64: 0>, test.trace_arguments ="))
        report = json.loads(invoke(tool, ["--structured-trace"], path).stdout)
        assert report["accepted"] and not report["allocated"] and report["allocation_unchanged_on_failure"]
    print(f"regional allocation: {len(cases) + 2} independent causal-reuse traces; "
          "sparse IDs and failure checks passed")


if __name__ == "__main__":
    main()
