# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Native shared-requirement placement and independent concrete prefix checks."""
import json
from pathlib import Path
import re
import subprocess
import sys

binary, source, helpers, prefix = sys.argv[1:]
sys.path.insert(0, helpers)
from compare_boundaries import run as trace
from run_qwen_additions import analyze

original = Path(source).read_text()
loop_match = re.search(r"(?m)^( +)scf.for (%[\w]+) = (%[\w]+) to (%[\w]+) step (%[\w]+) \{", original)
assert loop_match, "expected the unchanged QK loop"
indent, iv, zero, upper, step = loop_match.groups()
body_begin = original.index("\n", loop_match.start()) + 1
body_end = original.index("\n" + indent + "}", body_begin)
body = original[body_begin:body_end]
loads = re.findall(r"(?m)^.*pto.tload[^\n]+", original)
assert len(loads) == 4
panels = [re.search(r"outs\((%[\w]+)", line)[1] for line in loads[:2]]
first_extract = next(line for line in body.splitlines() if "pto.textract" in line)
# Same producer/read/reuse relationships, different names and surrounding guards.
renamed = original.replace("@qk_matmul", "@independent_preloads")
additional = original.replace(first_extract, first_extract.replace("ins(" + panels[0] + ",",
                                                                   "ins(" + panels[1] + ",") + "\n" + first_extract, 1)
allocation = re.search(re.escape(panels[1]) + r" = pto.alloc_tile addr = (%[\w]+)", original)
address = allocation[1]
overlap = re.sub(re.escape(address) + r" = arith.constant 4096 : i64",
                 address + " = arith.constant 0 : i64", original)
assert overlap != original

def guarded(predicate, reverse):
    condition = indent + "  %readers = arith.cmpi " + predicate + ", %arg3, " + zero + " : index\n"
    guarded_body = "\n".join("  " + line for line in body.splitlines())
    inner = indent + "  scf.if %readers {\n"
    if reverse:
        inner += indent + "  } else {\n"
    inner += guarded_body + "\n" + indent + "  }"
    return original[:body_begin] + condition + inner + original[body_end:]

variants = {"unchanged": original, "renamed": renamed, "additional_reader": additional,
            "overlap": overlap, "skipped_eq": guarded("eq", False), "skipped_ne": guarded("ne", True)}
two = re.search(r"(%[\w]+) = arith.constant 2 : index", original)[1]
start = original.index(loads[0])
end = re.search(r"(?m)^" + indent + "return", original).start()
outer_body = "\n".join("  " + line for line in original[start:end].splitlines())
variants["outer_invocations"] = (original[:start] + indent + f"scf.for %outer = {zero} to {two} step {step} {{\n"
                                 + outer_body + "\n" + indent + "}\n" + original[end:])
outputs, reports = {}, {}
for name, text in variants.items():
    path = Path(prefix + "." + name + ".input.pto")
    output = Path(prefix + "." + name + ".output.pto")
    path.write_text(text)
    command = [binary, str(path), "--mlir-disable-threading", "--mlir-print-op-on-diagnostic=false",
               "--pto-insert-sync=buffer-generations=true defer-same-pipe=true "
               "mmad-chains=true gm-alias=assume-disjoint-arguments", "-o", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, (name, completed.stderr)
    Path(prefix + "." + name + ".stderr").write_text(completed.stderr)
    before, after = analyze(path), analyze(output)
    for key in ("payload", "allocations", "views", "abi"):
        assert before[key] == after[key], (name, key)
    count = re.search(r"generation_publications_advanced = (\d+)", output.read_text())
    advanced = int(count[1]) if count else 0
    if name != "outer_invocations":
        assert advanced == (1 if name in ("unchanged", "renamed") else 0), (name, advanced, completed.stderr)
    outputs[name] = output
    reports[name] = {"advanced": advanced, "static": after["mechanisms"], "executions": []}
    for trips in (-1, 0, 1, 2, 3, 16):
        for selector in ((0, 1) if name.startswith("skipped") else (0,)):
            scenario = {"arguments": ["Q", "OUT", "K", selector, trips, 0, 24]}
            observed, metrics = trace(output, scenario)
            assert metrics["counts"]["pto.set_flag"] == metrics["counts"]["pto.wait_flag"]
            # Fresh concrete token replay must retire both outside preloads even
            # when the loop is empty or every reader is skipped.
            retired = max([observed.drained.get("PIPE_MTE2", -1)] +
                          [v.get("PIPE_MTE2", -1) for v in observed.completed.values()])
            assert retired >= observed.issued["PIPE_MTE2"], (name, trips, selector, "undrained producer")
            extracts = [i for i,payload in enumerate(observed.payload) if payload[0] == "pto.textract"]
            if extracts:
                first = extracts[0]
                available = observed.before[first]["completed"].get("PIPE_MTE2", -1)
                if name in ("unchanged", "renamed"):
                    assert first == 3 and available == 0, (name, trips, available)
                    # Q1's consumers still acquire Q1, via the independent K
                    # publication. Four extracts precede the first Q1 extract.
                    assert observed.before[extracts[4]]["completed"].get("PIPE_MTE2", -1) >= 1
                elif name in ("overlap", "additional_reader"):
                    assert available >= 1, (name, trips, "lost actual Q1 requirement")
            reports[name]["executions"].append({"trips": trips, "selector": selector,
                "sets": metrics["counts"]["pto.set_flag"], "waits": metrics["counts"]["pto.wait_flag"],
                "scalar": metrics["scalar_counts"], "scalar_steps": metrics["scalar_steps"]})

# Break the emitted empty-path transfer, without changing the payload. These
# checks must fail independently of the compiler's output status attributes.
text = outputs["unchanged"].read_text()
empty = re.search(r"(%[\w]+) = arith.cmpi sge, [^\n]+\n\s*scf.if \1 \{\n\s*(pto.set_flag[^\n]+)", text)
assert empty, text
for name, broken, trips in (
        ("missing_empty_publication", text[:empty.start(2)] + text[empty.end(2):], 0),
        ("double_publication", text.replace("arith.cmpi sge,", "arith.cmpi slt,", 1), 1)):
    path = Path(prefix + "." + name + ".pto")
    path.write_text(broken)
    try:
        trace(path, {"arguments": ["Q", "OUT", "K", 0, trips, 0, 24]})
    except ValueError as error:
        assert "publication" in str(error) or "overwritten" in str(error), str(error)
    else:
        raise AssertionError(name + " was not detected")
Path(prefix + ".report.json").write_text(json.dumps(reports, indent=2) + "\n")
print("Shared publications: unchanged QK improved; rename, overlap, extra reader, equivalent guards, "
      "empty/skipped recurrence and broken token transfers checked")
