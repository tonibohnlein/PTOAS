# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check one prefill invocation's fixed logical protocol, preserving all payloads.

Block zero, seqlens=[0,4], cache[0]=0, initial[0]=1, original flag reads=1.
Valid initialized storage and the original nonalias premise are assumed. The
bound IR is trace-only; its allocation metadata must never be used for lowering.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from check_fixed_protocol_capacity import (capacity, check_protocol_comparison, feedback_case,
                                           same_fixed_protocol)

BINDINGS = {"block_index": 0, "sequence_start": 0, "sequence_end": 4,
            "cache_index": 0, "initial": 1, "flag_read_0": 1,
            "flag_read_1": 1, "flag_read_2": 1}


def bind_prefill(text, length=4, initial=1):
    assert "test.trace_existing" not in text
    function = re.search(r"func\.func @causal_conv1d_prefill\(([^\n]+)\) attributes \{", text)
    assert function, "expected exact original prefill signature"
    args = re.findall(r"(%\w+)\s*:", function.group(1))
    assert len(args) == 7
    definitions = dict(re.findall(r"^\s*(%\w+) = ([^\n]+)$", text, re.M))
    block = [name for name, operation in definitions.items() if operation == "pto.get_block_idx"]
    assert len(block) == 1
    block_index = [name for name, operation in definitions.items()
                   if operation == f"arith.index_cast {block[0]} : i64 to index"]
    assert len(block_index) == 1
    bindings = {block[0]: (0, "i64", "block_index")}
    loads = re.findall(r"^\s*(%\w+) = pto\.load (%\w+)\[(%\w+)\] : !pto.ptr<i32, gm> -> i32$",
                       text, re.M)
    assert len(loads) == 4
    seen = set()
    for result, pointer, index in loads:
        assert pointer in args[3:6]
        if pointer == args[3]:
            if index == block_index[0]:
                value, key = 0, "sequence_start"
            else:
                addition = re.fullmatch(r"arith.addi (%\w+), (%\w+) : index", definitions[index])
                assert addition and addition.group(1) == block_index[0]
                assert definitions[addition.group(2)] == "arith.constant 1 : index"
                value, key = length, "sequence_end"
        else:
            assert index == block_index[0]
            value, key = (0, "cache_index") if pointer == args[4] else (initial, "initial")
        assert key not in seen
        seen.add(key)
        bindings[result] = (value, "i32", key)
    assert seen == {"sequence_start", "sequence_end", "cache_index", "initial"}
    reads = re.findall(r"^\s*(%\w+) = pto\.tgetval ins\((%\w+), (%\w+) : [^\n]+ outs : i32$",
                       text, re.M)
    assert len(reads) == 3
    assert reads[0][1] == reads[1][1] and reads[0][1] != reads[2][1]
    for ordinal, (result, tile, index) in enumerate(reads):
        assert definitions[index] == "arith.constant 0 : index"
        writes = re.findall(r"pto\.texpands ins\((%\w+) : i32\) outs\(" + re.escape(tile) + r" :", text)
        assert len(writes) == 1 and definitions[writes[0]] == "arith.constant 1 : i32"
        # Other writes to these flag tiles would invalidate this narrow valuation.
        consumers = [line for line in text.splitlines() if re.search(r"\bouts\(" + re.escape(tile) + r"\s*:", line)]
        assert len(consumers) == 1
        bindings[result] = (1, "i32", f"flag_read_{ordinal}")
    rewritten = []
    applied = set()
    for line in text.splitlines():
        match = re.match(r"^(\s*)(%\w+) = (.*)$", line)
        if match and match.group(2) in bindings:
            indent, result, operation = match.groups()
            value, ty, key = bindings[result]
            rewritten.append(f"{indent}%prefill_observation_{result[1:]} = {operation}")
            rewritten.append(f"{indent}{result} = arith.constant {value} : {ty}")
            applied.add(result)
        else:
            rewritten.append(line)
    assert applied == set(bindings)
    output = "\n".join(rewritten) + "\n"
    output, count = re.subn(r"(func\.func @causal_conv1d_prefill\([^\n]+\) attributes \{)",
                            r"\1test.trace_existing, test.trace_visit_limit = 20000000 : i64, ", output)
    assert count == 1
    # Every PTO operation, including the observed loads/getvals, stays in order.
    inventory = lambda body: re.findall(r"\bpto\.[a-zA-Z_]+(?= |\n)", body)
    assert inventory(output) == inventory(text)
    return output, {key: {"ssa": result, "value": value, "type": ty}
                    for result, (value, ty, key) in bindings.items()}


def payloads(report):
    return [event for event in report["trace"]["events"] if event["kind"] == "payload"]


def check(original, logical):
    for report in (original, logical):
        assert report["accepted"] and report["trace"]["error"] == "", "trace not accepted"
    assert all(event["kind"] == "payload" for event in original["trace"]["events"])
    assert len(original["trace"]["events"]) == original["trace"]["payloads"]
    assert payloads(original) == payloads(logical), "original payload trace changed"
    result = capacity(logical)
    assert result["payloads"] == 97 and result["commands"] == 124
    selected = next(row for row in result["directions"] if row["direction"] == [4, 1])
    assert selected["handoffs"] == selected["minimum_ids"] == 20
    assert selected["reuse_edges"] == 0 and len(selected["antichain"]) == 20
    assert selected["minimum_ids"] > 6
    return result


def controls(original, logical):
    feedback_case(False)
    feedback_case(True)
    check_protocol_comparison()
    for damage in ("original-payload", "logical-payload", "unmatched", "duplicate"):
        left, right = deepcopy(original), deepcopy(logical)
        if damage.endswith("payload"):
            target = left if damage == "original-payload" else right
            next(event for event in target["trace"]["events"] if event["kind"] == "payload")["type"] += 1
        elif damage == "unmatched":
            events = right["trace"]["events"]
            events.pop(next(i for i, event in enumerate(events) if event["kind"] == "wait"))
        else:
            events = right["trace"]["events"]
            index = next(i for i, event in enumerate(events) if event["kind"] == "set")
            events.insert(index, dict(events[index]))
        try:
            check(left, right)
        except AssertionError:
            continue
        raise AssertionError("accepted damaged prefill evidence: " + damage)


def replay(source, logical, tracer):
    reports = []
    # Large logical input belongs on disk, not the machine's RAM-backed /tmp.
    with tempfile.TemporaryDirectory(prefix="prefill-capacity-", dir=logical.parent) as scratch:
        for name, path in (("original", source), ("logical", logical)):
            bound, rows = bind_prefill(path.read_text())
            assert {key: row["value"] for key, row in rows.items()} == BINDINGS
            temporary = Path(scratch) / (name + ".pto")
            temporary.write_text(bound)
            run = subprocess.run([tracer, "--structured-trace", str(temporary)], capture_output=True,
                                 text=True, check=False, timeout=180)
            if run.returncode:
                raise RuntimeError(f"{name} replay failed ({run.returncode}): {run.stderr}")
            reports.append(json.loads(run.stdout))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--logical", type=Path)
    parser.add_argument("--tracer")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    root = Path(__file__).resolve().parents[4]
    source = root / fixture["input"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == fixture["input_sha256"]
    assert fixture["bindings"] == BINDINGS
    original, logical = fixture["original_report"], fixture["report"]
    result = check(original, logical)
    # JSON serializes identity tuples as arrays; compare the complete certificate.
    assert json.loads(json.dumps(result)) == fixture["expected"]
    controls(original, logical)
    if bool(args.logical) != bool(args.tracer):
        parser.error("--logical and --tracer must be supplied together")
    if args.logical:
        executable = shutil.which(args.tracer)
        if executable is None:
            parser.error("--tracer must name an available executable")
        executable = str(Path(executable).resolve(strict=True))
        fresh_original, fresh_logical = replay(source, args.logical.resolve(strict=True), executable)
        assert same_fixed_protocol(fresh_original["trace"], original["trace"]), "original execution changed"
        assert same_fixed_protocol(fresh_logical["trace"], logical["trace"]), "fixed logical protocol changed"
        assert check(fresh_original, fresh_logical) == result
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
