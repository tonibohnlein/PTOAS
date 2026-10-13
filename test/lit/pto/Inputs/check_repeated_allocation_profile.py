# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Allocation inventories are opt-in, nested, and absent for logical-only requests."""
from collections import Counter
from pathlib import Path
import re
import shutil
import subprocess
import sys

from check_frontier_stage_profile import check_stages, require, split_report


def compile_plan(tool, source, profile, allocation):
    flags = f"profile-stages={str(profile).lower()} prepare-allocation={str(allocation).lower()}"
    return subprocess.run([tool, "--mlir-disable-threading", "--pto-frontier-analysis=" + flags, "-"],
                          input=source, capture_output=True, text=True, check=False, timeout=90)


def same_boolean_order(left, right):
    # The existing late compactor can order i1 AND operands by allocation
    # address. Check their complete operand multisets and every other IR line;
    # do not ignore guards, certificates, commands, or arithmetic differences.
    names = {}
    definition = re.compile(r"^\s*(%[\w.$-]+) = arith\.andi (%[\w.$-]+), (%[\w.$-]+) : i1$")
    token = re.compile(r"%[\w.$-]+")

    def normalize(source):
        scopes, result = [(-1, {})], []
        for line in source.splitlines():
            indent = len(line) - len(line.lstrip())
            while scopes[-1][0] > indent:
                scopes.pop()
            if scopes[-1][0] < indent:
                scopes.append((indent, dict(scopes[-1][1])))
            operands = scopes[-1][1]
            match = definition.fullmatch(line)
            if match:
                value, a, b = match.groups()
                operands[value] = operands.get(a, Counter({a: 1})) + operands.get(b, Counter({b: 1}))
                continue
            # A sibling structured region can reuse printed SSA names.
            assigned = re.match(r"^\s*(%[\w.$-]+) =", line)
            if assigned:
                operands.pop(assigned.group(1), None)

            def replace(match):
                value = match.group()
                if value not in operands:
                    return value
                key = tuple(sorted(operands[value].items()))
                return "<and:" + str(names.setdefault(key, len(names))) + ">"

            result.append(token.sub(replace, line))
        return result

    return normalize(left) == normalize(right)


def main():
    require(len(sys.argv) == 3, "expected optimizer and repeated fixture")
    tool = shutil.which(sys.argv[1])
    require(tool is not None, "optimizer unavailable")
    source = Path(sys.argv[2]).read_text(encoding="utf-8")
    # Retain the nested, guarded repetition while keeping the finite child small.
    source = source.replace("%c32 = arith.constant 32", "%c32 = arith.constant 2")
    source = source.replace("%c4 = arith.constant 4", "%c4 = arith.constant 2")
    for allocation in (False, True):
        plain = compile_plan(tool, source, False, allocation)
        measured = compile_plan(tool, source, True, allocation)
        reports, _, diagnostics = split_report(measured.stderr)
        require(plain.returncode == measured.returncode == 0, plain.stderr + measured.stderr)
        require(same_boolean_order(plain.stdout, measured.stdout) and plain.stderr == diagnostics,
                "profiling changed the plan")
        require(len(reports) == 1, "missing owned profile")
        check_stages(reports[0])
        work = [stage for stage in reports[0]["stages"] if stage["stage"] == "repeated-allocation-work"]
        if not allocation:
            require(not work, "logical-only compilation constructed repeated allocation support")
            continue
        require(bool(work), "repeated allocation was not measured")
        require(any(stage["work"]["matrix_pairs"] for stage in work), "fixture did not exercise sharing")
        for stage in work:
            counters = stage["work"]
            require(stage["parent"] > 0 and stage["method"], "unowned repeated-region inventory")
            require(stage["elapsed_ns"] == stage["exclusive_ns"] == 0, "inventory double-counted summary time")
            for request, computation in (("relative_queries", "relative_computations"),
                                         ("implication_requests", "implication_computations"),
                                         ("constant_requests", "constant_computations")):
                require(0 <= counters[computation] <= counters[request], "invalid work counters")
            if counters["implication_computations"]:
                require(counters["proof_node_visits"] > 0, "proof traversal was not measured")
    print("repeated allocation profile: unchanged plans and logical-only exports; owned work counters")


if __name__ == "__main__":
    main()
