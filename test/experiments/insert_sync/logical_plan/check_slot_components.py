#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check independent compact/general slot components in one native function.

Component receipts identify the actual physical tables, rather than inferring
representation choice from aggregate counters. A successful strict output also
receives concrete access-order, payload, event and retirement checks. Explicit
discovery-only mode tests selection without claiming emitted effectiveness.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from compare_boundaries import Boundaries
from measure import children, fingerprint, replay
from observations import project

HERE = Path(__file__).resolve().parent


def mixed_source():
    template = (HERE / "inputs/slots/slot_guard_binding.pto").read_text()
    body = template[template.index("    %n ="):template.rindex("    return")]

    def component(prefix, slots, outputs, trips=4, reversed_table=False):
        result = body.replace("array<i64: 0, 512>", f"array<i64: {slots[0]}, {slots[1]}>")
        result = result.replace("%n = arith.constant 4 : index", f"%n = arith.constant {trips} : index")
        result = result.replace("4096 : i64", f"{outputs[0]} : i64")
        result = result.replace("4608 : i64", f"{outputs[1]} : i64")
        if reversed_table:
            allocation = next(line for line in result.splitlines() if "%slots = pto.alloc_multi_tile" in line)
            other = allocation.replace("%slots =", "%other_slots =").replace(
                f"array<i64: {slots[0]}, {slots[1]}>", f"array<i64: {slots[1]}, {slots[0]}>")
            result = result.replace(allocation, allocation + "\n" + other)
            result = result.replace("%b = pto.multi_tile_get %slots", "%b = pto.multi_tile_get %other_slots")
        return re.sub(r"%([A-Za-z_][A-Za-z_0-9]*)",
                      lambda match: "%src" if match[1] == "src" else "%" + prefix + match[1], result)

    return '''module attributes {pto.target_arch = "a3"} {
  func.func @mixed_slot_components(%src: !pto.ptr<f16>) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
''' + component("h_", [8192, 8704], [16384, 16896]) + component(
        # One reversed-table iteration exercises genuine physical overlap and
        # the general representation without claiming recurring allocation for
        # that currently unsupported component. The compact component repeats.
        "r_", [0, 512], [20480, 20992], trips=1, reversed_table=True) + "    return\n  }\n}\n"


def walk(op):
    yield op
    for child in children(op):
        yield from walk(child)


class ConcreteAccesses:
    def __init__(self, function):
        self.phases = [op for op in walk(function) if op.name in {"pto.tload", "pto.tabs"}]
        assert [op.name for op in self.phases] == ["pto.tload", "pto.tload", "pto.tabs", "pto.tabs"] * 2
        self.ids = {op: index for index, op in enumerate(self.phases)}
        self.allocations, self.events = {}, []

    def observe(self, op, point, signature):
        if op.name == "pto.alloc_tile":
            assert str(op.results[0].type) == "!pto.tile_buf<vec, 16x16xf16>"
            base = signature[1][0]
            assert isinstance(base, int)
            self.allocations[fingerprint(signature)] = {"scope": "vec", "begin": base, "bytes": 512}
        if op not in self.ids:
            return
        values = [value for value, operand in zip(signature[1], op.operands)
                  if "!pto.tile_buf<vec," in str(operand.type)]
        regions = [value["local_tile"] if isinstance(value, dict) else self.allocations[value] for value in values]
        assert all(region["scope"] == "vec" and region["bytes"] == 512 for region in regions)
        load = op.name == "pto.tload"
        assert len(regions) == (1 if load else 2)
        self.events.append({"phase": self.ids[op], "lane": "PIPE_MTE2" if load else "PIPE_V",
                            "reads": [] if load else regions[:-1], "writes": regions[-1:]})


def inspect_emitted(source, emitted, ir, pto):
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        modules = [ir.Module.parse(text) for text in (source, emitted)]
        projections = [project(module) for module in modules]
        for key in ("payload", "allocations", "views", "abi"):
            assert projections[0][key] == projections[1][key], key
        functions = [next(op for op in children(module.operation) if op.name == "func.func") for module in modules]
        original = ConcreteAccesses(functions[0])
        original_metric = replay(functions[0], ["source"], observer=original.observe)
        observer = Boundaries()
        actual_metric = replay(functions[1], ["source"], observer=observer.observe)
        assert original_metric["payload_sha256"] == actual_metric["payload_sha256"]
        assert len(original.events) == len(observer.before) == 20
        assert Counter(event["phase"] for event in original.events) == Counter({
            phase: 4 if phase < 4 else 1 for phase in range(8)})
        assert not observer.tokens
        conflicts = Counter()
        for target, current in enumerate(original.events):
            for source_index, previous in enumerate(original.events[:target]):
                for kind, read, write in (("RAW", "writes", "reads"), ("WAR", "reads", "writes"), ("WAW", "writes", "writes")):
                    for a in previous[read]:
                        for b in current[write]:
                            if max(a["begin"], b["begin"]) >= min(a["begin"] + a["bytes"], b["begin"] + b["bytes"]):
                                continue
                            assert (previous["phase"] < 4) == (current["phase"] < 4), "components overlap physically"
                            conflicts[kind] += 1
                            assert observer.before[target]["completed"].get(previous["lane"], -1) >= source_index, (
                                "missing concrete local ordering", kind, source_index, target)
        assert all(conflicts[kind] > 0 for kind in ("RAW", "WAR", "WAW"))
        for lane, last in observer.issued.items():
            assert observer.drained.get(lane, -1) >= last, ("missing retirement", lane, last)
        return {"payload_occurrences": len(original.events), "conflicts": dict(conflicts),
                "commands": actual_metric["counts"], "scalar_steps": actual_metric["scalar_steps"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discovery-only", action="store_true",
                        help="Accept checked component selection after a controlled refusal; no effectiveness claim")
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Slot-component acceptance requires assertions")
    args.output.mkdir(parents=True, exist_ok=False)
    source = mixed_source()
    path = args.output / "mixed_slot_components.pto"
    path.write_text(source)
    sidecar = args.output / "discovery.json"
    driver_hash = hashlib.sha256(args.driver.read_bytes()).hexdigest()
    command = [str(args.driver.resolve()), str(path.resolve()), "retirement", "--discovery-output=" + str(sidecar.resolve())]
    env = dict(os.environ, PTOAS_LOGICAL_TRACE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    start = time.monotonic()
    try:
        process = subprocess.run(command, text=True, capture_output=True, env=env, timeout=90)
    except subprocess.TimeoutExpired as error:
        for name, value in (("stdout.json", error.stdout), ("stderr", error.stderr)):
            (args.output / name).write_text(value.decode() if isinstance(value, bytes) else value or "")
        (args.output / "summary.json").write_text(json.dumps({"status": "timeout", "accepted": False,
            "effectiveness": "not established", "command": command, "seconds": time.monotonic() - start,
            "driver_sha256": driver_hash, "source_sha256": hashlib.sha256(source.encode()).hexdigest()}, indent=2) + "\n")
        raise SystemExit(1)
    (args.output / "stdout.json").write_text(process.stdout)
    (args.output / "stderr").write_text(process.stderr)
    assert process.returncode in (0, 1), process.stderr
    answer = json.loads(process.stdout)
    discovery = json.loads(sidecar.read_text())
    assert discovery["discovery_only"] and discovery["export_complete"] and answer["discovery_written"]
    assert len(discovery["phases"]) == 8
    for field in ("accesses", "phases", "requirements", "points", "orders"):
        assert discovery[field] == answer[field], ("discovery changed", field)
    members = [tuple(map(int, match)) for match in re.findall(
        r"logical slot_component_member index (\d+) scope (\d+) base (\d+) slots (\d+) compact ([01])", process.stderr)]
    assert len(members) == 4 and len({member[0] for member in members}) == 4, members
    assert len({member[1] for member in members}) == 1, members
    assert Counter((base, count, compact) for _, _, base, count, compact in members) == Counter({
        (8192, 2, 1): 2, (0, 2, 0): 1, (512, 2, 0): 1}), members
    counts = re.findall(r"logical slot_queries range_builds (\d+) equality_builds (\d+) domain_builds (\d+) geometry_builds (\d+)", process.stderr)
    assert len(counts) == 1, counts
    ranges, equalities, domains, geometries = map(int, counts[0])
    assert ranges > 0 and equalities > 0 and domains > 0 and geometries > 0, counts
    summary = {"component_selection_passed": True, "component_members": members,
               "strict_status": answer["status"], "reason": answer["reason"], "accepted": False,
               "command": command, "seconds": time.monotonic() - start, "driver_sha256": driver_hash,
               "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
               "query_counts": {"range": ranges, "equality": equalities, "slot_domain": domains, "geometry": geometries}}
    if answer["applied"]:
        assert process.returncode == 0 and answer["status"] == "applied" and answer["invoked"]
        sys.path.insert(0, str(args.python_root.resolve()))
        from ptoas.mlir import ir
        from ptoas.mlir.dialects import pto
        emitted = 'module attributes {pto.target_arch = "a3"} {\n' + answer["emitted_ir"] + '\n}'
        (args.output / "logical.pto").write_text(emitted)
        summary["observations"] = inspect_emitted(source, emitted, ir, pto)
        summary.update(status="passed", accepted=True, effectiveness="strict construction and emitted checks passed")
    else:
        assert process.returncode == 1 and answer["original_preserved"] and not answer["emitted_ir"]
        assert answer["status"] in {"unsupported", "analysis-limit", "allocation-failure", "unproved"}, answer
        summary.update(status="discovery-only", effectiveness="not established; controlled constructor refusal")
    assert hashlib.sha256(args.driver.read_bytes()).hexdigest() == driver_hash, "driver changed during test"
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Slot components:", summary["status"], "with homogeneous compact and reversed general tables")
    if not summary["accepted"] and not args.discovery_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
