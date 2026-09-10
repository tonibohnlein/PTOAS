# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Frozen research observers only; no synchronization planner or compilation arm."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from measure import SYNC, attrs, children, static_metrics
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

SERIAL_DRIVER = """import sys
sys.path.insert(0, sys.argv.pop(1))
from pathlib import Path
from ptoas import _cli
from ptoas.mlir import ir
class SerialContext(ir.Context):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_multithreading(False)
ir.Context = SerialContext
raise SystemExit(_cli.launch(sys.argv[1:], wrapper=Path(_cli.__file__)))
"""

def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def project(module):
    """Exact non-sync structure, attrs, types and SSA wiring (no algebraic rewrites).

Compiler status attrs and the explicitly recorded GM contract are removed.
Synchronization-only branches and comparisons used exclusively by those branches
are recorded separately. Payload control, constants, views and effects remain.
"""
    values, terms, records, placements, allocations, views, abi = {}, {}, [], [], [], [], []
    sync_control = Counter()

    control_arithmetic = {"arith.constant", "arith.cmpi", "arith.subi", "arith.addi", "arith.remsi", "arith.remui", "arith.andi", "arith.ori", "arith.xori"}

    def only_sync(op, local=None):
        if op.name in SYNC or op.name == "scf.yield":
            return not op.results
        if op.name in control_arithmetic and local is not None:
            return all(use.owner in local for value in op.results for use in value.uses)
        if op.name == "scf.if" and not op.results:
            private = set(walk(op)) if local is None else local
            return all(only_sync(view.operation, private) for region in op.regions
                       for block in region.blocks for view in block.operations)
        return False

    def walk(op):
        for child in children(op):
            yield child
            yield from walk(child)

    excluded = {op for op in walk(module.operation) if op.name == "scf.if" and only_sync(op)}
    while True:
        found = {op for op in walk(module.operation) if op not in excluded and op.name in control_arithmetic
                 and op.results and any(value.uses for value in op.results)
                 and all(use.owner in excluded for value in op.results for use in value.uses)}
        if not found:
            break
        excluded.update(found)

    def visit(op, path, guards=()):
        properties = attrs(op)
        if op in excluded:
            sync_control[op.name] += 1
            if op.name in control_arithmetic:
                for i, value in enumerate(op.results):
                    terms[value] = hashed([op.name, properties, [terms[v] for v in op.operands], i])
            else:
                for ri, region in enumerate(op.regions):
                    branch = guards + ((terms[op.operands[0]], ri == 0),)
                    for block in region.blocks:
                        for view in block.operations:
                            child = view.operation
                            if child.name != "scf.yield":
                                visit(child, path, branch)
            return
        if op.name in SYNC:
            placements.append({"point": path, "op": op.name, "attrs": properties, "guards": guards})
            return
        for key in list(properties):
            if key.startswith("pto.insert_sync."):
                del properties[key]
            elif key == "pto.gm_alias":
                if properties[key] != '"assume-disjoint-arguments"':
                    raise ValueError("unexpected GM allocation contract")
                del properties[key]
        record = {"point": path, "op": op.name, "attrs": properties,
                  "operands": [values[v] for v in op.operands],
                  "types": [str(v.type) for v in op.results]}
        operand_terms = [terms[v] for v in op.operands]
        for i, value in enumerate(op.results):
            values[value] = [path, "result", i]
            terms[value] = hashed([op.name, properties, record["types"], operand_terms, i])
        records.append(record)
        if op.name in {"pto.alloc_tile", "pto.alloc", "memref.alloc", "memref.alloca"}:
            allocations.append({**record, "operand_expression_sha256": operand_terms})
        if op.name in {"pto.make_tensor_view", "pto.partition_view", "pto.subview",
                       "pto.multi_tile_get", "pto.bind_tile"}:
            views.append({**record, "operand_expression_sha256": operand_terms})
        if op.name == "func.func":
            abi.append(properties)
        for ri, region in enumerate(op.regions):
            for bi, block in enumerate(region.blocks):
                block_path = path + ["region", ri, "block", bi]
                records.append({"point": block_path, "block_types": [str(v.type) for v in block.arguments]})
                for ai, value in enumerate(block.arguments):
                    values[value] = [block_path, "argument", ai]
                    terms[value] = hashed([values[value], str(value.type)])
                index = 0
                for view in block.operations:
                    child = view.operation
                    visit(child, block_path + [index], guards)
                    index += child.name not in SYNC and child not in excluded
    visit(module.operation, [])
    return {"payload": records, "placements": placements, "allocations": allocations,
            "views": views, "abi": abi, "sync_control": dict(sync_control)}

def analyze(path):
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        module = ir.Module.parse(path.read_text())
        if not module.operation.verify():
            raise ValueError(f"invalid emitted IR: {path}")
        metrics = static_metrics(module.operation)
        result = project(module)
        counts = metrics["counts"]
        barriers = Counter()
        for key, value in counts.items():
            if key.startswith("barrier:"):
                barriers[re.search(r"PIPE_[A-Z0-9]+", key)[0]] += value
        result["mechanisms"] = {"sets": counts.get("pto.set_flag", 0),
                                "waits": counts.get("pto.wait_flag", 0),
                                "PIPE_ALL": barriers.pop("PIPE_ALL", 0),
                                "named": dict(barriers)}
        result["event_ids_by_direction"] = metrics["event_ids_by_direction"]
        result["status_attributes"] = [
            {k: v for k, v in attrs(op).items() if k.startswith("pto.insert_sync.")}
            for op in children(module.operation) if op.name == "func.func"
        ]
        return result

def boundary_evidence(seed, trial, scenarios):
    from compare_boundaries import ObserverUnsupported, compare, run
    rows = []
    for scenario in scenarios:
        observations, unknown = {}, {}
        for label, path in (("seed", seed), ("handoff", trial)):
            try:
                observations[label], _ = run(path, scenario)
            except ObserverUnsupported as error:
                unknown[label] = str(error)
        if unknown:
            rows.append({"scenario": scenario["name"], "status": "unsupported", "reasons_by_arm": unknown})
        else:
            before, after = observations["seed"], observations["handoff"]
            differences = compare(before, after)
            if any(item["automatic_requires_later_prefix"] for item in differences):
                raise ValueError("new blocking in concrete payload prefix")
            rows.append({"scenario": scenario["name"], "status": "checked",
                         "weaker_prefixes": differences})
    return rows

def population():
    manifest = json.loads((HERE / "checkpoint/manifest.json").read_text())
    for case in manifest["cases"]:
        case["source"] = ROOT / case["source"]
        if hashlib.sha256(case["source"].read_bytes()).hexdigest() != case["sha256"]:
            raise ValueError("changed frozen source: " + case["case_id"])
        for arm, artifact in case.get("artifacts", {}).items():
            path = HERE / artifact["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("changed checkpoint artifact: " + case["case_id"] + "/" + arm)
    return manifest["cases"]
