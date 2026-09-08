#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare a native local projection against scalar replay and native footprints.

This checks interchange, not native synchronization safety. GM and target
resource obligations stay explicitly outside the reference projection. The
scalar replay reads the captured PTO body independently of the isl schedule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "performance"))
from measure import replay
from eventlab.intervals import Footprint
from eventlab.symbolic import Program


def walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk(child.operation)


def check(record, arguments):
    if record.get("schema") != "ptoas.handoff-facts.v1":
        raise ValueError("unknown native schema")
    if record.get("native_transformation_proof") is not False:
        raise ValueError("diagnostic export must not claim transformation proof")
    if hashlib.md5(record["context_json"].encode()).hexdigest() != record["context_md5"]:
        raise ValueError("module context digest mismatch")
    captured_context = json.loads(record["context_json"])
    if record["status"] != "supported-local-projection":
        return {"status": "UNSUPPORTED", "reason": record["reason"]}
    if record["scope"] != "conservative-local-ordering-projection":
        raise ValueError("unknown qualification")
    if hashlib.md5(record["input_ir"].encode()).hexdigest() != record["input_ir_md5"]:
        raise ValueError("captured input digest mismatch")
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    program = Program(record["model"])
    native = record["native"]
    if captured_context["module_attributes_nearest_first"][0] != record["parent_module_attributes"]:
        raise ValueError("analyzed module context differs from captured context")
    parameters = {item["name"]: arguments[item["argument"]] for item in native["parameter_bindings"]}
    occurrences = program.instances(parameters, limit=8192)
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        attributes = record.get("parent_module_attributes", "{}")
        module = ir.Module.parse("module attributes " + attributes + " {\n" + record["input_ir"] + "\n}")
        function = next(op for op in walk(module.operation) if op.name == "func.func")
        operations = list(walk(function))
        mapping = {}
        for phase in native["phases"]:
            operation = operations[phase["operation_ordinal"]]
            if operation.name != phase["operation"]:
                raise ValueError("native phase identity lost")
            mapping[operation] = phase["id"]
        executed = []
        def observe(op, point, signature):
            if op in mapping:
                executed.append(mapping[op])
        replay(function, arguments, observer=observe)
    expected_ids = ["P" + str(phase) for phase in executed]
    actual_ids = [occurrence["identity"][0] for occurrence in occurrences]
    if actual_ids != expected_ids:
        raise ValueError("occurrence schedule differs from actual structured scalar replay")
    lanes = {phase["id"]: phase["lane"] for phase in native["phases"]}
    omitted = {entry["access"] for entry in native["omitted_effects"]}
    footprints = {}
    access_modes = {}
    for access in native["accesses"]:
        if access["space"] != "GM" and len(access["addresses"]) == 1:
            key = access["phase"], access["space"], int(access["addresses"][0]), int(access["allocation_bytes"])
            access_modes[key] = access_modes.get(key, 0) | (2 if access["write"] else 1)
    complete = {(item["space"], int(item["begin"]), int(item["bytes"]))
                for item in native["projections"] if item["complete"]}
    retained = {}
    for obligation in native["obligations"]:
        def access_key(side):
            access = obligation[side + "_access"]
            key = obligation[side], access["space"], int(access["begin"]), int(access["bytes"])
            modes = (1 if access["read"] else 0) | (2 if access["write"] else 0)
            if not modes or modes & ~access_modes.get(key, 0):
                raise ValueError("native obligation has an invalid access identity or mode")
            return key
        source_key, target_key = access_key("source"), access_key("target")
        hazards = obligation["hazards"]
        if not isinstance(hazards, int) or not hazards or hazards & ~15:
            raise ValueError("invalid native hazard kinds")
        key = source_key, target_key
        retained[key] = retained.get(key, 0) | hazards
    for phase in lanes:
        for mode in (False, True):
            intervals = []
            for access in native["accesses"]:
                if access["phase"] != phase or access["write"] != mode:
                    continue
                if access["space"] == "GM":
                    if access["id"] not in omitted:
                        raise ValueError("GM effect disappeared without an explicit boundary")
                    continue
                if not access["known_physical_addresses"] or access["unknown_range"]:
                    raise ValueError("unknown native range entered the projection")
                for address in access["addresses"]:
                    lo = int(address)
                    intervals.append((access["space"], lo, lo + int(access["allocation_bytes"])))
            footprints[phase, mode] = Footprint(intervals)
    for phase, occurrence in zip(executed, occurrences):
        if occurrence["lane"] != lanes[phase]:
            raise ValueError("physical lane mismatch")
        for mode, label in ((False, "reads"), (True, "writes")):
            if occurrence[label] != footprints[phase, mode]:
                raise ValueError("native footprint differs from reference access expression")
    # Compare dense access-order relationships without using isl's access-flow
    # discovery. The schedule above comes from the original scalar control.
    expected = set()
    expected_native = set()
    for target, b in enumerate(executed):
        for source, a in enumerate(executed[:target]):
            if (footprints[a, True] & (footprints[b, False] | footprints[b, True]) or
                    footprints[a, False] & footprints[b, True]):
                expected.add((occurrences[source]["identity"], occurrences[target]["identity"]))
                for source_key, source_modes in access_modes.items():
                    if source_key[0] != a or source_key[1:] not in complete:
                        continue
                    target_key = (b, *source_key[1:])
                    target_modes = access_modes.get(target_key, 0)
                    hazards = (1 if source_modes & 2 and target_modes & 1 else 0) | (
                        2 if source_modes & 1 and target_modes & 2 else 0) | (
                        4 if source_modes & 2 and target_modes & 2 else 0)
                    if hazards:
                        expected_native.add((source_key, target_key, hazards))
    for source_key, target_key, hazards in expected_native:
        if hazards & ~retained.get((source_key, target_key), 0):
            raise ValueError("native requirement inventory omits a migrated concrete conflict")
    dependencies = program.discover()
    actual = set(dependencies.dense.points(parameters, limit=1000000))
    if actual != expected:
        raise ValueError("reference dependencies differ from native conservative access requirements")
    # Challenge the separately retained native LocalStorageRequirements too.
    # Native rows quantify all ordered feasible occurrences; exclude only their
    # explicitly typed ACC resource part, which this memory model cannot prove.
    typed = {1: set(dependencies.raw.points(parameters, limit=1000000)),
             2: set(dependencies.war.points(parameters, limit=1000000)),
             4: set(dependencies.waw.points(parameters, limit=1000000))}
    native_checked = 0
    for obligation in native["obligations"]:
        a, b = obligation["source_access"], obligation["target_access"]
        left = Footprint([(a["space"], int(a["begin"]), int(a["begin"]) + int(a["bytes"]))])
        right = Footprint([(b["space"], int(b["begin"]), int(b["begin"]) + int(b["bytes"]))])
        if not left & right:
            continue
        for target, target_phase in enumerate(executed):
            if target_phase != obligation["target"]:
                continue
            for source, source_phase in enumerate(executed[:target]):
                if source_phase != obligation["source"]:
                    continue
                pair = occurrences[source]["identity"], occurrences[target]["identity"]
                for hazard, relation in typed.items():
                    if obligation["hazards"] & hazard:
                        if pair not in relation:
                            raise ValueError("retained native obligation lost by reference discovery")
                        native_checked += 1
    return {"status": "PASS", "scope": record["scope"], "parameters": parameters,
            "physical_occurrences": len(occurrences), "dense_requirements": len(expected),
            "native_memory_obligation_occurrences_checked": native_checked,
            "native_resource_obligations_outside_reference": sum(
                bool(row["hazards"] & 8) for row in native["obligations"]),
            "omitted_gm_effects": len(omitted), "isl_version": program.isl.version,
            "native_safety_or_device_proof": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--arguments", required=True, help="JSON array of concrete arguments")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = check(json.loads(options.record.read_text()), json.loads(options.arguments))
    options.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
