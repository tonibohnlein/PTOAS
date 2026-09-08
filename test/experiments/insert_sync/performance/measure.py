#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bounded scalar replay of emitted MLIR. Counts actions; does not simulate hardware.

Unknown scalar/control operations fail explicitly. Payloads remain symbolic:
their trace captures operation order, concrete descriptors and physical storage,
but neither numerical correctness nor asynchronous completion is checked here.
"""

from collections import Counter, defaultdict
import hashlib
import json
import re


SYNC = {"pto.set_flag", "pto.wait_flag", "pto.barrier"}
FIXED_PROTOCOL = {"pto.sync.set", "pto.sync.wait", "pto.set_ffts"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def attrs(op):
    return {entry.name: str(entry.attr) for entry in op.attributes}


def children(op):
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child.operation


def static_metrics(op):
    """Record placements relative to payload/control, and directed event footprint."""
    counts = Counter()
    keys = defaultdict(set)
    placements = []
    queue_counts = Counter()
    protocol = []

    def visit(node, path, depth):
        name = node.name
        if name in FIXED_PROTOCOL:
            protocol.append([path, name, attrs(node)])
        if name in {"pto.talloc", "pto.tpush", "pto.tpop", "pto.tfree", "pto.initialize_l2g2l_pipe"}:
            queue_counts[name] += 1
        if name in SYNC:
            properties = attrs(node)
            counts[name] += 1
            if name == "pto.barrier":
                counts[f"barrier:{properties['pipe']}:{'loop' if depth else 'outside-loop'}"] += 1
            else:
                direction = f"{properties['src_pipe']}->{properties['dst_pipe']}"
                keys[direction].add(properties["event_id"])
            placements.append([path, name, properties])
        for index, child in enumerate(children(node)):
            visit(child, path + [index], depth + (name == "scf.for"))

    visit(op, [], 0)
    result = {"counts": dict(counts), "event_ids_by_direction": {k: sorted(v) for k, v in keys.items()},
              "placements": placements, "placement_sha256": fingerprint(placements)}
    if queue_counts:
        result["fixed_queue_operations"] = dict(queue_counts)
    if protocol:
        result["fixed_protocol_operations"] = dict(Counter(item[1] for item in protocol))
        result["fixed_protocol_sha256"] = fingerprint(protocol)
    return result


def replay(function, arguments, block_idx=0, block_num=1, budget=2000000, observer=None):
    """Evaluate a concrete launch instance with bounded positive-step scf loops."""
    env = {}
    counts = Counter()
    scalar_counts = Counter()
    payload = hashlib.sha256()
    actions = hashlib.sha256()
    remaining = budget
    executed = 0

    def record(stream, value):
        stream.update(json.dumps(value, sort_keys=True).encode() + b"\n")

    def block(body, values=(), depth=0):
        nonlocal remaining, executed
        if len(values) != len(body.arguments):
            raise ValueError("block argument mismatch")
        env.update(zip(body.arguments, values))
        for view in body.operations:
            remaining -= 1
            if remaining < 0:
                raise ValueError("scalar replay budget exceeded")
            op = view.operation
            name = op.name
            if name.startswith("arith.") or name.startswith("scf."):
                scalar_counts[name] += 1
            properties = attrs(op)
            operands = [env[v] for v in op.operands]
            result = None
            if name in ("func.return", "scf.yield"):
                return operands
            if name == "scf.for":
                lo, hi, step, *carry = operands
                if not all(isinstance(v, int) for v in (lo, hi, step)) or step <= 0:
                    raise ValueError("unsupported loop bounds")
                if len(range(lo, hi, step)) > remaining:
                    raise ValueError("loop replay budget exceeded")
                for iv in range(lo, hi, step):
                    carry = block(op.regions[0].blocks[0], [iv] + carry, depth + 1)
                env.update(zip(op.results, carry))
                continue
            if name == "scf.if":
                if not isinstance(operands[0], (int, bool)):
                    raise ValueError("unknown branch condition")
                region = 0 if operands[0] else 1
                values = []
                if region < len(op.regions) and len(op.regions[region].blocks):
                    values = block(op.regions[region].blocks[0], depth=depth)
                env.update(zip(op.results, values))
                continue
            if name in ("pto.section.cube", "pto.section.vector"):
                block(op.regions[0].blocks[0], depth=depth)
                continue
            if name == "arith.constant":
                from ptoas.mlir import ir
                value = op.attributes["value"]
                result = ir.FloatAttr(value).value if ir.FloatAttr.isinstance(value) else ir.IntegerAttr(value).value
            elif name == "arith.negf":
                result = -operands[0] if isinstance(operands[0], (int, float)) else fingerprint([name, operands])
            elif name in ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui"):
                result = operands[0]
            elif name in ("arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.divui",
                          "arith.remsi", "arith.remui", "arith.andi", "arith.ori", "arith.xori"):
                a, b = operands
                if not isinstance(a, int) or not isinstance(b, int):
                    raise ValueError(f"unknown arithmetic operands: {name}")
                if name in ("arith.divui", "arith.remui") and (a < 0 or b < 0):
                    raise ValueError("unsigned negative operand unsupported")
                if name in ("arith.divsi", "arith.remsi", "arith.divui", "arith.remui"):
                    quotient = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                    result = a - quotient * b if "rem" in name else quotient
                else:
                    result = {"arith.addi": lambda: a + b, "arith.subi": lambda: a - b,
                              "arith.muli": lambda: a * b, "arith.andi": lambda: a & b,
                              "arith.ori": lambda: a | b, "arith.xori": lambda: a ^ b}[name]()
            elif name == "arith.cmpi":
                from ptoas.mlir import ir
                predicate = ir.IntegerAttr(op.attributes["predicate"]).value
                a, b = operands
                if not isinstance(a, int) or not isinstance(b, int) or predicate > 5:
                    raise ValueError("unsupported comparison")
                result = (a == b, a != b, a < b, a <= b, a > b, a >= b)[predicate]
            elif name == "arith.select":
                if not isinstance(operands[0], (int, bool)):
                    raise ValueError("unknown select condition")
                result = operands[1] if operands[0] else operands[2]
            elif name in ("arith.maxsi", "arith.minsi"):
                if not all(isinstance(v, int) for v in operands):
                    raise ValueError("unknown signed min/max operands")
                result = (max if name == "arith.maxsi" else min)(operands)
            elif name == "pto.get_block_idx":
                result = block_idx
            elif name == "pto.get_block_num":
                result = block_num
            elif name in SYNC:
                counts[name] += 1
                if name == "pto.barrier":
                    counts[f"barrier:{properties['pipe']}:{'loop' if depth else 'outside-loop'}"] += 1
                record(actions, [executed, name, properties])
                if observer:
                    observer(op, executed, None)
            elif name.startswith("pto.") and not op.regions:
                # Explicit allowlist prevents unknown sync/helper operations being
                # misreported as ordinary payload with a successful metric verdict.
                if name not in {"pto.alloc_tile", "pto.make_tensor_view", "pto.partition_view",
                                "pto.tload", "pto.tstore", "pto.textract", "pto.tmov", "pto.tabs",
                                "pto.tmatmul", "pto.tmatmul.acc", "pto.tsort32", "pto.tmrgsort",
                                "pto.tgather", "pto.bitcast", "pto.taxpy", "pto.tsetval", "pto.tgetval",
                                "pto.texpands", "pto.tcvt", "pto.tcolexpand", "pto.tmul", "pto.texp",
                                "pto.tfillpad", "pto.tadd", "pto.tsub", "pto.tmax",
                                "pto.trowexpanddiv", "pto.trowexpandmul"} | FIXED_PROTOCOL:
                    raise ValueError(f"unsupported payload: {name}")
                signature = [name, operands, properties, [str(v.type) for v in op.results]]
                record(payload, signature)
                if observer:
                    observer(op, executed, signature)
                executed += 1
                counts[name] += 1
                if name in {"pto.sync.set", "pto.sync.wait"}:
                    counts[f"{name}:{properties['event_id']}"] += 1
                # Fixed-length symbolic descriptors keep nested expressions bounded.
                result = fingerprint(signature)
            else:
                raise ValueError(f"unsupported operation: {name}")
            if len(op.results) == 1:
                if result is None:
                    raise ValueError(f"missing result: {name}")
                env[op.results[0]] = result
            elif op.results:
                raise ValueError(f"unsupported multiple results: {name}")
        return []

    block(function.regions[0].blocks[0], arguments)
    return {"counts": dict(counts), "scalar_counts": dict(scalar_counts), "payload_sha256": payload.hexdigest(),
            "action_trace_sha256": actions.hexdigest(), "scalar_steps": budget - remaining}


def normalize_gm_pipe_assembly(text):
    """Bridge the current custom-printer/parser mismatch without changing attrs.

    Only the exact GM-only form (one tensor operand, no local address) is
    recognized. Unknown attributes/forms remain untouched and fail parsing.
    """
    pattern = (r'pto\.initialize_l2g2l_pipe\{dir_mask = (\d+), slot_size = (\d+), '
               r'slot_num = (\d+), flag_base = (\d+), nosplit = (true|false)\} '
               r'\((%[\w]+) : (!pto\.tensor_view<[^>]+>)\) -> !pto\.pipe')

    def generic(match):
        direction, size, slots, flag, nosplit, operand, ty = match.groups()
        return (f'"pto.initialize_l2g2l_pipe"({operand}) <{{dir_mask = {direction} : i8, '
                f'slot_size = {size} : i32, slot_num = {slots} : i32, flag_base = {flag} : i32, '
                f'nosplit = {nosplit}, operandSegmentSizes = array<i32: 1, 0, 0>}}> : ({ty}) -> !pto.pipe')

    return re.subn(pattern, generic, text)


def measure(path, scenarios, mode="replay", normalize_gm_pipes=False):
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        original = path.read_text()
        normalization = None
        if normalize_gm_pipes:
            assembly, count = normalize_gm_pipe_assembly(original)
            if count:
                path.with_suffix('.metrics.pto').write_text(assembly)
                normalization = {"kind": "generic-gm-only-pipe", "operations": count,
                                 "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                                 "normalized_sha256": hashlib.sha256(assembly.encode()).hexdigest()}
        else:
            assembly = original
        module = ir.Module.parse(assembly)
        if mode == "static":
            result = {"static": static_metrics(module.operation), "scenarios": {},
                      "dynamic_measurement": "unsupported helper/queue domain; not replayed"}
            if normalization:
                result["assembly_normalization"] = normalization
            return result
        functions = [op for op in children(module.operation) if op.name == "func.func"]
        by_name = {ir.StringAttr(op.attributes["sym_name"]).value: op for op in functions}
        result = {"static": static_metrics(module.operation), "scenarios": {}}
        for scenario in scenarios:
            if "function" in scenario:
                function = by_name[scenario["function"]]
            elif len(functions) == 1:
                function = functions[0]
            else:
                raise ValueError("multi-function benchmark requires an explicit scenario function")
            result["scenarios"][scenario["name"]] = replay(
                function, scenario["arguments"], scenario.get("block_idx", 0),
                scenario.get("block_num", 1))
        return result
