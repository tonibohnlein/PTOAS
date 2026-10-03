# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent finite command-contract oracle for the actual compiled loop."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from ptoas.mlir.ir import Context, DenseI64ArrayAttr, DictAttr, IntegerAttr, Module, BoolAttr
from ptoas.mlir.dialects import pto


def close(edges):
    reach = [set(row) for row in edges]
    for middle in range(len(reach)):
        for row in reach:
            if middle in row:
                row.update(reach[middle])
    return reach


def native(payloads):
    graph = [set() for _ in range(2 * len(payloads))]
    last = {}
    for site, pipe in enumerate(payloads):
        graph[2 * site].add(2 * site + 1)
        if pipe in last:
            prior = last[pipe]
            graph[2 * prior].add(2 * site)
            graph[2 * prior + 1].add(2 * site + 1)
        last[pipe] = site
    return graph


def requirement_graph(function, n, correlated, active=None, scalar_inputs=None):
    effects = []
    commands = execute(function, n, correlated, effects, active, scalar_inputs)
    pipes = [argument for kind, argument in commands if kind == "payload"]
    graph = native(pipes)
    for source, (reads, writes) in enumerate(effects):
        for target in range(source + 1, len(effects)):
            target_reads, target_writes = effects[target]
            if writes & target_reads or reads & target_writes or writes & target_writes:
                graph[2 * source + 1].add(2 * target)
    return graph


def expected(function, n, correlated, active=None, scalar_inputs=None):
    return close(requirement_graph(function, n, correlated, active, scalar_inputs))


def attr_name(attribute):
    match = re.search(r"PIPE_[A-Z0-9]+", str(attribute))
    if match is None:
        raise ValueError("missing pipe attribute")
    return match.group()


def tile_bytes(tile_type):
    text = str(tile_type)
    shape = re.search(r"rows=(\d+), cols=(\d+)", text)
    if shape:
        rows, cols = map(int, shape.groups())
    else:
        shape = re.search(r"(\d+)x(\d+)x(?:i|f)(\d+)", text)
        assert shape, "missing original static physical tile shape"
        rows, cols = map(int, shape.groups()[:2])
    element = re.search(r"dtype=(?:i|f)(\d+)", text) or re.search(r"x(?:i|f)(\d+)", text)
    assert element, "missing original element width"
    return rows * cols * int(element.group(1)) // 8


def physical_cells(address, size):
    # Test-only32-byte interval cells preserve cross-root overlap. Every
    # admitted fixture operand has block-aligned geometry; no compiler model
    # or dependency result is consulted here.
    assert address % 32 == 0 and size % 32 == 0
    return set(range(address, address + size, 32))


def execute(function, n, correlated=False, effects=None, active=None, scalar_inputs=None,
            values_out=None, occurrences_out=None):
    values = {}
    for index, argument in enumerate(function.regions[0].blocks[0].arguments):
        if str(argument.type) == "i1":
            assert isinstance(active, bool), "original Boolean argument requires an explicit test valuation"
            values[argument] = active
        elif scalar_inputs is not None and index in scalar_inputs:
            values[argument] = scalar_inputs[index]
        else:
            values[argument] = n if index == 2 else {"gm"}
    commands = []
    frames = []
    payload_sites = {}
    if occurrences_out is not None:
        def collect(region_block):
            for view in region_block.operations:
                operation = view.operation
                if operation.name in ("pto.tload", "pto.txor", "pto.trowexpandmul", "pto.tstore", "pto.load"):
                    payload_sites[operation] = len(payload_sites)
                for region in operation.regions:
                    for nested in region.blocks:
                        collect(nested)
        collect(function.regions[0].blocks[0])

    def block(region_block):
        for view in region_block.operations:
            operation = view.operation
            name = operation.name
            operands = [values.get(value) for value in operation.operands]
            result = None
            results = None
            if name in ("scf.yield", "func.return"):
                return operands
            if name == "arith.constant":
                text = str(operation.attributes["value"])
                if str(operation.results[0].type) == "f32":
                    result = float(text.split(" : ")[0])
                elif text in ("true", "false"):
                    result = int(text == "true")
                else:
                    # MLIR's Python IntegerAttr.value narrows to int64. Parse
                    # its exact decimal form for widened endpoint constants.
                    match = re.match(r"(-?\d+)(?: : .*)?$", text)
                    assert match, "unsupported scalar integer constant"
                    result = int(match.group(1))
            elif name in ("arith.addi", "arith.subi", "arith.remui", "arith.muli", "arith.divui", "arith.floordivsi"):
                a, b = operands
                if name == "arith.remui":
                    if b <= 0:
                        raise ValueError("invalid modulo")
                    result = a % b
                elif name == "arith.floordivsi":
                    assert b > 0, "represented endpoint floor divisor must be positive"
                    result = a // b
                elif name == "arith.divui":
                    if b <= 0:
                        raise ValueError("invalid unsigned divisor")
                    width = int(str(operation.results[0].type)[1:])
                    result = (a % (1 << width)) // b
                elif name == "arith.addi":
                    result = a + b
                elif name == "arith.subi":
                    result = a - b
                else:
                    result = a * b
            elif name in ("arith.maxsi", "arith.minsi"):
                result = max(operands) if name == "arith.maxsi" else min(operands)
            elif name == "arith.cmpi":
                predicate = IntegerAttr(operation.attributes["predicate"]).value
                a, b = operands
                result = {0: a == b, 1: a != b, 2: a < b, 3: a <= b, 4: a > b, 5: a >= b}[predicate]
            elif name in ("arith.index_cast", "arith.extsi", "arith.extui"):
                result = operands[0]
                operand_type = str(operation.operands[0].type)
                if name == "arith.extsi" and operand_type.startswith("i"):
                    width = int(operand_type[1:])
                    result %= 1 << width
                    if result >= 1 << (width - 1):
                        result -= 1 << width
            elif name in ("arith.ori", "arith.andi", "arith.xori"):
                a, b = operands
                result = (a | b) if name == "arith.ori" else ((a & b) if name == "arith.andi" else (a ^ b))
            elif name == "arith.select":
                result = operands[1] if operands[0] else operands[2]
            elif name == "scf.for":
                nested = operation.regions[0].blocks[0]
                carried = operands[3:]
                for iteration in range(*operands[:3]):
                    values[nested.arguments[0]] = iteration
                    for argument, value in zip(nested.arguments[1:], carried, strict=True):
                        values[argument] = value
                    frames.append(iteration)
                    carried = block(nested)
                    frames.pop()
                results = carried
            elif name == "scf.if":
                arm = 0 if operands[0] else 1
                if arm < len(operation.regions) and len(operation.regions[arm].blocks):
                    results = block(operation.regions[arm].blocks[0])
                else:
                    results = []
            elif name == "pto.alloc_tile":
                result = physical_cells(operands[0], tile_bytes(operation.results[0].type))
            elif name == "pto.alloc_multi_tile":
                slots = int(re.search(r"count=(\d+)", str(operation.results[0].type)).group(1))
                result = (operands[0], slots, tile_bytes(operation.results[0].type))
            elif name == "pto.multi_tile_get":
                address, slots, size = operands[0]
                result = set().union(*(physical_cells(address + size * slot, size) for slot in
                                     ([operands[1]] if correlated else range(slots))))
            elif name in ("pto.make_tensor_view", "pto.partition_view"):
                # These fixtures model GM through the shared conservative class.
                result = operands[0]
            elif name in ("pto.tload", "pto.txor", "pto.trowexpandmul", "pto.tstore", "pto.load"):
                pipe = {"pto.tload": "PIPE_MTE2", "pto.txor": "PIPE_V", "pto.trowexpandmul": "PIPE_V",
                        "pto.tstore": "PIPE_MTE3", "pto.load": "PIPE_S"}[name]
                if name == "pto.tstore" and ("loc=acc" in str(operation.operands[0].type) or
                                             "<acc," in str(operation.operands[0].type)):
                    pipe = "PIPE_FIX"
                commands.append(("payload", pipe))
                if occurrences_out is not None:
                    occurrences_out.append((payload_sites[operation], list(frames)))
                if effects is not None:
                    if name == "pto.load":
                        effects.append((operands[0], set()))
                    elif name in ("pto.txor", "pto.trowexpandmul"):
                        effects.append((operands[0] | operands[1] | operands[2], operands[2] | operands[3]))
                    else:
                        effects.append((operands[0], operands[1]))
                if name == "pto.load":
                    result = 0
            elif name in ("pto.set_flag", "pto.wait_flag"):
                key = (attr_name(operation.attributes["src_pipe"]),
                       attr_name(operation.attributes["dst_pipe"]), str(operation.attributes["event_id"]))
                commands.append(("set" if name == "pto.set_flag" else "wait", key))
            elif name in ("pto.set_flag_dyn", "pto.wait_flag_dyn"):
                key = (attr_name(operation.attributes["src_pipe"]),
                       attr_name(operation.attributes["dst_pipe"]), f"#pto.event<EVENT_ID{operands[0]}>")
                commands.append(("set" if name == "pto.set_flag_dyn" else "wait", key))
            elif name == "pto.barrier":
                commands.append(("barrier", attr_name(operation.attributes["pipe"])))
            elif name not in ("pto.alloc_tile", "pto.alloc_multi_tile", "pto.multi_tile_get", "scf.yield",
                              "func.return"):
                raise ValueError("uninterpreted compiled operation: " + name)
            if results is not None:
                for value, computed in zip(operation.results, results, strict=True):
                    values[value] = computed
            elif len(operation.results) == 1:
                values[operation.results[0]] = result
    block(function.regions[0].blocks[0])
    if values_out is not None:
        values_out.update(values)
    return commands


def command_graph(commands, details=None):
    graph = []
    payloads = []
    starts = {}
    completions = {}
    prefix = {}
    gate = {}
    publications = {}
    events = []

    def vertex():
        graph.append(set())
        return len(graph) - 1

    def edge(source, target):
        graph[source].add(target)

    for kind, argument in commands:
        if kind == "payload":
            pipe = argument
            start, completion = vertex(), vertex()
            edge(start, completion)
            if pipe in starts:
                edge(starts[pipe], start)
                edge(completions[pipe], completion)
            if pipe in gate:
                edge(gate[pipe], start)
            starts[pipe], completions[pipe] = start, completion
            prefix.setdefault(pipe, []).append(completion)
            payloads.extend((start, completion))
            continue
        finish = vertex()
        events.append((kind, argument, finish))
        pipe = argument[0] if kind == "set" else argument[1] if kind == "wait" else argument
        pipes = list(prefix) if pipe == "PIPE_ALL" else [pipe]
        for current in pipes:
            if current in starts:
                edge(starts[current], finish)
            if current in gate:
                edge(gate[current], finish)
            if kind != "wait":
                for previous in prefix.get(current, []):
                    edge(previous, finish)
        if kind == "set":
            if argument in publications:
                raise ValueError("republication before consumption")
            publications[argument] = finish
        elif kind == "wait":
            if argument not in publications:
                raise ValueError("acquisition without publication")
            edge(publications.pop(argument), finish)
            gate[pipe] = finish
        elif pipe != "PIPE_ALL":
            gate[pipe] = finish
        for current in pipes:
            prefix.setdefault(current, []).append(finish)
    if publications:
        raise ValueError("unconsumed invocation publication")
    reach = close(graph)
    if details is not None:
        details.update(reach=reach, events=events)
    return [{target for target, event in enumerate(payloads) if event in reach[source]} for source in payloads]


def run(tool, source, algorithm, extra=""):
    profile = ""
    return subprocess.run([tool, "--mlir-disable-threading", "--pto-insert-sync-debug=2",
                           "--pto-insert-sync=algorithm=" + algorithm + profile + extra, source, "-o", "-"],
                          capture_output=True, text=True, check=False, timeout=120)


def activation_certificate(original, output):
    arguments = list(original.regions[0].blocks[0].arguments)
    booleans = [index for index, argument in enumerate(arguments) if str(argument.type) == "i1"]
    common = next((operation for operation in original.regions[0].blocks[0].operations
                   if operation.operation.name == "scf.if" and any(
                       nested.operation.name == "scf.for"
                       for nested in operation.regions[0].blocks[0].operations)), None)
    if common is None:
        return
    summary = DictAttr(DictAttr(output.attributes["pto.frontier.analysis"])["single_stream"])
    binding = DictAttr(summary["activation_binding"])
    if booleans:
        assert len(booleans) == 1
        assert common.operands[0] == arguments[booleans[0]]
        assert str(binding["origin"]) == '"argument"'
        assert IntegerAttr(binding["argument"]).value == booleans[0]
    else:
        comparison = common.operands[0].owner
        assert comparison.name == "arith.cmpi"
        assert str(binding["origin"]) == '"comparison-result"'
        assert IntegerAttr(summary["activation_argument"]).value == -1
        assert IntegerAttr(binding["predicate"]).value == IntegerAttr(comparison.attributes["predicate"]).value
        argument_first = BoolAttr(binding["argument_first"]).value
        argument = comparison.operands[0 if argument_first else 1]
        literal = comparison.operands[1 if argument_first else 0].owner
        assert argument == arguments[IntegerAttr(binding["argument"]).value]
        assert literal.name == "arith.constant" and binding["constant"] == literal.attributes["value"]
        compiled_common = next(operation for operation in output.regions[0].blocks[0].operations
                               if operation.operation.name == "scf.if")
        compiled_comparison = compiled_common.operands[0].owner
        assert IntegerAttr(compiled_comparison.attributes["pto.frontier.source"]).value == \
            IntegerAttr(binding["result_source"]).value
        assert IntegerAttr(binding["result_index"]).value == 0
        compiled_literal = compiled_comparison.operands[1 if argument_first else 0].owner
        assert IntegerAttr(compiled_literal.attributes["pto.frontier.source"]).value == \
            IntegerAttr(binding["constant_source"]).value
    assert IntegerAttr(summary["participation_source"]).value >= 0
    shared = DictAttr(DictAttr(output.attributes["pto.frontier.analysis"])["shared_target"])
    shared_participation = DictAttr(shared["common_participation"])
    assert shared_participation["activation_binding"] == summary["activation_binding"]
    assert shared_participation["source"] == summary["participation_source"]
    if IntegerAttr(summary["publication_cut_source"]).value >= 0:
        compiled_common = next(operation for operation in output.regions[0].blocks[0].operations
                               if operation.operation.name == "scf.if")
        compiled_loop = next(operation for operation in compiled_common.regions[0].blocks[0].operations
                             if operation.operation.name == "scf.for")
        assert IntegerAttr(compiled_loop.attributes["pto.frontier.source"]).value == \
            IntegerAttr(summary["publication_cut_source"]).value
    for raw in summary["minimum"]:
        piece = DictAttr(raw)
        width = IntegerAttr(piece["source_depth"]).value + IntegerAttr(piece["consumer_depth"]).value + 3
        rows = list(DenseI64ArrayAttr(piece["equalities"]))
        required = [0] * (width - 2) + [1, -1]
        assert required in [rows[begin:begin + width] for begin in range(0, len(rows), width)], \
            "retained family omitted original activation"
    for raw in summary["bound_ports"]:
        port = DictAttr(raw)
        assert port["activation_binding"] == summary["activation_binding"]
        assert list(DenseI64ArrayAttr(port["presence_equalities"])) == [0, 1, -1]
        assert list(DenseI64ArrayAttr(port["presence_inequalities"])) == [1, 0, -1]



def participation_valuations(function, n):
    arguments = list(function.regions[0].blocks[0].arguments)
    common = next((operation for operation in function.regions[0].blocks[0].operations
                   if operation.operation.name == "scf.if" and any(
                       nested.operation.name == "scf.for"
                       for nested in operation.regions[0].blocks[0].operations)), None)
    if common is None:
        return [(None, None)]
    condition = common.operands[0]
    if condition in arguments:
        return [(value, None) for value in (False, True)]
    comparison = condition.owner
    assert comparison.name == "arith.cmpi"
    ordinal = next(index for index, argument in enumerate(arguments) if argument in comparison.operands)
    assert str(arguments[ordinal].type) == "index"
    result = []
    for value in (-1, 0, 1):
        scalars = {ordinal: value}
        values = {}
        execute(function, n, scalar_inputs=scalars, values_out=values)
        active = values[condition]
        assert isinstance(active, bool)
        result.append((active, scalars))
    return result


def retained_covers(function, n, active=None):
    analysis = DictAttr(function.attributes["pto.frontier.analysis"])
    summary = DictAttr(analysis["single_stream"])
    pieces = summary["minimum"]
    binding = DictAttr(summary["activation_binding"]) if "activation_binding" in summary else None
    activated = binding is not None and str(binding["origin"]) != '"none"'
    if activated:
        if str(binding["origin"]) == '"argument"':
            identity = IntegerAttr(binding["argument"]).value
            assert str(function.regions[0].blocks[0].arguments[identity].type) == "i1"
        assert isinstance(active, bool)
    count = max(0, n) if not activated or active else 0
    bodies = list(DenseI64ArrayAttr(summary["body_sites"]))
    repeated = "protocol" in summary and "repeated-ready-release" in str(summary["protocol"])
    prefix = "prefix_realization" in summary and "first original" in str(summary["prefix_realization"])
    occurrences = ([] if repeated and not prefix else [(0, [])]) + [
        (site, [i]) for i in range(count) for site in bodies]
    if count:
        occurrences.append((bodies[-1] + 1, []))
    covers = set()
    for raw in pieces:
        piece = DictAttr(raw)
        source_id = IntegerAttr(piece["source"]).value
        consumer_id = IntegerAttr(piece["consumer"]).value
        for source, (source_site, source_coordinates) in enumerate(occurrences):
            for target, (target_site, target_coordinates) in enumerate(occurrences):
                if source_site != source_id or target_site != consumer_id:
                    continue
                coordinates = source_coordinates + target_coordinates + [n] + ([int(active)] if activated else []) + [1]
                width = len(coordinates)
                admitted = True
                for name in ("equalities", "inequalities"):
                    rows = list(DenseI64ArrayAttr(piece[name]))
                    assert len(rows) % width == 0, "malformed durable affine row dimensions"
                    for offset in range(0, len(rows), width):
                        value = sum(a * b for a, b in zip(rows[offset:offset + width], coordinates))
                        admitted &= value == 0 if name == "equalities" else value >= 0
                if admitted:
                    covers.add((source, target))
    return covers


def independent_covers(function, n, active=None, scalar_inputs=None):
    graph = requirement_graph(function, n, False, active, scalar_inputs)
    retained = set()
    for source in range(len(graph) // 2):
        for target in range(source + 1, len(graph) // 2):
            if 2 * target not in graph[2 * source + 1]:
                continue
            alternative = [set(row) for row in graph]
            alternative[2 * source + 1].remove(2 * target)
            if 2 * target not in close(alternative)[2 * source + 1]:
                retained.add((source, target))
    return retained


def rejected_original(tool, candidate, name):
    parsed = subprocess.run([tool, "--mlir-disable-threading", str(candidate), "-o", "-"],
                            capture_output=True, text=True, check=False, timeout=30)
    assert parsed.returncode == 0, parsed.stderr
    failed = subprocess.run([tool, "--mlir-disable-threading", "--mlir-print-ir-after-failure",
                             "--mlir-print-ir-module-scope", "--pto-insert-sync=algorithm=frontier-synch"
                             "", str(candidate), "-o", "-"],
                            capture_output=True, text=True, check=False, timeout=120)
    assert failed.returncode != 0, name + " escaped bounded admission"
    assert "certified-single-generation-boundaries" not in failed.stderr
    marker = failed.stderr.index("// -----// IR Dump After")
    output = failed.stderr[marker:]
    output = output[output.index("module attributes"):].strip()
    assert output == parsed.stdout.strip(), name + " rejection modified original IR"


def negative_stages(tool, source):
    original = Path(source).read_text(encoding="utf-8")
    second = re.findall(r"pto.txor[^\n]+", original)[1]
    allocated = original.replace("    pto.tload ins", "    %a_other = arith.constant 90112 : i64\n"
                                 "    %other_tmp = pto.alloc_tile addr = %a_other : !vec\n    pto.tload ins", 1)
    if "%other_tmp =" in original:
        allocated = original
    alternatives = {
        "missing-adjacency": allocated.replace(second, second.replace("%tmp", "%other_tmp")
                                                .replace("%output, %output", "%input, %input")),
        "missing-scratch": original.replace(second, re.sub(r", %\w+ : !vec, !vec, !vec", " : !vec, !vec", second)),
        "skipped-body": original.replace(second, "scf.if %skip {\n        " + second + "\n      }")
            .replace("      %slotB =", "      %skip = arith.cmpi sgt, %i, %zero : index\n      %slotB ="),
    }
    if "stages_3" in Path(source).name:
        third = re.findall(r"pto.txor[^\n]+", original)[2]
        # Preserve both lexical RAWs but remove the original union wrap: three
        # disjoint output families and a different scratch for the last phase.
        missing_wrap = allocated.replace("    pto.tload ins",
            "    %ac = arith.constant 150016 : i64\n"
            "    %banksC = pto.alloc_multi_tile addr = %ac : !pto.multi_tile_buf<!vec, count=2>\n"
            "    pto.tload ins", 1)
        missing_wrap = missing_wrap.replace(third, third.replace("%tmp", "%other_tmp"))
        missing_wrap = missing_wrap.replace("%outputC = pto.multi_tile_get %banks[%slot]",
                                            "%outputC = pto.multi_tile_get %banksC[%slot]")
        before_exit, after_exit = missing_wrap.split("    %nonempty", 1)
        after_exit = after_exit.replace("%banks[%slot]", "%banksC[%slot]")
        alternatives["missing-wrap"] = before_exit + "    %nonempty" + after_exit
    prefix, exit_part = original.split("    %nonempty", 1)
    if "stages_2" in Path(source).name:
        exit_part = exit_part.replace("arith.remui %last, %countB", "arith.remui %last, %count")
        exit_part = exit_part.replace("%banksB[%slot] : !pto.multi_tile_buf<!vec, count=3>",
                                     "%banks[%slot] : !pto.multi_tile_buf<!vec, count=2>")
    else:
        exit_part = exit_part.replace("arith.remui %last, %count", "arith.remui %last, %countB")
        exit_part = exit_part.replace("%banks[%slot] : !pto.multi_tile_buf<!vec, count=2>",
                                     "%banksB[%slot] : !pto.multi_tile_buf<!vec, count=5>")
    alternatives["exit-earlier-family"] = prefix + "    %nonempty" + exit_part
    with tempfile.TemporaryDirectory(prefix="banked-stages-") as directory:
        for name, text in alternatives.items():
            candidate = Path(directory) / (name + ".pto")
            candidate.write_text(text, encoding="utf-8")
            rejected_original(tool, candidate, name)


def negative_sources(tool, source):
    original = Path(source).read_text(encoding="utf-8")
    payload = "pto.txor ins(%input, %input, %tmp : !vec, !vec, !vec) outs(%output : !vec)"
    alternatives = {
        "skipped-payload": original.replace(payload, "scf.if %nonempty_body {\n        " + payload + "\n      }")
            .replace("%slot = arith.remui %i, %count : index", "%nonempty_body = arith.cmpi sgt, %i, %zero : index\n"
                     "      %slot = arith.remui %i, %count : index"),
        "changing-scratch": original.replace("    pto.tload ins", "    %a3 = arith.constant 90112 : i64\n"
            "    %alternate = pto.alloc_tile addr = %a3 : !vec\n    pto.tload ins", 1)
            .replace(payload, "pto.txor ins(%input, %input, %changing : !vec, !vec, !vec) outs(%output : !vec)")
            .replace("%slot = arith.remui %i, %count : index", "%first = arith.cmpi eq, %i, %zero : index\n"
                     "      %changing = arith.select %first, %tmp, %alternate : !vec\n"
                     "      %slot = arith.remui %i, %count : index"),
    }
    with tempfile.TemporaryDirectory(prefix="banked-xor-") as directory:
        duplicate = Path(directory) / "duplicate-payload.pto"
        duplicate.write_text(original.replace(payload, payload + "\n      " + payload), encoding="utf-8")
        doubled = run(tool, str(duplicate), "frontier-synch")
        assert doubled.returncode == 0, doubled.stderr
        assert "certified-single-generation-boundaries" in doubled.stdout
        constant_source = original.replace("    %count =", "    %upper = arith.constant 4 : index\n    %count =", 1)
        constant_source = constant_source.replace("to %n step", "to %upper step")
        constant_source = constant_source.replace("sgt, %n, %zero", "sgt, %upper, %zero")
        constant_source = constant_source.replace("subi %n, %one", "subi %upper, %one")
        constant_path = Path(directory) / "constant-after-entry.pto"
        constant_path.write_text(constant_source, encoding="utf-8")
        positive = run(tool, str(constant_path), "frontier-synch")
        assert positive.returncode == 0, positive.stderr
        assert "certified-single-generation-boundaries" in positive.stdout
        for name, text in alternatives.items():
            candidate = Path(directory) / (name + ".pto")
            candidate.write_text(text, encoding="utf-8")
            rejected_original(tool, candidate, name)


def row_expand_source_mutations(original):
    alternatives = {
        "scratch-alias": original.replace("131072 : i64", "0 : i64"),
        "missing-explicit-scratch": original.replace(
            "%data, %coeff, %tmp : !vec, !coeff, !scratch", "%data, %coeff : !vec, !coeff"),
        "partial-valid": original.replace("v_row=16, v_col=128", "v_row=15, v_col=128").replace(
            "v_row=16, v_col=1,", "v_row=15, v_col=1,").replace(
                "!pto.partition_tensor_view<16x128xf32>", "!pto.partition_tensor_view<15x128xf32>"),
        "unaccounted-metadata": original.replace("    // These constants",
            "    %rows, %cols = pto.get_validshape %data : !vec\n    // These constants"),
    }
    alternatives["row-major-undrained"] = original.replace(
        "rows=16, cols=1, v_row=16, v_col=1, blayout=col_major",
        "rows=16, cols=8, v_row=16, v_col=8, blayout=row_major").replace(
            "%data, %coeff, %tmp : !vec, !coeff, !scratch", "%data, %coeff : !vec, !coeff")
    # Unknown original alternatives cannot inherit the first disjoint origin.
    alternatives["alternative-scratch-alias"] = original.replace("    pto.tload ins",
        "    %alias = pto.alloc_tile addr = %a0 : !scratch\n    pto.tload ins", 1).replace(
            "      pto.trowexpandmul", "      %first = arith.cmpi eq, %i, %zero : index\n"
            "      %selected = arith.select %first, %tmp, %alias : !scratch\n      pto.trowexpandmul").replace(
                "%data, %coeff, %tmp :", "%data, %coeff, %selected :")
    return alternatives


def row_expand_negatives(tool, source):
    original = Path(source).read_text(encoding="utf-8")
    alternatives = row_expand_source_mutations(original)
    with tempfile.TemporaryDirectory(prefix="row-expand-") as directory:
        malformed = Path(directory) / "misaligned-scalar.pto"
        malformed.write_text(original.replace("8192 : i64", "8196 : i64"), encoding="utf-8")
        verified = subprocess.run([tool, "--mlir-disable-threading", str(malformed), "-o", "-"],
                                  capture_output=True, text=True, check=False, timeout=30)
        assert verified.returncode != 0 and "addr must be aligned to 32 bytes" in verified.stderr
        relocated = Path(directory) / "initial-coefficient-reader.pto"
        relocated.write_text(original.replace("8192 : i64", "32768 : i64"), encoding="utf-8")
        accepted = run(tool, str(relocated), "frontier-synch")
        baseline = run(tool, str(relocated), "existing")
        assert accepted.returncode == baseline.returncode == 0, accepted.stderr + baseline.stderr
        shared = lambda output: [line for line in output.splitlines() if line.startswith("Shared")]
        assert shared(accepted.stderr) == shared(baseline.stderr), "relocated input witness parity"
        with Context() as context:
            pto.register_dialect(context)
            source_module = Module.parse(relocated.read_text(encoding="utf-8"))
            compiled_module = Module.parse(accepted.stdout)
            source_function = list(source_module.body.operations)[0]
            compiled_function = list(compiled_module.body.operations)[0]
            source_values = {}
            execute(source_function, 1, values_out=source_values)
            root = source_function.regions[0].blocks[0]
            load = next(view.operation for view in root.operations if view.operation.name == "pto.tload")
            loop = next(view.operation for view in root.operations if view.operation.name == "scf.for")
            multiply = next(view.operation for view in loop.regions[0].blocks[0].operations
                            if view.operation.name == "pto.trowexpandmul")
            loaded = source_values[load.operands[1]]
            assert loaded & source_values[multiply.operands[0]], "data RAW vanished"
            assert not loaded & source_values[multiply.operands[1]], "initial coefficient falsely has RAW"
            for trips in (-2, 0, 1, 2, 5):
                assert retained_covers(compiled_function, trips) == independent_covers(source_function, trips)
                assert command_graph(execute(compiled_function, trips)) == expected(source_function, trips, False)
        for name, text in alternatives.items():
            candidate = Path(directory) / (name + ".pto")
            candidate.write_text(text, encoding="utf-8")
            rejected_original(tool, candidate, name)


def main():
    tool, source = sys.argv[1:]
    existing = run(tool, source, "existing")
    compiled = run(tool, source, "frontier-synch")
    assert existing.returncode == 0, existing.stderr
    assert compiled.returncode == 0, compiled.stderr
    shared = lambda output: [line for line in output.splitlines() if line.startswith("Shared")]
    assert shared(existing.stderr) == shared(compiled.stderr), "shared input/selector evidence diverged"
    assert any(line.startswith("SharedSlots") and "unit_ordinal_modulo = true" in line
               for line in shared(compiled.stderr)), "shared occurrence correlation missing"
    detailed = run(tool, source, "frontier-synch", " dump-frontier-demands=true report-frontier-costs=true")
    assert detailed.returncode == 0, detailed.stderr
    assert detailed.stdout == compiled.stdout, "diagnostics changed output"
    assert "supported-source-arith-scf-guards" in detailed.stderr
    assert "equal to selected closure under recorded modeled requirements" in detailed.stderr
    banks = int(re.search(r"count=(\d+)", Path(source).read_text(encoding="utf-8")).group(1))
    with Context() as context:
        pto.register_dialect(context)
        module = Module.parse(compiled.stdout)
        function = list(module.body.operations)[0]
        original = Module.parse(Path(source).read_text(encoding="utf-8"))
        source_function = list(original.body.operations)[0]
        source_text = Path(source).read_text(encoding="utf-8")
        moduli = [int(value) for value in re.findall(r"count=(\d+)", source_text)]
        activation_certificate(source_function, function)
        if "row_expand" in Path(source).name:
            original_values = {}
            execute(source_function, 1, values_out=original_values)
            operations = []
            def collect(region):
                for operation in region.operations:
                    operations.append(operation)
                    for nested in operation.regions:
                        for block in nested.blocks:
                            collect(block)
            collect(source_function.regions[0].blocks[0])
            load = next(operation for operation in operations if operation.name == "pto.tload")
            multiply = next(operation for operation in operations if operation.name == "pto.trowexpandmul")
            loaded = original_values[load.operands[1]]
            for operand in multiply.operands[:2]:
                assert operand != load.operands[1], "cross-root descriptor fixture lost"
                assert loaded & original_values[operand], "packed-load RAW input missing"
        trips = {-2, 0, 1, 2, banks - 1, banks, banks + 1, 2 * max(moduli) + 1}
        for n in sorted(trips):
            for active, scalar_inputs in participation_valuations(source_function, n):
                assert retained_covers(function, n, active) == independent_covers(
                    source_function, n, active, scalar_inputs), (
                    banks, n, "advertised F* covers")
                commands = execute(function, n, active=active, scalar_inputs=scalar_inputs)
                projected = command_graph(commands)
                assert projected == expected(source_function, n, False, active, scalar_inputs), \
                    (banks, n, "modeled closure")
                correlated = expected(source_function, n, True, active, scalar_inputs)
                assert all(row <= projected[i] for i, row in enumerate(correlated)), (banks, n, "correlated coverage")
                if active is False or n <= 0:
                    assert not any(kind in ("set", "wait") for kind, _ in commands)
                    surplus = list(commands) + [("set", ("PIPE_MTE2", "PIPE_V", "EVENT_ID0"))]
                    try:
                        command_graph(surplus)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("unguarded surplus publication escaped independent oracle")
                if n > 0 and active is not False:
                    mutated = list(commands)
                    removed = next(i for i, command in enumerate(mutated) if command[0] == "wait")
                    mutated.pop(removed)
                    try:
                        equal = command_graph(mutated) == expected(source_function, n, False, active, scalar_inputs)
                    except ValueError:
                        equal = False
                    assert not equal, "missing acquisition escaped independent oracle"
                    # SET is an observation, not a gate: publishing only after the
                    # first consumer cannot establish its preceding prerequisite.
                    moved = list(commands)
                    publication = next(i for i, command in enumerate(moved) if command[0] == "set")
                    source_publication = moved.pop(publication)
                    consumer = next(i for i, command in enumerate(moved)
                                    if command == ("payload", "PIPE_V"))
                    moved.insert(consumer + 1, source_publication)
                    try:
                        equal = command_graph(moved) == expected(source_function, n, False, active, scalar_inputs)
                    except ValueError:
                        equal = False
                    assert not equal, "moved publication escaped independent oracle"
                    duplicated = list(commands)
                    publication = next(i for i, command in enumerate(duplicated) if command[0] == "set")
                    duplicated.insert(publication, duplicated[publication])
                    try:
                        command_graph(duplicated)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("duplicate publication escaped independent oracle")
    if "guarded_entry" in Path(source).name:
        with tempfile.TemporaryDirectory(prefix="guarded-entry-") as directory:
            sources = {
                "intervening-payload": source_text.replace("    %zero =", "    pto.tload ins(%gm0 : "
                    "!pto.partition_tensor_view<16x128xi32>) outs(%tmp : !vec)\n    %zero =", 1),
                "intervening-barrier": source_text.replace("    %zero =", "    pto.barrier <PIPE_V>\n    %zero =", 1),
                "unsupported-guard": source_text.replace("    scf.if %active {", "    %combined = arith.ori "
                    "%active, %active : i1\n    scf.if %combined {", 1),
            }
            for name, text in sources.items():
                candidate = Path(directory) / (name + ".pto")
                candidate.write_text(text, encoding="utf-8")
                rejected_original(tool, candidate, name)
    elif "_stages_" in Path(source).name:
        negative_stages(tool, source)
    elif "row_expand" in Path(source).name:
        row_expand_negatives(tool, source)
    elif banks == 2:
        negative_sources(tool, source)
    print("verified banked output and F* covers, parity, zero/short/incomplete trips, "
          "union closure equality, physical coverage and three protocol mutations")


if __name__ == "__main__":
    main()
