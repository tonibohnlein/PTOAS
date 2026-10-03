# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent upper-closure, command and excess oracle for the V-only fixture.

Reuse the test command contracts and original byte/range effects; neither the
production origin/reduction engine nor its numerical frontier queries run here.
Selected metadata is checked against its declared generators and actual output.
"""
from pathlib import Path
import sys

from ptoas.mlir.ir import ArrayAttr, Context, DictAttr, IntegerAttr, Module, StringAttr
from ptoas.mlir.dialects import pto
import check_banked_xor as reference


def operations(block):
    for view in block.operations:
        operation = view.operation
        yield operation
        for region in operation.regions:
            for nested in region.blocks:
                yield from operations(nested)


def integer(attribute):
    return IntegerAttr(attribute).value


def string(attribute):
    return StringAttr(attribute).value


def records(attribute):
    return list(ArrayAttr(attribute))


def selected_graph(metadata, trips, body_size, demands):
    count = max(0, trips)
    if demands:
        graph = reference.native(["PIPE_V"] * (count * body_size))
        edges = [(2 * integer(row[0]) + 1, 2 * integer(row[1]), int(string(row[2])))
                 for row in (records(item) for item in records(metadata["demands"]))]
    else:
        graph = [set() for _ in range(2 * count * body_size)]
        edges = [(integer(row[0]), integer(row[1]), int(string(row[2])))
                 for row in (records(item) for item in records(metadata["closure_graph"]))]
    for source, consumer, distance in edges:
        assert 0 <= source < 2 * body_size and 0 <= consumer < 2 * body_size
        assert distance >= 0, "unsupported backwards quotient generator"
        for period in range(count):
            target = period + distance
            if target < count:
                graph[2 * body_size * period + source].add(2 * body_size * target + consumer)
    return reference.close(graph)


def profile(reach):
    # One physical V pipe; ranks are executed payload ordinals, not loop indices.
    return [max([0] + [source + 1 for source in range(len(reach) // 2)
                       if 2 * consumer in reach[2 * source + 1]])
            for consumer in range(len(reach) // 2)]


def run(source_path, output_path):
    with Context() as context:
        pto.register_dialect(context)
        original_module = Module.parse(Path(source_path).read_text(encoding="utf-8"))
        output_module = Module.parse(Path(output_path).read_text(encoding="utf-8"))
        original = list(original_module.body.operations)[0]
        output = list(output_module.body.operations)[0]
        source_block = original.regions[0].blocks[0]
        output_block = output.regions[0].blocks[0]
        original_ops = list(operations(source_block))
        source_loops = [operation for operation in original_ops if operation.name == "scf.for"]
        assert len(source_loops) == 1
        bound = source_loops[0].operands[1]
        bound_indices = [index for index, value in enumerate(source_block.arguments) if value == bound]
        assert len(bound_indices) == 1 and str(bound.type) == "index"
        bound_index = bound_indices[0]
        output_loops = [operation for operation in operations(output_block) if operation.name == "scf.for"]
        assert len(output_loops) == 1 and output_loops[0].operands[1] == output_block.arguments[bound_index]
        payloads = [operation for operation in original_ops if operation.name == "pto.txor"]
        assert len(payloads) == 3, "this independent oracle qualifies the original three-stage V fixture"
        assert all(operation.name not in ("pto.tload", "pto.tstore") for operation in original_ops)
        metadata = DictAttr(output.attributes["pto.frontier.analysis"])
        assert string(metadata["demand_name"]) == "F_hat"
        assert string(metadata["selected_closure"]) == "sound-upper"
        assert string(metadata["quality"]) == "selected-order-covers"
        assert string(metadata["requirements"]) == "shared-modeled"
        assert string(metadata["lower_baseline"]).startswith("native-only;")
        assert string(output.attributes["pto.frontier.physical_status"]) == "certified-periodic-locals"
        assert integer(metadata["vertices"]) == 2 * len(payloads)
        terms = [DictAttr(term) for term in records(metadata["excess_terms"])]
        gamma = int(string(metadata["quadratic_numerator"]))
        assert integer(metadata["quadratic_denominator"]) == 2
        assert gamma == sum(int(string(term["count"])) for term in terms)
        measured_upper = {}
        mutations = 0
        for trips in (-2, 0, 1, 2, 3, 4):
            valuations = {bound_index: trips}
            modeled = reference.expected(original, trips, False, scalar_inputs=valuations)
            selected = selected_graph(metadata, trips, len(payloads), True)
            assert selected == selected_graph(metadata, trips, len(payloads), False), "quotient/generator mismatch"
            commands = reference.execute(output, trips, scalar_inputs=valuations)
            actual = reference.command_graph(commands)
            assert actual == selected, "physical barriers changed the declared selected closure"
            assert all(row <= selected[index] for index, row in enumerate(modeled)), "upper lost a requirement"
            assert all(index not in row for index, row in enumerate(selected)), "selected cycle"
            upper_prefix = profile(selected)
            original_prefix = profile(modeled)
            lower_prefix = profile(reference.close(reference.native(["PIPE_V"] * len(upper_prefix))))
            assert not any(lower_prefix), "native-only lower must add no completion-to-start prerequisite"
            upper_sum = sum(upper_prefix)
            measured_upper[trips] = upper_sum
            excess = sum(a - b for a, b in zip(upper_prefix, original_prefix))
            declared = sum(max(0, int(string(term["count"])) * period + int(string(term["offset"])))
                           for term in terms for period in range(max(0, trips)))
            assert declared == upper_sum and 0 <= excess <= declared, "native-only excess bound false"
            if trips > 0:
                assert excess > 0, "fixture must exercise a strictly conservative selected order"
            for position, command in enumerate(commands):
                if command != ("barrier", "PIPE_V"):
                    continue
                changed = reference.command_graph(commands[:position] + commands[position + 1:])
                assert changed != selected, "removed local acquisition incorrectly accepted"
                mutations += 1
        assert mutations > 0
        assert measured_upper[4] - 2 * measured_upper[3] + measured_upper[2] == gamma
        print("upper physical closure/coverage/native-only excess: six trip valuations, "
              f"{mutations} missing-barrier mutations")


if __name__ == "__main__":
    run(*sys.argv[1:])
