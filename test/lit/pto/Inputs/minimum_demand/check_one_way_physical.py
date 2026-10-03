# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent actual-IR checker for the finite eight-cover repair fixture."""
from collections import Counter
from pathlib import Path
import re
import sys

from ptoas.mlir.ir import ArrayAttr, Context, DictAttr, IntegerAttr, Module, StringAttr
from ptoas.mlir.dialects import pto
import check_banked_xor as reference


def integer(attribute):
    return IntegerAttr(attribute).value


def string(attribute):
    return StringAttr(attribute).value


def records(attribute):
    return list(ArrayAttr(attribute))


def source_profile(reach, pipes):
    sources = [index for index, pipe in enumerate(pipes) if pipe == "PIPE_MTE2"]
    return [max([0] + [rank + 1 for rank, source in enumerate(sources)
                       if 2 * consumer in reach[2 * source + 1]])
            for consumer in range(len(pipes))]


def exhaustive_excess(covers, consumers, capacity):
    best = None
    # Enumerate every contiguous partition with <=capacity blocks. Compute
    # its actual acquired prefix at each consumer, without DP block weights.
    for mask in range(1 << (len(covers) - 1)):
        cuts = [0] + [gap + 1 for gap in range(len(covers) - 1) if mask & (1 << gap)] + [len(covers)]
        if len(cuts) - 1 > capacity:
            continue
        cost = 0
        for consumer in range(1, consumers + 1):
            required = max([0] + [source for source, target in covers if target <= consumer])
            acquired = max([0] + [covers[last - 1][0] for first, last in zip(cuts, cuts[1:])
                                 if covers[first][1] <= consumer])
            assert acquired >= required
            cost += acquired - required
        best = cost if best is None else min(best, cost)
    assert best is not None
    return best


def run(source_path, output_path):
    with Context() as context:
        pto.register_dialect(context)
        source_module = Module.parse(Path(source_path).read_text(encoding="utf-8"))
        output_module = Module.parse(Path(output_path).read_text(encoding="utf-8"))
        original = list(source_module.body.operations)[0]
        output = list(output_module.body.operations)[0]
        # The existing test interpreter's legacy arg2 default is a loop bound;
        # this fixture instead supplies eight readonly GM arguments explicitly.
        valuations = {index: {"gm"} for index, _ in enumerate(original.regions[0].blocks[0].arguments)}
        original_effects, output_effects = [], []
        original_commands = reference.execute(original, 0, effects=original_effects, scalar_inputs=valuations)
        output_commands = reference.execute(output, 0, effects=output_effects, scalar_inputs=valuations)
        assert original_effects == output_effects, "repair changed original payload effects/order"
        pipes = [argument for kind, argument in original_commands if kind == "payload"]
        assert pipes == ["PIPE_MTE2"] * 8 + ["PIPE_V"] * 8
        assert [argument for kind, argument in output_commands if kind == "payload"] == pipes
        original_barriers = [command for command in original_commands if command[0] == "barrier"]
        assert original_barriers == [("barrier", "PIPE_MTE2")] * 7
        assert [command for command in output_commands if command == ("barrier", "PIPE_MTE2")] == original_barriers
        # The required baseline includes the original command projection,
        # independent of storage conflicts or any production native recipe.
        storage = reference.requirement_graph(original, 0, False, scalar_inputs=valuations)
        original_projection = reference.command_graph(original_commands)
        required = reference.close([a | b for a, b in zip(storage, original_projection)])
        metadata = DictAttr(output.attributes["pto.frontier.analysis"])
        demands = [tuple(integer(item) for item in records(row)) for row in records(metadata["demands"])]
        reconstructed = reference.native(pipes)
        for source, consumer in demands:
            assert 0 <= source < len(pipes) and 0 <= consumer < len(pipes)
            reconstructed[2 * source + 1].add(2 * consumer)
        assert reference.close(reconstructed) == required, "retained demands lost source baseline or storage"
        actual = reference.command_graph(output_commands)
        assert all(row <= actual[index] for index, row in enumerate(required)), "repair lost a requirement"
        assert all(index not in row for index, row in enumerate(actual)), "repaired cycle"
        physical = DictAttr(output.attributes["pto.frontier.physical"])
        assert string(output.attributes["pto.frontier.physical_status"]) == "certified-finite-one-way-repair"
        assert string(physical["permitted_family"]) == "explicit finite-one-way restriction"
        assert string(physical["guarantee"]) == "restricted minimality and minimum total repair excess"
        assert integer(physical["fixed_required"]) == 8
        pool_rows = records(physical["pools"])
        assert len(pool_rows) == 1
        pool = DictAttr(pool_rows[0])
        eligible = {integer(item) for item in records(pool["eligible"])}
        reserved = {integer(item) for item in records(pool["reserved"])}
        assert eligible == set(range(6)) and not eligible & reserved
        assert integer(pool["required"]) == 6
        publications = Counter(argument for kind, argument in output_commands if kind == "set")
        acquisitions = Counter(argument for kind, argument in output_commands if kind == "wait")
        assert publications == acquisitions and len(publications) == 6 and set(publications.values()) == {1}
        ids = set()
        for source, target, event in publications:
            assert (source, target) == ("PIPE_MTE2", "PIPE_V")
            match = re.search(r"EVENT_ID(\d+)", event)
            assert match is not None
            identity = int(match.group(1))
            assert identity in eligible and identity not in reserved
            ids.add(identity)
        assert len(ids) == 6, "numeric IDs aliased distinct logical publications"
        covers = [(source + 1, consumer - 7) for source, consumer in demands
                  if pipes[source] != pipes[consumer]]
        assert covers == [(rank, rank) for rank in range(1, 9)]
        supplied = [tuple(integer(item) for item in records(row)) for row in records(physical["required_covers"])]
        assert supplied == [(rank - 1, rank + 7) for rank in range(1, 9)]
        measured = sum(a - b for a, b in zip(source_profile(actual, pipes), source_profile(required, pipes)))
        assert measured == int(string(physical["repair_excess"])) == 2
        assert measured == exhaustive_excess(covers, 8, len(eligible)), "repair is not minimum excess"
        # No additional Q completion prerequisite is hidden in the scalar sum.
        for source in range(8, 16):
            for consumer in range(16):
                assert (2 * consumer in actual[2 * source + 1]) == (2 * consumer in required[2 * source + 1])
        mutations = 0
        for position, (kind, argument) in enumerate(output_commands):
            if kind not in ("set", "wait") and (kind, argument) != ("barrier", "PIPE_MTE2"):
                continue
            changed = output_commands[:position] + output_commands[position + 1:]
            try:
                projection = reference.command_graph(changed)
            except ValueError:
                assert kind in ("set", "wait"), "original drain removal changed matching"
            else:
                assert any(not row <= projection[index] for index, row in enumerate(required)), \
                    "missing endpoint/original drain still covers required closure"
            mutations += 1
        assert mutations == 19
        print("finite physical one-way: eight selected covers, six paired eligible IDs, "
              f"minimum excess {measured}; {mutations} missing endpoint/drain mutations")


if __name__ == "__main__":
    run(*sys.argv[1:])
