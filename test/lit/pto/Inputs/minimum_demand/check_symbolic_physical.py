# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Unfold actual nested IR and independently evaluate the recorded selected R."""
import sys
from pathlib import Path
from ptoas.mlir.ir import ArrayAttr, Context, DictAttr, IntegerAttr, Module, StringAttr
from ptoas.mlir.dialects import pto
from check_banked_xor import close, command_graph, execute, expected
from check_periodic_physical import rearm, rejected, check_costs


def integers(attribute):
    return [IntegerAttr(value).value for value in ArrayAttr(attribute)]


def selected_closure(metadata, occurrences, scalar_inputs):
    period = int(StringAttr(metadata["period"]).value)
    parameters = [scalar_inputs[index] for index in integers(metadata["parameters"])]
    graph = [set() for _ in range(2 * len(occurrences))]
    for field in ("native", "demands"):
        for record in ArrayAttr(metadata[field]):
            piece = DictAttr(record)
            source, source_kind = integers(piece["source"])
            target, target_kind = integers(piece["target"])
            residues = [int(StringAttr(value).value) for value in ArrayAttr(piece["residues"])]
            for a, (source_site, source_coordinates) in enumerate(occurrences):
                if source_site != source:
                    continue
                for b, (target_site, target_coordinates) in enumerate(occurrences):
                    if target_site != target:
                        continue
                    axes = (source_coordinates, target_coordinates, parameters)
                    flattened = source_coordinates + target_coordinates + parameters
                    if any(value % period != residue for value, residue in zip(flattened, residues, strict=True)):
                        continue
                    valid = True
                    for row_attribute in ArrayAttr(piece["rows"]):
                        row = DictAttr(row_attribute)
                        total = 0
                        for term in ArrayAttr(row["terms"]):
                            role, axis, sign = integers(term)
                            total += sign * (axes[role][axis] // period)
                        if total > int(StringAttr(row["bound"]).value):
                            valid = False
                            break
                    if valid and 2 * a + source_kind != 2 * b + target_kind:
                        graph[2 * a + source_kind].add(2 * b + target_kind)
    return close(graph)


def main():
    with Context() as context:
        pto.register_dialect(context)
        original_module = Module.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
        compiled_module = Module.parse(Path(sys.argv[2]).read_text(encoding="utf-8"))
        pairs = zip(original_module.body.operations, compiled_module.body.operations, strict=True)
        for original_view, compiled_view in pairs:
            original, compiled = original_view.operation, compiled_view.operation
            status = StringAttr(compiled.attributes["pto.frontier.physical_status"]).value
            if status == "certified-finite-guarded-pools":
                for n in (-1, 0, 1, 2):
                    inputs = {2: n}
                    required = expected(original, n, True, scalar_inputs=inputs)
                    commands = execute(compiled, n, correlated=True, scalar_inputs=inputs)
                    assert command_graph(commands) == required, "guarded payload order changed"
                    rearm(commands)
                continue
            assert status == "certified-symbolic-pools"
            metadata = DictAttr(compiled.attributes["pto.frontier.analysis"])
            guarded = str(original.regions[0].blocks[0].arguments[2].type) == "i1"
            cases = [(0, False), (0, True)] if guarded else [(n, None) for n in (-1, 0, 1, 2, 3, 5)]
            for n, active in cases:
                occurrences = []
                inputs = {2: int(active) if guarded else n}
                execute(original, n, correlated=True, active=active, scalar_inputs=inputs, occurrences_out=occurrences)
                selected = selected_closure(metadata, occurrences, inputs)
                required = expected(original, n, True, active=active, scalar_inputs=inputs)
                assert all(row <= selected[index] for index, row in enumerate(required)), "selected R misses storage"
                commands = execute(compiled, n, correlated=True, active=active, scalar_inputs=inputs)
                assert command_graph(commands) == selected, f"selected payload order changed at n={n}"
                rearm(commands)
                for kind in ("set", "wait"):
                    position = next((index for index, command in enumerate(commands) if command[0] == kind), None)
                    if position is not None:
                        mutation = commands[:position] + commands[position + 1:]
                        assert rejected(mutation, selected), "unmatched endpoint mutation escaped oracle"
        if len(sys.argv) > 3:
            check_costs(sys.argv[3], {StringAttr(view.operation.attributes["sym_name"]).value
                                    for view in original_module.body.operations})
        print("symbolic physical selected closure, matching, reuse and endpoint mutations verified")


if __name__ == "__main__":
    main()
