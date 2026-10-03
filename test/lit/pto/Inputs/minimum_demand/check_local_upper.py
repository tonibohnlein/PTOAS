# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent emitted-command/selected-closure checks for delayed local replacement."""
import sys
from pathlib import Path
from ptoas.mlir.ir import ArrayAttr, Context, DictAttr, IntegerAttr, Module, StringAttr
from ptoas.mlir.dialects import pto
from check_banked_xor import close, command_graph, execute, expected, native
from check_periodic_physical import rearm, rejected
from check_symbolic_physical import integers, selected_closure


def postorder(operation):
    result = []
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                result.extend(postorder(child.operation))
    result.append(operation)
    return result


def finite_selected(metadata, original, commands, occurrences, inputs, values):
    pipes = [argument for kind, argument in commands if kind == "payload"]
    graph = native(pipes)
    executed = {site: index for index, (site, _) in enumerate(occurrences)}
    predicates = []
    if StringAttr(metadata["representation"]).value == "finite-guarded-circuit":
        operations = postorder(original)
        for record in ArrayAttr(metadata["predicates"]):
            node = DictAttr(record)
            kind = IntegerAttr(node["kind"]).value
            first, second = (IntegerAttr(node[field]).value for field in ("first", "second"))
            if kind == 0:
                value = False
            elif kind == 1:
                value = True
            elif kind == 2:
                if "argument" in node:
                    value = bool(inputs[IntegerAttr(node["argument"]).value])
                else:
                    anchor = operations[IntegerAttr(node["value_anchor"]).value]
                    value = bool(values[anchor.results[IntegerAttr(node["result"]).value]])
            elif kind == 3:
                value = not predicates[first]
            elif kind == 4:
                value = predicates[first] and predicates[second]
            elif kind == 5:
                value = predicates[first] or predicates[second]
            else:
                raise AssertionError("unknown predicate kind")
            predicates.append(value)
    for record in ArrayAttr(metadata["demands"]):
        edge = integers(record)
        if len(edge) == 3 and not predicates[edge[2]]:
            continue
        source, consumer = edge[:2]
        assert source in executed and consumer in executed, "selected absent endpoint"
        graph[2 * executed[source] + 1].add(2 * executed[consumer])
    return close(graph)


def main():
    with Context() as context:
        pto.register_dialect(context)
        source = Module.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
        output = Module.parse(Path(sys.argv[2]).read_text(encoding="utf-8"))
        pairs = zip(source.body.operations, output.body.operations, strict=True)
        for source_view, output_view in pairs:
            original, compiled = source_view.operation, output_view.operation
            metadata = DictAttr(compiled.attributes["pto.frontier.analysis"])
            assert StringAttr(metadata["selected_closure"]).value == "sound-upper"
            for n in (-1, 0, 1, 2, 3):
                for active in (False, True):
                    inputs = {2: n, 3: int(active)}
                    occurrences, values = [], {}
                    old_commands = execute(original, n, correlated=True, active=active, scalar_inputs=inputs,
                                           occurrences_out=occurrences, values_out=values)
                    required = expected(original, n, True, active=active, scalar_inputs=inputs)
                    if StringAttr(metadata["representation"]).value == "signed-piece-union":
                        selected = selected_closure(metadata, occurrences, inputs)
                    else:
                        selected = finite_selected(metadata, original, old_commands, occurrences, inputs, values)
                    covered = all(row <= selected[index] for index, row in enumerate(required))
                    assert covered, "lost original requirement"
                    commands = execute(compiled, n, correlated=True, active=active, scalar_inputs=inputs)
                    assert command_graph(commands) == selected, "physical plan differs from selected upper"
                    rearm(commands)
                    local = next((index for index, (kind, pipe) in enumerate(commands)
                                  if kind == "barrier" and pipe != "PIPE_ALL"), None)
                    if local is not None:
                        mutation = commands[:local] + commands[local + 1:]
                        assert rejected(mutation, selected), "missing local barrier escaped"
        print("local upper original coverage, selected-order equality, guarded adjacency and mutations verified")


if __name__ == "__main__":
    main()
