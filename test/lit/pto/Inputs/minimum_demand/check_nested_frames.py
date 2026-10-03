# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Original-IR oracle for compact nested/re-entry frame certificates."""
from pathlib import Path
import itertools
import math
import sys
from ptoas.mlir.ir import Context, Module, DictAttr, IntegerAttr, DenseI64ArrayAttr, BoolAttr, StringAttr
from ptoas.mlir.dialects import pto
from check_banked_xor import run, execute, command_graph, expected, independent_covers, close, requirement_graph


def admitted(piece, coordinates):
    for name in ("equalities", "inequalities"):
        rows = list(DenseI64ArrayAttr(piece[name]))
        assert len(rows) % len(coordinates) == 0
        for begin in range(0, len(rows), len(coordinates)):
            value = sum(a * b for a, b in zip(rows[begin:begin + len(coordinates)], coordinates))
            if (value != 0 if name == "equalities" else value < 0):
                return False
    return True


def check(tool, source):
    compiled = run(tool, source, "frontier-synch")
    assert compiled.returncode == 0, compiled.stderr
    with Context() as context:
        pto.register_dialect(context)
        original = Module.parse(Path(source).read_text(encoding="utf-8"))
        output = Module.parse(compiled.stdout)
        source_function = list(original.body.operations)[0]
        function = list(output.body.operations)[0]
        summary = DictAttr(DictAttr(function.attributes["pto.frontier.analysis"])["single_stream"])
        repeated = StringAttr(summary["protocol"]).value.startswith("repeated-")
        parameters = list(DenseI64ArrayAttr(summary["parameter_arguments"]))
        assert len(parameters) >= 2
        bool_arguments = [arg for arg in source_function.regions[0].blocks[0].arguments if str(arg.type) == "i1"]
        activities = (False, True) if bool_arguments else (None,)
        cases = 0
        constants = [None if parameter >= 0 else IntegerAttr(DictAttr(frame)["parameter_constant"]).value
                     for parameter, frame in zip(parameters, summary["frames"])]
        domains = [(-2, 0, 1, 2, 3) if value is None else (value,) for value in constants]
        extreme_cases = [tuple(value if value is not None else (-(1 << 63) if axis == negative else 2)
                               for axis, value in enumerate(constants))
                         for negative, parameter in enumerate(parameters) if parameter >= 0]
        bound_cases = itertools.chain(itertools.product(*domains), extreme_cases)
        for bounds in bound_cases:
            if math.prod(max(0, bound) for bound in bounds) > 12:
                continue
            scalars = {parameter: bound for parameter, bound in zip(parameters, bounds) if parameter >= 0}
            if any(parameter >= 0 and scalars[parameter] != bound
                   for parameter, bound in zip(parameters, bounds)):
                continue
            for active in activities:
                occurrences = []
                execute(source_function, 0, active=active, scalar_inputs=scalars, occurrences_out=occurrences)
                coordinates = list(bounds) + ([int(active)] if bool_arguments else []) + [1]
                covers = set()
                for raw in summary["minimum"]:
                    piece = DictAttr(raw)
                    for a, (source_site, source_coordinate) in enumerate(occurrences):
                        for b, (target_site, target_coordinate) in enumerate(occurrences):
                            if source_site == IntegerAttr(piece["source"]).value and \
                                    target_site == IntegerAttr(piece["consumer"]).value and \
                                    admitted(piece, source_coordinate + target_coordinate + coordinates):
                                covers.add((a, b))
                assert covers == independent_covers(source_function, 0, active, scalars), (bounds, active, covers)
                commands = execute(function, 0, active=active, scalar_inputs=scalars)
                projected = command_graph(commands)
                reference = expected(source_function, 0, False, active, scalars)
                assert projected == reference, (bounds, active, "payload projection")
                correlated = expected(source_function, 0, True, active, scalars)
                assert all(row <= projected[i] for i, row in enumerate(correlated))
                body = [i for i, (_, coordinate) in enumerate(occurrences) if coordinate]
                closure = close(requirement_graph(source_function, 0, False, active, scalars))
                for raw in summary["frame_port_queries"]:
                    query = DictAttr(raw)
                    actual = any(admitted(DictAttr(guard), coordinates) for guard in query["guard"])
                    if not body:
                        assert not actual, (bounds, active, "absent port")
                        continue
                    a = body[-1] if BoolAttr(query["source_last"]).value else body[0]
                    b = body[-1] if BoolAttr(query["target_last"]).value else body[0]
                    a = 2 * a + int(BoolAttr(query["source_completion"]).value)
                    b = 2 * b + int(BoolAttr(query["target_completion"]).value)
                    assert actual == (a == b or b in closure[a]), (bounds, active, "event port")
                if body:
                    # READY per tuple, RELEASE except globally last tuple, final once.
                    generations = 2 * math.prod(max(0, bound) for bound in bounds) if repeated else 2
                    assert sum(kind == "set" for kind, _ in commands) == generations
                    assert sum(kind == "wait" for kind, _ in commands) == generations
                    missing = list(commands)
                    del missing[next(i for i, (kind, _) in enumerate(missing) if kind == "wait")]
                    try:
                        mutated = command_graph(missing)
                    except ValueError:
                        pass
                    else:
                        assert mutated != reference
                else:
                    assert not any(kind in ("set", "wait") for kind, _ in commands)
                cases += 1
        print(f"nested original-IR frame/cover/all-event/projection checks: {cases} valuations")


if __name__ == "__main__":
    check(*sys.argv[1:])
