# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent actual command graph for the exact affine counted-reader route."""
import sys
from pathlib import Path
from ptoas.mlir.ir import Context, Module, StringAttr
from ptoas.mlir.dialects import pto
from check_banked_xor import command_graph, execute, expected
from check_periodic_physical import rearm, rejected, check_costs


def main():
    with Context() as context:
        pto.register_dialect(context)
        source = Module.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
        output = Module.parse(Path(sys.argv[2]).read_text(encoding="utf-8"))
        original = source.body.operations[0].operation
        compiled = output.body.operations[0].operation
        assert StringAttr(compiled.attributes["pto.frontier.physical_status"]).value == "certified-general-pools"
        for n in (-1, 0, 1, 2, 4, 6):
            for offset in (-2, 0, 1, 3):
                inputs = {1: n, 2: offset}
                required = expected(original, n, True, scalar_inputs=inputs)
                commands = execute(compiled, n, correlated=True, scalar_inputs=inputs)
                assert command_graph(commands) == required, f"general ordering changed at n={n}, c={offset}"
                rearm(commands)
                for kind in ("set", "wait"):
                    position = next((i for i, command in enumerate(commands) if command[0] == kind), None)
                    if position is not None:
                        mutation = commands[:position] + commands[position + 1:]
                        assert rejected(mutation, required), "missing general endpoint escaped oracle"
        check_costs(sys.argv[3], {"general_physical"})
        print("general affine physical order, matching, reuse and endpoint mutations verified")


if __name__ == "__main__":
    main()
