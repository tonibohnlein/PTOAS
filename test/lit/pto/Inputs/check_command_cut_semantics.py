# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Pin asymmetric command order at a shared original payload cut."""
from check_logical_insertion import closure_with_commands


def check_cut(barrier):
    # A(P), B(Q), C(Q), D(R); communications A->C and B->D.
    # The outgoing SET after B precedes the incoming WAIT before C in source
    # order. Neither that order nor native I_B->I_C proves C_B->I_C.
    pipes = [0, 1, 1, 2]
    commands = [
        {"kind": "set", "gap": 1, "pipe": 0, "identity": (0,)},
        {"kind": "set", "gap": 2, "pipe": 1, "identity": (1,)},
    ]
    if barrier:
        commands.append({"kind": "barrier", "gap": 2, "pipe": 1})
    commands.extend([
        {"kind": "wait", "gap": 2, "pipe": 1, "identity": (0,)},
        {"kind": "wait", "gap": 3, "pipe": 2, "identity": (1,)},
    ])
    graph = closure_with_commands(pipes, commands, include_commands=True)
    outgoing = 2 * len(pipes) + 1
    incoming = 2 * len(pipes) + 2 + int(barrier)
    assert bool(graph[outgoing] & (1 << incoming)) == barrier
    assert bool(graph[3] & (1 << 4)) == barrier  # C_B -> I_C
    assert graph[2] & (1 << 4)                  # native I_B -> I_C
    assert graph[3] & (1 << 5)                  # native C_B -> C_C
    assert graph[1] & (1 << 4)                  # A -> C matching
    assert graph[3] & (1 << 6)                  # B -> D matching
    assert not (graph[incoming] & (1 << outgoing))
    assert not (graph[1] & (1 << 6))            # no spurious A -> D


def main():
    check_cut(False)
    check_cut(True)
    print("command-cut semantics: SET/WAIT independence and local-barrier order passed")


if __name__ == "__main__":
    main()
