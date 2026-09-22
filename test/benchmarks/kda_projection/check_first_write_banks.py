# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""Check separate operand-bank deadlines in actual native emitted plans.

Uses the explicit-command issue/completion oracle. This is a host placement
regression, not device correctness or a model of queue internals.
"""
import argparse
from pathlib import Path
from compare_order import run

PIPES = {"tstore": "FIX", "textract": "MTE1", "tmatmul.acc": "M"}


def check(baseline, candidate):
    cases = removed = 0
    for entries in (0, 1, 2, 5):
        bindings = {"%arg1": entries}
        old, new = [run(path, "first_write_banks", bindings, PIPES)
                    for path in (baseline, candidate)]
        assert old.names == new.names, "payload sequence changed"
        for before, after in zip(old.relations(), new.relations()):
            assert not after & ~before, "first-write refinement added payload order"
            removed += (before & ~after).bit_count()
        extracts = []
        for i, (name, _) in enumerate(new.names):
            if name == "textract":
                extracts.append(i)
            elif name == "tmatmul.acc" and len(extracts) == 4:
                issue = new.nodes[2 * i]
                for j in extracts[:2]:
                    assert new.order[issue] & (1 << (2 * j + 1)), "bank A readiness missing"
                for j in extracts[2:]:
                    assert not new.order[issue] & (1 << (2 * j + 1)), "bank B gates bank A compute"
                extracts.clear()
                cases += 1
        assert not extracts
    assert cases == 16, cases
    print(f"first-write-banks paths=4 bank-deadlines={cases} removed={removed} added=0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    check(args.baseline, args.candidate)
