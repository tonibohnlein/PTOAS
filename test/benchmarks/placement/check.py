# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Local ordinary-prefix oracle for generate.py's dense vector witnesses.

Reuses the independent issue/finish graph and scalar interpreter, not OAHS's
causal analysis. The parser-only tabs -> textract spelling adapter below means
an out-of-place read/write; this subclass assigns it to V. GM visibility and
numerics are checked by the device task, not this local-footprint oracle.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "oahs"))
from check_carried_slot_trace import Trace, execute, parse


class VectorTrace(Trace):
    def __init__(self):
        super().__init__()
        self.operations = []

    def payload(self, name, effects, context, acc_order=False):
        pipe = {"tload": "MTE2", "textract": "V", "tstore": "MTE3"}[name]
        launch = self.issue(pipe)
        finish = self.vertex([launch])
        for _, old_effects, _, old_finish in self.operations:
            for (space, address, size), write in effects:
                for (old_space, old_address, old_size), old_write in old_effects:
                    if (space == old_space and (write or old_write)
                            and max(address, old_address) < min(address + size, old_address + old_size)):
                        assert self.ancestors[launch] & (1 << old_finish), ("missing memory", name, effects)
                        self.required_checks += 1
        self.finishes.setdefault(pipe, []).append(finish)
        self.operations.append((name, effects, launch, finish))

    def relations(self):
        return {(i, j) for j, (_, _, launch, _) in enumerate(self.operations)
                for i, (_, _, _, finish) in enumerate(self.operations[:j])
                if self.ancestors[launch] & (1 << finish)}


def run(path, active):
    lines = path.read_text().replace("pto.tabs", "pto.textract").splitlines()
    at = next(i for i, line in enumerate(lines) if "func.func @" in line)
    arguments = re.findall(r"(%\w+): i32", lines[at])
    nodes, _ = parse(lines, at + 1)
    trace = VectorTrace()
    execute(nodes, {name: active for name in arguments}, trace)
    assert not trace.live, "unconsumed event"
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="micro/<kernel>/<arm>/plan.pto tree")
    args = parser.parse_args()
    rows = []
    for kernel in ("source_gap", "deferred_ack", "class_invariant"):
        for active in ([0, 1] if kernel == "deferred_ack" else [0]):
            candidate = run(args.directory / kernel / "candidate/plan.pto", active)
            # Class invariance extends construction coverage: default fails.
            # Existing InsertSync's output lacks a same-pipe WAW edge required
            # here. Keep it as a device comparator, not a certified baseline.
            if kernel == "class_invariant":
                rows.append(dict(kernel=kernel, active=active, reference=None,
                                 conflicts=candidate.required_checks,
                                 after_pairs=sum(n for (kind, _, _), n in candidate.sync_counts.items()
                                                 if kind == "set_flag")))
                continue
            reference = "default"
            baseline = run(args.directory / kernel / reference / "plan.pto", active)
            identity = lambda trace: [(name, effects) for name, effects, _, _ in trace.operations]
            assert identity(candidate) == identity(baseline), "changed local payload"
            added = candidate.relations() - baseline.relations()
            removed = baseline.relations() - candidate.relations()
            if kernel != "class_invariant":
                assert not added, (kernel, "added ordering", added)
            if kernel == "source_gap":
                # R store y (2), Q last x reader (4), P overwrite x (6).
                assert (2, 6) in baseline.relations()
                assert (2, 6) not in candidate.relations()
            row = dict(kernel=kernel, active=active, reference=reference,
                       added=len(added), removed=len(removed), conflicts=candidate.required_checks,
                       before_pairs=sum(n for (kind, _, _), n in baseline.sync_counts.items() if kind == "set_flag"),
                       after_pairs=sum(n for (kind, _, _), n in candidate.sync_counts.items() if kind == "set_flag"))
            rows.append(row)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
