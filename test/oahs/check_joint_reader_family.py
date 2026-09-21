# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare the six prepared projection cases in a paired corpus directory.

Each pypto_lib__prefill_fwd__N contains baseline/plan.pto and candidate/plan.pto.
Uses the independent local-footprint/event oracle with no ACC shortcut. Aliased
module rows are covered by corpus construction; the paths here are finite tests.
"""
import argparse
import json
import re
from pathlib import Path
from check_carried_slot_trace import Trace, execute, parse
from diagnose_projection_last_reader import payload_order


class ProjectionFamilyTrace(Trace):
    def __init__(self):
        super().__init__()
        self.starts = []

    def payload(self, name, effects, context, acc_order=False):
        self.starts.append(len(self.ancestors))
        super().payload(name, effects, context, False)


def run(path, arguments):
    source = re.sub(r"%c-(\d+)", r"%cneg\1", path.read_text().replace("xbf16", "xf16"))
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if "func.func @" in line) + 1
    nodes, _ = parse(lines, start)
    trace = ProjectionFamilyTrace()
    execute(nodes, dict(arguments), trace)
    assert not trace.live, "unconsumed event at invocation exit"
    return trace


def cases():
    for tiles in (0, 1, 2):
        for chunks in (0, 1, 2, 3, 4, 17):
            yield 0, f"{tiles}tiles_{chunks}chunks", {
                "%arg3": 20 * tiles, "%arg4": chunks, "%arg5": 0, "%arg6": 0,
                "%arg7": 4352, "%arg8": 0, "%arg9": 20}
    for number, core_arg, cores in ((2, 8, (0, 7, 16, 17, 23)),
                                   (4, 7, (0, 3, 4, 7, 8)),
                                   (7, 5, (0, 19, 20)), (9, 5, (0, 19, 20)),
                                   (6, 5, (744, 768, 792))):
        for core in cores:
            env = {f"%arg{i}": 0 for i in range(10)}
            env.update({"%arg4": 0, "%arg6": 5120, "%arg7": 5120})
            env[f"%arg{core_arg}"] = core
            if number in (7, 9):
                env["%arg3"], env["%arg4"] = 0, 5120
            if number == 6:
                env["%arg3"], env["%arg4"] = 16, 16
            yield number, f"start{core}", env


def check(corpus, output):
    rows = []
    for number, label, arguments in cases():
        name = f"pypto_lib__prefill_fwd__{number}"
        a = run(corpus / name / "baseline/plan.pto", arguments)
        b = run(corpus / name / "candidate/plan.pto", arguments)
        identity = lambda t: [(p[0], p[2], tuple(v for _, v in p[3])) for p in t.payloads]
        assert identity(a) == identity(b), (name, label, "payload change")
        assert a.sync_counts == b.sync_counts, (name, label, "executed sync/fence population change")
        before, after = payload_order(a), payload_order(b)
        assert not after - before, (name, label, "added complete payload ordering")
        rows.append(dict(case=name, path=label, conflicts=b.required_checks,
                         before=len(before), after=len(after),
                         removed=len(before - after), added=len(after - before)))
    report = dict(paths=len(rows), conflicts=sum(r["conflicts"] for r in rows),
                  removed=sum(r["removed"] for r in rows), added=0, rows=rows)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.corpus, args.output)
