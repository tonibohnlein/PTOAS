# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check compiler-emitted first/final reader composition on down_proj.

The optional baseline comparison checks complete payload relation sets and
executed event/fence counts. This is a finite local/event oracle, not numerical
device execution or a replacement for the native reconstructed-plan checker.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from check_projection_trace import run
from diagnose_projection_last_reader import payload_order


def check(path, baseline=None, output=None):
    rows = []
    for chunks in (0, 1, 2, 3, 4, 17):
        for tiles in (0, 1, 2):
            after = run(path, chunks, tiles, True, True)
            row = dict(chunks=chunks, tiles=tiles, conflicts=after.required_checks,
                       payloads=len(after.payloads),
                       sync_counts={str(k): v for k, v in after.sync_counts.items()})
            if baseline:
                before = run(baseline, chunks, tiles, True, True)
                assert [(p[0], p[2], p[3]) for p in before.payloads] == [
                    (p[0], p[2], p[3]) for p in after.payloads], "payload identity changed"
                assert before.sync_counts == after.sync_counts, "executed synchronization population changed"
                a, b = payload_order(before), payload_order(after)
                assert not b-a, ("added payload ordering", chunks, tiles, sorted(b-a)[:8])
                row.update(before=len(a), after=len(b), removed=len(a-b), added=len(b-a))
            rows.append(row)
    # Real physical lifetime: final B0 extraction no longer gates next A0 load.
    trace = run(path, 4, 1, True, True)
    assert trace.payloads[13][0] == "textract" and trace.payloads[28][0] == "tload"
    assert (("mat", 65536, 131072), False) in trace.payloads[13][2]
    assert (("mat", 0, 65536), True) in trace.payloads[28][2]
    for source in (trace.starts[13], trace.payloads[13][1]):
        for target in (trace.starts[28], trace.payloads[28][1]):
            assert not trace.ancestors[target] & (1 << source), "final B0 read still gates A0 overwrite"
    if baseline:
        before = run(baseline, 4, 1, True, True)
        assert before.ancestors[before.starts[28]] & (1 << before.payloads[13][1]), "baseline is not discriminating"
    source = path.read_text()
    assert re.search(r"arith\.cmpi sle, %\w+, %c128\w* : index", source), "missing original-step final predicate"
    report = dict(scope="compiler-emitted finite payload/event graphs", rows=rows,
                  candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if baseline:
        report["baseline_sha256"] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    if output:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(paths=len(rows), conflicts=sum(r["conflicts"] for r in rows),
                          removed=sum(r.get("removed", 0) for r in rows),
                          added=sum(r.get("added", 0) for r in rows))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check(args.plan, args.baseline, args.output)
