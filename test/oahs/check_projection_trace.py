# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Concrete down_proj local memory/event and operand-overlap regression.

Executes original scalar conditions and emitted words. GM arguments are separate;
this checks local byte footprints, not numerical execution or device latency.
BF16 footprints are 16 bits; no native ACC exemption is used for this fixture.
"""
import argparse
import re
from pathlib import Path
from check_carried_slot_trace import Trace, execute, parse


def overlap(a, b):
    return a[0] == b[0] and a[1] < b[1] + b[2] and b[1] < a[1] + a[2]


class ProjectionTrace(Trace):
    def __init__(self, check_mat=False, check_release=False):
        super().__init__()
        self.early_checks = 0
        self.reuse_checks = 0
        self.starts = []
        self.check_mat = check_mat
        self.mat_checks = 0
        self.check_release = check_release
        self.release_checks = 0

    def payload(self, name, effects, context, acc_order=False):
        pipe = {"tload": "MTE2", "textract": "MTE1", "tmatmul": "M",
                "tmatmul.acc": "M", "tstore": "FIX"}[name]
        issued = self.issue(pipe)
        self.starts.append(issued)
        done = self.vertex([issued])
        before = self.ancestors[issued]
        for old in self.payloads:
            for a, aw in old[2]:
                for b, bw in effects:
                    if (aw or bw) and overlap(a, b):
                        assert before & (1 << old[1]), ("missing local completion", old, name, b)
                        self.required_checks += 1
        if self.check_release and name == "tload" and any(
                write and a[0] == "mat" and a[1] == 0 for a, write in effects):
            # The next first MAT pair must not wait for reads of the preceding
            # second pair. Its own previous readers still remain mandatory.
            for old in self.payloads:
                if old[0] != "textract" or len(old[3]) != len(context) + 1:
                    continue
                if (old[3][:-2] != context[:-1] or old[3][-2][0] != context[-1][0]
                        or old[3][-2][1] + 2 != context[-1][1]):
                    continue
                if any(not write and a[0] == "mat" and a[1] >= 196608
                       for a, write in old[2]):
                    assert not before & (1 << old[1]), "second-child MAT reader gates first-pair refill"
                    self.release_checks += 1
        if pipe == "M":
            inputs = [a for a, write in effects if not write and a[0] in ("left", "right")]
            required = [i for i, old in enumerate(self.payloads) if old[0] == "textract" and
                        any(write and any(overlap(a, b) for b in inputs) for a, write in old[2])]
            assert required, "missing operand preparation"
            for old in self.payloads[max(required) + 1:]:
                if old[0] == "textract" and old[3] == context:
                    assert not before & (1 << old[1]), "later operand preparation gates early matrix use"
                    self.early_checks += 1
        if name == "textract" and self.check_mat:
            # This fixture's first MAT B bank is [65536, 196608). Its
            # readiness must not observe the later A1/B1 loads. The second
            # child's entry placement is a separate, still-open deadline.
            reads = [a for a, write in effects if not write and
                     a[0] == "mat" and a[1] == 65536]
            if reads:
                producers = [i for i, old in enumerate(self.payloads) if
                             old[0] == "tload" and any(write and any(overlap(a, b) for b in reads)
                                                       for a, write in old[2])]
                assert producers, "missing original B producer"
                producer = producers[-1]
                for old in self.payloads[producer + 1:]:
                    if old[0] == "tload" and old[3] == self.payloads[producer][3]:
                        assert not before & (1 << old[1]), "later MAT load gates early B extraction"
                        self.mat_checks += 1
        if name == "textract":
            outputs = [a for a, write in effects if write and a[0] in ("left", "right")]
            readers = [i for i, old in enumerate(self.payloads) if old[0].startswith("tmatmul") and
                       any(not write and any(overlap(a, b) for b in outputs) for a, write in old[2])]
            if readers:
                previous = self.payloads[readers[-1]]
                # Consecutive uses within the same original child invocation.
                if (previous[3][:-1] == context[:-1] and previous[3][-1][0] == context[-1][0]
                        and previous[3][-1][1] < context[-1][1]):
                    for old in self.payloads[readers[-1] + 1:]:
                        if old[0].startswith("tmatmul") and old[3] == previous[3]:
                            assert not before & (1 << old[1]), "other bank's reader gates this bank's refill"
                            self.reuse_checks += 1
        self.payloads.append((name, done, effects, context))
        self.finishes.setdefault(pipe, []).append(done)


def run(path, chunks, tiles, check_mat=False, check_release=False):
    # Only normalize spelling for the shared scalar/footprint parser. The
    # payload checker above deliberately ignores its optional ACC shortcut.
    source = path.read_text().replace("xbf16", "xf16")
    source = re.sub(r"%c-(\d+)", r"%cneg\1", source)
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if "func.func @down_proj" in line) + 1
    nodes, _ = parse(lines, start)
    trace = ProjectionTrace(check_mat=check_mat, check_release=check_release)
    execute(nodes, {"%arg3": 20 * tiles, "%arg4": chunks, "%arg5": 0, "%arg6": 0,
                    "%arg7": 4352, "%arg8": 0, "%arg9": 20}, trace)
    assert not trace.live, "unconsumed event at invocation exit"
    assert trace.sync_counts.get(("barrier", "ALL", None)) == 1
    if chunks and tiles:
        assert trace.early_checks and trace.reuse_checks, "operand quality assertions were not exercised"
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--require-early-mat", action="store_true",
                        help="require first-consumer placement before unrelated later loads")
    parser.add_argument("--require-early-release", action="store_true")
    args = parser.parse_args()
    checks = early = reuse = mat = releases = 0
    for chunks in (0, 1, 2, 3, 4, 17):
        for tiles in (0, 1, 2):
            trace = run(args.path, chunks, tiles, check_mat=args.require_early_mat,
                        check_release=args.require_early_release)
            checks += trace.required_checks
            early += trace.early_checks
            reuse += trace.reuse_checks
            mat += trace.mat_checks
            releases += trace.release_checks
    if args.require_early_mat:
        assert mat, "MAT placement assertions were not exercised"
        print(f"projection MAT target: {mat} unrelated-load edges absent")
    if args.require_early_release:
        assert releases, "MAT release assertions were not exercised"
        print(f"projection MAT release: {releases} unrelated-child edges absent")
    print(f"projection: 18 traces, {checks} local conflicts, {early} early-readiness and {reuse} reuse checks")


if __name__ == "__main__":
    main()
