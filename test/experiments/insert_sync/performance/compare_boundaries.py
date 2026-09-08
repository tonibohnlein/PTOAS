#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Compare concrete handoff boundaries, independent of event-ID spelling.

This observes the existing scalar replay. It reports source-prefix completions
required by flags at each physical operation. It is neither a latency model nor
a correctness proof; intrinsic ordering and same-pipe hazards are not inferred.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from measure import SYNC, attrs, children, replay


class Boundaries:
    def __init__(self):
        self.issued = {}
        self.completed = {}
        self.drained = {}
        self.tokens = {}
        self.payload = []
        self.before = []
        self.handoffs = []

    @staticmethod
    def join(left, right):
        for lane, point in right.items():
            left[lane] = max(left.get(lane, -1), point)

    def action(self, name, properties, lane=None, signature=None):
        if name in SYNC:
            pipe = lambda key: re.search(r"PIPE_[A-Z0-9]+", properties[key]).group()
            if name == "pto.barrier":
                all_pipes = pipe("pipe") == "PIPE_ALL"
                if all_pipes:
                    self.join(self.drained, self.issued)
                lanes = list(self.issued) if all_pipes else [pipe("pipe")]
                for target in lanes:
                    self.join(self.completed.setdefault(target, {}), self.issued if all_pipes else
                              {target: self.issued.get(target, -1)})
                return
            source, target = pipe("src_pipe"), pipe("dst_pipe")
            key = (source, target, properties["event_id"])
            if name == "pto.set_flag":
                if key in self.tokens:
                    raise ValueError(f"live token overwritten during concrete replay: {key}")
                snapshot = dict(self.completed.get(source, {}))
                self.join(snapshot, self.drained)
                snapshot[source] = self.issued.get(source, -1)
                self.tokens[key] = snapshot
            else:
                if key not in self.tokens:
                    raise ValueError(f"wait without publication during concrete replay: {key}")
                snapshot = self.tokens.pop(key)
                self.join(self.completed.setdefault(target, {}), snapshot)
                self.handoffs.append({"before_physical": len(self.payload), "source": source,
                                      "target": target, "completed_prefix": snapshot})
            return
        if lane is None:
            return
        point = len(self.payload)
        self.join(self.completed.setdefault(lane, {}), self.drained)
        self.payload.append(signature)
        self.before.append({"lane": lane, "completed": dict(self.completed.get(lane, {}))})
        self.issued[lane] = point

    def observe(self, op, point, signature):
        lane = {"pto.tload": "PIPE_MTE2", "pto.textract": "PIPE_MTE1",
                "pto.tmatmul": "PIPE_M", "pto.tmatmul.acc": "PIPE_M",
                "pto.tabs": "PIPE_V", "pto.tadd": "PIPE_V"}.get(op.name)
        if op.name == "pto.tstore":
            lane = "PIPE_FIX" if "tile_buf<acc," in str(op.operands[0].type) else "PIPE_MTE3"
        if op.name == "pto.textract" and "tile_buf<mat," not in str(op.operands[0].type):
            raise ValueError("boundary observer only qualifies MAT operand extraction")
        if op.name not in SYNC and op.name.startswith("pto.t") and lane is None:
            raise ValueError(f"no qualified physical lane in boundary observer: {op.name}")
        self.action(op.name, attrs(op), lane, signature)


def compare(manual, automatic):
    if manual.payload != automatic.payload:
        raise ValueError("physical payload traces differ")
    differences = []
    for point, (left, right) in enumerate(zip(manual.before, automatic.before)):
        if left["lane"] != right["lane"]:
            raise ValueError("physical lane traces differ")
        for source in sorted(left["completed"].keys() | right["completed"].keys()):
            if source == left["lane"]:
                continue
            a, b = left["completed"].get(source, -1), right["completed"].get(source, -1)
            if a != b:
                differences.append({"target": point, "target_op": manual.payload[point][0],
                                    "source_lane": source, "target_lane": left["lane"],
                                    "manual_prefix": a, "automatic_prefix": b,
                                    "automatic_requires_later_prefix": b > a})
    return differences


def run(path, scenario):
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        module = ir.Module.parse(path.read_text())
        function = next(op for op in children(module.operation) if op.name == "func.func")
        result = Boundaries()
        metric = replay(function, scenario["arguments"], scenario.get("block_idx", 0),
                        scenario.get("block_num", 1), observer=result.observe)
        if result.tokens:
            raise ValueError("undrained event tokens in concrete replay")
        return result, metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manual", type=Path)
    parser.add_argument("automatic", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument("--case", default="historical_gemm")
    parser.add_argument("--scenario", default="outer_reuse")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = next(c for c in json.loads(args.manifest.read_text())["cases"] if c["case_id"] == args.case)
    scenario = next(s for s in case["scenarios"] if s["name"] == args.scenario)
    manual, m = run(args.manual, scenario)
    automatic, a = run(args.automatic, scenario)
    if m["payload_sha256"] != a["payload_sha256"]:
        raise ValueError("replayed payload hashes differ")
    differences = compare(manual, automatic)
    summary = Counter((d["source_lane"], d["target_lane"], d["automatic_requires_later_prefix"])
                      for d in differences)
    report = {"case": args.case, "scenario": scenario, "physical_operations": len(manual.payload),
              "interpretation": "concrete required source prefixes; not timing or a correctness proof",
              "summary": [{"source": s, "target": t, "automatic_later": later, "observations": n}
                          for (s, t, later), n in sorted(summary.items())],
              "differences": differences, "payload": manual.payload,
              "manual_handoffs": manual.handoffs, "automatic_handoffs": automatic.handoffs}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
