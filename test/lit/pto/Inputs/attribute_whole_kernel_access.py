# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Attribute finite graph growth; derived upper graphs are not production runs."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from check_whole_kernel_access import independent, inputs
from whole_kernel_reference import reference


def excess(rows, concrete):
    return sum(
        bool(bits >> b & 1 and not concrete[a] >> b & 1)
        for a, bits in enumerate(rows)
        if a % 2
        for b in range(0, len(rows), 2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, label, text, strategy, trips in inputs(Path.cwd()):
        items, actual, _, scalar = independent(text, strategy, trips)
        exact, _ = reference(items, scalar)
        if exact != actual:
            raise RuntimeError(
                f"Independent graph implementations disagree: {name}/{label}"
            )
        possible, _ = reference(items, scalar, possible_alias=True)
        coarse_items = deepcopy(items)
        for item in coarse_items:
            for effect in item["effects"]:
                if effect["space"] == 1:
                    effect.update(base=0, intervals=[(0, 1)])
        coarse, _ = reference(coarse_items, scalar)
        for alias in ("may-not-alias", "may-alias"):
            directory = (
                "semantics-pypto"
                if strategy == "pypto"
                else "semantics-cumsum"
                if name == "gdn_chunk_cumsum"
                else "semantics-bounded"
            )
            path = (
                args.receipts
                / directory
                / name
                / label
                / alias
                / "completion-trace.json"
            )
            trace = json.loads(path.read_text())["trace"]
            events = trace["events"]
            next_payload = [e for e in events if e["kind"] == "payload"]
            barriers = [
                {
                    "gap": e["gap"],
                    "next_type": next_payload[e["gap"]]["type"]
                    if e["gap"] < len(next_payload)
                    else "return",
                    "pipe": e["pipe"],
                }
                for e in events
                if e["kind"] == "barrier"
            ]
            rows.append(
                {
                    "kernel": name,
                    "configuration": label,
                    "alias": alias,
                    "payloads": len(items),
                    "possible_alias_extra_ci": excess(possible, actual),
                    "whole_gm_class_extra_ci": excess(coarse, actual),
                    "scope": (
                        "Independent derived upper graph over disjoint actual layouts; "
                        "not exclusive SCC attribution"
                    ),
                    "completion_barriers": barriers,
                }
            )
    (args.receipts / "attribution-results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
