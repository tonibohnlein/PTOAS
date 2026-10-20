# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Diagnose exact outer replay without changing production selection or limits.

Every available replay interface is checked independently. An unavailable
interface remains an explicit missing-export outcome, never an exact fallback.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from check_compact_persistent_contract import annotated, variant
from compact_persistent_oracle import boundary, occurrences, probes, required


def request(tool, text, path):
    path.write_text(text)
    command = [str(tool), "--compact-region-contract", str(path)]
    begin = time.perf_counter()
    run = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=35
    )
    path.with_suffix(".json").write_text(run.stdout)
    path.with_suffix(".stderr").write_text(run.stderr)
    if run.returncode:
        raise RuntimeError(run.stderr)
    return json.loads(run.stdout), {
        "command": command,
        "returncode": run.returncode,
        "seconds": time.perf_counter() - begin,
    }


def check_available(result, items, cells):
    if result["error"] or result["retained_demand_error"]:
        raise AssertionError(result)
    assert result["present"] == [x["active"] for x in items]
    _, _, covers, rows = required(items)
    expected = [
        "".join("1" if row >> b & 1 else "0" for b in range(len(rows))) for row in rows
    ]
    assert result["event_rows"] == expected, "four-kind reachability differs"
    active = [i for i, x in enumerate(items) if x["active"]]
    assert {tuple(x) for x in result["retained_demands"]} == {
        (active[a // 2], active[b // 2]) for a, b in covers
    }, "actual retained covers differ"
    assert not result["boundaries"]["error"]
    assert len(result["boundaries"]["rows"]) == len(cells)
    for cell, actual in zip(cells, result["boundaries"]["rows"]):
        actual = {
            k: {p: v for p, v in value.items() if v}
            if isinstance(value, dict)
            else value
            for k, value in actual.items()
        }
        assert actual == boundary(items, cell), "storage boundary differs"
    return {
        "active_occurrences": len(active),
        "covers": len(covers),
        "event_query_comparisons": len(rows) ** 2,
        "local_cell_queries": len(cells),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-tool", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.trace_tool = args.trace_tool.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    source = args.input.read_text()
    rows = []
    for epochs in (0, 1, 2, 3):
        # Always keep the explicit original epoch coordinate, including one visit.
        text = variant(source, 2, 2, 2, 0, 2).replace(
            "%epoch_limit = arith.constant 2", f"%epoch_limit = arith.constant {epochs}"
        )
        items = occurrences(2, 2, 2, 0, epochs if epochs != 1 else 2)
        if epochs == 1:
            items = [x for x in items if x["coordinates"][0] == 0]
        cells = probes(items)
        text = annotated(text, items, cells)
        result, run = request(
            args.trace_tool, text, args.output / f"e{epochs}-contract.pto"
        )
        row = {"epochs": epochs, "production_contract": run, "error": result["error"]}
        if not result["error"]:
            row["independent_contract"] = check_available(result, items, cells)
        else:
            raise AssertionError(
                "Qualified replay interface regressed: " + str(result["error"])
            )
        # A separate exact mathematics request diagnoses whether root mathematics
        # was merely retained behind an endpoint refusal. This is additional work.
        math_text = text.replace(
            "attributes {test.contract_occurrences",
            "attributes {test.contract_math_only, test.contract_occurrences",
            1,
        )
        math, run = request(
            args.trace_tool, math_text, args.output / f"e{epochs}-mathematics.pto"
        )
        row["separate_mathematics_request"] = run
        row["mathematics_error"] = math["error"]
        row["mathematics_ready"] = not bool(math["error"])
        if not result["error"]:
            # Check output membership on every row boundary and every hole,
            # independently for each explicit replay count. No byte sampling
            # is used as a producer-side uniformity certificate.
            gm_cells = sorted(
                {
                    (1, 2, point)
                    for item in items
                    for access in item["effects"]
                    if access["space"] == 1 and access["base"] == 2
                    for begin, end in access["intervals"]
                    for point in (
                        begin,
                        min(begin + 1, end - 1),
                        (begin + end) // 2,
                        end - 1,
                        end,
                    )
                }
            )
            gm_cells = gm_cells or [(1, 2, 0)]
            plain = variant(source, 2, 2, 2, 0, 2).replace(
                "%epoch_limit = arith.constant 2",
                f"%epoch_limit = arith.constant {epochs}",
            )
            gm, gm_run = request(
                args.trace_tool,
                annotated(plain, items, gm_cells),
                args.output / f"e{epochs}-gm-selectors.pto",
            )
            row["output_selector_request"] = gm_run
            row["output_selector"] = gm.get("boundaries", {"error": gm.get("error")})
            if not row["output_selector"].get("error"):
                row["independent_gm_selector"] = check_available(gm, items, gm_cells)
            else:
                raise AssertionError(
                    "Repaired output domain must provide selectors: "
                    + str(row["output_selector"])
                )
            # Remove GM effects only in this independent attribution experiment.
            # Equal closures show that overlap adds no missing prerequisite;
            # actual retained covers above still use the complete access model.
            local_items = [
                dict(
                    item,
                    effects=[
                        access for access in item["effects"] if access["space"] != 1
                    ],
                )
                for item in items
            ]
            row["gm_conflicts_already_in_local_closure"] = (
                required(local_items)[3] == required(items)[3]
            )
        rows.append(row)
        print(
            "replay",
            epochs,
            "available" if not result["error"] else "unavailable",
            flush=True,
        )
    receipt = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_source": True,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "tool_sha256": hashlib.sha256(args.trace_tool.read_bytes()).hexdigest(),
        "scope": "supplied NZ layout; no native/device qualification; no allocation change",
        "rows": rows,
    }
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
