# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bounded region survey on pinned pre-synchronization physical PTO inputs.

No source kernel rewrite or allocation change. Constant entry bindings and
explicit loop-prefix derivatives are named in each receipt. Model checks consume
original declarations, while retained-cover checks consume actual region records.
"""

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from check_compact_persistent_contract import annotated, check_accesses
from compact_persistent_oracle import boundary
from pypto_region_oracle import interpret, mathematical


def specialize(text, arguments, loop_prefix=None):
    lines = text.splitlines()
    at = next(i for i, line in enumerate(lines) if "func.func @" in line)
    types = dict(re.findall(r"(%arg\d+): (index|i32)(?=[,)])", lines[at]))
    if set(types) != set(arguments):
        raise ValueError(
            f"Incomplete scalar configuration {types.keys()} != {arguments.keys()}"
        )
    body = "\n".join(lines[at + 1 :])
    for name in sorted(arguments, key=len, reverse=True):
        body = re.sub(
            re.escape(name) + r"(?![A-Za-z_0-9])", "%survey_" + name[1:], body
        )
    constants = [
        f"    %survey_{name[1:]} = arith.constant {arguments[name]} : {types[name]}"
        for name in types
    ]
    if loop_prefix:
        old, limit = loop_prefix
        body, count = re.subn(old, "to %survey_limit step", body, count=1)
        if count != 1:
            raise ValueError("Unknown loop-prefix derivative")
        constants.append(f"    %survey_limit = arith.constant {limit} : index")
    return "\n".join(lines[: at + 1] + constants) + "\n" + body + "\n"


def child_source(text, name, context=0):
    """Instantiate an original contiguous child; retain declarations and loops."""
    iv = "%arg10" if name == "qwen-down" else "%arg7"
    outer = text.index("scf.for " + iv)
    opening = text.index("{", outer)
    if name == "qwen-down":
        inner = text.index("scf.for %arg11", opening)
        end = text.index("{", inner)
        depth, end = 1, end + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        body = text[opening + 1 : end]
    else:
        depth, end = 1, opening + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        body = text[opening + 1 : end - 1]
    return (
        text[:outer]
        + f"{iv} = arith.constant {context} : index\n"
        + body
        + "\n return\n }\n}\n"
    )


def invoke(command, path, timeout=35):
    begin = time.perf_counter()
    with path.open("w") as out, path.with_suffix(path.suffix + ".stderr").open(
        "w"
    ) as err:
        try:
            result = subprocess.run(
                list(map(str, command)),
                stdout=out,
                stderr=err,
                timeout=timeout,
                check=False,
            )
            rc = result.returncode
            status = (
                "success"
                if rc == 0
                else "process_signal"
                if rc < 0
                else "compiler_failure"
            )
        except subprocess.TimeoutExpired:
            rc, status = None, "watchdog"
    return {
        "command": list(map(str, command)),
        "status": status,
        "returncode": rc,
        "seconds": time.perf_counter() - begin,
    }


def local_probes(items):
    cuts = {}
    for item in items:
        for effect in item["effects"]:
            if effect["space"] == 1:
                continue
            key = (effect["space"], effect["base"])
            for begin, end in effect["intervals"]:
                cuts.setdefault(key, set()).update((begin, end))
    return [
        (space, base, byte)
        for (space, base), points in sorted(cuts.items())
        for byte in sorted(points)
    ]


def validate(result, items, cells):
    outcome = {"status": "unavailable", "error": result.get("error")}
    if result.get("error"):
        return outcome
    reach, covers = mathematical(items)
    expected = [
        "".join("1" if row >> b & 1 else "0" for b in range(len(reach)))
        for row in reach
    ]
    outcome.update(
        presence=result["present"] == [True] * len(items),
        closure=result["event_rows"] == expected,
        covers=(
            not result.get("retained_demand_error")
            and {tuple(x) for x in result["retained_demands"]}
            == {(a // 2, b // 2) for a, b in covers}
        ),
        cover_error=result.get("retained_demand_error"),
        selectors=False,
        selector_error=result["boundaries"].get("error"),
        actual_cover_count=len(result["retained_demands"]),
        oracle_cover_count=len(covers),
        event_query_comparisons=len(reach) ** 2,
    )
    if not outcome["selector_error"]:
        normalized = [
            {
                k: {p: v for p, v in value.items() if v}
                if isinstance(value, dict)
                else value
                for k, value in row.items()
            }
            for row in result["boundaries"]["rows"]
        ]
        outcome["selectors"] = normalized == [boundary(items, cell) for cell in cells]
    outcome["status"] = (
        "passed"
        if all(outcome[k] for k in ["presence", "closure", "covers", "selectors"])
        else "mismatch_or_unavailable"
    )
    return outcome


def configurations(name):
    if name == "qwen-down":
        for k in (1, 2, 3, 4, 8, 16):
            yield (
                f"k{k}",
                {
                    "%arg3": 20,
                    "%arg4": k,
                    "%arg5": 0,
                    "%arg6": 0,
                    "%arg7": 4352,
                    "%arg8": 0,
                    "%arg9": 20,
                },
                None,
            )
    elif name == "qwen-finalize":
        for trips in (0, 1, 2, 4, 8):
            yield (
                f"visits{trips}",
                {"%arg3": 128, "%arg4": 0, "%arg5": 0, "%arg6": 48},
                (r"to %c1024 step", trips * 48),
            )
        yield (
            "skipped",
            {"%arg3": 0, "%arg4": 0, "%arg5": 0, "%arg6": 48},
            (r"to %c1024 step", 96),
        )
    else:
        for trips in (0, 1, 2):
            yield (
                f"reduce{trips}",
                {"%arg3": 8, "%arg4": 0, "%arg5": 4},
                (r"to %c7168 step", trips * 3584),
            )


def finite_required(items):
    reach, covers = mathematical(items)
    return items, reach, covers, reach


def check(args, name, source, label, bindings, prefix):
    directory = args.output / name / label
    directory.mkdir(parents=True, exist_ok=True)
    text = specialize(source, bindings, prefix)
    plain = directory / "input.pto"
    plain.write_text(text)
    constants = {}
    items = interpret(text, constants)
    cells = local_probes(items)
    row = {
        "name": name,
        "configuration": label,
        "scalar_bindings": bindings,
        "loop_prefix": prefix,
        "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "payloads": len(items),
        "local_probes": len(cells),
        "stages": {},
    }
    contract = directory / "contract.pto"
    valued = annotated(text, items, cells)
    bindings = ", ".join(
        f"array<i64: {line}, {next(iter(observed))}>"
        for line, observed in sorted(constants.items())
        if len(observed) == 1
    )
    valued = valued.replace(
        "attributes {test.contract_occurrences",
        "attributes {test.contract_constants = ["
        + bindings
        + "], test.contract_occurrences",
        1,
    )
    contract.write_text(valued)
    output = directory / "contract.json"
    row["stages"]["regional_contract"] = invoke(
        [args.trace_tool, "--compact-region-contract", contract], output
    )
    if row["stages"]["regional_contract"]["status"] == "success":
        result = json.loads(output.read_text())
        row["regional_contract"] = validate(result, items, cells)
    if label in ("child-k2", "child-work0", "reduce2"):
        global_cells = sorted(
            {
                (effect["space"], effect["base"], effect["intervals"][0][0])
                for item in items
                for effect in item["effects"]
                if effect["space"] == 1 and effect["intervals"]
            }
        )[:4]
        description = ", ".join(
            f"array<i64: {space}, {base}, {byte}>" for space, base, byte in global_cells
        )
        global_text = re.sub(
            r"test.contract_cells = \[[^\]]*\]",
            "test.contract_cells = [" + description + "]",
            valued,
            count=1,
        )
        global_input = directory / "global-selectors.pto"
        global_input.write_text(global_text)
        global_output = directory / "global-selectors.json"
        stage = invoke(
            [args.trace_tool, "--compact-region-contract", global_input], global_output
        )
        row["stages"]["global_selectors"] = stage
        if stage["status"] == "success":
            exported = json.loads(global_output.read_text())
            result = exported.get("boundaries", {"error": exported.get("error")})
            row["global_selectors"] = {
                "cells": global_cells,
                "result": result,
                "status": "unavailable" if result.get("error") else "available",
            }
            if not result.get("error"):
                actual = [
                    {
                        k: {p: v for p, v in value.items() if v}
                        if isinstance(value, dict)
                        else value
                        for k, value in r.items()
                    }
                    for r in result["rows"]
                ]
                if actual != [boundary(items, cell) for cell in global_cells]:
                    row["global_selectors"]["status"] = "mismatch"
                    row["global_selectors"]["expected"] = [
                        boundary(items, cell) for cell in global_cells
                    ]
                else:
                    row["global_selectors"]["status"] = "passed"
    if label in ("child-work0", "reduce2"):
        # Query outside every modeled GM range, separately from the known
        # nonempty witness. Missing coverage must not destroy proved absence.
        absent_cells = []
        for space, base in sorted(
            {
                (v["space"], v["base"])
                for item in items
                for v in item["effects"]
                if v["space"] == 1
            }
        ):
            upper = max(
                end
                for item in items
                for v in item["effects"]
                if (v["space"], v["base"]) == (space, base)
                for _, end in v["intervals"]
            )
            absent_cells.append((space, base, upper + 16))
        description = ", ".join(
            f"array<i64: {space}, {base}, {byte}>" for space, base, byte in absent_cells
        )
        absent_text = re.sub(
            r"test.contract_cells = \[[^\]]*\]",
            "test.contract_cells = [" + description + "]",
            valued,
            count=1,
        )
        absent_input = directory / "absent-selectors.pto"
        absent_input.write_text(absent_text)
        absent_output = directory / "absent-selectors.json"
        stage = invoke(
            [args.trace_tool, "--compact-region-contract", absent_input], absent_output
        )
        row["stages"]["absent_selectors"] = stage
        exported = (
            json.loads(absent_output.read_text())
            if stage["status"] == "success"
            else {}
        )
        result = exported.get(
            "boundaries", {"error": exported.get("error", "request failed")}
        )
        row["absent_selectors"] = {
            "cells": absent_cells,
            "result": result,
            "status": "unavailable" if result.get("error") else "passed",
        }
        if not result.get("error"):
            if any(
                any(value.values()) if isinstance(value, dict) else bool(value)
                for entry in result["rows"]
                for value in entry.values()
            ):
                row["absent_selectors"]["status"] = "mismatch"
        if args.require_supported and row["absent_selectors"]["status"] == "mismatch":
            raise AssertionError(row["absent_selectors"])
    # Retained representation profiling is a production logical request, not an audit.
    row["stages"]["logical"] = invoke(
        [
            args.optimizer,
            "--mlir-disable-threading",
            "--pto-frontier-analysis=gm-alias=may-not-alias prepare-allocation=false profile-stages=true",
            plain,
            "-o",
            directory / "logical.pto",
        ],
        directory / "logical.stdout",
    )
    for line in (directory / "logical.stdout.stderr").read_text().splitlines():
        if line.startswith("frontier-retained-inventory "):
            row["retained_inventory"] = json.loads(line.split(" ", 1)[1])
    # Full extraction/flat-oracle checks on small positive prefixes only. Production
    # IR remains nested; unrolling here serves only the independent access oracle.
    if label in (
        "k1",
        "k2",
        "visits1",
        "visits2",
        "reduce1",
        "reduce2",
        "child-k2",
        "child-work0",
    ):
        try:
            count = check_accesses(
                SimpleNamespace(optimizer=args.optimizer, trace_tool=args.trace_tool),
                text,
                items,
                directory,
                required_order=finite_required,
            )
            row["access_and_flat_oracle"] = {
                "status": "passed",
                "access_comparisons": count,
            }
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            row["access_and_flat_oracle"] = {
                "status": "unqualified",
                "error": str(error)[:1500],
            }
    print(name, label, row.get("regional_contract", {}).get("status"), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("optimizer", "trace-tool", "input-root", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("qwen-down", "qwen-finalize", "deepseek-rms"),
        default=("qwen-down", "qwen-finalize", "deepseek-rms"),
    )
    parser.add_argument("--children", action="store_true")
    parser.add_argument("--require-supported", action="store_true")
    args = parser.parse_args()
    args.optimizer, args.trace_tool = (
        Path(shutil.which(str(args.optimizer)) or args.optimizer).resolve(),
        Path(shutil.which(str(args.trace_tool)) or args.trace_tool).resolve(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    rows = []
    for name in args.cases:
        source = (args.input_root / name / "input.pto").read_text()
        configs = list(configurations(name))
        if args.children:
            if name == "deepseek-rms":
                continue
            if name == "qwen-down":
                source = child_source(source, name)
                configs = [
                    ("child-" + label, bindings, None)
                    for label, bindings, _ in configs
                    if label in ("k2", "k4", "k8", "k16")
                ]
            else:
                original = source
                configs = [
                    (
                        f"child-work{context}",
                        {"%arg3": 128, "%arg4": 0, "%arg5": 0, "%arg6": 48},
                        None,
                    )
                    for context in (0, 48)
                ]
                source = child_source(original, name)
        for label, bindings, prefix in configs:
            if args.children and name == "qwen-finalize":
                source = child_source(
                    original, name, int(label.removeprefix("child-work"))
                )
            row = check(args, name, source, label, bindings, prefix)
            rows.append(row)
            if args.require_supported:
                if row.get("regional_contract", {}).get("status") != "passed":
                    raise AssertionError(row.get("regional_contract"))
                if row.get("global_selectors", {}).get("status") == "mismatch":
                    raise AssertionError(row["global_selectors"])
                if (
                    row.get("access_and_flat_oracle", {}).get("status", "passed")
                    != "passed"
                ):
                    raise AssertionError(row["access_and_flat_oracle"])
            (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
