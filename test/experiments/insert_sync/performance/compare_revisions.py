#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run original/revised InsertSync on identical frozen automatic inputs, serially."""

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare_native import compile_arm, compare, digest
from measure import measure


def totals(counts):
    values = {name: counts.get("pto." + name, 0) for name in ("set_flag", "wait_flag", "barrier")}
    values["total"] = sum(values.values())
    values["pipe_all"] = sum(value for key, value in counts.items() if key.startswith("barrier:#pto.pipe<PIPE_ALL>:"))
    return values


def campaign(args):
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "inputs"
    frozen.mkdir()
    sys.path.insert(0, str(args.revised_python_root.resolve()))
    cases = []
    for filename in ("manifest.json", "kernel-pairs-manifest.json"):
        manifest = json.loads((root / filename).read_text())
        (output / filename).write_text(json.dumps(manifest, indent=2) + "\n")
        for case in manifest["cases"]:
            source = case["sources"]["auto"]
            path = root / source["path"]
            if digest(path) != source["sha256"]:
                raise ValueError(f"input identity mismatch: {path}")
            target = frozen / f"{case['case_id']}.pto"
            target.write_bytes(path.read_bytes())
            original = measure(target, [], "static")
            if totals(original["static"]["counts"])["total"]:
                raise ValueError(f"automatic input already has local synchronization: {path}")
            cases.append(case)
    tsv = output / "inputs.tsv"
    with tsv.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["case_id", "source", "sha256", "level"])
        for case in cases:
            name = case["case_id"]
            writer.writerow([name, name + ".pto", digest(frozen / (name + ".pto")), case["level"]])
    report = {
        "measurement": "Inserted local set_flag + wait_flag + barrier operations; "
        "fixed cross-core/FIFO operations excluded",
        "device_execution": "not-run",
        "original_source": args.original_source,
        "revised_source": args.revised_source,
        "input_manifest_sha256": digest(tsv),
        "gm_contract": "disjoint arguments; original uses legacy default, "
        "revision explicitly selects assume-disjoint-arguments",
        "allow_rejections": args.allow_rejections,
        "failures": [],
        "architectures": {},
    }
    for arch in args.arch:
        arms = {}
        for arm, runtime in [
            ("main", args.original_python_root),
            ("combined", args.revised_python_root),
            ("staged", args.revised_python_root),
        ]:
            destination = output / arch / arm
            destination.parent.mkdir(exist_ok=True)
            compile_arm(
                SimpleNamespace(
                    input_root=frozen,
                    manifest=tsv,
                    python_root=runtime,
                    python=Path(sys.executable),
                    output=destination,
                    population="seven kernel families plus four controls",
                    arm=arm,
                    arch=arch,
                    gm_alias=None if arm == "main" else "assume-disjoint-arguments",
                    audit=None if arm == "main" else "report",
                    timeout=args.timeout,
                )
            )
            compilation = json.loads((destination / "results.json").read_text())
            measured = {
                "native_sha256": compilation["native_sha256"],
                "cli_sha256": compilation["cli_sha256"],
                "native_unchanged": compilation["native_unchanged"],
                "rows": {},
            }
            for case, result in zip(cases, compilation["rows"]):
                name = case["case_id"]
                row = {"status": "emitted", "runs": result["runs"]}
                diagnostics = {kind: (destination / name / f"{kind}.stderr").read_text() for kind in ("pto", "cpp")}
                row["diagnostics"] = [
                    f"{kind}: {line}"
                    for kind, diagnostic in diagnostics.items()
                    for line in diagnostic.splitlines()
                    if "error:" in line
                ]
                if any(run["status"] != "pass" for run in result["runs"].values()):
                    expected = case.get("expected_autosync_rejection")
                    rejected = expected and all(
                        run["status"] != "pass"
                        and run["returncode"] is not None
                        and run["returncode"] > 0
                        and expected in diagnostics[kind]
                        for kind, run in result["runs"].items()
                    )
                    row["status"] = "rejected" if rejected else "compile-failure"
                elif (
                    not result["runs"]["pto"]["analysis_executed"]
                    or not result["runs"]["pto"]["allocation_executed"]
                    or not result["runs"]["pto"]["functions"]
                    or any(f["explicit_sync_bypass"] for f in result["runs"]["pto"]["functions"])
                ):
                    row["status"] = "unconfirmed-insertion"
                if row["status"] == "emitted":
                    path = destination / name / "output.pto"
                    try:
                        row["metrics"] = measure(path, [], "static", case.get("normalize_gm_pipe_assembly", False))
                        row["inserted_static"] = totals(row["metrics"]["static"]["counts"])
                        if case.get("measurement", "replay") == "replay":
                            replayed = measure(path, case["scenarios"])
                            row["scenarios"] = replayed["scenarios"]
                            row["inserted_dynamic"] = {
                                key: totals(value["counts"]) for key, value in replayed["scenarios"].items()
                            }
                    except Exception as error:
                        row["status"] = "measurement-failure"
                        row["measurement_error"] = f"{type(error).__name__}: {error}"
                measured["rows"][name] = row
                if row["status"] != "emitted" and not (row["status"] == "rejected" and args.allow_rejections):
                    report["failures"].append(
                        f"{arch}/{arm}/{name}: {row['status']} {row.get('measurement_error', '')}".rstrip()
                    )
            arms[arm] = measured
            report["architectures"][arch] = arms
            (output / "counts.json").write_text(json.dumps(report, indent=2) + "\n")
        compare([output / arch / arm for arm in arms], output / arch / "diffs")
    lines = [
        "# Original versus revised InsertSync counts",
        "",
        f"Original source: `{args.original_source}`. Revised source: `{args.revised_source}`.",
        "",
        "Each cell is **sets / waits / barriers = total** in emitted IR. "
        "Fixed cross-core and FIFO operations are excluded.",
        "All input local-sync counts are zero. Rejection has no inserted count, rather than a count of zero.",
        "Successful emission and fewer synchronization actions do not establish correctness or runtime speed.",
        "",
    ]
    for arch, arms in report["architectures"].items():
        lines += [
            f"## {arch.upper()}",
            "",
            "| Case | Original | Revised combined | Revised staged |",
            "| --- | --- | --- | --- |",
        ]
        for case in cases:
            name = case["case_id"]
            cells = []
            for arm in ("main", "combined", "staged"):
                row = arms[arm]["rows"][name]
                value = row.get("inserted_static")
                cell = (
                    f"{value['set_flag']} / {value['wait_flag']} / {value['barrier']} = {value['total']}"
                    if value
                    else row["status"]
                )
                if row.get("measurement_error"):
                    cell += " (measurement-failure; see counts.json)"
                cells.append(cell)
            lines.append("| " + " | ".join([name] + cells) + " |")
        lines.append("")
    lines += ["Failures:", ""] + (report["failures"] or ["None."])
    (output / "counts.md").write_text("\n".join(lines) + "\n")
    print(f"Count report: {output / 'counts.md'}", flush=True)
    return bool(report["failures"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-python-root", type=Path, required=True)
    parser.add_argument("--revised-python-root", type=Path, required=True)
    parser.add_argument("--original-source", required=True)
    parser.add_argument("--revised-source", required=True)
    parser.add_argument("--arch", choices=("a2", "a3"), nargs="+", default=["a2", "a3"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--allow-rejections",
        action="store_true",
        help="Allow manifest-declared coverage rejections when comparing historical compilers",
    )
    raise SystemExit(campaign(parser.parse_args()))
