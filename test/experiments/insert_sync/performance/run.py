#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Serial manual/combined/staged InsertSync regression campaign; no device timing."""

import argparse
import copy
import difflib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compare_native import digest, participation, run
from measure import measure
from generate_kernel_pairs import strip_local_sync


def regression_signature(row):
    result = {key: copy.deepcopy(row.get(key)) for key in ("status", "metrics", "audit_verdicts")}
    if result["metrics"]:
        # Full placements remain in results.json; their digest is enough for
        # a small checked-in checkpoint. Compile durations are never a gate.
        result["metrics"]["static"].pop("placements", None)
    return result


def compare(before, after):
    """Gate all changes, including reductions: lower counts need overlap review."""
    if before["manifest_sha256"] != after["manifest_sha256"] or before["arch"] != after["arch"]:
        raise ValueError("baseline has different inputs/scenarios or architecture")
    changes = []
    for key in ("rows",):
        left = before[key]
        right = after[key]
        if set(left) != set(right):
            raise ValueError("baseline has different arms/cases")
        for name, row in right.items():
            old = regression_signature(left[name])
            new = regression_signature(row)
            for field in ("status", "metrics", "audit_verdicts"):
                if old[field] != new[field]:
                    changes.append(f"{name}: {field} changed")
    return changes


def campaign(args):
    root = Path(__file__).resolve().parent
    manifest_path = args.manifest.resolve() if args.manifest else root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for source in manifest.get("reference_files", []) + manifest.get("support_files", []):
        path = (root / source["path"]).resolve()
        if not path.is_relative_to(root) or digest(path) != source["sha256"]:
            raise ValueError(f"support/reference identity mismatch: {path}")
    for case in manifest["cases"]:
        for source in case["sources"].values():
            path = (root / source["path"]).resolve()
            if not path.is_relative_to(root) or digest(path) != source["sha256"]:
                raise ValueError(f"source identity mismatch: {path}")
        if case.get("pair_contract") == "remove-local-sync-only":
            manual = (root / case["sources"]["manual"]["path"]).read_text()
            automatic = (root / case["sources"]["auto"]["path"]).read_text()
            if strip_local_sync(manual) != automatic:
                raise ValueError(f"non-sync source change in pair: {case['case_id']}")
    runtime = args.python_root.resolve()
    sys.path.insert(0, str(runtime))
    native = runtime / "ptoas/mlir/_mlir_libs/libPTOASCompiler.so"
    cli = runtime / "ptoas/_cli.py"
    report = {
        "version": 1,
        "arch": args.arch,
        "manifest_sha256": digest(manifest_path),
        "native_sha256": digest(native),
        "cli_sha256": digest(cli),
        "python_root": str(runtime),
        "gm_alias": manifest["gm_alias"],
        "measurement": "host scalar replay; no completion/latency simulation",
        "device_runtime": "not-run",
        "rows": {},
        "failures": [],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    # Freeze the input population beside each result, including failed rows.
    (args.output / "manifest.json").write_text(manifest_path.read_text())
    for source in manifest.get("reference_files", []) + manifest.get("support_files", []):
        target = args.output / "support" / source["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / source["path"]).read_bytes())
    prefix = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "from pathlib import Path; from ptoas import _cli; "
        "raise SystemExit(_cli.launch(sys.argv[1:], wrapper=Path(_cli.__file__)))",
        str(runtime),
    ]
    for case in manifest["cases"]:
        for arm in ("manual", "combined", "staged"):
            key = f"{case['case_id']}/{arm}"
            directory = args.output / key
            directory.mkdir(parents=True)
            source = root / case["sources"]["manual" if arm == "manual" else "auto"]["path"]
            frozen = directory / "input.pto"
            frozen.write_bytes(source.read_bytes())
            row = {"status": "pass", "input_sha256": digest(frozen), "runs": {}}
            expected_rejection = case.get("expected_autosync_rejection") if arm != "manual" else None
            rejection_matches = []
            for kind in ("pto", "cpp"):
                output = directory / f"output.{kind}"
                command = prefix + [f"--pto-arch={args.arch}", f"--pto-level={case['level']}"]
                if arm != "manual":
                    command += ["--enable-insert-sync", f"--insert-sync-gm-alias={manifest['gm_alias']}"]
                    if arm == "staged":
                        command.append("--insert-sync-defer-same-pipe")
                if kind == "pto":
                    command.append("--emit-pto-ir")
                    if arm != "manual":
                        command += [
                            "--insert-sync-audit=report",
                            "--pto-insert-sync-debug=1",
                            "--mlir-print-ir-before=pto-insert-sync",
                            "--mlir-print-ir-after=pto-insert-sync",
                        ]
                command += [str(frozen.resolve()), "-o", str(output.resolve())]
                code, stdout, stderr, elapsed = run(command, args.timeout)
                (directory / f"{kind}.stdout").write_text(stdout)
                (directory / f"{kind}.stderr").write_text(stderr)
                record = {"command": command, "returncode": code, "seconds": elapsed}
                if expected_rejection:
                    rejection_matches.append(code not in (0, None) and expected_rejection in stderr)
                if code != 0 or not output.is_file():
                    row["status"] = "compile-failure"
                else:
                    record["output_sha256"] = digest(output)
                if kind == "pto" and arm != "manual":
                    record["participation"] = participation(stderr)
                    active = record["participation"]
                    if code == 0 and (
                        not active["analysis_executed"]
                        or not active["allocation_executed"]
                        or not active["functions"]
                        or any(f["explicit_sync_bypass"] for f in active["functions"])
                    ):
                        row["status"] = "autosync-bypass"
                row["runs"][kind] = record
            if expected_rejection and all(rejection_matches):
                # A rejected input remains in the population. Check plain lowering
                # separately so a parser/emitter failure cannot masquerade as a
                # missing synchronization contract. It is not an admitted kernel.
                row["status"] = "unsupported-contract"
                row["expected_diagnostic"] = expected_rejection
                for kind in ("pto", "cpp"):
                    output = directory / f"unsynchronized.{kind}"
                    command = prefix + [f"--pto-arch={args.arch}", f"--pto-level={case['level']}"]
                    if kind == "pto":
                        command.append("--emit-pto-ir")
                    command += [str(frozen.resolve()), "-o", str(output.resolve())]
                    code, stdout, stderr, elapsed = run(command, args.timeout)
                    (directory / f"plain-{kind}.stdout").write_text(stdout)
                    (directory / f"plain-{kind}.stderr").write_text(stderr)
                    row["runs"][f"plain-{kind}"] = {"command": command, "returncode": code, "seconds": elapsed}
                    if code != 0 or not output.is_file():
                        row["status"] = "plain-lowering-failure"
                if row["status"] == "unsupported-contract":
                    row["input_pair_parity"] = case.get("pair_contract") == "remove-local-sync-only"
                    try:
                        replay_plain = case.get("replay_plain_lowering", False)
                        row["metrics"] = measure(
                            directory / "unsynchronized.pto",
                            case["scenarios"] if replay_plain else [],
                            "replay" if replay_plain else "static",
                            case.get("normalize_gm_pipe_assembly", False),
                        )
                        if replay_plain:
                            manual = report["rows"][f"{case['case_id']}/manual"]
                            for name, metric in row["metrics"]["scenarios"].items():
                                reference = manual["metrics"]["scenarios"][name]
                                if metric["payload_sha256"] != reference["payload_sha256"]:
                                    raise ValueError(f"manual/plain payload or allocation differs: {name}")
                            row["plain_payload_parity"] = True
                    except Exception as error:
                        row["status"] = "measurement-failure"
                        row["error"] = str(error)
            if row["status"] == "pass":
                output = directory / "output.pto"
                row["audit_verdicts"] = re.findall(r'pto\.insert_sync\.audit = "([^"]+)"', output.read_text())
                try:
                    row["metrics"] = measure(
                        output,
                        case["scenarios"],
                        case.get("measurement", "replay"),
                        case.get("normalize_gm_pipe_assembly", False),
                    )
                    if arm != "manual":
                        manual = report["rows"][f"{case['case_id']}/manual"]
                        if "metrics" not in manual:
                            raise ValueError("manual reference did not produce metrics")
                        for name, metric in row["metrics"]["scenarios"].items():
                            reference = manual["metrics"]["scenarios"][name]
                            if metric["payload_sha256"] != reference["payload_sha256"]:
                                raise ValueError(f"manual/auto payload or allocation differs: {name}")
                except Exception as error:
                    # Preserve an unsupported parser/binding/control result as
                    # a failed row and continue measuring the frozen population.
                    row["status"] = "measurement-failure"
                    row["error"] = str(error)
            report["rows"][key] = row
            if row["status"] not in ("pass", "unsupported-contract"):
                report["failures"].append(f"{key}: {row['status']} {row.get('error', '')}")
            (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{key}: {row['status']}", flush=True)
        for arm in ("combined", "staged"):
            left = args.output / case["case_id"] / "manual/output.pto"
            right = args.output / case["case_id"] / arm / "output.pto"
            if left.is_file() and right.is_file():
                difference = difflib.unified_diff(
                    left.read_text().splitlines(True), right.read_text().splitlines(True), fromfile="manual", tofile=arm
                )
                (right.parent / "manual.diff").write_text("".join(difference))
    report["native_unchanged"] = digest(native) == report["native_sha256"] and digest(cli) == report["cli_sha256"]
    if not report["native_unchanged"]:
        report["failures"].append("compiler changed during campaign")
    report["baseline_changes"] = compare(json.loads(args.baseline.read_text()), report) if args.baseline else []
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# InsertSync host regression",
        "",
        f"Native SHA-256: `{report['native_sha256']}`",
        "",
        "Counts describe static IR and concrete scalar replay, not device performance.",
        "",
        "| Case / arm | Status | Static set / wait | Static barriers |",
        "| --- | --- | ---: | ---: |",
    ]
    for key, row in report["rows"].items():
        counts = row.get("metrics", {}).get("static", {}).get("counts", {})
        lines.append(
            f"| {key} | {row['status']} | {counts.get('pto.set_flag', '-')} / "
            f"{counts.get('pto.wait_flag', '-')} | {counts.get('pto.barrier', '-')} |"
        )
    lines += ["", "Failures / baseline changes:", ""] + report["failures"] + report["baseline_changes"]
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    return bool(report["failures"] or report["baseline_changes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-root", type=Path, required=True, help="Complete matching PTOAS Python/native runtime")
    parser.add_argument(
        "--manifest", type=Path, help="Alternate frozen case manifest; inputs remain relative to this runner"
    )
    parser.add_argument("--arch", choices=("a2", "a3"), default="a3")
    parser.add_argument("--output", type=Path, required=True, help="New disk-backed results directory")
    parser.add_argument("--baseline", type=Path, help="Earlier results.json; any metric change requires review")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    return campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
