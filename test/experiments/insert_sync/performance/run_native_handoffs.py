#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Serial native seed/handoff acceptance on 19 pinned inputs; no device execution.

Reuse the established payload/allocation projection, scalar replay and concrete
boundary observer. Keep unsupported observations separate from failed checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from measure import measure, normalize_gm_pipe_assembly
from run_qwen_additions import REVISED_FLAGS, SERIAL_DRIVER, analyze

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def population():
    cases = []
    for name in ("manifest.json", "kernel-pairs-manifest.json"):
        manifest = json.loads((HERE / name).read_text())
        for case in manifest["cases"]:
            source = case["sources"]["auto"]
            cases.append({**case, "source": HERE / source["path"], "sha256": source["sha256"]})
    additions = json.loads((HERE / "qwen-additions-manifest.json").read_text())
    for case in additions["cases"]:
        scenarios = []
        if case["case_id"] == "online_softmax":
            scenarios = [{"name": str(n), "arguments": [0, 0, 0, 0, n, 0, 0]} for n in (0, 1, 2, 16)]
        elif case["case_id"] == "q_proj":
            # Native caller block parameters, without changing tile/loop bounds.
            scenarios = [{"name": "core0", "arguments": [0, 0, 0, 0, 1]}]
        elif case["case_id"] in ("qk_matmul", "sv_matmul"):
            scenarios = [{"name": str(n), "arguments":
                ([0, 0, 0, 0, n, 0, 1] if case["case_id"] == "qk_matmul" else [0, 0, 0, n, 0, 0, 1])}
                for n in (0, 1, 2, 16)]
        cases.append({**case, "source": ROOT / case["source"], "scenarios": scenarios})
    names = [case["case_id"] for case in cases]
    if len(set(names)) != 19 or len(names) != 19:
        raise ValueError("frozen benchmark population changed")
    for case in cases:
        if not case["source"].resolve().is_relative_to(ROOT) or digest(case["source"]) != case["sha256"]:
            raise ValueError("frozen source changed: " + case["case_id"])
    return cases


def compare_evidence(seed, trial):
    """Changed synchronization is permitted; changed payload contracts are not."""
    for key in ("payload", "allocations", "views", "abi", "sync_control"):
        if seed["projection"][key] != trial["projection"][key]:
            raise ValueError("changed " + key)
    left, right = seed["metrics"]["scenarios"], trial["metrics"]["scenarios"]
    if left.keys() != right.keys():
        raise ValueError("missing replay scenario")
    for name in left:
        for key in ("payload_sha256", "scalar_counts"):
            if left[name][key] != right[name][key]:
                raise ValueError("changed replay " + key + ": " + name)


def boundary_evidence(seed, trial, scenarios):
    from compare_boundaries import ObserverUnsupported, compare, run
    rows = []
    for scenario in scenarios:
        observations, unknown = {}, {}
        for label, path in (("seed", seed), ("handoff", trial)):
            try:
                observations[label], _ = run(path, scenario)
            except ObserverUnsupported as error:
                unknown[label] = str(error)
        if unknown:
            rows.append({"scenario": scenario["name"], "status": "unsupported", "reasons_by_arm": unknown})
        else:
            before, after = observations["seed"], observations["handoff"]
            differences = compare(before, after)
            if any(item["automatic_requires_later_prefix"] for item in differences):
                raise ValueError("new blocking in concrete payload prefix")
            rows.append({"scenario": scenario["name"], "status": "checked",
                         "weaker_prefixes": differences})
    return rows


def require_effectiveness(row):
    # Frozen regression expectations, never input to the compiler's selector.
    # Copying the seed after an unsupported analysis is not effectiveness.
    if row["case"] == "online_softmax":
        seed, trial = [row["arms"][a]["metrics"]["scenarios"]["16"]["counts"] for a in ("seed", "handoff")]
        for action in ("pto.set_flag", "pto.wait_flag"):
            if trial.get(action, 0) >= seed.get(action, 0) or trial.get(action, 0) > 96:
                raise ValueError("online softmax command improvement lost")
    if row["case"] == "q_proj":
        seed, trial = [row["arms"][a]["projection"]["mechanisms"] for a in ("seed", "handoff")]
        if any(trial[k] >= seed[k] or trial[k] > 29 for k in ("sets", "waits")):
            raise ValueError("Q projection command improvement lost")


def write_report(output, report):
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Native handoff regression", "", "Campaign: " + report["status"] +
             f' ({len(report["rows"])}/{len(report["population"])} rows).', "",
             "Static sites; mechanisms are never combined. No device timing.", "",
             "| Fixture | Seed sets/waits | Handoff sets/waits | Seed named / ALL | Handoff named / ALL | Status | Prefix observations |",
             "| --- | ---: | ---: | --- | --- | --- | --- |"]
    for row in report["rows"]:
        cells = [row["case"]]
        arms = [row["arms"].get(arm, {}).get("projection", {}).get("mechanisms") for arm in ("seed", "handoff")]
        cells += [f'{a["sets"]}/{a["waits"]}' if a else "unavailable" for a in arms]
        cells += [(str(a["named"]) + " / " + str(a["PIPE_ALL"])) if a else "unavailable" for a in arms]
        cells += [row["status"]]
        coverage = row.get("observation_coverage", {})
        cells += [(f'{coverage["prefix_checked"]} checked, {coverage["prefix_unsupported"]} unsupported'
                   if coverage.get("prefix_status") == "observed" else "not-run")]
        lines.append("| " + " | ".join(cells) + " |")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Focused validation only; default is all 19")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    cases = population()
    if args.case:
        unknown = set(args.case) - {c["case_id"] for c in cases}
        if unknown:
            parser.error("unknown cases: " + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c["case_id"] in args.case]
    runtime = args.python_root.resolve()
    sys.path.insert(0, str(runtime))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = runtime / "ptoas/mlir/_mlir_libs/libPTOASCompiler.so"
    evidence_paths = {str(path.resolve()): path for path in [Path(__file__), HERE / "compare_boundaries.py",
        HERE / "measure.py", HERE / "run_qwen_additions.py", HERE.parent / "compare_native.py",
        runtime / "ptoas/_cli.py", *runtime.glob("ptoas/_core*.so")]}
    evidence_hashes = {name: digest(path) for name, path in evidence_paths.items()}
    report = {"native_sha256": digest(library), "source_commit": subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": digest(Path(__file__)),
        "evidence_sha256": evidence_hashes,
        "manifest_sha256": {name: digest(HERE / name) for name in
            ("manifest.json", "kernel-pairs-manifest.json", "qwen-additions-manifest.json")},
        "population": [c["case_id"] for c in cases], "rows": [], "failures": [], "status": "running",
        "contract": "a3, distinct GM arguments, identical REVISED_FLAGS except handoff option",
        "device": "not-run", "measurement": "static and scalar replay, not asynchronous device correctness"}
    for case in cases:
        directory = output / case["case_id"]
        directory.mkdir()
        source = directory / "input.pto"
        source.write_bytes(case["source"].read_bytes())
        row = {"case": case["case_id"], "input_sha256": digest(source), "arms": {}, "status": "pass"}
        for arm in ("seed", "handoff"):
            record = {"runs": []}
            row["arms"][arm] = record
            try:
                for kind in ("pto", "cpp"):
                    path = directory / (arm + "." + kind)
                    command = [sys.executable, "-c", SERIAL_DRIVER, str(runtime), "--pto-arch=a3",
                               "--pto-level=" + case["level"], "--enable-insert-sync", *REVISED_FLAGS]
                    if arm == "handoff":
                        command.append("--insert-sync-handoff-planning")
                    if kind == "pto":
                        command.append("--emit-pto-ir")
                    command += [str(source), "-o", str(path)]
                    start = time.monotonic()
                    try:
                        result = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
                        code, stdout, stderr = result.returncode, result.stdout, result.stderr
                    except subprocess.TimeoutExpired as error:
                        code, stdout, stderr = None, error.stdout or b"", error.stderr or b""
                        stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
                        stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
                    path.with_suffix("." + kind + ".log").write_text(stdout + stderr)
                    record["runs"].append({"command": command, "returncode": code, "seconds": time.monotonic() - start})
                    if code != 0 or not path.is_file():
                        raise ValueError("compilation failed: " + kind + " code=" + str(code))
                    record[kind + "_sha256"] = digest(path)
                path = directory / (arm + ".pto")
                record["metrics"] = measure(path, case.get("scenarios", []), case.get("measurement", "replay"),
                                            case.get("normalize_gm_pipe_assembly", False))
                normalized, edits = normalize_gm_pipe_assembly(path.read_text()) if case.get("normalize_gm_pipe_assembly") else (path.read_text(), 0)
                if edits:
                    path = directory / (arm + ".projection.pto")
                    path.write_text(normalized)
                record["projection"] = analyze(path)
                record["projection_input_sha256"] = digest(path)
            except Exception as error:
                record["error"] = str(error)
                row["status"] = "failure"
        if row["status"] == "pass":
            try:
                compare_evidence(row["arms"]["seed"], row["arms"]["handoff"])
                require_effectiveness(row)
                row["payload_contract_parity"] = True
                row["boundary_observations"] = boundary_evidence(directory / "seed.pto", directory / "handoff.pto",
                    case.get("scenarios", []) if case.get("measurement") != "static" else [])
                row["observation_coverage"] = {"replayed_scenarios": len(row["arms"]["handoff"]["metrics"]["scenarios"]),
                    "prefix_checked": sum(r["status"] == "checked" for r in row["boundary_observations"]),
                    "prefix_unsupported": sum(r["status"] == "unsupported" for r in row["boundary_observations"]),
                    "prefix_status": "observed" if row["boundary_observations"] else "not-run"}
            except Exception as error:
                row["status"], row["error"] = "failure", str(error)
        if row["status"] != "pass":
            report["failures"].append(case["case_id"])
        report["rows"].append(row)
        write_report(output, report)
        print(case["case_id"], row["status"], flush=True)
    report["native_unchanged"] = digest(library) == report["native_sha256"]
    report["evidence_unchanged"] = all(digest(path) == evidence_hashes[name] for name, path in evidence_paths.items())
    if not report["native_unchanged"]:
        report["failures"].append("compiler changed during campaign")
    if not report["evidence_unchanged"]:
        report["failures"].append("evidence tooling changed during campaign")
    report["status"] = "completed-with-failures" if report["failures"] else "completed"
    write_report(output, report)
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
