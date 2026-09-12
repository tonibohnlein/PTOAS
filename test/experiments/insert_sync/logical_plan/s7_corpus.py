#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Discover/freeze and compare A2/A3 inputs without conflating preparation stages.

Discovery is deliberately a RAW compatibility population, not a production
admission claim. For production admission, supply a manifest of snapshots
captured immediately before InsertSync after the real preparation pipeline.
Every arm consumes the identical immutable snapshot. No legacy seed is used.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SCHEMA = "oahs.s7.corpus.v1"
ARMS = ("existing", "structured", "structured-hardware")
CONTRACT = "a2a3-mmad-acc-v1"
REGRESSIONS = ("q_proj", "online_softmax", "qk_matmul", "hc_head_linear", "hc_pre_linear", "weights_proj")
STAGES = ("raw-addressed-compatibility", "prepared-pass-entry", "historical-archive")
CLASSES = ("production", "focused", "intentional-refusal", "historical")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def discover(repo, all_samples=False):
    """Exact filenames/contents, no inference of missing inputs or stage fixes."""
    repo = Path(repo).resolve()
    choices = {}
    for path in sorted((repo / "test/samples").rglob("*.pto")):
        if not all_samples and path.stem not in REGRESSIONS:
            continue
        text = path.read_text()
        arch = re.search(r'pto\.target_arch\s*=\s*"(a2a3|a2|a3)"', text)
        if arch:
            choices[path] = ("production", "raw-addressed-compatibility")
    # Preserve the existing seven immutable benchmark sources when available.
    frozen = repo / "test/experiments/insert_sync/logical_plan/checkpoint/manifest.json"
    if frozen.exists():
        for case in json.loads(frozen.read_text())["cases"]:
            path = repo / case["source"]
            if not path.is_file() or digest(path.read_bytes()) != case["sha256"]:
                raise ValueError("missing/changed frozen population source: " + case["case_id"])
            choices[path] = ("production" if "test/samples/" in case["source"] else "focused",
                             "raw-addressed-compatibility")
    for path in sorted((repo / "test/experiments/insert_sync/logical_plan/hardware_inputs").glob("*.pto")):
        choices[path] = ("focused", "raw-addressed-compatibility")
    archives = sorted((repo / "test/experiments/insert_sync/logical_plan/archive/historical_gemm").glob("*.pto"))
    for path in archives:
        choices[path] = ("historical", "historical-archive")
    cases = [dict(id=str(p.relative_to(repo)), path=str(p.relative_to(repo)), sha256=digest(p.read_bytes()),
                  classification=c, stage=s, gm_contract="may-alias")
             for p, (c, s) in sorted(choices.items())]
    present = {p.stem for p in choices}
    missing = [name for name in REGRESSIONS if name not in present]
    if not archives:
        missing.append("historical_gemm: run the pinned recover_historical_gemm.py first")
    return dict(schema=SCHEMA, root=str(repo), cases=cases, missing_requested=missing,
                note="Raw compatibility only. Classify real pre-pass snapshots separately. GM defaults to may-alias; change only for a qualified caller contract.")


def validate_manifest(manifest):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("wrong corpus schema")
    ids = set()
    for case in manifest["cases"]:
        if not case.get("id") or case["id"] in ids:
            raise ValueError("missing/duplicate input identity")
        ids.add(case["id"])
        if case.get("stage") not in STAGES or case.get("classification") not in CLASSES:
            raise ValueError("explicit stage/classification required")
        if case.get("gm_contract") not in ("may-alias", "assume-disjoint-arguments"):
            raise ValueError("unknown GM contract; no all-access nonalias mode")
        if not re.fullmatch(r"[0-9a-f]{64}", case.get("sha256", "")):
            raise ValueError("input fingerprint required")
        if case["stage"] == "prepared-pass-entry" and not case.get("preparation"):
            raise ValueError("prepared snapshot requires pipeline/build provenance")
    if not ids:
        raise ValueError("empty corpus is not an admission study")


def prepare_bytes(data, stage):
    # This is target selection only. Do not repair geometry, inject temporaries,
    # erase manual synchronization, or change payload/control to admit an input.
    text = data.decode()
    changes = []
    if stage == "prepared-pass-entry":
        if re.search(r'pto\.target_arch\s*=\s*"a2a3"', text):
            raise ValueError("prepared snapshot must already select a concrete target")
    else:
        text, n = re.subn(r'(pto\.target_arch\s*=\s*)"a2a3"', r'\1"a3"', text)
        if n:
            changes.append("explicit target binding a2a3 -> a3; no payload edits")
    return text.encode(), changes


def summarize(rows):
    out = {}
    for stage in STAGES:
        selected = [r for r in rows if r["stage"] == stage]
        if not selected:
            continue
        pools = {}
        for classification in CLASSES:
            population = [r for r in selected if r["classification"] == classification]
            if not population:
                continue
            pools[classification] = dict(inputs=len(population),
                accepted={a: sum(r["arms"][a]["status"] == "applied" for r in population) for a in ARMS})
        failures = Counter()
        common = []
        for r in selected:
            if r["arms"]["existing"]["status"] == "applied":
                if r["arms"]["structured-hardware"]["status"] != "applied":
                    failures[r["arms"]["structured-hardware"].get("reason", "unclassified failure")] += 1
                else:
                    common.append(r["id"])
        out[stage] = dict(pools=pools, existing_successful_hardware_refusals=dict(failures),
                          common_success_ids=common)
    return out


def metrics(path, python_root):
    # Parse actual IR; no grep over comments, and no summing different mechanisms.
    sys.path.insert(0, str(python_root))
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    from measure import static_metrics
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        module = ir.Module.parse(path.read_text())
        if not module.operation.verify():
            raise ValueError("invalid emitted IR")
        data = static_metrics(module.operation)
    counts = data["counts"]
    named, retirement = Counter(), 0
    for key, count in counts.items():
        if key.startswith("barrier:"):
            pipe = re.search(r"PIPE_[A-Z0-9]+", key).group(0)
            if pipe == "PIPE_ALL":
                retirement += count  # reported ALL sites, not assumed all terminal
            else:
                named[pipe] += count
    return dict(sets=counts.get("pto.set_flag", 0), waits=counts.get("pto.wait_flag", 0),
                named_barriers=dict(named), pipe_all=retirement,
                event_ids_by_direction=data["event_ids_by_direction"],
                placements_sha256=data["placement_sha256"])


def sweep(manifest, opt, python_root, output, timeout, require_historical):
    validate_manifest(manifest)
    root = Path(manifest["root"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for key in ("PTOAS_STRUCTURED_PLAN_JSON", "PTOAS_LOGICAL_TRACE"):
        env.pop(key, None)
    rows = []
    for number, case in enumerate(manifest["cases"]):
        source = Path(case["path"])
        if not source.is_absolute():
            source = root / source
        data = source.read_bytes()
        if digest(data) != case["sha256"]:
            raise ValueError("changed corpus input " + case["id"])
        folder = output / f"{number:04d}"
        folder.mkdir()
        prepared, changes = prepare_bytes(data, case["stage"])
        snapshot = folder / "input.pto"
        snapshot.write_bytes(prepared)
        row = dict(case, input_sha256=digest(prepared), adaptations=changes, arms={})
        # Rotate order: this remains one diagnostic round, NOT a timing gate.
        arms = ARMS[number % 3:] + ARMS[:number % 3]
        for arm in arms:
            planner = "existing" if arm == "existing" else "structured"
            contract = CONTRACT if arm == "structured-hardware" else "conservative"
            path = folder / (arm + ".pto")
            option = f"-pto-insert-sync=planner={planner} hardware-contract={contract} gm-alias={case['gm_contract']}"
            command = [str(opt), str(snapshot), option, "-o", str(path)]
            start = time.monotonic()
            try:
                result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=timeout)
                code, stdout, stderr = result.returncode, result.stdout, result.stderr
            except subprocess.TimeoutExpired as error:
                code = "harness-timeout"
                def text(value):
                    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
                stdout, stderr = text(error.stdout), text(error.stderr)
            seconds = time.monotonic() - start
            (folder / (arm + ".stdout")).write_text(stdout)
            (folder / (arm + ".stderr")).write_text(stderr)
            item = dict(status="applied" if code == 0 else "refused", exit=code, command=command, seconds=seconds)
            if code == 0:
                try:
                    item["mechanisms"] = metrics(path, python_root)
                except Exception as error:
                    item["status"] = "measurement-failed"
                    item["reason"] = type(error).__name__ + ": " + str(error)
                item["output_sha256"] = digest(path.read_bytes())
            else:
                diagnostic = next((line.strip() for line in stderr.splitlines() if "error:" in line), stderr[-1000:])
                item["reason"] = (diagnostic.partition("error:")[2].strip() if "error:" in diagnostic else diagnostic) or str(code)
            row["arms"][arm] = item
        rows.append(row)
        (output / "results.json").write_text(json.dumps(dict(cases=rows), indent=2) + "\n")
    result = dict(status="completed", cases=rows, summary=summarize(rows),
                  missing_requested=manifest.get("missing_requested", []),
                  opt_sha256=digest(opt.read_bytes()), device="NOT_RUN",
                  timing="single diagnostic round, no matched performance acceptance claim")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if require_historical and not any(r["classification"] == "historical" for r in rows):
        raise RuntimeError("historical GEMM missing: results retained, coverage request incomplete")
    print(json.dumps(result["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--repo", type=Path, default=ROOT)
    d.add_argument("--all-a2a3", action="store_true")
    d.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("run")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--opt", type=Path, required=True)
    s.add_argument("--python-root", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--timeout", type=float, default=120, help="external process timeout; NOT an analysis quota")
    s.add_argument("--require-historical", action="store_true")
    args = parser.parse_args()
    if args.command == "discover":
        report = discover(args.repo, args.all_a2a3)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps(dict(inputs=len(report["cases"]), missing=report["missing_requested"])))
    else:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        sweep(json.loads(args.manifest.read_text()), args.opt.resolve(), args.python_root.resolve(),
              args.output.resolve(), args.timeout, args.require_historical)


if __name__ == "__main__":
    main()
