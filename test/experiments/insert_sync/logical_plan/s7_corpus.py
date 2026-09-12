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
LOCK_SCHEMA = "oahs.s7.corpus-lock.v1"
PORTABLE_ROOT = "$REPO"
ARMS = ("existing", "structured", "structured-hardware")
CONTRACT = "a2a3-mmad-acc-v1"
REGRESSIONS = ("q_proj", "online_softmax", "qk_matmul", "hc_head_linear", "hc_pre_linear", "weights_proj")
STAGES = ("raw-addressed-compatibility", "prepared-pass-entry", "historical-archive")
CLASSES = ("production", "focused", "intentional-refusal", "historical")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def discover(repo, all_samples=False, portable=False):
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
    return dict(schema=SCHEMA, root=PORTABLE_ROOT if portable else str(repo),
                discovery=dict(all_a2a3=bool(all_samples),
                    preparation="identical source bytes; bind module target a2a3 to a3 only"),
                cases=cases, missing_requested=missing,
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


def manifest_digest(manifest):
    return digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())


def verify_discovery(manifest, repo):
    """Require the checked-in identity population to equal fresh discovery."""
    validate_manifest(manifest)
    discovery = manifest.get("discovery")
    if not isinstance(discovery, dict) or not isinstance(discovery.get("all_a2a3"), bool):
        raise ValueError("frozen manifest requires exact discovery parameters")
    actual = discover(repo, discovery["all_a2a3"], portable=True)
    if manifest != actual:
        expected = {case["id"]: case["sha256"] for case in manifest["cases"]}
        found = {case["id"]: case["sha256"] for case in actual["cases"]}
        missing = sorted(set(expected) - set(found))
        added = sorted(set(found) - set(expected))
        changed = sorted(case for case in set(expected) & set(found)
                         if expected[case] != found[case])
        raise ValueError("frozen corpus differs from discovery: " + json.dumps(
            dict(missing=missing, added=added, changed=changed,
                 expected_missing=manifest.get("missing_requested", []),
                 actual_missing=actual.get("missing_requested", []))))
    return dict(inputs=len(actual["cases"]), manifest_sha256=manifest_digest(manifest),
                missing_requested=actual["missing_requested"])


def compact_lock(result, manifest, implementation="same source tree as this lock"):
    """Stable evidence only: no host paths, commands, timings, or binary hashes."""
    expected = {case["id"]: case["sha256"] for case in manifest["cases"]}
    observed = {case["id"]: case["sha256"] for case in result["cases"]}
    if observed != expected:
        raise ValueError("result population does not match the frozen manifest")
    rows = []
    refusal_reasons = sorted({item.get("reason", "")
        for case in result["cases"] for item in case["arms"].values()
        if item["status"] != "applied"})
    reason_ids = {reason: index for index, reason in enumerate(refusal_reasons)}
    for case in result["cases"]:
        arms = {}
        for arm in ARMS:
            item = case["arms"][arm]
            evidence = dict(status=item["status"])
            if item["status"] == "applied":
                mechanisms = item["mechanisms"]
                evidence["mechanism_counts"] = {
                    key: mechanisms[key] for key in
                    ("sets", "waits", "named_barriers", "pipe_all")}
                evidence["mechanisms_sha256"] = digest(json.dumps(
                    mechanisms, sort_keys=True, separators=(",", ":")).encode())
                evidence["output_sha256"] = item["output_sha256"]
            else:
                evidence["reason_id"] = reason_ids[item.get("reason", "")]
            arms[arm] = evidence
        rows.append(dict(id=case["id"], source_sha256=case["sha256"],
                         pre_sync_ir_sha256=case["input_sha256"],
                         adaptations=case["adaptations"], arms=arms))
    aggregate = {stage: dict(pools=data["pools"],
        existing_successful_hardware_refusals=data["existing_successful_hardware_refusals"])
        for stage, data in result["summary"].items()}
    return dict(schema=LOCK_SCHEMA, manifest_sha256=manifest_digest(manifest),
                implementation=implementation,
                preparation=manifest.get("discovery", {}).get("preparation"),
                refusal_reasons=refusal_reasons,cases=rows, summary=aggregate,
                missing_requested=result["missing_requested"], device="NOT_RUN")


def validate_lock(lock, manifest):
    validate_manifest(manifest)
    if lock.get("schema") != LOCK_SCHEMA or \
       lock.get("manifest_sha256") != manifest_digest(manifest):
        raise ValueError("corpus lock does not name the exact manifest")
    if not isinstance(lock.get("implementation"), str) or not lock["implementation"]:
        raise ValueError("corpus lock requires implementation provenance")
    expected = {case["id"]: case["sha256"] for case in manifest["cases"]}
    rows = lock.get("cases")
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("corpus lock population differs from manifest")
    reasons=lock.get("refusal_reasons")
    if not isinstance(reasons,list) or any(not isinstance(reason,str) for reason in reasons):
        raise ValueError("corpus lock refusal dictionary is invalid")
    seen = set()
    for row in rows:
        if row.get("id") in seen or expected.get(row.get("id")) != row.get("source_sha256"):
            raise ValueError("corpus lock input identity differs from manifest")
        seen.add(row["id"])
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("pre_sync_ir_sha256", "")):
            raise ValueError("corpus lock requires a pre-sync IR hash")
        if set(row.get("arms", {})) != set(ARMS) or any(
                row["arms"][arm].get("status") not in
                ("applied", "refused", "measurement-failed") for arm in ARMS):
            raise ValueError("corpus lock has an invalid arm result")
        for arm in ARMS:
            item=row["arms"][arm]
            if item["status"]!="applied" and (not isinstance(item.get("reason_id"),int) or
                                               item["reason_id"]<0 or item["reason_id"]>=len(reasons)):
                raise ValueError("corpus lock has an invalid refusal reference")
    if seen != set(expected):
        raise ValueError("corpus lock omits a manifest input")
    return dict(inputs=len(rows), manifest_sha256=lock["manifest_sha256"])


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


def sweep(manifest, opt, python_root, output, timeout, require_historical,
          repo=ROOT, lock_output=None):
    validate_manifest(manifest)
    root = Path(repo).resolve() if manifest["root"] == PORTABLE_ROOT else Path(manifest["root"])
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
    lock = compact_lock(result, manifest)
    (output / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    if lock_output:
        with Path(lock_output).open("x") as stream:
            json.dump(lock, stream, indent=2)
            stream.write("\n")
    if require_historical and not any(r["classification"] == "historical" for r in rows):
        raise RuntimeError("historical GEMM missing: results retained, coverage request incomplete")
    print(json.dumps(result["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover")
    d.add_argument("--repo", type=Path, default=ROOT)
    d.add_argument("--all-a2a3", action="store_true")
    d.add_argument("--portable", action="store_true",
                   help="write a repository-relative manifest suitable for version control")
    d.add_argument("--output", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--repo", type=Path, default=ROOT)
    v.add_argument("--lock", type=Path)
    l = sub.add_parser("lock")
    l.add_argument("--manifest", type=Path, required=True)
    l.add_argument("--summary", type=Path, required=True)
    l.add_argument("--implementation", required=True)
    l.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("run")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--opt", type=Path, required=True)
    s.add_argument("--python-root", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--repo", type=Path, default=ROOT,
                   help="repository root for a portable $REPO manifest")
    s.add_argument("--lock-output", type=Path,
                   help="also create a compact version-controlled result lock")
    s.add_argument("--timeout", type=float, default=120, help="external process timeout; NOT an analysis quota")
    s.add_argument("--require-historical", action="store_true")
    args = parser.parse_args()
    if args.command == "discover":
        report = discover(args.repo, args.all_a2a3, args.portable)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(json.dumps(dict(inputs=len(report["cases"]), missing=report["missing_requested"])))
    elif args.command == "verify":
        manifest = json.loads(args.manifest.read_text())
        report = verify_discovery(manifest, args.repo)
        if args.lock:
            report["lock"] = validate_lock(json.loads(args.lock.read_text()), manifest)
        print(json.dumps(report, indent=2))
    elif args.command == "lock":
        manifest = json.loads(args.manifest.read_text())
        lock = compact_lock(json.loads(args.summary.read_text()), manifest,
                            args.implementation)
        validate_lock(lock, manifest)
        with args.output.open("x") as stream:
            json.dump(lock, stream, indent=2)
            stream.write("\n")
        print(json.dumps(dict(inputs=len(lock["cases"]),
                              manifest_sha256=lock["manifest_sha256"])))
    else:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        sweep(json.loads(args.manifest.read_text()), args.opt.resolve(), args.python_root.resolve(),
              args.output.resolve(), args.timeout, args.require_historical,
              args.repo.resolve(), args.lock_output)


if __name__ == "__main__":
    main()
