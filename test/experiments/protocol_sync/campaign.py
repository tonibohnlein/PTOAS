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

"""Run a bounded, read-only ProtocolSync corpus campaign and preserve its inputs."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from census import summarize
from provenance import build_metadata, sha256
from records import parse_diagnostics


REPO = Path(__file__).resolve().parents[3]
DUMPS = {"lane": "lane-frontiers", "storage": "storage-tracks", "concrete": "concrete-verification",
         "acceptance": "storage-tracks"}


def write_json(path, value):
    """Write generated evidence in a deterministic JSON representation."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_output(*arguments):
    return subprocess.run(
        ["git", "-C", str(REPO), *arguments], capture_output=True,
        text=True, check=True, timeout=30,
    ).stdout.strip()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(DUMPS), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ptoas", type=Path, default=REPO / "build/tools/ptoas/ptoas")
    parser.add_argument("--build-root", type=Path, default=REPO / "build")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--arch", choices=("a2", "a3"), default="a3")
    parser.add_argument("--gm-alias", choices=("may-alias", "assume-disjoint-arguments"))
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    for key in ("input_root", "ptoas", "python", "build_root"):
        setattr(args, key, getattr(args, key).absolute())
    args.results = args.results.resolve()
    return args


def inputs(args):
    if args.mode == "concrete":
        rows = [{"case_id": path.parents[1].name, "source": str(path.relative_to(args.input_root))}
                for path in sorted(args.input_root.glob("*/frontier_on/out.pto"))]
    else:
        if args.manifest is None:
            raise ValueError("lane/storage campaigns require --manifest")
        with args.manifest.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    seen = set()
    for row in rows:
        case = row["case_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", case) or case in seen:
            raise ValueError(f"invalid or duplicate case identifier: {case!r}")
        seen.add(case)
        source = (args.input_root / row["source"]).resolve(strict=True)
        if not source.is_relative_to(args.input_root.resolve()):
            raise ValueError(f"input escapes corpus root: {case}")
        row["input_sha256"] = sha256(source)
        if row.get("level", "level3") not in {"level2", "level3"}:
            raise ValueError(f"unsupported input pipeline level: {case}")
        if row.get("sha256") and row["sha256"] != row["input_sha256"]:
            raise ValueError(f"input differs from manifest hash: {case}")
        if row.get("bytes") and int(row["bytes"]) != source.stat().st_size:
            raise ValueError(f"input differs from manifest size: {case}")
    if len(rows) != args.expected_rows or not rows:
        raise ValueError(f"expected {args.expected_rows} inputs, found {len(rows)}")
    return rows


def probe_command(args, row, dump, strict=False):
    """Use one declared architecture and alias contract for every probe."""
    flags = ["--protocol-sync-mixed", "--protocol-sync-fallback=fail"] if strict else [
        "--enable-insert-sync", "--protocol-sync-analysis-only"]
    if args.gm_alias is not None:
        flags.append(f"--protocol-sync-gm-alias={args.gm_alias}")
    return [str(args.python), str(args.ptoas), f"--pto-arch={args.arch}",
            f"--pto-level={row.get('level', 'level3')}", "--emit-pto-ir", *flags,
            f"--protocol-sync-dump={dump}", "--protocol-sync-statistics",
            str(args.input_root / row["source"]), "-o", "/dev/null"]


def execute_probe(args, row, dump, suffix="", strict=False):
    """Keep actual strict admission separate from every analysis-only blocker."""
    command = probe_command(args, row, dump, strict)
    try:
        result = subprocess.run(command, capture_output=True, timeout=120, check=False)
        output = result.stdout + result.stderr
        return_code = result.returncode
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"") + (error.stderr or b"")
        return_code = 124
    diagnostic = args.results / "diagnostics" / f"{row['case_id']}{suffix}.txt.gz"
    diagnostic.write_bytes(gzip.compress(output, mtime=0))
    record = dict(row, command=command, return_code=return_code,
                  diagnostics_sha256=sha256(diagnostic), target_arch=args.arch,
                  gm_alias_override=args.gm_alias)
    record.update(parse_diagnostics(output.decode("utf-8", errors="replace")))
    return record


def execute_row(args, row):
    """Collect residuals even when strict emission stops at the extraction gate."""
    record = execute_probe(args, row, DUMPS[args.mode])
    if args.mode == "acceptance":
        record["empty_world"] = execute_probe(args, row, "residuals", ".residuals")
        record["strict_mixed"] = execute_probe(args, row, "plan", ".strict", strict=True)
    return record


def prepare_campaign(args):
    """Validate inputs and freeze provenance before any compiler subprocess."""
    tracked_diff = git_output("diff", "HEAD", "--")
    experiment_path = str(Path(__file__).parent.relative_to(REPO))
    experiment_dirty = git_output("status", "--porcelain", "--", experiment_path)
    if (tracked_diff or experiment_dirty) and not args.allow_dirty:
        raise ValueError("tracked worktree is dirty; commit the experiment or use --allow-dirty for debugging")
    rows = inputs(args)
    toolchain = build_metadata(args.build_root, args.ptoas)
    args.results.mkdir(parents=True, exist_ok=False)
    (args.results / "diagnostics").mkdir()
    (args.results / "scripts").mkdir()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        (args.results / "scripts" / path.name).write_bytes(path.read_bytes())
    metadata = {
        "source_commit": git_output("rev-parse", "HEAD"),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff.encode()).hexdigest(),
        "tracked_clean": not tracked_diff,
        "experiment_clean": not experiment_dirty,
        "toolchain": toolchain,
        "mode": args.mode, "workers": args.workers,
        "target_arch": args.arch, "gm_alias_override": args.gm_alias,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": sys.argv, "python_version": sys.version,
        "input_root": str(args.input_root),
        "inputs_sha256": hashlib.sha256(json.dumps(
            [(row["case_id"], row["input_sha256"]) for row in rows]).encode()).hexdigest(),
        "script_sha256": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
    }
    if args.manifest:
        metadata["manifest_sha256"] = sha256(args.manifest)
    if tracked_diff:
        (args.results / "tracked.patch").write_text(tracked_diff, encoding="utf-8")
    write_json(args.results / "run.json", metadata)
    return rows, metadata, tracked_diff


def run_rows(args, rows):
    """Persist each completed row while keeping compiler concurrency bounded."""
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (args.results / "rows.jsonl").open(
            "w", encoding="utf-8") as stream:
        for index, record in enumerate(pool.map(lambda row: execute_row(args, row), rows), 1):
            results.append(record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(rows):
                print(f"completed {index}/{len(rows)}", flush=True)
    return results


def finish_campaign(args, metadata, tracked_diff, summary):
    """Reject moving-source evidence and hash the finalized artifact set."""
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["source_stable"] = (
        metadata["source_commit"] == git_output("rev-parse", "HEAD")
        and tracked_diff == git_output("diff", "HEAD", "--")
        and metadata["toolchain"]["compiler_library"]["sha256"] == sha256(
            Path(metadata["toolchain"]["compiler_library"]["path"]))
        and metadata["script_sha256"] == {
            path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    )
    write_json(args.results / "run.json", metadata)
    write_json(args.results / "summary.json", summary)
    write_json(args.results / "hashes.json", {
        str(path.relative_to(args.results)): sha256(path)
        for path in sorted(args.results.rglob("*")) if path.is_file()
    })


def main():
    args = arguments()
    rows, metadata, tracked_diff = prepare_campaign(args)
    results = run_rows(args, rows)
    summary = summarize(results, args.mode)
    finish_campaign(args, metadata, tracked_diff, summary)
    print(json.dumps(summary["totals"], indent=2, sort_keys=True))
    return int(summary["totals"]["failed_rows"] != 0
               or summary["totals"]["missing_statistics_rows"] != 0
               or summary["totals"].get("failed_acceptance_probes", 0) != 0 or not metadata["source_stable"])


if __name__ == "__main__":
    sys.exit(main())
