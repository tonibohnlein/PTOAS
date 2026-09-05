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

"""Check C++ emission and concrete verification after native corpus admission."""

import argparse
import gzip
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from acceptance import classify_strict
from campaign import REPO, write_json
from provenance import sha256
from records import parse_diagnostics


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--examples", nargs="+", default=["k005", "k106", "k128", "k358", "k010"])
    args = parser.parse_args()
    args.results = args.results.resolve()
    return args


def checked_path(root, name, expected_hash):
    """Do not trust stored absolute paths or execute recorded command vectors."""
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or sha256(path) != expected_hash:
        raise ValueError(f"artifact escaped its root or changed: {name}")
    return path


def collect_jobs(args):
    """Construct explicit probes from validated inputs and declared contracts."""
    jobs = []
    provenance = []
    legacy_seen = set()
    modes = set()
    for directory in args.campaign:
        directory = directory.resolve(strict=True)
        metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if not metadata.get("source_stable") or metadata["mode"] != "acceptance":
            raise ValueError("follow-up requires a completed, source-stable acceptance campaign")
        arch = metadata["target_arch"]
        alias = metadata["gm_alias_override"]
        if arch not in {"a2", "a3"} or alias not in {"may-alias", "assume-disjoint-arguments"}:
            raise ValueError("follow-up requires explicit supported target and GM contracts")
        mode = f"{arch}-{alias}"
        if mode in modes:
            raise ValueError(f"duplicate campaign mode: {mode}")
        modes.add(mode)
        toolchain = metadata["toolchain"]
        for key in ("compiler_library", "launcher"):
            checked_path(REPO, toolchain[key]["path"], toolchain[key]["sha256"])
        provenance.append({"directory": str(directory), "run_sha256": sha256(directory / "run.json"),
                           "rows_sha256": sha256(directory / "rows.jsonl"), "metadata": metadata})
        rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in rows:
            case = row["case_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", case) or row.get("level", "level3") not in {"level2", "level3"}:
                raise ValueError("invalid case identifier or pipeline level")
            source = checked_path(Path(metadata["input_root"]), row["source"], row["input_sha256"])
            base = [sys.executable, toolchain["launcher"]["path"], f"--pto-arch={arch}",
                    f"--pto-level={row.get('level', 'level3')}"]
            native = [f"--protocol-sync-gm-alias={alias}", "--protocol-sync-fallback=fail",
                      "--protocol-sync-statistics"]
            probes = []
            if classify_strict(row["strict_mixed"]) == "admitted":
                emitted = row["strict_mixed"]["emitted_ir"]
                ir_path = checked_path(directory, emitted["path"], emitted["sha256"])
                probes.append(("mixed-cpp", base + native + ["--protocol-sync-mixed", str(source)], ".cpp"))
                # Emitted IR is already physically assigned, regardless of input level.
                concrete = base[:3] + ["--pto-level=level3", f"--protocol-sync-gm-alias={alias}",
                                      "--emit-pto-ir", "--protocol-sync-analysis-only",
                                      "--protocol-sync-dump=concrete-verification", str(ir_path)]
                probes.append(("concrete", concrete, ".pto"))
            if case in args.examples:
                probes.append(("direct-cpp", base + native + ["--protocol-sync-direct-repair", str(source)], ".cpp"))
                if (arch, case) not in legacy_seen:
                    probes.append(("legacy-cpp", base + ["--enable-insert-sync", str(source)], ".cpp"))
                    legacy_seen.add((arch, case))
            for kind, command, suffix in probes:
                jobs.append({"mode": mode, "case_id": case, "kind": kind, "command": command,
                             "source": row["source"], "input_sha256": row["input_sha256"], "suffix": suffix})
    return jobs, provenance


def successful_probe(kind, return_code, functions, output_exists):
    """An analysis-only exit code is not a concrete-verifier acceptance verdict."""
    if return_code != 0 or not output_exists:
        return False
    if kind == "concrete":
        return bool(functions) and all(f.get("verdict", {}).get("status") == "accepted" for f in functions)
    if kind in {"mixed-cpp", "direct-cpp"}:
        return bool(functions) and all(f.get("statistics", {}).get("producer") ==
                                      "protocol-plus-direct-residuals" for f in functions)
    return True


def execute(job, results):
    """Keep raw diagnostics and generated source, never invoke a device toolchain."""
    prefix = f"{job['mode']}-{job['case_id']}-{job['kind']}"
    output_path = results / (prefix + job["suffix"])
    command = job["command"] + ["-o", str(output_path)]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=120)
        output = result.stdout + result.stderr
        return_code = result.returncode
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"") + (error.stderr or b"")
        return_code = 124
    diagnostic = results / (prefix + ".txt.gz")
    diagnostic.write_bytes(gzip.compress(output, mtime=0))
    functions = parse_diagnostics(output.decode("utf-8", errors="replace"))["functions"]
    verdict = successful_probe(job["kind"], return_code, functions, output_path.is_file())
    return dict(job, command=command, return_code=return_code, successful=verdict, functions=functions,
                diagnostic_sha256=sha256(diagnostic),
                output_sha256=sha256(output_path) if output_path.exists() else None)


def main():
    args = arguments()
    jobs, provenance = collect_jobs(args)
    args.results.mkdir(parents=True, exist_ok=False)
    (args.results / "native_followup.py").write_bytes(Path(__file__).read_bytes())
    write_json(args.results / "provenance.json", provenance)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (args.results / "rows.jsonl").open(
            "w", encoding="utf-8") as stream:
        for index, record in enumerate(pool.map(lambda job: execute(job, args.results), jobs), 1):
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            if index % 25 == 0 or index == len(jobs):
                print(f"completed {index}/{len(jobs)}", flush=True)
    stable = all(sha256(Path(p["metadata"]["toolchain"]["compiler_library"]["path"])) ==
                 p["metadata"]["toolchain"]["compiler_library"]["sha256"] for p in provenance)
    summary = {"compiler_stable": stable, "counts": dict(Counter(
        f"{r['mode']}/{r['kind']}/{'passed' if r['successful'] else 'failed'}" for r in records))}
    write_json(args.results / "summary.json", summary)
    write_json(args.results / "hashes.json", {p.name: sha256(p) for p in sorted(args.results.iterdir()) if p.is_file()})
    print(json.dumps(summary, indent=2, sort_keys=True))
    # Example direct-only rejection is a measurement, not a failed audit.
    return int(not stable or any(not r["successful"] for r in records if r["kind"] in {"mixed-cpp", "concrete"}))


if __name__ == "__main__":
    sys.exit(main())
