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

"""Collect frozen example/model entries without runtime or auto-sync execution.

This adapter covers source-backed static entry points, not the parameterized
test population. Every inventory seed is recorded, including failed imports,
unresolved factories, and declared drafts. Additional adapters must preserve
their separate collection denominators when combined into an acceptance set.
"""

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from campaign import write_json
from frontend_inventory import inventory_source, source_revision
from provenance import artifact, sha256


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypto-root", type=Path, required=True)
    parser.add_argument("--lib-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.timeout < 1 or args.timeout > 600:
        parser.error("--timeout must be between 1 and 600 seconds")
    for key in ("pypto_root", "lib_root", "python"):
        setattr(args, key, getattr(args, key).absolute())
    args.results = args.results.resolve()
    if not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) > 2:
        parser.error("run under a CPU affinity mask of at most two CPUs")
    return args


def worker_environment(args):
    environment = dict(os.environ)
    environment.update(PYTHONPATH=os.pathsep.join((str(args.pypto_root / "python"), str(args.lib_root))),
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       NUMEXPR_NUM_THREADS="1", PYPTO_CODEGEN_MAX_WORKERS="1",
                       PYPTO_PROG_BUILD_DIR=str(args.results / "outputs"))
    return environment


def run_entry(args, entry):
    record = dict(entry)
    if entry.get("kind") == "syntax-error":
        return dict(record, status="source-syntax-error", inputs=[])
    if entry["draft"]:
        return dict(record, status="declared-draft", inputs=[])
    if not entry["top_level"]:
        return dict(record, status="needs-construction-adapter", inputs=[])
    root = args.pypto_root if entry["family"] == "pypto" else args.lib_root
    output = args.results / "outputs" / entry["case_id"]
    command = [str(args.python), str(args.results / "scripts/frontend_worker.py"),
               "--root", str(root), "--source", entry["source"], "--entry", entry["entry"],
               "--kind", entry["kind"], "--output", str(output)]
    try:
        process = subprocess.run(command, env=worker_environment(args), cwd=args.results,
                                 capture_output=True, timeout=args.timeout, check=False)
        log, code = process.stdout + process.stderr, process.returncode
    except subprocess.TimeoutExpired as error:
        log, code = (error.stdout or b"") + (error.stderr or b""), 124
    diagnostic = args.results / "logs" / f"{entry['case_id']}.txt.gz"
    diagnostic.write_bytes(gzip.compress(log, mtime=0))
    inputs = harvest_inputs(args.results, output, entry)
    status = "collected" if code == 0 and inputs else "collection-failed"
    return dict(record, status=status, return_code=code, command=command, inputs=inputs,
                diagnostic_sha256=sha256(diagnostic))


def harvest_inputs(results, output, entry):
    """Keep partial outputs too; their parent collection failure remains visible."""
    inputs = []
    for index, path in enumerate(sorted(output.rglob("*.pto"))):
        if not path.resolve(strict=True).is_relative_to(output.resolve()):
            raise ValueError("generated input escapes its collection directory")
        case = f"{entry['case_id']}_{index:03d}"
        destination = results / "inputs" / f"{case}.pto"
        destination.write_bytes(path.read_bytes())
        inputs.append({"case_id": case, "source": destination.relative_to(results).as_posix(),
                       "sha256": sha256(destination), "bytes": destination.stat().st_size,
                       "level": "level3", "parent_seed": entry["case_id"],
                       "generated_path": path.relative_to(results).as_posix()})
    return inputs


def prepare(args):
    roots = {"pypto": args.pypto_root, "pypto-lib": args.lib_root}
    sources, entries = {}, []
    for family, root in roots.items():
        sources[family], found = inventory_source(root, family)
        entries.extend(found)
    extensions = sorted((args.pypto_root / "python/pypto").glob("pypto_core*.so"))
    if len(extensions) != 1:
        raise ValueError("require exactly one locally built frontend extension in the frozen checkout")
    probe = [str(args.python), "-c", "import json, platform, importlib.metadata as m; "
             "print(json.dumps({'python': platform.python_version(), 'platform': platform.platform(), "
             "'packages': {name: m.version(name) for name in ('torch', 'numpy', 'pytest', 'nanobind')}}))"]
    runtime = subprocess.run(probe, env=worker_environment(args), capture_output=True,
                             check=True, text=True, timeout=30)
    metadata = {"sources": sources, "frontend_extension": artifact(extensions[0]),
                "frontend_python": artifact(args.python), "runtime": json.loads(runtime.stdout),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "frontend_cmake_cache": artifact(args.pypto_root / "build-corpus/CMakeCache.txt"),
                "invocation": sys.argv, "python": str(args.python), "serial_workers": 1,
                "worker_environment": {key: value for key, value in worker_environment(args).items()
                                       if key in {"PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                  "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                                                  "PYPTO_CODEGEN_MAX_WORKERS", "PYPTO_PROG_BUILD_DIR"}},
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "scope": "static example/model entry points; parameterized tests require a separate adapter"}
    args.results.mkdir(parents=True, exist_ok=False)
    for child in ("logs", "inputs", "outputs", "scripts"):
        (args.results / child).mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        (args.results / "scripts" / path.name).write_bytes(path.read_bytes())
    write_json(args.results / "run.json", metadata)
    write_json(args.results / "inventory.json", entries)
    return metadata, entries


def main():
    args = arguments()
    metadata, entries = prepare(args)
    rows = []
    for index, entry in enumerate(entries, 1):
        row = run_entry(args, entry)
        rows.append(row)
        write_json(args.results / "collection.json", rows)
        label = f"{entry['family']}:{entry['source']}:{entry['entry']}"
        print(f"{index}/{len(entries)} {row['status']} {label}", flush=True)
    inputs = [source for row in rows for source in row["inputs"]]
    with (args.results / "manifest.tsv").open("w", encoding="utf-8", newline="") as stream:
        columns = ("case_id", "source", "sha256", "bytes", "level", "parent_seed", "generated_path")
        writer = csv.DictWriter(stream, columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(inputs)
    stable = all(source_revision(Path(value["root"])) == value for value in metadata["sources"].values())
    for key in ("frontend_extension", "frontend_python", "frontend_cmake_cache"):
        stable = stable and sha256(Path(metadata[key]["path"])) == metadata[key]["sha256"]
    summary = {"source_stable": stable, "seeds": len(entries), "inputs": len(inputs),
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "seed_status": dict(Counter(row["status"] for row in rows)), "scope": metadata["scope"]}
    write_json(args.results / "summary.json", summary)
    write_json(args.results / "hashes.json", {str(path.relative_to(args.results)): sha256(path)
               for path in sorted(args.results.rglob("*")) if path.is_file()})
    # Failed collection rows are evidence, but never successful acceptance rows.
    return int(not stable or not inputs)


if __name__ == "__main__":
    sys.exit(main())
