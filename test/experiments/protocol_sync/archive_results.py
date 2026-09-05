# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Validate and archive raw campaign evidence with compact reviewable summaries."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from campaign import write_json
from provenance import sha256


def verify_campaign(root):
    """Validate every archived file against the runner's completion manifest."""
    manifest = json.loads((root / "hashes.json").read_text(encoding="utf-8"))
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("campaign artifacts must not contain symbolic links")
    for relative, expected in manifest.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"invalid campaign artifact: {relative}")
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual_files != set(manifest) | {"hashes.json"}:
        raise ValueError("campaign contains files outside its completion manifest")
    metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if not metadata["source_stable"]:
        raise ValueError("cannot freeze a moving-source campaign")
    return metadata


def compact_records(root):
    """Retain input, diagnostic and exact per-row hashes without duplicating IR."""
    records = []
    with (root / "rows.jsonl").open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            records.append({key: row[key] for key in ("case_id", "input_sha256", "diagnostics_sha256", "return_code")})
            records[-1]["row_sha256"] = hashlib.sha256(line).hexdigest()
    return records


def summarize_campaign(root, metadata):
    """Keep detailed classification rows in the archive and all aggregates here."""
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summary = {key: value for key, value in summary.items() if not key.endswith("_records")}
    if "acceptance" in summary:
        summary["acceptance"] = {key: value for key, value in summary["acceptance"].items() if key != "records"}
    return {"run": metadata, "summary": summary, "row_hashes": compact_records(root),
            "artifact_manifest_sha256": sha256(root / "hashes.json"),
            "rows_jsonl_sha256": sha256(root / "rows.jsonl")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, action="append", required=True)
    parser.add_argument("--include", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    roots = [path.resolve(strict=True) for path in args.campaign]
    args.include = [path.resolve(strict=True) for path in args.include]
    args.archive = args.archive.resolve()
    args.output = args.output.resolve()
    if any(args.archive.is_relative_to(root) or args.output.is_relative_to(root) for root in roots):
        raise ValueError("archive and summary outputs must be outside input campaigns")
    if len({path.name for path in roots}) != len(roots):
        raise ValueError("campaign archive names must be distinct")
    if len({path.name for path in args.include}) != len(args.include) or any(
            not path.is_file() for path in args.include):
        raise ValueError("included artifacts must be distinct regular files")
    metadata = [verify_campaign(root) for root in roots]
    args.output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive, "x:gz") as archive:
        for root in roots:
            archive.add(root, arcname=root.name)
        for path in args.include:
            archive.add(path, arcname=f"inputs/{path.name}")
    for root, run in zip(roots, metadata):
        write_json(args.output / f"{root.name}.json", summarize_campaign(root, run))
    write_json(args.output / "archive.json", {
        "archive": str(args.archive), "sha256": sha256(args.archive), "bytes": args.archive.stat().st_size,
        "clean_committed_campaigns": all(run["tracked_clean"] and run["experiment_clean"] for run in metadata),
        "included_artifacts": {path.name: sha256(path) for path in args.include},
        "generator_sha256": sha256(Path(__file__)),
    })


if __name__ == "__main__":
    main()
