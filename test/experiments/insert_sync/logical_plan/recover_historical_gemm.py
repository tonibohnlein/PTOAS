# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""Restore the exact historical fixture from a pinned local Git object.

Read-only Git commands only: no checkout, fetch, reset, or branch change. The
archive is outside automatic lit discovery because it retains historical RUN
lines and syntax. This utility does not claim current OAHS/GEMM admission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SPEC = {
    "repository": "https://github.com/tonibohnlein/PTOAS.git",
    "commit": "5eab65e919ca053cc93a96ab09efd3ffa434a70b",
    "path": "test/lit/pto/canonical_sync_historical_gemm_noalias.pto",
    "git_blob": "940f08d2bc0eddabb75db413aa629ccaac3c5219",
    "frontend_source": "huawei-csl/pto-dsl@36cd417e230a84a80870f564de27539ccc7c4a75",
    "frontend_version": "20260305",
    "raw_generated_sha256": "3db5d353475ee920857d16e5c3533ab23e0c6d2c38e22b57235c0e85a9a6a380",
    "planned_presync_sha256": "d5c3e847cc70185581b87fa686a0ccf285e515cb9b7c7dc8b4f123be2028dcf1",
    "alias_contract": "Retain original pairwise A/B/C contract; never assume all GM accesses are disjoint.",
    "qualification": "Historical source archive only; current syntax adaptation, construction and device execution unrun."
}


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def extract(repo, spec):
    result = subprocess.run(["git", "-C", str(repo), "show", spec["commit"] + ":" + spec["path"]],
                            capture_output=True)
    if result.returncode:
        raise RuntimeError("Pinned object is not available locally. Review and run:\n"
                           f"git -C {repo!s} fetch --no-tags {spec['repository']} {spec['commit']}\n"
                           "Then rerun this read-only extraction. Git error: " +
                           result.stderr.decode(errors="replace"))
    if blob_sha(result.stdout) != spec["git_blob"]:
        raise RuntimeError("Historical fixture blob mismatch; no output written")
    return result.stdout


def archive(output, data, spec):
    if blob_sha(data) != spec["git_blob"]:
        raise ValueError("unverified fixture bytes")
    output = Path(output).absolute()
    for parent in (output, *output.parents):
        if parent.is_symlink():
            raise ValueError("symlink output path refused")
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing archive")
    output.mkdir(parents=True, exist_ok=False)
    try:
        name = Path(spec["path"]).name
        with (output / name).open("xb") as stream:
            stream.write(data)
        manifest = {**spec, "archived_sha256": hashlib.sha256(data).hexdigest(),
                    "archived_file": name, "byte_count": len(data), "adaptations": []}
        with (output / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=HERE / "archive/historical_gemm")
    args = parser.parse_args()
    data = extract(args.repo, SPEC)
    manifest = archive(args.output, data, SPEC)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
