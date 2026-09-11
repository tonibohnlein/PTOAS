# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
"""Tests runner accounting and archive integrity, not native synchronization."""
import json
from pathlib import Path
import subprocess
import tempfile
from benchmark_buffers import ratio_summary
from recover_historical_gemm import archive, blob_sha, extract


def main():
    if not __debug__:
        raise RuntimeError("assertions required")
    rows = []
    for i in range(3):
        rows.extend([dict(round=i, arm="existing", status="applied", seconds=1),
                     dict(round=i, arm="structured", status="applied", seconds=1.5+i*0.1)])
    result = ratio_summary(rows, "structured", 3)
    assert result["complete"] and result["paired_samples"] == 3 and result["median_ratio"] == 1.6
    failed = [dict(r) for r in rows]
    failed[-1].update(status="harness-timeout", seconds=0.01)
    result = ratio_summary(failed, "structured", 3)
    assert not result["complete"] and result["paired_samples"] == 2
    result = ratio_summary(rows[:-1], "structured", 3)
    assert not result["complete"]
    result = ratio_summary(rows + [dict(round=-1, arm="structured", status="applied", seconds=999)], "structured", 3)
    assert result["median_ratio"] == 1.6
    try:
        ratio_summary(rows + [rows[0]], "structured", 3)
        raise AssertionError("duplicate paired sample accepted")
    except ValueError:
        pass
    # Synthetic local Git object tests the extraction mechanism. It is NOT a
    # substitute for having fetched the historical GEMM object in this runtime.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary);repo = root / "repo";repo.mkdir()
        def git(*args):
            return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL)
        git("init", "-q")
        source = b"// synthetic archival witness\nmodule {}\n"
        (repo / "source.pto").write_bytes(source)
        git("add", "source.pto")
        git("-c", "user.name=OAHS test", "-c", "user.email=test@invalid", "commit", "-qm", "fixture")
        spec = dict(repository="unused", commit=git("rev-parse", "HEAD").decode().strip(),
                    path="source.pto", git_blob=blob_sha(source))
        assert extract(repo, spec) == source
        archive(root / "archive", source, spec)
        assert (root / "archive/source.pto").read_bytes() == source
        assert json.loads((root / "archive/manifest.json").read_text())["adaptations"] == []
        try:
            archive(root / "archive", source, spec)
            raise AssertionError("overwrote existing archive")
        except FileExistsError:
            pass
        try:
            archive(root / "bad", source+b"x", spec)
            raise AssertionError("accepted altered source")
        except ValueError:
            pass
        (root / "link").symlink_to(root / "archive", target_is_directory=True)
        try:
            archive(root / "link/subdir", source, spec)
            raise AssertionError("accepted a symlink output path")
        except ValueError:
            pass
        try:
            extract(repo, {**spec, "git_blob": "0"*40})
            raise AssertionError("accepted mismatched Git blob")
        except RuntimeError:
            pass
    print(json.dumps(dict(status="passed", ratio_cases=5, archive_cases=6,
                          native="NOT_RUN", historical_object="NOT_FETCHED", device="NOT_RUN")))


if __name__ == "__main__":
    main()
