# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Capture the host compiler artifacts actually used by a configured-tree run."""

import hashlib
import os
import platform
import re
import subprocess
from pathlib import Path


SOURCE_ROOTS = ("include", "lib", "tools", "test/experiments", "test/lit")


def untracked_sources(repo):
    """Include new compiler/test sources without traversing unrelated checkouts."""
    output = checked_output(["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard",
                             "-z", "--", *SOURCE_ROOTS])
    result = {}
    for name in filter(None, output.split("\0")):
        path = repo / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(repo.resolve()):
            raise ValueError(f"untracked source is not a regular in-repository file: {name}")
        result[name] = sha256(path)
    return result


def snapshot_untracked(repo, destination, fingerprints):
    """Retain actual new source contents, not just hashes of unavailable files."""
    for name, fingerprint in fingerprints.items():
        source = repo / name
        if sha256(source) != fingerprint:
            raise ValueError(f"untracked source changed while snapshotting: {name}")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def sha256(path):
    """Hash a file with bounded memory use."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_output(command):
    """Run a bounded read-only provenance query without a command interpreter."""
    return subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout.strip()


def artifact(path):
    """Record both a logical path and the resolved contents of a build artifact."""
    return {"path": str(path), "resolved_path": str(path.resolve(strict=True)), "sha256": sha256(path)}


def build_metadata(build_root, launcher):
    """Fail if the configured compiler/toolchain cannot be identified exactly."""
    cache_path = build_root / "CMakeCache.txt"
    cache = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"(\w+):[^=]+=(.*)", line)
        if match:
            cache[match[1]] = match[2]
    llvm_config = Path(cache["LLVM_DIR"]) / "LLVMConfig.cmake"
    llvm_text = llvm_config.read_text(encoding="utf-8")
    llvm_source_match = re.search(r'set\(LLVM_BUILD_MAIN_SRC_DIR "([^"\n]+)"\)', llvm_text)
    llvm_version_match = re.search(r"set\(LLVM_PACKAGE_VERSION ([^\s)]+)\)", llvm_text)
    if not llvm_source_match or not llvm_version_match:
        raise ValueError("LLVM source/version metadata missing from configured build")
    llvm_source = Path(llvm_source_match[1])
    llvm_diff = checked_output(["git", "-C", str(llvm_source), "diff", "HEAD", "--"])
    compiler_library = build_root / "python/ptoas/mlir/_mlir_libs/libPTOASCompiler.so"
    return {
        "build_root": str(build_root), "cmake_cache": artifact(cache_path),
        "launcher": artifact(launcher), "compiler_library": artifact(compiler_library),
        "llvm_version": llvm_version_match[1],
        "llvm_source_commit": checked_output(["git", "-C", str(llvm_source), "rev-parse", "HEAD"]),
        "llvm_tracked_clean": not llvm_diff,
        "llvm_tracked_patch": llvm_diff,
        "llvm_tracked_diff_sha256": hashlib.sha256(llvm_diff.encode()).hexdigest(),
        "python_extensions": [artifact(path) for path in sorted((build_root / "python/ptoas").glob("_core*.so"))],
        "llvm_config": artifact(llvm_config),
        "mlir_config": artifact(Path(cache["MLIR_DIR"]) / "MLIRConfig.cmake"),
        "cxx_version": checked_output([cache["CMAKE_CXX_COMPILER"], "--version"]),
        "cmake_version": checked_output([cache["CMAKE_COMMAND"], "--version"]),
        "platform": platform.platform(),
        "environment": {key: os.environ.get(key) for key in (
            "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "LLVM_BUILD_DIR", "ASCEND_HOME_PATH", "CANN_INSTALL_DIR")},
        "hardware_evidence": "none: host compiler only; CANN/PTO-ISA runtimes not exercised",
    }
