# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Native scalar protection needs no address or loop dependence solver."""
from pathlib import Path
import re
import shutil
import subprocess
import sys
from check_frontier_public_pipeline import unchanged_failure, without_evidence


def run(tool, source, option=None):
    arguments = [tool, "--mlir-disable-threading"] + ([option] if option else [])
    return subprocess.run(arguments, input=source,
                          text=True, capture_output=True, timeout=30)


def original_body(tool, source):
    canonical = run(tool, without_evidence(source))
    assert canonical.returncode == 0, canonical.stderr
    return canonical.stdout


def check_preserved_scalar_barrier(tool, source):
    # Existing barriers are preserved metadata. Scalar protection discharges
    # same-PIPE_S storage hazards without requiring generated event commands.
    # Do not invent values for the original data-dependent scalar loads: exact
    # canonical SSA/control equality preserves every possible original trace.
    assert source.count("    return") == 1
    barrier = source.replace("    return", "    pto.barrier #pto.pipe<PIPE_S>\n    return")
    logical = run(tool, barrier, "--pto-frontier-analysis")
    assert logical.returncode == 0, logical.stderr
    assert not re.search(r"pto\.(?:logical_set|logical_wait|set_flag|wait_flag)", logical.stdout)
    assert logical.stdout.count("pto.barrier <PIPE_S>") == 1
    assert logical.stdout.count("pto.barrier <PIPE_ALL>") == 1
    # Remove only the new final drain. Reparse both complete modules so printed
    # SSA names and empty report-attribute dictionaries compare canonically.
    restored, count = re.subn(r"(?m)^    pto\.barrier <PIPE_ALL>\n(?=    return\b)", "", logical.stdout)
    assert count == 1, "new drain was not at the original function exit"
    assert original_body(tool, restored) == original_body(tool, barrier), "original scalar program changed"
    physical = run(tool, logical.stdout, "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5")
    assert physical.returncode == 0, physical.stderr
    assert original_body(tool, physical.stdout) == original_body(tool, logical.stdout)
    # An external effectful call has neither closed matching nor a completed
    # exit certificate. Retain this real unsupported-effect/atomicity control.
    declaration = "  func.func private @unmodeled(!pto.ptr<i32, gm>)\n"
    unsupported = barrier.replace("  func.func @", declaration + "  func.func @", 1)
    unsupported = unsupported.replace("    return",
                                      "    func.call @unmodeled(%arg0) : (!pto.ptr<i32, gm>) -> ()\n    return")
    refusal = unchanged_failure(tool, unsupported, ["--pto-frontier-analysis"])
    assert "error:" in refusal.stderr


def main():
    tool = shutil.which(sys.argv[1])
    assert tool, "missing optimizer executable"
    tool = str(Path(tool).resolve(strict=True))
    samples = Path(sys.argv[2]).read_text().split("// -----")
    for source in samples:
        logical = run(tool, source, "--pto-frontier-analysis")
        assert logical.returncode == 0, logical.stderr
        assert "pto.logical_set" not in logical.stdout
        assert "pto.logical_wait" not in logical.stdout
        assert "<PIPE_S>" not in logical.stdout
        assert logical.stdout.count("pto.barrier <PIPE_ALL>") == 1
        for op in ("pto.load", "pto.store", "scf.for", "scf.if"):
            assert source.count(op) == logical.stdout.count(op), op
        physical = run(tool, logical.stdout,
                       "--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5")
        assert physical.returncode == 0, physical.stderr
        assert not re.search(r"pto.(?:set_flag|wait_flag)", physical.stdout)
    check_preserved_scalar_barrier(tool, samples[0])
    print("native scalar corpus: three structured programs allocate with zero event IDs")


if __name__ == "__main__":
    main()
