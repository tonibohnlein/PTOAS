# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Plan failures retain original IR; source drains are not input admission certificates."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile


def run(tool, arguments):
    return subprocess.run([tool, "--mlir-disable-threading", *arguments],
                          capture_output=True, text=True, check=False, timeout=120)


def reject(tool, directory, name, source, family, expected):
    path = Path(directory) / (name + ".pto")
    path.write_text(source, encoding="utf-8")
    parsed = run(tool, [str(path), "-o", "-"])
    assert parsed.returncode == 0, name + ": invalid original fixture\n" + parsed.stderr
    option = "--pto-insert-sync=algorithm=frontier-synch " + \
        "frontier-repair-family=" + family + \
        " report-frontier-costs=true"
    failed = run(tool, ["--mlir-print-ir-after-failure", "--mlir-print-ir-module-scope",
                        option, str(path), "-o", "-"])
    assert failed.returncode != 0, name + ": unsupported plan admitted"
    assert expected in failed.stderr, name + ": wrong failure stage\n" + failed.stderr
    reports = [json.loads(line) for line in failed.stderr.splitlines() if line.startswith("{")]
    costs = [item for item in reports if item.get("report") == "frontier-costs-v1"]
    assert len(costs) == 1, name + ": missing structured failure report"
    if name != "fixed-capacity":
        # A genuine earlier fixed-allocation counterexample stays in the
        # ordered ledger; the final construction failure must be distinct.
        assert "requires 8 IDs" not in costs[0]["reason"], name + ": qualification mislabeled scarcity"
    assert costs[0]["failure_category"] == "NoCertifiedPlan"
    if name == "fixed-capacity":
        attempts = costs[0]["attempts"]
        assert any(item["route"] == "physical/fixed-handoffs" and
                   item["outcome"] == "CapacityInsufficient" for item in attempts)
        assert [item["outcome"] for item in attempts if item["route"].startswith("physical/supplied-")] == \
            ["NotSupplied", "NotSupplied"]
    marker = failed.stderr.index("// -----// IR Dump After")
    dumped = failed.stderr[marker:]
    dumped = dumped[dumped.index("module attributes"):].strip()
    assert dumped == parsed.stdout.strip(), name + ": failed realization changed original IR"


def main(tool, one_way_path, upper_path):
    one_way = Path(one_way_path).read_text(encoding="utf-8")
    upper = Path(upper_path).read_text(encoding="utf-8")
    no_drain = one_way.replace("    pto.barrier <PIPE_MTE2>\n", "", 1)
    local = one_way.replace(
        "    pto.txor ins(%data1,", "    pto.barrier <PIPE_V>\n    pto.txor ins(%data1,", 1)
    nested = upper.replace(
        "    scf.for %i = %zero to %n step %one {",
        "    scf.for %outer = %zero to %n step %one {\n"
        "    scf.for %i = %zero to %n step %one {", 1)
    nested = nested.replace("    }\n    return", "    }\n    }\n    return", 1)
    with tempfile.TemporaryDirectory(prefix="frontier-repair-failures-") as directory:
        reject(tool, directory, "fixed-capacity", one_way, "fixed-only", "requires 8 IDs; available 6")
        path = Path(directory) / "missing-source-drain.pto"
        path.write_text(no_drain, encoding="utf-8")
        options = "--pto-insert-sync=algorithm=frontier-synch " + \
            "" + \
            "frontier-repair-family=finite-one-way report-frontier-costs=true"
        admitted = run(tool, [options, str(path), "-o", "-"])
        assert "qualification before demand reduction" not in admitted.stderr, admitted.stderr
        reports = [json.loads(line) for line in admitted.stderr.splitlines() if line.startswith("{")]
        assert any(r.get("report") == "frontier-costs-v1" and r["analysis"] != "not-attempted"
                   for r in reports), "source drain absence vetoed shared analysis"
        reject(tool, directory, "variable-target-local", local, "finite-one-way", "variable local readiness")
        reject(tool, directory, "nested-upper", nested, "fixed-only", "physical")
    print("three transactional plan failures and source-drain-independent shared admission")


if __name__ == "__main__":
    main(*sys.argv[1:])
