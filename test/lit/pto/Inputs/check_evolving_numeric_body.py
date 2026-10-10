# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check numeric phase bodies with exact evolving GM maps and refused approximations."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from check_composed_phases import check


def invoke(tool, policy, mode, path):
    """Run one compiler probe serially; the watchdog is not an admission limit."""
    result = subprocess.run([tool, f"--gm-alias={policy}", mode, str(path)],
                            capture_output=True, text=True, check=False, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return result.stdout


def require(condition, message, report):
    """Keep regression checks active even when Python assertions are disabled."""
    if not condition:
        raise RuntimeError(f"{message}: {report}")


def quotient_origin(source):
    """Equivalent physical translation with an otherwise unneeded SSA quotient."""
    return source.replace(
        "      %column = arith.muli %visit, %owned_stride overflow<nsw> : index",
        "      %twice = arith.muli %visit, %two overflow<nsw> : index\n"
        "      %quotient = arith.divui %twice, %two : index\n"
        "      %column = arith.muli %quotient, %owned_stride overflow<nsw> : index")


def moving_local(source):
    """Keep the bank period but add a genuinely moving local destination."""
    source = source.replace("      %value = pto.tgetval",
        "      %drift_stride = arith.constant 65536 : index\n"
        "      %drift = arith.muli %visit, %drift_stride overflow<nsw> : index\n"
        "      %drift64 = arith.index_cast %drift : index to i64\n"
        "      %moving_address = arith.addi %base, %drift64 : i64\n"
        "      %moving = pto.alloc_tile addr = %moving_address : !cell\n"
        "      %value = pto.tgetval")
    return source.replace(
        'outs(%selected : !cell) {test.label = "compute"}',
        'outs(%moving : !cell) {test.label = "compute"}')


def unequal_startup(source):
    """Use two finite startup visits and one steady visit without modulo control."""
    start = source.index("      scf.for %i = %zero to %inner_trips")
    end = source.index("      pto.tsetval", start)
    body = source[start:end]
    steady = body.replace("to %inner_trips", "to %one")
    branch = ("      %first_visit = arith.cmpi eq, %visit, %zero : index\n"
              "      scf.if %first_visit {\n" + body + "      } else {\n" + steady + "      }\n")
    return source[:start] + branch + source[end:]


def check_positive(tool, policy, path, source):
    """Require phased numeric selection and check each phase's actual word."""
    path.write_text(source, encoding="utf-8")
    report = json.loads(invoke(tool, policy, "--sequence-analysis", path))
    selected = (not report["error"] and report["prepared"] and report["unchanged"]
                and report["queries_available"] and report["phase_descriptions"] > 0
                and report["numeric_visits"] > 0)
    require(selected, "numeric evolving-body adapter was not consumed", report)
    invoke(tool, policy, "--insert-logical-library", path)
    for outer, inner in ((0, 0), (1, 0), (1, 1), (2, 2), (3, 1), (5, 2), (3, 3)):
        case = source.replace("array<i64: 3, 2, 65536>", f"array<i64: {outer}, {inner}, 65536>")
        case = case.replace("%inner_trips = arith.constant 2 : index",
                            f"%inner_trips = arith.constant {inner} : index")
        path.write_text(case, encoding="utf-8")
        document = json.loads(invoke(tool, policy, "--structured-trace", path))
        check(document, outer, inner, owned_stride=32768, owned_consumers=False, read_square=True)


def main():
    tool_name, fixture_name = sys.argv[1:]
    tool = shutil.which(tool_name)
    if tool is None:
        raise RuntimeError(f"compiler probe not found: {tool_name}")
    source = Path(fixture_name).resolve(strict=True).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="evolving-numeric-body-") as scratch:
        path = Path(scratch) / "case.pto"
        for policy in ("may-not-alias", "may-alias"):
            for positive in (source, quotient_origin(source)):
                check_positive(tool, policy, path, positive)
            overlap = source.replace("%owned_stride = arith.constant 8192 : index",
                                     "%owned_stride = arith.constant 4 : index")
            for negative in (overlap, moving_local(source)):
                path.write_text(negative, encoding="utf-8")
                report = json.loads(invoke(tool, policy, "--sequence-analysis", path))
                require(report["unchanged"] and report["phase_descriptions"] == 0,
                        "uncertified evolving map was frozen or omitted", report)
            for spelling in (source, quotient_origin(source)):
                path.write_text(unequal_startup(spelling), encoding="utf-8")
                report = json.loads(invoke(tool, policy, "--sequence-analysis", path))
                rejected = (report["unchanged"] and not report["prepared"] and not report["emitted"]
                            and report["phase_descriptions"] == 0
                            and "symbolic storage crossings" in report["error"])
                require(rejected, "unequal startup storage adapter obligation was hidden", report)
    print("evolving numeric body: 28 exact byte closures, consumed fallback, policy isolation, "
          "overlap/local-motion refusals, explicit unequal-startup obligation")


if __name__ == "__main__":
    main()
