# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Both algorithms consume identical shared records without native admission."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    require(len(sys.argv) == 3, "usage: check_shared_admission.py compiler input")
    tool = shutil.which(sys.argv[1])
    require(tool is not None, "compiler missing")
    source = Path(sys.argv[2]).resolve(strict=True)
    common = [tool, "--mlir-disable-threading", "--pto-insert-sync-debug=2"]
    baseline = subprocess.run(common + ["--pto-insert-sync=algorithm=existing", str(source), "-o", "-"],
                              capture_output=True, text=True, timeout=120, check=False)
    require(baseline.returncode == 0, baseline.stderr)
    expected = re.findall(r"^SharedTarget .*$", baseline.stderr, re.MULTILINE)
    require(len(expected) == 1, "expected shared target records")
    parsed = subprocess.run(common + [str(source), "-o", "-"], capture_output=True, text=True,
                            timeout=120, check=False)
    require(parsed.returncode == 0, parsed.stderr)
    outcomes = []
    for flags in ["", " report-frontier-costs=true", " dump-frontier-demands=true",
                  " dump-frontier-demands=true report-frontier-costs=true"]:
        result = subprocess.run(common + ["--mlir-print-ir-after-failure", "--mlir-print-ir-module-scope",
                                "--pto-insert-sync=algorithm=frontier-synch" + flags, str(source), "-o", "-"],
                                capture_output=True, text=True, timeout=120, check=False)
        require("qualification before demand reduction" not in result.stderr, result.stderr)
        require(re.findall(r"^SharedTarget .*$", result.stderr, re.MULTILINE) == expected,
                "algorithms consumed different shared records")
        reports = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
        costs = [r for r in reports if r.get("report") == "frontier-costs-v1"]
        if "report-frontier-costs" in flags:
            require(len(costs) == 1, "missing analysis report")
            require(costs[0]["analysis"] != "not-attempted", "shared input was vetoed before analysis")
            require(costs[0]["stages"]["effect_recovery"]["invocations"] >= 2,
                    "shared input did not reach trace analysis")
            require(costs[0].get("qualification") != "unmet-native-order", "native admission remains")
        else:
            require(not costs, "cost report emitted without its option")
        if result.returncode == 0:
            outcomes.append((0, result.stdout))
        else:
            marker = "// -----// IR Dump After"
            require(marker in result.stderr, "missing failed-plan IR")
            dumped = result.stderr[result.stderr.index(marker):]
            start = dumped.find("module attributes")
            require(start >= 0, "missing rejected module")
            require(dumped[start:].strip() == parsed.stdout.strip(), "failed plan changed original IR")
            outcomes.append((result.returncode, dumped[start:].strip()))
    require(all(value == outcomes[0] for value in outcomes), "diagnostic flags changed pass behavior")
    print("verified shared-input admission, record parity, diagnostic invariance and transactional realization")


if __name__ == "__main__":
    main()
