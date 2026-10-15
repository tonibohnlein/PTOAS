# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check process isolation, outcome classification and streamed inventories."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile


def main():
    source = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("control_campaign", source)
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    with tempfile.TemporaryDirectory(prefix="control-campaign-") as scratch:
        root = Path(scratch)
        code = ("import os,resource; print(sorted(os.sched_getaffinity(0))); "
                "print(resource.getrlimit(resource.RLIMIT_AS))")
        result = campaign.run_stage(root, "bounds", [sys.executable, "-c", code], 5, 128)
        assert result["success"] and result["outcome_kind"] == "success", result
        lines = (root / "bounds.out").read_text().splitlines()
        assert json.loads(lines[0]) == result["cpu_affinity"], (lines, result)
        assert lines[1] == str((128 * 1024 * 1024,) * 2), lines
        result = campaign.run_stage(root, "nonzero", [sys.executable, "-c",
            "import sys; [print('error: expected',file=sys.stderr) for _ in range(200)]; sys.exit(7)"], 5, 128)
        assert result["outcome_kind"] == "nonzero_exit" and result["returncode"] == 7, result
        assert len(result["diagnostics"]) == 4 and not result["success"], result
        result = campaign.run_stage(root, "signal", [sys.executable, "-c",
            "import os,signal; os.kill(os.getpid(), signal.SIGABRT)"], 5, 128)
        assert result["outcome_kind"] == "signal" and result["returncode"] == -signal.SIGABRT, result
        result = campaign.run_stage(root, "timeout", [sys.executable, "-c", "while True: pass"], .1, 128)
        assert result["outcome_kind"] == "timeout" and result["returncode"] < 0, result
        assert json.loads((root / "timeout.json").read_text()) == result
        fixture = root / "inventory.pto"
        fixture.write_text("%v = arith.select %x, %y, %z : i32\npto.logical_set\npto.logical_wait\n")
        inventory = campaign.metrics(fixture)
        assert inventory["arithmetic"] == inventory["selects"] == inventory["sets"] == inventory["waits"] == 1
    print("control campaign: process limits, timeout/signal/nonzero separation and streamed inventories passed")


if __name__ == "__main__":
    main()
