# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Prepare inputs and check the unfolded closure; lit owns compiler execution."""
import json
from pathlib import Path
import re
import sys
from check_mixed_crossing_families import check


def main():
    mode, *args = sys.argv[1:]
    if mode == "input":
        fixture, guard, destination = args
        assert guard in ("0", "1")
        source = Path(fixture).read_text().replace("attributes {test.trace_arguments",
            "attributes {test.allocation_candidates, test.trace_arguments")
        Path(destination).write_text(source.replace("array<i64: 1>", f"array<i64: {guard}>"))
    elif mode == "trace":
        source, destination = args
        logical, count = re.subn(r"(func\.func\s+@\w+\([^\n]*?\)\s*)attributes\s*\{",
            r"\1attributes {test.trace_existing, ", Path(source).read_text(), count=1)
        assert count == 1
        Path(destination).write_text(logical)
    elif mode == "check":
        source, guard = args
        check(json.loads(Path(source).read_text()), 4, 4, int(guard))
        print("regional allocation candidate: unfolded matching and payload closure passed")
    else:
        raise ValueError("expected input, trace or check mode")


if __name__ == "__main__":
    main()
