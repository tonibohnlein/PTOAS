# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Check the maintained parser against the actual adversarial compiler dump."""

import argparse
from pathlib import Path

from census import summarize
from records import parse_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    args = parser.parse_args()
    row = dict(parse_diagnostics(args.diagnostic.read_text(encoding="utf-8")),
               case_id="adversarial", source="protocol_sync_storage_adversarial.pto", return_code=0)
    summary = summarize([row], "storage")
    if summary["totals"]["missing_statistics_rows"] or not summary["totals"]["transitions"]:
        raise ValueError("adversarial fixture has no complete storage evidence")
    for function in row["functions"]:
        if not function["regions"] or "op" not in function["regions"][0]:
            raise ValueError("missing region-operation provenance")
        for family in function["families"]:
            if "root-lexical-scope" not in family:
                raise ValueError("missing root-scope provenance")
    if not summary["multi_family_records"]:
        raise ValueError("missing adversarial multi-family records")


if __name__ == "__main__":
    main()
