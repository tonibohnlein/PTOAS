# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Evaluate exported residue systems directly with Python integer arithmetic."""

def membership_index(document):
    grouped = {}
    for relation in document["relations"]:
        key = (relation["kind"], relation["source"], relation["target"],
               relation["source_event"], relation["target_event"])
        for piece in relation["pieces"]:
            if not piece["empty"]:
                grouped.setdefault((key, tuple(piece["residues"])), []).append(piece["rows"])
    period = document["period"]

    def contains(key, values):
        residues = tuple(x % period for x in values)
        quotients = tuple(x // period for x in values)
        for rows in grouped.get((key, residues), []):
            valid = True
            for row in rows:
                assert len(row["coefficients"]) == len(values)
                value = row["constant"] + sum(a * b for a, b in zip(row["coefficients"], quotients))
                if not (value == 0 if row["equality"] else value >= 0):
                    valid = False
                    break
            if valid:
                return True
        return False
    return contains

