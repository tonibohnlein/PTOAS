#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Accounting/partition tests only: synthetic records are not native evidence."""
import copy
import json
from s6_report import parse_reports, validate_report


def main():
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    source = dict(core="AIV", pipe="MTE2")
    target = dict(core="AIV", pipe="V")
    a = dict(action=0, anchor=0, kind="set", after=True, source_lane=source,
             target_lane=target, key=0, planner_order="0", participation="every",
             ordinal_distance="0", existence_residue="0", invocation=0,
             invocation_body_guard=False, invocation_residue="0")
    b = dict(a, action=1, anchor=1, planner_order="1")
    report = dict(schema="oahs.s6.plan.v1", producer="structured", new_hardware_elision=False,
                  retirement_sites=1, generated_static_sites_including_retirement="2",
                  units=[dict(unit=0, period="1", wrapper_count=0,
                              atoms=[dict(atom=0, original_phase=7, role="initial"),
                                     dict(atom=1, original_phase=7, role="steady")],
                              actions=[a, b], sites=[dict(site=0, actions=[0, 1])],
                              requirements=[], deletion_audit=[])])
    assert validate_report(report)["static_sites"] == 2
    assert parse_reports("other diagnostic\nOAHS_PLAN " + json.dumps(report))[0] == report
    bad = []
    for field, value in (("key", 1), ("ordinal_distance", "1"), ("participation", "last"),
                         ("invocation", 1)):
        x = copy.deepcopy(report);x["units"][0]["actions"][1][field] = value;bad.append(x)
    x = copy.deepcopy(report);x["units"][0]["atoms"][1]["original_phase"] = 8;bad.append(x)
    x = copy.deepcopy(report);x["units"][0]["sites"][0]["actions"] = [0, 0];bad.append(x)
    x = copy.deepcopy(report);x["units"][0]["wrapper_count"] = 1;bad.append(x)
    x = copy.deepcopy(report);x["generated_static_sites_including_retirement"] = "3";bad.append(x)
    x = copy.deepcopy(report);x["units"][0]["deletion_audit"] = [dict(completion_unproved_without=[99])];bad.append(x)
    for x in bad:
        try:
            validate_report(x)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid diagnostic accepted")
    for text in ("", "old trace only", 'OAHS_PLAN {"schema":"other"}'):
        try:
            parse_reports(text)
        except ValueError:
            pass
        else:
            raise AssertionError("missing or wrong-schema report accepted")
    print(json.dumps(dict(status="passed", positive_checks=2, negative_checks=len(bad)+3,
                          native="NOT_RUN", device="NOT_RUN")))


if __name__ == "__main__":
    main()
