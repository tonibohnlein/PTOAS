# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare root-relative arithmetic primitives against independent executions."""
import itertools
import json
import shutil
import subprocess
import sys

from arithmetic_membership import membership_index


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_region(document):
    name = document["region"]
    if name in ("local-nonlinear", "local-loaded-bound"):
        require(not document["sites"] and not document["parameters"] and not document["relations"],
                "local nonlinear access was accepted or exported partially")
        return 0
    expected_kinds = {"outer-binding": ["enclosing"], "entry-value": ["enclosing", "entry-value", "function"],
                      "loaded-bound": ["entry-value"], "root-conditional": ["function", "enclosing"]}
    require(document["parameter_kinds"] == expected_kinds[name], name + ": incorrect context bindings")
    sites = 2 if name == "outer-binding" else 1
    depth = 0 if name == "root-conditional" else 1
    require(len(document["sites"]) == sites and all(x["depth"] == depth for x in document["sites"]),
            name + ": enclosing loop leaked into occurrence coordinates")
    if name == "loaded-bound":
        require(document["incoming_prerequisites"] > 0, "lost incoming scalar completion prerequisite")
    contains = membership_index(document)
    checks = 0
    for context in range(-1, 5):
        for flag in ((0, 1) if name in ("entry-value", "root-conditional") else (1,)):
            parameters = (context, context * context, flag) if name == "entry-value" else (context,)
            if name == "root-conditional":
                parameters = (flag, context)
            iterations = range(1, context + 1, 2) if name == "outer-binding" else range(context)
            if name == "entry-value":
                iterations = range(context * context)
            trace = [(site, (j,)) for j in iterations for site in range(sites)] if flag else []
            if name == "root-conditional":
                trace = [(0, ())] if flag else []
            ranks = {occurrence: i for i, occurrence in enumerate(trace)}
            candidates = [(site, (j,)) for j in range(-1, 6) for site in range(sites)]
            if name == "root-conditional":
                candidates = [(0, ())]
            for a in candidates:
                site, coords = a
                require(contains((1, site, -1, 0, 0), coords + parameters) == (a in ranks),
                        name + ": occurrence domain mismatch")
                checks += 1
                address = 4 * (coords[0] + context if name == "outer-binding" else (coords[0] if coords else context))
                if name == "root-conditional":
                    address = 4 * context
                for byte in range(-4, 40):
                    for kind in (4, 5):
                        writes = name == "outer-binding" and site == 0
                        expected = a in ranks and (kind == 5) == writes and address <= byte < address + 4
                        require(contains((kind, site, -1, 0, 0), coords + (byte,) + parameters) == expected,
                                name + ": physical access mismatch")
                        checks += 1
                for b in candidates:
                    before = a in ranks and b in ranks and ranks[a] < ranks[b]
                    values = coords + b[1] + parameters
                    require(contains((2, site, b[0], 0, 0), values) == before, name + ": order mismatch")
                    for source, target in itertools.product((1, 2), repeat=2):
                        expected = ((before and (source, target) != (2, 1)) or
                                    (a == b and a in ranks and (source, target) == (1, 2)))
                        require(contains((3, site, b[0], source, target), values) == expected,
                                name + ": native-order mismatch")
                        checks += 1
    return checks


def main():
    executable = shutil.which(sys.argv[1])
    require(executable is not None, "test executable unavailable")
    result = subprocess.run([executable, "--arithmetic", sys.argv[2]],
                            check=True, capture_output=True, text=True)
    documents = [json.loads(line.removeprefix("arithmetic-json ")) for line in result.stdout.splitlines()
                 if line.startswith("arithmetic-json ")]
    regions = [doc for doc in documents if "region" in doc]
    require(len(regions) == 6, "missing regional extraction fixtures")
    checks = sum(check_region(document) for document in regions)
    print("regional arithmetic: 4 accepted, 2 rejected; independent checks:", checks)


if __name__ == "__main__":
    main()
