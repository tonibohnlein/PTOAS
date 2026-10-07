# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check complete scalar exports and absence of partial exports on rejection."""
import itertools
import json
import shutil
import subprocess
import sys

from arithmetic_membership import membership_index

executable = shutil.which(sys.argv[1])
if executable is None:
    raise RuntimeError("arithmetic test executable unavailable")
result = subprocess.run([executable, "--arithmetic", sys.argv[2]],
                        check=True, capture_output=True, text=True, timeout=30)
documents = [json.loads(line.removeprefix("arithmetic-json ")) for line in result.stdout.splitlines()
             if line.startswith("arithmetic-json ")]
if len(documents) != 7:
    raise RuntimeError("expected all seven arithmetic fixtures")
accepted = {"nested_reset": (3, 2), "triangular": (2, 1), "siblings": (4, 2), "empty_minimum": (1, 0),
            "ssa_prerequisite": (2, 0), "opaque_bound": (1, 2), "existing_sync": (0, 0)}
for document in documents:
    name = document["function"]
    if name in accepted:
        sites, parameters = accepted[name]
        if (len(document["sites"]) != sites or len(document["parameters"]) != parameters
                or not document["relations"]):
            raise RuntimeError(name + ": incomplete arithmetic exports")
    elif document["relations"] or document["sites"] or document["parameters"]:
        raise RuntimeError(name + ": partial exports on rejection")

def explicit_trace(name, bounds):
    """Independent source-order executions of the accepted fixture programs."""
    if name == "nested_reset":
        n, k = bounds
        return [(site, coords) for t in range(n)
                for site, coords in ([(site, (t, i)) for i in range(k) for site in (0, 1)] + [(2, (t,))])]
    if name == "triangular":
        return [(site, coords) for t in range(bounds[0])
                for site, coords in ([(0, (t,))] + [(1, (t, i)) for i in range(t)])]
    if name == "siblings":
        return [(0, ())] + [(1, (i,)) for i in range(bounds[0])] + [
            (2, (i,)) for i in range(bounds[1])] + [(3, ())]
    if name == "opaque_bound":
        # The once-evaluated machine expression is an entry parameter; its
        # value is not assumed to equal mathematical n+1 after overflow.
        return [(0, (i,)) for i in range(bounds[1])]
    if name == "ssa_prerequisite":
        return [(0, ()), (1, ())]
    return []  # empty_minimum has a negative upper bound.





checks = 0
for document in documents:
    name = document["function"]
    if name not in accepted:
        continue
    contains = membership_index(document)
    candidates = [(site, coords) for site, data in enumerate(document["sites"])
                  for coords in itertools.product(range(-1, 4), repeat=data["depth"])]
    for bounds in itertools.product(range(-1, 4), repeat=len(document["parameters"])):
        trace = explicit_trace(name, bounds)
        ranks = {occurrence: rank for rank, occurrence in enumerate(trace)}
        for a in candidates:
            site, coords = a
            actual = contains((1, site, -1, 0, 0), coords + bounds)
            assert actual == (a in ranks), (name, "occurrence", a, bounds)
            checks += 1
            base = 1024 * (coords[-1] % 2) if name == "nested_reset" and site < 2 else 0
            writes = site == 1 if name == "ssa_prerequisite" else (site == 0 or (name == "siblings" and site == 2))
            for kind in (4, 5):
                for byte in (-1, 0, 1, 3, 4, 1023, 1024, 1025, 1027, 1028):
                    expected = a in ranks and (kind == 5) == writes and base <= byte < base + 4
                    actual = contains((kind, site, -1, 0, 0), coords + (byte,) + bounds)
                    assert actual == expected, (name, "access", a, bounds, kind, byte)
                    checks += 1
            for b in candidates:
                before = a in ranks and b in ranks and ranks[a] < ranks[b]
                values = coords + b[1] + bounds
                assert contains((2, site, b[0], 0, 0), values) == before, (name, "order", a, b, bounds)
                checks += 1
                same_pipe = document["sites"][site]["pipe"] == document["sites"][b[0]]["pipe"]
                for source, target in itertools.product((1, 2), repeat=2):
                    expected = same_pipe and ((before and (source, target) != (2, 1)) or
                                             (a == b and a in ranks and (source, target) == (1, 2)))
                    if name == "ssa_prerequisite" and a == (0, ()) and b == (1, ()) and (source, target) == (2, 1):
                        expected = True
                    actual = contains((3, site, b[0], source, target), values)
                    assert actual == expected, (name, "native", a, b, bounds, source, target)
                    checks += 1
print("arithmetic exports: 7 accepted; finite semantic checks:", checks)
