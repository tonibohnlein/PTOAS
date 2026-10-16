# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Cross-check the membership oracle, including machine rows and locals."""
import itertools
import random
from arithmetic_membership import integer_membership, membership_index

rng = random.Random(1183)
for _ in range(1000):
    # Bounded locals permit complete enumeration by an independent flat oracle.
    rows = [((0, 1, 0), 3), ((0, -1, 0), 2), ((0, 0, 1), 2), ((0, 0, -1), 3)]
    rows += [(tuple(rng.randrange(-3, 4) for _ in range(3)), rng.randrange(-5, 6)) for _ in range(3)]
    congruences = [(tuple(rng.randrange(-3, 4) for _ in range(3)), rng.randrange(1, 7), rng.randrange(-4, 5))]
    for x in range(-3, 4):
        expected = any(all(sum(a*b for a, b in zip(c, (x, y, z))) <= bound for c, bound in rows)
                       and all((sum(a*b for a, b in zip(c, (x, y, z))) - residue) % modulus == 0
                               for c, modulus, residue in congruences)
                       for y, z in itertools.product(range(-2, 4), range(-3, 3)))
        assert integer_membership(rows, congruences, (x,)) == expected

key = (1, 0, -1, 0, 0)
wide = 1 << 100
piece = {"empty": False, "residues": [0], "rows": [],
         "machine_rows": [{"coefficients": ["-1"], "bound": str(-wide)},
                          {"coefficients": ["1"], "bound": str(wide + 20)}],
         "congruences": [{"coefficients": ["1"], "modulus": "7", "residue": "-2"}]}
relation = dict(zip(("kind", "source", "target", "source_event", "target_event"), key), pieces=[piece])
contains = membership_index({"period": 1, "relations": [relation]})
for x in range(wide - 2, wide + 23):
    assert contains(key, (x,)) == (wide <= x <= wide + 20 and x % 7 == 5)
for rows in ([((1, 1), 0)], [((1, 1), 0), ((1,), 1)]):
    try:
        integer_membership(rows, [], (0,))
    except ValueError:
        pass
    else:
        raise AssertionError("unresolved locals/schema must not become a membership answer")
print("integer membership: 7000 bounded systems, wide machine rows, congruences and explicit refusals")
