# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent integer membership for exported rows and machine constraints.

Existential locals are evaluated only over bounds proved by supplied unary
constraints. An unsupported/unbounded coupled system raises, never guesses a
membership answer or silently ignores its locals.
"""
import itertools
import math


def integer_membership(rows, congruences, values):
    dimensions = {len(coefficients) for coefficients, _ in rows}
    dimensions.update(len(coefficients) for coefficients, _, _ in congruences)
    if len(dimensions) > 1 or (dimensions and min(dimensions) < len(values)):
        raise ValueError("inconsistent exported coordinate dimensions")
    locals_count = next(iter(dimensions), len(values)) - len(values)
    residual = [(coefficients[len(values):], bound - sum(a * b for a, b in zip(coefficients, values)))
                for coefficients, bound in rows]
    modular = [(coefficients[len(values):], modulus,
                residue - sum(a * b for a, b in zip(coefficients, values)))
               for coefficients, modulus, residue in congruences]
    lower, upper = [None] * locals_count, [None] * locals_count
    for coefficients, bound in residual:
        used = [(i, a) for i, a in enumerate(coefficients) if a]
        if not used and bound < 0:
            return False
        if len(used) == 1:
            i, a = used[0]
            if a > 0:
                hi = bound // a
                upper[i] = hi if upper[i] is None else min(hi, upper[i])
            else:
                lo = -(bound // -a)
                lower[i] = lo if lower[i] is None else max(lo, lower[i])
    ranges = []
    for i in range(locals_count):
        used = any(c[i] for c, _ in residual) or any(c[i] for c, _, _ in modular)
        if not used:
            ranges.append(range(1))
        elif lower[i] is not None and upper[i] is not None:
            if lower[i] > upper[i]:
                return False
            ranges.append(range(lower[i], upper[i] + 1))
        else:
            raise ValueError("oracle requires certified finite bounds for existential locals")
    if math.prod(len(r) for r in ranges) > 4096:
        raise ValueError("oracle local enumeration exceeds its test-work bound")
    for assignment in itertools.product(*ranges):
        if (all(sum(a * b for a, b in zip(c, assignment)) <= bound for c, bound in residual)
                and all(sum(a * b for a, b in zip(c, assignment)) % modulus == residue % modulus
                        for c, modulus, residue in modular)):
            return True
    return False


def membership_index(document):
    grouped = {}
    period = document["period"]
    if period <= 0:
        raise ValueError("nonpositive exported period")
    for relation in document["relations"]:
        key = (relation["kind"], relation["source"], relation["target"],
               relation["source_event"], relation["target_event"])
        for piece in relation["pieces"]:
            if piece["empty"]:
                continue
            rows = []
            for row in piece["rows"]:
                coefficients = tuple(int(a) for a in row["coefficients"])
                rows.append((tuple(-a for a in coefficients), int(row["constant"])))
                if row["equality"]:
                    rows.append((coefficients, -int(row["constant"])))
            machine = piece.get("machine_rows", [])
            congruences = piece.get("congruences", [])
            if (machine or congruences) and period != 1:
                raise ValueError("machine constraints require original coordinates at period one")
            rows.extend((tuple(int(a) for a in row["coefficients"]), int(row["bound"])) for row in machine)
            modular = [(tuple(int(a) for a in row["coefficients"]), int(row["modulus"]), int(row["residue"]))
                       for row in congruences]
            if any(modulus <= 0 for _, modulus, _ in modular):
                raise ValueError("nonpositive congruence modulus")
            grouped.setdefault((key, tuple(piece["residues"])), []).append((rows, modular))

    def contains(key, values):
        residues = tuple(x % period for x in values)
        quotients = tuple(x // period for x in values)
        return any(integer_membership(rows, modular, quotients)
                   for rows, modular in grouped.get((key, residues), []))
    return contains
