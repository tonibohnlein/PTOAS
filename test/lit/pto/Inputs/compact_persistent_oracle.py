# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent supplied-layout model; does not consume PTOAS generators or cells."""

from collections import defaultdict
from check_periodic_demands import closure, native

PIPES = (4, 4, 4, 4, 3, 3, 2, 2, 10)


def merge(intervals):
    result = []
    for begin, end in sorted(intervals):
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((begin, end))
    return result


def effect(space, base, mode, intervals):
    return {"space": space, "base": base, "mode": mode, "intervals": merge(intervals)}


def effects(kind, coordinates, core):
    wave = coordinates[-1] if kind in (0, 1, 8) else coordinates[-2] if kind in (2, 3) else coordinates[-3]
    k = 0 if kind in (0, 1, 8) else coordinates[-1] + 1 if kind in (2, 3) else coordinates[-2]
    kk = coordinates[-1] if kind in (4, 5, 6, 7) else 0
    task = 20 * wave + core
    row, col = (task // 4) * 128, (task % 4) * 256
    a_slot, b_slot = (k % 2) * 65536, 131072 + (k % 2) * 131072
    left, right = (kk % 2) * 16384, (kk % 2) * 32768
    local = lambda space, mode, start, size: effect(space, -1, mode, [(start, start + size)])
    if kind in (0, 2):
        gm = [((row + r) * 16384 + k * 512, (row + r) * 16384 + (k + 1) * 512) for r in range(128)]
        return [effect(1, 0, "read", gm), local(2, "write", a_slot, 65536)]
    if kind in (1, 3):
        gm = [((k * 256 + r) * 2048 + col * 2, (k * 256 + r) * 2048 + col * 2 + 512) for r in range(256)]
        return [effect(1, 1, "read", gm), local(2, "write", b_slot, 131072)]
    if kind == 4:
        return [local(2, "read", a_slot + kk * 16384, 16384), local(3, "write", left, 16384)]
    if kind == 5:
        # NZ rows select four of sixteen 16x16 blocks in every column group.
        strips = [(b_slot + c * 8192 + kk * 2048, b_slot + c * 8192 + kk * 2048 + 2048) for c in range(16)]
        return [effect(2, -1, "read", strips), local(4, "write", right, 32768)]
    if kind in (6, 7):
        reads = [local(3, "read", left, 16384), local(4, "read", right, 32768)]
        if kind == 7:
            reads.insert(0, local(5, "read", 0, 131072))
        return reads + [local(5, "write", 0, 131072)]
    gm = [((row + r) * 2048 + col * 2, (row + r) * 2048 + col * 2 + 512) for r in range(128)]
    return [local(5, "read", 0, 131072), effect(1, 2, "write", gm)]


def occurrences(waves, trips, inner, core, epochs=1):
    result = []

    def append(kind, coordinates, active):
        result.append(
            {
                "phase": kind,
                "coordinates": coordinates,
                "active": active,
                "pipe": PIPES[kind],
                "effects": effects(kind, coordinates, core),
            }
        )

    for epoch in range(epochs):
        prefix = [epoch] if epochs != 1 else []
        for wave in range(waves):
            active = 20 * wave + core < 256
            for kind in (0, 1):
                append(kind, prefix + [wave], active)
            for k in range(trips):
                for kind in (2, 3):
                    append(kind, prefix + [wave, k], active and k + 1 < trips)
                for kk in range(inner):
                    for kind in (4, 5, 6, 7):
                        enabled = active and (kind != 6 or k == kk == 0) and (kind != 7 or k != 0 or kk != 0)
                        append(kind, prefix + [wave, k, kk], enabled)
            append(8, prefix + [wave], active)
    return result


def overlap(a, b):
    return (
        a["space"] == b["space"]
        and a["base"] == b["base"]
        and any(x < v and u < y for x, y in a["intervals"] for u, v in b["intervals"])
    )


def required(items):
    active = [i for i, item in enumerate(items) if item["active"]]
    word = [items[i] for i in active]
    edges = native([item["pipe"] for item in word])
    for a, first in enumerate(word):
        for b in range(a + 1, len(word)):
            second = word[b]
            protected = (
                first["phase"] in (6, 7)
                and second["phase"] in (6, 7)
                and first["coordinates"][:-2] == second["coordinates"][:-2]
            )
            conflict = any(
                (x["mode"] == "write" or y["mode"] == "write") and overlap(x, y) and not (protected and x["space"] == 5)
                for x in first["effects"]
                for y in second["effects"]
            )
            if conflict:
                edges.add((2 * a + 1, 2 * b))
    reach, covers = closure(2 * len(word), edges)
    rows = [0] * (2 * len(items))
    for a, source in enumerate(active):
        for kind in (0, 1):
            for b, target in enumerate(active):
                for other in (0, 1):
                    if reach[2 * a + kind] >> (2 * b + other) & 1:
                        rows[2 * source + kind] |= 1 << (2 * target + other)
    return word, reach, covers, rows


def probes(items):
    cuts = defaultdict(set)
    for item in items:
        for access in item["effects"]:
            if access["space"] == 1:
                continue  # Root GM-discharge exports are investigated separately.
            for begin, end in access["intervals"]:
                cuts[access["space"]].update((begin, end))
    return [(space, -1, begin) for space, values in sorted(cuts.items()) for begin in sorted(values)]


def boundary(items, probe):
    space, base, byte = probe
    accesses = []
    for i, item in enumerate(items):
        if not item["active"]:
            continue
        modes = {
            e["mode"]
            for e in item["effects"]
            if e["space"] == space and e["base"] == base and any(begin <= byte < end for begin, end in e["intervals"])
        }
        if modes:
            accesses.append((i, item["pipe"], modes))
    writers = [i for i, _, modes in accesses if "write" in modes]
    first, last = defaultdict(list), defaultdict(list)
    for i, pipe, modes in accesses:
        if "write" in modes:
            continue
        if not writers or i < writers[0]:
            if not first[pipe]:
                first[pipe] = [i]
        if not writers or i > writers[-1]:
            last[pipe] = [i]
    return {
        "first_writers": writers[:1],
        "last_writers": writers[-1:],
        "first_readers": {str(k): v for k, v in first.items()},
        "last_readers": {str(k): v for k, v in last.items()},
    }
