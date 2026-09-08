# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Original Event Lab reference code retains its MIT notice below and in LICENSE.
# SPDX-License-Identifier: MIT
"""Exact finite memory sets stored as normalized half-open intervals.

This module is independent of symbolic dependence discovery. No byte is sampled
or dropped: intersection preserves the exact represented overlap. Its work is
proportional to the number of intervals, not their byte lengths.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, init=False)
class Footprint:
    intervals: tuple[tuple[str, int, int], ...]

    def __init__(self, intervals: Iterable[tuple[str, int, int]] = ()):
        source = list(intervals)
        for space, lo, hi in source:
            if not isinstance(space, str) or type(lo) is not int or type(hi) is not int:
                raise ValueError('memory intervals require a space and integer endpoints')
            if hi < lo:
                raise ValueError('reversed memory interval')
        merged: list[tuple[str, int, int]] = []
        for space, lo, hi in sorted(source):
            if hi == lo:
                continue
            if merged and merged[-1][0] == space and lo <= merged[-1][2]:
                old_space, old_lo, old_hi = merged[-1]
                merged[-1] = (old_space, old_lo, max(hi, old_hi))
            else:
                merged.append((space, lo, hi))
        object.__setattr__(self, 'intervals', tuple(merged))

    @staticmethod
    def convert(value):
        if isinstance(value, Footprint):
            return value
        # Compatibility with older review models and mutation tests. Only these
        # explicitly supplied finite sets are converted; the new importer never
        # enumerates byte sets.
        if isinstance(value, (set, frozenset)):
            return Footprint((space, coordinates[0], coordinates[0] + 1)
                             for space, coordinates in value)
        return NotImplemented

    def __bool__(self):
        return bool(self.intervals)

    @property
    def byte_count(self):
        return sum(hi - lo for _, lo, hi in self.intervals)

    def __or__(self, other):
        other = self.convert(other)
        if other is NotImplemented:
            return NotImplemented
        return Footprint(self.intervals + other.intervals)

    __ror__ = __or__

    def __and__(self, other):
        other = self.convert(other)
        if other is NotImplemented:
            return NotImplemented
        result = []
        a, b = self.intervals, other.intervals
        i = j = 0
        while i < len(a) and j < len(b):
            sa, la, ha = a[i]
            sb, lb, hb = b[j]
            if sa < sb:
                i += 1
            elif sb < sa:
                j += 1
            else:
                lo, hi = max(la, lb), min(ha, hb)
                if lo < hi:
                    result.append((sa, lo, hi))
                if ha <= hb:
                    i += 1
                else:
                    j += 1
        return Footprint(result)

    __rand__ = __and__

    def report(self):
        return {'intervals': [list(x) for x in self.intervals],
                'bytes': self.byte_count}
