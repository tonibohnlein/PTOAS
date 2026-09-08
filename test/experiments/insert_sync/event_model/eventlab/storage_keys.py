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
"""Construct candidate event-key functions from physical conflict footprints.

Storage is proposal evidence, never a reuse certificate. The caller must prove
all next-same-key consumptions in the unchanged plan. No iterator position,
variable name, kernel topology, or sampled launch defines the key rule here.
"""
from __future__ import annotations
from dataclasses import dataclass
from .islwrap import Rel


@dataclass
class StorageKeyCandidate:
    labels: Rel                       # publication -> finite descriptor rank
    slots: list[tuple[int, int, int]] # address-space rank, start, byte extent
    selected_footprints: Rel

    def keys(self, program, count):
        color = program.relation([
            f'StorageRank[r] -> Key[r % {count}] : 0 <= r < {len(self.slots)}'])
        return self.labels.then(color)

    def report(self):
        return {'kind': 'shared_conflict_write_footprint',
                'descriptors': [list(x) for x in self.slots],
                'selection': str(self.selected_footprints),
                'qualification': 'candidate only; complete symbolic rearm proof is mandatory'}


def derive_storage_candidate(plan, domain, max_descriptors=64):
    """Recover a canonical local write footprint participating in each handoff.

    Ready: source write overlapping the matching consumer's access.
    Release: matching target write overlapping the source's access.

    This deliberately uses write-footprint starts, NOT the first read byte.
    An inner panel reader may read a subrange, while the next writer replaces
    the full physical slot. For multi-footprint handoffs lexmin is a deterministic
    proposal; final causal proof, not this choice, establishes safe assignment.
    """
    program = plan.program
    handoff = plan.handoffs[domain]
    footprints = program.empty()
    all_accesses = program.reads | program.writes
    space_codes = {space: i for i, space in enumerate(('MAT', 'LEFT', 'RIGHT', 'ACC', 'VEC'))}
    matching_targets = handoff.then(all_accesses)
    matching_sources = handoff.reverse().then(all_accesses)
    for op in program.statements:
        for access in op.get('writes', []):
            space = access['space']
            if space not in space_codes:
                continue
            start, size = access['address'], access['size']
            occurrence = program.instance(op)
            condition = program.condition(op)
            write = program.relation([
                occurrence + f' -> {space}[byte] : {condition} and '
                f'({start}) <= byte < ({start}) + {size}'])
            descriptor = program.relation([
                occurrence + f' -> StorageSlot[{space_codes[space]}, ({start}), {size}] : {condition}'])
            source_overlap = write & matching_targets
            target_overlap = write & matching_sources
            footprints = footprints | descriptor.filter(domain=source_overlap)
            footprints = footprints | handoff.then(descriptor.filter(domain=target_overlap))

    selected = footprints.lexmin().filter(domain=handoff)
    # Use domain identities, not a quadratic selected ; inverse(selected)
    # relation, to check that every publication received a descriptor.
    if not handoff.domain_identity().subset(selected.domain_identity()):
        return None, 'some publication has no represented shared local write footprint'
    try:
        # Existentially project all launch parameters before enumerating only
        # the bounded physical descriptor set. This is NOT sampling launches
        # and it never enumerates panel bytes or runtime generations.
        values = selected.range_identity(project_parameters=True).points({}, limit=max_descriptors)
    except ValueError as error:
        if str(error) not in ('enumeration budget exceeded', 'point enumeration requires a bounded specialization'):
            raise
        return None, 'physical descriptor range is unbounded or exceeds the candidate budget'
    slots = sorted({coordinates for (_, coordinates), _ in values})
    if not slots:
        return None, 'empty descriptor range'
    ranks = program.relation([
        f'StorageSlot[{space}, {start}, {size}] -> StorageRank[{rank}]'
        for rank, (space, start, size) in enumerate(slots)])
    labels = selected.then(ranks)
    if not labels.single() or not handoff.domain_identity().subset(labels.domain_identity()):
        raise ValueError('internal storage candidate mapping is incomplete or ambiguous')
    return StorageKeyCandidate(labels, slots, selected), None
