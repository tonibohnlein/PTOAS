# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Regression checks for the evidence required by publication classification."""
import unittest

from coverage_campaign import witness_classes


class PublicationWitnessTests(unittest.TestCase):
    def classify(self, left, right, overlap=True, **contracts):
        return witness_classes([
            dict(function='test', accesses=[dict(id=0, **left), dict(id=1, **right)],
                 publication_pairs=[dict(writer=0, reader=1, native_overlap=overlap)], **contracts)])[0]

    def test_coarsened_alias_cell_is_lost_precision(self):
        left = dict(roots=['A'], origins_complete=True, gm_lower='0', gm_upper='512')
        right = dict(roots=['A'], origins_complete=True, gm_lower='512', gm_upper='1024')
        for overlap in (True, False):
            row = self.classify(left, right, overlap)
            self.assertEqual(row['classification'], 'lost-provenance-or-range-precision')
            self.assertTrue(row['proven_disjoint_ranges'])

    def test_unknown_range_does_not_establish_actual_overlap(self):
        unknown = dict(roots=['A'], origins_complete=True)
        self.assertEqual(self.classify(unknown, unknown)['classification'], 'unresolved-evidence')

    def test_intersecting_exact_ranges_allow_possible_overlap(self):
        left = dict(roots=['A'], origins_complete=True, gm_lower='0', gm_upper='512')
        right = dict(roots=['A'], origins_complete=True, gm_lower='256', gm_upper='768')
        self.assertEqual(self.classify(left, right)['classification'], 'genuine-possible-overlap')

    def test_offsets_on_distinct_roots_are_not_comparable(self):
        left = dict(roots=['A'], origins_complete=True, gm_lower='0', gm_upper='512')
        right = dict(roots=['B'], origins_complete=True, gm_lower='512', gm_upper='1024')
        self.assertFalse(self.classify(left, right)['proven_disjoint_ranges'])
        self.assertEqual(self.classify(left, right)['classification'], 'unresolved-evidence')
        self.assertEqual(self.classify(left, right, False)['classification'], 'unresolved-evidence')

    def test_coarsening_does_not_erase_disjointness_evidence(self):
        left = dict(roots=['A'], origins_complete=True, gm_lower='0', gm_upper='512')
        right = dict(roots=['B'], origins_complete=True, gm_lower='0', gm_upper='512')
        for contracts in (dict(alias_contract='assume-disjoint-arguments'),
                          dict(disjoint_root_pairs=[['A', 'B']])):
            self.assertEqual(self.classify(left, right, **contracts)['classification'],
                             'lost-provenance-or-range-precision')
            self.assertEqual(self.classify(left, right, False, **contracts)['classification'],
                             'established-disjointness-contract')

    def test_union_requires_every_cross_root_contract(self):
        left = dict(roots=['A', 'B'], origins_complete=True)
        right = dict(roots=['C'], origins_complete=True)
        incomplete = self.classify(left, right, disjoint_root_pairs=[['A', 'C']])
        self.assertFalse(incomplete['proven_disjoint_contract'])
        complete = self.classify(left, right, disjoint_root_pairs=[['A', 'C'], ['B', 'C']])
        self.assertTrue(complete['proven_disjoint_contract'])
        self.assertEqual(complete['classification'], 'lost-provenance-or-range-precision')


if __name__ == '__main__':
    unittest.main()
