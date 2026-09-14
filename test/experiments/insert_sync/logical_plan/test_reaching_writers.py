# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import unittest
from reaching_writers import WriterFlow


def report(structure, accesses=None):
    return dict(nodes=[dict(id=i, kind=kind, children=children) for i, (kind, children) in enumerate(structure)],
                accesses=accesses or [dict(node=i, macro_phase=None) for i, (kind, _) in enumerate(structure) if kind == 0])


class WriterFlowTests(unittest.TestCase):
    def relation(self, structure, writer=0, reader=1):
        return WriterFlow(report(structure)).relation(dict(node=writer), dict(node=reader))

    def test_order_and_opposite_arms(self):
        leaves = [(0, []), (0, [])]
        self.assertEqual(self.relation(leaves + [(2, [0, 1])]), 'may-reach-without-backedge')
        self.assertEqual(self.relation(leaves + [(2, [1, 0])]), 'cannot-precede-within-invocation')
        self.assertEqual(self.relation(leaves + [(3, [0, 1])]), 'cannot-precede-within-invocation')

    def test_loop_backedge_and_zero_trip_exit(self):
        self.assertEqual(self.relation([(0, []), (0, []), (2, [1, 0]), (4, [2])]),
                         'may-reach-via-backedge')
        self.assertEqual(self.relation([(0, []), (4, [0]), (0, []), (2, [1, 2])], 0, 2),
                         'may-reach-without-backedge')
        # This says MAY, not a guaranteed write on the zero-trip path.

    def test_while_before_after_and_exit(self):
        shape = [(0, []), (0, []), (5, [1, 0])]
        self.assertEqual(self.relation(shape), 'may-reach-via-backedge')
        self.assertEqual(self.relation(shape, 1, 0), 'may-reach-without-backedge')

    def test_macro_phase_order(self):
        accesses = [dict(node=0, macro_phase=i) for i in range(2)]
        flow = WriterFlow(report([(1, [])], accesses))
        self.assertEqual(flow.relation(accesses[1], accesses[0]), 'cannot-precede-within-invocation')
        self.assertEqual(flow.relation(accesses[0], accesses[1]), 'may-reach-without-backedge')

    def test_exhaustion_and_missing_mapping_are_unknown(self):
        r = report([(0, []), (0, []), (2, [0, 1])])
        self.assertEqual(WriterFlow(r, limit=0).relation(dict(node=0), dict(node=1)), 'analysis-exhausted')
        r['report_exhausted'] = True
        self.assertEqual(WriterFlow(r).relation(dict(node=0), dict(node=1)), 'incomplete-report')

    def test_paths_name_actual_backedge_owners(self):
        flow = WriterFlow(report([(0, []), (0, []), (2, [1, 0]), (4, [2])]))
        witness = flow.witness(dict(node=0), dict(node=1))
        self.assertEqual(witness['backedge_owners'], [3])
        self.assertEqual(witness['path'][0], (0, None))
        self.assertEqual(witness['path'][-1], (1, None))
        for a, b in zip(witness['path'], witness['path'][1:]):
            self.assertTrue(any(point == b for point, _ in flow.graph[a]))
        forward = flow.witness(dict(node=1), dict(node=0))
        self.assertEqual(forward['backedge_owners'], [])
        # Path materialization has the same aggregate allowance as traversal.
        flow.limit = flow.work
        self.assertEqual(flow.witness(dict(node=0), dict(node=1))['relation'], 'analysis-exhausted')


if __name__ == '__main__':
    unittest.main()
