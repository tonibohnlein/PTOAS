# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Concrete event balance must not manufacture asynchronous reuse evidence."""
from pathlib import Path
import tempfile
import unittest

from quality_matrix import endpoint_replay


class EndpointEvidenceTests(unittest.TestCase):
    def run_events(self, body):
        with tempfile.TemporaryDirectory(prefix='oahs-endpoint-') as scratch:
            source = Path(scratch) / 'input.pto'
            source.write_text('module { func.func @f() {\n' + body + '\nreturn\n} }')
            return endpoint_replay(source, [dict(name='one', arguments=[])])['executions'][0]

    @staticmethod
    def transfer(source, observer):
        return (f'pto.set_flag[<PIPE_{source}>, <PIPE_{observer}>, <EVENT_ID0>]\n'
                f'pto.wait_flag[<PIPE_{source}>, <PIPE_{observer}>, <EVENT_ID0>]\n')

    def test_return_receipt_orders_rearm(self):
        exchange = self.transfer('V', 'MTE2') + self.transfer('MTE2', 'V')
        evidence = self.run_events(exchange * 2 + 'pto.barrier <PIPE_ALL>')
        self.assertTrue(evidence['consumption_before_rearm_proven'])
        self.assertTrue(evidence['all_generations_consumed'])
        self.assertEqual([event['generation'] for event in evidence['events']], [1] * 4 + [2] * 4)

    def test_equal_counts_without_receipt_do_not_prove_rearm(self):
        evidence = self.run_events(self.transfer('V', 'MTE2') * 2 + 'pto.barrier <PIPE_ALL>')
        self.assertTrue(evidence['all_generations_consumed'])
        self.assertFalse(evidence['consumption_before_rearm_proven'])
        self.assertEqual(len(evidence['unproven_rearms']), 1)

    def test_missing_acquisition_refused(self):
        with self.assertRaises(ValueError):
            self.run_events('pto.set_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]')

    def test_unknown_helper_refused(self):
        with self.assertRaisesRegex(ValueError, 'unqualified helper'):
            self.run_events('func.call @f() : () -> ()')


if __name__ == '__main__':
    unittest.main()
