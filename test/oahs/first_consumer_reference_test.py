# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Reference protocol regression for invariant MAT first-consumer placement.

These are ordinary prefix-command graphs, not constructor output or device
predictions. The native projection comparison is recorded in the design note.
Every conflict and rearming edge must come from the actual selected words.
"""
import unittest

from check_carried_slot_trace import Trace


class FirstConsumerTrace(Trace):
    def __init__(self):
        super().__init__()
        self.starts = []

    def payload(self, *args, **kwargs):
        self.starts.append(len(self.ancestors))
        super().payload(*args, **kwargs)

    def ordered(self, source, consumer):
        return bool(self.ancestors[self.starts[consumer]] &
                    (1 << self.payloads[source][1]))

    def relations(self):
        return {(i, j) for j in range(len(self.payloads)) for i in range(j)
                if self.ordered(i, j)}


def protocol(lengths, placement="first", repeat_receipt=False, missing=False):
    trace = FirstConsumerTrace()
    deadlines = []
    a, b = ("mat", 0, 64), ("mat", 64, 64)

    def event(kind, source, observer, key):
        trace.sync(kind, source, observer, key)

    def acquire_b():
        if missing:
            return
        event("wait_flag", "MTE2", "MTE1", 1)
        # Retain real consumption evidence before the next B publication.
        event("set_flag", "MTE1", "MTE2", 1)
        event("wait_flag", "MTE1", "MTE2", 1)

    for episode, length in enumerate(lengths):
        # A skipped *whole* episode preserves the interface. This does not
        # model a producer executing before a possibly empty reader loop.
        if not length:
            continue
        context = (("episode", episode),)
        trace.payload("tload", [(a, True)], context)
        event("set_flag", "MTE2", "MTE1", 0)
        load_b = len(trace.payloads)
        trace.payload("tload", [(b, True)], context)
        if placement != "late" and not missing:
            event("set_flag", "MTE2", "MTE1", 1)
        late_load = len(trace.payloads)
        trace.payload("tload", [(("mat", 128 + 64 * episode, 64), True)], context)
        event("wait_flag", "MTE2", "MTE1", 0)
        if placement == "entry":
            acquire_b()
        for iteration in range(length):
            use = context + (("inner", iteration),)
            first_a = len(trace.payloads)
            trace.payload("textract", [(a, False)], use)
            if iteration == 0 or repeat_receipt:
                if placement == "late" and not missing:
                    event("set_flag", "MTE2", "MTE1", 1)
                if placement != "entry":
                    acquire_b()
            first_b = len(trace.payloads)
            trace.payload("textract", [(b, False)], use)
            if iteration == 0:
                deadlines.append((load_b, late_load, first_a, first_b))
        # Required by the next actual overwrite of A and B. It also rearms
        # A readiness and itself through the next A-readiness acquisition.
        event("set_flag", "MTE1", "MTE2", 0)
        event("wait_flag", "MTE1", "MTE2", 0)
    trace.sync("barrier", "ALL")
    assert not trace.live
    return trace, deadlines


class FirstConsumerPlacementTest(unittest.TestCase):
    def test_first_consumer_preserves_early_prefix(self):
        for lengths in ((), (0,), (1,), (2,), (2, 0, 3, 1), (0, 4, 0, 2)):
            with self.subTest(lengths=lengths):
                good, deadlines = protocol(lengths)
                late, _ = protocol(lengths, "late", repeat_receipt=True)
                self.assertEqual([(p[0], p[2:]) for p in good.payloads],
                                 [(p[0], p[2:]) for p in late.payloads])
                self.assertFalse(good.relations() - late.relations())
                for load_b, unrelated, first_a, first_b in deadlines:
                    self.assertTrue(good.ordered(load_b, first_b))
                    self.assertFalse(good.ordered(load_b, first_a))
                    self.assertFalse(good.ordered(unrelated, first_b))
                    self.assertTrue(late.ordered(unrelated, first_b))

    def test_entry_wait_is_safe_but_orders_unrelated_first_consumer(self):
        good, _ = protocol((2,))
        entry, deadlines = protocol((2,), "entry")
        load_b, _, first_a, _ = deadlines[0]
        self.assertFalse(good.ordered(load_b, first_a))
        self.assertTrue(entry.ordered(load_b, first_a))
        self.assertTrue(entry.relations() - good.relations())

    def test_missing_readiness_rejected(self):
        with self.assertRaisesRegex(AssertionError, "missing physical conflict"):
            protocol((2,), missing=True)

    def test_one_token_cannot_be_consumed_on_every_iteration(self):
        with self.assertRaisesRegex(AssertionError, "unseeded wait"):
            protocol((2,), repeat_receipt=True)


if __name__ == "__main__":
    unittest.main()
