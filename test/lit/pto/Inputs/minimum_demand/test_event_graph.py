# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Mutation tests for the independent event-graph oracle."""

import unittest
from event_graph import Command
from event_graph import InvalidPlan
from event_graph import Payload
from event_graph import check
from event_graph import minimum_generators


def payload(pipe, reads=(), writes=()):
    """Build a concrete occurrence with cell sets."""
    return Payload(pipe, frozenset(reads), frozenset(writes))


def flag(kind, handoff, src="p", dst="q", event_id=0):
    """Build an explicitly matched flag endpoint."""
    return Command(kind, handoff, src, dst, event_id)


class EventGraphTests(unittest.TestCase):
    """Check the oracle against valid plans and semantic mutations."""

    def setUp(self):
        self.pools = {("p", "q"): set(range(8)), ("q", "p"): set(range(8)),
                      ("r", "p"): set(range(8))}
        self.items = [payload("p", writes=("x",)), payload("q", reads=("x",))]
        self.streams = {"p": [0, flag("set", "a")], "q": [flag("wait", "a"), 1]}

    def test_direct(self):
        check(self.items, self.streams, self.pools)

    def test_missing_dependency(self):
        with self.assertRaisesRegex(InvalidPlan, "missing storage"):
            check(self.items, {"p": [0], "q": [1]}, self.pools)

    def test_missing_wait(self):
        with self.assertRaisesRegex(InvalidPlan, "unmatched"):
            check(self.items, {"p": self.streams["p"], "q": [1]}, self.pools)

    def test_incorrect_matching(self):
        streams = {"p": self.streams["p"], "q": [flag("wait", "a", event_id=1), 1]}
        with self.assertRaisesRegex(InvalidPlan, "incorrect physical matching"):
            check(self.items, streams, self.pools)

    def test_reserved_id(self):
        with self.assertRaisesRegex(InvalidPlan, "reserved physical ID"):
            check(self.items, self.streams, {("p", "q"): {1}})

    def test_unsafe_reuse_despite_issue_order(self):
        items = self.items + [payload("p", writes=("y",)), payload("q", reads=("y",))]
        streams = {"p": [0, flag("set", "a"), 2, flag("set", "b")],
                   "q": [flag("wait", "a"), 1, flag("wait", "b"), 3]}
        with self.assertRaisesRegex(InvalidPlan, "consumption-before-republication"):
            check(items, streams, self.pools)

    def test_causal_reuse(self):
        items = self.items + [payload("p", writes=("x",)), payload("q", reads=("x",))]
        streams = {"p": [0, flag("set", "a"), flag("wait", "back", "q", "p"),
                         2, flag("set", "b")],
                   "q": [flag("wait", "a"), 1, flag("set", "back", "q", "p"),
                         flag("wait", "b"), 3]}
        check(items, streams, self.pools)

    def test_set_after_acquisition_adds_prerequisite(self):
        items = [payload("p", writes=("x",)), payload("r", writes=("y",)),
                 payload("p", reads=("y",)), payload("q", reads=("x",))]
        streams = {"p": [0, flag("set", "a"), flag("wait", "b", "r", "p"), 2],
                   "r": [1, flag("set", "b", "r", "p")],
                   "q": [flag("wait", "a"), 3]}
        check(items, streams, self.pools)
        streams["p"] = [0, flag("wait", "b", "r", "p"), flag("set", "a"), 2]
        with self.assertRaisesRegex(InvalidPlan, "added payload prerequisite"):
            check(items, streams, self.pools)

    def test_nonadjacent_barrier_adds_prerequisite(self):
        items = [payload("p", writes=("x",)), payload("p", writes=("y",)),
                 payload("p", reads=("x",))]
        with self.assertRaisesRegex(InvalidPlan, "added payload prerequisite"):
            check(items, {"p": [0, 1, Command("barrier"), 2]}, self.pools)

    def test_adjacent_barrier(self):
        items = [payload("p", writes=("x",)), payload("p", reads=("x",))]
        check(items, {"p": [0, Command("barrier"), 1]}, self.pools)

    def test_macro_boundary_widens_earlier_phase_release(self):
        # The shared A2/A3 compare TGATHER has V/S/V/S effects. This tests
        # those effects under the core native contract, not undocumented
        # internal macro command edges or a device execution.
        items = [payload("v", reads=("src",), writes=("tmp",)),
                 payload("s", writes=("tmp",)),
                 payload("v", reads=("tmp",), writes=("dst",)),
                 payload("s", writes=("count",)),
                 payload("mte2", writes=("src",))]
        pools = {("v", "s"): {0}, ("s", "v"): {1}, ("v", "mte2"): {0}}
        release = flag("set", "release", "v", "mte2")
        streams = {"v": [0, flag("set", "a", "v", "s"), release,
                         flag("wait", "b", "s", "v", 1), 2],
                   "s": [flag("wait", "a", "v", "s"), 1,
                         flag("set", "b", "s", "v", 1), 3],
                   "mte2": [flag("wait", "release", "v", "mte2"), 4]}
        check(items, streams, pools)
        streams["v"] = [0, flag("set", "a", "v", "s"),
                        flag("wait", "b", "s", "v", 1), 2, release]
        with self.assertRaisesRegex(InvalidPlan, "added payload prerequisite"):
            check(items, streams, pools)

    def test_no_command_for_independent_payloads(self):
        items = [payload("p", reads=("x",)), payload("p", reads=("x",))]
        check(items, {"p": [0, 1]}, self.pools)

    def test_initial_reader_release_and_read_modify_write(self):
        items = [payload("q", reads=("x",)), payload("p", reads=("x",), writes=("x",))]
        streams = {"q": [0, flag("set", "release", "q", "p")],
                   "p": [flag("wait", "release", "q", "p"), 1]}
        check(items, streams, self.pools)

    def test_cycle(self):
        streams = {"p": [flag("wait", "b", "q", "p"), 0, flag("set", "a")],
                   "q": [flag("wait", "a"), 1, flag("set", "b", "q", "p")]}
        with self.assertRaisesRegex(InvalidPlan, "cyclic"):
            check(self.items, streams, self.pools)

    def test_duplicate_endpoint(self):
        streams = {"p": [0, flag("set", "a"), flag("set", "a")],
                   "q": self.streams["q"]}
        with self.assertRaisesRegex(InvalidPlan, "duplicate logical endpoint"):
            check(self.items, streams, self.pools)

    def test_duplicate_payload(self):
        with self.assertRaisesRegex(InvalidPlan, "payload missing or repeated"):
            check(self.items, {"p": [0, 0], "q": [1]}, self.pools)

    def test_extra_wait(self):
        streams = {"p": self.streams["p"],
                   "q": self.streams["q"] + [flag("wait", "extra")]}
        with self.assertRaisesRegex(InvalidPlan, "unmatched logical handoff"):
            check(self.items, streams, self.pools)

    def test_reversed_payload_order(self):
        items = [payload("p", writes=("x",)), payload("p", writes=("y",))]
        with self.assertRaisesRegex(InvalidPlan, "cyclic command graph"):
            check(items, {"p": [1, 0]}, self.pools)

    def test_supplied_prerequisite(self):
        items = [payload("p"), payload("q")]
        check(items, self.streams, self.pools, prerequisites=((0, 1),))

    def test_reader_mediated_waw_cover(self):
        items = self.items + [payload("p", writes=("x",))]
        expected = {(0, 1), (1, 2)}
        self.assertEqual(minimum_generators(items), expected)
        self.assertEqual(minimum_generators(items, expected), expected)
        self.assertEqual(minimum_generators(items, expected | {(0, 2)}), expected)

    def test_partial_write_preserves_old_writer(self):
        items = [payload("p", writes=("x0", "x1")), payload("q", writes=("x0",)),
                 payload("r", reads=("x1",))]
        self.assertEqual(minimum_generators(items), {(0, 1), (0, 2)})

    def test_skipped_store_cross_cell_reduction(self):
        items = [payload("l", writes=("x0",)), payload("v", reads=("x0",), writes=("y0",)),
                 payload("l", writes=("x1",)), payload("v", reads=("x1",), writes=("y1",)),
                 payload("s", reads=("y1",)), payload("l", writes=("x0",)),
                 payload("v", reads=("x0",), writes=("y0",)), payload("s", reads=("y0",))]
        covers = minimum_generators(items)
        self.assertEqual(covers, {(0, 1), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7)})
        self.assertNotIn((1, 6), covers)
        self.assertTrue(all(items[a].pipe != items[b].pipe for a, b in covers))

    def test_double_buffer_zero_one_many_unfoldings(self):
        for trips in range(8):
            items = []
            for iteration in range(trips):
                cell = f"x{iteration % 2}"
                items.extend((payload("p", writes=(cell,)), payload("q", reads=(cell,))))
            expected = {(2 * i, 2 * i + 1) for i in range(trips)}
            expected.update((2 * i + 1, 2 * (i + 2)) for i in range(max(0, trips - 2)))
            self.assertEqual(minimum_generators(items), expected)

    def test_read_modify_write_cover(self):
        items = [payload("p", writes=("x",)), payload("q", reads=("x",), writes=("x",)),
                 payload("r", reads=("x",))]
        self.assertEqual(minimum_generators(items), {(0, 1), (1, 2)})

    def test_retained_nonadjacent_local_cover(self):
        items = [payload("p", writes=("x",)), payload("p", writes=("y",)),
                 payload("p", reads=("x",))]
        self.assertEqual(minimum_generators(items), {(0, 2)})

    def test_empty_execution(self):
        check([], {}, self.pools)


if __name__ == "__main__":
    unittest.main()
