# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Abstract reuse-order oracle; physical claims additionally assume local adjacency."""
import shutil
import sys
from check_periodic_demands import requests, run, unfolded


def handoffs(case, length, source_pipe, target_pipe):
    reach, covers = unfolded(case, length)
    types = len(case["pipes"])
    pairs = sorted((a // 2, b // 2) for a, b in covers
                   if case["pipes"][(a // 2) % types] == source_pipe
                   and case["pipes"][(b // 2) % types] == target_pipe)
    return reach, pairs


def lifetimes(reach, pairs):
    count = len(pairs)
    ends = []
    for index, (_, consumer) in enumerate(pairs):
        ends.append(next((later for later in range(index + 1, count)
                          if (reach[2 * consumer + 1] >> (2 * pairs[later][0])) & 1), count))
    # Explicit interval overlap, rather than the periodic maximum-phase formula.
    width = max((sum(start <= point < end for start, end in enumerate(ends))
                 for point in range(count)), default=0)
    assert width == max((end - start for start, end in enumerate(ends)), default=0)
    return ends, width


def check(case, result):
    allocation = result["allocation"]
    assert not allocation["error"], allocation["error"]
    assert allocation["budgets_ready"] and not allocation["physical_ids_ready"] and not allocation["ir_emitted"]
    types = len(case["pipes"])
    for pool, values in zip(allocation["directions"], allocation["checks"]):
        source, target = pool["source_pipe"], pool["target_pipe"]
        assert source != target
        # A shortest return path uses <=2m-1 quotient edges, each <=3 in
        # these random cases. Include target delay and a full following period.
        reach, pairs = handoffs(case, types * (6 * types + 8), source, target)
        ends, _ = lifetimes(reach, pairs)
        phases = len(pool["handoffs"])
        expected_reuse = [None if ends[r] == len(pairs) else ends[r] - r for r in range(phases)]
        assert [entry["first_reuse"] for entry in pool["handoffs"]] == expected_reuse
        uniform = None if None in expected_reuse else max(expected_reuse)
        assert pool["uniform_budget"] == uniform
        for capacity, answer in zip(case["capacities"], values["uniform"]):
            assert answer == {"error": 0, "sufficient": uniform is not None and capacity >= uniform}
        for prefix, measured in zip(case["allocation_prefixes"], values["finite"]):
            reach, pairs = handoffs(case, prefix, source, target)
            ends, width = lifetimes(reach, pairs)
            assert measured["count_error"] == measured["budget_error"] == 0
            assert measured["handoffs"] == len(pairs) and measured["budget"] == width
            for capacity, answer in zip(case["capacities"], measured["fits"]):
                assert answer == {"error": 0, "sufficient": capacity >= width}
            # Independently verify EVERY same-ID pair under cyclic assignment.
            if width:
                for i in range(len(pairs)):
                    for j in range(i + width, len(pairs), width):
                        assert (reach[2 * pairs[i][1] + 1] >> (2 * pairs[j][0])) & 1
                if width > 1:
                    assert any(not ((reach[2 * pairs[i][1] + 1] >> (2 * pairs[j][0])) & 1)
                               for i in range(len(pairs)) for j in range(i + width - 1, len(pairs), width - 1))


def main():
    tool = shutil.which(sys.argv[1])
    assert tool
    cases = [case for case in requests() if "scan" not in case]
    for case in cases:
        size = len(case["pipes"])
        case["allocation"] = True
        case["allocation_prefixes"] = sorted({0, 1, size, size + 1, 2 * size, 3 * size - 1, 4 * size} - {-1})
        case["capacities"] = [0, 1, 2, 3, 8, 32]
    results = run(tool, cases)
    for case, result in zip(cases, results):
        check(case, result)
    maximum = 2**64 - 1
    special = [
        {"pipes": [0, 1], "records": [[0, 1, 0]], "allocation": True,
         "allocation_prefixes": [0, 1, 2, 5, maximum], "capacities": [0, 1, maximum],
         "allocation_contracts": True,
         "offset_queries": [[0, 0, maximum, maximum], [0, 0, 1, 0], [0, 1, 0, 2]]},
        {"pipes": [0, 1], "records": [[0, 1, 0], [1, 0, 4]], "allocation": True,
         "allocation_prefixes": [0, 1, 2, 7, 8, 9, 10, 20, maximum], "capacities": [0, 3, 4]},
        {"pipes": [0, 0, 1, 1], "records": [[0, 2, 0], [1, 3, 0]], "allocation": True,
         "allocation_prefixes": [maximum], "capacities": [0],
         "offset_queries": [[0, 0, maximum, maximum], [0, 1, maximum, maximum],
                            [0, 1, maximum, 13]]},
    ]
    output = run(tool, special)
    one = output[0]["allocation"]
    assert one["contract_checks"]
    assert one["directions"][0]["uniform_budget"] is None
    assert [item["budget"] for item in one["checks"][0]["finite"]] == [0, 0, 1, 2, maximum // 2]
    assert one["offset_answers"] == [{"error": 0, "value": 0}, {"error": 1, "value": 0},
                                      {"error": 1, "value": 0}]
    rotation = output[1]["allocation"]
    assert [pool["uniform_budget"] for pool in rotation["directions"]] == [4, 4]
    for pool, answers in zip(rotation["directions"], rotation["checks"]):
        for item in answers["finite"]:
            periods = item["prefix"] // 2 if pool["source_pipe"] == 0 else (item["prefix"] + 1) // 2
            handoffs = periods if pool["source_pipe"] == 0 else max(0, periods - 4)
            assert item["handoffs"] == handoffs and item["budget"] == min(4, handoffs)
    wide = output[2]["allocation"]
    assert wide["offset_answers"] == [{"error": 0, "value": 0}, {"error": 0, "value": 1},
                                       {"error": 0, "value": (2 * maximum + 1) % 13}]
    print(f"periodic allocation: {len(cases)} abstract reuse-order oracles and large-prefix cases passed")


if __name__ == "__main__":
    main()
