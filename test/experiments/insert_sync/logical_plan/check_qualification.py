#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Challenge the extracted physical admission boundary on actual native IR."""
import argparse
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seed = (HERE / 'inputs/skipped_reader.pto').read_text()
    memory = seed.replace('scf.for %iv = %c0 to %trip step %c1 {',
        '%loop_result = scf.for %iv = %c0 to %trip step %c1 iter_args(%carried = %in0) -> (!pto.tile_buf<vec, 16x16xf16>) {')
    memory = memory.replace('    }\n    return', '      scf.yield %carried : !pto.tile_buf<vec, 16x16xf16>\n    }\n    return')
    cases = {
        'positive': (seed, True),
        'missing_core': (seed.replace('attributes {pto.kernel_kind = #pto.kernel_kind<vector>}', ''), False),
        'wrong_core': (seed.replace('#pto.kernel_kind<vector>', '#pto.kernel_kind<cube>'), False),
        'unqualified_arch': (seed.replace('pto.target_arch = "a3"', 'pto.target_arch = "a5"'), False),
        'unknown_address': (seed.replace('%ia0 = arith.constant 0 : i64', '%ia0 = arith.index_cast %trip : index to i64'), False),
        'memory_carry': (memory, False),
        'while': (seed.replace('    return', '''    scf.while : () -> () {
      %never = arith.constant false
      scf.condition(%never)
    } do {
      scf.yield
    }
    return'''), False),
        'authored_sync': (seed.replace('    return', '    pto.barrier #pto.pipe<PIPE_ALL>\n    return'), False),
        'missing_adapter': (seed.replace('pto.tabs ins(%in0 : !pto.tile_buf<vec, 16x16xf16>)',
            'pto.tdiv ins(%in0, %in0 : !pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 16x16xf16>)'), False),
    }
    rows = []
    for name, (source, accepted) in cases.items():
        path = args.output / (name + '.pto')
        path.write_text(source)
        process = subprocess.run([str(args.driver.resolve()), str(path.resolve()), 'none'],
            capture_output=True, text=True, timeout=90)
        (args.output / (name + '.stderr')).write_text(process.stderr)
        assert process.returncode == (0 if accepted else 1), (name, process.stderr)
        result = json.loads(process.stdout)
        assert result['applied'] == accepted, (name, result)
        if not accepted:
            assert result['original_preserved'] and not result['invoked'], (name, result)
            assert result['status'] == 'unsupported', (name, result)
        if name in ('memory_carry', 'while'):
            assert 'memory loop forwarding' in result['reason'], (name, result)
        rows.append({'case': name, 'accepted': accepted, 'reason': result['reason'], 'work': result['work']})
        print(name, 'accepted' if accepted else result['reason'], flush=True)
    access_seed = (HERE / "inputs/one_buffer.auto.pto").read_text()
    access_rows = []
    for name, source, disjoint in (
        ("dynamic_bound", access_seed, False),
        ("literal_partition", access_seed.replace("to %trip step", "to %c16 step"), True),
        ("overlapping_output", access_seed.replace("to %trip step", "to %c16 step").replace(
            "offsets = [%or0_0, %c0]", "offsets = [%c0, %c0]"), False),
    ):
        path = args.output / (name + ".pto")
        path.write_text(source)
        process = subprocess.run([str(args.driver.resolve()), str(path.resolve()), "access-queries"],
            capture_output=True, text=True, timeout=90)
        assert process.returncode == 0, (name, process.stderr)
        data = json.loads(process.stdout)
        assert data["accesses"] == 2 and len(data["queries"]) == 4, data
        output = next(q for q in data["queries"] if q["source_write"] and q["target_write"])
        assert output["source"] == output["target"] and output["may_alias"], data
        assert output["distinct_occurrences_disjoint"] == disjoint, (name, data)
        for query in data["queries"]:
            if query["source_write"] != query["target_write"]:
                assert query["may_alias"] and not query["disjoint_arguments_alias"], (name, data)
                assert not query["distinct_occurrences_disjoint"], (name, data)
        access_rows.append({"case": name, **data})
    (args.output / "access_queries.json").write_text(json.dumps(access_rows, indent=2) + "\n")
    (args.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
