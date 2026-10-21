# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check a production tuple-map repair using independent original-access graphs."""

import argparse
import json
import os
from pathlib import Path
import resource
import shutil
from types import SimpleNamespace

from assess_whole_kernel_access import annotate, compile_stage, digest, invoke
from check_finite_allocation import physical_check
from check_whole_kernel_access import independent
from compact_vector_oracle import command_order
from whole_kernel_reference import reference


def trace(args, folder, source):
    path = folder / (source.stem + '-trace.pto')
    path.write_text(annotate(source.read_text(), 'test.trace_existing'))
    run = invoke([args.trace_tool, '--structured-trace', path], path.with_suffix('.json'), 35)
    assert run['status'] == 'success', run
    result = json.loads(Path(run['stdout']).read_text())
    assert result.get('accepted') and not result['trace']['error'], result
    return result['trace']


def pairs(rows):
    return {(a, b) for a, row in enumerate(rows) for b in range(len(rows))
            if a != b and row >> b & 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('optimizer', 'trace-tool', 'input', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('optimizer', 'trace_tool', 'input', 'output'):
        path = getattr(args, name)
        setattr(args, name, Path(shutil.which(str(path)) or path).resolve())
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args.output.mkdir(parents=True, exist_ok=True)
    original = args.input.read_text()
    results = {'input_sha256': digest(args.input), 'tools': {
        str(p): digest(p) for p in (args.optimizer, args.trace_tool)},
        'reference': 'Original operand/layout intervals; scratch is a supplied enclosure', 'cases': []}
    # The original full four-visit kernel exercises scalar cyclic rules inside
    # the nested family inventory. Short loops and a skipped subblock test the
    # surrounding realization independently, without changing allocation policy.
    for count, subblock in ((0, 0), (1, 0), (2, 0), (4, 0), (4, 1)):
        text = original.replace('pto.get_block_idx', 'arith.constant 0 : i64')
        text = text.replace('pto.get_subblock_idx', f'arith.constant {subblock} : i64')
        if count != 4:
            text = text.replace('    %c0 =',
                f'    %test_limit = arith.constant {count} : index\n    %c0 =', 1)
            text = text.replace('to %c4 step', 'to %test_limit step')
        items, required, _, scalar = independent(text, 'vector', count)
        for alias in ('may-not-alias', 'may-alias'):
            folder = args.output / f'trips{count}-subblock{subblock}' / alias
            folder.mkdir(parents=True, exist_ok=True)
            source = folder / 'input.pto'
            source.write_text(text)
            config = SimpleNamespace(optimizer=args.optimizer, timeout=35)
            # The may-alias export gap is intentionally not weakened. Its
            # supported conservative policy remains an explicit request.
            policy = '' if alias == 'may-not-alias' else ' completion-only=true'
            logical = compile_stage(config, folder, 'logical', source,
                '--pto-frontier-analysis=gm-alias=' + alias + policy)
            row = {'trips': count, 'subblock': subblock, 'alias': alias,
                   'policy': 'production' if not policy else 'explicit-completion', 'logical': logical}
            results['cases'].append(row)
            assert logical['status'] == 'success', logical
            physical = compile_stage(config, folder, 'physical', Path(logical['output']),
                '--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5')
            row['physical'] = physical
            assert physical['status'] == 'success', physical
            word = trace(args, folder, Path(logical['output']))
            machine = trace(args, folder, Path(physical['output']))
            actual = [e for e in word['events'] if e['kind'] == 'payload']
            assert [(e['type'], e['coordinates'], e['pipe']) for e in actual] == [
                (e['phase'], e['coordinates'], e['pipe']) for e in items]
            physical_check({'accepted': True, 'allocated': True,
                            'trace': word, 'physical': machine}, set(range(6)))
            reach, _ = command_order(word, scalar)
            assert not any(row >> i & 1 for i, row in enumerate(reach)), 'Command cycle'
            wanted, got = pairs(required), pairs(reach)
            assert not wanted - got, sorted(wanted - got)[:5]
            # All-root possible overlap is a selected upper model, not a
            # comparison of offsets relative to unrelated pointer names.
            upper, _ = reference(items, scalar, possible_alias=alias == 'may-alias')
            assert not pairs(upper) - got, sorted(pairs(upper) - got)[:5]
            row.update(payloads=len(items), commands=len(word['events'])-len(items),
                       matching='passed', causal_reuse='passed', missing_event_pairs=0,
                       excess_ci=sum(a % 2 == 1 and b % 2 == 0 for a, b in got-wanted))
            (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print('production tuple mapping: ten original-layout guard/trip/alias cases passed')


if __name__ == '__main__':
    main()
