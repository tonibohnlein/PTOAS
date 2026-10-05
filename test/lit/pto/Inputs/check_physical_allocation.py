# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check actual physical IDs, matching, causal reuse and allocation failure."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from check_logical_insertion import closure_with_commands, invoke, recognized, validate


def check_physical(template, report, eligible):
    assert report['allocated'] and not report['physical']['error']
    logical, physical = report['logical'], report['physical']
    validate(template, logical)
    assert physical['payloads'] == logical['payloads']
    assert len(logical['events']) == len(physical['events'])
    pipes = [e['pipe'] for e in logical['events'] if e['kind'] == 'payload']
    commands, uses = [], []
    for old, new in zip(logical['events'], physical['events']):
        if old['kind'] in ('payload', 'barrier'):
            assert old == new
        if old['kind'] == 'payload':
            continue
        assert (old['kind'], old['gap'], old['pipe']) == (new['kind'], new['gap'], new['pipe'])
        command = dict(kind=old['kind'], gap=old['gap'], pipe=old['pipe'])
        if old['kind'] != 'barrier':
            assert (old['source_pipe'], old['target_pipe']) == (new['source_pipe'], new['target_pipe'])
            command['identity'] = (old['plan'], old['record'], old['source_ordinal'])
            assert new['physical_id'] in eligible
            event = (new['source_pipe'], new['target_pipe'], new['physical_id'])
            uses.append((len(commands), new['kind'], event, command['identity']))
        commands.append(command)
    reach = closure_with_commands(pipes, commands, include_commands=True)
    live, last_consumption = {}, {}
    for index, kind, event, identity in uses:
        vertex = 2 * len(pipes) + index
        if kind == 'set':
            assert event not in live, 'publication overwrites an outstanding notification'
            if event in last_consumption:
                assert (reach[last_consumption[event]] >> vertex) & 1, 'reuse has no causal witness'
            live[event] = identity
        else:
            assert live.pop(event, None) == identity, 'WAIT consumes the wrong logical generation'
            last_consumption[event] = vertex
    assert not live, 'notifications remain at return'


def opt(tool, path, passes, success=True):
    result = subprocess.run([tool, '--mlir-disable-threading', *passes, str(path)], capture_output=True,
                            text=True, timeout=90, check=False)
    assert (result.returncode == 0) == success, result.stderr + result.stdout
    return result


def run_checks(tool, optimizer, source):
    original = source.read_text()
    allocate = '--pto-frontier-allocate=eligible-ids=1,3'
    summaries = []
    with tempfile.TemporaryDirectory(prefix='physical-allocation-') as scratch:
        path = Path(scratch) / 'case.pto'
        for upper in [-2, 1, 2, 5, 8, 13]:
            path.write_text(original.replace('array<i64: 8>', f'array<i64: {upper}>'))
            template = recognized(tool, path)
            report = json.loads(invoke(tool, '--physical-trace', path))
            check_physical(template, report, {1, 3})
            domains = {}
            for event in report['physical']['events']:
                if event['kind'] == 'set':
                    domains.setdefault((event['source_pipe'], event['target_pipe']), set()).add(event['physical_id'])
            if report['physical']['payloads'] >= 8:
                assert len(domains) == 2 and all(ids == {1, 3} for ids in domains.values())
            summaries.append(report['physical']['payloads'])
        physical = opt(optimizer, path, ['--pto-frontier-analysis', allocate]).stdout
        assert 'pto.logical_' not in physical and 'pto.cyclic_allocation' not in physical
        assert 'pto.set_flag_dyn' in physical and 'pto.wait_flag_dyn' in physical
        for name in ['scf.for', 'pto.tload', 'pto.textract']:
            assert physical.count(name) == original.count(name), 'original payload/control structure changed'
        # Mutating a real physical assignment must be caught independently.
        broken = copy.deepcopy(report)
        commands = [e for e in broken['physical']['events'] if e['kind'] in ('set', 'wait')]
        for command in commands:
            command['physical_id'] = 1
        try:
            check_physical(template, broken, {1, 3})
        except AssertionError:
            pass
        else:
            raise AssertionError('oracle accepted forced physical-ID collisions')
        # Insufficient, absent, duplicate and out-of-range IDs do not mutate IR.
        for ids in ['1', '', '1, 1', '-1, 1', '1, 6', '1, 7', '1, 8']:
            path.write_text(original.replace('array<i64: 1, 3>', f'array<i64: {ids}>' if ids else 'array<i64>'))
            report = json.loads(invoke(tool, '--physical-trace', path))
            assert not report['allocated'] and report['unchanged_on_failure']
        path.write_text(original)
        for reserved in [6, 7]:
            result = opt(optimizer, path, ['--pto-frontier-analysis',
                         f'--pto-frontier-allocate=eligible-ids=1,{reserved}'], success=False)
            assert 'reserved for system/framework use' in result.stderr
        result = opt(optimizer, path, ['--pto-frontier-analysis',
                     '--pto-frontier-allocate=eligible-ids=0,1,2,3,4,5'])
        assert 'pto.logical_' not in result.stdout
        logical = opt(optimizer, path, ['--pto-frontier-analysis']).stdout
        for changed, expected in [
            (logical.replace('version = 1 : i64', 'version = 2 : i64'), 'malformed'),
            (logical.replace('budget = 2 : i64', 'budget = -1 : i64', 1), 'no finite'),
            (logical.replace('plan 0 record', 'plan 1 record', 1), 'does not match'),
            ('\n'.join(line for line in logical.splitlines() if 'pto.logical_wait' not in line), 'missing logical'),
        ]:
            path.write_text(changed)
            result = opt(optimizer, path, [allocate], success=False)
            assert expected in result.stderr, result.stderr
        path.write_text(original)
        result = opt(optimizer, path, [allocate], success=False)
        assert 'requires a cyclic allocation certificate' in result.stderr
        # A statically empty invocation consumes no IDs and needs no commands.
        path.write_text(original.replace('to %n step', 'to %one step'))
        physical = opt(optimizer, path, ['--pto-frontier-analysis', '--pto-frontier-allocate']).stdout
        assert 'pto.logical_' not in physical and 'pto.set_flag' not in physical
        # Nested selected-arm case exercises constant IDs and local barriers.
        nested = source.with_name('sync_logical_insertion.pto').read_text()
        nested = nested.replace('test.trace_arguments =',
                                'test.eligible_ids = array<i64: 2, 5>, test.trace_arguments =')
        path.write_text(nested)
        template = recognized(tool, path)
        check_physical(template, json.loads(invoke(tool, '--physical-trace', path)), {2, 5})
        physical = opt(optimizer, path, ['--pto-frontier-analysis', '--pto-frontier-allocate=eligible-ids=2,5']).stdout
        assert 'pto.set_flag[' in physical and 'pto.wait_flag[' in physical and 'pto.barrier' in physical
    print('physical allocation: actual matching/reuse, untouched failures, cyclic/static lowering passed', summaries)


def main():
    tool, optimizer = shutil.which(sys.argv[1]), shutil.which(sys.argv[2])
    assert tool and optimizer
    run_checks(tool, optimizer, Path(sys.argv[3]))


if __name__ == '__main__':
    main()
