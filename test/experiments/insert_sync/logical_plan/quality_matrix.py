# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Controlled scalar-contract/hardware matrix with concrete endpoint generations.

Replay establishes concrete event participation and prefix boundaries, not
numerical device correctness or an independent proof of asynchronous reuse.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

from benchmark_buffers import demand_population, digest, dump, invoke, provenance
from observations import SERIAL_DRIVER, analyze
from measure import SYNC, attrs, children, replay
from compare_boundaries import Boundaries, run as checked_boundaries


def endpoint_replay(path, scenarios):
    from ptoas.mlir import ir
    from ptoas.mlir.dialects import pto
    with ir.Context() as context:
        context.enable_multithreading(False)
        pto.register_dialect(context, load=True)
        module = ir.Module.parse(path.read_text())
        functions = [op for op in children(module.operation) if op.name == 'func.func']
        if len(functions) != 1:
            raise ValueError('endpoint replay requires exactly one function')
        sites, inventory = {}, []

        def walk(op, owners=()):
            if op.name in SYNC:
                sites[op] = len(inventory)
                inventory.append(dict(site=len(inventory), operation=str(op), properties=attrs(op),
                                      owners=list(owners)))
            for ri, region in enumerate(op.regions):
                for block in region.blocks:
                    owner = dict(operation=op.name, region=ri, operands=[str(v) for v in op.operands])
                    for view in block.operations:
                        walk(view.operation, owners + (owner,))
        walk(functions[0])
        executions = []
        for scenario in scenarios:
            # Reuse the observer's fail-closed helper and physical-context
            # qualification before recording detailed event generations.
            checked_boundaries(path, scenario)
            boundaries = Boundaries()
            generations, live, events = Counter(), {}, []
            receipts, snapshots, unproven_rearms = {}, {}, []

            def join(left, right):
                for key, generation in right.items():
                    left[key] = max(left.get(key, 0), generation)

            def observe(op, point, signature):
                properties = attrs(op)
                if op.name in ('pto.set_flag', 'pto.wait_flag'):
                    key = tuple(properties[k] for k in ('src_pipe', 'dst_pipe', 'event_id'))
                    if op.name == 'pto.set_flag':
                        if key in live:
                            raise ValueError('publication before preceding generation consumed')
                        known = receipts.setdefault(key[0], {})
                        if generations[key] > known.get(key, 0):
                            unproven_rearms.append(dict(site=sites[op], key=key,
                                                       previous_generation=generations[key]))
                        snapshots[key] = dict(known)
                        generations[key] += 1
                        live[key] = generations[key]
                    elif key not in live:
                        raise ValueError('consumption without a participating publication')
                    events.append(dict(site=sites[op], operation=op.name, key=key,
                                       generation=live[key], before_physical=len(boundaries.payload)))
                    if op.name == 'pto.wait_flag':
                        known = receipts.setdefault(key[1], {})
                        join(known, snapshots.pop(key))
                        known[key] = live[key]
                        del live[key]
                elif op.name == 'pto.barrier' and 'PIPE_ALL' in properties['pipe']:
                    retired_receipts = {}
                    for known in receipts.values():
                        join(retired_receipts, known)
                    for known in receipts.values():
                        join(known, retired_receipts)
                boundaries.observe(op, point, signature)
            metrics = replay(functions[0], scenario['arguments'], scenario.get('block_idx', 0),
                             scenario.get('block_num', 1), observer=observe)
            if live or boundaries.tokens:
                raise ValueError('unconsumed event generation at return')
            retired = all(boundaries.drained.get(lane, -1) >= point
                          for lane, point in boundaries.issued.items())
            executions.append(dict(scenario=scenario, metrics=metrics, events=events,
                                   all_generations_consumed=True, final_payload_retired=retired,
                                   consumption_before_rearm_proven=not unproven_rearms,
                                   unproven_rearms=unproven_rearms,
                                   handoffs=boundaries.handoffs, payload=boundaries.payload,
                                   before=boundaries.before))
        return dict(inventory=inventory, executions=executions,
                    scope='Concrete scalar participation and prefix observation; asynchronous reuse needs emitted verification')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-root', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.python_root.resolve()))
    args.output.mkdir(parents=True, exist_ok=False)
    case = next(c for c in demand_population() if c['case_id'] == 'historical_gemm')
    raw = case['source'].read_text()
    contract = json.loads(args.contract.read_text())
    scalars = []
    for argument, bounds in sorted(contract['arguments'].items(), key=lambda pair: int(pair[0])):
        multiple = int(bounds['multiple'])
        scalars.extend((int(argument), int(bounds['min']), int(bounds['max']) // multiple * multiple, multiple))
    metadata = 'pto.scalar_argument_preconditions = array<i64: ' + ', '.join(map(str, scalars)) + '>'
    marker = 'pto.noalias_pairs = array<i64: 0, 1, 0, 2, 1, 2>'
    if raw.count(marker) != 1 or 'pto.scalar_argument_preconditions' in raw:
        raise ValueError('unexpected scalar contract boundary')
    qualified = args.output / 'abi-qualified.pto'
    qualified.write_text(raw.replace(marker, marker + ', ' + metadata, 1))
    runner_hash = digest(__file__)
    observers = {name: digest(Path(__file__).with_name(name)) for name in
                 ('measure.py', 'observations.py', 'compare_boundaries.py')}
    result = dict(provenance=provenance(args.python_root.resolve()), matrix_runner_sha256=runner_hash,
                  observer_sha256=observers, source_sha256=digest(case['source']),
                  contract_sha256=digest(args.contract), scalar_preconditions=scalars,
                  alias=case['gm_contract'], scenarios=case['scenarios'], rows=[], device='NOT_RUN')
    env = dict(os.environ, PTOAS_LOGICAL_TRACE='1', PTOAS_COMPOSITION_WITNESSES='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    reference = None
    for source_name, source in (('original', case['source']), ('qualified', qualified)):
        for planner, hardware in (('composition', 'conservative'), ('composition', 'a2a3-mmad-acc-v1'),
                                  ('existing', 'conservative')):
            name = source_name + '-' + planner + '-' + hardware
            output = args.output / (name + '.pto')
            command = [sys.executable, '-c', SERIAL_DRIVER, str(args.python_root.resolve()),
                       '--pto-arch=a3', '--pto-level=level3', '--enable-insert-sync',
                       '--insert-sync-planner=' + planner, '--insert-sync-gm-alias=' + case['gm_contract'],
                       '--insert-sync-structured-precision=true', '--insert-sync-logical-work-budget=0',
                       '--insert-sync-hardware-contract=' + hardware,
                       '--emit-pto-ir', str(source.resolve()), '-o', str(output.resolve())]
            row = invoke(command, args.output / name, env, 120)
            row.update(name=name, input_sha256=digest(source))
            if row['returncode'] != 0:
                raise RuntimeError('controlled matrix arm refused: ' + name)
            report = analyze(output)
            projection = json.loads(json.dumps({k: report[k] for k in ('payload', 'allocations', 'views', 'abi')}))
            for record in projection['payload'] + projection['abi']:
                properties = record.get('attrs', record)
                properties.pop('pto.scalar_argument_preconditions', None)
            if reference is None:
                reference = projection
            if projection != reference:
                raise ValueError('payload/assignments/ABI changed beyond scalar preconditions')
            row.update(mechanisms=report['mechanisms'], guards=report['sync_control'],
                       keys=report['event_ids_by_direction'], status_attributes=report['status_attributes'],
                       output_sha256=digest(output))
            witness = [json.loads(line.removeprefix('OAHS_WITNESS '))
                       for line in Path(row['stderr']).read_text().splitlines() if line.startswith('OAHS_WITNESS ')]
            dump(args.output / (name + '.witness.json'), witness)
            endpoints = endpoint_replay(output, case['scenarios'])
            dump(args.output / (name + '.endpoints.json'), endpoints)
            row['families'] = [{k: v for k, v in w.items() if k.startswith('families_') or 'rejection' in k}
                               for w in witness]
            result['rows'].append(row)
            dump(args.output / 'summary.json', result)
            print(name, row['mechanisms'], row['families'], flush=True)
    if (provenance(args.python_root.resolve()) != result['provenance'] or digest(__file__) != runner_hash or
            any(digest(Path(__file__).with_name(name)) != value for name, value in observers.items())):
        raise ValueError('source or binary changed during matrix')


if __name__ == '__main__':
    main()
