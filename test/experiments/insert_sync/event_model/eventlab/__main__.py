# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Original Event Lab reference code retains its MIT notice below and in LICENSE.
# SPDX-License-Identifier: MIT
"""Command line for the executable symbolic specification."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .islwrap import ISLError
from .symbolic import Program
from .physical import lower, allocate, verify_async
from .reuse import allocate_symbolically


def concrete_report(plan, parameters, pool=6, verify=False, states=200000, strategy="symbolic", symbolic=None):
    concrete = lower(plan, parameters)
    finite = allocate(concrete, pool)
    symbolic = symbolic if symbolic is not None else allocate_symbolically(plan, pool)
    if strategy == 'symbolic' and symbolic.accepted:
        assignment = symbolic.specialize(plan, concrete, parameters)
    elif strategy == 'finite':
        assignment = finite
    else:
        from .physical import Allocation
        assignment = Allocation(False, {}, {}, {}, 'symbolic key search did not prove this domain; use --allocation finite only as a finite oracle')
    report = {
        'parameters': parameters,
        'evidence_scope': 'finite specialization of hand-authored/exported facts; not native kernel execution',
        'mechanisms': concrete.metrics(),
        'memory_representation': 'exact_half_open_intervals',
        'operations': [dict(identity=o['identity'], lane=o['lane'], time=o['time'],
                            reads=o['reads'].report(), writes=o['writes'].report())
                       for o in concrete.operations],
        'handoffs': [asdict(h) for h in concrete.handoffs],
        'per_lane_commands': concrete.streams,
        'assignment': asdict(assignment),
        'allocation_strategy': strategy,
        'symbolic_key_proof': symbolic.report(),
        'finite_chain_cover_oracle': asdict(finite),
    }
    if verify and assignment.accepted:
        report['asynchronous_check'] = verify_async(concrete, assignment, states)
    else:
        report['asynchronous_check'] = {'status':'NOT_RUN',
            'reason':'not requested' if not verify else 'finite keys not assigned'}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','realize'])
    parser.add_argument('facts', type=Path)
    parser.add_argument('--params', nargs='*', default=[], metavar='NAME=INTEGER')
    parser.add_argument('--pool', type=int, default=6)
    parser.add_argument('--allocation', choices=['symbolic','finite'], default='symbolic')
    parser.add_argument('--verify', action='store_true', help='exhaustively explore finite interleavings')
    parser.add_argument('--states', type=int, default=200000)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.facts.read_text())
        parameters = {}
        for item in args.params:
            name, value = item.split('=', 1)
            if name in parameters: raise ValueError('duplicate parameter: '+name)
            parameters[name] = int(value)
        plan = Program(data).plan()
        result = {'input':str(args.facts),'provenance':data.get('provenance','not supplied'),
                  'symbolic_plan':plan.report(),
                  'symbolic_key_plan':allocate_symbolically(plan,args.pool).report()}
        if args.command == 'realize':
            result['concrete'] = concrete_report(plan, parameters, args.pool, args.verify, args.states, args.allocation)
        elif parameters or args.verify:
            raise ValueError('--params and --verify apply to realize, not plan')
        text=json.dumps(result, indent=2)+'\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
            print(str(args.output))
        else:
            print(text, end='')
        concrete=result.get('concrete')
        if concrete:
            if not concrete['assignment']['accepted']: return 3
            if concrete['mechanisms']['missing_memory_requirements']: return 2
            status=concrete['asynchronous_check']['status']
            if status=='FAIL':return 2
            if status=='LIMIT':return 4
        return 0
    except (ISLError, ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({'status':'UNPROVED_OR_INVALID','error':str(error)}),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
