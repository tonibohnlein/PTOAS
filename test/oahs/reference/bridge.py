#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared pinned-reference loader and independent original-control builder.

Used by the causal-frontier bridge. The superseded candidate-constructor
interchange driver and its construction campaign have been removed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor' / 'v08'
sys.path.insert(0, str(VENDOR))
from causal_interface import Command, Interface, Rejected  # noqa: E402

KINDS = {'sequence': 0, 'choice': 1, 'for': 2, 'while': 3, 'operation': 4}

def verify_vendor(root: Path = VENDOR) -> dict:
    manifest = json.loads((root / 'PROVENANCE.json').read_text())
    for name, expected in manifest['files'].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'pinned reference source/artifact changed: {name}')
    return manifest

def key_name(key):
    return ':'.join(map(str, key))

def control(case):
    """Independent construction from Region, not C++ Control.h's edge table."""
    n = len(case['operations'])
    edges = [[] for _ in range(n + 1)]
    def node():
        edges.append([])
        return len(edges)-1
    def build(r, continuation):
        kind = r['kind']
        children = r.get('children', [])
        if kind == 'operation':
            at = r['operation']
            edges[at] = [continuation]
            return at
        if kind == 'sequence':
            result = continuation
            for c in reversed(children):
                result = build(c, result)
            return result
        if kind == 'choice':
            at = node()
            edges[at] = [build(c, continuation) for c in children]
            return at
        if kind == 'for':
            header = node()
            edges[header] = [build(children[0], header), continuation]
            return header
        if kind == 'while':
            decision = node()
            before = build(children[0], decision)
            after = build(children[1], before)
            edges[decision] = [after, continuation]
            return before
        raise ValueError(kind)
    body = case.get('body', {'kind': 'sequence'})
    if body['kind'] == 'sequence' and not body.get('children'):
        for at in range(n):
            edges[at] = [at+1]
        entry = 0
    else:
        entry = build(body, n)
    return edges, entry

def op(pipe, cell=0, read=False, write=False):
    return {'pipe':pipe, 'accesses':[[cell,int(read),int(write)]] if read or write else []}

def leaf(i):return {'kind':'operation','operation':i}
def seq(*rs):return {'kind':'sequence','children':list(rs)}
def loop(r):return {'kind':'for','children':[r],'zero_trip_possible':True}
def choice(a,b):return {'kind':'choice','children':[a,b]}
