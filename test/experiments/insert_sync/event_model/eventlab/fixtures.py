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
"""Semantic fixtures. These are reductions, not claimed captures of native kernels."""
from copy import deepcopy

def access(space,address,size=1,**kw):return dict(space=space,address=address,size=size,**kw)
def op(name,lane,indices,domain,time,reads=(),writes=()):
    return dict(id=name,lane=lane,iterators=indices,domain=domain,schedule=time,
                reads=list(reads),writes=list(writes))
def model(name,params,context,ops,scenarios):
    return dict(name=name,parameters=params,context=context,statements=ops,scenarios=scenarios,
                provenance='Hand-authored physical-facts reduction; not a native PTO benchmark capture')

def ring(slots=2,outer=False):
    indices=['o','i'] if outer else ['i']
    domain='0<=o<O and 0<=i<N' if outer else '0<=i<N'
    t=['o','i'] if outer else ['i']
    iteration='(32*o+i)' if outer else 'i'
    return model(f'ring_{slots}'+('_nested_reset' if outer else ''),['O','N'] if outer else ['N'],
                 '0<=O<=8 and 0<=N<=8' if outer else '0<=N<=1000000',[
        op('L','MTE2',indices,domain,t+['0'],[access('GM',f'16*{iteration}',2)],
           [access('VEC',f'4*(i%{slots})',2)]),
        op('C','V',indices,domain,t+['1'],[access('VEC',f'4*(i%{slots})',2)],
           [access('VEC',f'64+4*(i%{slots})',2)]),
        op('S','MTE3',indices,domain,t+['2'],[access('VEC',f'64+4*(i%{slots})',2)],
           [access('GM',f'100000000+16*{iteration}',2)])],
        [{'O':o,'N':n} for o,n in [(0,3),(1,0),(2,1),(2,3)]] if outer else
        [{'N':n} for n in [0,1,2,3,slots+2]])

def preload():
    return model('independent_preload_boundaries',['J','K'],'0<=J<=8 and 0<=K<=8',[
        op('A','MTE2',[],'true',['0','0'],writes=[access('MAT','0',4)]),
        op('B','MTE2',[],'true',['1','0'],writes=[access('MAT','16',4)]),
        op('UseA','MTE1',['j'],'0<=j<J',['2','j'],reads=[access('MAT','0',4)]),
        op('UseB','MTE1',['j'],'0<=j<K',['3','j'],reads=[access('MAT','16',4)])],
        [{'J':j,'K':k} for j,k in [(0,0),(0,2),(2,0),(1,1),(2,3)]])

def panel():
    return model('produced_panel_with_empty_or_repeated_readers',['O','J'],
                 '0<=O<=1000000 and 0<=J<=1000000',[
        op('Load','MTE2',['o'],'0<=o<O',['o','0','0'],writes=[access('MAT','0',4)]),
        op('Extract','MTE1',['o','j'],'0<=o<O and 0<=j<J',['o','1','j'],reads=[access('MAT','0',4)]),
        # Independent work MUST NOT be included in the panel release frontier.
        op('Independent','MTE1',['o'],'0<=o<O',['o','2','0'],writes=[access('LEFT','8*o',2)])],
        [{'O':o,'J':j} for o,j in [(0,0),(1,0),(2,0),(2,1),(2,3)]])

def bundle(early=False):
    rows=[op('Left','MTE1',[],'true',['0'],writes=[access('LEFT','0',4)])]
    if early: rows += [op('Early','M',[],'true',['1'],reads=[access('LEFT','0',4)])]
    rows += [op('Right','MTE1',[],'true',['2'],writes=[access('RIGHT','0',4)]),
             op('Matmul','M',[],'true',['3'],reads=[access('LEFT','0',4),access('RIGHT','0',4)])]
    return model('compatible_operand_bundle' if not early else 'bundle_with_early_reader',[], 'true',rows,[{}])

def accumulator():
    return model('accumulator_updates_are_not_replacements',['O','K'],'0<=O<=8 and 0<=K<=8',[
        op('Init','M',['o'],'0<=o<O',['o','0','0'],writes=[access('ACC','0',4)]),
        op('Update','M',['o','k'],'0<=o<O and 0<=k<K',['o','1','k'],
           reads=[access('ACC','0',4)],writes=[access('ACC','0',4)]),
        op('Store','FIX',['o'],'0<=o<O',['o','2','0'],reads=[access('ACC','0',4)],
           writes=[access('GM','32*o',4)])],
        [{'O':o,'K':k} for o,k in [(0,2),(2,0),(1,1),(2,2)]])

def partial(may=False):
    return model('partial_writes' if not may else 'may_write_keeps_preceding_definition',[], 'true',[
        op('Low','MTE2',[],'true',['0'],writes=[access('MAT','0',4)]),
        op('ReadLow','MTE1',[],'true',['1'],reads=[access('MAT','0',4)]),
        op('High','MTE2',[],'true',['2'],writes=[access('MAT','2' if may else '4',4,definite=not may)]),
        op('ReadAll','MTE1',[],'true',['3'],reads=[access('MAT','0',6 if may else 8)])],[{}])

def conditional():
    return model('guarded_reader_and_next_overwrite',['N','TAKE'],'0<=N<=8 and 0<=TAKE<=1',[
        op('P','MTE2',['i'],'0<=i<N',['i','0'],writes=[access('MAT','0',4)]),
        op('R','MTE1',['i'],'0<=i<N and TAKE=1 and i%2=0',['i','1'],reads=[access('MAT','0',4)])],
        [{'N':n,'TAKE':t} for n,t in [(0,0),(3,0),(3,1),(4,1)]])

def parity(form='i%2=0'):
    return model('parity_'+str(abs(sum(map(ord,form)))),['N'],'0<=N<=8',[
        op('P','MTE2',['i'],'0<=i<N',['i','0'],writes=[access('MAT','8*(i%2)',4)]),
        op('Even','MTE1',['i'],'0<=i<N and ('+form+')',['i','1'],reads=[access('MAT','8*(i%2)',4)]),
        op('Odd','MTE1',['i'],'0<=i<N and not ('+form+')',['i','2'],reads=[access('MAT','8*(i%2)',4)])],
        [{'N':n} for n in [0,1,3,4]])

def all_models():
    return [ring(k) for k in [1,2,3,4]]+[ring(2,True),preload(),panel(),bundle(),bundle(True),
            accumulator(),partial(),partial(True),conditional(),parity()]
