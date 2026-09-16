"""Supplied payload/resource profiles and programs; no native machine code."""
from phase_interface import *


def profile(blocks=2):
    cells=[Cell('left','LEFT',0,512),Cell('right','RIGHT',0,512),Cell('side','LOCAL',0,512)]
    cells += [Cell(f'u{j}','ACC',j*512,512) for j in range(blocks)]
    cells += [Cell(f'g{j}','GM',j*512,512) for j in range(blocks)]
    return Profile(('L','M','F','Q'),cells,[Group('acc',tuple(f'u{j}' for j in range(blocks)))],
                   {'lm':('L','M'),'ml':('M','L'),'mf':('M','F'),'fm':('F','M'),
                    'fq':('F','Q'),'qf':('Q','F'),'qm':('Q','M'),'mq':('M','Q')})


def producer(p, label='matrix'):
    return Action('produce','M',reads=('left','right'),group='acc',blocks=p.groups['acc'].blocks,label=label)


def consumer(p, label='output'):
    return Action('consume','F',writes=tuple(f'g{j}' for j in range(len(p.blocks))),
                  group='acc',blocks=p.groups['acc'].blocks,label=label)


def event(p,key,kind):
    engine=p.keys[key][0 if kind=='set' else 1]
    return Action(kind,engine,key=key,label=f'{kind}_{key}')


def handshake(p,key):
    return [event(p,key,'set'),event(p,key,'wait')]


def linear(actions):
    nodes=[Node(a,(i+1,)) for i,a in enumerate(actions)]
    nodes.append(Node(Action('nop',label='exit')))
    return Program(tuple(nodes),0,(len(nodes)-1,))


def loop(actions):
    # Zero or any finite number of body visits, no invented runtime predicates.
    nodes=[Node(Action('nop',label='loop_header'))]
    for i,a in enumerate(actions,1):nodes.append(Node(a,(i+1,)))
    end=len(nodes);nodes.append(Node(Action('nop',label='backedge'),(0,)))
    exit_=len(nodes);nodes.append(Node(Action('nop',label='exit')))
    nodes[0]=Node(nodes[0].action,(1,exit_))
    return Program(tuple(nodes),0,(exit_,))


def branched_loop(p):
    # Two equivalent source-level paths, one carrying unrelated work on Q.
    # Both return complete writable block groups. The join keeps whole states.
    actions1=[producer(p),consumer(p),Action('fence','F',label='output_reuse_fence')]
    actions2=[Action('op','Q',reads=('side',),label='unrelated_read'),producer(p),consumer(p),
              Action('fence','F',label='output_reuse_fence')]
    nodes=[Node(Action('nop',label='loop_header'))]
    starts=[]
    for aa in (actions1,actions2):
        starts.append(len(nodes))
        for a in aa:nodes.append(Node(a,(len(nodes)+1,)))
        nodes.append(Node(Action('nop',label='backedge'),(0,)))
    exit_=len(nodes);nodes.append(Node(Action('nop',label='exit')))
    nodes[0]=Node(nodes[0].action,tuple(starts)+(exit_,))
    return Program(tuple(nodes),0,(exit_,))
