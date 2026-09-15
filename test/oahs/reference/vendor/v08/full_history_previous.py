#!/usr/bin/env python3
"""Independent full-history oracle versus finite signatures.

The oracle stores all historical commands and access occurrences, and direct
edges to ALL preceding command finishes. It has no signature abstraction,
port projection, or CFG fixed point. Random graph prefixes can be unsafe;
reported missing obligations/rearm failures are compared before continuing
algebraically. This does not treat an unsafe prefix as an executable protocol.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from itertools import product
import random,json
from causal_interface import *

class FullHistory:
    def __init__(self,engines,keys):
        self.engines=tuple(engines); self.keys=keys; self.anc=[1]
        self.launch={p:0 for p in engines}; self.gate={p:0 for p in engines}
        self.finishes={p:[] for p in engines}; self.live={}; self.consumed={k:0 for k in keys}
        self.accesses=[]
    def node(self,preds):
        i=len(self.anc); bits=1<<i
        for p in preds: bits |= self.anc[p]
        self.anc.append(bits); return i
    def residual(self,c):
        ancestors=self.anc[self.launch[c.engine]]|self.anc[self.gate[c.engine]]
        rr,ww=set(c.reads),set(c.writes); result=set()
        for cell,engine,mode,f in self.accesses:
            if (cell in ww or (cell in rr and mode=='W')) and not ancestors&(1<<f):
                result.add((cell,engine,mode))
        return result
    def step(self,c):
        if c.kind=='nop': return None
        error=None
        if c.kind=='op' and self.residual(c): error='payload'
        p=c.engine
        i=self.node([self.launch[p],self.gate[p]])
        self.launch[p]=i
        if c.kind=='set':
            if c.key in self.live: raise ValueError('generator published full key')
            f=self.node([i,*self.finishes[p]])
            if not self.anc[f]&(1<<self.consumed[c.key]): error='rearm'
            self.live[c.key]=f
        elif c.kind=='wait':
            if c.key not in self.live: raise ValueError('generator acquired empty key')
            f=self.node([i,self.live.pop(c.key)])
            self.consumed[c.key]=f; self.gate[p]=f
        elif c.kind=='fence':
            f=self.node([i,*self.finishes[p]]); self.gate[p]=f
        else:
            f=self.node([i])
            self.accesses.extend((cell,p,'R',f) for cell in c.reads)
            self.accesses.extend((cell,p,'W',f) for cell in c.writes)
        self.finishes[p].append(f)
        return error

