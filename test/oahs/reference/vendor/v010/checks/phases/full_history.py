"""Independent adjacency-set/full-history oracle for the supplied phase contract.

Shares the typed operation descriptions, not live-interface transfers, closure,
projection, antichains or the paired-signature monitor. Source prefixes expand to
ALL earlier command finishes, and hazards to ALL earlier conflicting accesses.
"""
from __future__ import annotations
from copy import deepcopy
from phase_interface import Profile, Action, Rejected

class Graph:
    def __init__(self):
        self.edges=[set()];self.cache={}
    def node(self):
        self.edges.append(set());return len(self.edges)-1
    def add(self,s,t):
        self.edges[s].add(t);self.cache.clear()
    def descendants(self,s):
        if s not in self.cache:
            seen={s};stack=[s]
            while stack:
                v=stack.pop()
                for w in self.edges[v]:
                    if w not in seen:seen.add(w);stack.append(w)
            self.cache[s]=seen
        return self.cache[s]
    def reach(self,s,t):return t in self.descendants(s)

class FullHistory:
    def __init__(self,profile:Profile,exact=False):
        self.profile=profile;self.exact=exact
        self.a=Graph();self.r=Graph()
        self.ag={p:0 for p in profile.engines};self.rg=dict(self.ag)
        self.finishes={p:[] for p in profile.engines}
        self.pub={};self.consume={k:0 for k in profile.keys}
        self.permission={b:0 for b in profile.blocks};self.readable=set()
        self.history=[] # cell, mode, actual end, reference end
        self.observations=[] # same original event, actual id, reference id
        self.commands=0;self.pair_queries=0

    def transfer(self,cmd:Action,adversary_edges=(),adversary_internal=()):
        state=deepcopy(self)
        state._step(cmd,adversary_edges,adversary_internal)
        return state

    def _step(self,cmd,adversary_edges,adversary_internal):
        frag=self.profile.fragment(cmd)
        if cmd.kind=='nop':return
        p=cmd.engine;k=cmd.kind
        aa={name:self.a.node() for name in frag.nodes}
        for s,t in frag.edges:self.a.add(aa[s],aa[t])
        self.a.add(self.ag[p],aa['I'])
        if k in ('set','fence'):
            for finish in self.finishes[p]:self.a.add(finish,aa['C'])
        if k in ('set','wait'):
            if k=='set':
                if cmd.key in self.pub:raise Rejected('occupancy','occupied')
            else:
                if cmd.key not in self.pub:raise Rejected('matching','absent')
                self.a.add(self.pub[cmd.key],aa['C'])
        for ac in frag.accesses:
            if not ac.permission:continue
            if (ac.cell in self.readable)!=(ac.permission=='read'):
                raise Rejected('resource_balance','wrong role')
            self.a.add(self.permission[ac.cell],aa[ac.begin])
        for source,target in adversary_edges:
            prefix,name=source.split(':',1)
            if prefix=='T':
                for f in self.finishes[name]:self.a.add(f,aa[target])
            elif prefix=='A':self.a.add(self.ag[name],aa[target])
            elif prefix=='U':self.a.add(self.permission[name],aa[target])
            elif prefix=='D':self.a.add(self.consume[name],aa[target])
            elif prefix=='S':self.a.add(self.pub[name],aa[target])
            else:raise ValueError(prefix)
        for source,target in adversary_internal:self.a.add(aa[source],aa[target])
        if k=='set' and not self.a.reach(self.consume[cmd.key],aa['C']):
            raise Rejected('rearm','missing physical rearm path')
        prior=[]
        for ac in frag.accesses:
            for cell,mode,end,_ in self.history:
                if cell==ac.cell and ('W' in mode or 'W' in ac.mode) and not self.a.reach(end,aa[ac.begin]):
                    raise Rejected('payload','old conflict')
            for old in prior:
                if old.cell==ac.cell and ('W' in old.mode or 'W' in ac.mode) and not self.a.reach(aa[old.end],aa[ac.begin]):
                    raise Rejected('payload','within-fragment conflict')
            prior.append(ac)
        rr={}
        if frag.observable:
            rr={name:self.r.node() for name in frag.nodes}
            for s,t in frag.edges:self.r.add(rr[s],rr[t])
            self.r.add(self.rg[p],rr['I'])
            previous=list(self.history)
            for ac in frag.accesses:
                for cell,mode,_,end in previous:
                    if cell==ac.cell and ('W' in mode or 'W' in ac.mode):self.r.add(end,rr[ac.begin])
                previous.append((ac.cell,ac.mode,aa[ac.end],rr[ac.end]))
            if self.exact:
                for _,av,rv in self.observations:
                    for label in frag.observable:
                        self.pair_queries+=1
                        if self.a.reach(av,aa[label]) and not self.r.reach(rv,rr[label]):
                            raise Rejected('extra_order','old observable gets excess ordering')
                for u in frag.observable:
                    for v in frag.observable:
                        if u==v:continue
                        self.pair_queries+=1
                        if self.a.reach(aa[u],aa[v]) and not self.r.reach(rr[u],rr[v]):
                            raise Rejected('extra_order','fresh internal excess ordering')
            self.rg[p]=rr['I']
            self.observations += [(f'{self.commands}:{label}',aa[label],rr[label]) for label in frag.observable]
            self.history += [(ac.cell,ac.mode,aa[ac.end],rr[ac.end]) for ac in frag.accesses]
        if k=='set':self.pub[cmd.key]=aa['C']
        if k=='wait':del self.pub[cmd.key];self.consume[cmd.key]=aa['C']
        for ac in frag.accesses:
            if ac.permission:
                self.permission[ac.cell]=aa[ac.end]
                if ac.permission=='write':self.readable.add(ac.cell)
                else:self.readable.remove(ac.cell)
        self.ag[p]=aa['C'] if k in ('wait','fence') else aa['I']
        self.finishes[p].append(aa['C']);self.commands+=1

    def check_exit(self):
        if self.pub:raise Rejected('exit_event','live keys')
        if self.readable:raise Rejected('exit_resource','unconsumed blocks')
