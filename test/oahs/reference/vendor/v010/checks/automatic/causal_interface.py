#!/usr/bin/env python3
"""Finite live-port reference analysis for the issue-only prefix-event model.

This is a checker, NOT a complete synthesis implementation or hardware model.
It processes original CFG sites to a fixed point; it never unrolls loop visits.
Historical payload completions are represented by antichains of signatures
against live boundary ports. Exact for the graph contract and a fixed path;
control abstraction may overapproximate paths. Full disjunction is retained.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from typing import Iterable
from pathlib import Path
import json

@dataclass(frozen=True)
class Command:
    kind: str                 # op, set, wait, fence, nop
    engine: str = ''
    reads: tuple[str,...] = ()
    writes: tuple[str,...] = ()
    key: str = ''
    label: str = ''

@dataclass(frozen=True)
class State:
    reach: tuple[int,...]       # reflexive reachability between named ports
    live: int                  # physical event occupancy in reference traversal
    history: tuple[tuple[int,...], ...] # inclusion-minimal completion signatures

class Rejected(Exception):
    def __init__(self, kind: str, detail: str):
        self.kind, self.detail = kind, detail
        super().__init__(f'{kind}: {detail}')


def antichain(values: Iterable[int]) -> tuple[int,...]:
    """Keep hardest-to-cover signatures, i.e. inclusion-minimal ones."""
    kept: list[int] = []
    for value in sorted(set(values),key=lambda n:(n.bit_count(),n)):
        if not any((old & value)==old for old in kept):
            kept.append(value)
    return tuple(sorted(kept))


def close(rows: list[int]) -> list[int]:
    # Warshall with bit rows; includes the diagonal for reflexive queries.
    for k in range(len(rows)):
        bit=1<<k; successors=rows[k]
        for i in range(len(rows)):
            if rows[i]&bit: rows[i] |= successors
    return rows


class Interface:
    def __init__(self, engines: Iterable[str], cells: Iterable[str],
                 keys: dict[str,tuple[str,str]]):
        self.engines=tuple(engines); self.cells=tuple(cells); self.keys=dict(keys)
        if len(set(self.engines))!=len(self.engines): raise ValueError('duplicate engine')
        self.key_names=tuple(keys); self.ki={k:i for i,k in enumerate(keys)}
        self.ports=tuple([f'A:{p}' for p in self.engines]+[f'T:{p}' for p in self.engines]
                        +[f'S:{k}' for k in keys]+[f'D:{k}' for k in keys])
        self.pi={x:i for i,x in enumerate(self.ports)}; self.m=len(self.ports)
        self.classes=tuple((c,p,mode) for c in self.cells for p in self.engines for mode in ('R','W'))
        self.ci={c:i for i,c in enumerate(self.classes)}
        for src,dst in keys.values():
            if src not in self.engines or dst not in self.engines: raise ValueError('unknown endpoint')

    def initial(self) -> State:
        # Initial launch gates, prefix nodes, and consumed-generation anchors
        # all designate the initial root. Publication ports are absent.
        active=[i for i,p in enumerate(self.ports) if not p.startswith('S:')]
        mask=sum(1<<i for i in active)
        return State(tuple(mask if i in active else 0 for i in range(self.m)),0,
                     tuple(() for _ in self.classes))

    def transfer(self,state:State,cmd:Command,check:bool=True) -> State:
        if cmd.kind=='nop': return state
        if cmd.kind not in ('op','set','wait','fence'): raise ValueError(cmd.kind)
        p=cmd.engine
        if p not in self.engines: raise ValueError(f'unknown engine {p}')
        if any(c not in self.cells for c in cmd.reads+cmd.writes): raise ValueError('unknown cell')
        a=self.pi['A:'+p]; t=self.pi['T:'+p]
        if cmd.kind=='op' and check:
            rr,ww=set(cmd.reads),set(cmd.writes)
            for index,(cell,src,mode) in enumerate(self.classes):
                conflict=(cell in ww) or (cell in rr and mode=='W')
                if conflict and any(not (sig&(1<<a)) for sig in state.history[index]):
                    raise Rejected('payload',f'{cmd.label or p}: pending {mode} on {cell} from {src}')
        if cmd.kind in ('set','wait'):
            if cmd.key not in self.keys: raise ValueError('unknown key')
            src,dst=self.keys[cmd.key]
            if p!=(src if cmd.kind=='set' else dst): raise ValueError('wrong event endpoint')
            keybit=1<<self.ki[cmd.key]
            s=self.pi['S:'+cmd.key]; d=self.pi['D:'+cmd.key]
            if cmd.kind=='set' and state.live&keybit:
                raise Rejected('occupancy',f'publish over unconsumed {cmd.key}')
            if cmd.kind=='wait' and not state.live&keybit:
                raise Rejected('matching',f'acquire absent {cmd.key}')

        # Fresh launch i, finish f, and aggregate tnew. Edges never point
        # backward into forgotten history. Old reachability is already closed.
        n=self.m; i,f,tnew=n,n+1,n+2
        rows=list(state.reach)+[1<<j for j in (i,f,tnew)]
        rows[a] |= 1<<i
        rows[i] |= 1<<f
        if cmd.kind in ('set','fence'): rows[t] |= 1<<f
        if cmd.kind=='wait': rows[s] |= 1<<f
        rows[t] |= 1<<tnew; rows[f] |= 1<<tnew
        close(rows)
        if cmd.kind=='set' and check and not (rows[d]&(1<<f)):
            raise Rejected('rearm',f'latest consumption of {cmd.key} does not precede publication')

        mapping=list(range(n)); mapping[a]=f if cmd.kind in ('wait','fence') else i
        mapping[t]=tnew; live=state.live
        if cmd.kind=='set':
            mapping[s]=f; live |= keybit
        elif cmd.kind=='wait':
            mapping[s]=-1; mapping[d]=f; live &= ~keybit
        # Unused S ports remain absent, including on first encounters.
        for k,j in self.ki.items():
            if not live&(1<<j): mapping[self.pi['S:'+k]]=-1

        def project_reach(node:int) -> int:
            return sum(1<<q for q,v in enumerate(mapping) if v>=0 and rows[node]&(1<<v))
        newreach=tuple(project_reach(v) if v>=0 else 0 for v in mapping)
        def update_signature(signature:int)->int:
            reached=0
            for q in range(n):
                if signature&(1<<q): reached |= rows[q]
            return sum(1<<q for q,v in enumerate(mapping) if v>=0 and reached&(1<<v))
        history=[antichain(update_signature(sig) for sig in bucket) for bucket in state.history]
        if cmd.kind=='op':
            fsig=project_reach(f)
            for c in cmd.reads:
                idx=self.ci[c,p,'R']; history[idx]=antichain((*history[idx],fsig))
            for c in cmd.writes:
                idx=self.ci[c,p,'W']; history[idx]=antichain((*history[idx],fsig))
        return State(newreach,live,tuple(history))

    def payload_residual(self,state:State,cmd:Command) -> list[tuple[str,str,str]]:
        a=self.pi['A:'+cmd.engine]; rr,ww=set(cmd.reads),set(cmd.writes)
        return [(c,p,mode) for i,(c,p,mode) in enumerate(self.classes)
                if (c in ww or (c in rr and mode=='W'))
                and any(not sig&(1<<a) for sig in state.history[i])]

@dataclass(frozen=True)
class Site:
    command: Command
    successors: tuple[int,...]
    label: str=''

@dataclass
class CFG:
    sites: list[Site]
    entry: int
    exits: set[int]


def analyze(cfg:CFG,model:Interface,max_states:int=200000,closed_exit:bool=True,certificate_path:Path|None=None) -> dict:
    """Collecting fixed point of finite interface states over original sites."""
    initial=model.initial(); seen: list[set[State]]=[set() for _ in cfg.sites]
    queue=deque([(cfg.entry,initial)]); seen[cfg.entry].add(initial)
    previous={(cfg.entry,initial):None}; visits=0; max_width=0; exit_states=set()
    while queue:
        loc,state=queue.popleft(); visits+=1
        max_width=max(max_width,max((len(v) for v in state.history),default=0))
        try:
            if loc in cfg.exits and closed_exit and state.live:
                raise Rejected('exit',f'live keys at {cfg.sites[loc].label}')
            out=model.transfer(state,cfg.sites[loc].command)
        except Rejected as error:
            path=[]; cursor=(loc,state)
            while cursor is not None:
                at,_=cursor; path.append(cfg.sites[at].label or str(at)); cursor=previous[cursor]
            return {'accepted':False,'kind':error.kind,'reason':error.detail,
                    'static_sites':len(cfg.sites),'reachable_states':sum(map(len,seen)),
                    'processed':visits,'witness':path[::-1],'max_signature_antichain':max_width}
        if loc in cfg.exits: exit_states.add(out)
        for target in cfg.sites[loc].successors:
            if out not in seen[target]:
                seen[target].add(out); previous[target,out]=(loc,state); queue.append((target,out))
                if sum(map(len,seen))>max_states:
                    return {'accepted':False,'kind':'budget','static_sites':len(cfg.sites),
                            'reachable_states':sum(map(len,seen)),'processed':visits}
    if certificate_path is not None:
        cert={'schema':'oahs.live_interface_invariant.v1',
              'engines':model.engines,'cells':model.cells,'keys':model.keys,
              'cfg':{'entry':cfg.entry,'exits':sorted(cfg.exits),
                     'sites':[asdict(site) for site in cfg.sites]},
              'closed_exit':closed_exit,
              'invariant':[[asdict(s) for s in sorted(bucket,key=lambda s:(s.live,s.reach,s.history))]
                           for bucket in seen]}
        certificate_path.parent.mkdir(parents=True,exist_ok=True)
        certificate_path.write_text(json.dumps(cert,indent=2)+'\n')
    return {'accepted':True,'static_sites':len(cfg.sites),'reachable_states':sum(map(len,seen)),
            'processed':visits,'max_states_per_site':max(map(len,seen)),
            'exit_states':len(exit_states),'max_signature_antichain':max_width,
            'trip_count_bound':None,'method':'finite live-interface fixed point; no trace unrolling'}


class Builder:
    def __init__(self): self.sites=[]
    def node(self,command:Command|None=None,successors=(),label=''):
        idx=len(self.sites); self.sites.append(Site(command or Command('nop'),tuple(successors),label)); return idx
    def connect(self,idx,successors):
        old=self.sites[idx]; self.sites[idx]=Site(old.command,tuple(successors),old.label)
    def sequence(self,commands:list[Command],continuation:int,label=''):
        cur=continuation
        for j,c in reversed(list(enumerate(commands))): cur=self.node(c,(cur,),f'{label}{j}:{c.label or c.kind}')
        return cur
