"""Finite phase/event interface for supplied final-block-pair-v1.

Standard-library reference checker; NOT a native UnitFlag driver or a synthesizer.
The supplied profile has exact, disjoint physical cells and forward-reference
alternating final producer/consumer visits. Its hardware refinement is an explicit
premise. Resource bits describe reference balance, not instantaneous hardware state.
No synchronization can be inserted at the internal analytical phase boundaries.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
from typing import Iterable
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'automatic'))
from causal_interface import close, antichain, Rejected
from order_interface import dangerous_antichain

class QualificationError(ValueError):
    pass

@dataclass(frozen=True)
class Cell:
    name: str
    domain: str
    offset: int
    size: int

@dataclass(frozen=True)
class Group:
    name: str
    blocks: tuple[str, ...]
    producer: str = 'M'
    consumer: str = 'F'

@dataclass(frozen=True)
class Action:
    kind: str                         # op, produce, consume, set, wait, fence, nop
    engine: str = ''
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    key: str = ''
    group: str = ''
    blocks: tuple[str, ...] = ()
    mode: str = 'final'
    layout: str = 'exact-block-map-v1'
    label: str = ''

@dataclass(frozen=True)
class Access:
    cell: str
    mode: str
    begin: str
    end: str
    permission: str = ''               # write or read, only for protected blocks

@dataclass(frozen=True)
class Fragment:
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    accesses: tuple[Access, ...]
    observable: tuple[str, ...]

@dataclass(frozen=True)
class State:
    reach: tuple[int, ...]
    live: int
    readable: int
    history: tuple[tuple[int, ...], ...]
    reference: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]


def projection(rows: list[int], mapping: list[int], node: int) -> int:
    return sum(1 << i for i, v in enumerate(mapping) if v >= 0 and rows[node] & (1 << v))


def expanded_image(signature: int, rows: list[int], width: int) -> int:
    result = 0
    for q in range(width):
        if signature & (1 << q):
            result |= rows[q]
    return result


def image(signature: int, rows: list[int], mapping: list[int]) -> int:
    reached = expanded_image(signature, rows, len(mapping))
    return sum(1 << i for i, v in enumerate(mapping) if v >= 0 and reached & (1 << v))


class Profile:
    """Check a supplied exact block map and a very narrow operation vocabulary.

    Reference qualification only. No reset, partial block, KEEP, accumulating
    producer, hidden observer, dynamic layout, or native machine-code inference.
    Each protected group is wholly produced and wholly consumed by one operation.
    """
    name = 'final-block-pair-v1'
    def __init__(self, engines: Iterable[str], cells: Iterable[Cell],
                 groups: Iterable[Group], keys: dict[str, tuple[str, str]],
                 *, entry_writable: bool = True):
        self.engines = tuple(engines)
        cc = tuple(cells); gg = tuple(groups)
        if not self.engines or len(set(self.engines)) != len(self.engines):
            raise QualificationError('invalid engine population')
        if len({c.name for c in cc}) != len(cc):
            raise QualificationError('duplicate cell name')
        self.cells = {c.name: c for c in cc}
        self.groups = {g.name: g for g in gg}
        if len(self.groups) != len(gg):
            raise QualificationError('duplicate resource group')
        for c in cc:
            if c.offset < 0 or c.size <= 0:
                raise QualificationError('invalid physical interval')
        for i, a in enumerate(cc):
            for b in cc[i+1:]:
                if a.domain == b.domain and max(a.offset, b.offset) < min(a.offset+a.size, b.offset+b.size):
                    raise QualificationError('unpartitioned aliases or overlapping physical cells')
        blocks = []
        for g in gg:
            if not g.blocks or len(set(g.blocks)) != len(g.blocks):
                raise QualificationError('empty/duplicate block map')
            if g.producer not in self.engines or g.consumer not in self.engines or g.producer == g.consumer:
                raise QualificationError('invalid resource participants')
            for b in g.blocks:
                c = self.cells.get(b)
                if c is None or c.domain != 'ACC' or c.offset % 512 or c.size != 512:
                    raise QualificationError('protected blocks must be exact aligned 512-byte ACC cells')
                if b in blocks:
                    raise QualificationError('protected cell belongs to multiple groups')
                blocks.append(b)
        self.blocks = tuple(blocks)
        self.keys = dict(keys)
        for k, (p, q) in self.keys.items():
            if not k or p not in self.engines or q not in self.engines or p == q:
                raise QualificationError('invalid directed event key')
        if not entry_writable:
            raise QualificationError('this profile requires an explicitly writable incoming interface')

    def fragment(self, action: Action) -> Fragment:
        k, p = action.kind, action.engine
        if k == 'nop':
            if action.engine or action.reads or action.writes or action.key or action.group or action.blocks:
                raise QualificationError('no-op has hidden effects or protocol metadata')
            return Fragment((), (), (), ())
        if p not in self.engines:
            raise QualificationError('unknown engine')
        if k in ('set', 'wait', 'fence'):
            if action.reads or action.writes or action.group or action.blocks:
                raise QualificationError('synchronization command has unexpected payload effects')
            if k == 'fence' and action.key:
                raise QualificationError('fence cannot reset or name an event key')
            if k != 'fence':
                if action.key not in self.keys:
                    raise QualificationError('unknown event key')
                expected = self.keys[action.key][0 if k == 'set' else 1]
                if p != expected:
                    raise QualificationError('wrong event endpoint')
            return Fragment(('I', 'C'), (('I', 'C'),), (), ())
        if k not in ('op', 'produce', 'consume'):
            raise QualificationError('unsupported operation or resource mode')
        if action.key:
            raise QualificationError('payload operation has unexplained private event effects')
        if len(set(action.reads)) != len(action.reads) or len(set(action.writes)) != len(action.writes):
            raise QualificationError('duplicate footprint')
        if any(c not in self.cells for c in action.reads + action.writes):
            raise QualificationError('unknown effect cell')
        if any(c in self.blocks for c in action.reads + action.writes):
            raise QualificationError('ordinary/unscoped access to a protected block')
        if k == 'op':
            if action.group or action.blocks:
                raise QualificationError('ordinary operation has resource metadata')
            accesses = tuple(Access(c, ('R' if c in action.reads else '') + ('W' if c in action.writes else ''), 'I', 'C')
                             for c in dict.fromkeys(action.reads + action.writes))
            return Fragment(('I', 'C'), (('I', 'C'),), accesses, ('I', 'C'))
        if action.mode != 'final' or action.layout != 'exact-block-map-v1':
            raise QualificationError('KEEP/reset/unknown layout or mode is outside the profile')
        if action.group not in self.groups:
            raise QualificationError('unknown resource group')
        g = self.groups[action.group]
        if action.blocks != g.blocks:
            raise QualificationError('partial, permuted, or mismatched block coverage')
        if p != (g.producer if k == 'produce' else g.consumer):
            raise QualificationError('wrong resource-role engine')
        if k == 'produce':
            if len(action.reads) != 2 or action.writes or tuple(self.cells[c].domain for c in action.reads) != ('LEFT', 'RIGHT'):
                raise QualificationError('producer requires complete LEFT/RIGHT inputs and no extra output')
        elif action.reads or len(action.writes) != len(g.blocks) or any(self.cells[c].domain != 'GM' or self.cells[c].size != 512 or self.cells[c].offset % 512 for c in action.writes):
            raise QualificationError('consumer requires one complete GM-output cell per block')
        nodes = ['I', 'C']; edges = [('I', 'C')]; accesses = []
        if k == 'produce':
            # Coarse source reads remain live through the whole instruction.
            # A ready output block does not release either operand.
            accesses += [Access(c, 'R', 'I', 'C') for c in action.reads]
        for j, b in enumerate(g.blocks):
            begin, end = f'b{j}', f'e{j}'
            nodes += [begin, end]
            edges += [('I', begin), (begin, end), (end, 'C')]
            accesses.append(Access(b, 'W' if k == 'produce' else 'R', begin, end,
                                   'write' if k == 'produce' else 'read'))
            if k == 'consume':
                outbegin, outend = f'o{j}b', f'o{j}e'
                nodes += [outbegin, outend]
                # Identity copy may stream: require the whole output block to
                # finish after the input block read, NOT that its first write
                # starts after the whole input block has been read.
                edges += [('I', outbegin), (outbegin, outend), (end, outend), (outend, 'C')]
                accesses.append(Access(action.writes[j], 'W', outbegin, outend))
        return Fragment(tuple(nodes), tuple(edges), tuple(accesses), tuple(nodes))

    def json(self):
        return {'name': self.name, 'qualification': 'reference-contract-only',
                'engines': self.engines, 'cells': [asdict(c) for c in self.cells.values()],
                'groups': [asdict(g) for g in self.groups.values()], 'keys': self.keys,
                'entry_writable': True}

    @classmethod
    def from_json(cls, obj):
        if obj['name'] != cls.name or obj.get('qualification') != 'reference-contract-only':
            raise QualificationError('unknown/unqualified profile identity')
        return cls(obj['engines'], [Cell(**c) for c in obj['cells']],
                   [Group(g['name'], tuple(g['blocks']), g['producer'], g['consumer']) for g in obj['groups']],
                   {k: tuple(v) for k, v in obj['keys'].items()}, entry_writable=obj['entry_writable'])


class PhaseInterface:
    def __init__(self, profile: Profile, exact_order: bool = False):
        self.profile = profile; self.exact_order = exact_order
        self.keys = profile.keys; self.kn = tuple(self.keys); self.ki = {k:i for i,k in enumerate(self.kn)}
        self.blocks = profile.blocks; self.bi = {b:i for i,b in enumerate(self.blocks)}
        self.ports = tuple([f'A:{p}' for p in profile.engines] + [f'T:{p}' for p in profile.engines]
                           + [f'S:{k}' for k in self.keys] + [f'D:{k}' for k in self.keys]
                           + [f'U:{b}' for b in self.blocks])
        self.pi = {p:i for i,p in enumerate(self.ports)}; self.m = len(self.ports)
        self.classes = tuple((c,p,x) for c in profile.cells for p in profile.engines for x in ('R','W'))
        self.ci = {c:i for i,c in enumerate(self.classes)}
        self.rports = tuple([f'G:{p}' for p in profile.engines] + [f'{x}:{c}' for c in profile.cells for x in ('R','W')])
        self.ri = {p:i for i,p in enumerate(self.rports)}; self.n = len(self.rports)

    def initial(self):
        active = [i for i,p in enumerate(self.ports) if not p.startswith('S:')]
        mask = sum(1<<i for i in active)
        return State(tuple(mask if i in active else 0 for i in range(self.m)), 0, 0,
                     tuple(() for _ in self.classes), tuple([(1<<self.n)-1]*self.n), ())

    def transfer(self, state: State, action: Action, *,
                 adversary_edges: tuple[tuple[str,str], ...] = (),
                 adversary_internal: tuple[tuple[str,str], ...] = ()) -> State:
        """Atomic extension/check/projection. adversary_edges are TEST-ONLY.

        Their sources name OLD actual ports and targets name fresh endpoints;
        they are never accepted by the serialized emitter or profile vocabulary.
        They test added completion/phase order that an issue-only monitor misses.
        """
        frag = self.profile.fragment(action)
        if action.kind == 'nop': return state
        p = action.engine; a = self.pi['A:'+p]; t = self.pi['T:'+p]
        rows = list(state.reach); ids = {}
        for label in frag.nodes:
            ids[label] = len(rows); rows.append(1<<len(rows))
        for u,v in frag.edges: rows[ids[u]] |= 1<<ids[v]
        i,f = ids['I'],ids['C']; rows[a] |= 1<<i
        newt = len(rows); rows.append(1<<newt)
        rows[t] |= 1<<newt; rows[f] |= 1<<newt
        mapping = list(range(self.m)); mapping[a] = i; mapping[t] = newt
        live, readable = state.live, state.readable
        k = action.kind
        if k in ('set','fence'): rows[t] |= 1<<f
        if k in ('wait','fence'): mapping[a] = f
        if k in ('set','wait'):
            s = self.pi['S:'+action.key]; d = self.pi['D:'+action.key]; bit=1<<self.ki[action.key]
            if k == 'set':
                if live & bit: raise Rejected('occupancy', 'publish over live '+action.key)
                mapping[s] = f; live |= bit
            else:
                if not live & bit: raise Rejected('matching', 'acquire absent '+action.key)
                rows[s] |= 1<<f; mapping[s] = -1; mapping[d] = f; live &= ~bit
        for ac in frag.accesses:
            if not ac.permission: continue
            bit = 1<<self.bi[ac.cell]; port = self.pi['U:'+ac.cell]
            expects_readable = ac.permission == 'read'
            if bool(readable & bit) != expects_readable:
                raise Rejected('resource_balance', f'{action.label}: unexpected {ac.permission} on {ac.cell}')
            rows[port] |= 1<<ids[ac.begin]
            mapping[port] = ids[ac.end]
            readable ^= bit
        for source, target in adversary_edges:
            rows[self.pi[source]] |= 1<<ids[target]
        for source, target in adversary_internal:
            rows[ids[source]] |= 1<<ids[target]
        close(rows)
        if k == 'set' and not rows[d] & (1<<f):
            raise Rejected('rearm', 'old event consumption does not reach new publication of '+action.key)
        for key,j in self.ki.items():
            if not live & (1<<j): mapping[self.pi['S:'+key]] = -1

        # Check all old hazards and earlier accesses IN THIS fragment before
        # inserting any effects. Multiple modes on one access are one RMW phase.
        prior=[]
        for ac in frag.accesses:
            target=ids[ac.begin]
            for idx,(cell,engine,mode) in enumerate(self.classes):
                if cell==ac.cell and ('W' in ac.mode or mode=='W'):
                    if any(not expanded_image(sig,rows,self.m)&(1<<target) for sig in state.history[idx]):
                        raise Rejected('payload', f'{action.label}/{ac.begin}: pending {mode} {cell} from {engine}')
            for old in prior:
                if ac.cell == old.cell and ('W' in ac.mode or 'W' in old.mode) and not rows[ids[old.end]]&(1<<target):
                    raise Rejected('payload', 'unordered conflicting phases within operation')
            prior.append(ac)
        history=[antichain(image(sig,rows,mapping) for sig in bucket) for bucket in state.history]
        for ac in frag.accesses:
            sig=projection(rows,mapping,ids[ac.end])
            for mode in ac.mode:
                ix=self.ci[ac.cell,p,mode];history[ix]=antichain((*history[ix],sig))

        # Independent reference graph: fixed operation-internal edges plus
        # exactly native issue and all prior memory-conflict requirements.
        rrows=list(state.reference);rmapping=list(range(self.n));rids={}
        if frag.observable:
            for label in frag.nodes:
                rids[label]=len(rrows);rrows.append(1<<len(rrows))
            for u,v in frag.edges:rrows[rids[u]] |= 1<<rids[v]
            gate=self.ri['G:'+p];rrows[gate] |= 1<<rids['I'];rmapping[gate]=rids['I']
            current={z:rmapping[z] for z in range(self.n)}
            for ac in frag.accesses:
                idx=self.ri['W:'+ac.cell];rrows[current[idx]] |= 1<<rids[ac.begin]
                if 'W' in ac.mode:
                    idx=self.ri['R:'+ac.cell];rrows[current[idx]] |= 1<<rids[ac.begin]
                for mode in ac.mode:
                    idx=self.ri[mode+':'+ac.cell];end=len(rrows);rrows.append(1<<end)
                    rrows[current[idx]] |= 1<<end;rrows[rids[ac.end]] |= 1<<end
                    current[idx]=end;rmapping[idx]=end
            close(rrows)
            if self.exact_order:
                for x,y in state.pairs:
                    ar=expanded_image(x,rows,self.m);rr=expanded_image(y,rrows,self.n)
                    for label in frag.observable:
                        if ar&(1<<ids[label]) and not rr&(1<<rids[label]):
                            raise Rejected('extra_order', f'{action.label}/{label}: old observable gets new precedence')
                for src in frag.observable:
                    for dst in frag.observable:
                        if src != dst and rows[ids[src]]&(1<<ids[dst]) and not rrows[rids[src]]&(1<<rids[dst]):
                            raise Rejected('extra_order', f'{action.label}: new {src} -> {dst} absent in reference')
        pairs=[(image(x,rows,mapping),image(y,rrows,rmapping)) for x,y in state.pairs] if self.exact_order else []
        if frag.observable and self.exact_order:
            pairs.extend((projection(rows,mapping,ids[v]),projection(rrows,rmapping,rids[v])) for v in frag.observable)
        return State(tuple(projection(rows,mapping,v) if v>=0 else 0 for v in mapping),live,readable,
                     tuple(history),tuple(projection(rrows,rmapping,v) for v in rmapping),dangerous_antichain(pairs))

    def check_exit(self,state:State):
        if state.live: raise Rejected('exit_event', 'terminating path has a live software event')
        if state.readable: raise Rejected('exit_resource','terminating path has unconsumed final blocks')


@dataclass(frozen=True)
class Node:
    action: Action
    successors: tuple[int, ...] = ()

@dataclass(frozen=True)
class Program:
    nodes: tuple[Node, ...]
    entry: int
    exits: tuple[int, ...]


def program_json(program):
    return {'nodes':[{'action':asdict(n.action),'successors':n.successors} for n in program.nodes],
            'entry':program.entry,'exits':program.exits}


def program_from_json(obj):
    nodes=[]
    for n in obj['nodes']:
        a=dict(n['action'])
        for k in ('reads','writes','blocks'): a[k]=tuple(a[k])
        nodes.append(Node(Action(**a),tuple(n['successors'])))
    return Program(tuple(nodes),obj['entry'],tuple(obj['exits']))


def state_from_json(obj):
    return State(tuple(obj['reach']),obj['live'],obj['readable'],tuple(tuple(x) for x in obj['history']),
                 tuple(obj['reference']),tuple(tuple(x) for x in obj['pairs']))


def analyze(program:Program,model:PhaseInterface,path:Path|None=None,max_states:int=20000):
    if not program.nodes or not 0<=program.entry<len(program.nodes):raise QualificationError('invalid CFG')
    for n in program.nodes:
        model.profile.fragment(n.action)
        if any(not 0<=t<len(program.nodes) for t in n.successors):raise QualificationError('invalid successor')
    if set(program.exits)!={i for i,n in enumerate(program.nodes) if not n.successors}:
        raise QualificationError('exit set must equal all terminating sites')
    seen=[set() for _ in program.nodes];initial=model.initial();seen[program.entry].add(initial)
    queue=deque([(program.entry,initial)]);parents={(program.entry,initial):None};count=0;arcs=0
    while queue:
        loc,state=queue.popleft();count+=1
        try:
            out=model.transfer(state,program.nodes[loc].action)
            if loc in program.exits:model.check_exit(out)
        except Rejected as e:
            trace=[];cur=(loc,state)
            while cur is not None:
                trace.append(cur[0]);cur=parents[cur]
            return {'accepted':False,'kind':e.kind,'detail':e.detail,'witness_sites':trace[::-1],
                    'processed':count,'states':sum(map(len,seen))}
        for target in program.nodes[loc].successors:
            arcs+=1
            if out not in seen[target]:
                seen[target].add(out);queue.append((target,out));parents[target,out]=(loc,state)
        if sum(map(len,seen))>max_states:return {'accepted':False,'kind':'budget','states':sum(map(len,seen))}
    if path:
        obj={'schema':'oahs.supplied_final_block_certificate.v1','profile':model.profile.json(),
             'exact_order':model.exact_order,'program':program_json(program),
             'invariant':[[asdict(s) for s in sorted(v,key=repr)] for v in seen]}
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')
    return {'accepted':True,'states':sum(map(len,seen)),'sites':len(program.nodes),'successor_checks':arcs,
            'max_states_per_site':max(map(len,seen)),'exact_order':model.exact_order,'trip_count_bound':None}


def check_certificate(path:Path):
    obj=json.loads(path.read_text())
    if obj['schema']!='oahs.supplied_final_block_certificate.v1':raise QualificationError('unknown certificate')
    model=PhaseInterface(Profile.from_json(obj['profile']),obj['exact_order']);program=program_from_json(obj['program'])
    invariant=[set(state_from_json(x) for x in bucket) for bucket in obj['invariant']]
    if len(invariant)!=len(program.nodes) or model.initial() not in invariant[program.entry]:
        raise Rejected('certificate','missing entry')
    if set(program.exits)!={i for i,n in enumerate(program.nodes) if not n.successors}:raise QualificationError('invalid exits')
    arcs=0
    for i,bucket in enumerate(invariant):
        for state in bucket:
            out=model.transfer(state,program.nodes[i].action)
            if i in program.exits:model.check_exit(out)
            for t in program.nodes[i].successors:
                if out not in invariant[t]:raise Rejected('certificate','successor missing')
                arcs+=1
    return {'accepted':True,'states':sum(map(len,invariant)),'successor_checks':arcs,'fixed_point_discovery':False}


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('certificate',type=Path);args=ap.parse_args()
    print(json.dumps(check_certificate(args.certificate),indent=2))
