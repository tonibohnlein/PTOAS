"""Unsynchronized finite control products; no publication/wait locations supplied.
The node's observation is a guard expression over ORIGINAL control variables.
Control abstraction is separate from the synthesis/checking engine.
"""
from dataclasses import dataclass, asdict
from collections import deque
from math import gcd,lcm
from causal_interface import Command

@dataclass(frozen=True)
class Node:
    command: Command
    successors: tuple[int,...]
    observation: str|None = None
    label: str=''

@dataclass
class Program:
    name: str
    engines: tuple[str,...]
    cells: tuple[str,...]
    nodes: list[Node]
    entry: int
    exits: set[int]
    predicates: dict[str,str]
    assumptions: str=''

class Builder:
    def __init__(self): self.nodes=[]; self.predicates={}
    def node(self,command=None,succ=(),obs=None,label=''):
        n=len(self.nodes);self.nodes.append(Node(command or Command('nop'),tuple(succ),obs,label or obs or str(n)))
        if obs: self.predicates[obs]=obs
        return n
    def edge(self,n,succ):
        x=self.nodes[n];self.nodes[n]=Node(x.command,tuple(succ),x.observation,x.label)
    def cut(self,obs,succ=()): return self.node(succ=succ,obs=obs)
    def op(self,p,c,mode,succ=(),label=''):
        return self.node(Command('op',p,(c,) if 'R'in mode else (), (c,) if 'W'in mode else (),label=label or mode+c),succ,label=label)
    def program(self,name,engines,cells,entry,exit,assumptions=''):
        return Program(name,tuple(engines),tuple(cells),self.nodes,entry,{exit},self.predicates,assumptions)

def ring(slots=2):
    # For affine-modular address (a*i+b) mod m the recurrence period is
    # m/gcd(a,m). Here a=1; tests below also check the general formula.
    b=slots; g=Builder();end=g.node(label='exit')
    entry=g.node(label='entry: N >= 0')
    initial=[(0,0,r) for r in range(b+2)]
    todo=deque(initial);head={q:g.node(label=f'loop header {q}')for q in initial};done=set()
    while todo:
        q=todo.popleft()
        if q in done: continue
        done.add(q);s,elapsed,remaining=q
        if remaining==0:g.edge(head[q],(end,));continue
        nr=(remaining-1,) if remaining<=b else (b,b+1)
        successors=[((s+1)%b,min(elapsed+1,b),r)for r in nr]
        for z in successors:
            if z not in head:head[z]=g.node(label=f'loop header {z}');todo.append(z)
        cond=f's={s},past={int(elapsed==b)},future={int(remaining>b)}'
        after=g.cut('after read | '+cond,tuple(head[z]for z in successors))
        read=g.op('Q',f'x{s}','R',(after,),'read x[i mod B]')
        middle=g.cut('after write | '+cond,(read,))
        write=g.op('P',f'x{s}','W',(middle,),'write x[i mod B]')
        before=g.cut('before write | '+cond,(write,))
        g.edge(head[q],(before,))
        for name in ('after read','after write','before write'):
            g.predicates[name+' | '+cond]=f'{name}: i%B=={s} && '+('i>=B'if elapsed==b else'i<B')+' && '+('N-i>B'if remaining>b else'N-i<=B')
    g.edge(entry,tuple(head[q]for q in initial))
    return g.program(f'ring_{b}',('P','Q'),tuple(f'x{s}'for s in range(b)),entry,end,
        'N>=0, i starts at zero; saturated elapsed/remaining and modular address quotient include all finite N. No sync supplied.')

def nested():
    g=Builder();end=g.node(label='exit');last=g.op('P','x','W',(end,),'final overwrite')
    finalcut=g.cut('before final overwrite',(last,));head=g.node(label='outer header')
    after0=g.cut('inner exit | m==0',(head,));after1=g.cut('inner exit | m>0',(head,))
    # The inner loop has one original read operation and an arbitrary finite count.
    # A preheader/exit can use the bound m, but not hidden consumption history.
    test=g.node(label='inner continue')
    afterread=g.cut('after inner read',(test,))
    read=g.op('Q','x','R',(afterread,),'inner read')
    preread=g.cut('before inner read',(read,));g.edge(test,(preread,after1))
    nonempty=g.cut('inner preheader | m>0',(preread,))
    choose=g.node(succ=(after0,nonempty),label='test m>0')
    afterwrite=g.cut('after production',(choose,))
    write=g.op('P','x','W',(afterwrite,),'produce')
    beforewrite=g.cut('before production',(write,));g.edge(head,(beforewrite,finalcut))
    return g.program('nested_reads',('P','Q'),('x',),head,end,
        'Arbitrary finite outer visits and each inner m>=0. Bound test controls distinct existing edges.')

def mixed():
    g=Builder();end=g.node(label='exit');last=g.op('P','x','W',(end,),'final overwrite')
    finalcut=g.cut('before final overwrite',(last,));head=g.node(label='outer header')
    after=g.cut('after final Q read',(head,));q2=g.op('Q','x','R',(after,),'final Q read');join=g.cut('join before Q',(q2,))
    afterq=g.cut('true arm after Q',(join,));q1=g.op('Q','x','R',(afterq,),'optional Q read');yes=g.cut('true arm before Q',(q1,))
    afterr=g.cut('false arm after R',(join,));r=g.op('R','x','RW',(afterr,),'R update');no=g.cut('false arm before R',(r,))
    branch=g.node(succ=(yes,no),label='choose existing b')
    post=g.cut('after P before branch',(branch,));p=g.op('P','x','W',(post,),'P write');pre=g.cut('before P',(p,))
    g.edge(head,(pre,finalcut))
    return g.program('mixed_writers',('P','Q','R'),('x',),head,end,'Arbitrary finite branch histories; branch-arm cuts are original control observations.')

def readers():
    g=Builder();end=g.node(label='exit');last=g.op('P','x','W',(end,),'final overwrite');lastcut=g.cut('before final overwrite',(last,));head=g.node(label='outer header')
    ar=g.cut('after R',(head,));r=g.op('R','x','R',(ar,),'R read');br=g.cut('before R',(r,))
    aq=g.cut('after Q',(br,));q=g.op('Q','x','R',(aq,),'Q read');bq=g.cut('before Q',(q,))
    ap=g.cut('after P',(bq,));p=g.op('P','x','W',(ap,),'P write');bp=g.cut('before P',(p,));g.edge(head,(bp,lastcut))
    return g.program('independent_readers',('P','Q','R'),('x',),head,end)

def straight():
    g=Builder();end=g.node(label='exit');r=g.op('Q','x','R',(end,),'read x');cutr=g.cut('before Q read',(r,));y=g.op('P','y','W',(cutr,),'independent write y');cuty=g.cut('after write x',(y,));x=g.op('P','x','W',(cuty,),'write x')
    return g.program('early_prefix',('P','Q'),('x','y'),x,end)

def affine_mod_period(a,m):
    if m<=0: raise ValueError('positive modulus required')
    return m//gcd(abs(a),m)


@dataclass(frozen=True)
class ModCell:
    family: str
    coefficient: int = 1
    offset: int = 0
    modulus: int = 1
    def period(self):return affine_mod_period(self.coefficient,self.modulus)
    def at(self,i):return f"{self.family}{(self.coefficient*i+self.offset)%self.modulus}"

@dataclass(frozen=True)
class Payload:
    engine: str
    reads: tuple[ModCell,...]=()
    writes: tuple[ModCell,...]=()
    label: str=''

def periodic_loop(body:tuple[Payload,...],name='periodic_loop',max_period=8):
    """Compile ANY straight-line periodic-effect body to a finite control product.
    The LCM and startup/tail predicate *vocabulary* come from address syntax;
    no source/target event, direction, key, or synchronization guard is supplied.
    This small frontend handles constant-step, zero-based counted loops.
    General CFG input remains available for nested loops and choices.
    """
    if not body:raise ValueError('nonempty payload body required')
    exprs=[e for op in body for e in op.reads+op.writes]
    period=lcm(*(e.period()for e in exprs)) if exprs else 1
    if period>max_period:raise ValueError('control precision budget exceeded')
    engines=tuple(dict.fromkeys(op.engine for op in body))
    cells=tuple(sorted({e.at(i)for e in exprs for i in range(period)}))
    g=Builder();end=g.node(label='exit');entry=g.node(label='enter counted loop')
    initial=[(0,0,r)for r in range(period+2)];heads={q:g.node(label=f'head {q}')for q in initial};todo=deque(initial);done=set()
    while todo:
        q=todo.popleft()
        if q in done:continue
        done.add(q);residue,past,remaining=q
        if not remaining:g.edge(heads[q],(end,));continue
        nr=(remaining-1,)if remaining<=period else(period,period+1)
        nextqs=[((residue+1)%period,min(past+1,period),r)for r in nr]
        for z in nextqs:
            if z not in heads:heads[z]=g.node(label=f'head {z}');todo.append(z)
        cond=f'r={residue},past={int(past==period)},future={int(remaining>period)}'
        continuation=g.node(succ=tuple(heads[z]for z in nextqs),label='advance original i')
        for j in range(len(body),-1,-1):
            ob=f'body-cut-{j} | '+cond
            cut=g.cut(ob,(continuation,))
            g.predicates[ob]=f'body cut {j}: i%{period}=={residue} && '+(f'i>={period}'if past==period else f'i<{period}')+' && '+(f'N-i>{period}'if remaining>period else f'N-i<={period}')
            if j:
                op=body[j-1]
                c=Command('op',op.engine,tuple(e.at(residue)for e in op.reads),tuple(e.at(residue)for e in op.writes),label=op.label or f'payload {j-1}')
                continuation=g.node(c,(cut,),label=c.label)
            else:continuation=cut
        g.edge(heads[q],(continuation,))
    g.edge(entry,tuple(heads[q]for q in initial))
    return g.program(name,engines,cells,entry,end,
        f'Generated from periodic address expressions; period={period}. All N>=0. Loop/predicate arithmetic uses unbounded mathematical integers; native fixed-width translation requires separate no-overflow facts.')
