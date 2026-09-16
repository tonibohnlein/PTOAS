"""Finite paired-interface check of added ORIGINAL payload-event ordering.

Actual side: issue-only prefix-event contract from causal_interface.py.
Reference side: issue order plus every earlier conflicting access completion.
History keeps maximally dangerous pairs: more actual reach, less reference
reach. Both issue and completion signatures are retained. No trace bound.
"""
from dataclasses import dataclass
from causal_interface import State,Command,Interface,Rejected,close

@dataclass(frozen=True)
class OrderedState:
    base: State
    reference: tuple[int,...]
    pairs: tuple[tuple[int,int],...]
    @property
    def live(self):return self.base.live

def dangerous_antichain(values):
    vals=sorted(set(values),key=lambda z:(-z[0].bit_count(),z[1].bit_count(),z))
    out=[]
    for a,b in vals:
        if not any((a&x)==a and (y&b)==y for x,y in out):out.append((a,b))
    return tuple(sorted(out))

class OrderedInterface:
    def __init__(self,base):
        self.base=base;self.keys=base.keys;self.ki=base.ki
        self.rports=tuple(['G:'+p for p in base.engines]+[f'{mode}:{c}'for c in base.cells for mode in ('R','W')])
        self.ri={p:i for i,p in enumerate(self.rports)};self.n=len(self.rports)
    def initial(self):
        return OrderedState(self.base.initial(),tuple([(1<<self.n)-1]*self.n),())
    def transfer(self,state,cmd,check=True):
        if cmd.kind=='nop':return state
        newbase=self.base.transfer(state.base,cmd,check)
        # Independently expose the small extension's signature transformer.
        b=self.base;n=b.m;p=cmd.engine;a=b.pi['A:'+p];t=b.pi['T:'+p]
        i,f,tn=n,n+1,n+2;rows=list(state.base.reach)+[1<<j for j in (i,f,tn)]
        rows[a]|=1<<i;rows[i]|=1<<f;rows[t]|=1<<tn;rows[f]|=1<<tn
        if cmd.kind in ('set','fence'):rows[t]|=1<<f
        if cmd.kind=='wait':rows[b.pi['S:'+cmd.key]]|=1<<f
        close(rows);mapping=list(range(n));mapping[a]=f if cmd.kind in ('wait','fence')else i;mapping[t]=tn
        if cmd.kind=='set':mapping[b.pi['S:'+cmd.key]]=f
        if cmd.kind=='wait':mapping[b.pi['S:'+cmd.key]]=-1;mapping[b.pi['D:'+cmd.key]]=f
        for e,j in b.ki.items():
            if not newbase.live&(1<<j):mapping[b.pi['S:'+e]]=-1
        def image(sig,rr,mm):
            anc=0
            for q in range(len(mm)):
                if sig&(1<<q):anc|=rr[q]
            return sum(1<<q for q,v in enumerate(mm)if v>=0 and anc&(1<<v))
        def point(v,rr,mm):return sum(1<<q for q,w in enumerate(mm)if w>=0 and rr[v]&(1<<w))
        # Reference payload graph is independent of emitted synchronization.
        rn=self.n;rrows=list(state.reference);rmapping=list(range(rn))
        if cmd.kind=='op':
            ii,ff=rn,rn+1;rrows += [1<<ii,1<<ff]
            gate=self.ri['G:'+p];rrows[gate]|=1<<ii;rrows[ii]|=1<<ff;rmapping[gate]=ii
            for c in set(cmd.reads)|set(cmd.writes):rrows[self.ri['W:'+c]]|=1<<ii
            for c in cmd.writes:rrows[self.ri['R:'+c]]|=1<<ii
            for mode,cs in (('R',cmd.reads),('W',cmd.writes)):
                for c in cs:
                    old=self.ri[mode+':'+c];new=len(rrows);rrows.append(1<<new)
                    rrows[old]|=1<<new;rrows[ff]|=1<<new;rmapping[old]=new
            close(rrows)
        newpairs=[(image(x,rows,mapping),image(y,rrows,rmapping))for x,y in state.pairs]
        if cmd.kind=='op':
            abit=1<<a;gbit=1<<self.ri['G:'+p]
            if check and any(x&abit and not y&gbit for x,y in newpairs):
                raise Rejected('extra_order',f'{cmd.label}: emitted synchronization adds payload ordering')
            newpairs += [(point(i,rows,mapping),point(ii,rrows,rmapping)),
                         (point(f,rows,mapping),point(ff,rrows,rmapping))]
        reference=tuple(point(v,rrows,rmapping)for v in rmapping)
        return OrderedState(newbase,reference,dangerous_antichain(newpairs))
