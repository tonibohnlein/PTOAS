# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from pathlib import Path
import sys,json,argparse
p=argparse.ArgumentParser(description='Independent no-motion GEMM memory/event/ordering comparison')
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--default',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
args=p.parse_args();sys.path.insert(0,str(args.repo/'test/oahs'))
from check_carried_slot_trace import Trace,parse,execute
class Capture(Trace):
 def __init__(self):super().__init__(check_outer=True,check_bank=True);self.starts=[]
 def payload(self,name,effects,context,acc_order=False):
  self.starts.append(len(self.ancestors));super().payload(name,effects,context,acc_order)
def run(path,step):
 lines=path.read_text().splitlines();nodes,_=parse(lines,next(i for i,l in enumerate(lines) if 'func.func @' in l)+1)
 t=Capture();execute(nodes,{'%arg3':0,'%arg4':step},t);assert not t.live
 assert t.sync_counts.get(('barrier','ALL',None))==1
 assert not any(n for (kind,pipe,_),n in t.sync_counts.items() if kind=='barrier' and pipe!='ALL')
 return t
def rel(t):return {(i,j) for j,at in enumerate(t.starts) for i,p in enumerate(t.payloads[:j]) if t.ancestors[at]&(1<<p[1])}
rows=[]
for step in (256,128,64):
 a=run(args.default,step)
 b=run(args.candidate,step)
 identity=lambda t:[(p[0],p[2],tuple(v for _,v in p[3])) for p in t.payloads]
 assert identity(a)==identity(b);ra,rb=rel(a),rel(b)
 x=dict(tiles=256//step,added=len(rb-ra),removed=len(ra-rb),baseline_pairs=sum(n for (k,_,_),n in a.sync_counts.items() if k=='set_flag'),candidate_pairs=sum(n for (k,_,_),n in b.sync_counts.items() if k=='set_flag'),conflicts=b.required_checks)
 assert x['added']==0,x
 rows.append(x);print(x,flush=True)
print(json.dumps(rows,indent=2))
