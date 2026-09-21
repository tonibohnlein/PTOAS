# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent local-effect ordering check of emitted dense vector witnesses.
No GM visibility, numerics, native phase ordering, or device timing is inferred.
"""
import json,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'placement'))
from check import VectorTrace, execute, parse

def run(path):
    # Parser adapter only: both operands and the output retain their effects;
    # VectorTrace assigns the adapted op to V, not the matrix/extract engines.
    lines=path.read_text().replace('pto.tadd ', 'pto.textract ').replace('pto.tabs ', 'pto.textract ').splitlines()
    at=next(i for i,l in enumerate(lines) if 'func.func @' in l)
    nodes,_=parse(lines,at+1); t=VectorTrace(); execute(nodes,{},t)
    assert not t.live, 'unconsumed events'
    return t

def complete(t):
    vertices=[v for _,_,launch,finish in t.operations for v in (launch,finish)]
    return {(i,j) for i,a in enumerate(vertices) for j,b in enumerate(vertices)
            if t.ancestors[b] & (1<<a)}

def compare(before,after):
    a,b=run(before),run(after)
    assert [(n,e) for n,e,_,_ in a.operations]==[(n,e) for n,e,_,_ in b.operations]
    x,y=complete(a),complete(b)
    assert not y-x, ('added payload order',sorted(y-x))
    return dict(payloads=len(a.operations),conflicts_before=a.required_checks,conflicts_after=b.required_checks,
        full_before=len(x),full_after=len(y),added=len(y-x),removed=len(x-y),
        completion_issue_removed=len(a.relations()-b.relations()),
        events_before=sum(v for (kind,_,_),v in a.sync_counts.items() if kind=='set_flag'),
        events_after=sum(v for (kind,_,_),v in b.sync_counts.items() if kind=='set_flag'))
if __name__=='__main__':
    print(json.dumps(compare(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
