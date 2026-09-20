# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Numerically simple native witnesses for placement experiments (A3 vector)."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
shape='32x256xf32';tile=f'!pto.tile_buf<vec, {shape}>';part=f'!pto.partition_tensor_view<{shape}>'
def header(name,nout,active=False):
 args=[f'%{v}: !pto.ptr<f32, gm>' for v in ['a','b']+[f'o{i}' for i in range(nout)]]
 if active:args.append('%active: i32')
 text=['module attributes {pto.backend = "emitc", pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a3"} {',f'  func.func @{name}('+', '.join(args)+') attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {','    %rows = arith.constant 32 : index','    %cols = arith.constant 256 : index','    %one = arith.constant 1 : index','    %zero = arith.constant 0 : index']
 for v in ['a','b']+[f'o{i}' for i in range(nout)]:
  text += [f'    %{v}view = pto.make_tensor_view %{v}, shape = [%rows, %cols], strides = [%cols, %one] {{layout = #pto.layout<nd>}} : !pto.tensor_view<?x?xf32>',f'    %{v}part = pto.partition_view %{v}view, offsets = [%zero, %zero], sizes = [%rows, %cols] : !pto.tensor_view<?x?xf32>']
 for i,v in enumerate(['x','y','z','w','unused']):
  text += [f'    %addr{i} = arith.constant {i*32768} : i64',f'    %{v} = pto.alloc_tile addr = %addr{i} : {tile}']
 return text
def load(src,dst):return f'    pto.tload ins(%{src}part : {part}) outs(%{dst} : {tile})'
def absolute(src,dst):return f'    pto.tabs ins(%{src} : {tile}) outs(%{dst} : {tile})'
def store(src,dst):return f'    pto.tstore ins(%{src} : {tile}) outs(%{dst}part : {part})'
def save(name,lines): (a.out/(name+'.pto')).write_text('\n'.join(lines+['    return','  }','}'])+'\n')
name='source_gap';lines=header(name,4)
lines += [load('a','y'),load('b','w'),store('y','o0'),load('b','x'),absolute('x','z'),absolute('w','y'),load('a','x'),store('z','o1'),store('y','o2'),store('x','o3')];save(name,lines)
name='deferred_ack';lines=header(name,4,True)
lines += [load('a','y'),load('b','x'),absolute('y','z'),'    %zero_i32 = arith.constant 0 : i32','    %take = arith.cmpi ne, %active, %zero_i32 : i32','    scf.if %take {',load('a','x'),'    }',absolute('x','y'),load('b','w'),store('z','o0'),store('y','o1'),store('w','o2'),store('x','o3')];save(name,lines)
name='class_invariant';lines=header(name,2)
lines += [load('a','x'),load('b','y'),load('a','unused'),'    %lb = arith.constant 3 : index','    %ub = arith.constant 9 : index','    %step = arith.constant 2 : index','    scf.for %i = %lb to %ub step %step {',absolute('x','z'),load('a','unused'),absolute('y','w'),'    }',store('z','o0'),store('w','o1')];save(name,lines)
(a.out/'numerical-contract.json').write_text(json.dumps(dict(shape=[32,256],dtype='float32',inputs='finite signed values; independent a and b',kernels={'source_gap':['a','abs(b)','abs(b)','a'],'deferred_ack':['abs(a)','abs(active ? a : b)','b','active ? a : b'],'class_invariant':['abs(a)','abs(b)']},active_values=[0,1],local_bytes=163840,oracle='exact for finite FP32 abs/load/store; ignore signed-zero sign or exclude input zeros',scope='synthetic placement witnesses, not substitutes for projection/attention payloads'),indent=2)+'\n')
