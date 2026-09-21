# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A3 retained-input witness; each parent uses different GM input/output tiles."""
import argparse
from pathlib import Path


def source(outer=4, readers=2):
    tile = "!pto.tile_buf<vec, 32x256xf32>"
    part = "!pto.partition_tensor_view<32x256xf32>"
    lines = [
        'module attributes {pto.backend = "emitc", pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a3"} {',
        '  func.func @retained_inputs(%a: !pto.ptr<f32, gm>, %b: !pto.ptr<f32, gm>, %out: !pto.ptr<f32, gm>) attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {',
        '    %zero = arith.constant 0 : index',
        '    %one = arith.constant 1 : index',
        '    %rows = arith.constant 32 : index',
        '    %cols = arith.constant 256 : index',
        f'    %grows = arith.constant {outer * 32} : index',
        f'    %outer = arith.constant {outer} : index',
        f'    %readers = arith.constant {readers} : index',
    ]
    for name in ('a', 'b', 'out'):
        lines.append(f'    %{name}view = pto.make_tensor_view %{name}, shape = [%grows, %cols], strides = [%cols, %one] {{layout = #pto.layout<nd>}} : !pto.tensor_view<?x?xf32>')
    for i, name in enumerate(('x', 'y', 'z', 'w')):
        lines += [f'    %addr{i} = arith.constant {i * 32768} : i64', f'    %{name} = pto.alloc_tile addr = %addr{i} : {tile}']
    lines += ['    scf.for %tile = %zero to %outer step %one {', '      %offset = arith.muli %tile, %rows : index']
    for name in ('a', 'b', 'out'):
        lines.append(f'      %{name}part = pto.partition_view %{name}view, offsets = [%offset, %zero], sizes = [%rows, %cols] : !pto.tensor_view<?x?xf32>')
    for name, local in (('a', 'x'), ('b', 'y')):
        lines.append(f'      pto.tload ins(%{name}part : {part}) outs(%{local} : {tile})')
    for i, (name, dest) in enumerate((('x', 'z'), ('y', 'w'), ('x', 'z'), ('y', 'w'))):
        lines.append(f'      scf.for %i{i} = %zero to %readers step %one {{')
        operation = f'pto.tabs ins(%{name} : {tile})' if i < 2 else f'pto.tadd ins(%{name}, %{dest} : {tile}, {tile})'
        lines += [f'        {operation} outs(%{dest} : {tile})', '      }']
    lines += [f'      pto.tadd ins(%z, %w : {tile}, {tile}) outs(%z : {tile})',
              f'      pto.tstore ins(%z : {tile}) outs(%outpart : {part})', '    }', '    return', '  }', '}']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for outer, readers in ((2, 1), (4, 2)):
        (args.out / f'retained_{outer}_{readers}.pto').write_text(source(outer, readers))
