# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Check matrix access precision against full rectangles and current metadata."""
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile


def run(tool, path, mode, *options):
    result = subprocess.run([tool, *options, mode, str(path)], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout


def main():
    tool, fixture = sys.argv[1:]
    text = Path(fixture).read_text()
    text = text[:text.index("  func.func @partial_shapes")] + "}\n"
    with tempfile.TemporaryDirectory(prefix="matrix-access-") as directory:
        path = Path(directory) / "case.pto"
        for element, width in [("f16", 2), ("bf16", 2), ("f32", 4)]:
            base = text.replace("f16", element).replace("constant 512 : i64", "constant 1024 : i64")
            # Retain all native layouts; resolve valid extents at the use.
            dynamic = re.sub(r"(tile_buf<\w+, 16x16x(?:bf16|f16|f32))", r"\1, valid=?x?", base)
            dynamic = re.sub(r"(pto.alloc_tile addr = %\w+)", r"\1 valid_row = %i16 valid_col = %i16", dynamic)
            for source in [base, dynamic]:
                path.write_text(source)
                effects = run(tool, path, "--storage-effects")
                assert "all-materialized=1" in effects, effects
                assert len(re.findall(r"effect \d+ .* intervals", effects)) == 17, effects
                # Whole 16x16 operands occupy exactly these physical byte spans.
                assert f"space=2 [0,{256 * width})" in effects, effects
                assert f"space=3 [0,{256 * width})" in effects, effects
                assert "space=5 [0,1024)" in effects, effects
                assert "effect 11 pto.tmatmul.acc read intervals" in effects
                assert "effect 14 pto.tmatmul.acc write intervals" in effects
                # Existing InsertSync consumes the same shared declarations.
                existing = run(tool, path, "--existing-dump")
                assert "pto.set_flag" in existing and "pto.wait_flag" in existing, existing
            # A later descriptor update cannot reuse the original full extent.
            changed = dynamic.replace("    pto.tmatmul ins",
                                      "    pto.set_validshape %al, %i1, %i16 : !left\n    pto.tmatmul ins")
            path.write_text(changed)
            effects = run(tool, path, "--storage-effects")
            assert "effect 5 pto.textract write intervals" in effects, effects
            assert "effect 8 pto.tmatmul read intervals" in effects, effects
            assert "effect 15 pto.tstore read intervals" in effects, effects
            restored = changed.replace("    pto.tmatmul.acc",
                                       "    pto.set_validshape %al, %i16, %i16 : !left\n    pto.tmatmul.acc")
            path.write_text(restored)
            effects = run(tool, path, "--storage-effects")
            assert "effect 8 pto.tmatmul read intervals" in effects, effects
            assert "effect 11 pto.tmatmul.acc read intervals" in effects, effects
            # An unknown extent remains conservative even with fixed capacity.
            unknown = dynamic.replace("%a: !pto.ptr", "%rows: index, %a: !pto.ptr")
            unknown = unknown.replace("%al = pto.alloc_tile addr = %base0 valid_row = %i16",
                                      "%al = pto.alloc_tile addr = %base0 valid_row = %rows")
            path.write_text(unknown)
            effects = run(tool, path, "--storage-effects")
            assert "effect 8 pto.tmatmul read intervals" in effects, effects
        # More precise input maps must not reject an otherwise read-only base
        # just because a varying max expression is outside template arithmetic.
        source = text.replace('f16', 'bf16').replace('%a: !pto.ptr', '%offset: index, %a: !pto.ptr')
        partition = next(line for line in source.splitlines() if '%avp = ' in line)
        source = source.replace(partition + '\n', '')
        loop = ('    scf.for %iv = %zero to %i16 step %i1 {\n'
                '    %row = arith.maxsi %iv, %offset : index\n' +
                partition.replace('offsets = [%zero, %zero]', 'offsets = [%row, %zero]') + '\n')
        source = source.replace('    pto.tload ins(%avp', loop + '    pto.tload ins(%avp', 1)
        source = source.replace('    return', '    }\n    return')
        store = next(line for line in source.splitlines() if line.lstrip().startswith('pto.tstore '))
        output_partition = next(line for line in source.splitlines() if '%cvp = ' in line)
        source = source.replace(output_partition + '\n', '')
        # Disjoint output tiles keep the writer's own iteration-private proof
        # from obscuring the input-origin alias-policy check.
        output_partition = output_partition.replace('offsets = [%zero, %zero]', 'offsets = [%out_row, %zero]')
        source = source.replace(store, '    %out_row = arith.muli %iv, %i16 : index\n' +
                                output_partition + '\n' + store)
        candidates = {
            'none': source.replace(store, ''),
            'same': source.replace('%cv = pto.make_tensor_view %c,', '%cv = pto.make_tensor_view %a,'),
            'distinct': source,
        }
        for writer, candidate in candidates.items():
            path.write_text(candidate)
            for policy in ('may-not-alias', 'may-alias'):
                output = run(tool, path, '--recognize', '--gm-alias=' + policy)
                docs = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
                attempts = [a for doc in docs for node in doc['nodes'] for a in node['attempts']
                            if a['route'] == 'numeric-template']
                expected = writer == 'none' or (writer == 'distinct' and policy == 'may-not-alias')
                assert attempts and any(a['state'] == 'applicable' for a in attempts) == expected, (
                    writer, policy, attempts)
    print("matrix accesses: F16/BF16/F32 rectangles, RMW, dynamic metadata and existing consumer passed")


if __name__ == "__main__":
    main()
