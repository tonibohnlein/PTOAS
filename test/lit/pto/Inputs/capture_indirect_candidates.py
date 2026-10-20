# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shape-only host capture; specialize and emit PTO, never invoke a device entry."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib-root", type=Path, required=True)
    parser.add_argument("--case", choices=("qwen-paged", "deepseek-swa"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    import torch
    from pypto import ir
    from pypto.pypto_core import passes

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    relative = (
        "models/qwen3_14b/test_paged_attention_pypto.py"
        if args.case == "qwen-paged"
        else "models/deepseek_v4_pro/prefill_attention_swa.py"
    )
    source = args.lib_root.resolve() / relative
    sys.path[:0] = [str(source.parent), str(args.lib_root.resolve())]
    spec = importlib.util.spec_from_file_location("indirect_capture", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case == "qwen-paged":
        # One batch, one physical page: addresses still come from loaded metadata.
        shapes = [
            ([128, 1024], torch.bfloat16),
            ([128, 1024], torch.bfloat16),
            ([1], torch.int32),
            ([1], torch.int32),
            ([1, 1], torch.float32),
            ([1], torch.int32),
            ([128, 128], torch.float32),
            ([128, 128], torch.float32),
            ([1, 5120], torch.float32),
            ([1, 1024], torch.float32),
            ([1, 1024], torch.float32),
            ([1, 128], torch.float32),
            ([1, 128], torch.float32),
        ]
        samples = [
            torch.empty(shape, dtype=dtype, device="meta") for shape, dtype in shapes
        ]
        samples.extend(
            [0, torch.empty([1, 40, 128], dtype=torch.bfloat16, device="meta")]
        )
        program = module.paged_attention_pypto_dynamic.specialize(*samples)
        signature = [{"shape": shape, "dtype": str(dtype)} for shape, dtype in shapes]
    else:
        program = module.prefill_attention_swa_test.specialize()
        signature = "original annotated static and symbolic dimensions"
    (args.output / "specialized.py").write_text(str(program))
    (args.output / "signature.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "signature": signature,
                "platform": "a2a3",
                "planner": "PYPTO",
                "samples": "meta tensors; no data execution",
            },
            indent=2,
        )
    )
    ir.compile(
        program,
        output_dir=str(args.output),
        platform="a2a3",
        skip_ptoas=True,
        dump_passes=True,
        memory_planner=passes.MemoryPlanner.PYPTO,
    )


if __name__ == "__main__":
    main()
