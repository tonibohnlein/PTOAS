# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under
# the terms and conditions of CANN Open Software License Agreement Version 2.0
# (the "License"). Please refer to the License for details. You may not use
# this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
# AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
# FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
# for the full text of the License.

"""Import a trusted frozen entry and emit PTO without invoking kernel execution.

The parent collector owns timeout, CPU affinity, and source/output validation.
This worker only accepts a pre-inventoried top-level JIT or program entry. It
does not guess sample shapes or scalar values when specialization needs them.
Importing the reviewed frontend source still executes its top-level Python.
"""

import argparse
import importlib
import inspect
import json
import os
import sys
import traceback
from pathlib import Path


def runtime_scalars(entry, language):
    """Preserve runtime scalar parameters; do not specialize arbitrary literals."""
    values = {}
    for name, parameter in inspect.signature(entry._func).parameters.items():
        annotation = str(parameter.annotation)
        if "Scalar" in annotation and parameter.default is inspect.Parameter.empty:
            values[name] = language.RUNTIME
    return values


def load_entry_module(root, source):
    """Preserve the sibling imports used by source-backed example drivers."""
    module_name = source.with_suffix("").as_posix().replace("/", ".")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str((root / source).parent))
    sys.argv = [str(root / source)]
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve(strict=True) != (root / source).resolve(strict=True):
        raise ValueError("imported entry module does not match the inventoried source")
    return module


def collect_entry(root, source, name, kind, output, samples=None):
    # Also bound direct smoke invocations, not only launches by the collector.
    for key in ("PYPTO_CODEGEN_MAX_WORKERS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import pypto.language as pl
    from pypto import ir
    from pypto.backend import BackendType
    from pypto.pypto_core import passes

    # Module import must see no collector flags in any module-local parser.
    module = load_entry_module(root, source)
    entry = getattr(module, name)
    if samples is not None:
        arguments, keywords = materialize_samples(samples)
        program = entry.specialize(*arguments, **keywords)
    elif kind == "jit":
        program = entry.specialize(**runtime_scalars(entry, pl))
    else:
        program = entry
    ir.compile(program, output_dir=str(output), backend_type=BackendType.Ascend910B,
               memory_planner=passes.MemoryPlanner.PYPTO, skip_ptoas=True,
               dump_passes=False)
    import pypto
    from pypto import pypto_core

    return {"source_module": module.__file__, "pypto_module": pypto.__file__,
            "frontend_extension": pypto_core.__file__, "python": sys.version}


def materialize_samples(samples):
    """Allocate only meta tensors; shared source identities stay shared objects."""
    import torch
    from frontend_samples import DTYPES

    if samples.get("status") != "source-samples":
        raise ValueError("driver arguments were not resolved")
    tensors = {}

    def materialize(value):
        if not isinstance(value, dict):
            if type(value) not in (int, float, bool):
                raise ValueError("unsupported scalar sample")
            return value
        identity = value["identity"]
        if value["dtype"] not in DTYPES:
            raise ValueError("unsupported tensor dtype")
        if identity not in tensors:
            tensors[identity] = (value, torch.empty(tuple(value["shape"]), dtype=getattr(torch, value["dtype"]),
                                                   device="meta"))
        if tensors[identity][0] != value:
            raise ValueError("conflicting tensor identity")
        return tensors[identity][1]

    return ([materialize(value) for value in samples["arguments"]],
            {name: materialize(value) for name, value in samples["keywords"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--kind", choices=("jit", "program"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    source = (root / args.source).resolve(strict=True)
    if not source.is_relative_to(root) or not args.entry.isidentifier():
        parser.error("entry must be a top-level identifier in the frozen source root")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "collected", "skip_ptoas": True, "memory_planner": "PYPTO",
              "backend": "Ascend910B", "level": "level3"}
    try:
        samples = json.loads(args.samples.read_text(encoding="utf-8")) if args.samples else None
        if samples is not None:
            from frontend_inventory import discover_entries
            from frontend_samples import discover_driver_calls

            source_text = source.read_text(encoding="utf-8")
            names = {entry["entry"] for entry in discover_entries(source_text)
                     if entry["kind"] == "jit" and entry["top_level"]}
            if samples.get("entry") != args.entry or samples not in discover_driver_calls(source_text, names):
                raise ValueError("sample record does not match the frozen source call")
        result["samples"] = samples
        result.update(collect_entry(root, source.relative_to(root), args.entry, args.kind, output, samples))
    except Exception as error:
        result.update(status="collection-failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    (output / "collection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return int(result["status"] != "collected")


if __name__ == "__main__":
    sys.exit(main())
