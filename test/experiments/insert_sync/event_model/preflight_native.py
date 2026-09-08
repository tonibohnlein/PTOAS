#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Probe the actual parser and flow API used by native-reference comparisons."""
import json
import sys
from eventlab.islwrap import ISL
from eventlab.symbolic import Program

try:
    runtime = ISL()
    model = {"parameters": ["N"], "context": "-10<=N<=16", "statements": [
        {"id": "W", "lane": "MTE2", "iterators": ["i"], "domain": "0<=i<N",
         "schedule": ["i", "0"], "writes": [{"space": "VEC", "address": "512*(i mod 2)", "size": 512}]},
        {"id": "R", "lane": "V", "iterators": ["i"], "domain": "0<=i<N and not (i=2)",
         "schedule": ["i", "1"], "reads": [{"space": "VEC", "address": "512*(i mod 2)", "size": 512}]}]}
    program = Program(model, runtime)
    dependencies = program.discover()
    assert len(program.instances({"N": 0})) == 0
    assert len(program.instances({"N": 3})) == 5
    assert dependencies.raw.subset(dependencies.dense)
    print(json.dumps({"status": "PASS", "isl": runtime.version,
                      "checks": ["modulo", "choice", "loop", "access-flow", "exact-composition"],
                      "native_PTOAS": "not-run", "device": "not-run"}))
except Exception as error:
    print(json.dumps({"status": "FAIL", "error": str(error)}))
    sys.exit(1)
