#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
# the License for details. You may not use this file except in compliance with the License. THIS
# SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
# License.
# Compile and run the actual original-interval core without LLVM/MLIR headers.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
scratch_root="${TMPDIR:-$root/.local/frontier-tests}"
mkdir -p "$scratch_root"
tmp=$(mktemp -d "$scratch_root/run.XXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
"${CXX:-c++}" -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -I"$root/include" -I"$root/lib/PTO/Transforms/FrontierSynch" \
  "$root/tools/pto-test-opt/pto-frontier-interval-core-test.cpp" \
  -o "$tmp/interval-core"
"$tmp/interval-core"
