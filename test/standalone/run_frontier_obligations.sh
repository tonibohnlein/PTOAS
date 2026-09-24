#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
# the License for details. You may not use this file except in compliance with the License. THIS
# SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
# License.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
scratch_root="${TMPDIR:-$root/.local/frontier-tests}"
mkdir -p "$scratch_root"
build=$(mktemp -d "$scratch_root/run.XXXXXX")
trap 'rm -rf -- "$build"' EXIT
cxx=${CXX:-c++}
flags=(-std=c++17 -Wall -Wextra -Werror -pedantic -O2)
if [[ ${SANITIZE:-0} == 1 ]]; then
  flags=(-std=c++17 -Wall -Wextra -Werror -pedantic -O0 -g1 -fsanitize=address,undefined -fno-omit-frame-pointer)
fi
"$cxx" "${flags[@]}" -I"$root/include" \
  "$root/test/standalone/frontier_obligations_test.cpp" -o "$build/model"
"$build/model"
# The same production factored transfer is independent of MLIR.
"$cxx" "${flags[@]}" -I"$root/include" \
  -I"$root/tools/pto-test-opt" "$root/test/standalone/frontier_factored_obligations_test.cpp" \
  -o "$build/transfer"
"$build/transfer"

# Lifetime integration runs as pto-frontier-lifetime-test in the native lit suite.
