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
scratch=${TMPDIR:-"$root/.local/frontier-tests"}
mkdir -p -- "$scratch"
build=$(mktemp -d "$scratch/d1.XXXXXX")
trap 'rm -rf -- "$build"' EXIT
flags=(-std=c++17 -O1 -g -Wall -Wextra -Werror)
if [[ ${SANITIZE:-0} == 1 ]]; then
    flags+=(-fsanitize=address,undefined -fno-omit-frame-pointer)
fi
"${CXX:-c++}" "${flags[@]}" -I"$root/include" \
    "$root/test/standalone/frontier_d1_sources_test.cpp" -o "$build/d1"
"$build/d1"
