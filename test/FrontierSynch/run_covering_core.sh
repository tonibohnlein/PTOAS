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
scratch=${PTO_FRONTIER_TEST_TMPDIR:-"$root/.local/frontier-tests"}
mkdir -p -- "$scratch"
work=$(mktemp -d -- "$scratch/covering-step11.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
flags=(-std=c++17 -Wall -Wextra -Werror -pedantic -O2)
if [[ ${SANITIZE:-0} == 1 ]]; then
    flags+=(-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined)
fi
"${CXX:-c++}" "${flags[@]}" -I "$root/include" -I "$root/lib/PTO/Transforms/FrontierSynch" \
    "$root/tools/pto-test-opt/pto-frontier-covering-core-test.cpp" -o "$work/covering"
"$work/covering"
