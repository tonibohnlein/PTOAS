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
scratch=${FRONTIER_TEST_TMPDIR:-"$root/.local/frontier-tests"}
mkdir -p -- "$scratch"
work=$(mktemp -d "$scratch/descriptors.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
flags=(-std=c++17 -fno-exceptions -Wall -Wextra -Werror -pedantic -O2 -DNDEBUG)
if [[ ${SANITIZE:-0} == 1 ]]; then
  flags+=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
fi
"${CXX:-c++}" "${flags[@]}" -I"$root/include" \
  "$root/test/standalone/frontier_descriptor_slots.cpp" -o "$work/descriptors"
"$work/descriptors"
