#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# The test driver invokes children serially with explicit timeouts.
set -eu
python_tool=$1
trace_tool=$2
optimizer=$3
fixture=$4
scratch=$5
helper=$(dirname "$0")/check_combined_family_output.py
mkdir -p "$scratch"
trap 'rm -rf "$scratch"' EXIT
for inner in 2 3 5; do
    "$python_tool" "$helper" prepare "$fixture" "$scratch/input.pto" "$inner"
    timeout 60 "$optimizer" --mlir-disable-threading \
        --pto-frontier-analysis="allocation-ids=0,1,2,3,4,5" \
        --pto-frontier-allocate="eligible-ids=0,1,2,3,4,5" "$scratch/input.pto" > "$scratch/split.pto"
    timeout 60 "$optimizer" --mlir-disable-threading \
        --pto-frontier-analysis="allocation-ids=0,1,2,3,4,5 physical-output=true profile-stages=true" \
        "$scratch/input.pto" > "$scratch/combined.pto" 2> "$scratch/profile.txt"
    "$python_tool" "$helper" profile "$scratch/profile.txt" "$scratch/combined.pto"
    for outer in 0 1 2 4; do
        for mode in split combined; do
            "$python_tool" "$helper" trace "$scratch/$mode.pto" "$scratch/trace.pto" "$outer" "$inner"
            timeout 60 "$trace_tool" --structured-trace "$scratch/trace.pto" > "$scratch/$mode.json"
        done
        "$python_tool" "$helper" compare "$scratch/split.json" "$scratch/combined.json"
    done
done
printf '%s\n' 'combined families: 12 zero/short/repeated physical traces match split output'
