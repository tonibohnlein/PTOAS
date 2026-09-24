#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to
# the License for details. You may not use this file except in compliance with the License. THIS
# SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the full text of the
# License.
# Compile the actual production coverage and partition headers without MLIR.
# Native importer/pass regressions are separate lit tests, not replaced here.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
scratch_root="${TMPDIR:-$root/.local/frontier-tests}"
mkdir -p "$scratch_root"
tmp="$(mktemp -d "$scratch_root/run.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/PTO/Transforms/FrontierSynch"
cat > "$tmp/PTO/Transforms/FrontierSynch/OriginalStructure.h" <<'HEADER'
#pragma once
#include "PTO/Transforms/FrontierSynch/WriteCoverage.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
namespace mlir::pto::frontiersynch {
inline constexpr auto NoControlId = std::numeric_limits<std::size_t>::max();
struct BaseMemInfo; // Opaque witness identity; never dereferenced by partitioning.
struct Access {
  std::size_t cell = 0;
  bool read = false, write = false, definiteWrite = false;
  const BaseMemInfo *memory = nullptr;
  std::size_t physicalRelation = NoControlId;
  WriteCoverage coverage = {};
};
struct Cell {
  bool exclusive = false;
  std::string addressSpace, provenance;
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  bool unknownRange = false;
  enum class Storage { Abstract, CanonicalInterval, OverlapWitness };
  Storage storage = Storage::Abstract;
  std::string coordinateSpace;
  std::vector<std::size_t> storageOrigins;
};
struct PhysicalOperation { std::vector<Access> accesses; };
struct OriginalStructure {
  std::vector<PhysicalOperation> operations;
  std::vector<Cell> cells;
};
}
HEADER
"${CXX:-c++}" -std=c++17 -O1 -g -Wall -Wextra -Werror \
  ${EXTRA_CXXFLAGS:-} -I"$tmp" -I"$root/include" \
  "$root/test/frontier_synch/write_coverage_standalone.cpp" -o "$tmp/check"
"$tmp/check"
