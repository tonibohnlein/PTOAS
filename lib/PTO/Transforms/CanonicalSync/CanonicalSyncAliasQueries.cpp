// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#include "CanonicalSyncInternal.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

bool intervalsOverlap(ArrayRef<CanonicalByteInterval> first,
                      ArrayRef<CanonicalByteInterval> second) {
  for (const CanonicalByteInterval &left : first) {
    for (const CanonicalByteInterval &right : second) {
      const std::optional<std::uint64_t> leftEnd = left.end();
      const std::optional<std::uint64_t> rightEnd = right.end();
      if (!leftEnd || !rightEnd ||
          (left.begin < *rightEnd && right.begin < *leftEnd)) {
        return true;
      }
    }
  }
  return false;
}

} // namespace

bool mlir::pto::canonical_sync_detail::accessReads(CanonicalAccessMode mode) {
  return mode == CanonicalAccessMode::Read ||
         mode == CanonicalAccessMode::ReadWrite;
}

bool mlir::pto::canonical_sync_detail::accessWrites(CanonicalAccessMode mode) {
  return mode == CanonicalAccessMode::Write ||
         mode == CanonicalAccessMode::ReadWrite;
}

bool mlir::pto::canonical_sync_detail::accessesMayAlias(
    const CanonicalAccess &first, const CanonicalAccess &second) {
  // An unknown address space may denote GM or any local physical storage.
  // Neither a differing default enum value nor distinct SSA roots can prove
  // such accesses disjoint.
  if (first.unknownSpace || second.unknownSpace) {
    return true;
  }
  if (first.space != second.space) {
    return false;
  }
  if (first.unknownRange || second.unknownRange) {
    return true;
  }
  if (first.space == AddressSpace::GM && first.aliasRoot != second.aliasRoot) {
    return true;
  }
  if (first.physical && second.physical) {
    return intervalsOverlap(first.intervals, second.intervals);
  }
  return first.aliasRoot == second.aliasRoot &&
         intervalsOverlap(first.intervals, second.intervals);
}

bool mlir::pto::canonical_sync_detail::controlsCanCoexecute(
    ArrayRef<CanonicalControlAtom> first,
    ArrayRef<CanonicalControlAtom> second) {
  for (const CanonicalControlAtom &left : first) {
    for (const CanonicalControlAtom &right : second) {
      if (left.choice == right.choice && left.arm != right.arm) {
        return false;
      }
    }
  }
  return true;
}

SmallVector<CanonicalControlAtom, 2>
mlir::pto::canonical_sync_detail::intersectControlPaths(
    ArrayRef<CanonicalControlAtom> first,
    ArrayRef<CanonicalControlAtom> second) {
  SmallVector<CanonicalControlAtom, 2> result(first.begin(), first.end());
  for (const CanonicalControlAtom &atom : second) {
    if (!llvm::is_contained(result, atom)) {
      result.push_back(atom);
    }
  }
  llvm::sort(result, [](const CanonicalControlAtom &left,
                        const CanonicalControlAtom &right) {
    return std::tie(left.choice, left.arm) < std::tie(right.choice, right.arm);
  });
  return result;
}

CanonicalRegionId mlir::pto::canonical_sync_detail::findRegionLca(
    const CanonicalSyncProgram &program, CanonicalRegionId first,
    CanonicalRegionId second) {
  SmallVector<CanonicalRegionId, 8> ancestors;
  for (CanonicalRegionId current = first; current != kInvalidCanonicalSyncId;
       current = program.getRegion(current).parent) {
    ancestors.push_back(current);
  }
  for (CanonicalRegionId current = second; current != kInvalidCanonicalSyncId;
       current = program.getRegion(current).parent) {
    if (llvm::is_contained(ancestors, current)) {
      return current;
    }
  }
  return 0;
}
