// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

#ifndef PTO_TRANSFORMS_INSERTSYNC_MMADCHAINDOMAIN_H
#define PTO_TRANSFORMS_INSERTSYNC_MMADCHAINDOMAIN_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

namespace mlir::pto::insert_sync_detail {

// A value-domain for the immediately preceding Cube instruction, NOT for
// completed work. The native adapter qualifies identities by physical context,
// exact ACC allocation, types, layout and valid M/N/K. Zero is unreachable;
// one is unknown (or different alternatives); values >=2 are qualified keys.
inline constexpr uint32_t kMmadUnreachable = 0;
inline constexpr uint32_t kMmadUnknown = 1;
inline uint32_t joinMmadPredecessor(uint32_t left, uint32_t right) {
  if (left == kMmadUnreachable)
    return right;
  if (right == kMmadUnreachable || left == right)
    return left;
  return kMmadUnknown;
}

// Deliberately narrower than the complete target API: nonempty, aligned f16
// matrix shapes are qualified by the native adapter. Never use allocation
// capacity in place of the actual valid M/N dimensions.
inline bool isLargeMmadShape(uint64_t m, uint64_t n, uint64_t k) {
  return m >= 16 && n >= 16 && k >= 16 && m <= 4095 && n <= 4095 &&
         k <= 4095 && m % 16 == 0 && n % 16 == 0 && k % 16 == 0 &&
         (m / 16) * (n / 16) >= 10;
}

struct MmadFlowNode {
  enum class Kind { PassThrough, Reset, Matrix };
  Kind kind = Kind::PassThrough;
  uint32_t key = kMmadUnknown;
  bool accumulate = false;
  std::vector<uint32_t> successors;
};

struct MmadFlowResult {
  enum class Status { Complete, InvalidGraph, LimitExceeded };
  Status status = Status::InvalidGraph;
  std::vector<uint32_t> incoming;
  std::vector<bool> eligible;
  std::size_t transfers = 0;
};

// Monotone dataflow over a conservative structured CFG. A loop is a backedge
// plus a zero-trip edge, not a sampled/unrolled body. A join keeps a key only
// when EVERY reachable incoming alternative has that same key. A Matrix node
// establishes a predecessor identity even if its own barrier cannot be elided.
// It establishes no memory completion, visibility, or event acknowledgement.
inline MmadFlowResult analyzeMmadFlow(const std::vector<MmadFlowNode> &nodes,
                                    uint32_t entry = 0,
                                    std::size_t maximumTransfers = 65536) {
  MmadFlowResult result;
  if (nodes.empty() || nodes.size() > 4096 || entry >= nodes.size()) {
    result.status = nodes.size() > 4096 ? MmadFlowResult::Status::LimitExceeded
                                      : MmadFlowResult::Status::InvalidGraph;
    return result;
  }
  for (const auto &node : nodes) {
    if (node.kind == MmadFlowNode::Kind::Matrix && node.key < 2)
      return result;
    for (uint32_t next : node.successors)
      if (next >= nodes.size())
        return result;
  }
  result.incoming.assign(nodes.size(), kMmadUnreachable);
  std::vector<uint32_t> outgoing(nodes.size(), kMmadUnreachable);
  std::vector<bool> queued(nodes.size(), false);
  std::deque<uint32_t> work;
  result.incoming[entry] = kMmadUnknown;
  work.push_back(entry);
  queued[entry] = true;
  while (!work.empty()) {
    if (result.transfers == maximumTransfers) {
      result.status = MmadFlowResult::Status::LimitExceeded;
      result.incoming.clear();
      result.eligible.clear();
      return result;
    }
    ++result.transfers;
    uint32_t current = work.front();
    work.pop_front();
    queued[current] = false;
    const auto &node = nodes[current];
    uint32_t after = result.incoming[current];
    if (after != kMmadUnreachable) {
      if (node.kind == MmadFlowNode::Kind::Reset)
        after = kMmadUnknown;
      else if (node.kind == MmadFlowNode::Kind::Matrix)
        after = node.key;
    }
    if (after == outgoing[current])
      continue;
    outgoing[current] = after;
    for (uint32_t next : node.successors) {
      uint32_t joined = joinMmadPredecessor(result.incoming[next], after);
      if (joined == result.incoming[next])
        continue;
      result.incoming[next] = joined;
      if (!queued[next]) {
        queued[next] = true;
        work.push_back(next);
      }
    }
  }
  result.eligible.assign(nodes.size(), false);
  for (std::size_t i = 0; i < nodes.size(); ++i)
    result.eligible[i] = nodes[i].kind == MmadFlowNode::Kind::Matrix &&
                         nodes[i].accumulate && result.incoming[i] == nodes[i].key;
  result.status = MmadFlowResult::Status::Complete;
  return result;
}
} // namespace mlir::pto::insert_sync_detail
#endif
